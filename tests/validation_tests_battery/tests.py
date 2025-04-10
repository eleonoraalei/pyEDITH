from pyEDITH import calculate_texp, lambda_d_to_arcsec
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from pyEDITH import AstrophysicalScene, Observation, Observatory, ObservatoryBuilder
from pyEDITH import calculate_exposure_time_or_snr, parse_input, lambda_d_to_arcsec
from pyEDITH.units import *
import os
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
import io

# LOAD HPIC
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(script_dir, "ETC_cal_detect.xlsx")
hpic = pd.read_csv(os.path.join(script_dir, "full_HPIC.txt"), sep="|")


def to_arcsec(quantity, observer_distance):
    return np.arctan(quantity / observer_distance).to(u.arcsec).value


def solar_longitude_to_ra_dec(solar_longitude, ecliptic_latitude):
    # Convert solar longitude to ecliptic longitude
    ecliptic_longitude = (solar_longitude + 180) % 360

    # Create a SkyCoord object in ecliptic coordinates
    ecliptic = SkyCoord(
        lon=ecliptic_longitude * u.deg,
        lat=ecliptic_latitude * u.deg,
        frame="barycentrictrueecliptic",
    )

    # Transform to ICRS (which gives RA and Dec)
    equatorial = ecliptic.transform_to("icrs")

    # Extract RA and Dec
    ra = equatorial.ra
    dec = equatorial.dec

    return ra.value, dec.value


def fluxes_to_magnitudes(F_star, F_p, F0):

    # Calculate dmag (magnitude difference between star and planet) for each wavelength
    dmag = -2.5 * np.log10(np.array(F_p) / np.array(F_star))

    return dmag


def run_single_test(name, input_params, expected_output):

    input_params["delta_mag"] = fluxes_to_magnitudes(
        input_params["Fstar"], input_params["Fp"], input_params["F0"]
    )
    # input_params["ra"], input_params["dec"] = solar_longitude_to_ra_dec(
    #     input_params["sollong"], input_params["ecllat"]
    # )
    import logging

    yippy_logger = logging.getLogger("yippy")
    yippy_logger.setLevel(logging.ERROR)
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        texp, validation_output = calculate_texp(input_params, verbose=False)

    errors = []
    # # Check specific variables
    for key, expected_value in expected_output.items():
        calculated_value = validation_output[0][key]
        if hasattr(calculated_value, "value"):
            calculated_value = calculated_value.value

        try:
            np.testing.assert_allclose(
                calculated_value,
                expected_value,
                rtol=1e-1,
                err_msg=f"Mismatch in {key} for test case: {name}",
            )
            # print("PASSED")
        except AssertionError:

            errors.append(
                f"-- {key}: FAILED - Expected: {expected_value}, Calculated: {calculated_value}"
            )

    if len(errors) == 0:
        print(f"Test case '{name}' passed successfully!")
    else:
        print(f"Test case '{name}' had some errors: \n" + "\n".join(errors))

    print(texp)


def read_from_excel_sheet(name):

    hip_name = int(name.strip("HIP "))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file_path = os.path.join(script_dir, "ETC_cal_detect.xlsx")

    df = pd.read_excel(
        excel_file_path,
        sheet_name=" ".join([name, "(Detection)"]),
        skiprows=[1, 2, 3, 4, 5, 6],
        usecols="A,D,J",
    )

    df.columns = ["parameter", "500nm", "1000nm"]

    for lamb in ["500nm", "1000nm"]:
        print(lamb)
        name_str = " ".join([name, lamb])
        input = {
            "F0": np.array([df.loc[df["parameter"] == "F_0", lamb].iloc[0]]),
            "mag": np.array([df.loc[df["parameter"] == "m_lambda", lamb].iloc[0]]),
            "Lstar": 10 ** float(hpic[hpic.hip_name == hip_name].st_lum.iloc[0]),
            "magV": float(hpic[hpic.hip_name == hip_name].sy_vmag.iloc[0]),
            "angdiam": np.round(
                to_arcsec(
                    2
                    * (
                        float(hpic[hpic.hip_name == hip_name].st_rad.iloc[0]) * u.Rsun
                    ).to(u.m),
                    float(hpic[hpic.hip_name == hip_name].sy_dist.iloc[0]) * u.pc,
                ),
                4,
            ),
            "distance": float(hpic[hpic.hip_name == hip_name].sy_dist.iloc[0]),
            "diameter": df.loc[df["parameter"] == "D", lamb].iloc[0],
            "unobscured_area": (1.0 - 0.121),
            "photap_rad": 0.85,
            "lambd": np.array(
                [df.loc[df["parameter"] == "λ", lamb].iloc[0] / 1000]
            ),  # nm to micron
            "bandwidth": df.loc[df["parameter"] == "Δλ", lamb].iloc[0]
            / df.loc[df["parameter"] == "λ", lamb].iloc[0],
            "nzodis": df.loc[df["parameter"] == "nzodis", lamb].iloc[0],
            "snr": np.array([df.loc[df["parameter"] == "SNR", lamb].iloc[0]]),
            "resolution": np.array([5]),
            "toverhead_fixed": df.loc[
                df["parameter"] == "t_overhead,static", lamb
            ].iloc[0],
            "toverhead_multi": df.loc[
                df["parameter"] == "t_overhead,dynamic", lamb
            ].iloc[0],
            "DC": np.array([df.loc[df["parameter"] == "det_DC", lamb].iloc[0]]),
            "RN": np.array([df.loc[df["parameter"] == "det_RN", lamb].iloc[0]]),
            "CIC": np.array([df.loc[df["parameter"] == "det_CIC", lamb].iloc[0]]),
            "dQE": np.array([df.loc[df["parameter"] == "dQE", lamb].iloc[0]]),
            "QE": np.array([df.loc[df["parameter"] == "QE", lamb].iloc[0]]),
            "Toptical": np.array(
                [df.loc[df["parameter"] == "T_optical", lamb].iloc[0]]
            ),
            # "sollong": 135,
            # "ecllat": 72.7,
            "ra": float(hpic[hpic.hip_name == hip_name].ra.iloc[0]),
            "dec": float(hpic[hpic.hip_name == hip_name].dec.iloc[0]),
            "sp": lambda_d_to_arcsec(
                value_lod=df.loc[df["parameter"] == "sp", lamb].iloc[0],
                wavelength=np.array(df.loc[df["parameter"] == "λ", lamb].iloc[0] / 1000)
                * u.micron,
                diameter=df.loc[df["parameter"] == "D", lamb].iloc[0] * u.m,
            ).value,
            "CRb_multiplier": 2.0,
            "Fstar": np.array([df.loc[df["parameter"] == "F_star", lamb].iloc[0]]),
            "Fp": np.array([df.loc[df["parameter"] == "F_p", lamb].iloc[0]]),
            "observatory_preset": "EAC1",
            "observing_mode": "IMAGER",
            "delta_mag_min": 25,
            # These parameters are specific to agree with the ETC validation
            "nchannels": 1,
            "noisefloor_factor": 0.029,
            "epswarmTrcold": [0],  ##### done to turn off thermal noise
        }
        output = {
            "F0": np.array(df.loc[df["parameter"] == "F_0", lamb].iloc[0]),
            "magstar": np.array(df.loc[df["parameter"] == "m_lambda", lamb].iloc[0]),
            "Lstar": np.float64(df.loc[df["parameter"] == "L_star", lamb].iloc[0]),
            "dist": np.float64(df.loc[df["parameter"] == "dist", lamb].iloc[0]),
            "D": np.float64(df.loc[df["parameter"] == "D", lamb].iloc[0]),
            "A_cm": np.float64(df.loc[df["parameter"] == "A", lamb].iloc[0]),
            "lambda": np.float64(df.loc[df["parameter"] == "λ", lamb].iloc[0]),
            "deltalambda_nm": np.float64(df.loc[df["parameter"] == "Δλ", lamb].iloc[0]),
            "snr": np.float64(df.loc[df["parameter"] == "SNR", lamb].iloc[0]),
            "nzodis": np.float64(df.loc[df["parameter"] == "nzodis", lamb].iloc[0]),
            "toverhead_fixed": np.float64(
                df.loc[df["parameter"] == "t_overhead,static", lamb].iloc[0]
            ),
            "toverhead_multi": np.float64(
                df.loc[df["parameter"] == "t_overhead,dynamic", lamb].iloc[0]
            ),
            "det_DC": np.float64(df.loc[df["parameter"] == "det_DC", lamb].iloc[0]),
            "det_RN": np.float64(df.loc[df["parameter"] == "det_RN", lamb].iloc[0]),
            "det_CIC": np.float64(df.loc[df["parameter"] == "det_CIC", lamb].iloc[0]),
            "det_tread": np.float64(
                df.loc[df["parameter"] == "det_tread", lamb].iloc[0]
            ),
            "det_pixscale_mas": np.float64(
                df.loc[df["parameter"] == "det_pixscale", lamb].iloc[0]
            ),
            "dQE": np.float64(df.loc[df["parameter"] == "dQE", lamb].iloc[0]),
            "QE": np.float64(df.loc[df["parameter"] == "QE", lamb].iloc[0]),
            "Toptical": np.float64(
                df.loc[df["parameter"] == "T_optical", lamb].iloc[0]
            ),
            "Fstar": np.float64(df.loc[df["parameter"] == "F_star", lamb].iloc[0]),
            "Fp": np.float64(df.loc[df["parameter"] == "F_p", lamb].iloc[0]),
            "Fzodi": np.float64(df.loc[df["parameter"] == "F_zodi", lamb].iloc[0]),
            "Fexozodi": np.array(df.loc[df["parameter"] == "F_exozodi", lamb]),
            "sp_lod": np.array(df.loc[df["parameter"] == "sp", lamb]),
            "omega_lod": np.float64(df.loc[df["parameter"] == "Ω_core", lamb].iloc[0]),
            "T_core or photap_frac": np.float64(
                df.loc[df["parameter"] == "T_core", lamb].iloc[0]
            ),
            "Istar*oneopixscale2 in (l/D)^-2": np.float64(
                df.loc[df["parameter"] == "I_star", lamb].iloc[0]
            ),
            # "contrast * offset PSF peak *oneopixscale2  in (l/D)^-2 (unused)": np.float64(
            #     3.9e-14
            # ),
            "skytrans*oneopixscale2  in (l/D)^-2": np.float64(
                df.loc[df["parameter"] == "skytrans", lamb].iloc[0]
            ),
            "det_npix": np.float64(df.loc[df["parameter"] == "det_npix", lamb].iloc[0]),
            "t_photon_count": np.float64(
                df.loc[df["parameter"] == "t_photon_count_ETCVALIDATION", lamb]
            ),
            "CRp": np.float64(df.loc[df["parameter"] == "CR_p", lamb].iloc[0]),
            "CRbs": np.float64(df.loc[df["parameter"] == "CR_bs", lamb].iloc[0]),
            "CRbz": np.float64(df.loc[df["parameter"] == "CR_bz", lamb].iloc[0]),
            "CRbez": np.array(df.loc[df["parameter"] == "CR_bez", lamb]),
            "CRbbin": np.float64(df.loc[df["parameter"] == "CR_bstray", lamb].iloc[0]),
            "CRbd": np.float64(df.loc[df["parameter"] == "CR_bd", lamb].iloc[0]),
            "CRnf": np.float64(df.loc[df["parameter"] == "CR_NF", lamb].iloc[0]),
            "sciencetime": np.float64(
                df.loc[df["parameter"] == "t_science", lamb].iloc[0]
            ),
            "exptime": np.float64(df.loc[df["parameter"] == "t_exp", lamb].iloc[0]),
        }
        print(np.round(input["angdiam"], 4))
        run_single_test(name_str, input, output)


names = ["HIP 32439", "HIP 77052", "HIP 79672", "HIP 26779", "HIP 113283"]
for name in names:
    print("NAME", name)
    read_from_excel_sheet(name)
