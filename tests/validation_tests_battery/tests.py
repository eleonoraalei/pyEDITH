import os
import pandas as pd
import numpy as np
from astropy import units as u
from pyEDITH import calculate_texp, lambda_d_to_arcsec
from pyEDITH.units import *
import matplotlib.pyplot as plt

# Load HPIC
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(script_dir, "ETC_cal_detect.xlsx")
hpic = pd.read_csv(os.path.join(script_dir, "full_HPIC.txt"), sep="|")


def to_arcsec(quantity, observer_distance):
    return np.arctan(quantity / observer_distance).to(u.arcsec).value


def fluxes_to_magnitudes(F_star, F_p, F0):
    return -2.5 * np.log10(np.array(F_p) / np.array(F_star))


def process_star(name):
    hip_name = int(name.strip("HIP "))

    for wavelength in ["500nm", "1000nm"]:
        print(f"\nProcessing {name} at {wavelength}")

        if wavelength == "500nm":
            columns = "A,D,E,F"
        else:
            columns = "A,J,K,L"

        df = pd.read_excel(
            excel_file_path,
            sheet_name=" ".join([name, "(Detection)"]),
            skiprows=[1, 2, 3, 4, 5, 6],
            usecols=columns,
        )

        df.columns = ["parameter", "AYO", "EBS", "EXOSIMS"]

        # Prepare input parameters for pyEDITH calculation
        input_params = prepare_input_params(df, hpic, hip_name, "AYO")

        import logging

        yippy_logger = logging.getLogger("yippy")
        yippy_logger.setLevel(logging.ERROR)
        # Run pyEDITH calculation
        texp, validation_output = calculate_texp(
            input_params, verbose=False, ETC_validation=True
        )

        # Compare with Ayo's results
        compare_with_ayo(
            name, wavelength, validation_output[0], df[["parameter", "AYO"]]
        )
        compare_all_codes(name, wavelength, validation_output[0], df)

        print(f"Exposure time: {texp}")


def prepare_input_params(df, hpic, hip_name, code):

    input = {
        "F0": np.array([df.loc[df["parameter"] == "F_0", code].iloc[0]]),
        "mag": np.array([df.loc[df["parameter"] == "m_lambda", code].iloc[0]]),
        "Lstar": 10 ** float(hpic[hpic.hip_name == hip_name].st_lum.iloc[0]),
        "magV": float(hpic[hpic.hip_name == hip_name].sy_vmag.iloc[0]),
        "angular_diameter": np.round(
            to_arcsec(
                2
                * (float(hpic[hpic.hip_name == hip_name].st_rad.iloc[0]) * u.Rsun).to(
                    u.m
                ),
                float(hpic[hpic.hip_name == hip_name].sy_dist.iloc[0]) * u.pc,
            ),
            4,
        ),
        "distance": float(hpic[hpic.hip_name == hip_name].sy_dist.iloc[0]),
        "diameter": df.loc[df["parameter"] == "D", code].iloc[0],
        "unobscured_area": (1.0 - 0.121),
        "photap_rad": 0.85,
        # "psf_trunc_ratio": df.loc[df["parameter"] == "psf_trunc_ratio", code].iloc[0],
        "det_npix_input": np.float64(
            df.loc[df["parameter"] == "det_npix", code].iloc[0]
        ),
        "wavelength": np.array(
            [df.loc[df["parameter"] == "λ", code].iloc[0] / 1000]
        ),  # nm to micron
        "bandwidth": df.loc[df["parameter"] == "Δλ", code].iloc[0]
        / df.loc[df["parameter"] == "λ", code].iloc[0],
        "nzodis": df.loc[df["parameter"] == "nzodis", code].iloc[0],
        "snr": np.array([df.loc[df["parameter"] == "SNR", code].iloc[0]]),
        "toverhead_fixed": df.loc[df["parameter"] == "t_overhead,static", code].iloc[0],
        "toverhead_multi": df.loc[df["parameter"] == "t_overhead,dynamic", code].iloc[
            0
        ],
        "DC": np.array([df.loc[df["parameter"] == "det_DC", code].iloc[0]]),
        "RN": np.array([df.loc[df["parameter"] == "det_RN", code].iloc[0]]),
        "CIC": np.array([df.loc[df["parameter"] == "det_CIC", code].iloc[0]]),
        "dQE": np.array([df.loc[df["parameter"] == "dQE", code].iloc[0]]),
        "QE": np.array([df.loc[df["parameter"] == "QE", code].iloc[0]]),
        "Toptical": np.array([df.loc[df["parameter"] == "T_optical", code].iloc[0]]),
        "ra": float(hpic[hpic.hip_name == hip_name].ra.iloc[0]),
        "dec": float(hpic[hpic.hip_name == hip_name].dec.iloc[0]),
        "separation": lambda_d_to_arcsec(
            value_lod=df.loc[df["parameter"] == "sp", code].iloc[0],
            wavelength=np.array(df.loc[df["parameter"] == "λ", code].iloc[0] / 1000)
            * u.micron,
            diameter=df.loc[df["parameter"] == "D", code].iloc[0] * u.m,
        ).value,
        "CRb_multiplier": 2.0,
        "Fstar": np.array([df.loc[df["parameter"] == "F_star", code].iloc[0]]),
        "Fp": np.array([df.loc[df["parameter"] == "F_p", code].iloc[0]]),
        "observatory_preset": "EAC1",
        "observing_mode": "IMAGER",
        "delta_mag_min": 25,
        "nchannels": 1,
        # "noisefloor_factor": 0.029,
        "epswarmTrcold": [0],
        "t_photon_count_input": np.float64(
            df.loc[df["parameter"] == "t_photon_count", code].iloc[0]
        ),
        "az_avg": True,
        "noisefloor_PPF": 1 / 0.029,
    }

    input["delta_mag"] = fluxes_to_magnitudes(input["Fstar"], input["Fp"], input["F0"])

    return input


def get_expected_output(df, code):
    return {
        "F0": np.array(df.loc[df["parameter"] == "F_0", code].iloc[0]),
        "magstar": np.array(df.loc[df["parameter"] == "m_lambda", code].iloc[0]),
        "Lstar": np.float64(df.loc[df["parameter"] == "L_star", code].iloc[0]),
        "dist": np.float64(df.loc[df["parameter"] == "dist", code].iloc[0]),
        "D": np.float64(df.loc[df["parameter"] == "D", code].iloc[0]),
        "A_cm": np.float64(df.loc[df["parameter"] == "A", code].iloc[0]),
        "wavelength": np.float64(df.loc[df["parameter"] == "λ", code].iloc[0]),
        "deltalambda_nm": np.float64(df.loc[df["parameter"] == "Δλ", code].iloc[0]),
        "snr": np.float64(df.loc[df["parameter"] == "SNR", code].iloc[0]),
        "nzodis": np.float64(df.loc[df["parameter"] == "nzodis", code].iloc[0]),
        "toverhead_fixed": np.float64(
            df.loc[df["parameter"] == "t_overhead,static", code].iloc[0]
        ),
        "toverhead_multi": np.float64(
            df.loc[df["parameter"] == "t_overhead,dynamic", code].iloc[0]
        ),
        "det_DC": np.float64(df.loc[df["parameter"] == "det_DC", code].iloc[0]),
        "det_RN": np.float64(df.loc[df["parameter"] == "det_RN", code].iloc[0]),
        "det_CIC": np.float64(df.loc[df["parameter"] == "det_CIC", code].iloc[0]),
        "det_tread": np.float64(df.loc[df["parameter"] == "det_tread", code].iloc[0]),
        "det_pixscale_mas": np.float64(
            df.loc[df["parameter"] == "det_pixscale", code].iloc[0]
        ),
        "dQE": np.float64(df.loc[df["parameter"] == "dQE", code].iloc[0]),
        "QE": np.float64(df.loc[df["parameter"] == "QE", code].iloc[0]),
        "Toptical": np.float64(df.loc[df["parameter"] == "T_optical", code].iloc[0]),
        "Fstar": np.float64(df.loc[df["parameter"] == "F_star", code].iloc[0]),
        "Fp": np.float64(df.loc[df["parameter"] == "F_p", code].iloc[0]),
        "Fzodi": np.float64(df.loc[df["parameter"] == "F_zodi", code].iloc[0]),
        "Fexozodi": np.array(df.loc[df["parameter"] == "F_exozodi", code]),
        "sp_lod": np.array(df.loc[df["parameter"] == "sp", code]),
        "omega_lod": np.float64(df.loc[df["parameter"] == "Ω_core", code].iloc[0]),
        "T_core or photap_frac": np.float64(
            df.loc[df["parameter"] == "T_core", code].iloc[0]
        ),
        "Istar*oneopixscale2 in (l/D)^-2": np.float64(
            df.loc[df["parameter"] == "I_star", code].iloc[0]
        ),
        # "contrast * offset PSF peak *oneopixscale2  in (l/D)^-2 (unused)": np.float64(
        #     3.9e-14
        # ),
        "skytrans*oneopixscale2  in (l/D)^-2": np.float64(
            df.loc[df["parameter"] == "skytrans", code].iloc[0]
        ),
        "det_npix": np.float64(df.loc[df["parameter"] == "det_npix", code].iloc[0]),
        # "t_photon_count_ETCVALIDATION": np.float64(
        #     df.loc[df["parameter"] == "t_photon_count", code].iloc[0]
        # ),
        "t_photon_count": np.float64(
            df.loc[df["parameter"] == "t_photon_count", code].iloc[0]
        ),
        "CRp": np.float64(df.loc[df["parameter"] == "CR_p", code].iloc[0]),
        "CRbs": np.float64(df.loc[df["parameter"] == "CR_bs", code].iloc[0]),
        "CRbz": np.float64(df.loc[df["parameter"] == "CR_bz", code].iloc[0]),
        "CRbez": np.array(df.loc[df["parameter"] == "CR_bez", code]),
        "CRbbin": np.float64(df.loc[df["parameter"] == "CR_bstray", code].iloc[0]),
        "CRbd": np.float64(df.loc[df["parameter"] == "CR_bd", code].iloc[0]),
        "CRnf": np.float64(df.loc[df["parameter"] == "CR_NF", code].iloc[0]),
        "sciencetime": np.float64(df.loc[df["parameter"] == "t_science", code].iloc[0]),
        "exptime": np.float64(df.loc[df["parameter"] == "t_exp", code].iloc[0]),
    }


def compare_with_ayo(name, lamb, pyedith_output, df):
    print(f"Comparing with Ayo's results for {name} at {lamb}")
    expected_output = get_expected_output(df, "AYO")
    errors = []

    for key, expected_value in expected_output.items():
        calculated_value = pyedith_output[key]
        if hasattr(calculated_value, "value"):
            calculated_value = calculated_value.value

        try:
            np.testing.assert_allclose(
                calculated_value,
                expected_value,
                rtol=1e-1,
                err_msg=f"Mismatch in {key} for test case: {name}",
            )
        except AssertionError as e:
            errors.append(
                f"-- {key}: FAILED - Expected: {expected_value}, Calculated: {calculated_value}"
            )

    if len(errors) == 0:
        print(f"Test case '{name}' at {lamb} passed successfully!")
    else:
        print(f"Test case '{name}' at {lamb} had some errors: \n" + "\n".join(errors))


def compare_all_codes(name, wavelength, pyedith_output, df):
    print(f"Comparing with all codes for {name} at {wavelength}")

    key_translation = {
        "Fstar": "F_star",
        "Fp": "F_p",
        "Fzodi": "F_zodi",
        "Fexozodi": "F_exozodi",
        "T_core or photap_frac": "T_core",
        "Istar*oneopixscale2 in (l/D)^-2": "I_star",
        "skytrans*oneopixscale2  in (l/D)^-2": "skytrans",
        "det_npix": "det_npix",
        "t_photon_count": "t_photon_count",
        "CRp": "CR_p",
        "CRbs": "CR_bs",
        "CRbez": "CR_bez",
        "CRbz": "CR_bz",
        "CRbd": "CR_bd",
        "CRnf": "CR_NF",
        "sciencetime": "t_science",
        "exptime": "t_exp",
    }
    renamed_pyedith_output = {}

    for old_key, value in pyedith_output.items():
        if old_key in key_translation:
            new_key = key_translation[old_key]
            renamed_pyedith_output[new_key] = value

    comparisons = {}
    for key in renamed_pyedith_output.keys():
        if key in df["parameter"].values:
            other_results = {
                code: df.loc[df["parameter"] == key, code].iloc[0]
                for code in ["AYO", "EBS", "EXOSIMS"]
            }
            comparisons[key] = compare_results(
                renamed_pyedith_output[key], other_results
            )

    visualize_comparisons(comparisons, name, wavelength)


def compare_results(pyedith_result, other_results):
    if hasattr(pyedith_result, "value"):
        pyedith_result = pyedith_result.value

    # calculate mean and std of the results from the other codes
    other_values = []
    for code in ["AYO", "EBS", "EXOSIMS"]:
        value = float(other_results[code])
        if not np.isnan(value):
            other_values.append(value)
        else:
            other_values.append(np.nan)

    mean = np.mean(other_values)
    std = np.std(other_values)

    return {
        "values": other_results,
        "pyedith": pyedith_result,
        "mean": mean,
        "std": std,
    }


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter


class ScientificNotationFormatter(ScalarFormatter):
    def __init__(self, useOffset=True, useMathText=True):
        super().__init__(useOffset=useOffset, useMathText=useMathText)
        self.set_scientific(True)
        self.set_powerlimits((-2, 2))

    def _set_format(self):
        self.format = "%1.1f"


def visualize_comparisons(comparisons, name, wavelength):
    filename = name + "_" + str(wavelength)
    n_cols = 6
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    # fig.suptitle(f"Comparisons for {name} at {wavelength}", fontsize=20, y=1.02)

    axes = axes.flatten()

    for i, (key, comparison) in enumerate(comparisons.items()):
        ax = axes[i]
        x = []
        y = []
        colors = []
        for j, code in enumerate(["AYO", "EBS", "EXOSIMS", "pyEDITH"]):
            try:
                if code == "pyEDITH":
                    value = float(comparison["pyedith"])
                else:
                    value = float(comparison["values"][code])
                if not np.isnan(value):
                    x.append(j)
                    y.append(value)
                    colors.append(["blue", "green", "red", "purple"][j])
                else:
                    ax.text(
                        j,
                        0.5,
                        "X",
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color=["blue", "green", "red", "purple"][j],
                    )
            except ValueError:
                ax.text(
                    j,
                    0.5,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=["blue", "green", "red", "purple"][j],
                )

        if y:
            ax.scatter(x, y, c=colors, s=80)
            if not np.isnan(comparison["mean"]):
                ax.axhline(
                    y=comparison["mean"],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label="Mean",
                )
                ax.axhspan(
                    comparison["mean"] - comparison["std"],
                    comparison["mean"] + comparison["std"],
                    alpha=0.2,
                    color="gray",
                    label="Std Dev",
                )

            # Set y-axis limits to zoom in on the data points
            y_min, y_max = min(y), max(y)
            y_range = y_max - y_min
            y_padding = 1 * y_range  # Add 100% padding
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Set y-axis to log scale if the values span more than 2 orders of magnitude
            if len(set(y)) > 1:
                if max(y) / min(y) > 100:
                    ax.set_yscale("log")
                    # For log scale, adjust limits to ensure all points are visible
                    ax.set_ylim(y_min / 1.1, y_max * 1.1)
                else:
                    ax.set_ylim(
                        bottom=max(0, y_min - y_padding)
                    )  # Ensure bottom limit is not negative for linear scale

            # Set custom formatter for y-axis ticks
            formatter = ScientificNotationFormatter()
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.offsetText.set_fontsize(7)

        else:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")

        ax.set_title(key, fontsize=12)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["AYO", "EBS", "EXO", "pyE"], rotation=45, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Only show legend for the first subplot
        # if i == 0:
        #     ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))

    # Remove any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
    plt.savefig(os.path.join(script_dir, filename + ".png"))


names = ["HIP 32439", "HIP 77052", "HIP 79672", "HIP 26779", "HIP 113283"]
for name in names:
    print("NAME", name)
    process_star(name)
