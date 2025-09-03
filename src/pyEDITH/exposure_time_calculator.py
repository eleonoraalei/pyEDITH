from typing import Tuple
import numpy as np
from pyEDITH import AstrophysicalScene, Observation, Observatory
from pyEDITH.components.coronagraphs import CoronagraphYIP, ToyModelCoronagraph
import astropy.constants as c
import astropy.units as u
from astropy.modeling import models
from .units import *
from . import utils
import pickle


def calculate_CRp(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    Fp_over_Fs: u.Quantity,
    area: u.Quantity,
    Upsilon: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the planet count rate.

    This function computes the detected count rate from a planet based on
    the stellar and planetary flux, telescope characteristics, and coronagraph
    performance parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux [dimensionless]
    Fp_over_Fs : u.Quantity
        Planet flux relative to star [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    Upsilon : u.Quantity
        Core throughput of the coronagraph [dimensionless]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    nchannels : int
        Number of channels

    Returns
    -------
    u.Quantity
        Planet count rate [electrons / s]
    """

    return (
        F0 * Fs_over_F0 * Fp_over_Fs * area * Upsilon * throughput * dlambda * nchannels
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbs(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    Istar: u.Quantity,
    area: u.Quantity,
    pixscale: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the stellar leakage count rate.

    This function computes the detected count rate from stellar leakage based
    on the stellar flux, coronagraph performance, and telescope parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux [dimensionless]
    Istar : u.Quantity
        Stellar intensity at the given pixel [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    pixscale : u.Quantity
        Pixel scale of the detector [lambda/D]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    nchannels : int
        Number of channels

    Returns
    -------
    u.Quantity
        Stellar leakage count rate [electrons / s]
    """

    return (
        F0
        * Fs_over_F0
        * Istar
        * area
        * throughput
        * dlambda
        * nchannels
        / (pixscale**2)
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbz(
    F0: u.Quantity,
    Fzodi: u.Quantity,
    lod_arcsec: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the local zodiacal light count rate.

    This function computes the detected count rate from local zodiacal light
    based on the zodiacal intensity, sky transmission, and telescope parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fzodi : u.Quantity
        Zodiacal light flux [dimensionless]
    lod_arcsec : u.Quantity
        Lambda/D in arcseconds [arcsec]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    nchannels : int
        Number of channels

    Returns
    -------
    u.Quantity
        Local zodiacal light count rate [electrons / s]
    """

    return (
        F0 * Fzodi * skytrans * area * throughput * dlambda * nchannels * lod_arcsec**2
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbez(
    F0: u.Quantity,
    Fexozodi: u.Quantity,
    lod_arcsec: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
    dist: u.Quantity,
    sp: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodiacal light count rate.

    This function computes the detected count rate from exozodiacal light
    based on the exozodiacal intensity, system geometry, and telescope parameters.
    It scales the exozodiacal intensity based on the distance to the star
    and the angular separation.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fexozodi : u.Quantity
        Exozodiacal light flux at reference position [dimensionless]
    lod_arcsec : u.Quantity
        Lambda/D in arcseconds [arcsec]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    nchannels : int
        Number of channels
    dist : u.Quantity
        Distance to the star [pc]
    sp : u.Quantity
        Separation of the planet [arcsec]

    Returns
    -------
    u.Quantity
        Exozodiacal light count rate [electrons / s]
    """
    # Calculate Fexozodi at the separation (scale the value of Fexozodi at 1 AU
    # to the separation in AU)
    scaling_factor = u.AU / arcsec_to_au(sp, dist)
    return (
        F0
        * (Fexozodi * scaling_factor**2)
        * skytrans
        * area
        * throughput
        * dlambda
        * nchannels
        * lod_arcsec**2
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )  # this is to simplify the arcsec^2/arcsec^2 that somehow does not simplify by itself


def calculate_CRbbin(
    F0: u.Quantity,
    Fbinary: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the count rate from neighboring stars.

    This function computes the detected count rate from binary or neighboring stars
    based on their flux, sky transmission, and telescope parameters.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fbinary : u.Quantity
        Flux from neighboring stars [dimensionless]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE) [electrons/photons]
    dlambda : u.Quantity
        Bandwidth [um]
    nchannels : int
        Number of channels

    Note
    ----
    IMPORTANT: Currently Fbinary is 0 by default, so this term is zero.
    It will need to be checked again in the future.

    Returns
    -------
    u.Quantity
        Count rate from neighboring stars [electrons / s]
    """

    return (F0 * Fbinary * skytrans * area * throughput * dlambda * nchannels).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbth(
    lam: u.Quantity,
    area: u.Quantity,
    dlambda: u.Quantity,
    temp: u.Quantity,
    lod_rad: u.Quantity,
    emis: u.Quantity,
    QE: u.Quantity,
    dQE: u.Quantity,
) -> u.Quantity:
    """
    Calculate background thermal count rate.

    This function computes the detected count rate from thermal emission
    of the telescope and instrument components based on their temperature,
    emissivity, and other system parameters. It uses a blackbody radiation
    model to calculate the thermal photon flux.

    Parameters
    ----------
    lam : u.Quantity
        Wavelength of observation [um]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    dlambda : u.Quantity
        Bandwidth [um]
    temp : u.Quantity
        Telescope mirror temperature [K]
    lod_rad : u.Quantity
        Lambda/D in radians [rad]
    emis : u.Quantity
        Effective emissivity for the observing system [dimensionless]
    QE : u.Quantity
        Quantum efficiency [electron/photon]
    dQE : u.Quantity
        Effective QE due to degradation [dimensionless]

    Returns
    -------
    u.Quantity
        Count rate from thermal background [electrons / s]
    """

    # Calculate blackbody radiation
    bb = models.BlackBody(
        temperature=temp, scale=1 * u.erg / (u.cm**2 * u.AA * u.s * u.sr)
    )
    Blambda_energy = bb(lam)

    # Convert to photon spectral radiance
    Blambda_photon = (Blambda_energy).to(
        u.photon / (u.cm**2 * u.nm * u.s * u.sr), equivalencies=u.spectral_density(lam)
    )

    # Calculate thermal background count rate
    return (Blambda_photon * dlambda * area * (lod_rad * lod_rad) * emis * QE * dQE).to(
        u.electron / u.s
    )


def calculate_t_photon_count(
    det_npix: u.Quantity,
    det_CR: u.Quantity,
) -> u.Quantity:
    """
    Calculate the photon counting time.

    This function computes the average time needed to detect one photon per pixel
    based on the detector count rate and number of pixels.

    Parameters
    ----------
    det_npix : u.Quantity
        Number of detector pixels [pix]
    det_CR : u.Quantity
        Detector count rate [photons / s]

    Returns
    -------
    u.Quantity
        Photon counting time (average time to detect one photon per pixel) [s * pixel / ph]
    """

    counts_per_second_per_pixel = det_CR / det_npix  #  electron / s / pix
    # NOTE: I am just extrapolating that 6.73 has units [pix*frame/electron]
    t_photon_count = 1.0 / (
        6.73 * PIXEL * FRAME / ELECTRON * counts_per_second_per_pixel
    )
    return t_photon_count


def calculate_CRbd(
    det_npix: u.Quantity,
    det_DC: u.Quantity,
    det_RN: u.Quantity,
    det_tread: u.Quantity,
    det_CIC: u.Quantity,
    t_photon_count: u.Quantity,
) -> u.Quantity:
    """
    Calculate the detector noise count rate.

    This function computes the total detector noise count rate by combining
    contributions from dark current, read noise, and clock-induced charge.

    Parameters
    ----------
    det_npix : u.Quantity
        Number of detector pixels [pix]
    det_DC : u.Quantity
        Dark current [electron / pix / s]
    det_RN : u.Quantity
        Read noise [electron / pix / read]
    det_tread : u.Quantity
        Read time [s]
    det_CIC : u.Quantity
        Clock-induced charge [electron / pix / photon]
    t_photon_count : u.Quantity
        Photon counting time [pix * s / electron]

    Returns
    -------
    u.Quantity
        Detector noise count rate [electrons / s]
    """

    # Using the variance of the read noise but keeping the same units as det_RN alone.
    read_noise_variance = det_RN * det_RN.value
    return (
        (det_DC + read_noise_variance / det_tread + det_CIC / t_photon_count) * det_npix
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRnf(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    area: u.Quantity,
    pixscale: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
    SNR: u.Quantity,
    noisefloor: u.Quantity,
) -> u.Quantity:
    """
    Calculate the noise floor count rate.

    This function computes the count rate corresponding to the noise floor
    based on the stellar flux, telescope parameters, and the specified noise floor level.
    The noise floor represents the limiting systematic noise that cannot be reduced
    through longer integration times.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux [dimensionless]
    area : u.Quantity
        Collecting area of the telescope [cm^2]
    pixscale : u.Quantity
        Pixel scale of the detector [lambda/D]
    throughput : u.Quantity
        Throughput of the system [dimensionless]
    dlambda : u.Quantity
        Bandwidth [um]
    nchannels : int
        Number of channels
    SNR : float
        Signal-to-noise ratio
    noisefloor : u.Quantity
        Noise floor level [dimensionless]

    Returns
    -------
    u.Quantity
        Noise floor count rate [photons / s]
    """

    return (
        SNR
        * (F0 * Fs_over_F0 * area * throughput * dlambda * nchannels / (pixscale**2))
        * noisefloor
    )


def calculate_CRnf_ez(
    CRbez: u.Quantity,
    SNR: u.Quantity,
    ez_PPF: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodi noise floor count rate.

    This function computes the noise floor contribution from exozodiacal light
    when it cannot be subtracted to the Poisson noise limit. It accounts for
    post-processing capabilities through the ez_PPF factor.

    Parameters
    ----------
    CRbez : u.Quantity
        Count rate of the exozodi [photons / s]
    SNR : float
        Signal-to-noise ratio
    ez_PPF : u.Quantity
        Post-processing factor for exozodi [dimensionless]

    Returns
    -------
    u.Quantity
        Exozodi noise floor count rate [photons / s]
    """

    return SNR * CRbez / ez_PPF


def measure_coronagraph_performance_at_IWA(
    photometric_aperture_throughput: u.Quantity,
    Istar_interp: u.Quantity,
    skytrans: u.Quantity,
    omega_lod: u.Quantity,
    npix: int,
    xcenter: u.Quantity,
    ycenter: u.Quantity,
    oneopixscale_arcsec: u.Quantity,
) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    Measure the performance of the coronagraph at the Inner Working Angle (IWA).

    This function determines the IWA and calculates various coronagraph performance
    parameters at that point. It identifies the IWA by finding where the photometric
    aperture throughput falls to half its maximum value and then measures stellar
    intensity, sky transmission, and other parameters in a 2-pixel annulus at the IWA.

    Parameters
    ----------
    photometric_aperture_throughput : u.Quantity
        Photometric aperture fractions [dimensionless]
    Istar_interp : u.Quantity
        Interpolated stellar intensity [dimensionless]
    skytrans : u.Quantity
        Sky transmission [dimensionless]
    omega_lod : u.Quantity
        Solid angle of photometric aperture [(lambda/D)^2]
    npix : int
        Number of pixels in each dimension
    xcenter : u.Quantity
        X-coordinate of the center [pixel]
    ycenter : u.Quantity
        Y-coordinate of the center [pixel]
    oneopixscale_arcsec : u.Quantity
        Inverse of pixel scale [1/arcsec]

    Returns
    -------
    Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]
        det_sep_pix: Separation at the IWA [pixel]
        det_sep: Separation at the IWA [arcsec]
        det_Istar: Maximum stellar intensity at the IWA [dimensionless]
        det_skytrans: Maximum sky transmission at the IWA [dimensionless]
        det_photometric_aperture_throughput: Maximum photometric aperture fraction at the IWA [dimensionless]
        det_omega_lod: Solid angle corresponding to max photometric_aperture_throughput at the IWA [(lambda/D)^2]
    """

    # Find psf_trunc_ratio closest to 0.3
    # Commenting this out for now since this is not used.
    # bestiratio = np.argmin(
    #     np.abs(psf_trunc_ratio - 0.3)
    # )  # NOT USED, in EDITH only one psf_trunc_ratio
    bestiratio = 0  # len = 1 array, so only one index to choose

    # Find maximum photometric_aperture_throughput in first half of image
    maxphotometric_aperture_throughput = np.max(
        photometric_aperture_throughput[: npix // 2, int(ycenter.value), bestiratio]
    )

    # Find IWA = where photometric_aperture_throughput is half the value of the maximum
    row = photometric_aperture_throughput[:, int(ycenter.value), bestiratio]
    iwa_index = np.where(
        row[: int(xcenter.value)] > 0.5 * maxphotometric_aperture_throughput
    )[0][-1]
    det_sep_pix = abs((iwa_index + 0.5) - xcenter.value) * PIXEL
    det_sep = det_sep_pix / oneopixscale_arcsec  # translates to arcsec

    # Calculate max values in 2-pixel annulus at det_sep
    y, x = np.ogrid[:npix, :npix]
    dist_from_center = (
        np.sqrt((x - xcenter.value + 0.5) ** 2 + (y - ycenter.value + 0.5) ** 2)
    ) * PIXEL

    mask = np.abs(dist_from_center.value - det_sep_pix.value) < 2
    # NOTE: We use .value because we want indices

    det_Istar = np.max(Istar_interp[mask]) * DIMENSIONLESS
    det_skytrans = np.max(skytrans[mask]) * DIMENSIONLESS

    photometric_aperture_throughput_masked = photometric_aperture_throughput[
        :, :, bestiratio
    ][mask]
    det_photometric_aperture_throughput = (
        np.max(photometric_aperture_throughput_masked) * DIMENSIONLESS
    )
    det_omega_lod = omega_lod[:, :, bestiratio][mask][
        np.argmax(photometric_aperture_throughput_masked)
    ]

    return (
        det_sep_pix,
        det_sep,
        det_Istar,
        det_skytrans,
        det_photometric_aperture_throughput,
        det_omega_lod,
    )


def calculate_exposure_time_or_snr(
    observation: Observation,
    scene: AstrophysicalScene,
    observatory: Observatory,
    verbose: bool,
    ETC_validation: bool = False,
    mode: str = "exposure_time",
) -> None:
    """
    Calculate exposure time or signal-to-noise ratio for an observation.

    This function performs detailed calculations of exposure time or signal-to-noise
    ratio for each wavelength in an observation, accounting for multiple noise sources,
    coronagraph performance, and detector characteristics. The function handles both
    'exposure_time' mode (calculating required exposure time for a given SNR) and
    'signal_to_noise' mode (calculating achievable SNR for a given exposure time).
    The function stores calculated photon counts, exposure times or SNR values
    directly in the observation object. For planets outside the working angle
    range or below the noise floor, infinity values are assigned.

    Parameters
    ----------
    observation : Observation
        Observation object containing observation parameters including wavelength,
        target SNR or exposure time, and bandwidth information
    scene : AstrophysicalScene
        AstrophysicalScene object containing scene parameters including planet
        contrast, stellar properties, and zodiacal light levels
    observatory : Observatory
        Observatory object containing telescope, detector, and coronagraph parameters
    verbose : bool
        If True, print detailed calculation information to the console
    ETC_validation : bool, optional
        If True, use specific parameter values for validation against the ETC,
        default is False
    mode : str, optional
        Calculation mode, either 'exposure_time' (to calculate required exposure
        time for a given SNR) or 'signal_to_noise' (to calculate achievable SNR
        for a given exposure time), default is 'exposure_time'

    Raises
    ------
    ValueError
        If an invalid mode is specified or if the observing_mode is not
        'IMAGER' or 'IFS'

    """

    # Check modes
    if mode not in ["exposure_time", "signal_to_noise"]:
        raise ValueError("Invalid mode. Use 'exposure_time' or 'signal_to_noise'.")

    observation.validation_variables = {}
    observation.photon_counts = {
        "CRp": np.empty(observation.nlambd),
        "CRbs": np.empty(observation.nlambd),
        "CRbz": np.empty(observation.nlambd),
        "CRbez": np.empty(observation.nlambd),
        "CRbbin": np.empty(observation.nlambd),
        "CRbth": np.empty(observation.nlambd),
        "CRbd": np.empty(observation.nlambd),
        "CRnf_s": np.empty(observation.nlambd),
        "CRnf_ez": np.empty(observation.nlambd),
        "CRnf": np.empty(observation.nlambd),
        "CRb": np.empty(observation.nlambd),
        "omega_lod": np.empty(observation.nlambd),
        "PPF_ez": np.empty(observation.nlambd),
    }

    for ilambd in range(observation.nlambd):

        # Take the lesser of the desired bandwidth
        # and what coronagraph allows
        if observatory.observing_mode == "IMAGER":
            deltalambda_nm = (
                np.min(
                    [
                        (observation.wavelength[ilambd].to(u.nm).value)
                        / observatory.coronagraph.coronagraph_spectral_resolution,
                        observatory.coronagraph.bandwidth
                        * (observation.wavelength[ilambd].to(u.nm).value),
                    ]
                )
                * u.nm
            )  # nanometers
            if (
                observatory.coronagraph.bandwidth
                * observation.wavelength[ilambd].to(u.nm).value
                >= observation.wavelength[ilambd].to(u.nm).value
                / observatory.coronagraph.coronagraph_spectral_resolution
            ):
                print(
                    "WARNING: Bandwidth larger than what the coronagraph allows. Selecting widest possible bandwidth..."
                )
        elif observatory.observing_mode == "IFS":
            # the effective bandwidth is the width of the spectral element
            deltalambda_nm = observation.delta_wavelength[ilambd].to(u.nm)
        else:
            raise ValueError("Invalid observation mode. Choose 'IMAGER' or 'IFS'.")

        # Calculate λ/D (dimensionless)
        lod = 1 * LAMBDA_D

        # Convert to radians
        # NOTE: using LENGTH here ensures that if we change the value of the unit,
        # this is still dimensionless
        lod_rad = lambda_d_to_radians(
            lod,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )

        # Convert to arcseconds
        lod_arcsec = lod_rad.to(u.arcsec)

        area_cm2 = observatory.telescope.Area.to(u.cm**2)

        detpixscale_lod = arcsec_to_lambda_d(
            observatory.detector.pixscale_mas.to(u.arcsec),
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )  # LAMBDA_D units

        stellar_diam_lod = arcsec_to_lambda_d(
            scene.stellar_angular_diameter_arcsec,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )  # LAMBDA_D units

        """
        WE DO NOT INTERPOLATE ANYMORE, INTERPOLATION IS DONE WITHIN YIPPY 
        # Interpolate Istar, noisefloor based on angular diameter
        # of the star (depends on the target). It reduces dimensionality
        # from 3D arrays [npix,npix,angdiam] to 2D arrays [npix,npix].
        # The interpolation is done based on the value of
        # stellar_diam_lod (dependence on istar)

        Istar_interp, noisefloor_interp = interpolate_arrays(
            observatory.coronagraph.Istar,
            observatory.coronagraph.noisefloor,
            observatory.coronagraph.npix,
            observatory.coronagraph.ndiams,
            stellar_diam_lod,
            observatory.coronagraph.angdiams,
        )
        """

        # Measure coronagraph performance at each IWA
        pixscale_rad = observatory.coronagraph.pixscale * lambda_d_to_radians(
            lod,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )  # going from LAMBDA_D to radians

        oneopixscale_arcsec = 1 * PIXEL / pixscale_rad.to(u.arcsec)

        # Measure coronagraph performance at each IWA
        (
            det_sep_pix,
            det_sep,
            det_Istar,
            det_skytrans,
            det_photometric_aperture_throughput,
            det_omega_lod,
        ) = measure_coronagraph_performance_at_IWA(
            # observation.psf_trunc_ratio, # this is no longer used in the function
            observatory.coronagraph.photometric_aperture_throughput,
            observatory.coronagraph.Istar,
            observatory.coronagraph.skytrans,
            observatory.coronagraph.omega_lod,
            observatory.coronagraph.npix,
            observatory.coronagraph.xcenter,
            observatory.coronagraph.ycenter,
            oneopixscale_arcsec,
        )

        if ETC_validation:
            print("Fixing det_npix for validation...")

            det_npix = observatory.detector.det_npix_input * PIXEL
        else:
            #  Calculate det_npix
            det_npix = (
                observatory.detector.npix_multiplier[ilambd]
                * det_omega_lod
                / (detpixscale_lod**2)
                * observatory.coronagraph.nchannels
            ) * PIXEL  # number of pixels in detector

        # Here we calculate detector noise, as it may depend on count rates
        # We don't know the count rates yet, so we make estimates based on
        # values near the IWA

        # Detector noise from signal itself (we budget for 10x
        # the planet count rate for the minimum detectable planet)
        det_CRp = calculate_CRp(
            scene.F0[ilambd],
            scene.Fs_over_F0[ilambd],
            10 * scene.Fp_min_over_Fs,
            area_cm2,
            det_photometric_aperture_throughput,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbs = calculate_CRbs(
            scene.F0[ilambd],
            scene.Fs_over_F0[ilambd],
            det_Istar,
            area_cm2,
            observatory.coronagraph.pixscale,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbz = calculate_CRbz(
            scene.F0[ilambd],
            scene.Fzodi_list[ilambd],
            lod_arcsec,
            det_skytrans,
            area_cm2,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbez = calculate_CRbez(
            scene.F0[ilambd],
            scene.Fexozodi_list[ilambd],
            lod_arcsec,
            det_skytrans,
            area_cm2,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
            scene.dist,
            det_sep,
        )

        det_CRbbin = calculate_CRbbin(
            scene.F0[ilambd],
            scene.Fbinary_list[ilambd],
            det_skytrans,
            area_cm2,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbth = (
            calculate_CRbth(
                observation.wavelength[ilambd],
                area_cm2,
                deltalambda_nm,
                observatory.telescope.temperature,
                lod_rad,
                observatory.epswarmTrcold[ilambd],
                observatory.detector.QE[ilambd,],
                observatory.detector.dQE[ilambd,],
            )
            * det_omega_lod
        )

        det_CR = det_CRp + det_CRbs + det_CRbz + det_CRbez + det_CRbbin + det_CRbth

        # Calculate position of the planet in the image
        # (from l/D to pixel)
        ix = (
            scene.xp * oneopixscale_arcsec + observatory.coronagraph.xcenter
        ).value  # this is the "index" of the position in pixel, i.e. the number of the pixel where the planet is
        iy = (
            scene.yp * oneopixscale_arcsec + observatory.coronagraph.ycenter
        ).value  # this is the "index" of the position in pixel, i.e. the number of the pixel where the planet is

        # Calculate separation (from arcsec to l/D)
        sp_lod = arcsec_to_lambda_d(
            scene.separation,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )

        # If planet is within the boundaries of the observatory.coronagraph
        # simulation and hard IWA/OWA cutoffs...
        if (
            (ix >= 0)  # check that x pixel is positive
            and (
                ix < observatory.coronagraph.npix
            )  # check that it is less than the maximum pixel number
            and (iy >= 0)  # check that the y pixel is positive
            and (
                iy < observatory.coronagraph.npix
            )  # check that it is less than the maximum pixel number
            and (
                sp_lod > observatory.coronagraph.minimum_IWA
            )  # check that the separation in l/D is more than the minimum allowed IWA
            and (
                sp_lod < observatory.coronagraph.maximum_OWA
            )  # check that the separation in l/D is less than the maximum allowed OWA
        ):

            for iratio in np.arange(observatory.coronagraph.npsfratios):
                # First we just calculate CRp and CRnoisefloor
                # to see if CRp > CRnoisefloor

                # PLANET COUNT RATE CRP
                CRp = calculate_CRp(
                    scene.F0[ilambd],
                    scene.Fs_over_F0[ilambd],
                    scene.Fp_over_Fs[ilambd],
                    area_cm2,
                    observatory.coronagraph.photometric_aperture_throughput[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ],
                    observatory.total_throughput[ilambd],
                    deltalambda_nm,
                    observatory.coronagraph.nchannels,
                )
                observation.photon_counts["CRp"][ilambd] = CRp.value

                # Calculate CRbez; this must happen here in order to estimate the exozodi noisefloor
                CRbez = calculate_CRbez(
                    scene.F0[ilambd],
                    scene.Fexozodi_list[ilambd],
                    lod_arcsec,
                    observatory.coronagraph.skytrans[
                        int(np.floor(iy)), int(np.floor(ix))
                    ],
                    area_cm2,
                    observatory.total_throughput[ilambd],
                    deltalambda_nm,
                    observatory.coronagraph.nchannels,
                    scene.dist,
                    scene.separation,
                )
                observation.photon_counts["CRbez"][ilambd] = (
                    CRbez.value
                    * observatory.coronagraph.omega_lod[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ].value
                )

                observation.photon_counts["omega_lod"][ilambd] = (
                    observatory.coronagraph.omega_lod[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ].value
                )

                # NOISE FLOOR CRNF
                if mode == "exposure_time":
                    # NOISE FLOOR CRNF
                    CRnf_s = calculate_CRnf(
                        scene.F0[ilambd],
                        scene.Fs_over_F0[ilambd],
                        area_cm2,
                        observatory.coronagraph.pixscale,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                        observation.SNR[ilambd],
                        observatory.coronagraph.noisefloor[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                    )

                    # calculate the exozodi noisefloor to account for imperfect exozodi removal
                    CRnf_ez = calculate_CRnf_ez(
                        CRbez
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value,
                        observation.SNR[ilambd],
                        scene.ez_PPF[ilambd],
                    )

                elif mode == "signal_to_noise":
                    #  NOTE THIS TIME THIS IS JUST THE NOISE
                    # FACTOR RATIO (i.e. we assume SNR =1 so
                    # that we can use it for the snr calculation later)

                    CRnf_s = calculate_CRnf(
                        scene.F0[ilambd],
                        scene.Fs_over_F0[ilambd],
                        area_cm2,
                        observatory.coronagraph.pixscale,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                        1,
                        observatory.coronagraph.noisefloor[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                    )
                    CRnf_ez = calculate_CRnf_ez(CRbez, 1, scene.ez_PPF[ilambd])

                # multiply by omega at that point
                CRnf_s *= observatory.coronagraph.omega_lod[
                    int(np.floor(iy)), int(np.floor(ix)), iratio
                ]
                observation.photon_counts["CRnf_s"][ilambd] = CRnf_s.value

                CRnf_ez *= observatory.coronagraph.omega_lod[
                    int(np.floor(iy)), int(np.floor(ix)), iratio
                ]
                observation.photon_counts["CRnf_ez"][ilambd] = CRnf_ez.value

                # total noisefloor
                CRnf = np.sqrt(CRnf_s**2 + CRnf_ez**2)
                observation.photon_counts["CRnf"][ilambd] = CRnf.value

                # NOTE: noisefloor_interp: technically the Y axis
                # is rows and the X axis is columns,
                # that is why they are inverted
                # NOTE: Evaluate if int(round(iy)) is better than
                # np.floor. Kept np.floor for consistency

                # Check if photometric aperture is large enough:
                if (
                    observatory.coronagraph.omega_lod[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ]
                    > detpixscale_lod**2
                ):

                    # (for exposure time mode) Check if it's above the noise floor
                    if mode == "exposure_time" and CRp <= CRnf:
                        print(
                            "WARNING: Count rate of the planet smaller than the noise floor. Hardcoded infinity results."
                        )

                        observation.exptime[ilambd] = np.inf
                        continue  # Skip to next iteration

                    # Calculate the rest of the background noise

                    # NOTE: WHEN CALCULATING THE COUNT RATES,
                    # WE NEED TO MULTIPLY BY OMEGA_LOD i.e.
                    # THE SOLID ANGLE OF THE PHOTOMETRIC APERTURE

                    # Calculate CRbs
                    CRbs = calculate_CRbs(
                        scene.F0[ilambd],
                        scene.Fs_over_F0[ilambd],
                        observatory.coronagraph.Istar[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.coronagraph.pixscale,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                    )
                    observation.photon_counts["CRbs"][ilambd] = (
                        CRbs.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbz
                    CRbz = calculate_CRbz(
                        scene.F0[ilambd],
                        scene.Fzodi_list[ilambd],
                        lod_arcsec,
                        observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                    )
                    observation.photon_counts["CRbz"][ilambd] = (
                        CRbz.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbbin
                    CRbbin = calculate_CRbbin(
                        scene.F0[ilambd],
                        scene.Fbinary_list[ilambd],
                        observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                    )
                    observation.photon_counts["CRbbin"][ilambd] = (
                        CRbbin.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbd
                    t_photon_count = calculate_t_photon_count(
                        det_npix,
                        det_CR,
                    )
                    if ETC_validation:
                        print("Fixing t_photon_count for validation...")
                        # the ETC validation (Stark+2025) fixed the frame rate
                        # t_photon_count = 1 / (det_CRp.value) * SECOND / FRAME
                        t_photon_count = observatory.detector.t_photon_count_input

                    CRbd = calculate_CRbd(
                        det_npix,
                        observatory.detector.DC[ilambd],
                        observatory.detector.RN[ilambd],
                        observatory.detector.tread[ilambd],
                        observatory.detector.CIC[ilambd],
                        t_photon_count,
                    )

                    observation.photon_counts["CRbd"][ilambd] = CRbd.value

                    CRbth = calculate_CRbth(
                        observation.wavelength[ilambd],
                        area_cm2,
                        deltalambda_nm,
                        observatory.telescope.temperature,
                        lod_rad,
                        observatory.epswarmTrcold[ilambd],
                        observatory.detector.QE[ilambd,],
                        observatory.detector.dQE[ilambd,],
                    )
                    observation.photon_counts["CRbth"][ilambd] = (
                        CRbth.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # TOTAL BACKGROUND NOISE
                    CRb = (
                        CRbs + CRbz + CRbez + CRbbin + CRbth
                    ) * observatory.coronagraph.omega_lod[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ]
                    observation.photon_counts["CRb"][ilambd] = CRb.value

                    # Add detector noise
                    CRb += CRbd

                    # EXPOSURE TIME
                    if mode == "exposure_time":
                        # count rate term
                        # NOTE this includes the systematic noise floor
                        # term a la Bijan Nemati
                        cp = (
                            (CRp + observation.CRb_multiplier * CRb)
                            / (CRp * CRp - CRnf * CRnf)
                            * u.electron
                        )

                        # UNITS:
                        # ([electron/s]+[electron/s])/([electron/s]^2+[electron/s]^2) =
                        # [s/electron]

                        # Calculate Exposure time
                        observation.exptime[ilambd] = (
                            observation.SNR[ilambd]
                            * observation.SNR[ilambd]
                            * cp
                            * observatory.telescope.toverhead_multi
                            + observatory.telescope.toverhead_fixed
                        )  # record exposure time with overheads

                        # UNITS:
                        # []^2*[s/electron]*[]+[s] == [s]
                        if observation.exptime[ilambd] < 0:
                            # time is past the systematic
                            # noise floor limit
                            observation.exptime[ilambd] = np.inf

                        if observation.exptime[ilambd] > observation.td_limit:
                            # treat as unobservable
                            # if beyond exposure time limit
                            observation.exptime[ilambd] = np.inf

                        if observatory.coronagraph.nrolls != 1:
                            # multiply by number of required rolls to
                            # achieve 360 deg coverage
                            # (after tlimit enforcement)
                            observation.exptime[
                                ilambd
                            ] *= observatory.coronagraph.nrolls
                    elif mode == "signal_to_noise":

                        # cp not used in this mode. Note: This will make the science time in
                        # validation variables be 0!
                        cp = 0
                        # SIGNAL-TO-NOISE
                        # time term
                        time_factors = (
                            observation.obstime / observatory.coronagraph.nrolls
                            - observatory.telescope.toverhead_fixed
                        ) / (
                            observatory.telescope.toverhead_multi
                            * ((CRp + observation.CRb_multiplier * CRb))
                        )
                        # UNITS:
                        # ([s]*[]-[s])/([electron/s]+[]*[electron/s])
                        # [s]/[electron/s]=[s^2/electron]

                        # Signal-to-noise
                        # observation.fullsnr[ilambd] = (
                        #     np.sqrt(
                        #         (time_factors * CRp**2)
                        #         / (1 * ELECTRON + time_factors * CRnf**2)
                        #     )
                        #     * DIMENSIONLESS
                        # )
                        # rewrote the above equation to properly evaluate the SNR when time = inf
                        observation.fullsnr[ilambd] = (
                            np.sqrt(CRp**2 / (1 * ELECTRON / time_factors + CRnf**2))
                            * DIMENSIONLESS
                        ).decompose()  # Ensure all units are simplified

                        # UNITS:
                        # ([s^2/electron]*[electron/s]^2)/([electron]+[s^2/electron]*[electron/s]^2)=
                        # [electron]/[electron] = []

                        observation.SNR[ilambd] = observation.fullsnr[
                            ilambd
                        ]  # this is the calculated snr now

                    # Store the variables of interest
                    observation.validation_variables[ilambd] = {
                        "F0": scene.F0[ilambd],
                        "magstar": scene.mag,
                        "dist": scene.dist,
                        "D": observatory.telescope.diameter,
                        "A_cm": area_cm2,
                        "wavelength": observation.wavelength[ilambd].to(u.nm),
                        "deltalambda_nm": deltalambda_nm,
                        "snr": observation.SNR[ilambd],
                        "nzodis": scene.nzodis,
                        "toverhead_fixed": observatory.telescope.toverhead_fixed,
                        "toverhead_multi": observatory.telescope.toverhead_multi,
                        "det_DC": observatory.detector.DC[ilambd],
                        "det_RN": observatory.detector.RN[ilambd],
                        "det_CIC": observatory.detector.CIC[ilambd],
                        "det_tread": observatory.detector.tread[ilambd],
                        "det_pixscale_mas": observatory.detector.pixscale_mas,
                        "dQE": observatory.detector.dQE[ilambd],
                        "QE": observatory.detector.QE[ilambd],
                        "T_optical": observatory.optics_throughput[ilambd],
                        "Fs_over_F0": scene.Fs_over_F0[ilambd] * scene.F0[ilambd],
                        "Fp": scene.Fs_over_F0[ilambd]
                        * scene.F0[ilambd]
                        * scene.Fp_over_Fs[ilambd],
                        "Fzodi": scene.Fzodi_list[ilambd] * scene.F0[ilambd],
                        "Fexozodi": scene.Fexozodi_list[ilambd]
                        * scene.F0[ilambd]
                        / (scene.separation**2 * scene.dist**2),
                        "sp_lod": arcsec_to_lambda_d(
                            scene.separation,
                            observation.wavelength[ilambd],
                            observatory.telescope.diameter,
                        ),
                        "omega_lod": observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        # "throughput": observatory.total_throughput[ilambd],
                        "T_core or photometric_aperture_throughput": observatory.coronagraph.photometric_aperture_throughput[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "Istar": observatory.coronagraph.Istar[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        "Istar*oneopixscale2 in (l/D)^-2": observatory.coronagraph.Istar[
                            int(np.floor(iy)), int(np.floor(ix))
                        ]
                        * (1 / observatory.coronagraph.pixscale) ** 2,
                        # "contrast * offset PSF peak *oneopixscale2  in (l/D)^-2 (unused)": 0.025
                        # * observatory.coronagraph.TLyot
                        # * observatory.coronagraph.contrast
                        # * (1 / observatory.coronagraph.pixscale) ** 2,
                        "skytrans": observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        "skytrans*oneopixscale2  in (l/D)^-2": observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ]
                        * (1 / observatory.coronagraph.pixscale) ** 2,
                        "det_npix": det_npix,
                        "t_photon_count": t_photon_count,
                        "CRp": CRp,
                        "CRbs": CRbs
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbz": CRbz.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbez": CRbez.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbbin": CRbbin
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbth": CRbth
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRb": CRb,
                        "CRbd": CRbd,
                        "CRnf": CRnf,
                        "sciencetime": observation.SNR[ilambd]
                        * observation.SNR[ilambd]
                        * cp,
                        "exptime": observation.exptime[ilambd],
                    }

                else:
                    print(
                        "WARNING: Photometric aperture is not large enough. Hardcoded infinity results."
                    )
                    if mode == "exposure_time":
                        observation.exptime[ilambd] = np.inf
                    elif mode == "signal_to_noise":
                        observation.fullsnr[ilambd] = np.inf

        else:
            print(
                "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results."
            )
            if mode == "exposure_time":
                observation.exptime[ilambd] = np.inf
            elif mode == "signal_to_noise":
                observation.fullsnr[ilambd] = np.inf

        if verbose:
            utils.print_all_variables(
                observation,
                scene,
                observatory,
                deltalambda_nm,
                lod,
                lod_rad,
                lod_arcsec,
                area_cm2,
                detpixscale_lod,
                stellar_diam_lod,
                pixscale_rad,
                oneopixscale_arcsec,
                det_sep_pix,
                det_sep,
                det_Istar,
                det_skytrans,
                det_photometric_aperture_throughput,
                det_omega_lod,
                det_CRp,
                det_CRbs,
                det_CRbz,
                det_CRbez,
                det_CRbbin,
                det_CRbth,
                det_CR,
                ix,
                iy,
                sp_lod,
                CRp,
                CRnf,
                CRbs,
                CRbz,
                CRbez,
                CRbbin,
                t_photon_count,
                CRbd,
                CRbth,
                CRb,
                # cp,
            )
            # Save the photon counts for later analysis
            pickle.dump(observation.photon_counts, open("photon_counts.pk", "wb"))

    return
