from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
from typing import Dict, Any


def average_over_bandpass(params: dict, wavelength_range: list) -> dict:
    # take the average within the specified wavelength range
    numpy_array_variables = {
        key: value for key, value in params.items() if isinstance(value, np.ndarray)
    }
    for key, value in numpy_array_variables.items():
        if key != "lam":
            params[key] = np.mean(
                params[key][
                    (params["lam"].value >= wavelength_range[0].value)
                    & (params["lam"].value <= wavelength_range[1].value)
                ]
            )
    return params


def interpolate_over_bandpass(params: dict, wavelengths: list) -> dict:
    # take the average within the specified wavelength range
    numpy_array_variables = {
        key: value for key, value in params.items() if isinstance(value, np.ndarray)
    }
    for key, value in numpy_array_variables.items():
        if key != "lam":
            interp_func = interp1d(params["lam"], params[key])
            ynew = interp_func(
                wavelengths
            )  # interpolates the CG throughput values onto native wl grid
            params[key] = ynew
    return params


def fill_parameters(class_obj, parameters, default_parameters):
    # Load parameters, use defaults if not provided
    for key, default_value in default_parameters.items():
        if key in parameters:
            # User provided a value
            user_value = parameters[key]
            if isinstance(default_value, u.Quantity):
                # Ensure the user value has the same unit as the default
                # TODO Implement conversion of units from the input file

                if isinstance(user_value, u.Quantity):
                    setattr(class_obj, key, user_value.to(default_value.unit))
                else:
                    setattr(class_obj, key, u.Quantity(user_value, default_value.unit))
            else:
                # For non-Quantity values (like integers), use as is
                setattr(class_obj, key, user_value)
        else:
            # Use default value
            setattr(class_obj, key, default_value)


def convert_to_numpy_array(class_obj, array_params):
    for param in array_params:
        attr_value = getattr(class_obj, param)
        if isinstance(attr_value, u.Quantity):
            # If it's already a Quantity, convert to numpy array while preserving units
            setattr(
                class_obj,
                param,
                u.Quantity(
                    np.array(attr_value.value, dtype=np.float64), attr_value.unit
                ),
            )
        else:
            # If it's not a Quantity, convert to numpy array without units
            setattr(class_obj, param, np.array(attr_value, dtype=np.float64))


def validate_attributes(obj: Any, expected_args: Dict[str, Any]) -> None:
    """
    Validate attributes of an object against expected types and units.

    Parameters
    ----------
    obj: object
        The object whose attributes are to be validated.
    expected_args: dict
        A dictionary where keys are attribute names and values are expected types or units.

    Raises
    ------
    AttributeError
        If a required attribute is missing.
    TypeError
        If an attribute has an incorrect type.
    ValueError
        If a Quantity attribute has incorrect units or if there's an unexpected type specification.
    """
    class_name = obj.__class__.__name__

    for arg, expected_type in expected_args.items():
        if not hasattr(obj, arg):
            raise AttributeError(f"{class_name} is missing attribute: {arg}")

        value = getattr(obj, arg)

        if expected_type is int:
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{class_name} attribute {arg} should be an integer")
        elif expected_type is float:
            if not isinstance(value, (float, np.floating)):
                raise TypeError(f"{class_name} attribute {arg} should be a float")
        elif isinstance(
            expected_type, (u.UnitBase, u.CompositeUnit, u.IrreducibleUnit)
        ):
            if not isinstance(value, u.Quantity):
                raise TypeError(f"{class_name} attribute {arg} should be a Quantity")
            if value.unit != expected_type:
                raise ValueError(
                    f"{class_name} attribute {arg} has incorrect units. "
                    f"Expected {expected_type}, got {value.unit}"
                )
        else:
            raise ValueError(f"Unexpected type specification for {arg}")


def print_array_info(file, name, arr, mode="full_info"):
    if mode == "full_info":
        file.write(f"{name}:\n")

        # Handle units
        if hasattr(arr, "unit"):
            if arr.unit == u.dimensionless_unscaled:
                file.write(" Unit: dimensionless\n")
            else:
                file.write(f" Unit: {arr.unit}\n")
        else:
            file.write(" Unit: N/A\n")

        # Convert to numpy array if it's not already
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # Handle shape
        if arr.size == 1:
            file.write(" Shape: scalar\n")
            if np.issubdtype(arr.dtype, np.integer):
                file.write(f" Value: {arr.item():d}\n")
            else:
                file.write(f" Value: {arr.item():.6e}\n")
        else:
            file.write(f" Shape: {arr.shape}\n")
            if arr.size > 0:
                max_val = np.max(arr)
                min_val = np.min(arr)
                max_coords = np.unravel_index(np.argmax(arr), arr.shape)
                min_coords = np.unravel_index(np.argmin(arr), arr.shape)
                file.write(f" Max value: {max_val} at coordinates: {max_coords}\n")
                file.write(f" Min value: {min_val} at coordinates: {min_coords}\n")
            else:
                file.write(" Array is empty\n")
    else:
        # C-like output for non-full_info mode
        file.write(f"{name}: ")

        # Convert to numpy array if it's not already
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        is_int = np.issubdtype(arr.dtype, np.integer)

        # Check if the array has units
        has_units = hasattr(arr, "unit")

        if arr.size == 1:
            if has_units:
                file.write(f"value: {arr.value.item():.6e}\n")
            else:

                file.write(f"value: {arr.item():.6e}\n")
        else:
            max_val = np.max(arr)
            min_val = np.min(arr)
            if has_units:

                file.write(
                    f"max value: {max_val.value:.6e}, min value: {min_val.value:.6e}\n"
                )
            else:

                file.write(f"max value: {max_val:.6e}, min value: {min_val:.6e}\n")


def print_all_variables(
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
):
    for mode in ["validation", "full_info"]:
        with open("pyedith_" + mode + ".txt", "w") as file:
            file.write("Input Objects and Their Relevant Properties:\n")
            file.write("1. Observation:\n")
            for item_name, item in [
                ("observation.wavelength", observation.wavelength),
                ("observation.SNR", observation.SNR),
                ("observation.td_limit", observation.td_limit),
                ("observation.CRb_multiplier", observation.CRb_multiplier),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n2. Scene:\n")
            for item_name, item in [
                ("scene.mag", scene.mag),
                (
                    "scene.stellar_angular_diameter_arcsec",
                    scene.stellar_angular_diameter_arcsec,
                ),
                ("scene.F0", scene.F0),
                ("scene.Fp_over_Fs", scene.Fp_over_Fs),
                ("scene.Fzodi_list", scene.Fzodi_list),
                ("scene.Fexozodi_list", scene.Fexozodi_list),
                ("scene.Fbinary_list", scene.Fbinary_list),
                ("scene.xp", scene.xp),
                ("scene.yp", scene.yp),
                ("scene.separation", scene.separation),
                ("scene.dist", scene.dist),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n3. Observatory:\n")
            file.write("Telescope:\n")
            for item_name, item in [
                ("observatory.telescope.diameter", observatory.telescope.diameter),
                (
                    "observatory.telescope.temperature",
                    observatory.telescope.temperature,
                ),
                (
                    "observatory.telescope.toverhead_multi",
                    observatory.telescope.toverhead_multi,
                ),
                (
                    "observatory.telescope.toverhead_fixed",
                    observatory.telescope.toverhead_fixed,
                ),
                ("observatory.total_throughput", observatory.total_throughput),
                ("observatory.epswarmTrcold", observatory.epswarmTrcold),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\nCoronagraph:\n")
            for item_name, item in [
                (
                    "observatory.coronagraph.bandwidth",
                    observatory.coronagraph.bandwidth,
                ),
                ("observatory.coronagraph.Istar", observatory.coronagraph.Istar),
                (
                    "observatory.coronagraph.noisefloor",
                    observatory.coronagraph.noisefloor,
                ),
                ("observatory.coronagraph.npix", observatory.coronagraph.npix),
                ("observatory.coronagraph.pixscale", observatory.coronagraph.pixscale),
                ("observation.psf_trunc_ratio", observation.psf_trunc_ratio),
                (
                    "observatory.coronagraph.photometric_aperture_throughput",
                    observatory.coronagraph.photometric_aperture_throughput,
                ),
                ("observatory.coronagraph.skytrans", observatory.coronagraph.skytrans),
                (
                    "observatory.coronagraph.omega_lod",
                    observatory.coronagraph.omega_lod,
                ),
                ("observatory.coronagraph.xcenter", observatory.coronagraph.xcenter),
                ("observatory.coronagraph.ycenter", observatory.coronagraph.ycenter),
                (
                    "observatory.coronagraph.nchannels",
                    observatory.coronagraph.nchannels,
                ),
                (
                    "observatory.coronagraph.minimum_IWA",
                    observatory.coronagraph.minimum_IWA,
                ),
                (
                    "observatory.coronagraph.maximum_OWA",
                    observatory.coronagraph.maximum_OWA,
                ),
                (
                    "observatory.coronagraph.npsfratios",
                    observatory.coronagraph.npsfratios,
                ),
                ("observatory.coronagraph.nrolls", observatory.coronagraph.nrolls),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\nDetector:\n")
            for item_name, item in [
                (
                    "observatory.detector.pixscale_mas",
                    observatory.detector.pixscale_mas,
                ),
                (
                    "observatory.detector.QE*observatory.detector.dQE",
                    observatory.detector.QE * observatory.detector.dQE,
                ),
                (
                    "observatory.detector.npix_multiplier",
                    observatory.detector.npix_multiplier,
                ),
                ("observatory.detector.DC", observatory.detector.DC),
                ("observatory.detector.RN", observatory.detector.RN),
                ("observatory.detector.tread", observatory.detector.tread),
                ("observatory.detector.CIC", observatory.detector.CIC),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\nCalculated Variables:\n")
            file.write("\n1. Initial Calculations:\n")
            for item_name, item in [
                ("Fs_over_F0", scene.Fs_over_F0),
                ("deltalambda_nm", deltalambda_nm),
                ("lod", lod),
                ("lod_rad", lod_rad),
                ("lod_arcsec", lod_arcsec),
                ("area_cm2", area_cm2),
                ("detpixscale_lod", detpixscale_lod),
                ("stellar_diam_lod", stellar_diam_lod),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n2. Interpolated Arrays:\n")
            for item_name, item in [
                ("Istar_interp", observatory.coronagraph.Istar),
                ("noisefloor_interp", observatory.coronagraph.noisefloor),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n3. Coronagraph Performance Measurements:\n")
            for item_name, item in [
                ("pixscale_rad", pixscale_rad),
                ("oneopixscale_arcsec", oneopixscale_arcsec),
                ("det_sep_pix", det_sep_pix),
                ("det_sep", det_sep),
                ("det_Istar", det_Istar),
                ("det_skytrans", det_skytrans),
                (
                    "det_photometric_aperture_throughput",
                    det_photometric_aperture_throughput,
                ),
                ("det_omega_lod", det_omega_lod),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n4. Detector Noise Calculations:\n")
            for item_name, item in [
                ("det_CRp", det_CRp),
                ("det_CRbs", det_CRbs),
                ("det_CRbz", det_CRbz),
                ("det_CRbez", det_CRbez),
                ("det_CRbbin", det_CRbbin),
                ("det_CRbth", det_CRbth),
                ("det_CR", det_CR),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n5. Planet Position and Separation:\n")
            for item_name, item in [("ix", ix), ("iy", iy), ("sp_lod", sp_lod)]:
                print_array_info(file, item_name, item, mode)

            file.write("\n6. Count Rates and Exposure Time Calculation:\n")
            for item_name, item in [
                ("CRp", CRp),
                ("CRnf", CRnf),
                ("CRbs", CRbs),
                ("CRbz", CRbz),
                ("CRbez", CRbez),
                ("CRbbin", CRbbin),
                ("t_photon_count", t_photon_count),
                ("CRbd", CRbd),
                ("CRbth", CRbth),
                ("CRb", CRb),
            ]:
                print_array_info(file, item_name, item, mode)

            file.write("\n7. Final Result:\n")
            for item_name, item in [
                ("observation.exptime", observation.exptime),
                ("observation.fullsnr", observation.fullsnr),
            ]:
                print_array_info(file, item_name, item, mode)


def synthesize_observation(
    snr_arr,
    exptime,
    ref_lam,
    observation,
    scene,
    random_seed=None,
    set_below_zero=np.nan,
    plotting=False,
):
    """
    Synthesizes an observation using the calculated SNRs for each wavelength bin
    IMPORTANT: You have to run the ETC in SNR mode with a given exposure time first
    (see spectroscopy tutorial)
    """

    # set a random seed if desired
    if random_seed is not None:
        np.random.seed(random_seed)

    noise = scene.Fp_over_Fs / snr_arr
    obs = scene.Fp_over_Fs + noise * np.random.randn(len(noise))

    obs[obs < 0] = (
        set_below_zero  # any observation that is below zero is set to whatever you want
    )

    return obs, noise


def wavelength_grid_fixed_res(x_min, x_max, res=-1):
    """
    Generates a wavelength grid at a fixed resolution of res.

    Parameters
    ----------
        x_min : float
            minimum wavelength
        x_max : float
            maximum wavelength
        res : float
            spectral resolution
    """
    x = [x_min]
    fac = (1 + 2 * res) / (2 * res - 1)
    i = 0
    while x[i] * fac < x_max:
        x = np.concatenate((x, [x[i] * fac]))
        i = i + 1
    Dx = x / res
    return np.squeeze(x), np.squeeze(Dx)


def gen_wavelength_grid(x_min, x_max, res):
    """
    Generates a wavelength grid at a fixed resolution for each spectral channel,
        then concatenates them to create a continuous wavelength grid
    inputs:
        x_min : 1D array
            minimum wavelength
        x_max : 1D array
            maximum wavelength
        res : 1D array
            spectral resolution
    """
    x, Dx = wavelength_grid_fixed_res(x_min[0], x_max[0], res=res[0])
    if len(x_min) > 1:
        for i in range(1, len(x_min)):
            xi, Dxi = wavelength_grid_fixed_res(x_min[i], x_max[i], res=res[i])
            x = np.concatenate((x, xi))
            Dx = np.concatenate((Dx, Dxi))
    Dx = [Dxs for _, Dxs in sorted(zip(x, Dx))]
    x = np.sort(x)
    return np.squeeze(x), np.squeeze(Dx)


def regrid_wavelengths(input_wls, res, lam_low=None, lam_high=None):
    """
    Creates a new wavelength grid given the resolution and channel boundaries for each spectral channel

    Parameters
    ----------
    input_wls : float, 1D arr
        the wavelength grid the user supplies
    res : float, 1D arr
        array of desired resolutions for each channel. should be length of the number of spectral channels
        for example, if we have a UV, VIS, and NIR channel, then we expect res = [R_UV, R_VIS, R_NIR], e.g. [7, 140, 40]
    lam_low : float, 1D arr
        array of the lower boundaries of spectral channels.
    lam_high : float, 1D arr
        array of the upper boundaries of spectral channels.
    """

    if lam_low is None and lam_high is None:
        lam_low = [np.min(input_wls[1:])]
        lam_high = [np.max(input_wls[:-1])]
    else:  # lam_low is not None and lam_high is not None:
        assert len(res) == len(lam_low) == len(lam_high)

    if len(res) > 1:
        assert (
            np.min(input_wls) < lam_low[0]
        ), "Your minimum input wavelength is greater than first channel lower boundary."
        assert (
            np.max(input_wls) > lam_high[-1]
        ), f"Your maximum input wavelength is less than last channel upper boundary."

        lam, dlam = gen_wavelength_grid(lam_low, lam_high, res)
    else:
        # no channel boundaries
        lam, dlam = gen_wavelength_grid(lam_low, lam_high, res)

    return lam, dlam


def regrid_spec_gaussconv(input_wls, input_spec, new_lam, new_dlam):
    """
    Regrids a spectrum onto a new wavelength grid using gaussian convolution.

    Inputs:
    input_wls : float, 1D arr
        the wavelength grid the user supplies
    input_spec : float, 1D arr
        the spectrum the user supplies
    new_lam : float, 1D arr
        the new wavelength grid we calculated for the ETC
    new_lam : float, 1D arr
        the new delta wavelength grid we calculated for the ETC

    """
    input_spec_unit = input_spec.unit

    R_arr = new_lam / new_dlam

    # interpolate original spectrum onto a fine log-lambda grid
    loglam_old = np.log(input_wls)
    interp_flux = interp1d(loglam_old, input_spec, bounds_error=False, fill_value=0.0)

    # make fine log-lambda grid
    dloglam = 1e-5
    loglam_grid = np.arange(loglam_old[0], loglam_old[-1], dloglam)
    lam_grid = np.exp(loglam_grid)
    flux_grid = interp_flux(loglam_grid)

    spec_regrid = np.zeros_like(new_lam)

    for i in range(len(new_lam)):
        lam = new_lam[i]
        R = R_arr[i]

        # get width of gaussian: sigma = FWHM / (2*np.sqrt(2*np.log(2))), where FWHM is dlam, but this is in logspace, so FWHM = 1/R
        sigma_loglam = 1.0 / (R * 2.0 * np.sqrt(2 * np.log(2)))

        # Gaussian kernel in log-space
        kernel_half_width = int(4 * sigma_loglam / dloglam)
        kernel_grid = np.arange(-kernel_half_width, kernel_half_width + 1)
        kernel = np.exp(-0.5 * (kernel_grid * dloglam / sigma_loglam) ** 2)
        kernel /= np.sum(kernel)

        # Find center index
        center_idx = np.searchsorted(lam_grid, lam)

        # Define convolution range safely
        i1 = max(center_idx - kernel_half_width, 0)
        i2 = min(center_idx + kernel_half_width + 1, len(flux_grid))
        k1 = kernel_half_width - (center_idx - i1)
        k2 = kernel_half_width + (i2 - center_idx)

        # Perform local convolution
        flux_segment = flux_grid[i1:i2]
        kernel_segment = kernel[k1:k2]
        spec_regrid[i] = np.sum(flux_segment * kernel_segment)

    return spec_regrid*input_spec_unit


def regrid_spec_interp(input_wls, input_spec, new_lam):
    """
    Regrids a spectrum onto a new wavelength grid using 1D interpolation.

    Parameters
    ----------
    input_wls : float, 1D arr
        the wavelength grid the user supplies
    input_spec : float, 1D arr
        the spectrum the user supplies
    new_lam : float, 1D arr
        the new wavelength grid we calculated for the ETC

    """
    input_spec_unit = input_spec.unit
    interp_func = interp1d(input_wls, input_spec)
    spec_regrid = interp_func(new_lam)
    return spec_regrid * input_spec_unit
