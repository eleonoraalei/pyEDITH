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

    Parameters:
    obj: object
        The object whose attributes are to be validated.
    expected_args: dict
        A dictionary where keys are attribute names and values are expected types or units.

    Raises:
    AttributeError: If a required attribute is missing.
    TypeError: If an attribute has an incorrect type.
    ValueError: If a Quantity attribute has incorrect units or if there's an unexpected type specification.
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
