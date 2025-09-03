from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
from typing import Dict, Any


def average_over_bandpass(params: dict, wavelength_range: list) -> dict:
    """
    Calculate the average of array parameters within a specified wavelength range.

    This function takes a dictionary of parameters and computes the mean value
    of all numpy array parameters (except wavelength) within the specified
    wavelength boundaries. The wavelength array is expected to be stored under
    the key "lam" in the params dictionary.

    Parameters
    ----------
    params : dict
        Dictionary containing parameters where numpy arrays represent wavelength-dependent
        quantities. Must include a "lam" key containing the wavelength array.
        These parameters come from EACy and follow that formatting.
    wavelength_range : list
        Two-element list containing the lower and upper wavelength boundaries
        for averaging, expected to have astropy units

    Returns
    -------
    dict
        Modified parameters dictionary with array values replaced by their mean
        values within the specified wavelength range
    """
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
    """
    Interpolate array parameters onto a new wavelength grid.

    This function takes a dictionary of parameters and interpolates all numpy array
    parameters (except the wavelength array itself) onto a new set of wavelength
    points using 1D linear interpolation. The original wavelength array is expected
    to be stored under the key "lam" in the params dictionary.

    Parameters
    ----------
    params : dict
        Dictionary containing parameters where numpy arrays represent wavelength-dependent
        quantities. Must include a "lam" key containing the original wavelength array.
        These parameters come from EACy and follow that formatting.

    wavelengths : list
        New wavelength points onto which to interpolate the parameter arrays

    Returns
    -------
    dict
        Modified parameters dictionary with array values interpolated onto the new
        wavelength grid specified by the wavelengths parameter
    """

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


def fill_parameters(
    class_obj: object, parameters: dict, default_parameters: dict
) -> None:
    """
    Populate class object attributes with user parameters or default values.

    This function sets attributes on a class object by loading user-provided
    parameters or falling back to default values. It handles both regular values
    and astropy Quantity objects with units, ensuring proper unit consistency
    when dealing with physical quantities.

    Parameters
    ----------
    class_obj : object
        Class instance whose attributes will be set
    parameters : dict
        Dictionary of user-provided parameter values
    default_parameters : dict
        Dictionary of default parameter values with keys matching expected
        attribute names. Values can be regular types or astropy Quantities
    """

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


def convert_to_numpy_array(class_obj: object, array_params: list) -> None:
    """
    Convert specified class attributes to numpy arrays with proper dtype.

    This function converts class attributes to numpy arrays with float64 dtype,
    while preserving astropy units for Quantity objects. Non-Quantity attributes
    are converted to plain numpy arrays, while Quantity attributes maintain their
    units but have their values converted to numpy arrays.

    Parameters
    ----------
    class_obj : object
        Class instance whose attributes will be converted
    array_params : list
        List of attribute names to convert to numpy arrays
    """

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

    This function checks that an object has all the required attributes and that
    each attribute has the correct type or units. It supports validation of
    integer and float types as well as astropy Quantity objects with specific units.

    Parameters
    ----------
    obj : object
        The object whose attributes are to be validated
    expected_args : dict
        A dictionary where keys are attribute names and values are expected types or units

    Raises
    ------
    AttributeError
        If a required attribute is missing
    TypeError
        If an attribute has an incorrect type
    ValueError
        If a Quantity attribute has incorrect units or if there's an unexpected type specification
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


def print_array_info(
    file: object, name: str, arr: np.ndarray, mode: str = "full_info"
) -> None:
    """
    Write detailed information about an array or variable to a file.

    This function writes comprehensive information about a given array or variable
    to a specified file, including its shape, data type, units (if applicable),
    and statistical properties such as minimum and maximum values. The output format
    depends on the specified mode.

    Parameters
    ----------
    file : object
        Open file object to write the information to
    name : str
        Name or identifier of the variable/array being described
    arr : np.ndarray
        The array or variable to analyze and describe. Can be a numpy array,
        an array-like object, or a scalar with or without astropy units
    mode : str, optional
        Output mode that determines the level of detail in the description.
        Default is "full_info" which provides comprehensive information.
        Other modes provide more concise output
    """

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
    observation: object,
    scene: object,
    observatory: object,
    deltalambda_nm: np.ndarray,
    lod: np.ndarray,
    lod_rad: np.ndarray,
    lod_arcsec: np.ndarray,
    area_cm2: np.ndarray,
    detpixscale_lod: np.ndarray,
    stellar_diam_lod: np.ndarray,
    pixscale_rad: np.ndarray,
    oneopixscale_arcsec: np.ndarray,
    det_sep_pix: np.ndarray,
    det_sep: np.ndarray,
    det_Istar: np.ndarray,
    det_skytrans: np.ndarray,
    det_photometric_aperture_throughput: np.ndarray,
    det_omega_lod: np.ndarray,
    det_CRp: np.ndarray,
    det_CRbs: np.ndarray,
    det_CRbz: np.ndarray,
    det_CRbez: np.ndarray,
    det_CRbbin: np.ndarray,
    det_CRbth: np.ndarray,
    det_CR: np.ndarray,
    ix: int,
    iy: int,
    sp_lod: float,
    CRp: np.ndarray,
    CRnf: np.ndarray,
    CRbs: np.ndarray,
    CRbz: np.ndarray,
    CRbez: np.ndarray,
    CRbbin: np.ndarray,
    t_photon_count: np.ndarray,
    CRbd: np.ndarray,
    CRbth: np.ndarray,
    CRb: np.ndarray,
) -> None:
    """
    Write comprehensive debug information to files for observation calculations.

    This function outputs detailed information about all relevant parameters
    and calculated variables used in the observation simulation to both validation
    and full_info text files. It includes observation parameters, scene properties,
    observatory characteristics, and all intermediate calculations.

    Parameters
    ----------
    observation : Observation
        Observation object containing observation-specific parameters
    scene : AstrophysicalScene
        Scene object containing astrophysical scene parameters
    observatory : Observatory
        Observatory object containing telescope, coronagraph, and detector parameters
    deltalambda_nm : np.ndarray
        Wavelength intervals in nanometers
    lod : np.ndarray
        Lambda over D values
    lod_rad : np.ndarray
        Lambda over D values in radians
    lod_arcsec : np.ndarray
        Lambda over D values in arcseconds
    area_cm2 : np.ndarray
        Telescope area in cm²
    detpixscale_lod : np.ndarray
        Detector pixel scale in λ/D units
    stellar_diam_lod : np.ndarray
        Stellar diameter in λ/D units
    pixscale_rad : np.ndarray
        Pixel scale in radians
    oneopixscale_arcsec : np.ndarray
        Single pixel scale in arcseconds
    det_sep_pix : np.ndarray
        Detector separation in pixels
    det_sep : np.ndarray
        Detector separation
    det_Istar : np.ndarray
        Detector stellar intensity
    det_skytrans : np.ndarray
        Detector sky transmission
    det_photometric_aperture_throughput : np.ndarray
        Detector photometric aperture throughput
    det_omega_lod : np.ndarray
        Detector solid angle in λ/D units
    det_CRp : np.ndarray
        Detector planet count rate
    det_CRbs : np.ndarray
        Detector background star count rate
    det_CRbz : np.ndarray
        Detector zodiacal background count rate
    det_CRbez : np.ndarray
        Detector exozodiacal background count rate
    det_CRbbin : np.ndarray
        Detector binary background count rate
    det_CRbth : np.ndarray
        Detector thermal background count rate
    det_CR : np.ndarray
        Total detector count rate
    ix : int
        X pixel coordinate
    iy : int
        Y pixel coordinate
    sp_lod : float
        Separation in λ/D units
    CRp : np.ndarray
        Planet count rate
    CRnf : np.ndarray
        Noise floor count rate
    CRbs : np.ndarray
        Background star count rate
    CRbz : np.ndarray
        Zodiacal background count rate
    CRbez : np.ndarray
        Exozodiacal background count rate
    CRbbin : np.ndarray
        Binary background count rate
    t_photon_count : np.ndarray
        Photon counting time
    CRbd : np.ndarray
        Detector background count rate
    CRbth : np.ndarray
        Thermal background count rate
    CRb : np.ndarray
        Total background count rate
    """

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
    snr_arr: np.ndarray,
    scene: object,
    random_seed: int = None,
    set_below_zero: float = np.nan,
) -> tuple:
    """
    Synthesize an observation using calculated SNRs for each wavelength bin.

    This function generates a synthetic observation by adding noise to the
    planet-to-star flux ratio based on the provided signal-to-noise ratio
    array. The noise is drawn from a normal distribution and scaled according
    to the SNR values. This function requires that the ETC has been run in
    SNR mode with a given exposure time first.

    Parameters
    ----------
    snr_arr : np.ndarray
        1D array containing SNR for each spectral bin
    scene : AstrophysicalScene
        Scene object containing astrophysical parameters including Fp_over_Fs
    random_seed : int, optional
        Random seed for reproducible noise generation. Default is None
    set_below_zero : float, optional
        Value to assign to measurements below zero. Default is np.nan

    Returns
    -------
    tuple
        A tuple containing:

        obs : np.ndarray
            1D array, spectrum with added noise

        noise : np.ndarray
            1D array, noise for each spectral bin
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


def wavelength_grid_fixed_res(x_min: float, x_max: float, res: float = -1) -> tuple:
    """
    Generate a wavelength grid at a fixed spectral resolution.

    This function creates a wavelength grid with constant resolution across
    the specified wavelength range. The grid spacing increases logarithmically
    to maintain constant R = λ/Δλ.

    Parameters
    ----------
    x_min : float
        Minimum wavelength
    x_max : float
        Maximum wavelength
    res : float, optional
        Spectral resolution R = λ/Δλ. Default is -1

    Returns
    -------
    tuple
        A tuple containing two 1D numpy arrays:

        wavelength : np.ndarray
            Wavelength grid

        delta_wavelength : np.ndarray
            Delta wavelength grid
    """

    x = [x_min]
    fac = (1 + 2 * res) / (2 * res - 1)
    i = 0
    while x[i] * fac < x_max:
        x = np.concatenate((x, [x[i] * fac]))
        i = i + 1
    Dx = x / res
    return np.squeeze(x), np.squeeze(Dx)


def gen_wavelength_grid(x_min: list, x_max: list, res: list) -> tuple:
    """
    Generate a continuous wavelength grid for multiple spectral channels.

    This function creates wavelength grids at fixed resolution for each spectral
    channel, then concatenates them to form a continuous wavelength grid covering
    all channels.

    Parameters
    ----------
    x_min : list
        Minimum wavelength for each spectral channel
    x_max : list
        Maximum wavelength for each spectral channel
    res : list
        Spectral resolution for each spectral channel

    Returns
    -------
    tuple
        A tuple containing two 1D numpy arrays:

        wavelength_grid : np.ndarray
            Combined wavelength grid for all channels

        delta_wavelength_grid : np.ndarray
            Combined delta wavelength grid for all channels
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


def regrid_wavelengths(
    input_wls: np.ndarray, res: list, lam_low: list = None, lam_high: list = None
) -> tuple:
    """
    Create a new wavelength grid with specified resolution and channel boundaries.

    This function generates a new wavelength grid given the resolution and
    channel boundaries for each spectral channel. If no boundaries are provided,
    it uses the full range of the input wavelengths.

    Parameters
    ----------
    input_wls : np.ndarray
        The wavelength grid supplied by the user
    res : list
        Array of desired resolutions for each channel. Should have length equal
        to the number of spectral channels. For example, for UV, VIS, and NIR
        channels: res = [R_UV, R_VIS, R_NIR], e.g. [7, 140, 40]
    lam_low : list, optional
        Array of the lower boundaries of spectral channels
    lam_high : list, optional
        Array of the upper boundaries of spectral channels

    Returns
    -------
    tuple
        A tuple containing two 1D numpy arrays:

        wavelength_grid : np.ndarray
            New wavelength grid

        delta_wavelength_grid : np.ndarray
            New delta wavelength grid
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


def regrid_spec_gaussconv(
    input_wls: np.ndarray,
    input_spec: np.ndarray,
    new_lam: np.ndarray,
    new_dlam: np.ndarray,
) -> np.ndarray:
    """
    Regrid a spectrum onto a new wavelength grid using Gaussian convolution.

    This function regrids a spectrum by convolving with Gaussian kernels to
    account for the spectral resolution at each wavelength point. The convolution
    is performed in log-wavelength space for accurate spectral line handling.

    Parameters
    ----------
    input_wls : np.ndarray
        The wavelength grid supplied by the user
    input_spec : np.ndarray
        The spectrum supplied by the user
    new_lam : np.ndarray
        The new wavelength grid calculated for the ETC
    new_dlam : np.ndarray
        The new delta wavelength grid calculated for the ETC

    Returns
    -------
    np.ndarray
        1D array containing the regridded spectrum with original units preserved
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

    return spec_regrid * input_spec_unit


def regrid_spec_interp(
    input_wls: np.ndarray, input_spec: np.ndarray, new_lam: np.ndarray
) -> np.ndarray:
    """
    Regrid a spectrum onto a new wavelength grid using 1D interpolation.

    This function regrids a spectrum using simple linear interpolation between
    the original and new wavelength grids. This method is faster than Gaussian
    convolution but does not account for spectral resolution effects.

    Parameters
    ----------
    input_wls : np.ndarray
        The wavelength grid supplied by the user
    input_spec : np.ndarray
        The spectrum supplied by the user
    new_lam : np.ndarray
        The new wavelength grid calculated for the ETC

    Returns
    -------
    np.ndarray
        1D array containing the regridded spectrum with original units preserved
    """

    input_spec_unit = input_spec.unit
    interp_func = interp1d(input_wls, input_spec)
    spec_regrid = interp_func(new_lam)
    return spec_regrid * input_spec_unit
