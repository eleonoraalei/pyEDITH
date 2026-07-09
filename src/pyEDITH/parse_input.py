from typing import Union, Dict, Tuple
from pathlib import Path
import astropy.units as u
import numpy as np
from .units import *
import pandas as pd
import os
import logging

logger = logging.getLogger("pyEDITH")


def parse_input_file(
    file_path: Union[Path, str], secondary_flag: bool
) -> Tuple[Dict, Dict]:
    """
    Parse an input file and extract variables and secondary variables.

    This function reads a configuration file, processes its contents line by line,
    and extracts primary and optional secondary variables. It handles various data
    types including arrays, strings, and numeric values, and performs special
    processing for IFS observing mode parameters. The function handles various data types including arrays, strings, and numbers.
    Comments in the input file should start with ';'.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the input file
    secondary_flag : bool
        Flag indicating whether secondary variables are expected

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple containing two dictionaries:

        variables: dict
            Primary variables extracted from the file

        secondary_variables: dict
            Secondary variables extracted from the file
            (any non-specified variable will be the same as in the variables dictionary)

    Raises
    ------
    KeyError
        If some parameters are missing
    FileNotFoundError
        If a specified spectrum file cannot be found
    ValueError
        If there are issues with the spectrum file

    """

    with open(file_path, "r") as file:
        content = file.read()

    # Remove comments and empty lines
    lines = [
        line.split(";")[0].strip()
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith(";")
    ]

    variables = {}
    secondary_variables = {}
    has_secondary = False

    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Handle arrays
            if value.startswith("[") and value.endswith("]"):
                value = [float(v.strip()) for v in value[1:-1].split(",")]
            # Handle strings
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            else:
                # Handle numbers
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string if it's not a number

            if "secondary" in key:
                has_secondary = True
                if secondary_flag:
                    secondary_variables[key[10:]] = (
                        value  # it replaces the default value
                    )
            else:
                variables[key] = value

    if secondary_flag and not has_secondary:
        raise KeyError(
            "Secondary flag is True but no secondary variables found in the input file."
        )

    # Handle IFS mode
    if variables.get("observing_mode") == "IFS":
        required_columns = ["wavelength", "Fstar_10pc", "Fp/Fs"]

        # Check if all required columns are provided as lists in the input file
        if all(
            col in variables and isinstance(variables[col], list)
            for col in required_columns
        ):
            # Ensure all lists have the same length
            lengths = [len(variables[col]) for col in required_columns]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All of {', '.join(required_columns)} must have the same length"
                )
            variables["nlambda"] = lengths[0]

        # If not all required columns are provided, try to read from spectrum file
        elif "spectrum_file" in variables:
            spectrum_file = variables["spectrum_file"]

            # Check if the file exists and is readable
            if not os.path.isfile(spectrum_file):
                raise FileNotFoundError(f"Spectrum file not found: {spectrum_file}")

            # Read the spectrum file
            spectrum_df = pd.read_csv(variables["spectrum_file"])

            # Ensure the file has exactly 3 columns
            if len(spectrum_df.columns) != 3:
                raise ValueError(
                    f"Spectrum file must contain exactly 3 columns (wavelength, stellar flux, planet contrast), but it has {len(spectrum_df.columns)}"
                )
            # Rename the columns to ensure they have the correct names
            spectrum_df.columns = ["wavelength", "Fstar_10pc", "Fp/Fs"]

            # Verify that all columns can be converted to float
            for column in spectrum_df.columns:
                try:
                    spectrum_df[column] = spectrum_df[column].astype(float)
                except ValueError:
                    raise ValueError(f"Column '{column}' contains non-numeric values")

            # Set the wavelength-dependent parameters from the file
            variables["wavelength"] = spectrum_df["wavelength"].tolist()
            variables["Fstar_10pc"] = spectrum_df["Fstar_10pc"].tolist()
            variables["Fp/Fs"] = spectrum_df["Fp/Fs"].tolist()
            variables["nlambda"] = len(spectrum_df["wavelength"].tolist())

        else:
            raise KeyError(
                "Required parameters 'wavelength', 'Fstar_10pc', and 'Fp/Fs' are not provided. Please write them explicitly or provide a spectrum_file path."
            )

    if variables.get("observing_mode") == "IMAGER" and isinstance(
        variables["wavelength"], list
    ):
        raise KeyError(
            "In IMAGER mode you can only use one wavelength at a time. If you are simulating photometry, please run every single wavelength separately. If you want to model a spectrum, please use IFS mode."
        )
    return variables, secondary_variables


def normalize_list_shapes(parameters, key, default_len):
    """
    Normalize the *shape* of a user-supplied parameter without binding it to a
    wavelength grid that may later change.

    Parsing here is deliberately grid-agnostic: we handle scalar broadcasting
    and preserve ``astropy.Quantity`` units, but we do NOT reject an array whose
    length happens to differ from ``default_len``. The final alignment onto the
    (possibly rebinned) resolved grid is deferred to ingestion time.

    Parameters
    ----------
    key : str
        Name of the parameter to look up in ``parameters``.
    default_len : int
        The expected length based on the *current* input grid. Used only for
        scalar broadcasting and (optionally) strict validation.

    """

    value = parameters[key]

    # Function to convert to float array, preserving Quantity if present
    def to_float_array(v):
        if isinstance(v, u.Quantity):
            return u.Quantity(np.array(v.value, dtype=np.float64), v.unit)
        else:
            return np.array(v, dtype=np.float64)

    if default_len > 1:
        # Case 1 & 1a: default_len > 1 but value is a pure scalar or single-element array
        if (
            np.isscalar(value)
            or (isinstance(value, u.Quantity) and value.isscalar)
            or (isinstance(value, (list, np.ndarray, tuple)) and len(value) == 1)
        ):
            logger.warning(
                f"{key} should be a list of length {default_len}. "
                "pyEDITH will create one assuming the input value for all the elements of the list."
            )
            if isinstance(value, u.Quantity):
                # If it's a single-element quantity array/list, treat it as a scalar
                if not value.isscalar and value.size == 1:
                    value = value[0]
                return u.Quantity(np.full(default_len, value.value), value.unit)
            else:
                # If it's a single-element list, array, or scalar
                if (isinstance(value, (list, np.ndarray, tuple))) and len(value) == 1:
                    value = value[0]
                return np.full(default_len, value, dtype=np.float64)

        # Case 2: default_len > 1 and value has length > 1 but != default_len
        # This is an error: the user provided a multi-element array that doesn't match
        elif (
            isinstance(value, (list, np.ndarray, u.Quantity))
            and len(value) > 1
            and len(value) != default_len
        ):
            raise ValueError(
                f"{key} should be a list of length {default_len}, but it has length {len(value)}."
            )

        # If value is already correct length, just convert to float array
        return to_float_array(value)

    else:
        # Case 3: default_len == 1
        # A scalar or single-element value becomes a length-1 array. A multi-element
        # value is preserved (grid-agnostic) so it can be regridded downstream,
        # unless strict mode collapses it to the first element as before.
        if isinstance(value, u.Quantity):
            if value.size > 1:
                logger.warning(
                    f"{key} has length {len(value)} but the expected input size is {default_len}; "
                    f"leaving it unchanged so it can be aligned/regridded when ingested."
                )
                return u.Quantity(np.array(value.value, dtype=np.float64), value.unit)
            else:
                return u.Quantity([value.value], value.unit)
        elif isinstance(value, (list, np.ndarray)) and len(value) > 1:
            logger.warning(
                f"{key} has length {len(value)} but the expected input size is {default_len}; "
                f"leaving it unchanged so it can be aligned/regridded when ingested."
            )
            return to_float_array(value)
        else:
            return to_float_array([value])


def parse_parameters(parameters: dict, nlambda: int = None) -> dict:
    """
    Parse and process input parameters for simulation.

    This function handles various parameter types including wavelength-dependent parameters,
    target-specific parameters, and scalar values. It converts parameters to appropriate
    data types and ensures arrays have the correct dimensions based on the number of
    wavelength points.

    Parameters
    ----------
    parameters : dict
        A dictionary of input parameters
    nlambda : int, optional
        Number of wavelength points, if not specified in parameters

    Returns
    -------
    dict
        A dictionary of parsed and processed parameters, including: arrays of length
        nlambda (wavelength-dependent parameters), Scalar parameters, Coronagraph
        specifications.

    Raises
    ------
    ValueError
        If wavelength information is missing and nlambda is not provided,
        or if array parameters have incorrect dimensions

    Note
    -----
    The function assumes one target (ntargs = 1) for now.
    nmeananom and norbits are defaulted to 1.
    """

    parsed_params = {}

    # NLAMBDA
    if "wavelength" in parameters.keys():
        if np.isscalar(parameters["wavelength"]) or (
            isinstance(parameters["wavelength"], u.Quantity)
            and np.isscalar(parameters["wavelength"].value)
        ):
            parsed_params["nlambda"] = 1

        else:
            parsed_params["nlambda"] = len(parameters["wavelength"])

        parsed_params["wavelength"] = normalize_list_shapes(
            parameters, "wavelength", parsed_params["nlambda"]
        )

    else:
        raise ValueError(
            "pyEDITH does not have access to wavelength here, please review your input."
        )

    # Use the determined or provided nlambda for array standardization
    nlambda = parsed_params["nlambda"]

    # ------ ARRAYS OF LENGTH NLAMBDA ------

    wavelength_params = [
        "snr",
        "T_optical",
        "epswarmTrcold",
        "npix_multiplier",
        "DC",
        "RN",
        "tread",
        "CIC",
        "QE",
        "dQE",
        "IFS_eff",
        "mag",  # used to be [ntargs x nlambda], now just [nlambda]
        "Fstar_10pc",
        "Fp/Fs",
        "ez_PPF",
        "delta_mag",  # used to be [nmeananom x norbits x ntargs]
        "F0",  # for validation purposes, the calculation of F0 is different in AYO
        "det_npix_input",  # for validation purposes
        "telescope_optical_throughput",
    ]

    parsed_params.update(
        {
            key: normalize_list_shapes(parameters, key, nlambda)
            for key in list(set(wavelength_params) & set(parameters.keys()))
        }
    )

    # ------ SCALARS (USED TO BE ARRAYS IN v. 0.2 and earlier) ------
    target_params = [
        "distance",  # used to be [ntargs]
        "magV",  # used to be [ntargs]
        "FstarV_10pc",
        "stellar_radius",  # used to be [ntargs]
        "nzodis",  # used to be [ntargs]
        "ra",  # used to be [ntargs]
        "dec",  # used to be [ntargs]
        "delta_mag_min",  # used to be [ntargs]
        "Fp_min/Fs",
        "semimajor_axis",
        "separation",  # used to be ARRAYS OF LENGTH  nmeananom x norbits x ntargs (but nmeananom and norbits are defaulted to 1
    ]

    for key in list(set(target_params) & set(parameters.keys())):
        parsed_params[key] = float(parameters[key])

    # ----- SCALARS ----
    scalar_params = [
        "photometric_aperture_radius",
        "psf_trunc_ratio",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "pixscale",  # pixel scale of the coronagraph
        "pixscale_mas",  # pixel scale of the detector
        "contrast",
        "noisefloor_factor",
        "noisefloor_PPF",
        "bandwidth",
        "Tcore",
        "TLyot",
        "unobscured_area",
        "temperature",
        "T_contamination",
        "CRb_multiplier",
        "t_photon_count_input",  # only for ETC validation
    ]

    for key in list(set(scalar_params) & set(parameters.keys())):
        parsed_params[key] = float(parameters[key])

    # ---- INTEGERS ---
    if "nrolls" in parameters.keys():
        parsed_params["nrolls"] = int(parameters["nrolls"])

    # ----- BOOLEANS ---
    for key in ["az_avg", "regrid_wavelength"]:
        if key in parameters.keys():
            value = parameters[key]
            if isinstance(value, str):
                value_lower = value.lower()
                if value_lower in ("true", "1", "yes"):
                    parsed_params[key] = True
                elif value_lower in ("false", "0", "no"):
                    parsed_params[key] = False
                else:
                    raise ValueError(
                        f"Invalid value '{value}' for parameter '{key}'. "
                        f"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
                        f"(case-insensitive)."
                    )
            elif isinstance(value, bool):
                parsed_params[key] = value
            elif isinstance(value, (int, float)):
                if value in (0, 1, 0.0, 1.0):
                    parsed_params[key] = bool(value)
                else:
                    raise ValueError(
                        f"Invalid numeric value '{value}' for parameter '{key}'. "
                        f"Expected 0 or 1 for boolean parameters."
                    )
            else:
                raise TypeError(
                    f"Invalid type {type(value).__name__} for parameter '{key}'. "
                    f"Expected boolean, string, or numeric (0/1)."
                )

    # ----- OBSERVATORY SPECS ---
    for key in [
        "observatory_preset",
        "telescope_type",
        "coronagraph_type",
        "detector_type",
        "observing_mode",
    ]:

        if key in parameters.keys():
            parsed_params[key] = parameters[key]
            if key == "observing_mode" and parameters[key] not in ["IMAGER", "IFS"]:
                raise KeyError("Invalid observing mode. Must be 'IMAGER' or 'IFS'.")

    # --- REQUIRED PARAMETERS IN SPECIFIC MODES ---
    if parsed_params.get("regrid_wavelength", False):
        required_keys = ["spectral_resolution", "lam_low", "lam_high"]
        for key in required_keys:
            if key not in parameters:
                raise KeyError(
                    f"regrid_wavelength is True, but '{key}' is missing. "
                    f"Required parameters: {', '.join(required_keys)}"
                )

        # Check that all required parameters are arrays/lists
        lengths = []
        for key in required_keys:
            val = parameters[key]
            if not isinstance(val, (list, np.ndarray, u.Quantity)):
                raise ValueError(
                    f"regrid_wavelength is True, but '{key}' is not an array. "
                    f"All of {', '.join(required_keys)} must be arrays of the same length."
                )
            lengths.append(len(val))

        # Check that all arrays have the same length
        if len(set(lengths)) != 1:
            raise ValueError(
                f"regrid_wavelength is True, but {', '.join(required_keys)} have different lengths: "
                f"{dict(zip(required_keys, lengths))}. All must have the same length."
            )

        # Add to parsed_params
        for key in required_keys:
            parsed_params[key] = normalize_list_shapes(parameters, key, lengths[0])

    return parsed_params


def read_configuration(
    input_file: Union[Path, str], secondary_flag: bool = False
) -> Tuple[Dict, Dict]:
    """
    Read and parse configuration from an input file.

    This function reads the input file, extracts parameters, and then parses both
    the primary and secondary parameters. It serves as a high-level wrapper around
    parse_input_file() and parse_parameters().

    Parameters
    ----------
    input_file : Union[Path, str]
        Path to the input configuration file
    secondary_flag : bool, optional
        Flag indicating whether secondary variables should be processed, default is False

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple containing two dictionaries:

        parsed_parameters: dict
            Parsed primary parameters

        parsed_secondary_parameters: dict
            Parsed secondary parameters (empty if secondary_flag is False)
    """

    parameters, secondary_parameters = parse_input_file(input_file, secondary_flag)
    parsed_parameters = parse_parameters(parameters)

    if secondary_flag:
        # Parse secondary parameters
        parsed_secondary_parameters = parse_parameters(secondary_parameters)
    else:
        parsed_secondary_parameters = {}

    return parsed_parameters, parsed_secondary_parameters


def get_observatory_config(parameters: Dict[str, str]) -> Union[str, Dict[str, str]]:
    """
    Generate observatory configuration from parameters.

    This function extracts observatory configuration information from the parameters
    dictionary. It either returns a preset name if specified or constructs a custom
    configuration dictionary with telescope, coronagraph, and detector components.

    Parameters
    ----------
    parameters : Dict[str, str]
        Dictionary containing configuration parameters

    Returns
    -------
    Union[str, Dict[str, str]]
        Either a string (if using a preset) or a dictionary (for custom configurations)

    Raises
    ------
    ValueError
        If any required component type is not specified
    """

    if "observatory_preset" in parameters:
        config = parameters["observatory_preset"]
    else:
        config = {}
        for component in ["telescope", "coronagraph", "detector"]:
            component_type = parameters.get(f"{component}_type")
            if component_type is None:
                raise ValueError(
                    f"{component.capitalize()} type not specified. Please provide a '{component}_type' parameter or use a preset."
                )
            config[component] = component_type

    print_observatory_config(config)
    return config


def print_observatory_config(config: Union[str, Dict[str, str]]) -> None:
    """
    Print the observatory configuration to the terminal.

    This function formats and displays the observatory configuration in a
    human-readable format, showing either the preset name or the individual
    component selections.

    Parameters
    ----------
    config : Union[str, Dict[str, str]]
        The observatory configuration, either as a string (preset) or
        a dictionary (custom configuration)
    """

    logger.info("Observatory Configuration:")
    if isinstance(config, str):
        logger.info(f"  Using preset: {config}")
    else:
        logger.info(f"  Telescope:   {config['telescope']}")
        logger.info(f"  Coronagraph: {config['coronagraph']}")
        logger.info(f"  Detector:    {config['detector']}")
    logger.info("")  # Add a blank line for better readability
