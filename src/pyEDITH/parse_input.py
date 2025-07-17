from typing import Union, Dict, Tuple
from pathlib import Path
import astropy.units as u
import numpy as np
from .units import *
import pandas as pd
import os


def parse_input_file(file_path: Union[Path, str], secondary_flag) -> Tuple[Dict, Dict]:
    """
    Parses an input file and extracts variables and secondary variables.

    Parameters:
    -----------
    file_path : Union[Path,str]
        Path to the input file.
    secondary_flag : bool
        Flag indicating whether secondary variables are expected.

    Returns:
    --------
    Tuple[Dict, Dict]
        A tuple containing two dictionaries:
        - variables: Primary variables extracted from the file.
        - secondary_variables: Secondary variables extracted from the file
          (any non-specified variable will be the same as in the variables dictionary).

    Notes:
    ------
    The function handles various data types including arrays, strings, and numbers.
    Comments in the input file should start with ';'.
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

        else:
            raise ValueError(
                "Required parameters 'wavelength', 'Fstar_10pc', and 'Fp/Fs' are not provided. Please write them explicitly or provide a spectrum_file path."
            )

    if variables.get("observing_mode") == "IMAGER" and isinstance(
        variables["wavelength"], list
    ):
        raise KeyError(
            "In IMAGER mode you can only use one wavelength at a time. If you are simulating photometry, please run every single wavelength separately. If you want to model a spectrum, please use IFS mode."
        )
    return variables, secondary_variables


def parse_parameters(parameters: dict, nlambda=None) -> dict:
    """
    Parses and processes input parameters for the Edith simulation.

    This function handles various parameter types including wavelength-dependent parameters,
    target-specific parameters, and scalar values. It also processes coronagraph specifications.

    Parameters:
    -----------
    parameters : dict
        A dictionary of input parameters.

    Returns:
    --------
    dict
        A dictionary of parsed and processed parameters, including:
        - Arrays of length nlambda (wavelength-dependent parameters)
        - Scalar parameters
        - Coronagraph specifications

    Notes:
    ------
    The function assumes one target (ntargs = 1) for now.
    nmeananom and norbits are defaulted to 1.
    """

    def parse_list_param(key, default_len):
        value = parameters[key]

        # Function to convert to float array, preserving Quantity if present
        def to_float_array(v):
            if isinstance(v, u.Quantity):
                return u.Quantity(np.array(v.value, dtype=np.float64), v.unit)
            else:
                return np.array(v, dtype=np.float64)

        if default_len > 1:
            # Case 1 & 1a: default_len > 1 but value is a pure scalar (including Quantity scalar)
            if np.isscalar(value) or (isinstance(value, u.Quantity) and value.isscalar):
                print(
                    f"WARNING: {key} should be a list of length {default_len}. pyEDITH will create one assuming the input value for all the elements of the list."
                )
                if isinstance(value, u.Quantity):
                    return u.Quantity(np.full(default_len, value.value), value.unit)
                else:
                    return np.full(default_len, value)

            # Case 2: default_len > 1 but value has a length > 1 and != default_len
            elif (
                isinstance(value, (list, np.ndarray, u.Quantity))
                and len(value) != default_len
            ):
                raise ValueError(
                    f"{key} should be a list of length {default_len}, but it has length {len(value)}."
                )

            # If value is already correct length, just convert to float array
            return to_float_array(value)

        else:
            # Case 3: default_len == 1, return a single element array
            if isinstance(value, u.Quantity):
                if value.size > 1:
                    print(
                        f"WARNING: {key} should be a list of length 1 but you assigned multiple values. pyEDITH will create a list assuming only the first input value."
                    )
                    return u.Quantity([value[0].value], value.unit)
                else:
                    return u.Quantity([value.value], value.unit)
            elif isinstance(value, (list, np.ndarray)) and len(value) > 1:
                print(
                    f"WARNING: {key} should be a list of length 1 but you assigned multiple values. pyEDITH will create a list assuming only the first input value."
                )
                return to_float_array([value[0]])
            else:
                return to_float_array([value])

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
        parsed_params["wavelength"] = parse_list_param(
            "wavelength", parsed_params["nlambda"]
        )

    elif nlambda is not None:
        parsed_params["nlambda"] = nlambda
    else:
        raise ValueError(
            "pyEDITH does not have access to wavelength here, you should provide nlambda as an argument to this function."
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
        "delta_mag",  # used to be [nmeananom x norbits x ntargs]
        "F0",  # for validation purposes, the calculation of F0 is different in AYO
        "det_npix_input",  # for validation purposes
    ]

    parsed_params.update(
        {
            key: parse_list_param(key, nlambda)
            for key in list(set(wavelength_params) & set(parameters.keys()))
        }
    )

    # ------ SCALARS (USED TO BE ARRAYS IN v. 0.2 and earlier) ------
    target_params = [
        "Lstar",  # used to be [ntargs]
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
        "minimum_IWA",
        "maximum_OWA",
        "contrast",
        "noisefloor_factor",
        "noisefloor_PPF",
        "ez_PPF",
        "bandwidth",
        "Tcore",
        "TLyot",
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

    return parsed_params


def read_configuration(
    input_file: Union[Path, str], secondary_flag=False
) -> Tuple[Dict, Dict]:
    """
    Reads and parses the configuration from an input file.

    This function reads the input file, extracts parameters, and then parses both
    the primary and secondary parameters.

    Parameters:
    -----------
    input_file : Union[Path,str]
        Path to the input configuration file.

    Returns:
    --------
    Tuple[Dict, Dict]
        A tuple containing two dictionaries:
        - parsed_parameters: Parsed primary parameters.
        - parsed_secondary_parameters: Parsed secondary parameters.

    Notes:
    ------
    This function uses parse_input_file() to read the raw parameters and
    parse_parameters() to process them.
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

    Returns either a string (if all components are from the same type) or a dictionary (for mixed configurations).
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

    Parameters:
    -----------
    config : Union[str, Dict[str, str]]
        The observatory configuration, either as a string (preset) or a dictionary (custom).
    """
    print("Observatory Configuration:")
    if isinstance(config, str):
        print(f"  Using preset: {config}")
    else:
        print(f"  Telescope:   {config['telescope']}")
        print(f"  Coronagraph: {config['coronagraph']}")
        print(f"  Detector:    {config['detector']}")
    print()  # Add a blank line for better readability
