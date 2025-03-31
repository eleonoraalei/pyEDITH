from typing import Union, Dict, Tuple
from pathlib import Path
import eacy
import astropy.units as u
import numpy as np
from .units import *
from scipy.interpolate import interp1d



def parse_input_file(file_path: Union[Path, str], secondary_flag) -> Tuple[Dict, Dict]:
    """
    Parses an input file and extracts variables and secondary variables.

    Parameters:
    -----------
    file_path : Union[Path,str]
        Path to the input file.

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
            # Handle numbers
            else:
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string if it's not a number

            if "secondary" in key and secondary_flag:
                secondary_variables[key[10:]] = value  # it replaces the default value
            else:
                variables[key] = value

    # Fill in missing secondary variables with primary variable values
    # if secondary_flag:
    #     for key in variables:
    #         if key not in secondary_variables:
    #             secondary_variables[key] = variables[key]

    return variables, secondary_variables


def checks_on_list_values(key: str, value, length: int) -> bool:
    """
    Check that a list has exactly the specified length.

    Parameters:
    -----------
    key : str
        The name of the parameter being checked.
    value : any
        The value to be checked.
    length : int
        The expected length of the list.

    Returns:
    --------
    bool
        True if the value is a list of the specified length.

    Raises:
    -------
    ValueError
        If the value is not a list or does not have the specified length.
    """

    if not isinstance(value, (list)) or len(list(value)) != length:
        raise ValueError(key + " should be a list of length " + str(length))
    return True


def parse_parameters(parameters: dict) -> dict:
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
        if default_len > 1:
            # it is supposed to be a list.
            checks_on_list_values(key, value, default_len)
            return [float(v) for v in value]
        else:
            return [float(value)]

    parsed_params = {}

    # CONSTANTS
    if isinstance(parameters["lambda"], list):
        parsed_params["nlambda"] = len(parameters["lambda"])
    else:
        parsed_params["nlambda"] = 1

    parsed_params["ntargs"] = 1  # For now, we assume one target

    # ------ ARRAYS OF LENGTH NLAMBDA ------
    parsed_params["lambd"] = parse_list_param("lambda", parsed_params["nlambda"])

    wavelength_params = [
        "resolution",
        "snr",
        "Toptical",
        "epswarmTrcold",
        "npix_multiplier",
        "dark_current",
        "read_noise",
        "read_time",
        "cic",
        "QE",
        "dQE",
        "IFS_eff",
        "mag",  # used to be [ntargs x nlambda], now just [nlambda]
    ]

    parsed_params.update(
        {
            key: parse_list_param(key, parsed_params["nlambda"])
            for key in list(set(wavelength_params) & set(parameters.keys()))
        }
    )

    # ------ SCALARS (USED TO BE ARRAYS IN v. 0.2 and earlier) ------
    target_params = [
        "Lstar",  # used to be [ntargs]
        "distance",  # used to be [ntargs]
        "magV",  # used to be [ntargs]
        "angdiam",  # used to be [ntargs]
        "nzodis",  # used to be [ntargs]
        "ra",  # used to be [ntargs]
        "dec",  # used to be [ntargs]
        "delta_mag_min",  # used to be [ntargs]
        "delta_mag",  # used to be [nmeananom x norbits x ntargs]
    ]

    for key in list(set(target_params) & set(parameters.keys())):
        parsed_params[key] = float(parameters[key])

    # ---- MORE SCALARS (used to be ARRAYS OF LENGTH  nmeananom x norbits x ntargs (but nmeananom and norbits are defaulted to 1)
    if "separation" in parameters.keys():
        parsed_params["sp"] = float(parameters["separation"])

    # ----- SCALARS ----
    scalar_params = [
        "photap_rad",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "minimum_IWA",
        "maximum_OWA",
        "contrast",
        "noisefloor_factor",
        "bandwidth",
        "Tcore",
        "TLyot",
        "temperature",
        "Tcontam",
        "CRb_multiplier",
    ]

    for key in list(set(scalar_params) & set(parameters.keys())):
        parsed_params[key] = float(parameters[key])

    # ---- CORONAGRAPH SPECS---
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
            ynew =  interp_func(wavelengths) # interpolates the CG throughput values onto native wl grid
            params[key] = ynew
    return params

