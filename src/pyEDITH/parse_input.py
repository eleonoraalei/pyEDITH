from typing import Union, Dict, Tuple
from pathlib import Path


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
        - Arrays of length ntargs (target-specific parameters)
        - Arrays of length ntargs x nlambda
        - Arrays of length nmeananom x norbits x ntargs
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
        "throughput",
        "npix_multiplier",
        "dark_current",
        "read_noise",
        "read_time",
        "cic",
    ]

    parsed_params.update(
        {
            key: parse_list_param(key, parsed_params["nlambda"])
            for key in list(set(wavelength_params) & set(parameters.keys()))
        }
    )

    # ------ ARRAYS OF LENGTH NTARGS ------
    target_params = [
        "Lstar",
        "distance",
        "magV",
        "angdiam",
        "nzodis",
        "ra",
        "dec",
        "delta_mag_min",
    ]

    parsed_params.update(
        {
            key: parse_list_param(key, parsed_params["ntargs"])
            for key in list(set(target_params) & set(parameters.keys()))
        }
    )

    # ----- ARRAYS OF LENGTH NLAMBDA x NTARGS ----
    if "mag" in parameters.keys():
        parsed_params["mag"] = [
            parse_list_param("mag", parsed_params["nlambda"])
            for targs in range(parsed_params["ntargs"])
        ]
        parsed_params["mag"] = list(map(list, zip(*parsed_params["mag"])))
    # ---- ARRAYS OF LENGTH  nmeananom x norbits x ntargs (but nmeananom and norbits are defaulted to 1)
    if "separation" in parameters.keys():
        parsed_params["sp"] = [
            [parse_list_param("separation", parsed_params["ntargs"])]
        ]
    if "delta_mag" in parameters.keys():
        parsed_params["delta_mag"] = [
            [parse_list_param("delta_mag", parsed_params["ntargs"])]
        ]

    # ----- SCALARS ----
    scalar_params = [
        "photap_rad",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "IWA",
        "OWA",
        "contrast",
        "noisefloor_factor",
        "bandwidth",
        "core_throughput",
        "Lyot_transmission",
    ]

    for key in list(set(scalar_params) & set(parameters.keys())):
        parsed_params[key] = float(parameters[key])

    # ---- CORONAGRAPH SPECS---
    if "type" in parameters.keys():
        parsed_params["coro_type"] = str(parameters["type"])
    if "nrolls" in parameters.keys():
        parsed_params["nrolls"] = int(parameters["nrolls"])

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
        parsed_secondary_parameters = parse_parameters(secondary_parameters)
    else:
        parsed_secondary_parameters = {}
    return parsed_parameters, parsed_secondary_parameters


# def read_inputs(args):
#     '''
#     The goal of this function is to read the three input files (.ayo file, .coro file, and the target list)
#     and return a list of parameters that will be ingested in the edith object'''
