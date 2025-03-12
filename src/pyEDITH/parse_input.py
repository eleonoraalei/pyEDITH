from typing import Union, Dict, Tuple
from pathlib import Path
import eacy
import astropy.units as u
import numpy as np


def parse_input_file(file_path: Union[Path, str], secondary_flag, units_flag) -> Tuple[Dict, Dict]:
    """
    Parses an input file and extracts variables and secondary variables.

    Parameters:
    -----------
    file_path : Union[Path,str]
        Path to the input file.
    units_flag : bool
        if true, units should be in input file as the first IDL comment.
        i.e. a line in the input file should look like this:
        parameter = value ;  unit ; parameter description

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

    if units_flag:
        import astropy.units as u
        # define a couple of custom units:
        readout = u.def_unit("readout")
        photon_count = u.def_unit("photon_count")
        units_arr = [line.split(";")[1].strip() for line in content.split("\n")
                if line.strip() and not line.strip().startswith(";")]
        
        # convert to astropy units:
        for u_i in range(len(units_arr)):
            if units_arr[u_i] in ["", None, "dimensionless"]:
                unit_astropy = u.dimensionless_unscaled
            elif units_arr[u_i] == "ct/pix/readout":
                unit_astropy = u.ct/u.pix/readout
            elif units_arr[u_i] == "ct/pix/photon_count":
                unit_astropy = u.ct / u.pix / photon_count
            else:
                try:
                    unit_astropy = u.Unit(units_arr[u_i])
                except ValueError:
                    raise ValueError(f"Invalid unit '{units_arr[u_i]}' ")
            units_arr[u_i] = unit_astropy
        assert len(lines) == len(units_arr)

    variables = {}
    secondary_variables = {}
    if units_flag:
        units = {}
    for i_line, line in enumerate(lines):
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if units_flag:
                val_unit = units_arr[i_line]
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
            if units_flag:
                if "secondary" in key and secondary_flag:
                    secondary_variables[key[10:]] = (value,val_unit)  # it replaces the default value
                else:
                    variables[key] = (value,val_unit)
            else:
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


def parse_parameters(parameters: dict, units_flag=False) -> dict:
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

    def parse_list_param(key, default_len, units_flag):
        if units_flag:
            value = parameters[key][0]
        else:
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
    parsed_params["lambd"] = parse_list_param("lambda", parsed_params["nlambda"], units_flag)
    
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
    ]

    parsed_params.update(
        {
            key: parse_list_param(key, parsed_params["nlambda"], units_flag)
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
            key: parse_list_param(key, parsed_params["ntargs"], units_flag)
            for key in list(set(target_params) & set(parameters.keys()))
        }
    )

    # ----- ARRAYS OF LENGTH NLAMBDA x NTARGS ----
    if "mag" in parameters.keys():
        parsed_params["mag"] = [
            parse_list_param("mag", parsed_params["nlambda"], units_flag)
            for targs in range(parsed_params["ntargs"])
        ]
        parsed_params["mag"] = list(map(list, zip(*parsed_params["mag"])))
    # ---- ARRAYS OF LENGTH  nmeananom x norbits x ntargs (but nmeananom and norbits are defaulted to 1)
    if "separation" in parameters.keys():
        parsed_params["sp"] = [
            [parse_list_param("separation", parsed_params["ntargs"], units_flag)]
        ]
    if "delta_mag" in parameters.keys():
        parsed_params["delta_mag"] = [
            [parse_list_param("delta_mag", parsed_params["ntargs"], units_flag)]
        ]

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
        "CRb_multiplier"
    ]

    for key in list(set(scalar_params) & set(parameters.keys())):
        if units_flag:
            parsed_params[key] = float(parameters[key][0])
        else:
            parsed_params[key] = float(parameters[key])

    # ---- CORONAGRAPH SPECS---
    if "nrolls" in parameters.keys():
        if units_flag:
            parsed_params["nrolls"] = int(parameters["nrolls"][0])
        else:
            parsed_params["nrolls"] = int(parameters["nrolls"])

    # ----- OBSERVATORY SPECS ---
    for key in [
        "observatory_preset",
        "telescope_type",
        "coronagraph_type",
        "detector_type",
    ]:

        if key in parameters.keys():
            parsed_params[key] = parameters[key]

    params_with_units = wavelength_params + target_params + scalar_params
    if units_flag:
        # do lambd since the name changed between parameters and parsed_params (Why is this??)
        parsed_params["lambd"] = (parsed_params["lambd"], parameters["lambda"][1])
        for key in list(set(params_with_units) & set(parameters.keys())):
            # include the unit
            parsed_params[key] = (parsed_params[key], parameters[key][1])

    return parsed_params


def read_configuration(
    input_file: Union[Path, str], secondary_flag=False, units_flag=False,
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

    parameters, secondary_parameters = parse_input_file(input_file, secondary_flag, units_flag)
    parsed_parameters = parse_parameters(parameters, units_flag=units_flag)
    if secondary_flag:
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
        telescope_type = parameters.get("telescope_type", "toymodel")
        coronagraph_type = parameters.get("coronagraph_type", "toymodel")
        detector_type = parameters.get("detector_type", "toymodel")

        if telescope_type == coronagraph_type == detector_type:
            config = telescope_type
        else:
            config = {
                "telescope": f"{telescope_type}Telescope",
                "coronagraph": f"{coronagraph_type}Coronagraph",
                "detector": f"{detector_type}Detector",
            }
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
                    (params["lam"].value >= wavelength_range[0])
                    & (params["lam"].value <= wavelength_range[1])
                ]
            )
    return params
