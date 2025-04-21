from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u

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