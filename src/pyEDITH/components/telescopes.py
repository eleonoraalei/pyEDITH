from abc import ABC, abstractmethod
import numpy as np


class Telescope(ABC):
    """
    A class representing a telescope for astronomical observations.

    This class provides methods to initialize and configure a telescope
    for simulating observations of exoplanets and their host stars.

    Attributes
    ----------
    diameter : float
        Circumscribed diameter of the telescope aperture in meters.
    Area : float
        Effective collecting area of the telescope in square meters.
    toverhead_fixed : float
        Fixed overhead time in seconds.
    toverhead_multi : float
        Multiplicative overhead time.
    throughput : numpy.ndarray
        Array of throughput values (excluding coronagraph core throughput).

    """

    @abstractmethod
    def load_configuration(self):
        pass

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "diameter": (float, np.floating),
            "Area": (float, np.floating),
            "toverhead_fixed": (float, np.floating),
            "toverhead_multi": (float, np.floating),
            "T_optical": np.ndarray,
        }

        for arg, expected_type in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Telescope is missing attribute: {arg}")
            if not isinstance(getattr(self, arg), expected_type):
                raise TypeError(
                    f"Telescope attribute {arg} should be of type {expected_type}, but is {type(getattr(self, arg))}"
                )


class ToyModelTelescope(Telescope):
    """
    A toy model telescope class that extends the base Telescope class.

    This class represents a simplified telescope model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "diameter": 7.87,  # circumscribed diameter of aperture (m, scalar)
        "unobscured_area": 1.0 - 0.121,  # unobscured area (percentage,scalar)
        "toverhead_fixed": 8.25e3,  # fixed overhead time (seconds,scalar)
        "toverhead_multi": 1.1,  # multiplicative overhead time (scalar)
        "T_optical": [0.5333333],  # Optical throughput (nlambd array)
    }

    def load_configuration(self, parameters, mediator) -> None:
        """
        Load configuration parameters for the simulation from a dictionary of
        parameters that was read from the input file. If not provided, use default values.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        ALL OF THE CLASSES
        Returns
        -------
        None
        """

        # load parameters from input file if available, otherwise use default
        for key, default_value in self.DEFAULT_CONFIG.items():
            setattr(self, key, parameters.get(key, default_value))

        # Convert to numpy array when appropriate
        array_params = [
            "T_optical",
        ]
        for param in array_params:
            setattr(self, param, np.array(getattr(self, param), dtype=np.float64))

        # Derived parameters
        # effective collecting area of telescope (m^2) # scalar
        self.Area = np.single(np.pi) / 4.0 * self.diameter**2.0 * self.unobscured_area
