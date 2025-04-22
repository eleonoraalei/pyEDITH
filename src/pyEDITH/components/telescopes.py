from abc import ABC, abstractmethod
import numpy as np
from .. import utils
import astropy.units as u
from ..units import *


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
    telescope_throughput : numpy.ndarray
        Array of throughput values.
    temperature : float
        Temperature of the warm optics.
    Tcontam : float
        Effective throughput factor to budget for contamination.
    """

    @abstractmethod
    def load_configuration(self):
        pass  # pragma: no cover

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "diameter": LENGTH,
            "Area": LENGTH**2,
            "toverhead_fixed": TIME,
            "toverhead_multi": DIMENSIONLESS,
            "telescope_throughput": DIMENSIONLESS,
            "temperature": TEMPERATURE,
            "Tcontam": DIMENSIONLESS,
        }

        utils.validate_attributes(self, expected_args)

        # for arg, expected_unit in expected_args.items():
        #     if not hasattr(self, arg):
        #         raise AttributeError(f"Telescope is missing attribute: {arg}")
        #     value = getattr(self, arg)
        #     if not isinstance(value, u.Quantity):
        #         raise TypeError(f"Telescope attribute {arg} should be a Quantity")
        #     if not value.unit == expected_unit:
        #         raise ValueError(
        #             f"Telescope attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
        #         )


class ToyModelTelescope(Telescope):
    """
    A toy model telescope class that extends the base Telescope class.

    This class represents a simplified telescope model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "diameter": 7.87 * LENGTH,  # circumscribed diameter of aperture (m, scalar)
        "unobscured_area": (1.0 - 0.121),  # unobscured area (percentage,scalar)
        "toverhead_fixed": 8.25e3 * TIME,  # fixed overhead time (seconds,scalar)
        "toverhead_multi": 1.1 * DIMENSIONLESS,  # multiplicative overhead time (scalar)
        "telescope_throughput": [0.823]
        * DIMENSIONLESS,  # Optical throughput (nlambd array) [made up from EAC1-ish]
        "temperature": 290 * TEMPERATURE,
        "Tcontam": 0.95 * DIMENSIONLESS,
    }

    def __init__(self, path=None, keyword=None):
        self.path = path
        self.keyword = keyword

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
        # Load parameters, use defaults if not provided
        utils.fill_parameters(self, parameters, self.DEFAULT_CONFIG)

        # Convert to numpy array when appropriate
        array_params = [
            "telescope_throughput",
        ]
        utils.convert_to_numpy_array(self, array_params)

        # Derived parameters
        # effective collecting area of telescope (m^2) # scalar
        self.Area = np.single(np.pi) / 4.0 * self.diameter**2.0 * self.unobscured_area


class EACTelescope(Telescope):
    """
    A toy model telescope class that extends the base Telescope class.

    This class represents a simplified telescope model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "diameter": None,  # circumscribed diameter of aperture (m, scalar)
        "unobscured_area": 1.0,  # unobscured area (percentage,scalar) ### NOTE default for now
        "toverhead_fixed": 8.25e3
        * TIME,  # fixed overhead time (seconds,scalar) ### NOTE default for now
        "toverhead_multi": 1.1
        * DIMENSIONLESS,  # multiplicative overhead time (scalar) ### NOTE default for now
        "telescope_throughput": None,  # Optical throughput (nlambd array)
        "Tcontam": 1.0
        * DIMENSIONLESS,  # Effective throughput factor to budget for contamination; NOTE: missing from YAML files
        "temperature": 290
        * TEMPERATURE,  # Temperature of the warm optics; NOTE: missing from YAML files
    }

    def __init__(self, path=None, keyword=None):
        self.path = path
        self.keyword = keyword

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
        # Check on possible modes
        if parameters["observing_mode"] not in ["IFS", "IMAGER"]:
            raise KeyError(
                f"Unsupported observing mode: {parameters['observing_mode']}"
            )

        from eacy import load_telescope

        # **** LOAD DEFAULTS FROM EAC YAML FILES AND UPDATE DEFAULT CONFIG ****

        # Load parameters from YAML files
        telescope_params = load_telescope(self.keyword).__dict__

        if parameters["observing_mode"] == "IMAGER":
            wavelength_range = [
                mediator.get_observation_parameter("wavelength")
                * (1 - 0.5 * mediator.get_coronagraph_parameter("bandwidth")),
                mediator.get_observation_parameter("wavelength")
                * (1 + 0.5 * mediator.get_coronagraph_parameter("bandwidth")),
            ] * WAVELENGTH

            telescope_params = utils.average_over_bandpass(
                telescope_params, wavelength_range
            )

        elif parameters["observing_mode"] == "IFS":
            # interpolate telescope throughput onto native wavelength grid
            telescope_params = utils.interpolate_over_bandpass(
                telescope_params, mediator.get_observation_parameter("wavelength")
            )

        # Load parameters that you need from the YAML files
        self.DEFAULT_CONFIG["diameter"] = telescope_params["diam_circ"] * LENGTH

        # Ensure telescope_throughput has dimensions nlambda
        if np.isscalar(telescope_params["total_tele_refl"]):
            self.DEFAULT_CONFIG["telescope_throughput"] = (
                np.array([telescope_params["total_tele_refl"]]) * DIMENSIONLESS
            )
        else:
            self.DEFAULT_CONFIG["telescope_throughput"] = (
                np.array(telescope_params["total_tele_refl"]) * DIMENSIONLESS
            )
        # ****** Update Default Config when necessary ******
        # TODO: wavelength_range probably should not depend on the coronagraph bandwidth; let's discuss
        # the coronagraph module needs the telescope module to be initialized first to get the telescope diameter

        # Load parameters, use defaults if not provided
        utils.fill_parameters(self, parameters, self.DEFAULT_CONFIG)

        # Derived parameters
        # effective collecting area of telescope (m^2) # scalar
        self.Area = np.single(np.pi) / 4.0 * self.diameter**2.0 * self.unobscured_area
