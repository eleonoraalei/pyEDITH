from abc import ABC, abstractmethod
import numpy as np
from .units import *
from . import utils


class Observatory(ABC):  # abstract class
    """
    Abstract base class for various astronomical observatories.

    This class defines the basic structure for modeling astronomical observatories
    used in performance calculations. It includes abstract methods that must be
    implemented by concrete subclasses and provides common functionality for
    calculating optical throughput, thermal properties, and component validation.

    Parameters
    ----------
    telescope : Telescope
        The telescope component of the observatory
    detector : Detector
        The detector component of the observatory
    coronagraph : Coronagraph
        The coronagraph component of the observatory
    optics_throughput : np.ndarray
        Optical throughput of the telescope and coronagraph system
    epswarmTrcold : np.ndarray
        Warm emissivity times cold transmission factor for thermal noise
    total_throughput : np.ndarray
        Combined optical and detector throughput
    observing_mode : str
        Observing mode ('IFS' or 'IMAGER')
    """

    def __init__(self):
        """
        Initialize an Observatory instance.

        This constructor initializes the basic components of the observatory
        (telescope, detector, and coronagraph) to None. These components
        should be properly initialized in subclasses using the `load_configuration` method.
        """

        self.telescope = None
        self.detector = None
        self.coronagraph = None

    def calculate_optics_throughput(self, parameters: dict, mediator: object) -> None:
        """
        Calculate the optical throughput of the observatory system.

        This method computes the optical throughput by either using a provided
        total optical throughput value (T_optical) from parameters, or by
        multiplying the telescope and coronagraph throughputs. For IFS mode,
        an additional IFS efficiency factor is applied. If optics_throughput is
        a scalar and wavelength array has multiple elements, the throughput is
        expanded to match the wavelength array length.

        Parameters
        ----------
        parameters : dict
            Configuration parameters dictionary that may contain 'T_optical',
            'observing_mode', and 'IFS_eff' keys
        mediator : ObservatoryMediator
            Mediator object providing access to observation parameters including
            wavelength array
        """

        if "T_optical" in parameters.keys():
            print("Calculating optics_throughput from input...")
            self.optics_throughput = parameters["T_optical"] * DIMENSIONLESS
        else:
            print("Calculating optics throughput from preset...")
            self.optics_throughput = (
                self.telescope.telescope_optical_throughput
                * self.coronagraph.coronagraph_optical_throughput
            )

        if parameters["observing_mode"] == "IFS":
            # multiply by the IFS efficiency if in spectroscopy mode
            # NOTE: this is a placeholder for now. Not yet included in YAML files. Name will probably change.
            # may also move to elsewhere in code.
            ifs_eff = u.Quantity(
                parameters.get("IFS_eff", 1.0), unit=u.dimensionless_unscaled
            )

            self.optics_throughput *= ifs_eff

        # if optics_throughput is a number and wavelength>1, make it an array of length nlambda
        if len(self.optics_throughput) == 1:
            self.optics_throughput = self.optics_throughput[0] * np.ones_like(
                mediator.get_observation_parameter("wavelength").value
            )

    def calculate_warmemissivity_coldtransmission(
        self, parameters: dict, mediator: object
    ) -> None:
        """
        Calculate the warm emissivity times cold transmission factor.

        This method computes the factor used for thermal noise calculations.
        It either uses a provided 'epswarmTrcold' value from parameters, or
        calculates it as (1 - optics_throughput).

        Parameters
        ----------
        parameters : dict
            Configuration parameters dictionary that may contain 'epswarmTrcold' key
        mediator : ObservatoryMediator
            Mediator object providing access to observation parameters including
            wavelength array
        """

        if "epswarmTrcold" in parameters.keys():
            print("Calculating epswarmTrcold from input...")
            self.epswarmTrcold = parameters["epswarmTrcold"] * DIMENSIONLESS
        else:
            print("Calculating epswarmTrcold as 1 - optics throughput...")
            self.epswarmTrcold = (
                np.ones_like(mediator.get_observation_parameter("wavelength").value)
                - self.optics_throughput
            )

    def calculate_total_throughput(self) -> None:
        """
        Calculate the total system throughput.

        This method computes the combined optical and detector throughput by
        multiplying the optics throughput with the detector quantum efficiency (QE),
        detector QE, and telescope contamination factor. This total throughput
        is used as a multiplicative factor in noise calculations.
        """

        self.total_throughput = (
            self.optics_throughput
            * self.detector.dQE
            * self.detector.QE
            * self.telescope.T_contamination
        )

    def load_configuration(
        self, parameters: dict, observation: object, scene: object
    ) -> None:
        """
        Load and configure all observatory components.

        This method initializes all observatory components (coronagraph, telescope,
        detector) with the provided parameters and calculates derived quantities
        like throughputs and thermal factors. Creates a mediator for component
        communication and sets the observing mode.

        Parameters
        ----------
        parameters : dict
            Configuration parameters dictionary containing observatory settings
        observation : Observation
            Observation object containing observational parameters
        scene : AstrophysicalScene
            Scene object containing target and environmental parameters
        """

        # Creates a mediator that picks selected variables from other classes
        mediator = ObservatoryMediator(self, observation, scene)

        self.coronagraph.load_configuration(parameters, mediator)
        self.telescope.load_configuration(parameters, mediator)
        self.detector.load_configuration(parameters, mediator)
        self.observing_mode = parameters["observing_mode"]  # IFS or IMAGER

        self.calculate_optics_throughput(parameters, mediator)
        self.calculate_warmemissivity_coldtransmission(parameters, mediator)
        self.calculate_total_throughput()

    def validate_configuration(self) -> None:
        """
        Validate that all observatory components and parameters are correctly configured.

        This method validates all sub-components (telescope, detector, coronagraph)
        and checks that required observatory-level attributes exist with correct
        types and units. Observatory-related parameters include total_throughput,
        optics_throughput, and epswarmTrcold.

        Raises
        ------
        AttributeError
            If required attributes are missing from the observatory or its components
        TypeError
            If an attribute has an incorrect type
        ValueError
            If a Quantity attribute has incorrect units
        """

        self.telescope.validate_configuration()
        self.detector.validate_configuration()
        self.coronagraph.validate_configuration()

        # Observatory-related args
        expected_args = {
            "total_throughput": QUANTUM_EFFICIENCY,
            "optics_throughput": DIMENSIONLESS,
            "epswarmTrcold": DIMENSIONLESS,
        }
        utils.validate_attributes(self, expected_args)
        # for arg, expected_unit in expected_args.items():
        #     if not hasattr(self, arg):
        #         raise AttributeError(f"Observatory is missing attribute: {arg}")
        #     value = getattr(self, arg)
        #     if not isinstance(value, u.Quantity):
        #         raise TypeError(f"Observatory attribute {arg} should be a Quantity")
        #     if not value.unit.is_equivalent(expected_unit):
        #         raise ValueError(
        #             f"Observatory attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
        #         )


class ObservatoryMediator:
    """
    Mediator class facilitating communication between observatory components.

    This class provides a centralized interface for accessing parameters from
    different components (observatory, observation, scene) without creating
    direct dependencies between them. It implements the mediator design pattern
    to decouple component interactions.

    Parameters
    ----------
    observatory : Observatory
        The observatory object
    observation : Observation
        The observation object containing observational parameters
    scene : AstrophysicalScene
        The scene object containing target and environmental parameters
    """

    def __init__(self, observatory: object, observation: object, scene: object):
        """
        Initialize the mediator with references to all major components.

        Parameters
        ----------
        observatory : Observatory
            The observatory object
        observation : Observation
            The observation object
        scene : AstrophysicalScene
            The scene object
        """

        self.observatory = observatory
        self.observation = observation
        self.scene = scene

    def get_telescope_parameter(self, param_name: str):
        """
        Retrieve a parameter from the telescope object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.observatory.telescope, param_name, None)

    def get_coronagraph_parameter(self, param_name: str):
        """
        Retrieve a parameter from the coronagraph object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.observatory.coronagraph, param_name, None)

    def get_detector_parameter(self, param_name: str):
        """
        Retrieve a parameter from the detector object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """

        return getattr(self.observatory.detector, param_name, None)

    def get_observation_parameter(self, param_name: str):
        """
        Retrieve a parameter from the observation object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.observation, param_name, None)

    def get_scene_parameter(self, param_name: str):
        """
        Retrieve a parameter from the scene object.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve

        Returns
        -------
        Any or None
            The parameter value if it exists, None otherwise
        """
        return getattr(self.scene, param_name, None)
