from abc import ABC, abstractmethod
import numpy as np
from .units import *


class Observatory(ABC):  # abstract class
    """
    Abstract base class for various Observatories.

    This class defines the basic structure for various Observatories.
    used in the calculation. It includes abstract methods that must be implemented
    by concrete subclasses.

    Attributes
    ----------
    telescope : Telescope
        The telescope component of the observatory.
    detector : Detector
        The detector component of the observatory.
    coronagraph : Coronagraph
        The coronagraph component of the observatory.

    Methods
    -------
    initialize(parameters: dict) -> None
        Abstract method to initialize the observatory with given parameters.
    """

    def __init__(self):
        """
        Initialize an Observatory instance.

        This constructor initializes the basic components of the observatory
        (telescope, detector, and coronagraph) to None. These components
        should be properly initialized in subclasses using the `initialize` method.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.telescope = None
        self.detector = None
        self.coronagraph = None

    def calculate_optics_throughput(self, parameters):
        """
        This function calculates the optical throughput. If there is a variable called
        total_optics_throughput (aka Toptical) in the parameters,
        use that one instead of telescope*coronagraph.
        """
        if "Toptical" in parameters.keys():
            print("Calculating optics_throughput from input...")
            self.optics_throughput = parameters["Toptical"] * DIMENSIONLESS
        else:
            print("Calculating optics throughput from preset...")
            self.optics_throughput = (
                self.telescope.telescope_throughput
                * self.coronagraph.coronagraph_throughput
            )

        if parameters["observing_mode"] == "IFS":
            # multiply by the IFS efficiency if in spectroscopy mode
            self.optics_throughput *= parameters["IFS_eff"] # NOTE: this is a placeholder for now. Not yet included in YAML files. Name will probably change.
        else:
            pass

    def calculate_warmemissivity_coldtransmission(self, parameters):
        """
        This function calculates the warm emissivity*cold transmission factor
        (for thermal noise). If there is a variable called
        epswarmTrcold in the parameters, use that one instead of 1-optics_throughput
        """
        if "epswarmTrcold" in parameters.keys():
            print("Calculating epswarmTrcold from input...")
            self.epswarmTrcold = parameters["epswarmTrcold"] * DIMENSIONLESS
        else:
            print("Calculating epswarmTrcold as 1 - optics throughput...")
            self.epswarmTrcold = 1 - self.optics_throughput

    def calculate_total_throughput(self):
        """
        This function calculates the optical (telescope + instrument path) + detector
        throughput, which is used as multiplicative factor when calculating the noise terms.
        If there is a variable called total_optics_throughput (aka Toptical) in the parameters,
        use that one instead of telescope*coronagraph.
        """
        self.total_throughput = (
            self.optics_throughput
            * self.detector.dQE
            * self.detector.QE
            * self.telescope.Tcontam
        )

    def load_configuration(self, parameters, observation, scene):

        # Creates a mediator that picks selected variables from other classes
        mediator = ObservatoryMediator(self, observation, scene)

        self.coronagraph.load_configuration(parameters, mediator)
        self.telescope.load_configuration(parameters, mediator)
        self.detector.load_configuration(parameters, mediator)

        self.calculate_optics_throughput(parameters)
        self.calculate_warmemissivity_coldtransmission(parameters)
        self.calculate_total_throughput()

    def validate_configuration(self):
        self.telescope.validate_configuration()
        self.detector.validate_configuration()
        self.coronagraph.validate_configuration()

        # Observatory-related args
        expected_args = {
            "total_throughput": QE,
            "optics_throughput": DIMENSIONLESS,
            "epswarmTrcold": DIMENSIONLESS,
        }

        for arg, expected_unit in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Observatory is missing attribute: {arg}")
            value = getattr(self, arg)
            if not isinstance(value, u.Quantity):
                raise TypeError(f"Observatory attribute {arg} should be a Quantity")
            if not value.unit.is_equivalent(expected_unit):
                raise ValueError(
                    f"Observatory attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
                )


class ObservatoryMediator:
    def __init__(self, observatory, observation, scene):
        self.observatory = observatory
        self.observation = observation
        self.scene = scene

    def get_telescope_parameter(self, param_name):
        return getattr(self.observatory.telescope, param_name, None)

    def get_coronagraph_parameter(self, param_name):
        return getattr(self.observatory.coronagraph, param_name, None)

    def get_detector_parameter(self, param_name):
        return getattr(self.observatory.detector, param_name, None)

    def get_observation_parameter(self, param_name):
        return getattr(self.observation, param_name, None)

    def get_scene_parameter(self, param_name):
        return getattr(self.scene, param_name, None)
