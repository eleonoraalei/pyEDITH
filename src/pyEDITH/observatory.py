from abc import ABC, abstractmethod
import numpy as np


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

    def load_configuration(self, parameters, observation, scene):

        # Creates a mediator that picks selected variables from other classes
        mediator = ObservatoryMediator(self, observation, scene)

        self.coronagraph.load_configuration(parameters, mediator)
        self.telescope.load_configuration(parameters, mediator)
        self.detector.load_configuration(parameters, mediator)

    def validate_configuration(self):
        self.telescope.validate_configuration()
        self.detector.validate_configuration()
        self.coronagraph.validate_configuration()


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
