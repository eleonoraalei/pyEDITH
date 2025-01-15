from abc import ABC, abstractmethod
from pyEDITH.telescope import Telescope
from pyEDITH.detector import Detector
from pyEDITH.coronagraph import Coronagraph


class Instrument(ABC):  # abstract class
    """
    Abstract base class for various instruments.

    This class defines the basic structure for various instruments
    used in the calculation. It includes abstract methods that must be implemented
    by concrete subclasses.

    Attributes
    ----------
    telescope : Telescope
        The telescope component of the instrument.
    detector : Detector
        The detector component of the instrument.
    coronagraph : Coronagraph
        The coronagraph component of the instrument.

    Methods
    -------
    initialize(parameters: dict) -> None
        Abstract method to initialize the instrument with given parameters.
    """

    def __init__(self):
        """
        Initialize an Instrument instance.

        This constructor initializes the basic components of the instrument
        (telescope, detector, and coronagraph) to None. These components
        should be properly initialized in subclasses using the `initialize` method.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.telescope: Telescope = None
        self.detector: Detector = None
        self.coronagraph: Coronagraph = None

    @abstractmethod
    def initialize(self, parameters: dict) -> None:
        """
        Initialize the instrument with given parameters.

        This abstract method should be implemented by concrete subclasses
        to set up the instrument with the provided parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters needed to initialize
            the instrument and its components.

        Returns
        -------
        None

        Notes
        -----
        This method is abstract and must be implemented by subclasses.
        It should typically initialize the telescope, detector, and
        coronagraph components of the instrument.
        """
        pass
