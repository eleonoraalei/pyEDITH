from abc import ABC, abstractmethod
from pyEDITH.telescope import Telescope
from pyEDITH.detector import Detector
from pyEDITH.coronagraph import Coronagraph


class Instrument(ABC):  # abstract class
    def __init__(self):
        self.telescope: Telescope = None
        self.detector: Detector = None
        self.coronagraph: Coronagraph = None

    @abstractmethod
    def initialize(self, parameters: dict) -> None:
        pass
