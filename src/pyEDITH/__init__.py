# Import main classes
from .astrophysical_scene import AstrophysicalScene
from .observation import Observation
from .instrument import Instrument
from .telescope import Telescope
from .coronagraph import Coronagraph
from .detector import Detector
from .edith import Edith


# Import main functions
from .exposure_time_calculator import calculate_exposure_time, calculate_signal_to_noise
from . import parse_input
from .coronagraph import generate_radii

# Import instrument-specific models
from .instruments.toymodel import ToyModel

# Import CLI functions
from .cli import main, calculate_texp, calculate_snr

# Set a __all__ variable to control what gets imported with "from pyEDITH import *"
__all__ = [
    "AstrophysicalScene",
    "Observation",
    "Instrument",
    "Telescope",
    "Coronagraph",
    "Detector",
    "Edith",
    "ToyModel",
    "calculate_exposure_time",
    "calculate_signal_to_noise",
    "main",
    "calculate_texp",
    "calculate_snr",
    "parse_input",
    "generate_radii",
]

__version__ = "0.2.0"
