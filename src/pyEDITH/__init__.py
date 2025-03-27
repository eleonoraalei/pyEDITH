# src/pyEDITH/__init__.py

# Import main classes
from .astrophysical_scene import AstrophysicalScene
from .observation import Observation
from .observatory import Observatory
from .observatory_builder import ObservatoryBuilder
from .components.coronagraphs import Coronagraph
from .components.telescopes import Telescope
from .components.detectors import Detector

# Import main functions
from .exposure_time_calculator import calculate_exposure_time_or_snr
from .components.coronagraphs import generate_radii
from . import parse_input
from .units import *

# Import CLI functions
from .cli import main, calculate_texp, calculate_snr

# Set a __all__ variable to control what gets imported with "from pyEDITH import *"
__all__ = [
    "AstrophysicalScene",
    "Observation",
    "Observatory",
    "ObservatoryBuilder",
    "Telescope",
    "Coronagraph",
    "Detector",
    "calculate_exposure_time_or_snr",
    "main",
    "calculate_texp",
    "calculate_snr",
    "parse_input",
    "generate_radii",
]

__version__ = "0.3.0"
