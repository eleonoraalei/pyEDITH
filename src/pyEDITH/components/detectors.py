from abc import ABC, abstractmethod
import numpy as np
from .. import parse_input


class Detector(ABC):
    """
    A class representing a detector for astronomical observations.

    This class manages detector-specific parameters and configurations
    used in astronomical simulations and observations.

    Attributes
    ----------
    pixscale_mas : float
        Detector pixel scale in milliarcseconds.
    npix_multiplier : ndarray
        Number of detector pixels per image plane "pixel".
    DC : ndarray
        Dark current in counts per pixel per second.
    RN : ndarray
        Read noise in counts per pixel per read.
    tread : ndarray
        Read time in seconds.
    CIC : ndarray
        Clock-induced charge in counts per pixel per photon count.
    dQE: ndarray
        Quantum efficiency of detector
    QE: ndarray
        Effective QE due to degradation, cosmic ray effects, readout inefficiencies
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
            "pixscale_mas": (float, np.floating),
            "npix_multiplier": np.ndarray,
            "DC": np.ndarray,
            "RN": np.ndarray,
            "tread": np.ndarray,
            "CIC": np.ndarray,
            "QE": np.ndarray,
            "dQE": np.ndarray,
        }

        for arg, expected_type in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Detector is missing attribute: {arg}")
            if not isinstance(getattr(self, arg), expected_type):
                raise TypeError(
                    f"Detector attribute {arg} should be of type {expected_type}, but is {type(getattr(self, arg))}"
                )


class ToyModelDetector(Detector):
    """
    A toy model detector class that extends the base Detector class.

    This class represents a simplified detector model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "pixscale_mas": None,  # Detector pixel scale in milliarcseconds.
        "npix_multiplier": [1],  # Number of detector pixels per image plane "pixel".
        "DC": [3e-5],  # Dark current (counts pix^-1 s^-1, nlambd array)
        "RN": [0.0],  # Read noise (counts pix^-1 read^-1, nlambd array)
        "tread": [1000],  # Read time (s, nlambd array)
        "CIC": [
            1.3e-3
        ],  # Clock-induced charge (counts pix^-1 photon_count^-1, nlambd array)
        "QE": [0.9],  # Quantum efficiency of detector
        "dQE": [
            0.75
        ],  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    }

    def load_configuration(self, parameters, mediator) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the Detector object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        mediator: ObservatoryMediator
        Returns
        -------
        None

        """

        # Calculate default detector pixel scale based on telescope
        self.DEFAULT_CONFIG["pixscale_mas"] = (
            0.5
            * (0.5e-6 / mediator.get_telescope_parameter("diameter"))
            * (180.0 / np.double(np.pi) * 60.0 * 60.0 * 1000.0)
        )

        # Load parameters, use defaults if not provided
        for key, default_value in self.DEFAULT_CONFIG.items():
            setattr(self, key, parameters.get(key, default_value))

        # Convert to numpy array when appropriate
        array_params = [
            "npix_multiplier",
            "DC",
            "RN",
            "tread",
            "CIC",
            "QE",
            "dQE",
        ]
        for param in array_params:
            setattr(self, param, np.array(getattr(self, param), dtype=np.float64))


class EAC1Detector(Detector):
    """
    A toy model detector class that extends the base Detector class.

    This class represents a simplified detector model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "pixscale_mas": None,  # Detector pixel scale in milliarcseconds.
        "npix_multiplier": [1],  # Number of detector pixels per image plane "pixel".
        "DC": None,  # Dark current (counts pix^-1 s^-1, nlambd array)
        "RN": None,  # Read noise (counts pix^-1 read^-1, nlambd array)
        "tread": [1000],  # Read time (s, nlambd array) # TO ADD TO YAML
        "CIC": [
            0
        ],  # Clock-induced charge (counts pix^-1 photon_count^-1, nlambd array) # TO ADD TO YAML
        "QE": None,  # Quantum efficiency of detector
        "dQE": None,  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    }

    def load_configuration(self, parameters, mediator) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the Detector object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        mediator: ObservatoryMediator
        Returns
        -------
        None

        """
        from eacy import load_detector

        # ****** Update Default Config when necessary ******
        wavelength_range = [
            mediator.get_observation_parameter("lambd")
            * (1 - 0.5 * mediator.get_coronagraph_parameter("bandwidth")),
            mediator.get_observation_parameter("lambd")
            * (1 + 0.5 * mediator.get_coronagraph_parameter("bandwidth")),
        ]
        detector_params = load_detector("IMAGER").__dict__
        detector_params = parse_input.average_over_bandpass(
            detector_params, wavelength_range
        )

        self.DEFAULT_CONFIG["DC"] = [
            (
                detector_params["dc_vis"]
                if mediator.get_observation_parameter("lambd") < 1
                else detector_params["dc_nir"]
            )
        ]
        # Dark current (counts pix^-1 s^-1, nlambd array)
        self.DEFAULT_CONFIG["RN"] = [
            (
                detector_params["rn_vis"]
                if mediator.get_observation_parameter("lambd") < 1
                else detector_params["rn_nir"]
            )
        ]
        self.DEFAULT_CONFIG["QE"] = [
            (
                detector_params["qe_vis"]
                if mediator.get_observation_parameter("lambd") < 1
                else detector_params["qe_nir"]
            )
        ]
        self.DEFAULT_CONFIG["dQE"] = [
            0.75
        ]  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies ## TO ADD TO YAML

        # Calculate default detector pixel scale based on telescope
        self.DEFAULT_CONFIG["pixscale_mas"] = (
            0.5
            * (0.5e-6 / mediator.get_telescope_parameter("diameter"))
            * (180.0 / np.double(np.pi) * 60.0 * 60.0 * 1000.0)
        )

        # ***** Load parameters, use defaults if not provided *****
        for key, default_value in self.DEFAULT_CONFIG.items():
            setattr(self, key, parameters.get(key, default_value))

        # ***** Convert to numpy array when appropriate *****
        array_params = [
            "npix_multiplier",
            "DC",
            "RN",
            "tread",
            "CIC",
            "QE",
            "dQE",
        ]
        for param in array_params:
            setattr(self, param, np.array(getattr(self, param), dtype=np.float64))
