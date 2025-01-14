import numpy as np


class Detector:
    """
    A class representing a detector for astronomical observations.

    This class manages detector-specific parameters and configurations
    used in astronomical simulations and observations.

    Attributes
    ----------
    det_pixscale_mas : float
        Detector pixel scale in milliarcseconds.
    det_npix_multiplier : ndarray
        Number of detector pixels per image plane "pixel".
    det_DC : ndarray
        Dark current in counts per pixel per second.
    det_RN : ndarray
        Read noise in counts per pixel per read.
    det_tread : ndarray
        Read time in seconds.
    det_CIC : ndarray
        Clock-induced charge in counts per pixel per photon count.

    Methods
    -------
    load_configuration(parameters)
        Load configuration parameters for the simulation from a dictionary.
    """

    def __init__(self) -> None:
        """
        Initialize a Detector object.

        This constructor initializes an empty Detector object. The detector's
        attributes are set using the `load_configuration` method.

        Returns
        -------
        None
        """
        pass

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the Detector object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        Returns
        -------
        None

        """

        # -------- INPUTS ---------

        # Detector parameters

        # Detector pixel scale (mas) (scalar)
        self.det_pixscale_mas = (
            0.5
            * (0.5e-6 / parameters["diameter"])
            * (180.0 / np.double(np.pi) * 60.0 * 60.0 * 1000.0)
        )

        # Number of detector pixels per image plane "pixel"
        # (nlambd array, 1 for detections or spectra w/ ERD,
        # ~6*(140/SR) for spectra with IFS)
        self.det_npix_multiplier = np.array(
            parameters["npix_multiplier"], dtype=np.float64
        )

        # Dark current (counts pix^-1 s^-1) (nlambd array)
        self.det_DC = np.array(parameters["dark_current"], dtype=np.float64)

        # Read noise (counts pix^-1 read^-1) (nlambd array)
        self.det_RN = np.array(parameters["read_noise"], dtype=np.float64)

        # Read time (s) (nlambd array)
        self.det_tread = np.array(parameters["read_time"], dtype=np.float64)

        # Clock-induced charge (counts pix^-1 photon_count^-1) (nlambd array)
        self.det_CIC = np.array(parameters["cic"], dtype=np.float64)
