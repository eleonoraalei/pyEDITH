import numpy as np


class Observation:
    """
    A class representing an astronomical observation.

    This class encapsulates various parameters and methods related to
    astronomical observations, including target star properties, planet
    characteristics, observational settings, telescope specifications,
    instrument details, and detector parameters.

    Attributes:
    -----------
    lambd : np.ndarray
        Wavelength array (in microns).
    nlambd : int
        Number of wavelength points.
    SR : np.ndarray
        Spectral resolution array.
    SNR : np.ndarray
        Signal-to-noise ratio array.
    photap_rad : float
        Photometric aperture radius (in units of lambda/D).

    Methods:
    --------
    __init__() -> None
        Initialize the Observation object.
    load_configuration(parameters: dict) -> None
        Load configuration parameters for the simulation.
    """

    def __init__(self) -> None:
        """
        Initialize the Observation object with default values for output arrays.
        """
        pass  # there are no default values, TODO it should fail if not provided

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary of
        parameters that was read from the input file.

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
        # Observational parameters
        self.lambd = np.array(
            parameters["lambd"], dtype=np.float64
        )  # wavelength # nlambd array #unit: micron
        self.nlambd = len(self.lambd)
        self.SR = np.array(
            parameters["resolution"], dtype=np.float64
        )  # spec res # nlambd array
        self.SNR = np.array(
            parameters["snr"], dtype=np.float64
        )  # signal to noise # nlambd array
        self.photap_rad = parameters["photap_rad"]  # (lambd/D) # scalar
