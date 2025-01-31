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
    tp : ndarray
        Exposure time of every planet (nmeananom x norbits x ntargs array).
    exptime : ndarray
        Exposure time for each target and wavelength.
    fullsnr : ndarray
        Signal-to-noise ratio for each target and wavelength.
    td_limit : float
        Limit placed on exposure times.
    nooptimize : int
        Flag to disable exposure time optimization.
    optimize_phase : int
        Flag to optimize the phase of the planet (not functional in this code).
    ntot : int
        Total number of something (purpose not specified).
    nmeananom : int
        Number of mean anomalies.
    norbits : int
        Number of orbits.


    """

    def __init__(self) -> None:
        """
        Initialize the default parameters of the Observation class.
        """
        # Misc parameters that probably don't need to be changed
        self.td_limit = 1e20  # limit placed on exposure times # scalar
        self.nooptimize = (
            0  # do not attempt to optimize exposure times for this code # scalar
        )
        self.optimize_phase = (
            0  # optimize the phase of the planet (does not work in this code) # scalar
        )

        # Some things specific to this code
        # self.ntargs = 1 #specified in the reading of the input
        # self.nlambd = 1 #specified in the reading of the input
        self.ntot = 1
        self.nmeananom = 1
        self.norbits = 1

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

    def set_output_arrays(self):
        """
        Initialize output arrays:

        - tp : ndarray
            Exposure time of every planet (nmeananom x norbits x ntargs array).
        - exptime : ndarray
            Exposure time for each target and wavelength.
        - fullsnr : ndarray
            Signal-to-noise ratio for each target and wavelength.
        """
        # Initialize some arrays needed for outputs...
        self.tp = np.array([[[0.0]]], dtype=np.float64)  # exposure time of every planet
        # (nmeananom x norbits x ntargs array), used in c function
        # [NOTE: nmeananom = nphases in C code]
        # NOTE: ntargs fixed to 1.
        self.exptime = np.full((1, self.nlambd), 0.0)

        # only used for snr calculation
        self.fullsnr = np.full((1, self.nlambd), 0.0)
