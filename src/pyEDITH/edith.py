import numpy as np
from pyEDITH.astrophysical_scene import AstrophysicalScene
from pyEDITH.observation import Observation


class Edith:
    """
    A class representing the E.D.I.T.H. (Exoplanet Detection Imaging and Throughput)
    simulation environment.

    This class handles the configuration, initialization, and calculations for
    exoplanet detection simulations.

    Parameters
    ----------
    scene : object
        An object representing the scene configuration.
    observation : object
        An object representing the observation parameters.

    Attributes
    ----------
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

    Methods
    -------
    load_default_parameters()
        Load default parameters for the simulation.
    """

    def __init__(self, scene: AstrophysicalScene, observation: Observation) -> None:
        """
        Initialize the Edith object. The following attributes are initialized:
        - tp : ndarray
            Exposure time of every planet (nmeananom x norbits x ntargs array).
        - exptime : ndarray
            Exposure time for each target and wavelength.
        - fullsnr : ndarray
            Signal-to-noise ratio for each target and wavelength.

        Parameters
        ----------
        scene : AstrophysicalScene
            An object representing the astrophysical scene configuration.
        observation : Observation
            An object representing the observation parameters.

        Returns
        -------
        None
        """

        # Initialize some arrays needed for outputs...
        self.tp = np.array([[[0.0]]], dtype=np.float64)  # exposure time of every planet
        # (nmeananom x norbits x ntargs array), used in c function
        # [NOTE: nmeananom = nphases in C code]
        self.exptime = np.full((scene.ntargs, observation.nlambd), 0.0)

        # only used for snr calculation
        self.fullsnr = np.full((scene.ntargs, observation.nlambd), 0.0)

    def load_default_parameters(self) -> None:
        """
        Load default parameters for the simulation.

        This method sets default values for various simulation parameters that
        typically don't need to be changed.

        Parameters
        ----------
        None

        Returns
        -------
        None
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
