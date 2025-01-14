import numpy as np


class Edith:
    """
    A class representing the E.D.I.T.H. (Exoplanet Detection Imaging and Throughput)
    simulation environment.

    This class handles the configuration, initialization, and calculations for
    exoplanet detection simulations.

    Methods:
    --------
    __init__()
    load_configuration(parameters)
    load_default_parameters()
    """

    def __init__(self, scene, observation) -> None:
        """
        Initialize the Edith object with default values for output arrays.
        """

        # Initialize some arrays needed for outputs...
        # self.besticoro = np.array([0])
        # self.bestilambd = np.array([0])
        # self.avgpsfomega = np.array([0.0])
        # self.minpsfomega = np.array([0.0])
        # self.maxpsfomega = np.array([0.0])
        # self.avgpsftruncratio = np.array([0.0])
        # self.minpsftruncratio = np.array([0.0])
        # self.maxpsftruncratio = np.array([0.0])
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
        typically don't need to be changed,
        including psf_trunc_ratio, td_limit, nooptimize, and optimize_phase.
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
