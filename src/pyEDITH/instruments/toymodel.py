from pyEDITH import Instrument, Coronagraph, Telescope, Detector, Observation
from pyEDITH import generate_radii
import numpy as np


class ToyModelCoronagraph(Coronagraph):
    """
    A toy model coronagraph class that extends the base Coronagraph class.

    This class implements a simplified coronagraph model with basic functionality
    for generating secondary parameters and setting up coronagraph characteristics.
    """

    def generate_secondary_parameters(self, observation: Observation) -> None:
        """
        Generate secondary parameters for the toy model coronagraph.

        This method initializes various coronagraph parameters based on
        a simplified model. It sets up the coronagraph's field of view,
        pixel scale, and performance characteristics.

        Parameters:
        -----------
        observation : Observation
            An object containing observational parameters.

        Returns:
        --------
        None
        """

        self.type = "Toy Model"

        # "Create" coronagraph1, the only coronagraph used in this code
        self.enable_coro = 1
        self.pixscale = 0.25  # lambd/D
        self.npix = int(2 * 60 / self.pixscale)
        self.xcenter = self.npix / 2.0
        self.ycenter = self.npix / 2.0
        self.angdiams = np.array([0.0, 10.0])
        self.ndiams = len(self.angdiams)
        self.npsfratios = 1
        self.r = (
            generate_radii(self.npix, self.npix) * self.pixscale
        )  # create an array of circumstellar separations in units of lambd/D centered on star
        self.omega_lod = np.full(
            (self.npix, self.npix, 1), float(np.pi) * observation.photap_rad**2
        )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))
        self.skytrans = np.full(
            (self.npix, self.npix), self.TLyot
        )  # skytrans at all separations

        self.photap_frac = np.full(
            (self.npix, self.npix, 1), self.Tcore
        )  # core throughput at all separations (npix,npix,len(psftruncratio))
        # TODO check change)
        # j = np.where(r lt self.IWA or r gt self.OWA)
        # find separations interior to IWA or exterior to OWA
        # if j[0] ne -1 then photap_frac1[j] = 0.0

        self.photap_frac[self.r < self.IWA] = 0.0  # index 0 is the
        self.photap_frac[self.r > self.OWA] = 0.0

        # put in the right dimensions (3d arrays), but third dimension
        # is 1 (number of psf_trunc_ratio)
        # self.omega_lod = np.array([self.omega_lod])
        # self.photap_frac = np.array([self.photap_frac])

        self.PSFpeak = (
            0.025 * self.TLyot
        )  # this is an approximation based on PAPLC results
        Istar = np.full((self.npix, self.npix), self.contrast * self.PSFpeak)

        self.Istar = np.zeros((self.npix, self.npix, len(self.angdiams)))
        for z in range(len(self.angdiams)):
            self.Istar[:, :, z] = Istar

        n_floor = np.full((self.npix, self.npix), self.noisefloor * self.PSFpeak)
        self.noisefloor = np.zeros((self.npix, self.npix, len(self.angdiams)))
        for z in range(len(self.angdiams)):
            self.noisefloor[:, :, z] = n_floor


class ToyModelTelescope(Telescope):
    """
    A toy model telescope class that extends the base Telescope class.

    This class represents a simplified telescope model for use in simulations.
    """

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the toy model telescope.

        Parameters
        ----------
        parameters : dict
            A dictionary containing configuration parameters for the telescope.

        Returns
        -------
        None
        """
        super().load_configuration(parameters)


class ToyModelDetector(Detector):
    """
    A toy model detector class that extends the base Detector class.

    This class represents a simplified detector model for use in simulations.
    """

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the toy model detector.

        Parameters
        ----------
        parameters : dict
            A dictionary containing configuration parameters for the detector.

        Returns
        -------
        None
        """
        super().load_configuration(parameters)


class ToyModel(Instrument):
    """
    A toy model instrument class that combines coronagraph, telescope, and detector.

    This class represents a simplified instrument model that includes a coronagraph,
    telescope, and detector for use in simulations.
    """

    def __init__(self):
        """
        Initialize the ToyModel instrument.

        Creates instances of ToyModelCoronagraph, ToyModelTelescope, and ToyModelDetector.
        """
        self.coronagraph = ToyModelCoronagraph()
        self.telescope = ToyModelTelescope()
        self.detector = ToyModelDetector()

    def initialize(self, parameters: dict):
        """
        Initialize the ToyModel instrument with given parameters.

        This method loads the configuration for the coronagraph, telescope, and detector
        components of the instrument.

        Parameters
        ----------
        parameters : dict
            A dictionary containing configuration parameters for the instrument components.

        Returns
        -------
        None
        """
        self.coronagraph.load_configuration(parameters)
        self.telescope.load_configuration(parameters)
        self.detector.load_configuration(parameters)
