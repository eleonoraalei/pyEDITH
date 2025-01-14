import numpy as np


def generate_radii(numx: int, numy: int = 0) -> np.array:
    """
    #### FORMER rgen.pro ####
    Generates a 2D distribution of radii.  The origin is assumed
    to be in the center of the matrix.  Radii are calculated
    assuming 1 pixel = 1 unit.  The number of pixels can be odd
    or even.

    numx = # of pixels in x direction
    numy = # of pixels in y direction
    x: the x coordinates
    y: the y coordinates
    """
    xo2 = int(numx / 2)  # if it is odd, gets the int (lower value)

    if numy == 0:
        numy = numx
    yo2 = int(numy / 2)

    xn1 = int(xo2)
    # xn2 = int(xo2-1)
    yn1 = int(yo2)
    # yn2 = int(yo2-1)

    oddx = 0
    oddy = 0
    if (xo2 + xo2) < numx:
        oddx = 1
        # xn2 = xo2
        xo2 += 1

    if (yo2 + yo2) < numy:
        oddy = 1
        # yn2 = yo2
        yo2 += 1

    xa = np.arange(0, xo2) + 0.5
    ya = np.arange(0, yo2) + 0.5
    if oddx == 1:
        xa -= 0.5
    if oddy == 1:
        ya -= 0.5

    xb = np.array([xa for i in np.arange(numy)])  # TODO check if type works

    yb = np.array([ya for i in np.arange(numx)]).T  # rotate by 90 deg CCW

    # FLIPPED BECAUSE IDL WORKS WITH COLUMNS X ROWS AND PYTHON THE OPPOSITE.
    # TODO consider changing this

    x = np.zeros((numy, numx))

    x[:, int(xn1) : int(numx)] = xb

    if oddx == 1:
        x[:, 0 : int(xn1) + 1] = -np.flip(xb, axis=1)
    else:
        x[:, 0 : int(xn1)] = -np.flip(xb, axis=1)

    y = np.zeros((numy, numx))
    y[int(yn1) : int(numy), :] = yb

    if oddy == 1:
        y[0 : int(yn1) + 1, :] = -np.flip(yb, axis=0)
    else:
        y[0 : int(yn1), :] = -np.flip(yb, axis=0)

    r = np.sqrt(x * x + y * y)
    return r


class Coronagraph:
    def __init__(self):
        # DEFAULT PARAMETERS FOR ANY CORONAGRAPH
        self.enable_coro = 0
        self.Istar = np.array([0.0, 0.0])
        self.noisefloor = np.array([0.0, 0.0])
        self.photap_frac = np.array([[0.0, 0.0]])
        self.omega_lod = np.array([0.0, 0.0])
        self.skytrans = np.array([0.0, 0.0])
        self.pixscale = 0.0
        self.npix = 0
        self.xcenter = 0.0
        self.ycenter = 0.0
        self.bw = 0.0  # called deltalambda but then assigned as bw
        self.angdiams = np.array([0.0, 0.0])
        self.ndiams = 0
        self.npsfratios = 0
        self.nrolls = 0
        self.psf_trunc_ratio = np.array([0.3], dtype=np.float64)  # array

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary of parameters that was read from the input file.

        Parameters:
        -----------
        parameters : dict
            A dictionary containing various simulation parameters including:
            - Target star parameters (ntargs, Lstar, dist, vmag, mag, angdiam_arcsec, nzodis, ra, dec)
            - Planet parameters (sp, deltamag, min_deltamag)
            - Observational parameters (lambd, SR, SNR, throughput, photap_rad)
            - Telescope & spacecraft parameters (D, toverhead_fixed, toverhead_multi)
            - Instrument parameters (IWA, OWA, contrast, noisefloor_factor, bandwidth, core_throughput, Lyot_transmission)
            - Detector parameters (npix_multiplier, dark_current, read_noise, read_time, cic)
            - Coronagraph parameters (coro_type, nrolls)
        """

        # -------- INPUTS ---------
        # Instrument parameters
        self.IWA = parameters["IWA"]  # smallest WA to allow (lambd/D) # scalar
        self.OWA = parameters["OWA"]  # largest WA to allow (lambd/D) # scalar
        self.contrast = parameters[
            "contrast"
        ]  # contrast of coronagraph (uniform over dark hole and unitless) # scalar
        self.noisefloor = (
            parameters["noisefloor_factor"] * self.contrast
        )  # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar
        self.bandwidth = parameters[
            "bandwidth"
        ]  # fractional bandwidth of coronagraph (unitless)
        self.Tcore = parameters[
            "core_throughput"
        ]  # core throughput of coronagraph (uniform over dark hole, unitless) # scalar
        self.TLyot = parameters[
            "Lyot_transmission"
        ]  # 1.6*Tcore # Lyot transmission of the coronagraph and the factor of 1.6 is just an estimate, used for skytrans

        # Coronagraph parameters
        self.coro_type = parameters["coro_type"]
        self.nrolls = parameters["nrolls"]

    def generate_secondary_parameters(self):
        """
        This function generates an error in the super-class, since it is specific of every subclass .
        It must be defined in each subclass. If not, it will fail here trying to read the non-implemented general one.
        """
        raise NotImplementedError


class ToyModel(Coronagraph):
    def generate_secondary_parameters(self, observation):

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
        # j = np.where(r lt self.IWA or r gt self.OWA)                        # find separations interior to IWA or exterior to OWA
        # if j[0] ne -1 then photap_frac1[j] = 0.0

        self.photap_frac[self.r < self.IWA] = 0.0  # index 0 is the
        self.photap_frac[self.r > self.OWA] = 0.0

        # put in the right dimensions (3d arrays), but third dimension is 1 (number of psf_trunc_ratio)
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
