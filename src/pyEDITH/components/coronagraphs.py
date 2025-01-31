from abc import ABC, abstractmethod
import numpy as np


def generate_radii(numx: int, numy: int = 0) -> np.ndarray:
    """
    Generate a 2D distribution of radii from the center of a matrix.

    This function creates a 2D numpy array representing the radial distance
    from the center for each element. The origin is assumed to be in the
    center of the matrix. Radii are calculated assuming 1 pixel = 1 unit.
    The number of pixels can be odd or even.

    Parameters:
    -----------
    numx : int
        Number of pixels in the x direction.
    numy : int, optional
        Number of pixels in the y direction. If 0 (default), it is set equal to numx.

    Returns:
    --------
    np.ndarray
        A 2D numpy array of shape (numy, numx) containing the radial distances
        from the center for each pixel.

    Notes:
    ------
    - The function handles both odd and even dimensions.
    - Formerly  in rgen.pro
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


class Coronagraph(ABC):
    """
    A base class representing a generic coronagraph.

    This class defines the basic structure and methods common to all coronagraphs.
    Specific coronagraph models should inherit from this class and implement
    their own `generate_secondary_parameters` method.

    Attributes:
    -----------
    Istar : np.ndarray
        Star intensity distribution.
    noisefloor : np.ndarray
        Noise floor of the coronagraph.
    photap_frac : np.ndarray
        Photometric aperture fraction.
    omega_lod : np.ndarray
        Solid angle of the photometric aperture in λ/D units.
    skytrans : np.ndarray
        Sky transmission.
    pixscale : float
        Pixel scale in λ/D units.
    npix : int
        Number of pixels in the image.
    xcenter : float
        X-coordinate of the image center.
    ycenter : float
        Y-coordinate of the image center.
    bandwidth : float
        Bandwidth of the coronagraph.
    angdiams : np.ndarray
        Angular diameters of the target objects.
    ndiams : int
        Number of angular diameters.
    npsfratios : int
        Number of PSF ratios.
    nrolls : int
        Number of roll angles.
    psf_trunc_ratio : np.ndarray
        PSF truncation ratio.
    minimum_IWA : float
        Minimum Inner Working Angle (lambd/D)
    maximum_OWA : float
        Maximum Outer Working Angle (lambd/D)
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
            "Istar": np.ndarray,
            "noisefloor": np.ndarray,
            "photap_frac": np.ndarray,
            "omega_lod": np.ndarray,
            "skytrans": np.ndarray,
            "pixscale": (float, np.floating),
            "npix": (int, np.integer),
            "xcenter": (float, np.floating),
            "ycenter": (float, np.floating),
            "bandwidth": (float, np.floating),
            "angdiams": np.ndarray,
            "ndiams": (int, np.integer),
            "npsfratios": (int, np.integer),
            "nrolls": (int, np.integer),
            "psf_trunc_ratio": np.ndarray,
            "minimum_IWA": (float, np.floating),
            "maximum_OWA": (float, np.floating),
            "bandwidth": (float, np.floating),
        }

        for arg, expected_type in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Coronagraph is missing attribute: {arg}")
            if not isinstance(getattr(self, arg), expected_type):
                raise TypeError(
                    f"Coronagraph attribute {arg} should be of type {expected_type}, but is {type(getattr(self, arg))}"
                )


class ToyModelCoronagraph(Coronagraph):
    """
    A toy model coronagraph class that extends the base Coronagraph class.

    This class implements a simplified coronagraph model with basic functionality
    for generating secondary parameters and setting up coronagraph characteristics.
    """

    DEFAULT_CONFIG = {
        "pixscale": 0.25,  # lambd/D
        "angdiams": [0.0, 10.0],
        "minimum_IWA": 2.0,  # smallest WA to allow (lambda/D) (scalar)
        "maximum_OWA": 100.0,  # largest WA to allow (lambda/D) (scalar)
        "contrast": 1.05e-13,  # contrast of coronagraph (uniform over dark hole and unitless)
        "noisefloor_factor": 0.03,  #  1 sigma systematic noise floor expressed as a multiplicative factor to the contrast (unitless)
        "bandwidth": 0.2,  # fractional bandwidth of coronagraph (unitless)
        "Tcore": 0.2968371,  # core throughput of coronagraph (uniform over dark hole, unitless, scalar)
        "TLyot": 0.65,  # Lyot transmission of the coronagraph and the factor of 1.6 is just an estimate, used for skytrans}
        "nrolls": 1,  # number of rolls
        "psf_trunc_ratio": [0.3],  # nlambda array
        "npsfratios": 1,  # NOTE UNUSED FOR NOW. Is it len(psf_trunc_ratio)?
    }

    def load_configuration(self, parameters, mediator):
        """
        Load configuration parameters for the simulation from a dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        observation: Observation
            An instance of the observation class.
        ALL OF THE CLASSES
        Returns
        -------
        None
        """

        # Load parameters, use defaults if not provided
        for key, default_value in self.DEFAULT_CONFIG.items():
            setattr(self, key, parameters.get(key, default_value))

        # Convert to numpy array when appropriate
        array_params = ["psf_trunc_ratio", "angdiams"]
        for param in array_params:
            setattr(self, param, np.array(getattr(self, param), dtype=np.float64))

        # Derived parameters
        self.npix = int(2 * 60 / self.pixscale)
        self.xcenter = self.npix / 2.0
        self.ycenter = self.npix / 2.0
        self.ndiams = len(self.angdiams)

        self.r = (
            generate_radii(self.npix, self.npix) * self.pixscale
        )  # create an array of circumstellar separations in units of lambd/D centered on star

        self.omega_lod = np.full(
            (self.npix, self.npix, len(self.psf_trunc_ratio)),
            float(np.pi) * mediator.get_observation_parameter("photap_rad") ** 2,
        )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

        self.skytrans = np.full(
            (self.npix, self.npix), self.TLyot
        )  # skytrans at all separations

        self.photap_frac = np.full(
            (self.npix, self.npix, len(self.psf_trunc_ratio)), self.Tcore
        )  # core throughput at all separations (npix,npix,len(psftruncratio))
        # TODO check change)
        # j = np.where(r lt self.IWA or r gt self.OWA)
        # find separations interior to IWA or exterior to OWA
        # if j[0] ne -1 then photap_frac1[j] = 0.0

        self.photap_frac[self.r < self.minimum_IWA] = 0.0  # index 0 is the
        self.photap_frac[self.r > self.maximum_OWA] = 0.0

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

        self.noisefloor = (
            self.noisefloor_factor * self.contrast
        )  # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar
        n_floor = np.full((self.npix, self.npix), self.noisefloor * self.PSFpeak)
        self.noisefloor = np.zeros((self.npix, self.npix, len(self.angdiams)))
        for z in range(len(self.angdiams)):
            self.noisefloor[:, :, z] = n_floor
