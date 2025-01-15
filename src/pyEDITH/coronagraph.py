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


class Coronagraph:
    """
    A base class representing a generic coronagraph.

    This class defines the basic structure and methods common to all coronagraphs.
    Specific coronagraph models should inherit from this class and implement
    their own `generate_secondary_parameters` method.

    Attributes:
    -----------
    enable_coro : int
        Flag to enable or disable the coronagraph (0: disabled, 1: enabled).
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
    bw : float
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
    """

    def __init__(self) -> None:
        """
        Initialize the Coronagraph object with default parameter values.
        """

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
        Load configuration parameters for the simulation from a dictionary.

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

    def generate_secondary_parameters(self) -> None:
        """
        Generate secondary parameters for the coronagraph.

        This method should be implemented by subclasses to generate
        coronagraph-specific secondary parameters.

        Raises:
        -------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError
