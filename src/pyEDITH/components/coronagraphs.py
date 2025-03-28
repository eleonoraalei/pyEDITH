from abc import ABC, abstractmethod
import numpy as np
from .. import parse_input
from astropy import units as u
from ..units import *
from scipy.interpolate import interp1d
from yippy import Coronagraph as yippycoro
from lod_unit import lod


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
    nchannels : int
        Number of channels.
    psf_trunc_ratio : np.ndarray
        PSF truncation ratio.
    minimum_IWA : float
        Minimum Inner Working Angle (lambd/D)
    maximum_OWA : float
        Maximum Outer Working Angle (lambd/D)
    coronagraph_throughput: np.ndarray
        Throughput for all coronagraph optics in the optical path
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
            "Istar": DIMENSIONLESS,  # contrast
            "noisefloor": DIMENSIONLESS,  # contrast
            "photap_frac": DIMENSIONLESS,  # throughput
            "omega_lod": LAMBDA_D**2,
            "skytrans": DIMENSIONLESS,  # throughput
            "pixscale": LAMBDA_D,
            "npix": int,
            "xcenter": PIXEL,
            "ycenter": PIXEL,
            "bandwidth": float,
            "angdiams": LAMBDA_D,
            "ndiams": int,
            "npsfratios": int,
            "nrolls": int,
            "nchannels": int,
            "psf_trunc_ratio": DIMENSIONLESS,
            "minimum_IWA": LAMBDA_D,
            "maximum_OWA": LAMBDA_D,
            "coronagraph_throughput": DIMENSIONLESS,
        }

        for arg, expected_type in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Coronagraph is missing attribute: {arg}")
            value = getattr(self, arg)
            if expected_type is int:
                if not isinstance(value, (int, np.integer)):
                    raise TypeError(f"Coronagraph attribute {arg} should be an integer")
            elif expected_type is float:
                if not isinstance(value, (float, np.floating)):
                    raise TypeError(f"Coronagraph attribute {arg} should be a float")
            elif isinstance(expected_type, u.UnitBase):
                if not isinstance(value, u.Quantity):
                    raise TypeError(f"Coronagraph attribute {arg} should be a Quantity")
                if not value.unit.is_equivalent(expected_type):
                    raise ValueError(
                        f"Coronagraph attribute {arg} has incorrect units. Expected {expected_type}, got {value.unit}"
                    )
            else:
                raise ValueError(f"Unexpected type specification for {arg}")


class ToyModelCoronagraph(Coronagraph):
    """
    A toy model coronagraph class that extends the base Coronagraph class.

    This class implements a simplified coronagraph model with basic functionality
    for generating secondary parameters and setting up coronagraph characteristics.
    """

    DEFAULT_CONFIG = {
        "pixscale": 0.25 * LAMBDA_D,  # lambd/D
        "angdiams": [0.0, 10.0] * LAMBDA_D,
        "minimum_IWA": 2.0 * LAMBDA_D,  # smallest WA to allow (lambda/D) (scalar)
        "maximum_OWA": 100.0 * LAMBDA_D,  # largest WA to allow (lambda/D) (scalar)
        "contrast": 1.05e-13
        * DIMENSIONLESS,  # contrast of coronagraph (uniform over dark hole and unitless)
        "noisefloor_factor": 0.03
        * DIMENSIONLESS,  #  1 sigma systematic noise floor expressed as a multiplicative factor to the contrast (unitless)
        "bandwidth": 0.2,  # fractional bandwidth of coronagraph (unitless)
        "Tcore": 0.2968371
        * DIMENSIONLESS,  # core throughput of coronagraph (uniform over dark hole, unitless, scalar)
        "TLyot": 0.65
        * DIMENSIONLESS,  # Lyot transmission of the coronagraph and the factor of 1.6 is just an estimate, used for skytrans}
        "nrolls": 1,  # number of rolls
        "nchannels": 2,  # number of channels
        "psf_trunc_ratio": [0.3] * DIMENSIONLESS,  # nlambda array
        "npsfratios": 1,  # NOTE UNUSED FOR NOW. Is it len(psf_trunc_ratio)?
        "coronagraph_throughput": 0.44
        * DIMENSIONLESS,  # Coronagraph throughput [made up from EAC1-ish]
    }

    def __init__(self, path=None, keyword=None):
        self.path = path
        self.keyword = keyword

    def load_configuration(self, parameters, mediator):
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

        # Load parameters, use defaults if not provided
        for key, default_value in self.DEFAULT_CONFIG.items():
            if key in parameters:
                # User provided a value
                user_value = parameters[key]
                if isinstance(default_value, u.Quantity):
                    # Ensure the user value has the same unit as the default
                    # TODO Implement conversion of units from the input file

                    if isinstance(user_value, u.Quantity):
                        setattr(self, key, user_value.to(default_value.unit))
                    else:
                        setattr(self, key, u.Quantity(user_value, default_value.unit))
                else:
                    # For non-Quantity values (like integers), use as is
                    setattr(self, key, user_value)
            else:
                # Use default value
                setattr(self, key, default_value)

        # Convert to numpy array when appropriate
        array_params = ["psf_trunc_ratio", "angdiams", "coronagraph_throughput"]
        for param in array_params:
            attr_value = getattr(self, param)
            if isinstance(attr_value, u.Quantity):
                # If it's already a Quantity, convert to numpy array while preserving units
                setattr(
                    self,
                    param,
                    u.Quantity(
                        np.array(attr_value.value, dtype=np.float64), attr_value.unit
                    ),
                )
            else:
                # If it's not a Quantity, convert to numpy array without units
                setattr(self, param, np.array(attr_value, dtype=np.float64))

        # Derived parameters
        self.npix = int(2 * 60 / self.pixscale)  # TODO check units here
        self.xcenter = self.npix / 2.0 * PIXEL
        self.ycenter = self.npix / 2.0 * PIXEL
        self.ndiams = len(self.angdiams)

        self.r = (
            generate_radii(self.npix, self.npix) * self.pixscale
        )  # create an array of circumstellar separations in units of lambd/D centered on star

        self.omega_lod = (
            np.full(
                (self.npix, self.npix, len(self.psf_trunc_ratio)),
                float(np.pi) * mediator.get_observation_parameter("photap_rad") ** 2,
            )
            * (mediator.get_observation_parameter("photap_rad").unit) ** 2
        )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

        self.skytrans = (
            np.full((self.npix, self.npix), self.TLyot) * self.TLyot.unit
        )  # skytrans at all separations

        self.photap_frac = (
            np.full((self.npix, self.npix, len(self.psf_trunc_ratio)), self.Tcore)
            * self.Tcore.unit
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
        self.Istar = (
            np.zeros((self.npix, self.npix, len(self.angdiams))) * DIMENSIONLESS
        )
        for z in range(len(self.angdiams)):
            self.Istar[:, :, z] = Istar

        self.noisefloor = (
            self.noisefloor_factor * self.contrast
        )  # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar
        n_floor = np.full((self.npix, self.npix), self.noisefloor * self.PSFpeak)
        self.noisefloor = (
            np.zeros((self.npix, self.npix, len(self.angdiams))) * DIMENSIONLESS
        )
        for z in range(len(self.angdiams)):
            self.noisefloor[:, :, z] = n_floor


class CoronagraphYIP(Coronagraph):
    """
    A coronagraph class that uses a Yield Input Package (YIP) for calculating
    coronagraph transmission, stellar intensity, and off-axis PSFs

    There are two components to this coronagraph simulation:
    1) the coronagraph optical throughput that comes from the HWO yaml files
    2) the coronagraph response that comes from the YIP

    """

    DEFAULT_CONFIG = {  # TODO: do we need this DEFAULT_CONFIG? everything we need is in the YIP or params
        "pixscale": 0.25,  # lambd/D
        "angdiams": [0.0, 10.0]
        * LAMBDA_D,  # NOTE: I don't fully understand this vs. the angdiam parameter elsewhere
        "angdiam": 0.01,  # NOTE: added this to match angdiam elsewhere in code
        "minimum_IWA": 2.0 * LAMBDA_D,  # smallest WA to allow (lambda/D) (scalar)
        "maximum_OWA": 100.0 * LAMBDA_D,  # largest WA to allow (lambda/D) (scalar)
        "contrast": 1.05e-13,  # contrast of coronagraph (uniform over dark hole and unitless)
        "noisefloor_factor": 0.03,  #  1 sigma systematic noise floor expressed as a multiplicative factor to the contrast (unitless)
        "bandwidth": 0.2,  # fractional bandwidth of coronagraph (unitless)
        "nrolls": 1,  # number of rolls
        "psf_trunc_ratio": [0.3] * DIMENSIONLESS,  # nlambda array
        "npsfratios": 1,  # NOTE UNUSED FOR NOW. Is it len(psf_trunc_ratio)?
        "coronagraph_throughput": None,
        "nchannels": 2,  # number of channels
    }

    def __init__(self, path=None, keyword=None):
        self.path = path
        self.keyword = keyword

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

        from eacy import load_instrument, load_telescope

        # ***** Set the bandwith *****
        setattr(
            self,
            "bandwidth",
            parameters.get("bandwidth", self.DEFAULT_CONFIG["bandwidth"]),
        )

        # ****** Update Default Config when necessary ******
        wavelength_range = [
            mediator.get_observation_parameter("lambd") * (1 - 0.5 * self.bandwidth),
            mediator.get_observation_parameter("lambd") * (1 + 0.5 * self.bandwidth),
        ]
        instrument_params = load_instrument("CI").__dict__

        instrument_params = parse_input.average_over_bandpass(
            instrument_params, wavelength_range
        )

        self.DEFAULT_CONFIG["coronagraph_throughput"] = instrument_params[
            "total_inst_refl"
        ]  # Optical throughput (nlambd array)

        # ***** Load parameters, use defaults if not provided *****
        for key, default_value in self.DEFAULT_CONFIG.items():
            setattr(self, key, parameters.get(key, default_value))

        # ***** Convert to numpy array when appropriate *****
        array_params = ["psf_trunc_ratio", "angdiam", "coronagraph_throughput"]
        for param in array_params:
            setattr(self, param, np.array(getattr(self, param), dtype=np.float64))

        self.psf_trunc_ratio *= DIMENSIONLESS
        self.minimum_IWA *= LAMBDA_D
        self.maximum_OWA *= LAMBDA_D
        self.coronagraph_throughput *= DIMENSIONLESS
        # ***** Load the YIP using yippy *****
        # TODO: this needs to be a parameter in the input file

        yippy_obj = yippycoro(self.path)
        # yippy_obj = yippycoro(parameters["YIP_dir"]) # TODO: this is the correct way to do it, but we need to add this to the input file

        # ***** Set all parameters that are defined in the YIP (unpack YIP metadata) *****
        self.pixscale = (
            yippy_obj.header.pixscale.value * LAMBDA_D
        )  # has units of lam/D / pix
        self.npix = yippy_obj.header.naxis1
        self.xcenter = yippy_obj.header.xcenter * PIXEL
        self.ycenter = yippy_obj.header.ycenter * PIXEL
        self.ndiams = len(self.angdiams)

        # Sky transmission map for extended sources
        self.skytrans = yippy_obj.sky_trans() * DIMENSIONLESS

        # Off axis throughput map: photap_frac
        self.r = (
            generate_radii(self.npix, self.npix) * self.pixscale
        )  # create an array of circumstellar separations in units of lambd/D centered on star
        self.omega_lod = (
            np.full(
                (self.npix, self.npix, len(self.psf_trunc_ratio)),
                float(np.pi) * mediator.get_observation_parameter("photap_rad") ** 2,
            )
            * DIMENSIONLESS
        )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

        self.photap_frac = (
            np.empty((self.npix, self.npix, len(self.psf_trunc_ratio))) * DIMENSIONLESS
        )
        sep_arr_lod, offax_tput_arr = yippy_obj.get_throughput_curve(
            aperture_radius_lod=self.psf_trunc_ratio, oversample=2, plot=False
        )  # changed from aperture_radius_lod=0.7
        offax_tput_func = interp1d(sep_arr_lod, offax_tput_arr)
        for i in range(self.npix):
            for j in range(self.npix):
                # get the off axis throughput at each separation
                self.photap_frac[i, j, 0] = offax_tput_func(self.r[i, j])

        # On-axis intensity map with a stellar diameter
        self.Istar = (
            np.zeros((self.npix, self.npix, 1)) * DIMENSIONLESS
        )  # TODO: should self.angdiam be an array??
        # for i_diam, angdiam in enumerate(self.angdiam):
        #     Istar = yippy_obj.stellar_intens(angdiam)
        #     self.Istar[:,:,i_diam] = Istar

        angdiam_arcsec = mediator.get_scene_parameter("angdiam_arcsec")
        lam = mediator.get_observation_parameter("lambd")
        telescope_params = load_telescope("EAC1").__dict__
        tele_diam = telescope_params["diam_circ"] * LENGTH
        # tele_diam = mediator.get_telescope_parameter("diameter") # NOTE: this doesn't work because the telescope object is not initialized yet

        angdiam_lod = arcsec_to_lambda_d(angdiam_arcsec, lam, tele_diam)

        Istar = yippy_obj.stellar_intens(angdiam_lod.value * lod)
        self.Istar[:, :, 0] = Istar

        # TODO: calculate noisefloor. Corey is working on this functionality for yippy that we can implement later.
        # by default, the noise floor is zero unless a second realization of stellar intensity is given in the YIP
        # currently I do not have a YIP with a second Istar realization, so I cannot develop this part yet.
        self.noisefloor = np.zeros((self.npix, self.npix, 1)) * DIMENSIONLESS

        # self.noisefloor = (
        #     self.noisefloor_factor * self.contrast
        # )  # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar
        # n_floor = np.full((self.npix, self.npix), self.noisefloor * self.PSFpeak)
        # n_floor =
        # self.noisefloor = np.zeros((self.npix, self.npix, len(self.angdiams)))
        # for z in range(len(self.angdiams)):
        #     self.noisefloor[:, :, z] = n_floor
