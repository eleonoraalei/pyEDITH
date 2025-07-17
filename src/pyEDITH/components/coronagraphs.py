from abc import ABC, abstractmethod
import numpy as np
from .. import utils
from astropy import units as u
from ..units import *
from scipy.interpolate import interp1d
from yippy import Coronagraph as yippycoro
from lod_unit import lod
from scipy.ndimage import convolve, zoom, rotate


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

    xb = np.array([xa for i in np.arange(numy)])

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
    Right now, there are two coronagraph sub-classes:
    -> ToyModelCoronagraph
        This is a simplistic coronagraph setup where the user can specify all
        coronagraph parameters. Use this for testing coronagraph parameters
        not defined by a Yield Input Package (files containing models of
        realistic coronagraph responses).

    -> CoronagraphYIP
        This is a coronagraph setup that is defined by a Yield Input Package (YIP),
        which contains models of realistic coronagraph responses. User this for
        testing specific coronagraph cases. Requires a path to a YIP.

    Attributes:
    -----------
    Istar : np.ndarray
        Star intensity distribution.
    noisefloor : np.ndarray
        Noise floor of the coronagraph.
    photometric_aperture_throughput : np.ndarray
        Photometric aperture throughput.
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
    stellar_angular_diameter : np.ndarray
        Angular diameters of the target objects.
    npsfratios : int
        Number of PSF ratios.
    nrolls : int
        Number of roll angles.
    nchannels : int
        Number of channels.
    minimum_IWA : float
        Minimum Inner Working Angle (lambd/D)
    maximum_OWA : float
        Maximum Outer Working Angle (lambd/D)
    coronagraph_optical_throughput: np.ndarray
        Throughput for all coronagraph optics in the optical path
    """

    @abstractmethod
    def load_configuration(self):  # pragma: no cover
        pass

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "Istar": DIMENSIONLESS,  # contrast
            "noisefloor": DIMENSIONLESS,  # contrast
            "photometric_aperture_throughput": DIMENSIONLESS,  # throughput
            "omega_lod": LAMBDA_D**2,
            "skytrans": DIMENSIONLESS,  # throughput
            "pixscale": LAMBDA_D,
            "npix": int,
            "xcenter": PIXEL,
            "ycenter": PIXEL,
            "bandwidth": float,
            "npsfratios": int,
            "nrolls": int,
            "nchannels": int,
            "minimum_IWA": LAMBDA_D,
            "maximum_OWA": LAMBDA_D,
            "coronagraph_optical_throughput": DIMENSIONLESS,
            "coronagraph_spectral_resolution": DIMENSIONLESS,
        }

        utils.validate_attributes(self, expected_args)


class ToyModelCoronagraph(Coronagraph):
    """
    A toy model coronagraph class that extends the base Coronagraph class.

    This class implements a simplified coronagraph model with basic functionality
    for generating secondary parameters and setting up coronagraph characteristics.
    """

    DEFAULT_CONFIG = {
        "pixscale": 0.25 * LAMBDA_D,
        "minimum_IWA": 2.0 * LAMBDA_D,  # smallest WA to allow (lambda/D) (scalar)
        "maximum_OWA": 100.0 * LAMBDA_D,  # largest WA to allow (lambda/D) (scalar)
        "contrast": 1.05e-13
        * DIMENSIONLESS,  # noise floor contrast of coronagraph (uniform over dark hole and unitless)
        "noisefloor_factor": 0.03
        * DIMENSIONLESS,  #  1 sigma systematic noise floor expressed as a multiplicative factor to the contrast (unitless)
        "bandwidth": 0.2,  # fractional bandwidth of coronagraph (unitless)
        "Tcore": 0.2968371
        * DIMENSIONLESS,  # core throughput of coronagraph (uniform over dark hole, unitless, scalar)
        "TLyot": 0.65
        * DIMENSIONLESS,  # Lyot transmission of the coronagraph and the factor of 1.6 is just an estimate, used for skytrans
        "nrolls": 1,  # number of rolls
        "nchannels": 2,  # number of channels
        "coronagraph_optical_throughput": [0.44]
        * DIMENSIONLESS,  # Coronagraph throughput [made up from EAC1-ish]
        "coronagraph_spectral_resolution": 1
        * DIMENSIONLESS,  # Set to default. It is used to limit the bandwidth if the coronagraph has a specific spectral window.
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

        utils.fill_parameters(self, parameters, self.DEFAULT_CONFIG)

        # Convert to numpy array when appropriate
        array_params = ["coronagraph_optical_throughput"]
        utils.convert_to_numpy_array(self, array_params)

        # Get PSF Truncation ratio from Observation
        # self.psf_trunc_ratio = mediator.get_observation_parameter("psf_trunc_ratio")

        # Derived parameters
        self.npsfratios = 1
        self.npix = int(2 * 60 / self.pixscale)
        self.xcenter = self.npix / 2.0 * PIXEL
        self.ycenter = self.npix / 2.0 * PIXEL

        self.r = (
            generate_radii(self.npix, self.npix) * self.pixscale
        )  # create an array of circumstellar separations in units of lambd/D centered on star

        self.omega_lod = (
            np.full(
                (self.npix, self.npix, self.npsfratios),
                float(np.pi)
                * mediator.get_observation_parameter("photometric_aperture_radius")
                ** 2,
            )
            * (mediator.get_observation_parameter("photometric_aperture_radius").unit)
            ** 2
        )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

        self.skytrans = (
            np.full((self.npix, self.npix), self.TLyot) * self.TLyot.unit
        )  # skytrans at all separations

        self.photometric_aperture_throughput = (
            np.full((self.npix, self.npix, self.npsfratios), self.Tcore)
            * self.Tcore.unit
        )  # core throughput at all separations (npix,npix,len(psftruncratio))
        # j = np.where(r lt self.IWA or r gt self.OWA)
        # find separations interior to IWA or exterior to OWA
        # if j[0] ne -1 then photometric_aperture_throughput1[j] = 0.0

        self.photometric_aperture_throughput[self.r < self.minimum_IWA] = (
            0.0  # index 0 is the
        )
        self.photometric_aperture_throughput[self.r > self.maximum_OWA] = 0.0

        # put in the right dimensions (3d arrays), but third dimension
        # is 1 (number of psf_trunc_ratio)
        # self.omega_lod = np.array([self.omega_lod])
        # self.photometric_aperture_throughput = np.array([self.photometric_aperture_throughput])

        self.PSFpeak = (
            0.025 * self.TLyot
        )  # this is an approximation based on PAPLC results

        """
        OLD VERSION
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
        """
        self.Istar = (
            np.full((self.npix, self.npix), self.contrast * self.PSFpeak)
            * DIMENSIONLESS
        )

        if "noisefloor_factor" not in parameters.keys():
            if "noisefloor_PPF" in parameters.keys():
                print(
                    "Noisefloor_PPF mode not implemented in ToyModel coronagraph. Please use the noisefloor_factor method."
                )
            print(
                "WARNING: noisefloor_factor value not provided. Using the default value: "
                + str(self.DEFAULT_CONFIG["noisefloor_factor"])
            )

        print(
            "Calculating noisefloor by multiplying noisefloor_factor="
            + str(self.noisefloor_factor)
            + ", contrast="
            + str(self.contrast)
            + ", PSFpeak="
            + str(self.PSFpeak)
        )
        # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) * PSF peak # scalar
        self.noisefloor = (
            np.full(
                (self.npix, self.npix),
                self.noisefloor_factor * self.contrast * self.PSFpeak,
            )
            * DIMENSIONLESS
        )


class CoronagraphYIP(Coronagraph):
    """
    A coronagraph class that uses a Yield Input Package (YIP) for calculating
    coronagraph transmission, stellar intensity, and off-axis PSFs

    There are two components to this coronagraph simulation:
    1) the coronagraph optical throughput that comes from the HWO yaml files
    2) the coronagraph response that comes from the YIP

    """

    DEFAULT_CONFIG = {
        "minimum_IWA": 2.0 * LAMBDA_D,  # smallest WA to allow (lambda/D) (scalar)
        "maximum_OWA": 100.0 * LAMBDA_D,  # largest WA to allow (lambda/D) (scalar)
        # "contrast": 1.05e-13,  #  noise floor contrast of coronagraph (uniform over dark hole and unitless)
        "noisefloor_PPF": 30.0,  # 30.0 #  divide Istar by this to get the noise floor (unitless)
        "bandwidth": 0.2,  # fractional bandwidth of coronagraph (unitless)
        "nrolls": 1,  # number of rolls
        "Tcore": 0.2968371
        * DIMENSIONLESS,  # core throughput within off-axis PSF (only used with photometric_aperture_radius method of calculating Omega)
        "coronagraph_optical_throughput": None,
        "coronagraph_spectral_resolution": 1
        * DIMENSIONLESS,  # Set to default. It is used to limit the bandwidth if the coronagraph has a specific spectral window.
        "nchannels": 2,  # number of channels
        # "TLyot": 0.65
        # * DIMENSIONLESS,  # Lyot transmission of the coronagraph and the factor of 1.6 is just an estimate, used for skytrans}
        "az_avg": True,  # azimuthally average the contrast maps and noise floor if True
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

        # ***** Load the YAML using EACy *****
        instrument_params = load_instrument("CI").__dict__

        # averaging over bandpass is only required for imaging mode.
        if parameters["observing_mode"] == "IMAGER":
            wavelength_range = [
                mediator.get_observation_parameter("wavelength")
                * (1 - 0.5 * self.bandwidth),
                mediator.get_observation_parameter("wavelength")
                * (1 + 0.5 * self.bandwidth),
            ]
            instrument_params = utils.average_over_bandpass(
                instrument_params, wavelength_range
            )
        elif parameters["observing_mode"] == "IFS":
            instrument_params = utils.interpolate_over_bandpass(
                instrument_params, mediator.get_observation_parameter("wavelength")
            )
        else:
            raise KeyError("Invalid observing mode. Must be 'IMAGER' or 'IFS'.")

        # Ensure coronagraph_optical_throughput has dimensions nlambda
        if np.isscalar(instrument_params["total_inst_refl"]):
            self.DEFAULT_CONFIG["coronagraph_optical_throughput"] = (
                np.array([instrument_params["total_inst_refl"]]) * DIMENSIONLESS
            )
        else:
            self.DEFAULT_CONFIG["coronagraph_optical_throughput"] = (
                np.array(instrument_params["total_inst_refl"]) * DIMENSIONLESS
            )

        # ***** Load the YIP using yippy *****
        yippy_obj = yippycoro(self.path)

        # get nrolls from yippy, if it exists in the YIP files
        try:
            self.DEFAULT_CONFIG["nrolls"] = yippy_obj.nrolls
        except AttributeError:
            # if yippy did not find nrolls, assume YIP has 36deg coverage and default to 1
            self.DEFAULT_CONFIG["nrolls"] = 1

        # ***** Set all parameters that are defined in the YIP (unpack YIP metadata) *****
        self.DEFAULT_CONFIG["pixscale"] = (
            yippy_obj.header.pixscale.value * LAMBDA_D
        )  # has units of lam/D / pix
        self.DEFAULT_CONFIG["npix"] = yippy_obj.header.naxis1
        self.DEFAULT_CONFIG["xcenter"] = yippy_obj.header.xcenter * PIXEL
        self.DEFAULT_CONFIG["ycenter"] = yippy_obj.header.ycenter * PIXEL

        # Sky transmission map for extended sources
        self.DEFAULT_CONFIG["skytrans"] = yippy_obj.sky_trans() * DIMENSIONLESS

        # create an array of circumstellar separations in units of lambd/D centered on star
        self.DEFAULT_CONFIG["r"] = (
            generate_radii(self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"])
            * self.DEFAULT_CONFIG["pixscale"]
        )

        # instantiate omega_lod and photometric_aperture_throughput
        self.DEFAULT_CONFIG["npsfratios"] = len(
            [mediator.get_observation_parameter("psf_trunc_ratio")]
        )
        omega_lod = np.zeros(
            (
                self.DEFAULT_CONFIG["npix"],
                self.DEFAULT_CONFIG["npix"],
                self.DEFAULT_CONFIG["npsfratios"],
            )
        )
        photometric_aperture_throughput = np.zeros(
            (
                self.DEFAULT_CONFIG["npix"],
                self.DEFAULT_CONFIG["npix"],
                self.DEFAULT_CONFIG["npsfratios"],
            )
        )

        # account for the method used to calculate omega (either use psf_trunc_ratio or photometric_aperture_radius, but not both)
        if (
            mediator.get_observation_parameter("psf_trunc_ratio") is None
            and mediator.get_observation_parameter("photometric_aperture_radius")
            is None
        ):
            raise KeyError(
                "WARNING: Neither psf_trunc_ratio or photometric_aperture_radius are specified. Specify one or the other to calculate Omega."
            )
        elif mediator.get_observation_parameter("psf_trunc_ratio") is not None:
            if (
                mediator.get_observation_parameter("photometric_aperture_radius")
                is not None
            ):
                print(
                    "WARNING: Both psf_trunc_ratio and photometric_aperture_radius are specified. Preferring psf_trunc_ratio going forward..."
                )
            print("Using psf_trunc_ratio to calculate Omega...")

            # Use the PSF truncation ratio method of calculating Omega with the YIP files

            # Get PSF Truncation ratio from Observation
            self.DEFAULT_CONFIG["psf_trunc_ratio"] = np.array(
                [mediator.get_observation_parameter("psf_trunc_ratio")]
            )  # TODO coronagraph needs it as an array, but it will be 1-dimensional. Reduce dimensions

            self.DEFAULT_CONFIG["psf_trunc_ratio"] = np.array(
                [mediator.get_observation_parameter("psf_trunc_ratio")]
            )
            offsets = np.sqrt(
                yippy_obj.offax.x_offsets**2 + yippy_obj.offax.y_offsets**2
            )
            noffsets = len(offsets)
            peakvals = np.zeros(noffsets)
            temp_omega_lod = np.zeros((noffsets, self.DEFAULT_CONFIG["npsfratios"]))
            temp_photometric_aperture_throughput = np.zeros(
                (noffsets, self.DEFAULT_CONFIG["npsfratios"])
            )
            resolvingfactor = int(np.ceil(self.DEFAULT_CONFIG["pixscale"] / 0.05))
            temppixomegalod = (self.DEFAULT_CONFIG["pixscale"] / resolvingfactor) ** 2
            resolvedPSFs = np.empty(
                (
                    noffsets,
                    resolvingfactor * self.DEFAULT_CONFIG["npix"],
                    resolvingfactor * self.DEFAULT_CONFIG["npix"],
                )
            )
            for i in range(noffsets):
                resolvedPSFs[i, :, :] = zoom(
                    yippy_obj.offax.reshaped_psfs[i, 0, :, :], resolvingfactor, order=3
                )

            resolvedPSFs = np.maximum(resolvedPSFs, 0)

            for i in range(noffsets):
                tempPSF = yippy_obj.offax.reshaped_psfs[i, 0, :, :]

                norm = np.sum(tempPSF)
                peakvals[i] = np.max(tempPSF)
                tempPSF_res = resolvedPSFs[i, :, :]

                norm2 = np.sum(tempPSF_res)
                if norm2 != 0:
                    tempPSF_res *= norm / norm2
                maxtempPSF = np.max(tempPSF_res)
                for j in range(self.DEFAULT_CONFIG["npsfratios"]):
                    thresh = self.DEFAULT_CONFIG["psf_trunc_ratio"][j] * maxtempPSF
                    k_idx = np.where(tempPSF_res > thresh)
                    if len(k_idx[0]) == 0:
                        k_idx = np.where(tempPSF_res == maxtempPSF)
                    temp_omega_lod[i, j] = len(k_idx[0]) * temppixomegalod
                    temp_photometric_aperture_throughput[i, j] = np.sum(
                        tempPSF_res[k_idx]
                    )

            # interpolate
            f_peak = interp1d(offsets, peakvals, bounds_error=False, fill_value=np.nan)
            PSFpeaks = f_peak(self.DEFAULT_CONFIG["r"])

            for j in range(self.DEFAULT_CONFIG["npsfratios"]):
                omega_lod[:, :, j] = np.interp(
                    self.DEFAULT_CONFIG["r"].ravel(), offsets, temp_omega_lod[:, j]
                ).reshape(self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"])
                photometric_aperture_throughput[:, :, j] = np.interp(
                    self.DEFAULT_CONFIG["r"].ravel(),
                    offsets,
                    temp_photometric_aperture_throughput[:, j],
                ).reshape(self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"])

        elif (
            mediator.get_observation_parameter("psf_trunc_ratio") is None
            and mediator.get_observation_parameter("photometric_aperture_radius")
            is not None
        ):
            print("Using photometric_aperture_radius to calculate Omega...")

            # Use the photometric_aperture_radius method of calculating Omega.
            # this method also requires you to set Tcore or uses the default one (does not use the YIP to calculate this)
            # ***** Tcore *****
            if "Tcore" in parameters.keys():
                print("Using user-defined Tcore...")
            else:
                print("Using default Tcore...")
            setattr(
                self,
                "Tcore",
                parameters.get("Tcore", self.DEFAULT_CONFIG["Tcore"]),
            )

            # simple omega calculation, omega = pi * (photometric_aperture_radius)**2, where photometric_aperture_radius is in lambda/D

            omega_lod = (
                np.full(
                    (self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"], 1),
                    float(np.pi)
                    * mediator.get_observation_parameter("photometric_aperture_radius")
                    ** 2,
                )
                * (
                    mediator.get_observation_parameter(
                        "photometric_aperture_radius"
                    ).unit
                )
                ** 2
            )  # size of photometric aperture at all separations (npix,npix,len(psftruncratio))

            photometric_aperture_throughput = (
                np.full(
                    (self.DEFAULT_CONFIG["npix"], self.DEFAULT_CONFIG["npix"], 1),
                    self.Tcore,
                )
                * self.Tcore.unit
            )  # core throughput at all separations (npix,npix,len(psftruncratio))

            photometric_aperture_throughput[
                self.DEFAULT_CONFIG["r"] < self.DEFAULT_CONFIG["minimum_IWA"]
            ] = 0.0  # index 0 is the
            photometric_aperture_throughput[
                self.DEFAULT_CONFIG["r"] > self.DEFAULT_CONFIG["maximum_OWA"]
            ] = 0.0

        omega_lod = np.maximum(omega_lod, 0)
        photometric_aperture_throughput = np.maximum(photometric_aperture_throughput, 0)

        self.DEFAULT_CONFIG["omega_lod"] = omega_lod * DIMENSIONLESS
        self.DEFAULT_CONFIG["photometric_aperture_throughput"] = (
            photometric_aperture_throughput * DIMENSIONLESS
        )

        stellar_angular_diameter_arcsec = mediator.get_scene_parameter(
            "stellar_angular_diameter_arcsec"
        )
        lam = mediator.get_observation_parameter("wavelength")

        # TODO how to behave when tele_diam has been overwritten by the user?
        telescope_params = load_telescope("EAC1").__dict__
        tele_diam = telescope_params["diam_circ"] * LENGTH

        stellar_angular_diameter_lod = arcsec_to_lambda_d(
            stellar_angular_diameter_arcsec, 0.55 * WAVELENGTH, tele_diam
        )  # NOTE TODO IMPORTANT! We have to know the wavelength that angdia_arcsec is given at. Right now, assumes 0.55 um. Suggest changing input file to take in angdiam_lod instead of angdiam_arcsec

        self.DEFAULT_CONFIG["Istar"] = (
            yippy_obj.stellar_intens(stellar_angular_diameter_lod.value * lod)[:, :]
            * DIMENSIONLESS
        )

        # Calculate noisefloor
        if "noisefloor_PPF" in parameters.keys():
            print("Setting the noise floor via user-supplied noisefloor_PPF...")
            self.DEFAULT_CONFIG["noisefloor_PPF"] = parameters["noisefloor_PPF"]

        else:
            if "noisefloor_factor" in parameters.keys():
                print(
                    "Noisefloor_factor mode not implemented in CoronagraphYIP coronagraph. Please use the noisefloor_PPF method."
                )
            print(
                "WARNING: noisefloor_PPF value not provided. Using the default value: "
                + str(self.DEFAULT_CONFIG["noisefloor_PPF"])
            )

        self.DEFAULT_CONFIG["noisefloor"] = (
            self.DEFAULT_CONFIG["Istar"] / self.DEFAULT_CONFIG["noisefloor_PPF"]
        )

        if self.DEFAULT_CONFIG["az_avg"]:
            # azimuthally average the stellar intensity to smooth it out
            print("Azimuthally averaging contrast maps and noise floor...")
            ntheta = 100
            theta_vals = np.linspace(0, 360, ntheta, endpoint=False)
            tempIstar = np.zeros_like(self.DEFAULT_CONFIG["Istar"])
            tempnoisefloor = np.zeros_like(self.DEFAULT_CONFIG["noisefloor"])
            for angle in theta_vals:
                tempIstar[:, :] += rotate(
                    self.DEFAULT_CONFIG["Istar"][:, :], angle, reshape=False, order=3
                )
                tempnoisefloor[:, :] += rotate(
                    self.DEFAULT_CONFIG["noisefloor"][:, :],
                    angle,
                    reshape=False,
                    order=3,
                )
            self.DEFAULT_CONFIG["Istar"] = tempIstar / ntheta
            self.DEFAULT_CONFIG["noisefloor"] = tempnoisefloor / ntheta

        # ***** REPLACE PARAMETERS WITH USER-SPECIFIED ONES ****
        # for coronagraph, allow replacement only in terms of scaling factors?
        # NOTE what should be allowed to be replaced?
        # Load parameters, use defaults if not provided
        utils.fill_parameters(self, parameters, self.DEFAULT_CONFIG)
