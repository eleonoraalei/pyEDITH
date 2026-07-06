import numpy as np
from .units import *
from . import utils
import logging
from pyEDITH import parse_input

logger = logging.getLogger("pyEDITH")


class Observation:
    """
    A class representing an astronomical observation.

    This class encapsulates various parameters and methods related to
    astronomical observations, including target star properties, planet
    characteristics, observational settings, telescope specifications,
    instrument details, and detector parameters.

    Parameters
    -----------
    wavelength : np.ndarray
        Wavelength array (in microns).
    nlambd : int
        Number of wavelength points.
    SNR : np.ndarray
        Desired bulk SNR.
    exptime : ndarray
        Exposure time per single wavelength datapoint.
    fullsnr : ndarray
        Signal-to-noise ratio per single wavelength datapoint.
    td_limit : float
        Limit placed on exposure times.

    """

    def __init__(self) -> None:
        """
        Initialize the default parameters of the Observation class.

        Sets the default exposure time limit (td_limit) to a large value.
        """

        # Misc parameters that probably don't need to be changed
        self.td_limit = 1e20 * TIME  # limit placed on exposure times # scalar

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the observation from a dictionary.

        This method initializes various observation parameters from the provided
        dictionary, including wavelength arrays, signal-to-noise ratios, and
        aperture settings. For IFS mode, it can calculate or regrid the wavelength
        grid based on specified parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing observation parameters including wavelengths,
            SNR values, aperture settings, and observation mode settings. Must
            include 'observing_mode', 'wavelength', 'snr'.

        Raises
        ------
        KeyError
            If required parameters are missing or if regridding is requested without
            necessary parameters
        """
        self.observing_mode = parameters["observing_mode"]
        if self.observing_mode not in ["IMAGER", "IFS"]:
            raise KeyError("Invalid observing mode. Must be 'IMAGER' or 'IFS'.")

        # -------- INPUTS ---------
        # Observational parameters
        if parameters["observing_mode"] == "IMAGER":
            self.wavelength = (
                parameters["wavelength"] * WAVELENGTH
            )  # wavelength # nlambd array #unit: micron
            self.SNR = (
                parameters["snr"] * DIMENSIONLESS
            )  # signal to noise # nlambd array

        elif (
            parameters["observing_mode"] == "IFS"
            and bool(parameters["regrid_wavelength"]) is False
        ):
            self.wavelength = (
                parameters["wavelength"] * WAVELENGTH
            )  # wavelength # nlambd array #unit: micron
            IFS_resolution = self.wavelength / np.gradient(
                self.wavelength
            )  # calculate the resolution from the wavelength grid
            dlam_um = np.gradient(self.wavelength)
            if ~np.isfinite(IFS_resolution).any():
                logger.warning(
                    "Wavelength grid is not valid. Using default spectral resolution of 140."
                )
                IFS_resolution = 140 * np.ones_like(
                    self.wavelength
                )  # default resolution
                dlam_um = self.wavelength / IFS_resolution
            self.delta_wavelength = dlam_um
            self.SNR = (
                parameters["snr"] * DIMENSIONLESS
            )  # signal to noise # nlambd array
        elif (
            parameters["observing_mode"] == "IFS"
            and bool(parameters["regrid_wavelength"]) is True
        ):
            logger.info("Calculating a new wavelength grid and re-gridding spectra...")
            if "spectral_resolution" not in parameters.keys():
                raise KeyError(
                    "regrid_wavelength is True; you must specify new resolution for each spectral channel: parameters['spectral_resolution']."
                )
            if "lam_low" not in parameters.keys():
                raise KeyError(
                    "regrid_wavelength is True; you must specify the wavelength boundaries between spectral channels: parameters['lam_low']."
                )
            if "lam_high" not in parameters.keys():
                raise KeyError(
                    "regrid_wavelength is True; you must specify the wavelength boundaries between spectral channels: parameters['lam_high']."
                )

            new_lam, new_dlam = utils.regrid_wavelengths(
                parameters["wavelength"],
                parameters["spectral_resolution"],
                parameters["lam_low"],
                parameters["lam_high"],
            )
            self.wavelength = (
                new_lam * WAVELENGTH
            )  # wavelength # nlambd array #unit: micron
            self.delta_wavelength = new_dlam * WAVELENGTH

            # PRELIMINARY to rebin SNR, treat it as a spectrum #TODO debate if it should be wavelength-dependent in the first place
            self.SNR = utils.regrid_spec_gaussconv(
                parameters["wavelength"],
                parameters["snr"] * DIMENSIONLESS,
                self.wavelength.value,
                self.delta_wavelength.value,
            )  # signal to noise # nlambd array

        self.CRb_multiplier = float(parameters["CRb_multiplier"])

        self.nlambd = len(self.wavelength)

    def set_output_arrays(self):
        """
        Initialize arrays for storing observation results.

        This method creates and initializes the arrays that will store the
        calculated exposure times and signal-to-noise ratios for each
        wavelength point in the observation.
        """

        # Initialize some arrays needed for outputs...
        self.exptime = np.full((self.nlambd), 0.0) * TIME

        # only used for snr calculation
        self.fullsnr = np.full((self.nlambd), 0.0) * DIMENSIONLESS

    def validate_configuration(self):
        """
        Validate that all required observation parameters are present and correctly formatted.

        This method checks that all mandatory attributes exist on the observation
        object and that they have the expected types and units.

        Raises
        ------
        TypeError
            If an attribute has an incorrect type
        ValueError
            If a Quantity attribute has incorrect units
        """
        expected_args = {
            "wavelength": WAVELENGTH,
            "nlambd": int,
            "SNR": DIMENSIONLESS,
            "CRb_multiplier": float,
        }

        utils.validate_attributes(self, expected_args)
