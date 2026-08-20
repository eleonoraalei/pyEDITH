import numpy as np
from .units import *
from . import utils
import logging
from pyEDITH import parse_input

logger = logging.getLogger("pyEDITH")


class Observation:
    """
    A class representing the observational parameters.

    This class collects the "Operations" setup, specifically initializing the
    output arrays and selecting the observing mode and filters that the user wants to use.

    Parameters
    -----------
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
        parameters = parse_input.parse_parameters(parameters)

        self.observing_mode = parameters["observing_mode"]

        # CHECK THAT WAVELENGTH IS IN FILTERS, OTHERWISE DEACTIVATE FILTER

        # assert (
        #     np.min(input_wls) <= lam_low
        # ), "Your minimum input wavelength is greater than the lower boundary."
        # assert (
        #     np.max(input_wls) >= lam_high
        # ), "Your maximum input wavelength is less than the upper boundary."

        # -------- FILTERS ---------
        # Observational parameters
        print("filter_list" in parameters.keys())
        if "filter_list" in parameters.keys():
            active_filters = []
            wl_min = np.min(parameters["wavelength"]) * WAVELENGTH
            wl_max = np.max(parameters["wavelength"]) * WAVELENGTH

            # Calculate input spectrum resolution if in IFS mode
            if self.observing_mode == "IFS":
                input_wavelengths = parameters["wavelength"] * WAVELENGTH
                input_dlam = np.gradient(input_wavelengths)
                input_resolution = input_wavelengths / input_dlam

            for f in parameters["filter_list"]:

                if wl_min <= f.low and wl_max >= f.high:
                    active_filters.append(f)

                    # Check spectral resolution compatibility for IFS mode
                    if self.observing_mode == "IFS":
                        # Find overlapping wavelength region
                        overlap_mask = (input_wavelengths >= f.low) & (
                            input_wavelengths <= f.high
                        )
                        if np.any(overlap_mask):
                            # Get median resolution in overlapping region
                            median_input_res = np.median(input_resolution[overlap_mask])

                            # Warn if input resolution is lower than filter resolution
                            if median_input_res < f.resolution:
                                logger.warning(
                                    f"Filter {f.name if hasattr(f, 'name') else f}: "
                                    f"Input spectrum resolution (R~{median_input_res:.1f}) is lower than "
                                    f"filter resolution (R~{f.resolution:.1f}). "
                                    f"Interpolation to filter wavelength grid may introduce artifacts."
                                )
                else:
                    logger.warning(
                        f"Filter {f.name if hasattr(f, 'name') else f} discarded: "
                        f"wavelength range [{wl_min}, {wl_max}] does not fully cover "
                        f"filter range [{f.low}, {f.high}]"
                    )

            parameters["filter_list"] = active_filters

            if len(active_filters) == 0:
                raise ValueError(
                    "No filters remain after filtering. Specify different filters or change spectrum."
                )
            else:
                # concatenate wavelengths from all active filters (to reproduce legacy behavior)
                filter_wavelengths = [f.wavelength for f in active_filters]
                self.wavelength = np.concatenate(filter_wavelengths)
                filter_deltawavelengths = [f.delta_wavelength for f in active_filters]
                self.delta_wavelength = np.concatenate(filter_deltawavelengths)
        else:
            self.legacy_helper(parameters)

        # ------------------------------------------------------------------
        # Length of the current resolved grid. Any per-wavelength parameter is
        # aligned against this via resample_to_wavelength_grid.
        # ------------------------------------------------------------------
        self.nlambda = len(self.wavelength)

        # ------------------------------------------------------------------
        # SNR: align onto the current resolved grid.
        # ------------------------------------------------------------------
        self.SNR = utils.resample_to_wavelength_grid(
            parameters["snr"] * DIMENSIONLESS,
            from_wavelength=parameters["wavelength"],
            to_wavelength=self.wavelength.value,
            name="snr",
            interpolation="1d",
        )  # signal to noise # nlambda array

        self.CRb_multiplier = float(parameters["CRb_multiplier"])

    def set_output_arrays(self):
        """
        Initialize arrays for storing observation results.

        This method creates and initializes the arrays that will store the
        calculated exposure times and signal-to-noise ratios for each
        wavelength point in the observation.
        """

        # Initialize some arrays needed for outputs...
        self.exptime = np.full((self.nlambda), 0.0) * TIME

        # only used for snr calculation
        self.fullsnr = np.full((self.nlambda), 0.0) * DIMENSIONLESS

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
            "nlambda": int,
            "SNR": DIMENSIONLESS,
            "CRb_multiplier": float,
        }

        utils.validate_attributes(self, expected_args)

    def legacy_helper(self, parameters):

        if parameters["observing_mode"] == "IMAGER":
            self.wavelength = (
                parameters["wavelength"] * WAVELENGTH
            )  # wavelength # nlambda array #unit: micron
            # IMAGER has no meaningful bin widths for regridding; broadcast-only
            self.delta_wavelength = None

        elif (
            parameters["observing_mode"] == "IFS"
            and bool(parameters["regrid_wavelength"]) is False
        ):
            self.wavelength = (
                parameters["wavelength"] * WAVELENGTH
            )  # wavelength # nlambda array #unit: micron
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

        elif (
            parameters["observing_mode"] == "IFS"
            and bool(parameters["regrid_wavelength"]) is True
        ):
            logger.info("Calculating a new wavelength grid and re-gridding spectra...")
            input_wls = parameters["wavelength"]

            assert (
                len(parameters["spectral_resolution"])
                == len(parameters["lam_low"])
                == len(parameters["lam_high"])
            )
            assert (
                np.min(input_wls) < parameters["lam_low"][0]
            ), "Your minimum input wavelength is greater than first channel lower boundary."
            assert (
                np.max(input_wls) > parameters["lam_high"][-1]
            ), f"Your maximum input wavelength is less than last channel upper boundary."

            new_lam = []
            new_dlam = []
            for i in range(0, len(parameters["spectral_resolution"])):
                res = parameters["spectral_resolution"][i]
                lam_low = parameters["lam_low"][i]
                lam_high = parameters["lam_high"][i]

                lam, dlam = utils.generate_wavelength_grid(res, lam_low, lam_high)

                new_lam = np.concatenate((new_lam, lam))
                new_dlam = np.concatenate((new_dlam, dlam))

            self.wavelength = (
                new_lam * WAVELENGTH
            )  # wavelength # nlambda array #unit: micron
            self.delta_wavelength = new_dlam * WAVELENGTH
