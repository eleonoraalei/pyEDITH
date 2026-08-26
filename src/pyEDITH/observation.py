import numpy as np
from .units import *
from . import utils
import logging
from pyEDITH import parse_input
from pyEDITH.filters import Filter

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

    def load_configuration(self, parameters: dict, filter: Filter = None) -> None:
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

        # -------- INPUTS ---------
        # Observational parameters

        if filter is not None:
            self.wavelength = filter.wavelength
            self.delta_wavelength = filter.delta_wavelength
            self.filter_bandwidth = filter.bandwidth
            self.wavelength_range = [
                filter.low,
                filter.high,
            ]

        else:
            raise ValueError("Please define a single filter for this simulation.")

        # ------------------------------------------------------------------
        # SNR: align onto the current resolved grid.
        # ------------------------------------------------------------------
        self.SNR = utils.resample_to_wavelength_grid(
            parameters["snr"] * DIMENSIONLESS,
            from_wavelength=parameters["wavelength"],  # input wavelength
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
        self.exptime = np.full(len(self.wavelength), 0.0) * TIME

        # only used for snr calculation
        self.fullsnr = np.full(len(self.wavelength), 0.0) * DIMENSIONLESS

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
            "SNR": DIMENSIONLESS,
            "CRb_multiplier": float,
        }

        utils.validate_attributes(self, expected_args)
