import numpy as np
from .units import *
from . import utils


class Observation:
    """
    A class representing an astronomical observation.

    This class encapsulates various parameters and methods related to
    astronomical observations, including target star properties, planet
    characteristics, observational settings, telescope specifications,
    instrument details, and detector parameters.

    Attributes:
    -----------
    lambd : np.ndarray
        Wavelength array (in microns).
    nlambd : int
        Number of wavelength points.
    SNR : np.ndarray
        Signal-to-noise ratio array.
    photap_rad : float
        Photometric aperture radius (in units of lambda/D).
    psf_trunc_ratio : np.ndarray
        PSF truncation ratio.
    tp : ndarray
        Exposure time of every planet (nmeananom x norbits x ntargs array).
    exptime : ndarray
        Exposure time for each target and wavelength.
    fullsnr : ndarray
        Signal-to-noise ratio for each target and wavelength.
    snr_ez : ndarray
        SNR on the exozodi; only used for when ez cannot be subtracted to Poisson limit
    td_limit : float
        Limit placed on exposure times.

    """

    def __init__(self) -> None:
        """
        Initialize the default parameters of the Observation class.
        """
        # Misc parameters that probably don't need to be changed
        self.td_limit = 1e20 * TIME  # limit placed on exposure times # scalar

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary of
        parameters that was read from the input file.

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
        # Observational parameters

        self.wavelength = (
            parameters["wavelength"] * WAVELENGTH
        )  # wavelength # nlambd array #unit: micron

        self.SNR = parameters["snr"] * DIMENSIONLESS  # signal to noise # nlambd array

        if "psf_trunc_ratio" in parameters.keys():
            self.psf_trunc_ratio = (
                parameters["psf_trunc_ratio"] * DIMENSIONLESS
            )  # scalar
        elif "photap_rad" in parameters.keys():
            self.photap_rad = parameters["photap_rad"] * LAMBDA_D  # (lambd/D) # scalar

        else:
            raise KeyError(
                "Either 'photap_rad' or 'psf_trunc_ratio' must be provided in the parameters."
            )
        # If both are provided, we'll use psf_trunc_ratio and ignore photap_rad
        if (
            "photap_rad" in parameters
            and "psf_trunc_ratio" in parameters
            and parameters["photap_rad"] is not None
            and parameters["psf_trunc_ratio"] is not None
        ):
            print(
                "Warning: Both 'photap_rad' and 'psf_trunc_ratio' provided. Using 'psf_trunc_ratio' and ignoring 'photap_rad'."
            )
            self.photap_rad = None  # ignore photap_rad by setting to None
            # TODO goes with coronagraph implementation of photap_rad function.

        self.CRb_multiplier = float(parameters["CRb_multiplier"])

        self.nlambd = len(self.wavelength)

    def set_output_arrays(self):
        """
        Initialize output arrays:

        - tp : ndarray
            Exposure time of every planet (nmeananom x norbits x ntargs array).
        - exptime : ndarray
            Exposure time for each target and wavelength.
        - fullsnr : ndarray
            Signal-to-noise ratio for each target and wavelength.
        - snr_ez : ndarray
            SNR on exozodi
        """
        # Initialize some arrays needed for outputs...
        self.tp = 0.0 * TIME  # exposure time of every planet
        # (nmeananom x norbits x ntargs array), used in c function
        # [NOTE: nmeananom = nphases in C code]
        # NOTE: ntargs fixed to 1.
        self.exptime = np.full((self.nlambd), 0.0) * TIME

        # only used for snr calculation
        self.fullsnr = np.full((self.nlambd), 0.0) * DIMENSIONLESS

        # only used for manual exozodi studies. Does not affect calculations on planet SNR.
        self.snr_ez = np.full((self.nlambd), 0.0) * DIMENSIONLESS

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "wavelength": WAVELENGTH,
            "nlambd": int,
            "SNR": DIMENSIONLESS,
            "CRb_multiplier": float,
        }

        # Either photap_rad or psf_trunc_ratio must be present
        if hasattr(self, "photap_rad") and getattr(self, "photap_rad") is not None:
            expected_args["photap_rad"] = LAMBDA_D
        elif (
            hasattr(self, "psf_trunc_ratio")
            and getattr(self, "psf_trunc_ratio") is not None
        ):
            expected_args["psf_trunc_ratio"] = DIMENSIONLESS
        else:
            raise AttributeError(
                "Observation must have either 'photap_rad' or 'psf_trunc_ratio' attribute"
            )

        utils.validate_attributes(self, expected_args)
        # for arg, expected_type in expected_args.items():
        #     if not hasattr(self, arg):
        #         raise AttributeError(f"Observation is missing attribute: {arg}")
        #     value = getattr(self, arg)

        #     if expected_type is int:
        #         if not isinstance(value, (int, np.integer)):
        #             raise TypeError(f"Observation attribute {arg} should be an integer")
        #     elif expected_type is float:
        #         if not isinstance(value, (float, np.floating)):
        #             raise TypeError(f"Observation attribute {arg} should be a float")
        #     elif expected_type in ALL_UNITS:
        #         if not isinstance(value, u.Quantity):
        #             raise TypeError(f"Observation attribute {arg} should be a Quantity")
        #         if not value.unit == expected_type:
        #             raise ValueError(f"Observation attribute {arg} has incorrect units")
        #     else:
        #         raise ValueError(f"Unexpected type specification for {arg}")

        #     # Additional check for numerical values
        #     if isinstance(value, u.Quantity):
        #         if not np.issubdtype(value.value.dtype, np.number):
        #             raise TypeError(
        #                 f"Observation attribute {arg} should contain numerical values"
        #             )
        #     elif not np.issubdtype(type(value), np.number):
        #         raise TypeError(f"Observation attribute {arg} should be a number")
