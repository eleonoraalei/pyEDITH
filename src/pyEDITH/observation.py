import numpy as np
from .units import *


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

        self.photap_rad = parameters["photap_rad"] * LAMBDA_D  # (lambd/D) # scalar

        self.psf_trunc_ratio = parameters["psf_trunc_ratio"] * DIMENSIONLESS  # scalar

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
        """
        # Initialize some arrays needed for outputs...
        self.tp = 0.0 * TIME  # exposure time of every planet
        # (nmeananom x norbits x ntargs array), used in c function
        # [NOTE: nmeananom = nphases in C code]
        # NOTE: ntargs fixed to 1.
        self.exptime = np.full((self.nlambd), 0.0) * TIME

        # only used for snr calculation
        self.fullsnr = np.full((self.nlambd), 0.0) * DIMENSIONLESS

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "wavelength": WAVELENGTH,
            "nlambd": int,
            "SNR": DIMENSIONLESS,
            "photap_rad": LAMBDA_D,
            "psf_trunc_ratio": DIMENSIONLESS,
            "CRb_multiplier": float,
        }

        for arg, expected_type in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Observation is missing attribute: {arg}")
            value = getattr(self, arg)

            if expected_type is int:
                if not isinstance(value, (int, np.integer)):
                    raise TypeError(f"Observation attribute {arg} should be an integer")
            elif expected_type is float:
                if not isinstance(value, (float, np.floating)):
                    raise TypeError(f"Observation attribute {arg} should be a float")
            elif expected_type in ALL_UNITS:
                if not isinstance(value, u.Quantity):
                    raise TypeError(f"Observation attribute {arg} should be a Quantity")
                if not value.unit.is_equivalent(expected_type):
                    raise ValueError(f"Observation attribute {arg} has incorrect units")
            else:
                raise ValueError(f"Unexpected type specification for {arg}")

            # Additional check for numerical values
            if isinstance(value, u.Quantity):
                if not np.issubdtype(value.value.dtype, np.number):
                    raise TypeError(
                        f"Observation attribute {arg} should contain numerical values"
                    )
            elif not np.issubdtype(type(value), np.number):
                raise TypeError(f"Observation attribute {arg} should be a number")
