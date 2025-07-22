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

    Parameters
    -----------
    lambd : np.ndarray
        Wavelength array (in microns).
    nlambd : int
        Number of wavelength points.
    SNR : np.ndarray
        Signal-to-noise ratio array.
    photometric_aperture_radius : float
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
        if parameters["observing_mode"] == "IMAGER":
            self.wavelength = (
                parameters["wavelength"] * WAVELENGTH
            )  # wavelength # nlambd array #unit: micron
        elif (
            parameters["observing_mode"] == "IFS"
            and parameters["regrid_wavelength"] is False
        ):
            self.wavelength = (
                parameters["wavelength"] * WAVELENGTH
            )  # wavelength # nlambd array #unit: micron
            IFS_resolution = self.wavelength / np.gradient(
                self.wavelength
            )  # calculate the resolution from the wavelength grid
            dlam_um = np.gradient(self.wavelength)
            if ~np.isfinite(IFS_resolution).any():
                print(
                    "WARNING: Wavelength grid is not valid. Using default spectral resolution of 140."
                )
                IFS_resolution = 140 * np.ones_like(
                    self.wavelength
                )  # default resolution
                dlam_um = self.wavelength / IFS_resolution
            self.delta_wavelength = dlam_um

        elif (
            parameters["observing_mode"] == "IFS"
            and parameters["regrid_wavelength"] is True
        ):
            print("Calculating a new wavelength grid and re-gridding spectra...")
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

        self.SNR = parameters["snr"] * DIMENSIONLESS  # signal to noise # nlambd array

        # Set defaults, replace if you can
        self.photometric_aperture_radius = None
        self.psf_trunc_ratio = None
        if "psf_trunc_ratio" in parameters.keys():
            self.psf_trunc_ratio = (
                parameters["psf_trunc_ratio"] * DIMENSIONLESS
            )  # scalar
        elif "photometric_aperture_radius" in parameters.keys():
            self.photometric_aperture_radius = (
                parameters["photometric_aperture_radius"] * LAMBDA_D
            )  # (lambd/D) # scalar

        else:
            raise KeyError(
                "Either 'photometric_aperture_radius' or 'psf_trunc_ratio' must be provided in the parameters."
            )
        # If both are provided, we'll use psf_trunc_ratio and ignore photometric_aperture_radius
        if (
            "photometric_aperture_radius" in parameters
            and "psf_trunc_ratio" in parameters
            and parameters["photometric_aperture_radius"] is not None
            and parameters["psf_trunc_ratio"] is not None
        ):
            print(
                "Warning: Both 'photometric_aperture_radius' and 'psf_trunc_ratio' provided. Using 'psf_trunc_ratio' and ignoring 'photometric_aperture_radius'."
            )
            self.photometric_aperture_radius = (
                None  # ignore photometric_aperture_radius by setting to None
            )

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
            "CRb_multiplier": float,
        }

        # Either photometric_aperture_radius or psf_trunc_ratio must be present
        if (
            hasattr(self, "photometric_aperture_radius")
            and getattr(self, "photometric_aperture_radius") is not None
        ):
            expected_args["photometric_aperture_radius"] = LAMBDA_D
        elif (
            hasattr(self, "psf_trunc_ratio")
            and getattr(self, "psf_trunc_ratio") is not None
        ):
            expected_args["psf_trunc_ratio"] = DIMENSIONLESS
        else:
            raise AttributeError(
                "Observation must have either 'photometric_aperture_radius' or 'psf_trunc_ratio' attribute"
            )

        utils.validate_attributes(self, expected_args)
