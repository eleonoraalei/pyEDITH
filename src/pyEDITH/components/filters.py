from abc import ABC, abstractmethod
import numpy as np
from ..units import *
import astropy.units as u
from .. import utils
import logging

logger = logging.getLogger("pyEDITH")


class Filter:
    """
    Represents a spectral channel with wavelength bounds and resolution.
    """

    def __init__(
        self,
        name: str,
        *,
        low: u.Quantity = None,
        high: u.Quantity = None,
        center: u.Quantity = None,
        bandpass: float = None,
        resolution: float = None,
        type: str = "IFS",
    ):
        """
        If type is 'IMAGER', save a filter object with wavelength and bandpass.
        If type is 'IFS', create a filter from either:
        - low, high, resolution (explicit bounds)
        - center, bandpass, resolution (center wavelength + fractional bandwidth)

        And generate wavelength array for the filters given those parameters.

        Parameters
        ----------
        name : str
            Filter identifier (e.g., "UV", "500nm")
        type :
        low : u.Quantity, optional
            Lower wavelength bound
        high : u.Quantity, optional
            Upper wavelength bound
        center : u.Quantity, optional
            Center wavelength
        bandpass : float, optional
            Fractional bandwidth (e.g., 0.2 for 20%)
        resolution : float
            Spectral resolution R = λ/Δλ
        """
        self.name = name
        self.type = type
        self.resolution = resolution

        if self.resolution is None:
            logging.warning(
                f"Filter {name}: resolution not set, will default to IMAGER type."
            )
        if low is not None and high is not None:
            self.low = low.to(WAVELENGTH)
            self.high = high.to(WAVELENGTH)
            self.center = (self.low + self.high) / 2
            self.bandpass = ((self.high - self.low) / self.center).to(DIMENSIONLESS)
            self.width = self.high - self.low

        elif center is not None and bandpass is not None:
            self.center = center.to(WAVELENGTH)
            self.bandpass = bandpass * DIMENSIONLESS
            self.low = self.center * (1 - self.bandpass.value / 2)
            self.high = self.center * (1 + self.bandpass.value / 2)
            self.width = self.high - self.low

        else:
            raise ValueError(
                f"Filter '{name}': Must provide either (low, high) or (center, bandpass)"
            )

        if self.type == "IMAGER" or self.resolution == None:
            ## BROADBAND PHOTOMETRY
            self.wavelength = self.center
            self.delta_wavelength = None

        elif self.type == "IFS":
            ## SPECTROSCOPY
            # Create wavelength array
            lam, dlam = utils.generate_wavelength_grid(
                res=self.resolution,
                lam_low=self.low.value,
                lam_high=self.high.value,
            )

            self.wavelength = lam * WAVELENGTH
            self.delta_wavelength = dlam * WAVELENGTH
        else:
            raise ValueError(
                f"Filter '{name}': Type must be either IMAGER (broadband photometry) or IFS (spectroscopy)."
            )

    def contains(self, wavelength: u.Quantity) -> bool:
        """Check if a wavelength falls within this filter."""
        wavelength = wavelength.to(WAVELENGTH)
        return (wavelength >= self.low) and (wavelength <= self.high)

    def __repr__(self):
        if self.type == "IMAGER" or self.resolution is None:
            return (
                f"Filter(name='{self.name}', type='{self.type}', "
                f"center={self.center:.3f}, bandpass={self.bandpass.value:.2%}, "
                f"width={self.width:.3f})"
            )
        else:

            return (
                f"Filter(name='{self.name}', type='{self.type}', "
                f"range={self.low:.3f}-{self.high:.3f}, center={self.center:.3f}, "
                f"R={self.resolution})"
            )


# # Predefined FULL_CHANNEL_FILTERS (to reproduce legacy behavior)
FULL_CHANNEL_FILTERS = {
    "UV": Filter("UV", low=0.2 * u.um, high=0.4 * u.um, resolution=7),
    "VIS": Filter("VIS", low=0.4 * u.um, high=1.0 * u.um, resolution=140),
    "NIR": Filter("NIR", low=1.0 * u.um, high=1.8 * u.um, resolution=70),
}


# def get_hwome_channels(channel_names: list = None) -> list:
#     """will be implemented when hwome is operational."""
#     return NotImplementedError
