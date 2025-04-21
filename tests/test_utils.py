import numpy as np
import astropy.units as u
from pyEDITH.units import (
    MAS,
    DIMENSIONLESS,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    WAVELENGTH,
    LENGTH,
    ARCSEC,
)
from pyEDITH.utils import average_over_bandpass,interpolate_over_bandpass

def test_average_over_bandpass():
    params = {
        "lam": np.array([0.4, 0.5, 0.6, 0.7, 0.8]) * WAVELENGTH,
        "value": np.array([1, 2, 3, 4, 5]) * DIMENSIONLESS,
    }
    wavelength_range = [0.45 * WAVELENGTH, 0.75 * WAVELENGTH]

    result = average_over_bandpass(params, wavelength_range)
    assert np.isclose(result["value"].value, 3)


def test_interpolate_over_bandpass():
    params = {
        "lam": np.array([0.4, 0.5, 0.6, 0.7, 0.8]) * WAVELENGTH,
        "value": np.array([1, 2, 3, 4, 5]) * DIMENSIONLESS,
    }
    wavelengths = u.Quantity([0.45, 0.55, 0.65, 0.75],WAVELENGTH)

    result = interpolate_over_bandpass(params, wavelengths)
    assert np.allclose(result["value"], np.array([1.5, 2.5, 3.5, 4.5]))
