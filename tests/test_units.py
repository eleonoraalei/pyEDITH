import pytest
from astropy import units as u
from pyEDITH.units import (
    lambda_d_to_radians,
    radians_to_lambda_d,
    lambda_d_to_arcsec,
    arcsec_to_lambda_d,
    arcsec_to_au,
    LAMBDA_D,
    LENGTH,
    ARCSEC,
)


def test_lambda_d_to_radians():
    wavelength = 500 * u.nm
    diameter = 10 * u.m
    value_lod = 2 * LAMBDA_D
    result = lambda_d_to_radians(value_lod, wavelength, diameter)
    expected = 1e-7 * u.rad
    assert pytest.approx(result.value) == expected.value
    assert result.unit == expected.unit


def test_radians_to_lambda_d():
    wavelength = 500 * u.nm
    diameter = 10 * u.m
    angle = 1e-7 * u.rad
    result = radians_to_lambda_d(angle, wavelength, diameter)
    expected = 2 * LAMBDA_D
    assert pytest.approx(result.value) == expected.value
    assert result.unit == expected.unit


def test_lambda_d_to_arcsec():
    wavelength = 500 * u.nm
    diameter = 10 * u.m
    value_lod = 2 * LAMBDA_D
    result = lambda_d_to_arcsec(value_lod, wavelength, diameter)
    expected = 0.02062648 * ARCSEC
    assert pytest.approx(result.value) == expected.value
    assert result.unit == expected.unit


def test_arcsec_to_lambda_d():
    wavelength = 500 * u.nm
    diameter = 10 * u.m
    angle = 0.02062648 * ARCSEC
    result = arcsec_to_lambda_d(angle, wavelength, diameter)
    expected = 2 * LAMBDA_D
    assert pytest.approx(result.value) == expected.value
    assert result.unit == expected.unit


def test_arcsec_to_au():
    # Use definition of parsec
    angle = 0.1 * ARCSEC
    distance = 10 * u.pc
    result = arcsec_to_au(angle, distance)
    expected = 1 * u.au
    assert pytest.approx(result.value) == expected.value
    assert result.unit == expected.unit
