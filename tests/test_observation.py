import pytest
import numpy as np
from astropy import units as u
import logging

from pyEDITH.observation import Observation
from pyEDITH.units import WAVELENGTH, DIMENSIONLESS, LAMBDA_D, TIME, MAGNITUDE
from pyEDITH.filters import Filter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ifs_observation_params():
    """Fixture providing IFS mode observation parameters."""
    return {
        "wavelength": np.linspace(0.2, 1.8, 100),
        "snr": [7.0] * np.ones(100),
        "regrid_wavelength": True,
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
    }


@pytest.fixture
def ifs_filter():
    return Filter("IFS", low=0.4, high=0.9, resolution=5)


@pytest.fixture
def basic_observation_params():
    """Fixture providing single wavelength observation parameters."""
    return {
        "wavelength": [0.5],
        "snr": [7.0],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }


@pytest.fixture
def basic_filter():
    return Filter("IMAGER", center=500 * u.nm, bandwidth=0.2)


# ============================================================================
# Tests for Observation initialization
# ============================================================================


def test_observation_init():
    """Test that Observation initializes with default td_limit."""
    obs = Observation()

    assert obs.td_limit == 1e20 * TIME


# ============================================================================
# Tests for Observation.load_configuration - Basic functionality
# ============================================================================


def test_observation_load_configuration_basic(basic_observation_params, basic_filter):
    """Test loading basic observation configuration."""
    obs = Observation()
    obs.load_configuration(basic_observation_params, filter=basic_filter)

    assert np.all(obs.wavelength == basic_filter.wavelength)
    assert np.all(obs.delta_wavelength == basic_filter.delta_wavelength)
    assert np.all(obs.filter_bandwidth == basic_filter.bandwidth)
    assert obs.wavelength_range == [
        basic_filter.wavelength * (1 - 0.5 * basic_filter.bandwidth),
        basic_filter.wavelength * (1 + 0.5 * basic_filter.bandwidth),
    ]

    assert np.all(
        obs.SNR
        == basic_observation_params["snr"][0]
        * np.ones_like(basic_filter.wavelength.value)
        * DIMENSIONLESS
    )
    assert obs.CRb_multiplier == basic_observation_params["CRb_multiplier"]


def test_observation_load_configuration_ifs_filter(ifs_observation_params, ifs_filter):
    """Test loading basic observation configuration."""
    obs = Observation()
    obs.load_configuration(ifs_observation_params, filter=ifs_filter)

    assert np.all(obs.wavelength == ifs_filter.wavelength)
    assert np.all(obs.delta_wavelength == ifs_filter.delta_wavelength)
    assert np.all(obs.filter_bandwidth == ifs_filter.bandwidth)
    assert obs.wavelength_range == [ifs_filter.low, ifs_filter.high]

    assert np.all(
        obs.SNR
        == ifs_observation_params["snr"][0] * np.ones_like(ifs_filter.wavelength.value)
    )

    assert obs.CRb_multiplier == ifs_observation_params["CRb_multiplier"]


# ============================================================================
# Tests for Observation.load_configuration - Error handling
# ============================================================================


def test_observation_load_configuration_invalid_observing_mode(
    basic_observation_params, basic_filter
):
    """Test that invalid observing mode raises KeyError with specific message."""
    obs = Observation()
    basic_observation_params["observing_mode"] = "Invalid"

    with pytest.raises(
        KeyError, match="Invalid observing mode. Must be 'IMAGER' or 'IFS'."
    ):
        obs.load_configuration(basic_observation_params, filter=basic_filter)


def test_observation_load_configuration_no_filter(basic_observation_params):
    """Test that not submitting a filter results in specific message."""
    obs = Observation()
    basic_observation_params["observing_mode"] = "IMAGER"

    with pytest.raises(
        ValueError, match="Please define a single filter for this simulation."
    ):
        obs.load_configuration(basic_observation_params)


# ============================================================================
# Tests for Observation.set_output_arrays
# ============================================================================


def test_observation_set_output_arrays(basic_observation_params, basic_filter):
    """Test that output arrays are initialized with correct shape and values."""
    obs = Observation()
    obs.load_configuration(basic_observation_params, basic_filter)
    obs.set_output_arrays()

    assert obs.exptime.shape == (1,)
    assert obs.fullsnr.shape == (1,)
    assert np.all(obs.exptime == 0.0 * TIME)
    assert np.all(obs.fullsnr == 0.0 * DIMENSIONLESS)


# ============================================================================
# Tests for Observation.validate_configuration
# ============================================================================


def test_observation_validate_configuration_valid(
    basic_observation_params, basic_filter
):
    """Test that validation passes with valid configuration."""
    obs = Observation()
    obs.load_configuration(basic_observation_params, basic_filter)

    # Should not raise any exception
    obs.validate_configuration()


def test_observation_validate_configuration_missing_attribute(
    basic_observation_params, basic_filter
):
    """Test that missing attribute raises AttributeError."""
    obs = Observation()
    obs.load_configuration(basic_observation_params, basic_filter)
    delattr(obs, "wavelength")

    with pytest.raises(AttributeError):
        obs.validate_configuration()


def test_observation_validate_configuration_invalid_type(
    basic_observation_params, basic_filter
):
    """Test that invalid attribute type raises TypeError."""
    obs = Observation()
    obs.load_configuration(basic_observation_params, basic_filter)
    obs.wavelength = "invalid"

    with pytest.raises(TypeError):
        obs.validate_configuration()


def test_observation_validate_configuration_incorrect_units(
    basic_observation_params, basic_filter
):
    """Test that incorrect units raise ValueError."""
    obs = Observation()
    obs.load_configuration(basic_observation_params, basic_filter)
    obs.wavelength = obs.wavelength.value * MAGNITUDE

    with pytest.raises(ValueError):
        obs.validate_configuration()
