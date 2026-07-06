import pytest
import numpy as np
from astropy import units as u
import logging

from pyEDITH.observation import Observation
from pyEDITH.units import WAVELENGTH, DIMENSIONLESS, LAMBDA_D, TIME, MAGNITUDE

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_observation_params():
    """Fixture providing basic observation parameters."""
    return {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }


@pytest.fixture
def ifs_observation_params():
    """Fixture providing IFS mode observation parameters."""
    return {
        "wavelength": np.linspace(0.2, 1.8, 1000),
        "snr": [7.0] * np.ones(1000),
        "spectral_resolution": [140, 40],
        "lam_low": [0.5, 1.0],
        "lam_high": [1.0, 1.7],
        "regrid_wavelength": True,
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
    }


@pytest.fixture
def single_wavelength_params():
    """Fixture providing single wavelength observation parameters."""
    return {
        "wavelength": [0.5],
        "snr": [7.0],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }


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


def test_observation_load_configuration_basic(basic_observation_params):
    """Test loading basic observation configuration."""
    obs = Observation()
    obs.load_configuration(basic_observation_params)

    assert np.all(obs.wavelength == basic_observation_params["wavelength"] * WAVELENGTH)
    assert np.all(obs.SNR == basic_observation_params["snr"] * DIMENSIONLESS)
    assert obs.CRb_multiplier == basic_observation_params["CRb_multiplier"]


def test_observation_load_configuration_single_wavelength(single_wavelength_params):
    """Test loading configuration with single wavelength."""
    obs = Observation()
    obs.load_configuration(single_wavelength_params)

    assert len(obs.wavelength) == 1
    assert len(obs.SNR) == 1


# ============================================================================
# Tests for Observation.load_configuration - IFS mode
# ============================================================================


def test_observation_load_configuration_ifs_mode(ifs_observation_params):
    """Test loading IFS mode configuration with spectral channels."""
    obs = Observation()
    obs.load_configuration(ifs_observation_params)

    # Check that spectral grid has correct resolution for each channel
    channel_1_mask = obs.wavelength.value < ifs_observation_params["lam_high"][0]
    channel_2_mask = obs.wavelength.value >= ifs_observation_params["lam_high"][1]

    assert np.all(
        obs.wavelength[channel_1_mask] / obs.delta_wavelength[channel_1_mask]
        == ifs_observation_params["spectral_resolution"][0]
    )
    assert np.all(
        obs.wavelength[channel_2_mask] / obs.delta_wavelength[channel_2_mask]
        == ifs_observation_params["spectral_resolution"][1]
    )
    assert len(obs.SNR) == len(obs.wavelength)
    assert len(obs.SNR) != len(ifs_observation_params["wavelength"])
    assert obs.SNR.unit == DIMENSIONLESS


def test_observation_load_configuration_ifs_missing_spectral_resolution():
    """Test that IFS mode without spectral_resolution raises KeyError."""
    obs = Observation()
    params = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": [7.0, 7.0, 7.0],
        "lam_low": [0.5, 1.0],
        "lam_high": [1.0, 1.7],
        "regrid_wavelength": True,
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
    }

    with pytest.raises(KeyError):
        obs.load_configuration(params)


def test_observation_load_configuration_ifs_missing_lam_low():
    """Test that IFS mode without lam_low raises KeyError."""
    obs = Observation()
    params = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": [7.0, 7.0, 7.0],
        "lam_high": [1.0, 1.7],
        "spectral_resolution": [140, 40],
        "regrid_wavelength": True,
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
    }

    with pytest.raises(KeyError):
        obs.load_configuration(params)


def test_observation_load_configuration_ifs_missing_lam_high():
    """Test that IFS mode without lam_high raises KeyError."""
    obs = Observation()
    params = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": [7.0, 7.0, 7.0],
        "lam_low": [0.5, 1.0],
        "spectral_resolution": [140, 40],
        "regrid_wavelength": True,
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
    }

    with pytest.raises(KeyError):
        obs.load_configuration(params)


# ============================================================================
# Tests for Observation.load_configuration - Warning handling
# ============================================================================


def test_observation_load_configuration_invalid_wavelength_grid(caplog):
    """Test that invalid wavelength grid triggers warning and uses default resolution."""
    obs = Observation()
    params = {
        "observing_mode": "IFS",
        "wavelength": [0.5, 0.5, 0.5],
        "snr": [7.0, 7.0, 7.0],
        "regrid_wavelength": False,
        "CRb_multiplier": 2.0,
    }

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        obs.load_configuration(params)

    assert any(
        "Wavelength grid is not valid. Using default spectral resolution of 140."
        in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


# ============================================================================
# Tests for Observation.load_configuration - Error handling
# ============================================================================


def test_observation_load_configuration_missing_aperture_params():
    """Test that missing photometric_aperture_radius and psf_trunc_ratio raises KeyError."""
    obs = Observation()
    params = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "CRb_multiplier": 2.0,
    }

    with pytest.raises(KeyError):
        obs.load_configuration(params)


def test_observation_load_configuration_invalid_key():
    """Test that invalid configuration key raises KeyError."""
    obs = Observation()

    with pytest.raises(KeyError):
        obs.load_configuration({"invalid_key": 0})


def test_observation_load_configuration_invalid_observing_mode():
    """Test that invalid observing mode raises KeyError with specific message."""
    obs = Observation()

    with pytest.raises(
        KeyError, match="Invalid observing mode. Must be 'IMAGER' or 'IFS'."
    ):
        obs.load_configuration({"observing_mode": "Invalid"})


# ============================================================================
# Tests for Observation.set_output_arrays
# ============================================================================


def test_observation_set_output_arrays(basic_observation_params):
    """Test that output arrays are initialized with correct shape and values."""
    obs = Observation()
    obs.load_configuration(basic_observation_params)
    obs.set_output_arrays()

    assert obs.exptime.shape == (3,)
    assert obs.fullsnr.shape == (3,)
    assert np.all(obs.exptime == 0.0 * TIME)
    assert np.all(obs.fullsnr == 0.0 * DIMENSIONLESS)


# ============================================================================
# Tests for Observation.validate_configuration
# ============================================================================


def test_observation_validate_configuration_valid(basic_observation_params):
    """Test that validation passes with valid configuration."""
    obs = Observation()
    obs.load_configuration(basic_observation_params)

    # Should not raise any exception
    obs.validate_configuration()


def test_observation_validate_configuration_missing_attribute(basic_observation_params):
    """Test that missing attribute raises AttributeError."""
    obs = Observation()
    obs.load_configuration(basic_observation_params)
    delattr(obs, "wavelength")

    with pytest.raises(AttributeError):
        obs.validate_configuration()


def test_observation_validate_configuration_invalid_type(basic_observation_params):
    """Test that invalid attribute type raises TypeError."""
    obs = Observation()
    obs.load_configuration(basic_observation_params)
    obs.wavelength = "invalid"

    with pytest.raises(TypeError):
        obs.validate_configuration()


def test_observation_validate_configuration_incorrect_units(basic_observation_params):
    """Test that incorrect units raise ValueError."""
    obs = Observation()
    obs.load_configuration(basic_observation_params)
    obs.wavelength = obs.wavelength.value * MAGNITUDE

    with pytest.raises(ValueError):
        obs.validate_configuration()
