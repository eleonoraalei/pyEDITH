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
# Check filters are within input lambda (LEGACY HELPER)
# ============================================================================


def test_generate_wavelength_grid_lower_boundary_outside_range(ifs_observation_params):
    """Test that lower boundary outside input range raises AssertionError."""
    ifs_observation_params["wavelength"] = np.linspace(0.8, 2.0, 1000)

    with pytest.raises(
        AssertionError,
        match="Your minimum input wavelength is greater than first channel lower boundary.",
    ):
        obs = Observation()
        obs.load_configuration(ifs_observation_params)


def test_generate_wavelength_grid_upper_boundary_outside_range(ifs_observation_params):
    """Test that upper boundary outside input range raises AssertionError."""
    ifs_observation_params["wavelength"] = np.linspace(0.1, 1.5, 1000)

    with pytest.raises(
        AssertionError,
        match="Your maximum input wavelength is less than last channel upper boundary.",
    ):
        obs = Observation()
        obs.load_configuration(ifs_observation_params)


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
    channel_2_mask = obs.wavelength.value >= ifs_observation_params["lam_low"][1]

    np.testing.assert_allclose(
        obs.wavelength[channel_1_mask].value
        / obs.delta_wavelength[channel_1_mask].value,
        ifs_observation_params["spectral_resolution"][0],
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        obs.wavelength[channel_2_mask].value
        / obs.delta_wavelength[channel_2_mask].value,
        ifs_observation_params["spectral_resolution"][1],
        rtol=1e-5,
    )
    assert len(obs.SNR) == len(obs.wavelength)
    assert len(obs.SNR) != len(ifs_observation_params["wavelength"])
    assert obs.SNR.unit == DIMENSIONLESS


def test_observation_load_configuration_ifs_missing_spectral_resolution():
    """Test that IFS mode without spectral_resolution raises KeyError."""
    obs = Observation()
    params = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": 7.0,
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
        "snr": 7.0,
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
        "snr": 7.0,
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
        obs.load_configuration({"wavelength": [0.5, 0.55, 0.6], "invalid_key": 0})


def test_observation_load_configuration_invalid_observing_mode():
    """Test that invalid observing mode raises KeyError with specific message."""
    obs = Observation()

    with pytest.raises(
        KeyError, match="Invalid observing mode. Must be 'IMAGER' or 'IFS'."
    ):
        obs.load_configuration(
            {"wavelength": [0.5, 0.55, 0.6], "observing_mode": "Invalid"}
        )


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


# ============================================================================
# Tests for Observation.load_configuration - Filter handling
# ============================================================================


class MockFilter:
    """Mock filter class for testing."""

    def __init__(self, name, low, high, wavelength, delta_wavelength, resolution=None):
        self.name = name
        self.low = low * WAVELENGTH
        self.high = high * WAVELENGTH
        self.wavelength = wavelength * WAVELENGTH
        self.delta_wavelength = delta_wavelength * WAVELENGTH
        self.resolution = resolution


@pytest.fixture
def template_filters():
    """Fixture providing observation parameters with filters."""
    filter1 = MockFilter(
        name="Filter1",
        low=0.5,
        high=0.7,
        wavelength=np.array([0.5, 0.6, 0.7]),
        delta_wavelength=np.array([0.05, 0.05, 0.05]),
    )
    filter2 = MockFilter(
        name="Filter2",
        low=0.8,
        high=1.0,
        wavelength=np.array([0.8, 0.9, 1.0]),
        delta_wavelength=np.array([0.05, 0.05, 0.05]),
    )
    return [filter1, filter2]


def test_observation_load_configuration_with_valid_filters(template_filters):
    """Test loading configuration with valid filters that fit within wavelength range."""

    filter_observation_params = {
        "wavelength": np.linspace(0.4, 1.2, 100),
        "snr": [7.0] * 100,
        "filter_list": template_filters,
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    obs.load_configuration(filter_observation_params)

    # Both filters should be active
    expected_wavelength = np.concatenate(
        [
            filter_observation_params["filter_list"][0].wavelength,
            filter_observation_params["filter_list"][1].wavelength,
        ]
    )
    expected_delta = np.concatenate(
        [
            filter_observation_params["filter_list"][0].delta_wavelength,
            filter_observation_params["filter_list"][1].delta_wavelength,
        ]
    )

    np.testing.assert_array_equal(obs.wavelength, expected_wavelength)
    np.testing.assert_array_equal(obs.delta_wavelength, expected_delta)
    assert obs.nlambda == 6


def test_observation_load_configuration_filter_outside_lower_bound(
    template_filters, caplog
):
    """Test that filter below wavelength range is discarded with warning."""
    # Set wavelength range that excludes first filter

    filter_observation_params = {
        "wavelength": np.linspace(0.75, 1.2, 100),
        "snr": [7.0] * 100,
        "filter_list": template_filters,
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }
    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(filter_observation_params)

    # Check warning was logged
    assert any(
        "Filter Filter1 discarded" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )

    # Only second filter should be active
    expected_wavelength = filter_observation_params["filter_list"][1].wavelength
    np.testing.assert_array_equal(obs.wavelength, expected_wavelength)
    assert obs.nlambda == 3


def test_observation_load_configuration_filter_outside_upper_bound(
    template_filters, caplog
):
    """Test that filter above wavelength range is discarded with warning."""
    # Set wavelength range that excludes second filter

    filter_observation_params = {
        "wavelength": np.linspace(0.4, 0.75, 100),
        "snr": [7.0] * 100,
        "filter_list": template_filters,
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(filter_observation_params)

    # Check warning was logged
    assert any(
        "Filter Filter2 discarded" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )

    # Only first filter should be active
    expected_wavelength = filter_observation_params["filter_list"][0].wavelength
    np.testing.assert_array_equal(obs.wavelength, expected_wavelength)
    assert obs.nlambda == 3


def test_observation_load_configuration_filter_partially_overlapping(
    template_filters, caplog
):
    """Test that partially overlapping filter is discarded."""
    # Set wavelength range that only partially covers first filter

    filter_observation_params = {
        "wavelength": np.linspace(0.6, 1.2, 100),
        "snr": [7.0] * 100,
        "filter_list": template_filters,
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(filter_observation_params)

    # Check warning was logged for first filter
    assert any(
        "Filter Filter1 discarded" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )

    # Only second filter should be active
    expected_wavelength = filter_observation_params["filter_list"][1].wavelength
    np.testing.assert_array_equal(obs.wavelength, expected_wavelength)


def test_observation_load_configuration_no_filters_remain(template_filters):
    """Test that ValueError is raised when all filters are outside wavelength range."""
    # Set wavelength range that excludes all filters

    filter_observation_params = {
        "wavelength": np.linspace(1.5, 2.0, 100),
        "snr": [7.0] * 100,
        "filter_list": template_filters,
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    with pytest.raises(
        ValueError,
        match="No filters remain after filtering. Specify different filters or change spectrum.",
    ):
        obs.load_configuration(filter_observation_params)


def test_observation_load_configuration_single_valid_filter():
    """Test configuration with only one valid filter."""
    single_filter = MockFilter(
        name="SingleFilter",
        low=0.5,
        high=0.7,
        wavelength=np.array([0.5, 0.6, 0.7]),
        delta_wavelength=np.array([0.05, 0.05, 0.05]),
    )

    params = {
        "wavelength": np.linspace(0.4, 0.8, 100),
        "snr": [7.0] * 100,
        "filter_list": [single_filter],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    obs.load_configuration(params)

    np.testing.assert_array_equal(obs.wavelength, single_filter.wavelength)
    np.testing.assert_array_equal(obs.delta_wavelength, single_filter.delta_wavelength)
    assert obs.nlambda == 3


def test_observation_load_configuration_empty_filter_list():
    """Test that empty filter list falls back to legacy helper."""
    params = {
        "wavelength": np.array([0.5, 0.6, 0.7]),
        "snr": [7.0, 7.0, 7.0],
        "filter_list": [],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    with pytest.raises(ValueError, match="No filters remain after filtering"):
        obs.load_configuration(params)


def test_observation_load_configuration_no_filter_list_uses_legacy():
    """Test that missing filter_list key uses legacy helper path."""
    params = {
        "wavelength": np.array([0.5, 0.6, 0.7]),
        "snr": [7.0, 7.0, 7.0],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    obs.load_configuration(params)

    # Should use legacy helper - wavelength should be directly assigned
    expected_wavelength = params["wavelength"] * WAVELENGTH
    np.testing.assert_array_equal(obs.wavelength, expected_wavelength)
    assert obs.delta_wavelength is None  # IMAGER mode has no delta_wavelength


def test_observation_load_configuration_filter_exact_boundaries(
    template_filters,
):
    """Test filters with wavelength range exactly matching filter boundaries."""
    # Set wavelength range to exactly match filter boundaries

    filter_observation_params = {
        "wavelength": np.array([0.5, 1.0]),
        "snr": [7.0] * 2,
        "filter_list": template_filters,
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    obs.load_configuration(filter_observation_params)

    # Both filters should be active (boundaries are inclusive)
    assert obs.nlambda == 6


def test_observation_load_configuration_multiple_filters_concatenation_order():
    """Test that multiple filters are concatenated in correct order."""

    filter1 = MockFilter(
        name="Filter1",
        low=0.5,
        high=0.6,
        wavelength=np.array([0.5, 0.55, 0.6]),
        delta_wavelength=np.array([0.025, 0.025, 0.025]),
    )
    filter2 = MockFilter(
        name="Filter2",
        low=0.7,
        high=0.8,
        wavelength=np.array([0.7, 0.75, 0.8]),
        delta_wavelength=np.array([0.025, 0.025, 0.025]),
    )
    filter3 = MockFilter(
        name="Filter3",
        low=0.9,
        high=1.0,
        wavelength=np.array([0.9, 0.95, 1.0]),
        delta_wavelength=np.array([0.025, 0.025, 0.025]),
    )

    params = {
        "wavelength": np.linspace(0.4, 1.2, 100),
        "snr": [7.0] * 100,
        "filter_list": [filter1, filter2, filter3],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    obs.load_configuration(params)

    # Check concatenation order
    expected_wavelength = np.concatenate(
        [filter1.wavelength, filter2.wavelength, filter3.wavelength]
    )
    np.testing.assert_array_equal(obs.wavelength, expected_wavelength)
    assert obs.nlambda == 9


def test_observation_load_configuration_filter_low_resolution_warning_ifs(caplog):
    """Test warning when input spectrum resolution is lower than filter resolution in IFS mode."""
    # Create filter with high resolution (R~100)
    high_res_filter = MockFilter(
        name="HighResFilter",
        low=0.5,
        high=0.7,
        wavelength=np.linspace(0.5, 0.7, 200),  # High resolution
        delta_wavelength=np.full(200, 0.001),
        resolution=600,  # R ~ 600
    )

    # Create low resolution input spectrum (R~20)
    low_res_wavelength = np.linspace(0.4, 0.8, 20)  # Low resolution

    params = {
        "wavelength": low_res_wavelength,
        "snr": [7.0] * 20,
        "filter_list": [high_res_filter],
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
        "regrid_wavelength": False,
    }

    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(params)

    # Check warning was logged
    assert any(
        "Input spectrum resolution" in record.message
        and "lower than" in record.message
        and "filter resolution" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_observation_load_configuration_filter_adequate_resolution_no_warning_ifs(
    caplog,
):
    """Test no warning when input spectrum resolution is adequate for filter in IFS mode."""
    # Create filter with resolution R~50
    filter_obj = MockFilter(
        name="MedResFilter",
        low=0.5,
        high=0.7,
        wavelength=np.linspace(0.5, 0.7, 100),
        delta_wavelength=np.full(100, 0.002),
        resolution=300,  # R ~ 300
    )

    # Create high resolution input spectrum (R~400)
    high_res_wavelength = np.linspace(0.4, 0.8, 500)

    params = {
        "wavelength": high_res_wavelength,
        "snr": [7.0] * 500,
        "filter_list": [filter_obj],
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
        "regrid_wavelength": False,
    }

    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(params)

    # Check no resolution warning was logged
    resolution_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Input spectrum resolution" in record.message
        and "lower than" in record.message
    ]
    assert len(resolution_warnings) == 0


def test_observation_load_configuration_filter_low_resolution_no_warning_imager(caplog):
    """Test no warning for low resolution in IMAGER mode."""
    # Create filter
    filter_obj = MockFilter(
        name="ImageFilter",
        low=0.5,
        high=0.7,
        wavelength=np.linspace(0.5, 0.7, 200),
        delta_wavelength=np.full(200, 0.001),
    )

    # Create low resolution input spectrum
    low_res_wavelength = np.linspace(0.4, 0.8, 20)

    params = {
        "wavelength": low_res_wavelength,
        "snr": [7.0] * 20,
        "filter_list": [filter_obj],
        "CRb_multiplier": 2.0,
        "observing_mode": "IMAGER",
    }

    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(params)

    # Check no resolution warning was logged (only relevant for IFS)
    resolution_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Input spectrum resolution" in record.message
    ]
    assert len(resolution_warnings) == 0


def test_observation_load_configuration_filter_resolution_warning_multiple_filters(
    caplog,
):
    """Test resolution warnings for multiple filters with different resolutions."""
    # Filter 1: Low resolution - should not trigger warning
    low_res_filter = MockFilter(
        name="LowResFilter",
        low=0.5,
        high=0.6,
        wavelength=np.linspace(0.5, 0.6, 20),
        delta_wavelength=np.full(20, 0.005),
        resolution=110,  # R ~ 110
    )

    # Filter 2: High resolution - should trigger warning
    high_res_filter = MockFilter(
        name="HighResFilter",
        low=0.7,
        high=0.8,
        wavelength=np.linspace(0.7, 0.8, 200),
        delta_wavelength=np.full(200, 0.0005),
        resolution=1500,  # R ~ 1500
    )

    # Medium resolution input spectrum
    input_wavelength = np.linspace(0.4, 0.9, 150)

    params = {
        "wavelength": input_wavelength,
        "snr": [7.0] * 150,
        "filter_list": [low_res_filter, high_res_filter],
        "CRb_multiplier": 2.0,
        "observing_mode": "IFS",
        "regrid_wavelength": False,
    }

    obs = Observation()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        obs.load_configuration(params)

    # Check warning was logged only for high resolution filter
    resolution_warnings = [
        record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Input spectrum resolution" in record.message
    ]

    assert len(resolution_warnings) == 1
    assert "HighResFilter" in resolution_warnings[0]
    assert "LowResFilter" not in resolution_warnings[0]
