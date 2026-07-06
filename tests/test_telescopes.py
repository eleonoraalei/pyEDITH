import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock
from pyEDITH.components.telescopes import ToyModelTelescope, EACTelescope
from pyEDITH.units import LENGTH, TIME, DIMENSIONLESS, TEMPERATURE, WAVELENGTH
from pyEDITH.utils import average_over_bandpass, interpolate_over_bandpass
from copy import deepcopy

# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator:
    """Mock mediator for testing telescope configurations."""

    def __init__(self, observing_mode="IMAGER"):
        self.observing_mode = observing_mode

    def get_observation_parameter(self, param):
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.1]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.7]) * WAVELENGTH
        if param == "delta_wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5 / 140, 0.7 / 140, 1.1 / 140]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.7 / 140]) * WAVELENGTH
        return 1.0

    def get_coronagraph_parameter(self, param):
        if param == "bandwidth":
            return 0.2 * DIMENSIONLESS
        return 1.0


@pytest.fixture
def mock_telescope_params():
    """Fixture providing mock telescope parameters from EAC."""

    class MockTelescope:
        def __init__(self):
            self.diam_circ = 8.0
            self.lam = u.Quantity([0.2, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5] * WAVELENGTH)
            self.total_tele_refl = np.array([0.9, 0.8, 0.9, 0.98, 0.75, 0.55, 0.34])

    return MockTelescope()


@pytest.fixture
def toy_telescope_parameters():
    """Fixture providing standard parameters for ToyModelTelescope testing."""
    return {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "telescope_optical_throughput": [0.85],
    }


@pytest.fixture
def full_toy_telescope_parameters():
    """Fixture providing complete parameters for ToyModelTelescope testing."""
    return {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "telescope_optical_throughput": [0.85],
        "temperature": 280,
        "T_contamination": 0.98,
    }


# ============================================================================
# Tests for ToyModelTelescope initialization
# ============================================================================


def test_toy_model_telescope_init():
    """Test that ToyModelTelescope initializes with None values."""
    telescope = ToyModelTelescope()

    assert telescope.path is None
    assert telescope.keyword is None


# ============================================================================
# Tests for ToyModelTelescope.load_configuration - Basic parameters
# ============================================================================


def test_toy_model_telescope_load_configuration_user_params(toy_telescope_parameters):
    """Test loading ToyModelTelescope configuration with user parameters."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(toy_telescope_parameters, mediator)

    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 0.9
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS
    assert np.all(telescope.telescope_optical_throughput == [0.85] * DIMENSIONLESS)


def test_toy_model_telescope_load_configuration_default_temperature(
    toy_telescope_parameters,
):
    """Test that default temperature is used when not provided."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(toy_telescope_parameters, mediator)

    assert telescope.temperature == 290 * TEMPERATURE


def test_toy_model_telescope_load_configuration_default_contamination(
    toy_telescope_parameters,
):
    """Test that default contamination factor is used when not provided."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(toy_telescope_parameters, mediator)

    assert telescope.T_contamination == 0.95 * DIMENSIONLESS


def test_toy_model_telescope_load_configuration_calculated_area(
    toy_telescope_parameters,
):
    """Test that telescope area is calculated correctly from diameter and obscuration."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(toy_telescope_parameters, mediator)

    assert np.isclose(telescope.Area.value, 45.2389, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for EACTelescope.load_configuration - IMAGER mode
# ============================================================================


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_imager_basic(
    mock_load_telescope, mock_telescope_params
):
    """Test basic EACTelescope configuration loading in IMAGER mode."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_imager_throughput_averaging(
    mock_load_telescope, mock_telescope_params
):
    """Test that telescope throughput is averaged over bandpass in IMAGER mode."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    telescope.load_configuration(parameters, mediator)

    wavelength = mediator.get_observation_parameter("wavelength")
    bandwidth = mediator.get_coronagraph_parameter("bandwidth")
    wavelength_range = [
        wavelength * (1 - 0.5 * bandwidth),
        wavelength * (1 + 0.5 * bandwidth),
    ]
    expected_throughput = average_over_bandpass(
        {
            "lam": mock_telescope_params.lam,
            "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
        },
        wavelength_range,
    )["total_tele_refl"]

    assert np.isclose(
        telescope.telescope_optical_throughput[0].value,
        expected_throughput,
        rtol=1e-5,
    )


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_imager_calculated_area(
    mock_load_telescope, mock_telescope_params
):
    """Test that telescope area is calculated correctly in IMAGER mode."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    telescope.load_configuration(parameters, mediator)

    assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for EACTelescope.load_configuration - IFS mode
# ============================================================================


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_ifs_basic(
    mock_load_telescope, mock_telescope_params
):
    """Test basic EACTelescope configuration loading in IFS mode."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 1.0
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 1.0 * DIMENSIONLESS


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_ifs_throughput_interpolation(
    mock_load_telescope, mock_telescope_params
):
    """Test that telescope throughput is interpolated onto wavelength grid in IFS mode."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    telescope.load_configuration(parameters, mediator)

    wavelengths = mediator.get_observation_parameter("wavelength")
    expected_throughput = interpolate_over_bandpass(
        {
            "lam": mock_telescope_params.lam,
            "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
        },
        wavelengths,
    )["total_tele_refl"]

    assert np.allclose(
        telescope.telescope_optical_throughput.value,
        expected_throughput,
        rtol=1e-5,
    )


@patch("eacy.load_telescope")
def test_eac_telescope_load_configuration_ifs_calculated_area(
    mock_load_telescope, mock_telescope_params
):
    """Test that telescope area is calculated correctly in IFS mode."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    telescope.load_configuration(parameters, mediator)

    assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


# ============================================================================
# Tests for EACTelescope.load_configuration - Error handling
# ============================================================================


def test_eac_telescope_load_configuration_invalid_mode():
    """Test that invalid observing mode raises KeyError."""
    telescope = EACTelescope()
    parameters = {"observing_mode": "INVALID"}
    mediator = MockMediator("IMAGER")

    with pytest.raises(KeyError, match="Unsupported observing mode: INVALID"):
        telescope.load_configuration(parameters, mediator)


# ============================================================================
# Tests for Telescope.validate_configuration
# ============================================================================


def test_telescope_validate_configuration_all_valid(full_toy_telescope_parameters):
    """Test that validation passes with all correct attributes."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_toy_telescope_parameters, mediator)

    # Should not raise
    telescope.validate_configuration()


def test_telescope_validate_configuration_missing_diameter(
    full_toy_telescope_parameters,
):
    """Test that missing diameter attribute raises AttributeError."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_toy_telescope_parameters, mediator)
    delattr(telescope, "diameter")

    with pytest.raises(
        AttributeError, match="Telescope is missing attribute: diameter"
    ):
        telescope.validate_configuration()


def test_telescope_validate_configuration_diameter_not_quantity(
    full_toy_telescope_parameters,
):
    """Test that non-Quantity diameter raises TypeError."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_toy_telescope_parameters, mediator)
    telescope.diameter = 8.0  # Not a Quantity

    with pytest.raises(
        TypeError, match="Telescope attribute diameter should be a Quantity"
    ):
        telescope.validate_configuration()


def test_telescope_validate_configuration_incorrect_diameter_units(
    full_toy_telescope_parameters,
):
    """Test that diameter with incorrect units raises ValueError."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration(full_toy_telescope_parameters, mediator)
    telescope.diameter = 8.0 * u.s  # Wrong unit

    with pytest.raises(
        ValueError, match="Telescope attribute diameter has incorrect units"
    ):
        telescope.validate_configuration()


# ============================================================================
# Tests for array parameter conversion
# ============================================================================


def test_toy_model_telescope_throughput_array_conversion():
    """Test that telescope throughput is converted to numpy array."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {
        "diameter": 8.0,
        "telescope_optical_throughput": [0.85, 0.90],  # List input
    }

    telescope.load_configuration(parameters, mediator)

    assert isinstance(telescope.telescope_optical_throughput.value, np.ndarray)
    assert len(telescope.telescope_optical_throughput) == 2


# ============================================================================
# Tests for derived parameter calculations
# ============================================================================


def test_toy_model_telescope_area_with_no_obscuration():
    """Test area calculation with no obscuration (unobscured_area = 1.0)."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {
        "diameter": 10.0,
        "unobscured_area": 1.0,
    }

    telescope.load_configuration(parameters, mediator)

    expected_area = np.pi * (10.0**2) / 4.0
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-6)


def test_toy_model_telescope_area_with_partial_obscuration():
    """Test area calculation with partial obscuration."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    parameters = {
        "diameter": 10.0,
        "unobscured_area": 0.8,
    }

    telescope.load_configuration(parameters, mediator)

    expected_area = np.pi * (10.0**2) / 4.0 * 0.8
    assert np.isclose(telescope.Area.value, expected_area, rtol=1e-6)


# ============================================================================
# Tests for default values
# ============================================================================


def test_toy_model_telescope_all_defaults():
    """Test that all default values are correctly applied."""
    telescope = ToyModelTelescope()
    mediator = MockMediator()

    telescope.load_configuration({}, mediator)

    assert telescope.diameter == 7.87 * LENGTH
    assert telescope.unobscured_area == (1.0 - 0.121)
    assert telescope.toverhead_fixed == 8.25e3 * TIME
    assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
    assert telescope.temperature == 290 * TEMPERATURE
    assert telescope.T_contamination == 0.95 * DIMENSIONLESS


@patch("eacy.load_telescope")
def test_eac_telescope_default_contamination(
    mock_load_telescope, mock_telescope_params
):
    """Test that EACTelescope uses default contamination factor."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    telescope.load_configuration(parameters, mediator)

    assert telescope.T_contamination == 1.0 * DIMENSIONLESS


@patch("eacy.load_telescope")
def test_eac_telescope_default_temperature(mock_load_telescope, mock_telescope_params):
    """Test that EACTelescope uses default temperature."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    telescope.load_configuration(parameters, mediator)

    assert telescope.temperature == 290 * TEMPERATURE


# ============================================================================
# Tests for throughput shape consistency
# ============================================================================


@patch("eacy.load_telescope")
def test_eac_telescope_throughput_shape_imager(
    mock_load_telescope, mock_telescope_params
):
    """Test that throughput has correct shape in IMAGER mode (scalar)."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IMAGER"}
    mediator = MockMediator("IMAGER")

    telescope.load_configuration(parameters, mediator)

    assert telescope.telescope_optical_throughput.shape == (1,)


@patch("eacy.load_telescope")
def test_eac_telescope_throughput_shape_ifs(mock_load_telescope, mock_telescope_params):
    """Test that throughput has correct shape in IFS mode (matches wavelength array)."""
    mock_load_telescope.return_value = deepcopy(mock_telescope_params)

    telescope = EACTelescope(keyword="EAC1")
    parameters = {"observing_mode": "IFS"}
    mediator = MockMediator("IFS")

    telescope.load_configuration(parameters, mediator)

    wavelengths = mediator.get_observation_parameter("wavelength")
    assert telescope.telescope_optical_throughput.shape == wavelengths.shape
