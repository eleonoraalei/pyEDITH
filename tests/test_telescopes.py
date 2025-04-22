import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock
from pyEDITH.components.telescopes import ToyModelTelescope, EACTelescope
from pyEDITH.units import LENGTH, TIME, DIMENSIONLESS, TEMPERATURE, WAVELENGTH
from pyEDITH.utils import average_over_bandpass, interpolate_over_bandpass
from copy import deepcopy


class MockMediator:
    def __init__(self, observing_mode="IMAGER"):
        self.observing_mode = observing_mode

    def get_observation_parameter(self, param):
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.1]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.7]) * WAVELENGTH
        else:
            return 1.0

    def get_coronagraph_parameter(self, param):
        if param == "bandwidth":
            return 0.2 * DIMENSIONLESS
        return 1.0


@pytest.fixture
def mock_telescope_params():
    class MockTelescope:
        def __init__(self):
            self.diam_circ = 8.0
            self.lam = u.Quantity([0.2, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5] * WAVELENGTH)
            self.total_tele_refl = np.array([0.9, 0.8, 0.9, 0.98, 0.75, 0.55, 0.34])

    return MockTelescope()


def test_toy_model_telescope_init():
    telescope = ToyModelTelescope()
    assert telescope.path is None
    assert telescope.keyword is None


def test_toy_model_telescope_load_configuration():
    telescope = ToyModelTelescope()
    parameters = {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "telescope_throughput": [0.85],
    }
    mediator = MockMediator()

    telescope.load_configuration(parameters, mediator)

    assert telescope.diameter == 8.0 * LENGTH
    assert telescope.unobscured_area == 0.9
    assert telescope.toverhead_fixed == 9000 * TIME
    assert telescope.toverhead_multi == 1.2 * DIMENSIONLESS
    assert np.all(telescope.telescope_throughput == [0.85] * DIMENSIONLESS)
    assert telescope.temperature == 290 * TEMPERATURE  # Test default
    assert telescope.Tcontam == 0.95 * DIMENSIONLESS  # Test default
    assert np.isclose(telescope.Area.value, 45.2389, rtol=1e-4)
    assert telescope.Area.unit == LENGTH**2


@pytest.mark.parametrize("observing_mode", ["IMAGER", "IFS"])
def test_eac_telescope_load_configuration(
    mock_telescope_params,
    observing_mode,
):
    with patch("eacy.load_telescope", return_value=deepcopy(mock_telescope_params)):
        telescope = EACTelescope(keyword="EAC1")
        parameters = {"observing_mode": observing_mode}
        mediator = MockMediator(observing_mode)

        telescope.load_configuration(parameters, mediator)

        assert telescope.diameter == 8.0 * LENGTH
        assert telescope.unobscured_area == 1.0
        assert telescope.toverhead_fixed == 8.25e3 * TIME
        assert telescope.toverhead_multi == 1.1 * DIMENSIONLESS
        assert telescope.temperature == 290 * TEMPERATURE
        assert telescope.Tcontam == 1.0 * DIMENSIONLESS

        if observing_mode == "IFS":
            wavelengths = mediator.get_observation_parameter("wavelength")
            expected_throughput = interpolate_over_bandpass(
                {
                    "lam": mock_telescope_params.lam,
                    "total_tele_refl": mock_telescope_params.total_tele_refl.copy(),
                },
                wavelengths,
            )["total_tele_refl"]
            print(expected_throughput)
            assert np.allclose(
                telescope.telescope_throughput.value, expected_throughput, rtol=1e-5
            )
        elif observing_mode == "IMAGER":
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
                telescope.telescope_throughput[0].value, expected_throughput, rtol=1e-5
            )

        assert np.isclose(telescope.Area.value, 50.2655, rtol=1e-4)
        assert telescope.Area.unit == LENGTH**2


def test_eac_detector_load_configuration_invalid():

    detector = EACTelescope()
    parameters = {"observing_mode": "INVALID"}
    mediator = MockMediator("IMAGER")

    with pytest.raises(KeyError, match="Unsupported observing mode: INVALID"):
        detector.load_configuration(parameters, mediator)


def test_telescope_validate_configuration():
    telescope = ToyModelTelescope()
    parameters = {
        "diameter": 8.0,
        "unobscured_area": 0.9,
        "toverhead_fixed": 9000,
        "toverhead_multi": 1.2,
        "telescope_throughput": [0.85],
        "temperature": 280,
        "Tcontam": 0.98,
    }
    mediator = MockMediator()
    telescope.load_configuration(parameters, mediator)

    # This should not raise any exception
    telescope.validate_configuration()

    # Test missing attribute
    delattr(telescope, "diameter")
    with pytest.raises(
        AttributeError, match="Telescope is missing attribute: diameter"
    ):
        telescope.validate_configuration()

    # Restore the attribute and test incorrect type
    setattr(telescope, "diameter", 8.0)  # Not a Quantity
    with pytest.raises(
        TypeError, match="Telescope attribute diameter should be a Quantity"
    ):
        telescope.validate_configuration()

    # Test incorrect units
    setattr(telescope, "diameter", 8.0 * u.s)
    with pytest.raises(
        ValueError, match="Telescope attribute diameter has incorrect units"
    ):
        telescope.validate_configuration()
