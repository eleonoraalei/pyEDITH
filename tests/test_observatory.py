import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from astropy import units as u
from pyEDITH.observatory import Observatory, ObservatoryMediator
from pyEDITH.components.telescopes import Telescope
from pyEDITH.components.detectors import Detector
from pyEDITH.components.coronagraphs import Coronagraph
from pyEDITH.observation import Observation
from pyEDITH.astrophysical_scene import AstrophysicalScene
from pyEDITH.units import (
    LENGTH,
    DIMENSIONLESS,
    WAVELENGTH,
    PHOTON_FLUX_DENSITY,
    LAMBDA_D,
)


class MockTelescope(Telescope):
    def load_configuration(self, parameters, mediator):
        self.diameter = 8.0 * LENGTH
        self.telescope_throughput = [0.9] * DIMENSIONLESS


class MockDetector(Detector):
    def load_configuration(self, parameters, mediator):
        self.pixscale_mas = 10 * u.mas
        self.QE = [0.9] * u.electron / u.photon
        self.QE = [0.9] * DIMENSIONLESS


class MockCoronagraph(Coronagraph):
    def load_configuration(self, parameters, mediator):
        self.minimum_IWA = 2 * LAMBDA_D
        self.maximum_OWA = 10 * LAMBDA_D


@pytest.fixture
def mock_observatory():
    obs = Observatory()
    obs.telescope = MockTelescope()
    obs.detector = MockDetector()
    obs.coronagraph = MockCoronagraph()
    return obs


@pytest.fixture
def mock_observation():
    obs = Observation()
    obs.wavelength = [0.5] * WAVELENGTH
    return obs


@pytest.fixture
def mock_scene():
    scene = AstrophysicalScene()
    scene.Fstar = [1e-8] * DIMENSIONLESS
    return scene


def test_observatory_init(mock_observatory):
    assert isinstance(mock_observatory.telescope, Telescope)
    assert isinstance(mock_observatory.detector, Detector)
    assert isinstance(mock_observatory.coronagraph, Coronagraph)


def test_observatory_load_configuration(mock_observatory, mock_observation, mock_scene):
    parameters = {
        "observatory_preset": "ToyModel",
        "observing_mode": "IMAGER",
        "Toptical": 0.8,
    }
    mock_observatory.load_configuration(parameters, mock_observation, mock_scene)

    assert mock_observatory.observing_mode == "IMAGER"
    assert np.isclose(mock_observatory.optics_throughput.value, 0.8)
    assert np.isclose(mock_observatory.epswarmTrcold.value, 0.2)
    assert np.isclose(mock_observatory.total_throughput.value, 0.8 * 0.9 * 0.9 * 0.95)


def test_observatory_calculate_optics_throughput(mock_observatory):
    parameters = {}
    mock_observatory.calculate_optics_throughput(parameters)
    assert np.isclose(mock_observatory.optics_throughput.value, 0.9 * 0.9)

    parameters = {"Toptical": 0.8}
    mock_observatory.calculate_optics_throughput(parameters)
    assert np.isclose(mock_observatory.optics_throughput.value, 0.8)


def test_observatory_calculate_warmemissivity_coldtransmission(mock_observatory):
    parameters = {}
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.calculate_warmemissivity_coldtransmission(parameters)
    assert np.isclose(mock_observatory.epswarmTrcold.value, 0.2)

    parameters = {"epswarmTrcold": 0.3}
    mock_observatory.calculate_warmemissivity_coldtransmission(parameters)
    assert np.isclose(mock_observatory.epswarmTrcold.value, 0.3)


def test_observatory_calculate_total_throughput(mock_observatory):
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.detector.dQE = [0.9] * DIMENSIONLESS
    mock_observatory.detector.QE = [0.9] * u.electron / u.photon
    mock_observatory.telescope.Tcontam = 0.95 * DIMENSIONLESS

    mock_observatory.calculate_total_throughput()
    assert np.isclose(mock_observatory.total_throughput.value, 0.8 * 0.9 * 0.9 * 0.95)


def test_observatory_mediator():
    telescope = MockTelescope()
    detector = MockDetector()
    coronagraph = MockCoronagraph()
    observation = mock_observation()
    scene = mock_scene()

    observatory = Observatory()
    observatory.telescope = telescope
    observatory.detector = detector
    observatory.coronagraph = coronagraph

    mediator = ObservatoryMediator(observatory, observation, scene)

    assert mediator.get_telescope_parameter("diameter") == 8.0 * LENGTH
    assert mediator.get_detector_parameter("pixscale_mas") == 10 * u.mas
    assert mediator.get_coronagraph_parameter("minimum_IWA") == 2 * u.lambda_over_D
    assert mediator.get_observation_parameter("wavelength") == [0.5] * WAVELENGTH
    assert mediator.get_scene_parameter("Fstar") == [1e-8] * DIMENSIONLESS
