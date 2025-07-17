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
    PHOTON_FLUX_DENSITY,
    DIMENSIONLESS,
    LENGTH,
    LAMBDA_D,
    ARCSEC,
    WAVELENGTH,
    TEMPERATURE,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    TIME,
    FRAME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    ELECTRON,
    PHOTON_COUNT,
    INV_SQUARE_ARCSEC,
    PIXEL,
    ZODI,
)


class MockTelescope(Telescope):
    def load_configuration(self, parameters, mediator):
        self.path = None
        self.keyword = "ToyModel"
        self.diameter = 7.87 * u.m
        self.unobscured_area = 0.879
        self.toverhead_fixed = 8381.3 * u.s
        self.toverhead_multi = 1.1 * DIMENSIONLESS
        self.telescope_optical_throughput = u.Quantity([0.823], DIMENSIONLESS)
        self.temperature = 290.0 * u.K
        self.T_contamination = 0.95 * DIMENSIONLESS
        self.Area = 42.75906827 * u.m**2


class MockDetector(Detector):
    def load_configuration(self, parameters, mediator):
        self.path = None
        self.keyword = "ToyModel"
        self.pixscale_mas = 6.55224925 * u.mas
        self.npix_multiplier = u.Quantity([1.0], DIMENSIONLESS)
        self.DC = u.Quantity([3.0e-05], DARK_CURRENT)
        self.RN = u.Quantity([0.0], READ_NOISE)
        self.tread = u.Quantity([1000.0], READ_TIME)
        self.CIC = u.Quantity([0.0013], CLOCK_INDUCED_CHARGE)
        self.QE = u.Quantity([0.897], QUANTUM_EFFICIENCY)
        self.dQE = u.Quantity([0.75], DIMENSIONLESS)


class MockCoronagraph(Coronagraph):
    def load_configuration(self, parameters, mediator):
        self.path = None
        self.keyword = "ToyModel"
        self.pixscale = 30.0 * LAMBDA_D
        self.minimum_IWA = 1.0 * LAMBDA_D
        self.maximum_OWA = 60.0 * LAMBDA_D
        self.contrast = 1.05e-13 * DIMENSIONLESS
        self.noisefloor_factor = 0.03 * DIMENSIONLESS
        self.bandwidth = 0.2
        self.Tcore = 0.2968371 * DIMENSIONLESS
        self.TLyot = 0.65 * DIMENSIONLESS
        self.nrolls = 1
        self.nchannels = 2
        self.coronagraph_optical_throughput = u.Quantity([0.44], DIMENSIONLESS)
        self.coronagraph_spectral_resolution = 1.0 * DIMENSIONLESS
        self.npsfratios = 1
        self.npix = 4
        self.xcenter = 2.0 * PIXEL
        self.ycenter = 2.0 * PIXEL
        self.r = (
            u.Quantity(
                [
                    [63.63961031, 47.4341649, 47.4341649, 63.63961031],
                    [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                    [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                    [63.63961031, 47.4341649, 47.4341649, 63.63961031],
                ],
                LAMBDA_D,
            ),
        )
        self.omega_lod = u.Quantity(
            [
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
            ],
            LAMBDA_D**2,
        )
        self.skytrans = u.Quantity(
            [
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
            ],
            DIMENSIONLESS,
        )
        self.photometric_aperture_throughput = u.Quantity(
            [
                [[0.0], [0.2968371], [0.2968371], [0.0]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.0], [0.2968371], [0.2968371], [0.0]],
            ],
            DIMENSIONLESS,
        )
        self.PSFpeak = u.Quantity(0.01625, DIMENSIONLESS)
        self.Istar = u.Quantity(
            [
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            ],
            DIMENSIONLESS,
        )
        self.noisefloor = u.Quantity(
            [
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            ],
            DIMENSIONLESS,
        )


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
    obs.td_limit = 1.0e20 * u.s
    obs.wavelength = u.Quantity([0.5], u.micron)
    obs.SNR = u.Quantity([7], DIMENSIONLESS)
    obs.photometric_aperture_radius = 0.85 * LAMBDA_D
    obs.CRb_multiplier = 2
    obs.nlambd = 1
    obs.tp = 0.0 * u.s
    obs.exptime = u.Quantity([0.0], u.s)
    obs.fullsnr = u.Quantity([0.0], DIMENSIONLESS)

    return obs


@pytest.fixture
def mock_scene():
    scene = AstrophysicalScene()
    scene.F0V = 10374.9964895 * u.photon / u.nm / u.s / u.cm**2
    scene.Lstar = 0.86 * u.L_sun
    scene.dist = 14.8 * u.pc
    scene.F0 = u.Quantity([12638.83670769], u.photon / u.nm / u.s / u.cm**2)
    scene.vmag = 5.84 * u.mag
    scene.mag = u.Quantity([6.189576], u.mag)
    scene.deltamag = u.Quantity([25.5], u.mag)
    scene.min_deltamag = 25 * u.mag
    scene.Fs_over_F0 = u.Quantity([0.00334326], DIMENSIONLESS)
    scene.Fp_over_Fs = u.Quantity([6.30957344e-11], DIMENSIONLESS)
    scene.Fp_min_over_Fs = 1.0e-10 * DIMENSIONLESS
    scene.stellar_angular_diameter_arcsec = 0.01 * ARCSEC
    scene.nzodis = 3 * ZODI
    scene.ra = 236.00757737 * u.deg
    scene.dec = 2.51516683 * u.deg
    scene.separation = 0.0628 * u.arcsec
    scene.xp = 0.0628 * u.arcsec
    scene.yp = 0.0 * u.arcsec
    scene.M_V = 4.98869142 * u.mag
    scene.Fzodi_list = (u.Quantity([6.11055505e-10], 1 / u.arcsec**2),)
    scene.Fexozodi_list = (u.Quantity([2.97724302e-09], 1 / u.arcsec**2),)
    scene.Fbinary_list = u.Quantity([0], DIMENSIONLESS)

    return scene


def test_observatory_init():
    obs = Observatory()
    assert obs.telescope is None
    assert obs.detector is None
    assert obs.coronagraph is None


def test_observatory_validate_configuration(mock_observatory):
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.total_throughput = [0.6] * QUANTUM_EFFICIENCY
    mock_observatory.epswarmTrcold = [0.2] * DIMENSIONLESS
    mock_observatory.telescope.load_configuration({}, {})
    mock_observatory.coronagraph.load_configuration({}, {})
    mock_observatory.detector.load_configuration({}, {})

    mock_observatory.validate_configuration()

    delattr(mock_observatory, "optics_throughput")
    with pytest.raises(
        AttributeError, match="Observatory is missing attribute: optics_throughput"
    ):
        mock_observatory.validate_configuration()

    mock_observatory.optics_throughput = 0.8  # Not a Quantity
    with pytest.raises(
        TypeError, match="Observatory attribute optics_throughput should be a Quantity"
    ):
        mock_observatory.validate_configuration()

    mock_observatory.optics_throughput = [0.8] * u.meter  # Wrong unit
    with pytest.raises(
        ValueError, match="Observatory attribute optics_throughput has incorrect units"
    ):
        mock_observatory.validate_configuration()


def test_calculate_optics_throughput(mock_observatory,mock_observation,mock_scene):
    """Test calculation of optics throughput"""
    # Test with T_optical parameter
    parameters = {"T_optical": 0.8, "observing_mode": "IMAGER"}
    mock_observatory.telescope.load_configuration({}, {})
    mock_observatory.coronagraph.load_configuration({}, {})
    mock_observatory.detector.load_configuration({}, {})
    mediator = ObservatoryMediator(mock_observatory, mock_observation, mock_scene)

    mock_observatory.calculate_optics_throughput(parameters,mediator)
    assert mock_observatory.optics_throughput.value == 0.8

    # Test with IFS mode
    parameters = {"T_optical": [0.8], "observing_mode": "IFS", "IFS_eff": [0.9]}
    mock_observatory.calculate_optics_throughput(parameters,mediator)
    assert mock_observatory.optics_throughput.value == [0.8 * 0.9]

    # Test without T_optical parameter
    parameters = {"observing_mode": "IMAGER"}
    mock_observatory.calculate_optics_throughput(parameters,mediator)
    expected = (
        mock_observatory.telescope.telescope_optical_throughput.value[0]
        * mock_observatory.coronagraph.coronagraph_optical_throughput.value[0]
    )
    assert mock_observatory.optics_throughput.value == expected


def test_calculate_warmemissivity_coldtransmission(mock_observatory,mock_observation,mock_scene):
    """Test calculation of warm emissivity and cold transmission"""
    mock_observatory.telescope.load_configuration({}, {})
    mock_observatory.coronagraph.load_configuration({}, {})
    mock_observatory.detector.load_configuration({}, {})
    mediator = ObservatoryMediator(mock_observatory, mock_observation, mock_scene)

    # Test with explicit parameter
    parameters = {"epswarmTrcold": 0.3}
    mock_observatory.calculate_warmemissivity_coldtransmission(parameters,mediator)
    assert mock_observatory.epswarmTrcold.value == 0.3

    # Test calculated from optics throughput
    parameters = {}
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.calculate_warmemissivity_coldtransmission(parameters,mediator)
    assert mock_observatory.epswarmTrcold.value == 1 - 0.8


def test_calculate_total_throughput(mock_observatory):
    """Test calculation of total throughput"""
    mock_observatory.optics_throughput = [0.8] * DIMENSIONLESS
    mock_observatory.detector.dQE = [0.9] * DIMENSIONLESS
    mock_observatory.detector.QE = [0.9] * QUANTUM_EFFICIENCY
    mock_observatory.telescope.T_contamination = 0.95 * DIMENSIONLESS

    mock_observatory.calculate_total_throughput()
    expected = 0.8 * 0.9 * 0.9 * 0.95
    assert np.isclose(mock_observatory.total_throughput.value, expected)
    assert mock_observatory.total_throughput.unit == QUANTUM_EFFICIENCY


def test_load_configuration(mock_observatory, mock_observation, mock_scene):
    """Test loading configuration"""
    parameters = {"observing_mode": "IMAGER", "T_optical": 0.8}

    mock_observatory.load_configuration(parameters, mock_observation, mock_scene)

    assert mock_observatory.observing_mode == "IMAGER"
    assert mock_observatory.optics_throughput.value == 0.8
    assert hasattr(mock_observatory, "epswarmTrcold")
    assert hasattr(mock_observatory, "total_throughput")


def test_observatory_mediator(mock_observatory, mock_observation, mock_scene):
    # Initialize mock observatory parameters
    mock_observatory.telescope.load_configuration({}, {})
    mock_observatory.coronagraph.load_configuration({}, {})
    mock_observatory.detector.load_configuration({}, {})
    mediator = ObservatoryMediator(mock_observatory, mock_observation, mock_scene)

    # Test getting telescope parameter
    assert (
        mediator.get_telescope_parameter("diameter")
        == mock_observatory.telescope.diameter
    )

    # Test getting coronagraph parameter
    assert (
        mediator.get_coronagraph_parameter("contrast")
        == mock_observatory.coronagraph.contrast
    )

    # Test getting detector parameter
    assert (
        mediator.get_detector_parameter("pixscale_mas")
        == mock_observatory.detector.pixscale_mas
    )

    # Test getting observation parameter
    assert (
        mediator.get_observation_parameter("wavelength") == mock_observation.wavelength
    )

    # Test getting scene parameter
    assert mediator.get_scene_parameter("vmag") == mock_scene.vmag

    # Test getting non-existent parameter
    assert mediator.get_telescope_parameter("nonexistent") is None
