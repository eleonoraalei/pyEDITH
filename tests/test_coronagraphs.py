import pytest
import numpy as np
from astropy import units as u
from pyEDITH.components.coronagraphs import (
    ToyModelCoronagraph,
    CoronagraphYIP,
    generate_radii,
)
from pyEDITH.units import (
    LAMBDA_D,
    DIMENSIONLESS,
    LENGTH,
    WAVELENGTH,
    PHOTON_FLUX_DENSITY,
    PIXEL,
    ARCSEC,
)
from unittest.mock import patch, MagicMock
import logging

# ============================================================================
# Mock Objects and Fixtures
# ============================================================================


class MockMediator_IMAGER:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return [0.7] * WAVELENGTH
        elif param == "observing_mode":
            return "IMAGER"
        else:
            return 1.0

    def get_scene_parameter(self, param):
        if param == "stellar_angular_diameter_arcsec":
            return 1e-3 * ARCSEC
        else:
            return 1.0


class MockMediator_IFS:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return [0.5, 0.6, 0.7] * WAVELENGTH
        elif param == "observing_mode":
            return "IFS"
        else:
            return 1.0

    def get_scene_parameter(self, param):
        if param == "stellar_angular_diameter_arcsec":
            return 1e-3 * ARCSEC
        else:
            return 1.0


@pytest.fixture
def mock_instrument():
    mock = MagicMock()
    mock.lam = np.linspace(0.3, 1.6, 10) * WAVELENGTH
    mock.total_inst_refl = np.array(
        [
            4.80759739e-29,
            4.05330898e-01,
            4.40641747e-01,
            3.94770896e-01,
            4.12956241e-01,
            5.15044124e-01,
            5.76293823e-01,
            5.38605236e-01,
            6.27117118e-01,
            6.63022075e-01,
        ]
    )
    return mock


@pytest.fixture
def mock_telescope():
    mock = MagicMock()
    mock.diam_circ = 8.0
    return mock


# ============================================================================
# Tests for generate_radii
# ============================================================================


def test_generate_radii_even_dimensions():
    """Test radii generation with even dimensions."""
    radii = generate_radii(10, 10)
    assert radii.shape == (10, 10)


def test_generate_radii_odd_dimensions():
    """Test radii generation with odd dimensions has zero at center."""
    radii = generate_radii(5, 5)

    assert np.isclose(radii[2, 2], 0.0)
    assert np.isclose(radii[0, 0], np.sqrt(radii[0, 2] ** 2 + radii[2, 0] ** 2))


def test_generate_radii_default_square():
    """Test radii generation defaults to square when y dimension not provided."""
    radii = generate_radii(5)

    assert radii.shape == (5, 5)
    assert np.isclose(radii[2, 2], 0.0)
    assert np.isclose(radii[0, 0], np.sqrt(radii[0, 2] ** 2 + radii[2, 0] ** 2))


# ============================================================================
# Tests for ToyModelCoronagraph initialization
# ============================================================================


def test_toy_model_coronagraph_init():
    """Test ToyModelCoronagraph initializes with None values."""
    coronagraph = ToyModelCoronagraph()

    assert coronagraph.path is None


# ============================================================================
# Tests for ToyModelCoronagraph.load_configuration
# ============================================================================


def test_toy_model_load_configuration_basic_parameters(caplog):
    """Test that basic parameters are loaded correctly."""
    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        coronagraph = ToyModelCoronagraph()
        parameters = {
            "pixscale": 0.3,
            "contrast": 1e-10,
            "noisefloor_factor": 0.05,
            "bandwidth": 0.1,
            "photometric_aperture_radius": 0.6,
            "Tcore": 0.3,
            "TLyot": 0.7 * DIMENSIONLESS,
            "nrolls": 2,
            "nchannels": 1,
        }
        mediator = MockMediator_IMAGER()

        coronagraph.load_configuration(parameters, mediator)

        assert coronagraph.pixscale == 0.3 * LAMBDA_D
        assert coronagraph.contrast == 1e-10 * DIMENSIONLESS
        assert coronagraph.noisefloor_factor == 0.05 * DIMENSIONLESS
        assert coronagraph.bandwidth == 0.1
        assert coronagraph.photometric_aperture_radius == 0.6 * LAMBDA_D
        assert coronagraph.Tcore == 0.3 * DIMENSIONLESS
        assert coronagraph.TLyot == 0.7 * DIMENSIONLESS
        assert coronagraph.nrolls == 2
        assert coronagraph.nchannels == 1


def test_toy_model_load_configuration_default_values():
    """Test that default values are used when not provided."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.coronagraph_optical_throughput == [0.44] * DIMENSIONLESS
    assert coronagraph.coronagraph_spectral_resolution == 1 * DIMENSIONLESS


def test_toy_model_load_configuration_calculated_attributes():
    """Test that derived attributes are calculated correctly."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7 * DIMENSIONLESS,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert hasattr(coronagraph, "npsfratios")
    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photometric_aperture_radius")
    assert hasattr(coronagraph, "photometric_aperture_throughput")
    assert hasattr(coronagraph, "PSFpeak")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")


def test_toy_model_load_configuration_pixel_grid():
    """Test that pixel grid parameters are calculated correctly."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7 * DIMENSIONLESS,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.npix == 400
    assert coronagraph.xcenter == 200 * PIXEL
    assert coronagraph.ycenter == 200 * PIXEL


def test_toy_model_load_configuration_radial_grid():
    """Test that radial separation grid is calculated correctly."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    assert np.isclose(coronagraph.r[0, 0], 84.641)


def test_toy_model_load_configuration_omega_lod():
    """Test that omega_lod is calculated with correct shape and values."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(
        coronagraph.omega_lod
        == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
    )


def test_toy_model_load_configuration_skytrans():
    """Test that sky transmission is set correctly."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert np.all(coronagraph.skytrans == 0.7 * DIMENSIONLESS)


def test_toy_model_load_configuration_photometric_aperture_throughput():
    """Test that photometric aperture throughput respects IWA and OWA."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.3 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )


def test_toy_model_load_configuration_psf_peak():
    """Test that PSF peak is calculated correctly."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert np.isclose(coronagraph.PSFpeak, 0.025 * 0.7 * DIMENSIONLESS)


def test_toy_model_load_configuration_istar():
    """Test that stellar intensity is calculated correctly."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert np.allclose(coronagraph.Istar.value, 1e-10 * 0.025 * 0.7, rtol=1e-6)
    assert coronagraph.Istar.unit == DIMENSIONLESS


def test_toy_model_load_configuration_noisefloor_with_factor(caplog):
    """Test noisefloor calculation using noisefloor_factor."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.6,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Calculating noisefloor by multiplying noisefloor_factor=0.05, contrast=1e-10, PSFpeak="
        + str(0.025 * 0.7)
        in record.message
        for record in caplog.records
    )

    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert np.allclose(
        coronagraph.noisefloor.value, 0.05 * 1e-10 * 0.025 * 0.7, rtol=1e-6
    )
    assert coronagraph.noisefloor.unit == DIMENSIONLESS


def test_toy_model_load_configuration_noisefloor_ppf_raises_error():
    """Test that providing noisefloor_PPF raises appropriate error."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "bandwidth": 0.1,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
        "noisefloor_PPF": 30,
    }
    mediator = MockMediator_IMAGER()

    with pytest.raises(
        KeyError,
        match="Noisefloor_PPF mode not implemented in ToyModel coronagraph",
    ):
        coronagraph.load_configuration(parameters, mediator)


def test_toy_model_load_configuration_default_noisefloor_factor(caplog):
    """Test that default noisefloor_factor is used when not provided."""
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "contrast": 1e-10,
        "bandwidth": 0.1,
        "Tcore": 0.3,
        "TLyot": 0.7,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "noisefloor_factor value not provided. Using the default value: 0.03"
        in record.message
        for record in caplog.records
    )
    assert any(
        "Calculating noisefloor by multiplying noisefloor_factor=0.03, contrast=1e-10, PSFpeak="
        + str(0.025 * 0.7)
        in record.message
        for record in caplog.records
    )

    assert coronagraph.noisefloor.unit == DIMENSIONLESS


# ============================================================================
# Tests for CoronagraphYIP initialization
# ============================================================================


def test_coronagraph_yip_init_with_path():
    """Test CoronagraphYIP initialization with path."""
    coronagraph = CoronagraphYIP(path="test_path")

    assert coronagraph.path == "test_path"
    assert coronagraph.yippy_coro is None


def test_coronagraph_yip_init_with_yippy_coro(yippy_coronagraph):
    """Test CoronagraphYIP initialization with pre-constructed yippy object."""
    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)

    assert coronagraph.yippy_coro is yippy_coronagraph
    assert coronagraph.path is None


def test_coronagraph_yip_init_requires_path_or_yippy():
    """Test that initialization requires either path or yippy_coro."""
    with pytest.raises(
        ValueError, match="Either a path or a yippy_coro must be provided"
    ):
        CoronagraphYIP()


def test_coronagraph_yip_init_not_both_path_and_yippy(yippy_coronagraph):
    """Test that initialization rejects both path and yippy_coro."""
    with pytest.raises(
        ValueError, match="Only one of path or yippy_coro can be provided"
    ):
        CoronagraphYIP(path="some_path", yippy_coro=yippy_coronagraph)


# ============================================================================
# Tests for CoronagraphYIP.load_configuration - IMAGER mode
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_basic_parameters(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that basic YIP parameters are loaded correctly in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.pixscale == yippy_coronagraph.header.pixscale.value * LAMBDA_D
    assert coronagraph.bandwidth == 0.1
    assert coronagraph.nrolls == 2
    assert coronagraph.nchannels == 1
    assert coronagraph.az_avg == True


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_pixel_grid(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that pixel grid parameters are loaded from YIP."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.npix == yippy_coronagraph.header.naxis1
    assert coronagraph.xcenter == yippy_coronagraph.header.xcenter * PIXEL
    assert coronagraph.ycenter == yippy_coronagraph.header.ycenter * PIXEL


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_has_required_attributes(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that all required attributes are created in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photometric_aperture_throughput")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_array_shapes(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that arrays have correct shapes in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_non_zero_values(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that arrays contain non-zero values where expected."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert not np.all(coronagraph.omega_lod == 0)
    assert not np.all(coronagraph.skytrans == 0)
    assert not np.all(coronagraph.photometric_aperture_throughput == 0)
    assert not np.all(coronagraph.Istar == 0)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_correct_units(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that arrays have correct units in IMAGER mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.omega_lod.unit == LAMBDA_D**2
    assert coronagraph.noisefloor.unit == DIMENSIONLESS


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_skytrans_from_yip(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that sky transmission matches YIP values."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert np.all(coronagraph.skytrans == yippy_coronagraph.sky_trans() * DIMENSIONLESS)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_optical_throughput(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that coronagraph optical throughput is loaded correctly."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert len(coronagraph.coronagraph_optical_throughput) == 1
    assert np.isclose(coronagraph.coronagraph_optical_throughput.value, 0.394770896)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_default_noisefloor_ppf(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
):
    """Test that default noisefloor_PPF is used when not provided."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "noisefloor_PPF value not provided. Using the default value: 30"
        in record.message
        for record in caplog.records
    )

    assert np.allclose(
        coronagraph.noisefloor,
        coronagraph.Istar / 30,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_custom_noisefloor_ppf(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
):
    """Test that custom noisefloor_PPF is used correctly."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
        "noisefloor_PPF": 35,
    }
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Setting the noise floor via user-supplied noisefloor_PPF..." in record.message
        for record in caplog.records
    )

    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    assert np.allclose(
        coronagraph.noisefloor,
        coronagraph.Istar / 35,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_noisefloor_factor_raises_error(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that noisefloor_factor raises appropriate error in YIP mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
        "noisefloor_factor": 1e-10,
    }
    mediator = MockMediator_IMAGER()

    with pytest.raises(
        ValueError,
        match="Noisefloor_factor mode not implemented in CoronagraphYIP",
    ):
        coronagraph.load_configuration(parameters, mediator)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_missing_aperture_raises_error(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that missing both aperture parameters raises error."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    with pytest.raises(
        KeyError,
        match="Either 'photometric_aperture_radius' or 'psf_trunc_ratio' must be provided",
    ):
        coronagraph.load_configuration(parameters, mediator)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_imager_az_avg_false(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that az_avg=False uses full 2D stellar intensity map."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": False,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.Istar.unit == DIMENSIONLESS
    assert not np.all(coronagraph.Istar == 0)

    # Verify it's 2D (not azimuthally averaged)
    # The stellar intensity should vary across the 2D map
    assert coronagraph.Istar.ndim == 2


# ============================================================================
# Tests for CoronagraphYIP.load_configuration - IFS mode
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_ifs_optical_throughput(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
):
    """Test that IFS mode correctly handles multiple wavelengths."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IFS",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "photometric_aperture_radius": 0.8,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator_ifs = MockMediator_IFS()

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator_ifs)

    assert any(
        "Both 'photometric_aperture_radius' and 'psf_trunc_ratio' provided"
        in record.message
        for record in caplog.records
    )

    assert len(coronagraph.coronagraph_optical_throughput) == 3
    assert np.isclose(
        coronagraph.coronagraph_optical_throughput.value,
        [0.41891199, 0.43711322, 0.40535648],
    ).all()


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_configuration_ifs_psf_trunc_ratio_warning(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
):
    """Test warning when both aperture methods provided in IFS mode."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IFS",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "photometric_aperture_radius": 0.8,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator_ifs = MockMediator_IFS()

    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator_ifs)

    assert any(
        "Using psf_trunc_ratio to calculate Omega..." in record.message
        for record in caplog.records
    )


# ============================================================================
# Tests for CoronagraphYIP with photometric_aperture_radius
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_photometric_aperture_with_custom_tcore(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
):
    """Test photometric aperture calculation with user-defined Tcore."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.85,
        "nrolls": 2,
        "nchannels": 1,
        "Tcore": 0.5 * DIMENSIONLESS,
    }
    mediator = MockMediator_IMAGER()

    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Using photometric_aperture_radius to calculate Omega..." in record.message
        for record in caplog.records
    )
    assert any(
        "Using user-defined Tcore..." in record.message for record in caplog.records
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_photometric_aperture_omega_calculation(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that omega_lod is calculated correctly with photometric aperture."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.85,
        "nrolls": 2,
        "nchannels": 1,
        "Tcore": 0.5 * DIMENSIONLESS,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(
        coronagraph.omega_lod
        == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_photometric_aperture_throughput_iwa_owa(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
):
    """Test that photometric aperture throughput respects IWA and OWA bounds."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "photometric_aperture_radius": 0.85,
        "nrolls": 2,
        "nchannels": 1,
        "Tcore": 0.5 * DIMENSIONLESS,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.5 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_photometric_aperture_default_tcore(
    mock_load_telescope,
    mock_load_instrument,
    yippy_coronagraph,
    mock_instrument,
    mock_telescope,
    caplog,
):
    """Test that default Tcore is used when not provided."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "photometric_aperture_radius": 0.85,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Using photometric_aperture_radius to calculate Omega..." in record.message
        for record in caplog.records
    )
    assert any("Using default Tcore..." in record.message for record in caplog.records)

    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(
        coronagraph.omega_lod
        == np.pi * parameters["photometric_aperture_radius"] ** 2 * LAMBDA_D**2
    )

    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.2968371 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )


# ============================================================================
# Tests for CoronagraphYIP with path
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_load_from_path(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    coronagraph_path,
):
    """Test that CoronagraphYIP can be constructed from a path."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(path=coronagraph_path)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.npix > 0
    assert hasattr(coronagraph, "Istar")


# ============================================================================
# Tests for CoronagraphYIP with pre-constructed yippy_coro
# ============================================================================


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_preconstruced_yippy_coro(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    yippy_coronagraph,
):
    """Test that pre-constructed yippy_coro is used directly."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.npix == yippy_coronagraph.header.naxis1
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_preconstruced_yippy_trunc_ratio_mismatch_warning(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    yippy_coronagraph,
    caplog,
):
    """Test warning when yippy_coro psf_trunc_ratio differs from parameters."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.99,  # Different from yippy_coro
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }
    mediator = MockMediator_IMAGER()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        coronagraph.load_configuration(parameters, mediator)

    assert any(
        "Pre-constructed yippy_coro has psf_trunc_ratio=" in record.message
        for record in caplog.records
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
def test_coronagraph_yip_nrolls_from_yippy_object(
    mock_load_telescope,
    mock_load_instrument,
    mock_instrument,
    mock_telescope,
    yippy_coronagraph,
    monkeypatch,
):
    """Test that nrolls is read from yippy object when available."""
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope

    monkeypatch.setattr(yippy_coronagraph, "nrolls", 4, raising=False)

    coronagraph = CoronagraphYIP(yippy_coro=yippy_coronagraph)
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "psf_trunc_ratio": 0.3,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }

    coronagraph.load_configuration(parameters, MockMediator_IMAGER())

    assert coronagraph.DEFAULT_CONFIG["nrolls"] == 4


# ============================================================================
# Tests for Coronagraph.validate_configuration
# ============================================================================


def test_validate_configuration_valid_setup():
    """Test that validate_configuration passes with valid configuration."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    # Should not raise any exception
    coronagraph.validate_configuration()


def test_validate_configuration_missing_istar():
    """Test that missing Istar attribute raises AttributeError."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    with pytest.raises(AttributeError, match="Coronagraph is missing attribute: Istar"):
        coronagraph.validate_configuration()


def test_validate_configuration_incorrect_npix_type():
    """Test that non-integer npix raises TypeError."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100.0  # Should be int
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    with pytest.raises(
        TypeError, match="Coronagraph attribute npix should be an integer"
    ):
        coronagraph.validate_configuration()


def test_validate_configuration_incorrect_bandwidth_type():
    """Test that non-float bandwidth raises TypeError."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = "0.1"  # Should be float
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    with pytest.raises(
        TypeError, match="Coronagraph attribute bandwidth should be a float"
    ):
        coronagraph.validate_configuration()


def test_validate_configuration_incorrect_pixscale_units():
    """Test that incorrect pixscale units raise ValueError."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * u.m  # Incorrect unit
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    with pytest.raises(
        ValueError, match="Coronagraph attribute pixscale has incorrect units"
    ):
        coronagraph.validate_configuration()


def test_validate_configuration_pixscale_not_quantity():
    """Test that pixscale without units raises TypeError."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1  # Missing unit
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    with pytest.raises(
        TypeError, match="Coronagraph attribute pixscale should be a Quantity"
    ):
        coronagraph.validate_configuration()


def test_validate_configuration_with_psf_trunc_ratio():
    """Test that validation passes with psf_trunc_ratio instead of aperture radius."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS
    coronagraph.psf_trunc_ratio = 0.3 * DIMENSIONLESS

    # Should not raise
    coronagraph.validate_configuration()


def test_validate_configuration_missing_both_aperture_params():
    """Test that missing both aperture parameters raises AttributeError."""
    coronagraph = ToyModelCoronagraph()

    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.photometric_aperture_throughput = np.ones((100, 100, 1)) * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * LAMBDA_D**2
    coronagraph.skytrans = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.pixscale = 0.1 * LAMBDA_D
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL
    coronagraph.bandwidth = 0.1
    coronagraph.npsfratios = 1
    coronagraph.nrolls = 1
    coronagraph.nchannels = 1
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    with pytest.raises(AttributeError, match="photometric_aperture_radius"):
        coronagraph.validate_configuration()
