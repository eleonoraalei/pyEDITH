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


class MockMediator:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return [0.5] * WAVELENGTH
        elif param == "psf_trunc_ratio":
            return [0.3] * DIMENSIONLESS
        elif param == "photap_rad":
            return 0.7 * LAMBDA_D
        else:
            return 1.0

    def get_coronagraph_parameter(self, param):
        return 0.2 if param == "bandwidth" else 1.0

    def get_scene_parameter(self, param):
        if param == "angular_diameter_arcsec":
            return 0.1 * ARCSEC
        else:
            return 1.0


def test_generate_radii():
    # Even number
    radii = generate_radii(10, 10)
    assert radii.shape == (10, 10)

    # Odd number
    radii = generate_radii(5, 5)
    assert np.isclose(radii[2, 2], 0.0)
    assert np.isclose(radii[0, 0], np.sqrt(radii[0, 2] ** 2 + radii[2, 0] ** 2))

    # missing y
    radii = generate_radii(5)
    assert np.isclose(radii[2, 2], 0.0)
    assert np.isclose(radii[0, 0], np.sqrt(radii[0, 2] ** 2 + radii[2, 0] ** 2))


def test_validate_configuration():
    coronagraph = ToyModelCoronagraph()

    # Set up a valid configuration
    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.noisefloor = np.ones((100, 100)) * DIMENSIONLESS
    coronagraph.photap_frac = np.ones((100, 100, 1)) * DIMENSIONLESS
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
    coronagraph.minimum_IWA = 2 * LAMBDA_D
    coronagraph.maximum_OWA = 10 * LAMBDA_D
    coronagraph.coronagraph_throughput = np.array([0.5]) * DIMENSIONLESS
    coronagraph.coronagraph_spectral_resolution = 1 * DIMENSIONLESS

    # Test valid configuration
    coronagraph.validate_configuration()  # This should not raise any exception

    # Test missing attribute
    delattr(coronagraph, "Istar")
    with pytest.raises(AttributeError, match="Coronagraph is missing attribute: Istar"):
        coronagraph.validate_configuration()
    coronagraph.Istar = np.ones((100, 100)) * DIMENSIONLESS  # Restore attribute

    # Test incorrect types
    original_npix = coronagraph.npix
    coronagraph.npix = 100.0  # Should be int, not float
    with pytest.raises(
        TypeError, match="Coronagraph attribute npix should be an integer"
    ):
        coronagraph.validate_configuration()
    coronagraph.npix = original_npix  # Restore correct type

    original_bandwidth = coronagraph.bandwidth
    coronagraph.bandwidth = "0.1"  # Should be float, not string
    with pytest.raises(
        TypeError, match="Coronagraph attribute bandwidth should be a float"
    ):
        coronagraph.validate_configuration()
    coronagraph.bandwidth = original_bandwidth  # Restore correct type

    # Test incorrect units
    original_pixscale = coronagraph.pixscale
    coronagraph.pixscale = 0.1 * u.m  # Incorrect unit
    with pytest.raises(
        ValueError, match="Coronagraph attribute pixscale has incorrect units"
    ):
        coronagraph.validate_configuration()
    coronagraph.pixscale = original_pixscale  # Restore correct unit

    # Test non-Quantity for Quantity attribute
    coronagraph.pixscale = 0.1  # Missing unit
    with pytest.raises(
        TypeError, match="Coronagraph attribute pixscale should be a Quantity"
    ):
        coronagraph.validate_configuration()
    coronagraph.pixscale = original_pixscale  # Restore correct Quantity


def test_toy_model_coronagraph_init():
    coronagraph = ToyModelCoronagraph()
    assert coronagraph.path is None
    assert coronagraph.keyword is None


def test_toy_model_coronagraph_load_configuration():
    coronagraph = ToyModelCoronagraph()
    parameters = {
        "pixscale": 0.3,
        "minimum_IWA": 2.5,
        "maximum_OWA": 90.0,
        "contrast": 1e-10,
        "noisefloor_factor": 0.05,
        "bandwidth": 0.1,
        "Tcore": 0.3,
        "TLyot": 0.7 * DIMENSIONLESS,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediator()

    coronagraph.load_configuration(parameters, mediator)

    assert coronagraph.pixscale == 0.3 * LAMBDA_D
    assert coronagraph.minimum_IWA == 2.5 * LAMBDA_D
    assert coronagraph.maximum_OWA == 90.0 * LAMBDA_D
    assert coronagraph.contrast == 1e-10 * DIMENSIONLESS
    assert coronagraph.noisefloor_factor == 0.05 * DIMENSIONLESS
    assert coronagraph.bandwidth == 0.1
    assert coronagraph.Tcore == 0.3 * DIMENSIONLESS
    assert coronagraph.TLyot == 0.7 * DIMENSIONLESS
    assert coronagraph.nrolls == 2
    assert coronagraph.nchannels == 1
    assert (
        coronagraph.coronagraph_throughput == [0.44] * DIMENSIONLESS
    )  # should grab from default values
    assert (
        coronagraph.coronagraph_spectral_resolution == 1 * DIMENSIONLESS
    )  # should grab from default values

    # Calculated  values
    assert hasattr(coronagraph, "psf_trunc_ratio")
    assert hasattr(coronagraph, "npsfratios")
    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photap_frac")
    assert hasattr(coronagraph, "PSFpeak")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")

    # Test calculated values
    assert coronagraph.npix == 400
    assert coronagraph.xcenter == 200 * PIXEL
    assert coronagraph.ycenter == 200 * PIXEL

    # Check r
    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    assert np.isclose(
        coronagraph.r[0, 0],
        84.641,
    )

    # Check omega_lod
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(coronagraph.omega_lod == np.pi * 0.7**2 * LAMBDA_D**2)

    # Check skytrans
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert np.all(coronagraph.skytrans == 0.7 * DIMENSIONLESS)

    # Check photap_frac
    assert coronagraph.photap_frac.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(
        (coronagraph.photap_frac == 0.3 * DIMENSIONLESS)
        | (coronagraph.photap_frac == 0.0 * DIMENSIONLESS)
    )
    assert np.all(
        coronagraph.photap_frac[coronagraph.r < coronagraph.minimum_IWA]
        == 0.0 * DIMENSIONLESS
    )
    assert np.all(
        coronagraph.photap_frac[coronagraph.r > coronagraph.maximum_OWA]
        == 0.0 * DIMENSIONLESS
    )

    # Check PSFpeak
    assert np.isclose(coronagraph.PSFpeak, 0.025 * 0.7 * DIMENSIONLESS)

    # Check Istar
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert np.allclose(coronagraph.Istar.value, 1e-10 * 0.025 * 0.7, rtol=1e-6)
    assert coronagraph.Istar.unit == DIMENSIONLESS

    # Check noisefloor
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert np.allclose(
        coronagraph.noisefloor.value, 0.05 * 1e-10 * 0.025 * 0.7, rtol=1e-6
    )
    assert coronagraph.noisefloor.unit == DIMENSIONLESS


def test_coronagraph_yip_init():
    coronagraph = CoronagraphYIP(path="test_path", keyword="usort")
    assert coronagraph.path == "test_path"
    assert coronagraph.keyword == "usort"


@pytest.fixture
def mock_yippy_object():
    mock_yippy = MagicMock()
    mock_yippy.header.pixscale.value = 0.25
    mock_yippy.header.naxis1 = 100
    mock_yippy.header.xcenter = 50
    mock_yippy.header.ycenter = 50
    mock_yippy.sky_trans.return_value = np.ones((100, 100))
    mock_yippy.offax.x_offsets = np.linspace(0, 10, 11)
    mock_yippy.offax.y_offsets = np.linspace(0, 10, 11)
    mock_yippy.offax.reshaped_psfs = np.random.rand(11, 1, 100, 100)
    mock_yippy.stellar_intens.return_value = np.random.rand(100, 100)
    return mock_yippy


@pytest.fixture
def mock_instrument():
    mock = MagicMock()
    mock.total_inst_refl = 0.9
    return mock


@pytest.fixture
def mock_telescope():
    mock = MagicMock()
    mock.diam_circ = 8.0
    return mock


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")  # Patch the alias used in your file
def test_coronagraph_yip_load_configuration(
    mock_yippycoro,
    mock_load_telescope,
    mock_load_instrument,
    mock_yippy_object,
    mock_instrument,
    mock_telescope,
):
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope
    mock_yippycoro.return_value = mock_yippy_object

    coronagraph = CoronagraphYIP(path="test_path")
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }

    mediator = MockMediator()

    coronagraph.load_configuration(parameters, mediator)

    # Check parameters from yippy + overwritten parameters
    assert coronagraph.pixscale == 0.25 * LAMBDA_D  # This comes from mock_yippy_object
    assert coronagraph.minimum_IWA == 2.0 * LAMBDA_D  # Default
    assert (
        coronagraph.maximum_OWA == 90.0 * LAMBDA_D
    )  # Overwritten value from parameters
    assert coronagraph.bandwidth == 0.1  # Overwritten value from parameters
    assert coronagraph.nrolls == 2
    assert coronagraph.nchannels == 1
    assert coronagraph.az_avg == True

    # Check calculated values
    assert coronagraph.npix == 100  # This comes from mock_yippy_object
    assert coronagraph.xcenter == 50 * PIXEL
    assert coronagraph.ycenter == 50 * PIXEL

    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photap_frac")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")

    # Check r
    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    # value of r checked in specific test

    # Check omega_lod
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert coronagraph.omega_lod.unit == LAMBDA_D**2

    # Check skytrans
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert np.all(
        coronagraph.skytrans == mock_yippy_object.sky_trans.return_value * DIMENSIONLESS
    )

    # Check photap_frac
    assert coronagraph.photap_frac.shape == (coronagraph.npix, coronagraph.npix, 1)
    # TODO evaluate photap_frac

    # Check Istar
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    # TODO evaluate Istar

    # Check noisefloor
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    # The exact calculation of noisefloor depends on your implementation
    assert np.all(coronagraph.noisefloor >= 0 * DIMENSIONLESS)

    # Check other attributes
    assert hasattr(coronagraph, "psf_trunc_ratio")
    assert hasattr(coronagraph, "npsfratios")
    assert hasattr(coronagraph, "coronagraph_throughput")
    assert hasattr(coronagraph, "coronagraph_spectral_resolution")


# @patch("eacy.load_instrument", mock_load_instrument)
# @patch("eacy.load_telescope", mock_load_telescope)
# @patch("yippy.Coronagraph", mock_yippycoro)
# def test_coronagraph_yip_ifs_mode(mock_mediator, mock_yippy_object):
#     mock_load_instrument.return_value = type(
#         "obj", (object,), {"__dict__": {"total_inst_refl": 0.9}}
#     )
#     mock_load_telescope.return_value = type(
#         "obj", (object,), {"__dict__": {"diam_circ": 8.0}}
#     )
#     mock_yippycoro.return_value = mock_yippy_object

#     coronagraph = CoronagraphYIP(path="test_path")
#     parameters = {
#         "observing_mode": "IFS",
#         "pixscale": 0.3,
#         "minimum_IWA": 2.5,
#         "maximum_OWA": 90.0,
#         "bandwidth": 0.1,
#         "nrolls": 2,
#         "nchannels": 1,
#     }

#     coronagraph.load_configuration(parameters, mock_mediator)

#     assert coronagraph.observing_mode == "IFS"
#     # Add more assertions specific to IFS mode


# @patch("eacy.load_instrument", mock_load_instrument)
# @patch("eacy.load_telescope", mock_load_telescope)
# @patch("yippy.Coronagraph", mock_yippycoro)
# def test_coronagraph_yip_validate_configuration(mock_mediator, mock_yippy_object):
#     mock_load_instrument.return_value = type(
#         "obj", (object,), {"__dict__": {"total_inst_refl": 0.9}}
#     )
#     mock_load_telescope.return_value = type(
#         "obj", (object,), {"__dict__": {"diam_circ": 8.0}}
#     )
#     mock_yippycoro.return_value = mock_yippy_object

#     coronagraph = CoronagraphYIP(path="test_path")
#     parameters = {
#         "observing_mode": "IMAGER",
#         "pixscale": 0.3,
#         "minimum_IWA": 2.5,
#         "maximum_OWA": 90.0,
#         "bandwidth": 0.1,
#         "nrolls": 2,
#         "nchannels": 1,
#     }
#     coronagraph.load_configuration(parameters, mock_mediator)

#     coronagraph.validate_configuration()  # Should not raise any exception

#     # Test missing attribute
#     delattr(coronagraph, "Istar")
#     with pytest.raises(AttributeError):
#         coronagraph.validate_configuration()

#     # Test incorrect units
#     coronagraph.Istar = np.zeros((coronagraph.npix, coronagraph.npix))  # Missing units
#     with pytest.raises(TypeError):
#         coronagraph.validate_configuration()
