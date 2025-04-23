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


class MockMediator_IMAGER:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return [0.7] * WAVELENGTH
        elif param == "psf_trunc_ratio":
            return [0.3] * DIMENSIONLESS
        elif param == "photometric_aperture_radius":
            return 0.7 * LAMBDA_D
        else:
            return 1.0

    def get_coronagraph_parameter(self, param):
        return 0.2 if param == "bandwidth" else 1.0

    def get_scene_parameter(self, param):
        if param == "stellar_angular_diameter_arcsec":
            return 0.1 * ARCSEC
        else:
            return 1.0


class MockMediator_IFS:
    def get_observation_parameter(self, param):
        if param == "wavelength":
            return [0.5, 0.6, 0.7] * WAVELENGTH
        elif param == "psf_trunc_ratio":
            return [0.3] * DIMENSIONLESS
        elif param == "photometric_aperture_radius":
            return 0.7 * LAMBDA_D
        else:
            return 1.0

    def get_coronagraph_parameter(self, param):
        return 0.2 if param == "bandwidth" else 1.0

    def get_scene_parameter(self, param):
        if param == "stellar_angular_diameter_arcsec":
            return 0.1 * ARCSEC
        else:
            return 1.0


class MockMediatorWithPhotapRad(MockMediator_IMAGER):
    def get_observation_parameter(self, param):
        if param == "psf_trunc_ratio":
            return None
        if param == "photometric_aperture_radius":
            return 0.7 * LAMBDA_D
        return super().get_observation_parameter(param)


class MockMediatorWithHighPSFTruncRatio(MockMediator_IMAGER):
    def get_observation_parameter(self, param):
        if param == "psf_trunc_ratio":
            return [1] * DIMENSIONLESS
        if param == "photometric_aperture_radius":
            return None
        return super().get_observation_parameter(param)


class MockMediatorNoParams(MockMediator_IMAGER):
    def get_observation_parameter(self, param):
        if param in ["psf_trunc_ratio", "photometric_aperture_radius"]:
            return None
        return super().get_observation_parameter(param)


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
    coronagraph.minimum_IWA = 2 * LAMBDA_D
    coronagraph.maximum_OWA = 10 * LAMBDA_D
    coronagraph.coronagraph_optical_throughput = np.array([0.5]) * DIMENSIONLESS
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
    mediator = MockMediator_IMAGER()

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
        coronagraph.coronagraph_optical_throughput == [0.44] * DIMENSIONLESS
    )  # should grab from default values
    assert (
        coronagraph.coronagraph_spectral_resolution == 1 * DIMENSIONLESS
    )  # should grab from default values

    # Calculated  values
    assert hasattr(coronagraph, "npsfratios")
    assert hasattr(coronagraph, "npix")
    assert hasattr(coronagraph, "xcenter")
    assert hasattr(coronagraph, "ycenter")
    assert hasattr(coronagraph, "r")
    assert hasattr(coronagraph, "omega_lod")
    assert hasattr(coronagraph, "skytrans")
    assert hasattr(coronagraph, "photometric_aperture_throughput")
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

    # Check photometric_aperture_throughput
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.3 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )
    assert np.all(
        coronagraph.photometric_aperture_throughput[
            coronagraph.r < coronagraph.minimum_IWA
        ]
        == 0.0 * DIMENSIONLESS
    )
    assert np.all(
        coronagraph.photometric_aperture_throughput[
            coronagraph.r > coronagraph.maximum_OWA
        ]
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


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")
def test_coronagraph_yip_load_configuration_IMAGER(
    mock_yippycoro,
    mock_load_telescope,
    mock_load_instrument,
    mock_yippy_object,
    mock_instrument,
    mock_telescope,
    capsys,
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

    mediator = MockMediator_IMAGER()

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
    assert hasattr(coronagraph, "photometric_aperture_throughput")
    assert hasattr(coronagraph, "Istar")
    assert hasattr(coronagraph, "noisefloor")

    # Check r
    assert coronagraph.r.shape == (coronagraph.npix, coronagraph.npix)
    # value of r checked in specific test

    # Check omega_lod
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert coronagraph.omega_lod.unit == LAMBDA_D**2
    assert not np.all(coronagraph.omega_lod == 0)

    # Check skytrans
    assert coronagraph.skytrans.shape == (coronagraph.npix, coronagraph.npix)
    assert np.all(
        coronagraph.skytrans == mock_yippy_object.sky_trans.return_value * DIMENSIONLESS
    )
    assert not np.all(coronagraph.skytrans == 0)

    # Check photometric_aperture_throughput
    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert not np.all(coronagraph.photometric_aperture_throughput == 0)

    # Check Istar
    assert coronagraph.Istar.shape == (coronagraph.npix, coronagraph.npix)
    assert not np.all(coronagraph.Istar == 0)

    # Test with noisefloor_factor
    base_parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
    }
    parameters_contrast = base_parameters.copy()
    parameters_contrast["noisefloor_factor"] = 1e-10 * DIMENSIONLESS
    coronagraph.load_configuration(parameters_contrast, mediator)

    captured = capsys.readouterr()
    assert (
        "Setting the noise floor via user-supplied noisefloor_factor..." in captured.out
    )

    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    assert not np.all(coronagraph.noisefloor == 0)

    # Test with noisefloor_PPF
    parameters_ppf = base_parameters.copy()
    parameters_ppf["noisefloor_PPF"] = 300.0
    coronagraph.load_configuration(parameters_ppf, mediator)

    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    assert not np.all(coronagraph.noisefloor == 0)

    captured = capsys.readouterr()
    assert "Setting the noise floor via user-supplied noisefloor_PPF..." in captured.out

    # # Test with neither noisefloor_factor nor noisefloor_PPF
    parameters_null = base_parameters.copy()
    parameters_null["noisefloor_PPF"] = None
    parameters_null["noisefloor_factor"] = None
    coronagraph.load_configuration(parameters_null, mediator)

    assert np.all(coronagraph.noisefloor == 0)
    assert coronagraph.noisefloor.shape == (coronagraph.npix, coronagraph.npix)
    assert coronagraph.noisefloor.unit == DIMENSIONLESS
    captured = capsys.readouterr()
    assert (
        "Neither noisefloor_factor or noisefloor_PPF was specified. Setting noise floor to zero."
        in captured.out
    )

    # Check coronagraph_optical_throughput
    assert len(coronagraph.coronagraph_optical_throughput) == 1
    assert np.isclose(coronagraph.coronagraph_optical_throughput.value, 0.394770896)

    # Check other attributes
    assert hasattr(coronagraph, "psf_trunc_ratio")
    assert hasattr(coronagraph, "npsfratios")
    assert hasattr(coronagraph, "coronagraph_optical_throughput")
    assert hasattr(coronagraph, "coronagraph_spectral_resolution")


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")
def test_coronagraph_yip_load_configuration_IFS(
    mock_yippycoro,
    mock_load_telescope,
    mock_load_instrument,
    mock_yippy_object,
    mock_instrument,
    mock_telescope,
    capsys,
):
    mock_load_instrument.return_value = mock_instrument
    mock_load_telescope.return_value = mock_telescope
    mock_yippycoro.return_value = mock_yippy_object

    coronagraph = CoronagraphYIP(path="test_path")
    parameters = {
        "observing_mode": "IFS",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }

    mediator_ifs = MockMediator_IFS()

    coronagraph.load_configuration(parameters, mediator_ifs)
    captured = capsys.readouterr()
    assert (
        "WARNING: Both psf_trunc_ratio and photometric_aperture_radius are specified. Preferring psf_trunc_ratio going forward..."
        in captured.out
    )
    assert "Using psf_trunc_ratio to calculate Omega..." in captured.out

    # Check coronagraph_optical_throughput
    assert len(coronagraph.coronagraph_optical_throughput) == 3
    assert np.isclose(
        coronagraph.coronagraph_optical_throughput.value,
        [0.41891199, 0.43711322, 0.40535648],
    ).all()


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")
def test_coronagraph_yip_load_configuration_INVALID(
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
        "observing_mode": "Invalid",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
        "az_avg": True,
    }

    mediator = MockMediator_IMAGER()

    with pytest.raises(KeyError, match="Unsupported observing mode: Invalid"):
        coronagraph.load_configuration(parameters, mediator)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")
def test_coronagraph_yip_load_configuration_no_psf_trunc_ratio_no_photometric_aperture_radius(
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
    }

    mediator = MockMediatorNoParams()

    with pytest.raises(
        KeyError,
        match="WARNING: Neither psf_trunc_ratio or photometric_aperture_radius are specified. Specify one or the other to calculate Omega.",
    ):
        coronagraph.load_configuration(parameters, mediator)


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")
def test_coronagraph_yip_load_configuration_with_photometric_aperture_radius(
    mock_yippycoro,
    mock_load_telescope,
    mock_load_instrument,
    mock_yippy_object,
    mock_instrument,
    mock_telescope,
    capsys,
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
        "Tcore": 0.5 * DIMENSIONLESS,
    }

    mediator = MockMediatorWithPhotapRad()

    coronagraph.load_configuration(parameters, mediator)
    captured = capsys.readouterr()
    assert "Using photometric_aperture_radius to calculate Omega..." in captured.out

    # Check that omega_lod and photometric_aperture_throughput are calculated correctly
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(coronagraph.omega_lod == np.pi * 0.7**2 * LAMBDA_D**2)

    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.5 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )
    assert np.all(
        coronagraph.photometric_aperture_throughput[
            coronagraph.r < coronagraph.minimum_IWA
        ]
        == 0.0 * DIMENSIONLESS
    )
    assert np.all(
        coronagraph.photometric_aperture_throughput[
            coronagraph.r > coronagraph.maximum_OWA
        ]
        == 0.0 * DIMENSIONLESS
    )

    # SAME TEST but no Tcore available, use default
    coronagraph = CoronagraphYIP(path="test_path")
    parameters = {
        "observing_mode": "IMAGER",
        "maximum_OWA": 90.0,
        "bandwidth": 0.1,
        "nrolls": 2,
        "nchannels": 1,
    }
    mediator = MockMediatorWithPhotapRad()

    coronagraph.load_configuration(parameters, mediator)
    captured = capsys.readouterr()
    assert "Using photometric_aperture_radius to calculate Omega..." in captured.out

    # Check that omega_lod and photometric_aperture_throughput are calculated correctly
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert np.all(coronagraph.omega_lod == np.pi * 0.7**2 * LAMBDA_D**2)

    assert coronagraph.photometric_aperture_throughput.shape == (
        coronagraph.npix,
        coronagraph.npix,
        1,
    )
    assert np.all(
        (coronagraph.photometric_aperture_throughput == 0.2968371 * DIMENSIONLESS)
        | (coronagraph.photometric_aperture_throughput == 0.0 * DIMENSIONLESS)
    )
    assert np.all(
        coronagraph.photometric_aperture_throughput[
            coronagraph.r < coronagraph.minimum_IWA
        ]
        == 0.0 * DIMENSIONLESS
    )
    assert np.all(
        coronagraph.photometric_aperture_throughput[
            coronagraph.r > coronagraph.maximum_OWA
        ]
        == 0.0 * DIMENSIONLESS
    )


@patch("eacy.load_instrument")
@patch("eacy.load_telescope")
@patch("pyEDITH.components.coronagraphs.yippycoro")
def test_coronagraph_yip_load_configuration_high_psf_trunc_ratio(
    mock_yippycoro,
    mock_load_telescope,
    mock_load_instrument,
    mock_yippy_object,
    mock_instrument,
    mock_telescope,
    capsys,
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
        "Tcore": 0.5 * DIMENSIONLESS,
    }

    mediator = MockMediatorWithHighPSFTruncRatio()

    coronagraph.load_configuration(parameters, mediator)
    captured = capsys.readouterr()
    assert "Using psf_trunc_ratio to calculate Omega..." in captured.out
    assert coronagraph.omega_lod.shape == (coronagraph.npix, coronagraph.npix, 1)
    assert coronagraph.omega_lod.unit == LAMBDA_D**2
    assert np.allclose(coronagraph.omega_lod.value, 0.0025, rtol=1e-6, atol=1e-9)
    # 0.0025 is (1*  (self.DEFAULT_CONFIG["pixscale"] / resolvingfactor) ** 2)    where resolvingfactor = int(np.ceil(self.DEFAULT_CONFIG["pixscale"] / 0.05)
