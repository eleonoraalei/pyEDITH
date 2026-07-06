import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pyEDITH.units import (
    LAMBDA_D,
    DIMENSIONLESS,
    WAVELENGTH,
    FRAME,
    INV_SQUARE_ARCSEC,
    QUANTUM_EFFICIENCY,
    LENGTH,
    PHOTON_FLUX_DENSITY,
)
import pytest
import os
from io import StringIO
import tempfile
from pyEDITH.utils import *
from pyEDITH import (
    Observation,
    AstrophysicalScene,
    Observatory,
)
from pyEDITH.components.telescopes import ToyModelTelescope
from pyEDITH.components.coronagraphs import ToyModelCoronagraph
from pyEDITH.components.detectors import ToyModelDetector

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_params_with_wavelength():
    """Fixture providing sample parameters with wavelength array for testing."""
    return {
        "lam": np.array([0.4, 0.5, 0.6, 0.7, 0.8]) * WAVELENGTH,
        "value": np.array([1, 2, 3, 4, 5]) * DIMENSIONLESS,
    }


@pytest.fixture
def test_object():
    """Fixture providing a fresh test object for each test."""

    class TestObject:
        pass

    return TestObject()


@pytest.fixture
def mock_observation():
    """Fixture providing a configured mock Observation object."""
    obs = Observation()
    obs.wavelength = [500, 600, 700] * u.nm
    obs.SNR = [10, 10, 10] * DIMENSIONLESS
    obs.td_limit = 24 * u.hour
    obs.CRb_multiplier = 1.0
    obs.fullsnr = [5, 6, 7] * DIMENSIONLESS
    obs.exptime = [1000, 1200, 1400] * u.s
    return obs


@pytest.fixture
def mock_scene():
    """Fixture providing a configured mock AstrophysicalScene object."""
    scene = AstrophysicalScene()
    scene.mag = 5.0
    scene.stellar_angular_diameter_arcsec = 0.1 * u.arcsec
    scene.F0 = [1e-8, 1e-8, 1e-8] * u.photon / (u.s * u.cm**2 * u.nm)
    scene.Fp_over_Fs = [1e-10, 1e-10, 1e-10] * DIMENSIONLESS
    scene.dist = 10 * u.pc
    scene.Fs_over_F0 = [1e-5, 1e-5, 1e-5]
    scene.Fzodi_list = [1e-7, 1e-7, 1e-7] * INV_SQUARE_ARCSEC
    scene.Fexozodi_list = [1e-8, 1e-8, 1e-8] * INV_SQUARE_ARCSEC
    scene.Fbinary_list = [1e-9, 1e-9, 1e-9] * DIMENSIONLESS
    scene.xp = 0.5 * u.arcsec
    scene.yp = 0.3 * u.arcsec
    scene.separation = 0.6 * u.arcsec
    return scene


@pytest.fixture
def mock_observatory():
    """Fixture providing a configured mock Observatory object."""
    observatory = Observatory()
    observatory.telescope = ToyModelTelescope()
    observatory.coronagraph = ToyModelCoronagraph()
    observatory.detector = ToyModelDetector()

    observatory.telescope.diameter = 2.4 * u.m
    observatory.telescope.temperature = 270 * u.K
    observatory.telescope.toverhead_multi = 1.1
    observatory.telescope.toverhead_fixed = 300 * u.s
    observatory.total_throughput = [0.3, 0.3, 0.3]
    observatory.epswarmTrcold = [0.1, 0.1, 0.1]

    observatory.coronagraph.bandwidth = 0.2
    observatory.coronagraph.Istar = np.ones((10, 10)) * 1e-10 * DIMENSIONLESS
    observatory.coronagraph.noisefloor = np.ones((10, 10)) * 1e-11 * DIMENSIONLESS
    observatory.coronagraph.npix = 100
    observatory.coronagraph.psf_trunc_ratio = 0.3
    observatory.coronagraph.pixscale = 0.1 * u.arcsec / u.pix
    observatory.coronagraph.photometric_aperture_throughput = (
        np.ones((10, 10, 1)) * 0.5 * DIMENSIONLESS
    )
    observatory.coronagraph.skytrans = np.ones((10, 10)) * 0.9 * DIMENSIONLESS
    observatory.coronagraph.omega_lod = np.ones((10, 10, 1)) * 0.1 * LAMBDA_D**2
    observatory.coronagraph.xcenter = 50 * u.pix
    observatory.coronagraph.ycenter = 50 * u.pix
    observatory.coronagraph.nchannels = 2
    observatory.coronagraph.npsfratios = 1
    observatory.coronagraph.nrolls = 1

    observatory.detector.pixscale_mas = 100 * u.mas
    observatory.detector.QE = np.array([0.8, 0.8, 0.8]) * QUANTUM_EFFICIENCY
    observatory.detector.dQE = np.array([1.0, 1.0, 1.0]) * DIMENSIONLESS
    observatory.detector.npix_multiplier = [1.0, 1.0, 1.0]
    observatory.detector.DC = [1e-3, 1e-3, 1e-3] * u.electron / u.s / u.pix
    observatory.detector.RN = [3, 3, 3] * u.electron / u.pix
    observatory.detector.tread = [100, 100, 100] * u.s
    observatory.detector.CIC = [1e-3, 1e-3, 1e-3] * u.electron / u.pix / FRAME

    return observatory


@pytest.fixture
def validation_kwargs():
    """Fixture providing keyword arguments for validation testing."""
    return {
        "deltalambda_nm": 1 * u.nm,
        "lod": 1 * u.dimensionless_unscaled,
        "lod_rad": 1 * u.rad,
        "lod_arcsec": 1 * u.arcsec,
        "area_cm2": 1 * u.cm**2,
        "detpixscale_lod": 1 * LAMBDA_D,
        "stellar_diam_lod": 1 * LAMBDA_D,
        "pixscale_rad": 1 * u.rad,
        "oneopixscale_arcsec": 1 / u.arcsec,
        "det_sep_pix": 1 * u.pix,
        "det_sep": 1 * u.arcsec,
        "det_Istar": 1 * u.dimensionless_unscaled,
        "det_skytrans": 1 * u.dimensionless_unscaled,
        "det_photometric_aperture_throughput": 1 * u.dimensionless_unscaled,
        "det_omega_lod": 1 * LAMBDA_D**2,
        "det_CRp": 1 * u.electron / u.s,
        "det_CRbs": 1 * u.electron / u.s,
        "det_CRbz": 1 * u.electron / u.s,
        "det_CRbez": 1 * u.electron / u.s,
        "det_CRbbin": 1 * u.electron / u.s,
        "det_CRbth": 1 * u.electron / u.s,
        "det_CR": 1 * u.electron / u.s,
        "ix": 1,
        "iy": 1,
        "sp_lod": 1 * LAMBDA_D,
        "CRp": 1 * u.electron / u.s,
        "CRnf": 1 * u.electron / u.s,
        "CRbs": 1 * u.electron / u.s,
        "CRbz": 1 * u.electron / u.s,
        "CRbez": 1 * u.electron / u.s,
        "CRbbin": 1 * u.electron / u.s,
        "t_photon_count": 1 * u.s,
        "CRbd": 1 * u.electron / u.s,
        "CRbth": 1 * u.electron / u.s,
        "CRb": 1 * u.electron / u.s,
    }


# ============================================================================
# Tests for average_over_bandpass
# ============================================================================


def test_average_over_bandpass_basic(sample_params_with_wavelength):
    """Test basic averaging over bandpass with simple values."""
    wavelength_range = [0.45 * WAVELENGTH, 0.75 * WAVELENGTH]

    result = average_over_bandpass(sample_params_with_wavelength, wavelength_range)

    assert np.isclose(result["value"].value, 3)


def test_average_over_bandpass_preserves_non_array_params(
    sample_params_with_wavelength,
):
    """Test that non-array parameters are preserved unchanged."""
    sample_params_with_wavelength["scalar_param"] = 42
    wavelength_range = [0.45 * WAVELENGTH, 0.75 * WAVELENGTH]

    result = average_over_bandpass(sample_params_with_wavelength, wavelength_range)

    assert result["scalar_param"] == 42


# ============================================================================
# Tests for interpolate_over_bandpass
# ============================================================================


def test_interpolate_over_bandpass_basic(sample_params_with_wavelength):
    """Test basic interpolation over bandpass with simple values."""
    wavelengths = u.Quantity([0.45, 0.55, 0.65, 0.75], WAVELENGTH)

    result = interpolate_over_bandpass(sample_params_with_wavelength, wavelengths)

    assert np.allclose(result["value"], np.array([1.5, 2.5, 3.5, 4.5]))


def test_interpolate_over_bandpass_preserves_non_array_params(
    sample_params_with_wavelength,
):
    """Test that non-array parameters are preserved unchanged."""
    sample_params_with_wavelength["scalar_param"] = 42
    wavelengths = u.Quantity([0.45, 0.55, 0.65, 0.75], WAVELENGTH)

    result = interpolate_over_bandpass(sample_params_with_wavelength, wavelengths)

    assert result["scalar_param"] == 42


# ============================================================================
# Tests for convert_to_numpy_array
# ============================================================================


def test_convert_to_numpy_array_with_quantity(test_object):
    """Test conversion of Quantity list to numpy array preserves units."""
    test_object.quantity_array = [1, 2, 3] * u.m
    array_params = ["quantity_array"]

    convert_to_numpy_array(test_object, array_params)

    assert isinstance(test_object.quantity_array, u.Quantity)
    assert isinstance(test_object.quantity_array.value, np.ndarray)
    assert test_object.quantity_array.unit == u.m
    assert np.array_equal(test_object.quantity_array.value, np.array([1, 2, 3]))
    assert test_object.quantity_array.dtype == np.float64


def test_convert_to_numpy_array_without_quantity(test_object):
    """Test conversion of regular list to numpy array."""
    test_object.regular_array = [4, 5, 6]
    array_params = ["regular_array"]

    convert_to_numpy_array(test_object, array_params)

    assert isinstance(test_object.regular_array, np.ndarray)
    assert np.array_equal(test_object.regular_array, np.array([4, 5, 6]))
    assert test_object.regular_array.dtype == np.float64


def test_convert_to_numpy_array_with_empty_list(test_object):
    """Test conversion of empty list produces empty numpy array."""
    test_object.empty_list = []
    array_params = ["empty_list"]

    convert_to_numpy_array(test_object, array_params)

    assert isinstance(test_object.empty_list, np.ndarray)
    assert test_object.empty_list.size == 0
    assert test_object.empty_list.dtype == np.float64


# ============================================================================
# Tests for validate_attributes - Basic validation
# ============================================================================


def test_validate_attributes_all_valid(test_object):
    """Test that validation passes with all correct attributes."""
    test_object.int_attr = 1
    test_object.float_attr = 1.0
    test_object.quantity_attr = 1.0 * u.m
    test_object.array_attr = np.array([1, 2, 3]) * u.m

    expected_args = {
        "int_attr": int,
        "float_attr": float,
        "quantity_attr": u.m,
        "array_attr": u.m,
    }

    # Should not raise
    validate_attributes(test_object, expected_args)


def test_validate_attributes_quantity_arrays(test_object):
    """Test validation of Quantity arrays."""
    test_object.array_attr = np.array([1, 2, 3]) * u.m

    expected_args = {"array_attr": u.m}

    # Should not raise
    validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Missing attributes
# ============================================================================


def test_validate_attributes_missing_attribute(test_object):
    """Test that missing attribute raises AttributeError."""
    test_object.int_attr = 1

    expected_args = {"int_attr": int, "missing_attr": int}

    with pytest.raises(
        AttributeError, match="TestObject is missing attribute: missing_attr"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Incorrect types
# ============================================================================


def test_validate_attributes_incorrect_int_type(test_object):
    """Test that float instead of int raises TypeError."""
    test_object.int_attr = 1.0

    expected_args = {"int_attr": int}

    with pytest.raises(
        TypeError, match="TestObject attribute int_attr should be an integer"
    ):
        validate_attributes(test_object, expected_args)


def test_validate_attributes_incorrect_float_type(test_object):
    """Test that int instead of float raises TypeError."""
    test_object.float_attr = 1

    expected_args = {"float_attr": float}

    with pytest.raises(
        TypeError, match="TestObject attribute float_attr should be a float"
    ):
        validate_attributes(test_object, expected_args)


def test_validate_attributes_non_quantity_for_quantity_attr(test_object):
    """Test that non-Quantity value for Quantity attribute raises TypeError."""
    test_object.quantity_attr = 1.0

    expected_args = {"quantity_attr": u.m}

    with pytest.raises(
        TypeError, match="TestObject attribute quantity_attr should be a Quantity"
    ):
        validate_attributes(test_object, expected_args)


def test_validate_attributes_non_quantity_array(test_object):
    """Test that non-Quantity array raises TypeError when Quantity expected."""
    test_object.array_attr = np.array([1, 2, 3])

    expected_args = {"array_attr": u.m}

    with pytest.raises(
        TypeError, match="TestObject attribute array_attr should be a Quantity"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Incorrect units
# ============================================================================


def test_validate_attributes_incorrect_units(test_object):
    """Test that Quantity with incorrect units raises ValueError."""
    test_object.quantity_attr = 1.0 * u.s

    expected_args = {"quantity_attr": u.m}

    with pytest.raises(
        ValueError, match="TestObject attribute quantity_attr has incorrect units"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for validate_attributes - Unexpected specifications
# ============================================================================


def test_validate_attributes_unexpected_type_specification(test_object):
    """Test that unexpected type specification raises ValueError."""
    test_object.unexpected_attr = 10

    expected_args = {"unexpected_attr": "unexpected"}

    with pytest.raises(
        ValueError, match="Unexpected type specification for unexpected_attr"
    ):
        validate_attributes(test_object, expected_args)


# ============================================================================
# Tests for print_array_info - Full info mode
# ============================================================================


def test_print_array_info_with_units():
    """Test printing array info with units in full_info mode."""
    mock_file = StringIO()
    test_array = np.array([1, 2, 3]) * u.m

    print_array_info(mock_file, "test_array", test_array, mode="full_info")

    output = mock_file.getvalue()
    assert "test_array:" in output
    assert "Unit: m" in output
    assert "Shape: (3,)" in output
    assert "Max value: 3" in output
    assert "Min value: 1" in output


def test_print_array_info_none_input():
    """Test that function returns early when None is passed."""
    file = StringIO()
    print_array_info(file, "none_input", None)
    output = file.getvalue()
    # Should produce no output when arr is None
    assert output == ""


def test_print_array_info_empty_numpy_array():
    """Test printing info for empty numpy array."""
    empty_array = np.array([])
    file = StringIO()

    print_array_info(file, "empty_numpy_array", empty_array, mode="full_info")

    output = file.getvalue()
    assert "empty_numpy_array:" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


def test_print_array_info_empty_list():
    """Test printing info for empty list."""
    empty_list = []
    file = StringIO()

    print_array_info(file, "empty_list", empty_list, mode="full_info")

    output = file.getvalue()
    assert "empty_list:" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


def test_print_array_info_empty_quantity():
    """Test printing info for empty Quantity array."""
    empty_quantity = u.Quantity([], unit=u.m)
    file = StringIO()

    print_array_info(file, "empty_quantity", empty_quantity, mode="full_info")

    output = file.getvalue()
    assert "empty_quantity:" in output
    assert "Unit: m" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


def test_print_array_info_integer_scalar():
    """Test printing info for integer scalar in full_info mode."""
    file = StringIO()
    int_array = np.array(42)

    print_array_info(file, "int_scalar", int_array, mode="full_info")

    output = file.getvalue()
    assert "int_scalar:" in output
    assert "Shape: scalar" in output
    assert "Value: 42" in output


def test_print_array_info_float_scalar_full_info():
    """Test printing info for float scalar in full_info mode."""
    file = StringIO()
    test_array = np.array(3.14)

    print_array_info(file, "test_scalar", test_array, mode="full_info")

    output = file.getvalue()
    assert "test_scalar:" in output
    assert "Shape: scalar" in output


def test_print_array_info_none_scalar_full_info():
    """Test printing info for None scalar in full_info mode."""
    file = StringIO()
    none_array = np.array(None)

    print_array_info(file, "none_scalar", none_array, mode="full_info")

    output = file.getvalue()
    assert "none_scalar:" in output
    assert "Value: None" in output


# ============================================================================
# Tests for print_array_info - Validation mode
# ============================================================================


def test_print_array_info_float_scalar_validation():
    """Test printing info for float scalar in validation mode."""
    file = StringIO()
    test_array = np.array(3.14)

    print_array_info(file, "test_scalar", test_array, mode="validation")

    output = file.getvalue()
    assert "test_scalar:" in output
    assert "value: 3.14" in output


def test_print_array_info_none_scalar_validation():
    """Test printing info for None scalar in validation mode."""
    file = StringIO()
    none_array = np.array(None)

    print_array_info(file, "none_scalar_val", none_array, mode="validation")

    output = file.getvalue()
    assert "none_scalar_val:" in output
    assert "value: None" in output


# ============================================================================
# Tests for print_all_variables
# ============================================================================


def test_print_all_variables_creates_files(
    mock_observation, mock_scene, mock_observatory
):
    """Test that print_all_variables creates both output files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            # A single call writes BOTH files via the internal `for mode` loop.
            assert os.path.exists("pyedith_validation.txt")
            assert os.path.exists("pyedith_full_info.txt")

        finally:
            os.chdir(original_dir)


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables_file_structure(
    mode, mock_observation, mock_scene, mock_observatory
):
    """Test that output file contains all expected sections."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check for main sections
            assert "Input Objects and Their Relevant Properties:" in content
            assert "1. Observation:" in content
            assert "2. Scene:" in content
            assert "3. Observatory:" in content
            assert "Telescope:" in content
            assert "Coronagraph:" in content
            assert "Detector:" in content
            assert "Calculated Variables:" in content

        finally:
            os.chdir(original_dir)


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables_includes_attributes(
    mode, mock_observation, mock_scene, mock_observatory
):
    """Test that output file includes specific object attributes."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check for some explicitly set attributes
            assert "observation.wavelength" in content
            assert "scene.mag" in content
            assert "observatory.telescope.diameter" in content
            assert "observatory.coronagraph.bandwidth" in content
            assert "observatory.detector.pixscale_mas" in content

        finally:
            os.chdir(original_dir)


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables_includes_calculated_vars(
    mode, mock_observation, mock_scene, mock_observatory
):
    """Test that output file includes calculated variables read from
    observation.validation_variables."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check for calculated variables
            assert "deltalambda_nm" in content
            assert "lod" in content
            assert "CRp" in content
            assert "CRb" in content

        finally:
            os.chdir(original_dir)


def test_print_all_variables_mode_specific_output_full_info(
    mock_observation, mock_scene, mock_observatory
):
    """Test that full_info mode includes shape and unit information."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open("pyedith_full_info.txt", "r") as file:
                content = file.read()

            assert "Shape:" in content
            assert "Unit:" in content

        finally:
            os.chdir(original_dir)


def test_print_all_variables_mode_specific_output_validation(
    mock_observation, mock_scene, mock_observatory
):
    """Test that validation mode includes value information."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            mock_observation.validation_variables = {
                "deltalambda_nm": np.array([10.0]),
                "lod": np.array([0.05]),
                "CRp": np.array([1.0]),
                "CRbs": np.array([0.5]),
                "CRb": np.array([2.0]),
            }

            print_all_variables(mock_observation, mock_scene, mock_observatory)

            with open("pyedith_validation.txt", "r") as file:
                content = file.read()

            assert "value:" in content
            assert "max value:" in content or "min value:" in content

        finally:
            os.chdir(original_dir)


# ============================================================================
# Tests for synthesize_observation
# ============================================================================


def test_synthesize_observation_basic():
    """Test basic observation synthesis returns expected shapes."""
    mock_scene = AstrophysicalScene()
    snr_arr = np.array([10, 15, 20])
    mock_scene.Fp_over_Fs = np.array([1e-6, 1.5e-6, 2e-6])

    obs, noise = synthesize_observation(snr_arr, mock_scene)

    assert obs.shape == (3,)
    assert noise.shape == (3,)
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(noise))


def test_synthesize_observation_reproducible_with_seed():
    """Test that random seed produces reproducible results."""
    mock_scene = AstrophysicalScene()
    snr_arr = np.array([10, 15, 20])
    mock_scene.Fp_over_Fs = np.array([1e-6, 1.5e-6, 2e-6])

    obs1, noise1 = synthesize_observation(snr_arr, mock_scene, random_seed=42)
    obs2, noise2 = synthesize_observation(snr_arr, mock_scene, random_seed=42)

    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(noise1, noise2)


def test_synthesize_observation_below_zero_handling():
    """Test that negative observations are handled with set_below_zero parameter."""
    mock_scene = AstrophysicalScene()
    snr_arr = np.array([10, 15, 20])
    mock_scene.Fp_over_Fs = np.array([1e-6, 1.5e-6, 2e-6])

    obs, noise = synthesize_observation(snr_arr, mock_scene, set_below_zero=-999)

    assert np.all(obs[obs < 0] == -999)


# ============================================================================
# Tests for wavelength_grid_fixed_res
# ============================================================================


def test_wavelength_grid_fixed_res_basic():
    """Test basic wavelength grid generation at fixed resolution."""
    x_min, x_max, res = 0.5, 1.0, 100

    x, Dx = wavelength_grid_fixed_res(x_min, x_max, res)

    assert x[0] == x_min
    assert x[-1] < x_max
    assert len(x) == len(Dx)
    assert np.all(np.diff(x) > 0)


def test_wavelength_grid_fixed_res_maintains_resolution():
    """Test that wavelength grid maintains constant resolution."""
    x_min, x_max, res = 0.5, 1.0, 100

    x, Dx = wavelength_grid_fixed_res(x_min, x_max, res)

    np.testing.assert_allclose(x / Dx, res, rtol=1e-5)


# ============================================================================
# Tests for gen_wavelength_grid
# ============================================================================


def test_gen_wavelength_grid_single_channel():
    """Test wavelength grid generation for single spectral channel."""
    x_min, x_max, res = [0.5], [1.0], [100]

    x, Dx = gen_wavelength_grid(x_min, x_max, res)

    assert x[0] == x_min[0]
    assert x[-1] < x_max[0]
    assert len(x) == len(Dx)
    assert np.all(np.diff(x) > 0)


def test_gen_wavelength_grid_multiple_channels():
    """Test wavelength grid generation for multiple spectral channels."""
    x_min, x_max, res = [0.5, 1.0], [1.0, 2.0], [100, 200]

    x, Dx = gen_wavelength_grid(x_min, x_max, res)

    assert x[0] == x_min[0]
    assert x[-1] < x_max[-1]
    assert len(x) == len(Dx)
    assert np.all(np.diff(x) > 0)


# ============================================================================
# Tests for regrid_wavelengths
# ============================================================================


def test_regrid_wavelengths_with_boundaries():
    """Test wavelength regridding with specified channel boundaries."""
    input_wls = np.linspace(0.2, 2.0, 100)
    res = [50, 100, 150]
    lam_low = [0.3, 0.5, 1.0]
    lam_high = [0.5, 1.0, 1.7]

    lam, dlam = regrid_wavelengths(input_wls, res, lam_low, lam_high)

    assert np.all(np.diff(lam) > 0)
    assert len(lam) == len(dlam)


def test_regrid_wavelengths_without_boundaries():
    """Test wavelength regridding without channel boundaries."""
    input_wls = np.linspace(0.2, 2.0, 100)

    lam, dlam = regrid_wavelengths(input_wls, [100], None, None)

    assert len(lam) > 0
    assert len(dlam) > 0


def test_regrid_wavelengths_lower_boundary_outside_range():
    """Test that lower boundary outside input range raises AssertionError."""
    input_wls = np.linspace(0.2, 2.0, 100)

    with pytest.raises(
        AssertionError,
        match="Your minimum input wavelength is greater than first channel lower boundary.",
    ):
        regrid_wavelengths(input_wls, [100, 200], [0.1, 1.0], [1.0, 1.7])


def test_regrid_wavelengths_upper_boundary_outside_range():
    """Test that upper boundary outside input range raises AssertionError."""
    input_wls = np.linspace(0.2, 2.0, 100)

    with pytest.raises(
        AssertionError,
        match="Your maximum input wavelength is less than last channel upper boundary.",
    ):
        regrid_wavelengths(input_wls, [100, 200], [0.5, 1.0], [1.0, 2.1])


def test_regrid_wavelengths_single_resolution():
    """Test wavelength regridding with single resolution value."""
    input_wls = np.linspace(0.2, 2.0, 100)

    lam, dlam = regrid_wavelengths(input_wls, [100])

    assert len(lam) > 0
    assert len(dlam) > 0


# ============================================================================
# Tests for regrid_spec_gaussconv
# ============================================================================


def test_regrid_spec_gaussconv_basic():
    """Test Gaussian convolution regridding produces correct output length."""
    input_wls = np.linspace(0.4, 2.0, 100)
    input_spec = np.random.rand(100) * PHOTON_FLUX_DENSITY
    new_lam = np.linspace(0.5, 1.9, 50)
    new_dlam = np.gradient(new_lam)

    spec_regrid = regrid_spec_gaussconv(input_wls, input_spec, new_lam, new_dlam)

    assert len(spec_regrid) == len(new_lam)


def test_regrid_spec_gaussconv_preserves_units():
    """Test that Gaussian convolution regridding preserves units."""
    input_wls = np.linspace(0.4, 2.0, 100)
    input_spec = np.random.rand(100) * PHOTON_FLUX_DENSITY
    new_lam = np.linspace(0.5, 1.9, 50)
    new_dlam = np.gradient(new_lam)

    spec_regrid = regrid_spec_gaussconv(input_wls, input_spec, new_lam, new_dlam)

    assert spec_regrid.unit == input_spec.unit


# ============================================================================
# Tests for regrid_spec_interp
# ============================================================================


def test_regrid_spec_interp_basic():
    """Test interpolation regridding produces correct output length."""
    input_wls = np.linspace(0.4, 2.0, 100) * WAVELENGTH
    input_spec = np.random.rand(100) * PHOTON_FLUX_DENSITY
    new_lam = np.linspace(0.5, 1.9, 50) * WAVELENGTH

    spec_regrid = regrid_spec_interp(input_wls, input_spec, new_lam)

    assert isinstance(spec_regrid, u.Quantity)
    assert len(spec_regrid) == len(new_lam)


def test_regrid_spec_interp_preserves_units():
    """Test that interpolation regridding preserves units."""
    input_wls = np.linspace(0.4, 2.0, 100) * WAVELENGTH
    input_spec = np.random.rand(100) * PHOTON_FLUX_DENSITY
    new_lam = np.linspace(0.5, 1.9, 50) * WAVELENGTH

    spec_regrid = regrid_spec_interp(input_wls, input_spec, new_lam)

    assert spec_regrid.unit == input_spec.unit
