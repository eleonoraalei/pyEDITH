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
    observatory.detector.npix_multiplier = 1.0 * DIMENSIONLESS
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
    input_spec = np.random.rand(100)
    new_lam = np.linspace(0.5, 1.9, 50)
    new_dlam = np.gradient(new_lam)

    spec_regrid = regrid_spec_gaussconv(input_wls, input_spec, new_lam, new_dlam)

    assert len(spec_regrid) == len(new_lam)


# ============================================================================
# Tests for regrid_spec_interp
# ============================================================================


def test_regrid_spec_interp_basic():
    """Test interpolation regridding produces correct output length."""
    input_wls = np.linspace(0.4, 2.0, 100)
    input_spec = np.random.rand(100)
    new_lam = np.linspace(0.5, 1.9, 50)

    spec_regrid = regrid_spec_interp(input_wls, input_spec, new_lam)

    assert len(spec_regrid) == len(new_lam)


# ============================================================================
# Tests for regrid_to_grid
# ============================================================================


def test_regrid_to_grid_broadcast_scalar():
    """Test that a single value is broadcast to the entire grid."""
    values = np.array([1.5])
    from_wavelength = np.array([1.0])
    to_wavelength = np.linspace(0.5, 2.0, 100)

    result = regrid_to_grid(values, from_wavelength, to_wavelength)

    assert len(result) == len(to_wavelength)
    assert np.all(result == 1.5)


def test_regrid_to_grid_passthrough():
    """Test that values already on the target grid pass through unchanged."""
    to_wavelength = np.linspace(0.5, 2.0, 50)
    values = np.random.rand(50)
    from_wavelength = to_wavelength.copy()

    result = regrid_to_grid(values, from_wavelength, to_wavelength)

    assert len(result) == len(to_wavelength)
    np.testing.assert_array_equal(result, values)


def test_regrid_to_grid_interpolation_1d():
    """Test regridding using 1D interpolation."""
    from_wavelength = np.linspace(0.4, 2.0, 100)
    values = np.random.rand(100)
    to_wavelength = np.linspace(0.5, 1.9, 50)

    result = regrid_to_grid(values, from_wavelength, to_wavelength, interpolation="1d")

    assert len(result) == len(to_wavelength)
    assert result.dtype == np.float64


def test_regrid_to_grid_interpolation_gaussian():
    """Test regridding using Gaussian convolution."""
    from_wavelength = np.linspace(0.4, 2.0, 100)
    values = np.random.rand(100)
    to_wavelength = np.linspace(0.5, 1.9, 50)
    to_delta_wavelength = np.gradient(to_wavelength)

    result = regrid_to_grid(
        values,
        from_wavelength,
        to_wavelength,
        to_delta_wavelength,
        interpolation="Gaussian",
    )

    assert len(result) == len(to_wavelength)
    assert result.dtype == np.float64


def test_regrid_to_grid_gaussian_missing_delta_wavelength():
    """Test that Gaussian interpolation raises error without delta_wavelength."""
    from_wavelength = np.linspace(0.4, 2.0, 100)
    values = np.random.rand(100)
    to_wavelength = np.linspace(0.5, 1.9, 50)

    with pytest.raises(ValueError, match="to_delta_wavelength.*not provided"):
        regrid_to_grid(values, from_wavelength, to_wavelength, interpolation="Gaussian")


def test_regrid_to_grid_invalid_interpolation():
    """Test that invalid interpolation method raises error."""
    values = np.array([1.0, 2.0, 3.0])
    from_wavelength = np.array([0.5, 1.0, 1.5])
    to_wavelength = np.linspace(0.5, 1.5, 10)

    with pytest.raises(ValueError, match="Unknown interpolation type"):
        regrid_to_grid(
            values, from_wavelength, to_wavelength, interpolation="invalid_method"
        )


def test_regrid_to_grid_with_quantity():
    """Test that astropy Quantity units are preserved."""
    values = np.array([1.0, 2.0, 3.0]) * u.Jy
    from_wavelength = np.array([0.5, 1.0, 1.5])
    to_wavelength = np.linspace(0.5, 1.5, 10)

    result = regrid_to_grid(values, from_wavelength, to_wavelength, interpolation="1d")

    assert isinstance(result, u.Quantity)
    assert result.unit == u.Jy
    assert len(result) == len(to_wavelength)


def test_regrid_to_grid_broadcast_quantity():
    """Test that a single Quantity value is broadcast correctly."""
    values = np.array([2.5]) * u.erg / u.s / u.cm**2
    from_wavelength = np.array([1.0])
    to_wavelength = np.linspace(0.5, 2.0, 50)

    result = regrid_to_grid(values, from_wavelength, to_wavelength)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.erg / u.s / u.cm**2
    assert len(result) == len(to_wavelength)
    assert np.all(result.value == 2.5)


def test_regrid_to_grid_name_parameter():
    """Test that custom name appears in log messages (requires log capture)."""
    # This test ensures the name parameter is accepted
    values = np.random.rand(10)
    from_wavelength = np.linspace(0.5, 1.5, 10)
    to_wavelength = np.linspace(0.5, 1.5, 20)

    result = regrid_to_grid(
        values, from_wavelength, to_wavelength, name="custom_flux", interpolation="1d"
    )

    assert len(result) == len(to_wavelength)


def test_regrid_to_grid_dtype_conversion():
    """Test that output is always float64."""
    values = np.array([1, 2, 3], dtype=np.int32)
    from_wavelength = np.array([0.5, 1.0, 1.5])
    to_wavelength = np.linspace(0.5, 1.5, 10)

    result = regrid_to_grid(values, from_wavelength, to_wavelength, interpolation="1d")

    assert result.dtype == np.float64


# ============================================================================
# Tests for fill_parameters
# ============================================================================


def test_fill_parameters_basic_default_values(test_object):
    """Test that default parameters are correctly assigned when no user params provided."""
    parameters = {}
    default_parameters = {
        "param1": 10,
        "param2": 20.5,
        "param3": "test_string",
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert test_object.param1 == 10
    assert test_object.param2 == 20.5
    assert test_object.param3 == "test_string"


def test_fill_parameters_user_override(test_object):
    """Test that user-provided parameters override defaults."""
    parameters = {
        "param1": 100,
        "param2": 200.5,
    }
    default_parameters = {
        "param1": 10,
        "param2": 20.5,
        "param3": "default_string",
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert test_object.param1 == 100
    assert test_object.param2 == 200.5
    assert test_object.param3 == "default_string"


def test_fill_parameters_locked_keys_prevent_override(test_object, caplog):
    """Test that locked keys cannot be overridden by user parameters."""
    parameters = {
        "locked_param": 999,
        "unlocked_param": 50,
    }
    default_parameters = {
        "locked_param": 42,
        "unlocked_param": 25,
    }
    locked_keys = {"locked_param"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(test_object, parameters, default_parameters, locked_keys)

    # Locked parameter should retain default value
    assert test_object.locked_param == 42
    # Unlocked parameter should use user value
    assert test_object.unlocked_param == 50
    # Warning should be logged
    assert "locked_param" in caplog.text
    assert "locked in this mode" in caplog.text


def test_fill_parameters_quantity_unit_conversion(test_object):
    """Test that Quantity parameters with matching units are converted correctly."""
    parameters = {
        "distance": 5.0 * u.pc,  # parsecs
    }
    default_parameters = {
        "distance": 10.0 * u.m,  # meters (different unit)
    }

    fill_parameters(test_object, parameters, default_parameters)

    # Should be converted to meters (the default unit)
    assert isinstance(test_object.distance, u.Quantity)
    assert test_object.distance.unit == u.m
    # 5 pc = 1.54285714e+17 m
    assert np.isclose(test_object.distance.value, 1.54285714e17, rtol=1e-4)


def test_fill_parameters_quantity_without_units(test_object):
    """Test that unitless user value gets assigned default units from Quantity default."""
    parameters = {
        "length": 100.0,  # No units
    }
    default_parameters = {
        "length": 1.0 * u.m,  # Has units
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert isinstance(test_object.length, u.Quantity)
    assert test_object.length.unit == u.m
    assert test_object.length.value == 100.0


def test_fill_parameters_quantity_user_override_with_units(test_object):
    """Test that user Quantity with units correctly overrides default."""
    parameters = {
        "wavelength": 500 * u.nm,
    }
    default_parameters = {
        "wavelength": 1.0 * u.um,  # Different value and unit
    }

    fill_parameters(test_object, parameters, default_parameters)

    assert isinstance(test_object.wavelength, u.Quantity)
    # Should be converted to microns (the default unit)
    assert test_object.wavelength.unit == u.um
    assert np.isclose(test_object.wavelength.value, 0.5)


def test_fill_parameters_mixed_locked_and_unlocked_quantities(test_object, caplog):
    """Test handling of both locked and unlocked Quantity parameters."""
    parameters = {
        "locked_distance": 100 * u.m,
        "unlocked_distance": 200 * u.m,
    }
    default_parameters = {
        "locked_distance": 10 * u.m,
        "unlocked_distance": 20 * u.m,
    }
    locked_keys = {"locked_distance"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(test_object, parameters, default_parameters, locked_keys)

    # Locked should use default
    assert test_object.locked_distance == 10 * u.m
    # Unlocked should use user value
    assert test_object.unlocked_distance == 200 * u.m
    # Warning should be logged only for locked parameter
    assert "locked_distance" in caplog.text


def test_fill_parameters_empty_locked_keys(test_object):
    """Test that passing None for locked_keys works correctly."""
    parameters = {
        "param1": 100,
    }
    default_parameters = {
        "param1": 10,
        "param2": 20,
    }

    # Should not raise any errors with locked_keys=None (default)
    fill_parameters(test_object, parameters, default_parameters, locked_keys=None)

    assert test_object.param1 == 100
    assert test_object.param2 == 20


def test_fill_parameters_all_locked_keys(test_object, caplog):
    """Test behavior when all parameters are locked."""
    parameters = {
        "param1": 999,
        "param2": 888,
    }
    default_parameters = {
        "param1": 10,
        "param2": 20,
    }
    locked_keys = {"param1", "param2"}

    with caplog.at_level(logging.WARNING):
        fill_parameters(test_object, parameters, default_parameters, locked_keys)

    # All should use defaults
    assert test_object.param1 == 10
    assert test_object.param2 == 20
    # Should have warnings for both
    assert "param1" in caplog.text
    assert "param2" in caplog.text
