import numpy as np
import astropy.units as u
from pyEDITH.units import (
    LAMBDA_D,
    DIMENSIONLESS,
    WAVELENGTH,
    FRAME,
    INV_SQUARE_ARCSEC,
    QUANTUM_EFFICIENCY,
    LENGTH,
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


def test_average_over_bandpass():
    params = {
        "lam": np.array([0.4, 0.5, 0.6, 0.7, 0.8]) * WAVELENGTH,
        "value": np.array([1, 2, 3, 4, 5]) * DIMENSIONLESS,
    }
    wavelength_range = [0.45 * WAVELENGTH, 0.75 * WAVELENGTH]

    result = average_over_bandpass(params, wavelength_range)
    assert np.isclose(result["value"].value, 3)


def test_interpolate_over_bandpass():
    params = {
        "lam": np.array([0.4, 0.5, 0.6, 0.7, 0.8]) * WAVELENGTH,
        "value": np.array([1, 2, 3, 4, 5]) * DIMENSIONLESS,
    }
    wavelengths = u.Quantity([0.45, 0.55, 0.65, 0.75], WAVELENGTH)

    result = interpolate_over_bandpass(params, wavelengths)
    assert np.allclose(result["value"], np.array([1.5, 2.5, 3.5, 4.5]))


class TestObject:
    pass


def test_convert_to_numpy_array():
    obj = TestObject()

    # Test with Quantity array
    obj.quantity_array = [1, 2, 3] * u.m

    # Test with non-Quantity array
    obj.regular_array = [4, 5, 6]

    array_params = [
        "quantity_array",
        "regular_array",
    ]

    convert_to_numpy_array(obj, array_params)

    # Check Quantity array
    assert isinstance(obj.quantity_array, u.Quantity)
    assert isinstance(obj.quantity_array.value, np.ndarray)
    assert obj.quantity_array.unit == u.m
    assert np.array_equal(obj.quantity_array.value, np.array([1, 2, 3]))
    assert obj.quantity_array.dtype == np.float64

    # Check non-Quantity array
    assert isinstance(obj.regular_array, np.ndarray)
    assert np.array_equal(obj.regular_array, np.array([4, 5, 6]))
    assert obj.regular_array.dtype == np.float64

    # Test with empty list
    obj.empty_list = []
    convert_to_numpy_array(obj, ["empty_list"])
    assert isinstance(obj.empty_list, np.ndarray)
    assert obj.empty_list.size == 0
    assert obj.empty_list.dtype == np.float64


def test_validate_attributes():
    obj = TestObject()
    obj.int_attr = 1
    obj.float_attr = 1.0
    obj.quantity_attr = 1.0 * u.m
    obj.array_attr = np.array([1, 2, 3]) * u.m

    expected_args = {
        "int_attr": int,
        "float_attr": float,
        "quantity_attr": u.m,
        "array_attr": u.m,
    }

    # Test valid case
    validate_attributes(obj, expected_args)

    # Test missing attribute
    with pytest.raises(
        AttributeError, match="TestObject is missing attribute: missing_attr"
    ):
        validate_attributes(obj, {**expected_args, "missing_attr": int})

    # Test incorrect type for int
    obj.int_attr = 1.0
    with pytest.raises(
        TypeError, match="TestObject attribute int_attr should be an integer"
    ):
        validate_attributes(obj, expected_args)
    obj.int_attr = 1  # Reset to correct type

    # Test incorrect type for float
    obj.float_attr = 1
    with pytest.raises(
        TypeError, match="TestObject attribute float_attr should be a float"
    ):
        validate_attributes(obj, expected_args)
    obj.float_attr = 1.0  # Reset to correct type

    # Test incorrect type for Quantity
    obj.quantity_attr = 1.0
    with pytest.raises(
        TypeError, match="TestObject attribute quantity_attr should be a Quantity"
    ):
        validate_attributes(obj, expected_args)
    obj.quantity_attr = 1.0 * u.m  # Reset to correct type

    # Test incorrect units for Quantity
    obj.quantity_attr = 1.0 * u.s
    with pytest.raises(
        ValueError, match="TestObject attribute quantity_attr has incorrect units"
    ):
        validate_attributes(obj, expected_args)
    obj.quantity_attr = 1.0 * u.m  # Reset to correct units

    # Test unexpected attribute
    with pytest.raises(
        AttributeError, match="TestObject is missing attribute: unexpected_attr"
    ):
        validate_attributes(obj, {**expected_args, "unexpected_attr": "unexpected"})

    # Test unexpected type specification
    with pytest.raises(
        ValueError, match="Unexpected type specification for unexpected_attr"
    ):
        obj.unexpected_attr = 10
        validate_attributes(obj, {**expected_args, "unexpected_attr": "unexpected"})

    # Test array of Quantity
    validate_attributes(
        obj, expected_args
    )  # This should pass as array_attr is already defined correctly

    # Test array of non-Quantity
    obj.array_attr = np.array([1, 2, 3])
    with pytest.raises(
        TypeError, match="TestObject attribute array_attr should be a Quantity"
    ):
        validate_attributes(obj, expected_args)
    obj.array_attr = np.array([1, 2, 3]) * u.m


### TESTING THE PLOTTING FUNCTIONS


def test_print_array_info():
    # Create a mock file object
    mock_file = StringIO()

    # Test array with units
    test_array = np.array([1, 2, 3]) * u.m

    # Call the function
    print_array_info(mock_file, "test_array", test_array, mode="full_info")

    # Get the output
    output = mock_file.getvalue()

    # Check the output
    assert "test_array:" in output
    assert "Unit: m" in output
    assert "Shape: (3,)" in output
    assert "Max value: 3" in output
    assert "Min value: 1" in output

    # Test with an empty numpy array
    empty_array = np.array([])
    file = StringIO()
    print_array_info(file, "empty_numpy_array", empty_array, mode="full_info")
    output = file.getvalue()
    assert "empty_numpy_array:" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output

    # Test with an empty list
    empty_list = []
    file = StringIO()
    print_array_info(file, "empty_list", empty_list, mode="full_info")
    output = file.getvalue()
    assert "empty_list:" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output

    # Test with an empty Quantity array
    empty_quantity = u.Quantity([], unit=u.m)
    file = StringIO()
    print_array_info(file, "empty_quantity", empty_quantity, mode="full_info")
    output = file.getvalue()
    assert "empty_quantity:" in output
    assert "Unit: m" in output
    assert "Shape: (0,)" in output
    assert "Array is empty" in output


@pytest.mark.parametrize("mode", ["validation", "full_info"])
def test_print_all_variables(mode):
    # Create mock objects using DefaultMagicMock
    mock_observation = Observation()
    mock_scene = AstrophysicalScene()
    mock_observatory = Observatory()
    mock_observatory.telescope = ToyModelTelescope()
    mock_observatory.coronagraph = ToyModelCoronagraph()
    mock_observatory.detector = ToyModelDetector()

    # Set some example attributes
    mock_observation.wavelength = [500, 600, 700] * u.nm
    mock_observation.SNR = [10, 10, 10] * DIMENSIONLESS
    mock_observation.td_limit = 24 * u.hour
    mock_observation.CRb_multiplier = 1.0
    mock_observation.fullsnr = [5, 6, 7] * DIMENSIONLESS
    mock_observation.psf_trunc_ratio = 0.3
    mock_observation.exptime = [1000, 1200, 1400] * u.s

    mock_scene.mag = 5.0
    mock_scene.angular_diameter_arcsec = 0.1 * u.arcsec
    mock_scene.F0 = [1e-8, 1e-8, 1e-8] * u.photon / (u.s * u.cm**2 * u.nm)
    mock_scene.Fp0 = [1e-10, 1e-10, 1e-10] * DIMENSIONLESS
    mock_scene.dist = 10 * u.pc
    mock_scene.Fstar = [1e-5, 1e-5, 1e-5]
    mock_scene.Fzodi_list = [1e-7, 1e-7, 1e-7] * INV_SQUARE_ARCSEC
    mock_scene.Fexozodi_list = [1e-8, 1e-8, 1e-8] * INV_SQUARE_ARCSEC
    mock_scene.Fbinary_list = [1e-9, 1e-9, 1e-9] * DIMENSIONLESS
    mock_scene.xp = 0.5 * u.arcsec
    mock_scene.yp = 0.3 * u.arcsec
    mock_scene.separation = 0.6 * u.arcsec

    mock_observatory.telescope.diameter = 2.4 * u.m
    mock_observatory.telescope.temperature = 270 * u.K
    mock_observatory.telescope.toverhead_multi = 1.1
    mock_observatory.telescope.toverhead_fixed = 300 * u.s
    mock_observatory.total_throughput = [0.3, 0.3, 0.3]
    mock_observatory.epswarmTrcold = [0.1, 0.1, 0.1]

    mock_observatory.coronagraph.bandwidth = 0.2
    mock_observatory.coronagraph.Istar = np.ones((10, 10)) * 1e-10 * DIMENSIONLESS
    mock_observatory.coronagraph.noisefloor = np.ones((10, 10)) * 1e-11 * DIMENSIONLESS
    mock_observatory.coronagraph.npix = 100
    mock_observatory.coronagraph.pixscale = 0.1 * u.arcsec / u.pix
    mock_observatory.coronagraph.photap_frac = (
        np.ones((10, 10, 1)) * 0.5 * DIMENSIONLESS
    )
    mock_observatory.coronagraph.skytrans = np.ones((10, 10)) * 0.9 * DIMENSIONLESS
    mock_observatory.coronagraph.omega_lod = np.ones((10, 10, 1)) * 0.1 * LAMBDA_D**2
    mock_observatory.coronagraph.xcenter = 50 * u.pix
    mock_observatory.coronagraph.ycenter = 50 * u.pix
    mock_observatory.coronagraph.nchannels = 2
    mock_observatory.coronagraph.minimum_IWA = 2 * LAMBDA_D
    mock_observatory.coronagraph.maximum_OWA = 10 * LAMBDA_D
    mock_observatory.coronagraph.npsfratios = 1
    mock_observatory.coronagraph.nrolls = 1

    mock_observatory.detector.pixscale_mas = 100 * u.mas
    mock_observatory.detector.QE = np.array([0.8, 0.8, 0.8]) * QUANTUM_EFFICIENCY
    mock_observatory.detector.dQE = np.array([1.0, 1.0, 1.0]) * DIMENSIONLESS
    mock_observatory.detector.npix_multiplier = [1.0, 1.0, 1.0]
    mock_observatory.detector.DC = [1e-3, 1e-3, 1e-3] * u.electron / u.s / u.pix
    mock_observatory.detector.RN = [3, 3, 3] * u.electron / u.pix
    mock_observatory.detector.tread = [100, 100, 100] * u.s
    mock_observatory.detector.CIC = [1e-3, 1e-3, 1e-3] * u.electron / u.pix / FRAME

    # Additional keyword arguments
    kwargs = {
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
        "det_photap_frac": 1 * u.dimensionless_unscaled,
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
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(tmpdirname)
        try:
            # Call the function
            print_all_variables(
                mock_observation, mock_scene, mock_observatory, **kwargs
            )

            # Check that the files were created
            assert os.path.exists(f"pyedith_{mode}.txt")

            # Read the content of the file
            with open(f"pyedith_{mode}.txt", "r") as file:
                content = file.read()

            # Check that the output contains expected sections
            assert "Input Objects and Their Relevant Properties:" in content
            assert "1. Observation:" in content
            assert "2. Scene:" in content
            assert "3. Observatory:" in content
            assert "Telescope:" in content
            assert "Coronagraph:" in content
            assert "Detector:" in content
            assert "Calculated Variables:" in content
            assert "1. Initial Calculations:" in content
            assert "2. Interpolated Arrays:" in content
            assert "3. Coronagraph Performance Measurements:" in content
            assert "4. Detector Noise Calculations:" in content
            assert "5. Planet Position and Separation:" in content
            assert "6. Count Rates and Exposure Time Calculation:" in content
            assert "7. Final Result:" in content

            # Check for some of the explicitly set attributes
            assert "observation.wavelength" in content
            assert "scene.mag" in content
            assert "observatory.telescope.diameter" in content
            assert "observatory.coronagraph.bandwidth" in content
            assert "observatory.detector.pixscale_mas" in content

            # Check for some of the calculated variables
            assert "deltalambda_nm" in content
            assert "lod" in content
            assert "CRp" in content
            assert "CRb" in content

            # Check mode-specific output
            if mode == "full_info":
                assert "Shape:" in content
                assert "Unit:" in content
            else:  # validation mode
                assert "value:" in content
                assert "max value:" in content or "min value:" in content

        finally:
            # Change back to the original directory
            os.chdir(original_dir)
