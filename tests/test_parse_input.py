import pytest
import numpy as np
from astropy import units as u
from pathlib import Path
import tempfile
import os
import logging

from pyEDITH.parse_input import *
from pyEDITH.units import WAVELENGTH, DIMENSIONLESS, LENGTH

# ============================================================================
# Fixtures - Temporary input files
# ============================================================================


@pytest.fixture
def sample_input_file():
    """Fixture providing a sample input file with valid parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        ; This is a comment
        wavelength = 0.5
        distance = 10
        magV = 5.0
        nzodis = 3.0
        observing_mode = IMAGER
        secondary_wavelength = 1.0
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_input_file_imager_multi_wavelength():
    """Fixture providing an IMAGER mode input file with multiple wavelengths (invalid)."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        ; This is a comment
        wavelength = [0.5, 0.6]
        distance = 10
        magV = 5.0
        nzodis = 3.0
        observing_mode = IMAGER
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_valid():
    """Fixture providing a valid IFS mode input file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_missing_keys():
    """Fixture providing an IFS mode input file with missing required keys."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def ifs_input_file_mismatched_lengths():
    """Fixture providing an IFS mode input file with mismatched column lengths."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = 'IFS'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def valid_spectrum_file():
    """Fixture providing a valid spectrum CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write(
            "wavelength,Fstar_10pc,Fp/Fs\n0.5,1e-9,1e-11\n0.6,1e-9,1e-11\n0.7,1e-8,1e-10"
        )
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def spectrum_file_invalid_columns():
    """Fixture providing a spectrum file with incorrect number of columns."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("wavelength,Fstar_10pc\n0.5,1e-8")
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def spectrum_file_non_numeric():
    """Fixture providing a spectrum file with non-numeric values."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("wavelength,Fstar_10pc,Fp/Fs\n0.5,1e-8,invalid")
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


# ============================================================================
# Tests for parse_input_file - Basic functionality
# ============================================================================


def test_parse_input_file_basic(sample_input_file):
    """Test parsing a basic input file with valid parameters."""
    variables, secondary_variables = parse_input_file(
        sample_input_file, secondary_flag=True
    )

    assert variables["wavelength"] == 0.5
    assert variables["distance"] == 10
    assert variables["magV"] == 5.0
    assert variables["nzodis"] == 3.0
    assert variables["observing_mode"] == "IMAGER"
    assert secondary_variables["wavelength"] == 1.0


def test_parse_input_file_secondary_prefix_removed(sample_input_file):
    """Test that 'secondary_' prefix is removed from secondary variable keys."""
    variables, secondary_variables = parse_input_file(
        sample_input_file, secondary_flag=True
    )

    # Should have 'wavelength', not 'secondary_wavelength'
    assert "wavelength" in secondary_variables
    assert "secondary_wavelength" not in secondary_variables


def test_parse_input_file_ifs_valid(ifs_input_file_valid):
    """Test parsing a valid IFS mode input file."""
    variables, secondary_variables = parse_input_file(
        ifs_input_file_valid, secondary_flag=False
    )

    assert np.all(variables["wavelength"] == [0.5, 0.6, 0.7])
    assert np.all(variables["Fstar_10pc"] == [1e-8, 1e-8, 1e-8])
    assert np.all(variables["Fp/Fs"] == [1e-10, 1e-10, 1e-10])
    assert variables["nlambda"] == 3


# ============================================================================
# Tests for parse_input_file - Error handling
# ============================================================================


def test_parse_input_file_imager_multi_wavelength_error(
    sample_input_file_imager_multi_wavelength,
):
    """Test that IMAGER mode with multiple wavelengths raises KeyError."""
    with pytest.raises(
        KeyError, match="In IMAGER mode you can only use one wavelength at a time"
    ):
        parse_input_file(
            sample_input_file_imager_multi_wavelength, secondary_flag=False
        )


def test_parse_input_file_secondary_flag_no_secondary_vars(
    sample_input_file_imager_multi_wavelength,
):
    """Test that secondary_flag=True without secondary variables raises KeyError."""
    with pytest.raises(
        KeyError,
        match="Secondary flag is True but no secondary variables found in the input file",
    ):
        parse_input_file(sample_input_file_imager_multi_wavelength, secondary_flag=True)


def test_parse_input_file_ifs_missing_keys(ifs_input_file_missing_keys):
    """Test that IFS mode with missing required keys raises ValueError."""
    with pytest.raises(
        KeyError,
        match="Required parameters 'wavelength', 'Fstar_10pc', and 'Fp/Fs' are not provided",
    ):
        parse_input_file(ifs_input_file_missing_keys, secondary_flag=False)


def test_parse_input_file_ifs_mismatched_lengths(ifs_input_file_mismatched_lengths):
    """Test that IFS mode with mismatched column lengths raises ValueError."""
    with pytest.raises(
        ValueError,
        match="All of wavelength, Fstar_10pc, Fp/Fs must have the same length",
    ):
        parse_input_file(ifs_input_file_mismatched_lengths, secondary_flag=False)


def test_parse_input_file_invalid_spectrum_file():
    """Test that non-existent spectrum file raises FileNotFoundError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write("""
        observing_mode = "IFS"
        spectrum_file = 'nonexistent_file.csv'
        """)
        tmp.flush()

        with pytest.raises(
            FileNotFoundError, match="Spectrum file not found: nonexistent_file.csv"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_spectrum_file_invalid_columns(spectrum_file_invalid_columns):
    """Test that spectrum file with incorrect number of columns raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = 'IFS'
        spectrum_file = '{spectrum_file_invalid_columns}'
        """)
        tmp.flush()

        with pytest.raises(
            ValueError, match="Spectrum file must contain exactly 3 columns"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


def test_parse_input_file_spectrum_file_non_numeric(spectrum_file_non_numeric):
    """Test that spectrum file with non-numeric values raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = 'IFS'
        spectrum_file = '{spectrum_file_non_numeric}'
        """)
        tmp.flush()

        with pytest.raises(
            ValueError, match="Column 'Fp/Fs' contains non-numeric values"
        ):
            parse_input_file(tmp.name, secondary_flag=False)

        os.unlink(tmp.name)


# ============================================================================
# Tests for parse_input_file - Spectrum file integration
# ============================================================================


def test_parse_input_file_with_valid_spectrum_file(valid_spectrum_file):
    """Test parsing input file with valid spectrum file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(f"""
        observing_mode = 'IFS'
        spectrum_file = '{valid_spectrum_file}'
        """)
        tmp.flush()

        variables, _ = parse_input_file(tmp.name, secondary_flag=False)

        assert variables["observing_mode"] == "IFS"
        assert "wavelength" in variables
        assert "Fstar_10pc" in variables
        assert "Fp/Fs" in variables
        assert len(variables["wavelength"]) == 3
        assert variables["nlambda"] == 3  # Should be set by parse_input_file
        np.testing.assert_almost_equal(variables["wavelength"], [0.5, 0.6, 0.7])
        np.testing.assert_almost_equal(variables["Fstar_10pc"], [1e-9, 1e-9, 1e-8])
        np.testing.assert_almost_equal(variables["Fp/Fs"], [1e-11, 1e-11, 1e-10])

        os.unlink(tmp.name)


# ============================================================================
# Tests for normalize_list_shapes
# ============================================================================


def test_normalize_list_shapes_scalar_single_wavelength():
    """Test scalar value with single wavelength (default_len=1)."""
    parameters = {"snr": 10.0}
    result = normalize_list_shapes(parameters, "snr", default_len=1)

    np.testing.assert_array_equal(result, np.array([10.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_scalar_multiple_wavelengths(caplog):
    """Test scalar value broadcast to multiple wavelengths (default_len>1)."""
    parameters = {"snr": 10.0}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 10.0, 10.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_single_element_list_broadcast(caplog):
    """Test single-element list broadcast to multiple wavelengths."""
    parameters = {"snr": [10.0]}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 10.0, 10.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_matching_length_list():
    """Test list with correct length passes through."""
    parameters = {"snr": [10.0, 20.0, 30.0]}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_mismatched_length_error():
    """Test list with wrong length raises ValueError."""
    parameters = {"snr": [10.0, 20.0]}

    with pytest.raises(
        ValueError, match="snr should be a list of length 3, but it has length 2"
    ):
        normalize_list_shapes(parameters, "snr", default_len=3)


def test_normalize_list_shapes_quantity_scalar_single_wavelength():
    """Test Quantity scalar with single wavelength."""
    parameters = {"DC": 10.0 * DARK_CURRENT}
    result = normalize_list_shapes(parameters, "DC", default_len=1)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0]))
    assert result.unit == DARK_CURRENT


def test_normalize_list_shapes_quantity_scalar_broadcast(caplog):
    """Test Quantity scalar broadcast to multiple wavelengths."""
    parameters = {"DC": 10.0 * DARK_CURRENT}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "DC", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0, 10.0, 10.0]))
    assert result.unit == DARK_CURRENT
    assert any(
        "DC should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_quantity_single_element_broadcast(caplog):
    """Test single-element Quantity array broadcast to multiple wavelengths."""
    parameters = {"DC": u.Quantity([10.0], DARK_CURRENT)}

    with caplog.at_level(logging.WARNING, logger="pyEDITH"):
        result = normalize_list_shapes(parameters, "DC", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0, 10.0, 10.0]))
    assert result.unit == DARK_CURRENT
    assert any(
        "DC should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_normalize_list_shapes_quantity_matching_length():
    """Test Quantity array with correct length passes through."""
    parameters = {"DC": u.Quantity([10.0, 20.0, 30.0], DARK_CURRENT)}
    result = normalize_list_shapes(parameters, "DC", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([10.0, 20.0, 30.0]))
    assert result.unit == DARK_CURRENT


def test_normalize_list_shapes_quantity_mismatched_length():
    """Test Quantity array with wrong length raises ValueError."""
    parameters = {"DC": u.Quantity([10.0, 20.0], DARK_CURRENT)}

    with pytest.raises(
        ValueError, match="DC should be a list of length 3, but it has length 2"
    ):
        normalize_list_shapes(parameters, "DC", default_len=3)


def test_normalize_list_shapes_excess_length_single_wavelength(caplog):
    """Test multi-element array preserved when default_len=1 for downstream regridding."""
    parameters = {"snr": [10.0, 20.0, 30.0]}

    with pytest.raises(
        ValueError, match="snr has length 3 but the expected input size is 1"
    ):
        normalize_list_shapes(parameters, "snr", default_len=1)


def test_normalize_list_shapes_excess_length_quantity_single_wavelength(caplog):
    """Test multi-element Quantity preserved when default_len=1 for downstream regridding."""
    parameters = {"DC": u.Quantity([10.0, 20.0, 30.0], DARK_CURRENT)}

    with pytest.raises(
        ValueError, match="DC has length 3 but the expected input size is 1"
    ):
        normalize_list_shapes(parameters, "DC", default_len=1)


def test_normalize_list_shapes_numpy_array():
    """Test that numpy arrays are properly converted."""
    parameters = {"snr": np.array([10.0, 20.0, 30.0])}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_tuple_input():
    """Test that tuples are properly converted to arrays."""
    parameters = {"snr": (10.0, 20.0, 30.0)}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_integer_values():
    """Test that integer values are converted to float64."""
    parameters = {"snr": [10, 20, 30]}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_mixed_numeric_types():
    """Test that mixed int/float values are converted to float64."""
    parameters = {"snr": [10, 20.5, 30]}
    result = normalize_list_shapes(parameters, "snr", default_len=3)

    np.testing.assert_array_equal(result, np.array([10.0, 20.5, 30.0]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_normalize_list_shapes_preserves_quantity_unit():
    """Test that Quantity units are preserved through conversion."""
    parameters = {"wavelength": u.Quantity([0.5, 0.6, 0.7], u.um)}
    result = normalize_list_shapes(parameters, "wavelength", default_len=3)

    assert isinstance(result, u.Quantity)
    np.testing.assert_array_equal(result.value, np.array([0.5, 0.6, 0.7]))
    assert result.unit == u.um


# ============================================================================
# Tests for parse_parameters - Wavelength handling
# ============================================================================


def test_parse_parameters_wavelength_scalar():
    """Test parsing wavelength as a scalar."""
    parsed = parse_parameters({"wavelength": 0.5})

    assert parsed["wavelength"] == np.array([0.5])
    assert isinstance(parsed["wavelength"], np.ndarray)


def test_parse_parameters_wavelength_list():
    """Test parsing wavelength as a list."""
    parsed = parse_parameters({"wavelength": [0.5, 0.6, 0.7]})

    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert isinstance(parsed["wavelength"], np.ndarray)


def test_parse_parameters_wavelength_scalar_quantity():
    """Test parsing wavelength as a scalar Quantity."""
    parsed = parse_parameters({"wavelength": 0.5 * u.um})

    assert parsed["wavelength"] == [0.5] * u.um
    assert isinstance(parsed["wavelength"], u.Quantity)


def test_parse_parameters_wavelength_list_quantity():
    """Test parsing wavelength as a list Quantity."""
    parsed = parse_parameters({"wavelength": [0.5, 0.6, 0.7] * u.um})

    assert np.all(parsed["wavelength"] == [0.5, 0.6, 0.7] * u.um)
    assert isinstance(parsed["wavelength"], u.Quantity)


# ============================================================================
# Tests for parse_parameters - Wavelength-dependent parameters
# ============================================================================


def test_parse_parameters_wavelength_dependent_single():
    """Test wavelength-dependent parameters with single wavelength."""
    wavelength_params = [
        "snr",
        "T_optical",
        "epswarmTrcold",
        "DC",
        "RN",
        "tread",
        "CIC",
        "QE",
        "dQE",
        "mag",
        "Fstar_10pc",
        "Fp/Fs",
        "delta_mag",
        "F0",
        "det_npix_input",
    ]

    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})

        assert np.all(parsed[param] == np.array([1.5]))
        assert isinstance(parsed[param], np.ndarray)


def test_parse_parameters_wavelength_dependent_multiple():
    """Test wavelength-dependent parameters with multiple wavelengths."""
    wavelengths = [0.5, 0.6, 0.7]
    wavelength_params = ["snr", "T_optical", "QE"]

    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": wavelengths, param: [1.5, 2.5, 3.5]})

        assert np.all(parsed[param] == np.array([1.5, 2.5, 3.5]))
        assert isinstance(parsed[param], np.ndarray)


def test_parse_parameters_wavelength_dependent_scalar_broadcast(caplog):
    """Test wavelength-dependent parameters broadcast from scalar."""
    wavelengths = [0.5, 0.6, 0.7]

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        parsed = parse_parameters({"wavelength": wavelengths, "snr": 10})

    assert np.all(parsed["snr"] == np.array([10, 10, 10]))
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_parse_parameters_wavelength_dependent_quantity_broadcast(caplog):
    """Test wavelength-dependent parameters broadcast from Quantity scalar."""
    wavelengths = [0.5, 0.6, 0.7]

    with caplog.at_level(logging.DEBUG, logger="pyEDITH"):
        parsed = parse_parameters(
            {"wavelength": wavelengths, "snr": 10 * u.dimensionless_unscaled}
        )

    assert np.all(parsed["snr"] == np.array([10, 10, 10]) * u.dimensionless_unscaled)
    assert any(
        "snr should be a list of length 3" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_parse_parameters_wavelength_dependent_mismatched_length():
    """Test that mismatched lengths raise ValueError."""
    wavelengths = [0.5, 0.6, 0.7]

    with pytest.raises(
        ValueError, match="snr should be a list of length 3, but it has length 2"
    ):
        parse_parameters({"wavelength": wavelengths, "snr": [10, 20]})


def test_parse_parameters_wavelength_dependent_excess_length_scalar(caplog):
    """Test that excess length triggers error for single wavelength."""
    with pytest.raises(
        ValueError, match="snr has length 3 but the expected input size is 1"
    ):
        parse_parameters({"wavelength": 0.5, "snr": [10, 20, 30]})


def test_parse_parameters_wavelength_dependent_excess_length_quantity(caplog):
    """Test that excess length triggers warning and uses first value."""

    with pytest.raises(
        ValueError, match="DC has length 3 but the expected input size is 1."
    ):
        parse_parameters({"wavelength": 0.5, "DC": [10, 20, 30] * DARK_CURRENT})


def test_parse_parameters_wavelength_dependent_with_quantity():
    """Test wavelength-dependent parameters with Quantity input."""
    wavelengths = [0.5, 0.6, 0.7]

    parsed = parse_parameters(
        {"wavelength": wavelengths, "snr": 1.5 * u.dimensionless_unscaled}
    )

    assert np.all(parsed["snr"] == np.array([1.5, 1.5, 1.5]) * u.dimensionless_unscaled)
    assert isinstance(parsed["snr"], u.Quantity)


# ============================================================================
# Tests for parse_parameters - nlambda handling
# ============================================================================


def test_parse_parameters_wavelength_not_provided():
    """Test that missing both wavelength and nlambda raises ValueError."""
    with pytest.raises(
        ValueError, match="pyEDITH does not have access to wavelength here"
    ):
        parse_parameters({"snr": 10})


# ============================================================================
# Tests for parse_parameters - Target parameters
# ============================================================================


def test_parse_parameters_target_params():
    """Test parsing target-specific parameters (scalars)."""
    target_params = [
        "distance",
        "magV",
        "FstarV_10pc",
        "stellar_radius",
        "nzodis",
        "ra",
        "dec",
        "delta_mag_min",
        "Fp_min/Fs",
        "separation",
        "semimajor_axis",
        "npix_multiplier",
    ]

    for param in target_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})

        assert parsed[param] == 1.5
        assert isinstance(parsed[param], float)


# ============================================================================
# Tests for parse_parameters - Scalar parameters
# ============================================================================


def test_parse_parameters_scalar_params():
    """Test parsing scalar observatory parameters."""
    scalar_params = [
        "photometric_aperture_radius",
        "psf_trunc_ratio",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "contrast",
        "noisefloor_factor",
        "noisefloor_PPF",
        "bandwidth",
        "Tcore",
        "TLyot",
        "temperature",
        "T_contamination",
        "CRb_multiplier",
        "t_photon_count_input",
    ]

    for param in scalar_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})

        assert parsed[param] == 1.5
        assert isinstance(parsed[param], float)


# ============================================================================
# Tests for parse_parameters - Integer parameters
# ============================================================================


def test_parse_parameters_integer_param():
    """Test parsing integer parameter (nrolls)."""
    parsed = parse_parameters({"wavelength": 0.5, "nrolls": 3})

    assert parsed["nrolls"] == 3
    assert isinstance(parsed["nrolls"], int)


# ============================================================================
# Tests for parse_parameters - Observatory specifications
# ============================================================================


def test_parse_parameters_observatory_specs():
    """Test parsing observatory specification strings."""
    observatory_specs = [
        "observatory_preset",
        "telescope_type",
        "coronagraph_type",
        "detector_type",
    ]

    for spec in observatory_specs:
        parsed = parse_parameters({"wavelength": 0.5, spec: "TestSpec"})

        assert parsed[spec] == "TestSpec"
        assert isinstance(parsed[spec], str)
    parsed = parse_parameters({"wavelength": 0.5, "observing_mode": "IMAGER"})

    assert parsed["observing_mode"] == "IMAGER"
    assert isinstance(parsed["observing_mode"], str)


# ============================================================================
# Tests for parse_parameters - Boolean parameters
# ============================================================================


def test_parse_parameters_boolean_true():
    """Test parsing boolean parameter as True."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": True,
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_false():
    """Test parsing boolean parameter as False."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": False,
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_true():
    """Test parsing boolean from 'true' string (case-insensitive)."""
    for true_string in ["true", "True", "TRUE"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": true_string,
            }
        )
        assert parsed["az_avg"] is True
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_false():
    """Test parsing boolean from 'false' string (case-insensitive)."""
    for false_string in ["false", "False", "FALSE"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": false_string,
            }
        )
        assert parsed["az_avg"] is False
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_one():
    """Test parsing boolean from '1' string."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": "1",
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_zero():
    """Test parsing boolean from '0' string."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": "0",
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_yes():
    """Test parsing boolean from 'yes' string (case-insensitive)."""
    for yes_string in ["yes", "Yes", "YES"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": yes_string,
            }
        )
        assert parsed["az_avg"] is True
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_string_no():
    """Test parsing boolean from 'no' string (case-insensitive)."""
    for no_string in ["no", "No", "NO"]:
        parsed = parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": no_string,
            }
        )
        assert parsed["az_avg"] is False
        assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_int_one():
    """Test parsing boolean from integer 1."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 1,
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_int_zero():
    """Test parsing boolean from integer 0."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 0,
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_float_one():
    """Test parsing boolean from float 1.0."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 1.0,
        }
    )
    assert parsed["az_avg"] is True
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_float_zero():
    """Test parsing boolean from float 0.0."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": 0.0,
        }
    )
    assert parsed["az_avg"] is False
    assert isinstance(parsed["az_avg"], bool)


def test_parse_parameters_boolean_invalid_string():
    """Test that invalid boolean string raises ValueError with helpful message."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value 'maybe' for parameter 'az_avg'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": "maybe",
            }
        )


def test_parse_parameters_boolean_invalid_numeric():
    """Test that invalid numeric boolean value raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value '2' for parameter 'az_avg'\. "
        r"Expected 0 or 1 for boolean parameters\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": 2,
            }
        )


def test_parse_parameters_boolean_invalid_numeric_float():
    """Test that invalid float boolean value raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value '0\.5' for parameter 'az_avg'\. "
        r"Expected 0 or 1 for boolean parameters\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": 0.5,
            }
        )


def test_parse_parameters_boolean_invalid_type():
    """Test that invalid type for boolean raises TypeError."""
    with pytest.raises(
        TypeError,
        match=r"Invalid type list for parameter 'az_avg'\. "
        r"Expected boolean, string, or numeric \(0/1\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": [True],
            }
        )


def test_parse_parameters_boolean_invalid_type_dict():
    """Test that dict type for boolean raises TypeError."""
    with pytest.raises(
        TypeError,
        match=r"Invalid type dict for parameter 'az_avg'\. "
        r"Expected boolean, string, or numeric \(0/1\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": {"value": True},
            }
        )


def test_parse_parameters_boolean_both_params():
    """Test parsing both boolean parameters (az_avg and regrid_wavelength)."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": True,
            "regrid_wavelength": "yes",
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["az_avg"] is True
    assert parsed["regrid_wavelength"] is True
    assert isinstance(parsed["az_avg"], bool)
    assert isinstance(parsed["regrid_wavelength"], bool)


def test_parse_parameters_boolean_mixed_formats():
    """Test parsing booleans with different format inputs."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "az_avg": "TRUE",
            "regrid_wavelength": 0,
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["az_avg"] is True
    assert parsed["regrid_wavelength"] is False


def test_parse_parameters_regrid_wavelength_invalid_string_value():
    """Test that invalid string for regrid_wavelength raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value 'invalid' for parameter 'regrid_wavelength'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": "invalid",
                "spectral_resolution": [100],
                "lam_low": [0.4],
                "lam_high": [1.0],
            }
        )


def test_parse_parameters_regrid_wavelength_invalid_numeric():
    """Test that invalid numeric for regrid_wavelength raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value '-1' for parameter 'regrid_wavelength'\. "
        r"Expected 0 or 1 for boolean parameters\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": -1,
                "spectral_resolution": [100],
                "lam_low": [0.4],
                "lam_high": [1.0],
            }
        )


def test_parse_parameters_boolean_empty_string():
    """Test that empty string for boolean raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value '' for parameter 'az_avg'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": "",
            }
        )


def test_parse_parameters_boolean_whitespace_string():
    """Test that whitespace-only string for boolean raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Invalid value '   ' for parameter 'az_avg'\. "
        r"Expected boolean or one of: 'true', 'false', '1', '0', 'yes', 'no' "
        r"\(case-insensitive\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": "   ",
            }
        )


def test_parse_parameters_boolean_none_type():
    """Test that None type for boolean raises TypeError."""
    with pytest.raises(
        TypeError,
        match=r"Invalid type NoneType for parameter 'az_avg'\. "
        r"Expected boolean, string, or numeric \(0/1\)\.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "az_avg": None,
            }
        )


def test_parse_parameters_regrid_wavelength_bool():
    """Test parsing regrid_wavelength as boolean."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": True,
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["regrid_wavelength"] is True
    assert isinstance(parsed["regrid_wavelength"], bool)

    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": False,
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["regrid_wavelength"] is False
    assert isinstance(parsed["regrid_wavelength"], bool)


def test_parse_parameters_regrid_wavelength_int():
    """Test parsing regrid_wavelength from integer."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": 1,
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["regrid_wavelength"] is True
    assert isinstance(parsed["regrid_wavelength"], bool)

    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": 0,
            "spectral_resolution": [100],
            "lam_low": [0.4],
            "lam_high": [1.0],
        }
    )
    assert parsed["regrid_wavelength"] is False
    assert isinstance(parsed["regrid_wavelength"], bool)


def test_parse_parameters_regrid_wavelength_missing_lam_low():
    """Test that regrid_wavelength=True with missing lam_low raises ValueError."""
    with pytest.raises(
        KeyError,
        match="regrid_wavelength is True, but 'lam_low' is missing. "
        "Required parameters: spectral_resolution, lam_low, lam_high",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": [100],
                "lam_high": [1.0],
            }
        )


def test_parse_parameters_regrid_wavelength_missing_lam_high():
    """Test that regrid_wavelength=True with missing lam_high raises ValueError."""
    with pytest.raises(
        KeyError,
        match="regrid_wavelength is True, but 'lam_high' is missing. "
        "Required parameters: spectral_resolution, lam_low, lam_high",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": [100],
                "lam_low": [0.4],
            }
        )


def test_parse_parameters_regrid_wavelength_missing_all_required():
    """Test that regrid_wavelength=True with all required parameters missing raises ValueError."""
    with pytest.raises(
        KeyError,
        match="regrid_wavelength is True, but 'spectral_resolution' is missing. "
        "Required parameters: spectral_resolution, lam_low, lam_high",
    ):
        parse_parameters({"wavelength": 0.5, "regrid_wavelength": True})


def test_parse_parameters_regrid_wavelength_spectral_resolution_not_array():
    """Test that regrid_wavelength=True with scalar spectral_resolution raises ValueError."""
    with pytest.raises(
        ValueError,
        match="regrid_wavelength is True, but 'spectral_resolution' is not an array. "
        "All of spectral_resolution, lam_low, lam_high must be arrays of the same length.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": 100,  # scalar instead of array
                "lam_low": [0.4],
                "lam_high": [1.0],
            }
        )


def test_parse_parameters_regrid_wavelength_lam_low_not_array():
    """Test that regrid_wavelength=True with scalar lam_low raises ValueError."""
    with pytest.raises(
        ValueError,
        match="regrid_wavelength is True, but 'lam_low' is not an array. "
        "All of spectral_resolution, lam_low, lam_high must be arrays of the same length.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": [100],
                "lam_low": 0.4,  # scalar instead of array
                "lam_high": [1.0],
            }
        )


def test_parse_parameters_regrid_wavelength_lam_high_not_array():
    """Test that regrid_wavelength=True with scalar lam_high raises ValueError."""
    with pytest.raises(
        ValueError,
        match="regrid_wavelength is True, but 'lam_high' is not an array. "
        "All of spectral_resolution, lam_low, lam_high must be arrays of the same length.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": [100],
                "lam_low": [0.4],
                "lam_high": 1.0,  # scalar instead of array
            }
        )


def test_parse_parameters_regrid_wavelength_mismatched_lengths_two_params():
    """Test that regrid_wavelength=True with two parameters of different lengths raises ValueError."""
    with pytest.raises(
        ValueError,
        match="regrid_wavelength is True, but spectral_resolution, lam_low, lam_high have different lengths: "
        ".*spectral_resolution.*2.*lam_low.*2.*lam_high.*3.* All must have the same length.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": [100, 200],
                "lam_low": [0.4, 0.5],
                "lam_high": [1.0, 1.5, 2.0],  # different length
            }
        )


def test_parse_parameters_regrid_wavelength_mismatched_lengths_all_different():
    """Test that regrid_wavelength=True with all parameters of different lengths raises ValueError."""
    with pytest.raises(
        ValueError,
        match="regrid_wavelength is True, but spectral_resolution, lam_low, lam_high have different lengths: "
        ".*spectral_resolution.*1.*lam_low.*2.*lam_high.*3.* All must have the same length.",
    ):
        parse_parameters(
            {
                "wavelength": 0.5,
                "regrid_wavelength": True,
                "spectral_resolution": [100],
                "lam_low": [0.4, 0.5],
                "lam_high": [1.0, 1.5, 2.0],
            }
        )


def test_parse_parameters_regrid_wavelength_valid_lists():
    """Test that regrid_wavelength=True with valid lists succeeds."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": True,
            "spectral_resolution": [100, 200],
            "lam_low": [0.4, 0.5],
            "lam_high": [1.0, 1.5],
        }
    )

    assert parsed["regrid_wavelength"] is True
    assert np.all(parsed["spectral_resolution"] == np.array([100, 200]))
    assert np.all(parsed["lam_low"] == np.array([0.4, 0.5]))
    assert np.all(parsed["lam_high"] == np.array([1.0, 1.5]))


def test_parse_parameters_regrid_wavelength_valid_numpy_arrays():
    """Test that regrid_wavelength=True with numpy arrays succeeds."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": True,
            "spectral_resolution": np.array([100, 200]),
            "lam_low": np.array([0.4, 0.5]),
            "lam_high": np.array([1.0, 1.5]),
        }
    )

    assert parsed["regrid_wavelength"] is True
    assert np.all(parsed["spectral_resolution"] == np.array([100, 200]))
    assert np.all(parsed["lam_low"] == np.array([0.4, 0.5]))
    assert np.all(parsed["lam_high"] == np.array([1.0, 1.5]))


def test_parse_parameters_regrid_wavelength_valid_quantities():
    """Test that regrid_wavelength=True with Quantity arrays succeeds."""
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": True,
            "spectral_resolution": [100, 200] * u.dimensionless_unscaled,
            "lam_low": [0.4, 0.5] * u.um,
            "lam_high": [1.0, 1.5] * u.um,
        }
    )

    assert parsed["regrid_wavelength"] is True
    assert isinstance(parsed["spectral_resolution"], u.Quantity)
    assert isinstance(parsed["lam_low"], u.Quantity)
    assert isinstance(parsed["lam_high"], u.Quantity)


def test_parse_parameters_regrid_wavelength_false_no_validation():
    """Test that regrid_wavelength=False skips validation of required parameters."""
    # Should not raise even though required parameters are missing
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
            "regrid_wavelength": False,
        }
    )

    assert parsed["regrid_wavelength"] is False


def test_parse_parameters_regrid_wavelength_absent_no_validation():
    """Test that absence of regrid_wavelength skips validation of required parameters."""
    # Should not raise even though required parameters are missing
    parsed = parse_parameters(
        {
            "wavelength": 0.5,
        }
    )

    assert "regrid_wavelength" not in parsed


# ============================================================================
# Tests for parse_parameters - Complete parameter set
# ============================================================================


def test_parse_parameters_complete():
    """Test parsing complete parameter set."""
    parameters = {
        "wavelength": [0.5, 0.6, 0.7],
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
        "snr": [10, 20, 30],
        "T_optical": 0.8,
        "diameter": 2.4,
        "toverhead_fixed": 300,
        "contrast": 1e-10,
        "nrolls": 3,
        "observatory_preset": "EAC1",
    }

    parsed = parse_parameters(parameters)

    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert parsed["distance"] == 10
    assert parsed["magV"] == 5.0
    assert parsed["nzodis"] == 3.0
    assert parsed["observing_mode"] == "IFS"
    assert parsed["nlambda"] == 3
    assert np.all(parsed["snr"] == np.array([10, 20, 30]))
    assert np.all(parsed["T_optical"] == np.array([0.8, 0.8, 0.8]))
    assert parsed["diameter"] == 2.4
    assert parsed["toverhead_fixed"] == 300
    assert parsed["contrast"] == 1e-10
    assert parsed["nrolls"] == 3
    assert parsed["observatory_preset"] == "EAC1"


# ============================================================================
# Tests for parse_parameters - IFS and IMAGER modes
# ============================================================================


def test_parse_parameters_ifs_mode():
    """Test parsing IFS mode parameters."""
    parameters = {
        "observing_mode": "IFS",
        "wavelength": [0.5, 0.6, 0.7],
        "Fstar_10pc": [1e-8, 1e-8, 1e-8],
        "Fp/Fs": [1e-10, 1e-10, 1e-10],
    }

    parsed = parse_parameters(parameters)

    assert parsed["observing_mode"] == "IFS"
    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert np.all(parsed["Fstar_10pc"] == np.array([1e-8, 1e-8, 1e-8]))
    assert np.all(parsed["Fp/Fs"] == np.array([1e-10, 1e-10, 1e-10]))


def test_parse_parameters_imager_mode():
    """Test parsing IMAGER mode parameters."""
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": [0.5],
    }

    parsed = parse_parameters(parameters)

    assert parsed["observing_mode"] == "IMAGER"
    assert np.all(parsed["wavelength"] == np.array([0.5]))


# ============================================================================
# Tests for read_configuration
# ============================================================================


def test_read_configuration_with_secondary(sample_input_file):
    """Test reading configuration with secondary parameters."""
    parsed_parameters, parsed_secondary_parameters = read_configuration(
        sample_input_file, secondary_flag=True
    )

    assert np.all(parsed_parameters["wavelength"] == np.array([0.5]))
    assert parsed_parameters["distance"] == 10
    assert parsed_parameters["magV"] == 5.0
    assert parsed_parameters["nzodis"] == 3.0
    assert parsed_parameters["observing_mode"] == "IMAGER"
    assert parsed_secondary_parameters["wavelength"] == np.array([1.0])


def test_read_configuration_without_secondary(sample_input_file):
    """Test reading configuration without secondary parameters."""
    parsed_parameters, parsed_secondary_parameters = read_configuration(
        sample_input_file, secondary_flag=False
    )

    assert parsed_secondary_parameters == {}


# ============================================================================
# Tests for get_observatory_config
# ============================================================================


def test_get_observatory_config_with_preset():
    """Test getting observatory config from preset."""
    parameters = {"observatory_preset": "EAC1"}

    config = get_observatory_config(parameters)

    assert config == "EAC1"


def test_get_observatory_config_with_custom_components():
    """Test getting observatory config from custom component specifications."""
    parameters = {
        "telescope_type": "EAC1",
        "coronagraph_type": "AAVC",
        "detector_type": "EAC1",
    }

    config = get_observatory_config(parameters)

    assert config == {"telescope": "EAC1", "coronagraph": "AAVC", "detector": "EAC1"}


def test_get_observatory_config_missing():
    """Test that missing observatory configuration raises ValueError."""
    with pytest.raises(ValueError):
        get_observatory_config({})


def test_get_observatory_config_missing_component():
    """Test that missing individual component raises ValueError with specific message."""
    parameters = {
        "telescope_type": "EAC1",
        "coronagraph_type": "AAVC",
        # detector_type missing
    }

    with pytest.raises(ValueError, match="Detector type not specified"):
        get_observatory_config(parameters)


def test_parsed_flag_prevents_reprocessing():
    """Test that _parsed flag prevents reprocessing of already-parsed parameters."""

    # Create a minimal set of parameters
    raw_params = {
        "wavelength": [1.0, 2.0, 3.0],
        "snr": [10.0, 15.0, 20.0],
        "distance": 10.0,
    }

    # Parse once
    parsed_once = parse_parameters(raw_params)

    # Verify _parsed flag is set
    assert "_parsed" in parsed_once
    assert parsed_once["_parsed"] is True

    # Store a reference to verify identity
    parsed_once_id = id(parsed_once)

    # Parse again - should return immediately without reprocessing
    parsed_twice = parse_parameters(parsed_once)

    # Should return the same object (not a copy)
    assert id(parsed_twice) == parsed_once_id
    assert parsed_twice is parsed_once

    # Verify all original parsed values are unchanged
    assert parsed_twice["nlambda"] == 3
    assert "_parsed" in parsed_twice


def test_parsed_flag_not_present_initially():
    """Test that _parsed flag is not present in raw parameters."""

    raw_params = {"wavelength": [1.0], "distance": 10.0}

    # Verify _parsed is not in raw params
    assert "_parsed" not in raw_params

    # Parse
    parsed = parse_parameters(raw_params)

    # Now it should be present
    assert "_parsed" in parsed
    assert parsed["_parsed"] is True


def test_parsed_flag_set_after_single_parse():
    """Test that _parsed flag is set after a single parsing operation."""

    raw_params = {"wavelength": 1.5, "snr": 25.0, "distance": 5.0}

    parsed = parse_parameters(raw_params)

    # Check the flag exists and is True
    assert parsed.get("_parsed") is True


# ============================================================================
# Tests for parse_parameters - overrides parameter
# ============================================================================


def test_parse_parameters_overrides_string():
    """Test parsing overrides parameter as a comma-separated string."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "DC,RN,QE"})

    assert "overrides" in parsed
    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_string_with_spaces():
    """Test parsing overrides parameter with spaces around commas."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "DC, RN, QE"})

    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_string_single_value():
    """Test parsing overrides parameter with single value string."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "DC"})

    assert parsed["overrides"] == ["DC"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_list():
    """Test parsing overrides parameter as a list."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": ["DC", "RN", "QE"]})

    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_tuple():
    """Test parsing overrides parameter as a tuple."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": ("DC", "RN")})

    assert parsed["overrides"] == ["DC", "RN"]
    assert isinstance(parsed["overrides"], list)


def test_parse_parameters_overrides_empty_string():
    """Test parsing overrides parameter as an empty string."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": ""})

    assert "overrides" not in parsed


def test_parse_parameters_overrides_empty_list():
    """Test parsing overrides parameter as an empty list."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": []})

    assert "overrides" not in parsed


def test_parse_parameters_overrides_not_present():
    """Test that absence of overrides parameter doesn't create key."""
    parsed = parse_parameters({"wavelength": 0.5})

    assert "overrides" not in parsed


def test_parse_parameters_overrides_string_extra_whitespace():
    """Test parsing overrides with extra whitespace."""
    parsed = parse_parameters({"wavelength": 0.5, "overrides": "  DC  ,  RN  ,  QE  "})

    assert parsed["overrides"] == ["DC", "RN", "QE"]
    assert isinstance(parsed["overrides"], list)
