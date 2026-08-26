import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pyEDITH.cli import main, calculate_texp, calculate_snr
from pyEDITH.filters import Filter
from io import StringIO
import astropy.units as u

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_args():
    """Fixture to create a mock arguments object simulating parsed CLI arguments."""

    class Args:
        edith = "test.edith"
        verbose = False
        time = 60.0

    return Args()


@pytest.fixture
def mock_parameters():
    """Fixture to create a mock parameters dictionary."""
    return {
        "wavelength": [0.5],
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IMAGER",
        "observatory_preset": "ToyModel",
        "filter_list": [
            Filter("photometry", center=0.5 * u.um, bandwidth=0.2, type="IMAGER")
        ],
    }


@pytest.fixture
def mock_parameters_ifs():
    """Fixture to create mock parameters for IFS mode."""
    return {
        "wavelength": np.linspace(0.2, 1.8, 100),
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
        "observatory_preset": "ToyModel",
        "filter_list": [Filter("VIS", low=0.4 * u.um, high=1.0 * u.um, resolution=140)],
    }


# ============================================================================
# Tests for main() - No arguments
# ============================================================================


def test_main_no_args_shows_help():
    """Test that main with no arguments displays usage help."""
    with patch("sys.argv", ["edith"]):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main()
            assert "usage: edith" in fake_out.getvalue()


# ============================================================================
# Tests for main() - Missing required arguments
# ============================================================================


@pytest.mark.parametrize(
    "subcommand, error_message",
    [
        ("etc", "--edith argument is required for etc subfunction."),
        (
            "snr",
            "Both --edith and --time arguments are required for snr subfunction.",
        ),
        ("etc2snr", "--edith argument is required for etc2snr subfunction."),
    ],
)
def test_main_missing_edith_argument(subcommand, error_message):
    """Test that missing --edith argument raises appropriate error for each subcommand."""
    with patch("sys.argv", ["edith", subcommand]):
        with pytest.raises(SyntaxError, match=error_message):
            main()


def test_main_snr_missing_time_argument():
    """Test that snr subcommand requires both --edith and --time arguments."""
    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith"]):
        with pytest.raises(
            SyntaxError,
            match="Both --edith and --time arguments are required for snr subfunction.",
        ):
            main()


# ============================================================================
# Tests for main() - etc subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_etc_basic_execution(
    mock_calculate_texp, mock_read_configuration, mock_args
):
    """Test that etc subcommand correctly calculates and prints exposure time."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["edith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([1.0]))


# ============================================================================
# Tests for main() - snr subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_snr")
def test_main_snr_basic_execution(
    mock_calculate_snr, mock_read_configuration, mock_args
):
    """Test that snr subcommand correctly calculates and prints SNR."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_snr.return_value = (np.array([10.0]), {})

    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith", "--time", "60"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([10.0]))


# ============================================================================
# Tests for main() - etc2snr subcommand
# ============================================================================


@patch("pyEDITH.cli.parse_input.read_configuration")
def test_main_etc2snr_missing_secondary_params(mock_read_configuration):
    """Test that etc2snr raises error when secondary parameters not specified."""
    mock_read_configuration.return_value = ({}, {})

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with pytest.raises(
            ValueError, match="The secondary parameters are not specified."
        ):
            main()


@patch("pyEDITH.cli.parse_input.read_configuration")
def test_main_etc2snr_multiple_primary_wavelengths(mock_read_configuration):
    """Test that etc2snr rejects multiple primary wavelengths."""
    mock_read_configuration.return_value = (
        {"wavelength": [0.5, 0.6]},
        {"wavelength": [0.7]},
    )

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with pytest.raises(
            TypeError, match="Cannot accept multiple lambdas as primary lambda"
        ):
            main()


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_etc2snr_infinite_exposure_time(
    mock_calculate_texp, mock_read_configuration
):
    """Test that etc2snr raises error when exposure time is infinite."""
    mock_read_configuration.return_value = (
        {"wavelength": [0.5]},
        {"wavelength": [0.6]},
    )
    mock_calculate_texp.return_value = (float("inf"), {})

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with pytest.raises(ValueError, match="Returned exposure time is infinity."):
            main()


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
@patch("pyEDITH.cli.calculate_snr")
def test_main_etc2snr_successful_calculation(
    mock_calculate_snr, mock_calculate_texp, mock_read_configuration, mock_args
):
    """Test that etc2snr correctly calculates exposure time and SNR for secondary wavelength."""
    mock_read_configuration.return_value = (
        {"wavelength": [0.5], "extra_param": 0.1},
        {"wavelength": [0.6]},
    )
    mock_calculate_texp.return_value = (np.array([1.0]), {})
    mock_calculate_snr.return_value = (np.array([10.0]), {})

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_any_call("Reference exposure time: ", np.array([1.0]))
            mock_print.assert_any_call(
                "SNR at the secondary lambda: ", np.array([10.0])
            )


# ============================================================================
# Tests for main() - Verbosity levels
# ============================================================================


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_default_verbosity_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that default verbosity (no flag) sets warning level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("warning")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_verbose_flag_sets_info_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that -v flag sets info verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-v", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("info")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_verbose_long_flag_sets_info_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that --verbose flag sets info verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "--verbose", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("info")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_double_verbose_flag_sets_debug_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that -vv flag sets debug verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-vv", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("debug")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_flag_sets_error_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that -q flag sets error verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "-q", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_long_flag_sets_error_level(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that --quiet flag sets error verbosity level."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["pyedith", "--quiet", "etc", "--edith", "test.edith"]):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


@patch("pyEDITH.cli.set_verbosity")
@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
def test_main_quiet_overrides_verbose(
    mock_calculate_texp, mock_read_configuration, mock_set_verbosity
):
    """Test that --quiet flag takes precedence over -vv flag."""
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch(
        "sys.argv", ["pyedith", "--quiet", "-vv", "etc", "--edith", "test.edith"]
    ):
        with patch("builtins.print"):
            main()
            mock_set_verbosity.assert_called_with("error")


# ============================================================================
# Tests for calculate_texp()
# ============================================================================


def test_calculate_texp_basic_execution(mock_parameters):
    """Test that calculate_texp correctly initializes objects and returns exposure time."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        texp, validation_variables = calculate_texp(mock_parameters, False)

        # Verify results
        filter_name = mock_parameters["filter_list"][0].name
        assert filter_name in texp
        assert "wavelength" in texp[filter_name]
        assert "exposure_time" in texp[filter_name]
        assert np.array_equal(texp[filter_name]["exposure_time"], np.array([1.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}


def test_calculate_texp_calls_all_components(mock_parameters):
    """Test that calculate_texp calls all required component methods."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        calculate_texp(mock_parameters, False)

        # Verify all components were called
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_calculate.assert_called_once()


def test_calculate_texp_ifs_mode_with_regridding(mock_parameters_ifs):
    """Test that calculate_texp correctly handles IFS mode with wavelength regridding."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        texp, validation_variables = calculate_texp(mock_parameters_ifs, False)

        # Verify results
        filter_name = mock_parameters_ifs["filter_list"][0].name
        assert filter_name in texp
        assert "wavelength" in texp[filter_name]
        assert "exposure_time" in texp[filter_name]
        assert np.array_equal(texp[filter_name]["exposure_time"], np.array([1.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}

        # Verify scene regridding was called
        mock_scene_instance.regrid_spectra.assert_called_once()


# ============================================================================
# Tests for calculate_snr()
# ============================================================================


def test_calculate_snr_basic_execution(mock_parameters):
    """Test that calculate_snr correctly initializes objects and returns SNR."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        snr, validation_variables = calculate_snr(mock_parameters, 1.0)

        # Verify results
        filter_name = mock_parameters["filter_list"][0].name
        assert filter_name in snr
        assert "wavelength" in snr[filter_name]
        assert "snr" in snr[filter_name]
        assert np.array_equal(snr[filter_name]["snr"], np.array([10.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}


def test_calculate_snr_calls_all_components(mock_parameters):
    """Test that calculate_snr calls all required component methods."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        calculate_snr(mock_parameters, 1.0)

        # Verify all components were called
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_observatory.assert_called_once()
        mock_calculate.assert_called_once()


def test_calculate_snr_ifs_mode_with_regridding(mock_parameters_ifs):
    """Test that calculate_snr correctly handles IFS mode with wavelength regridding."""
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.Observatory") as mock_observatory,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):
        # Set up mock objects
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory_instance = MagicMock()
        mock_observatory.return_value = mock_observatory_instance

        # Execute
        snr, validation_variables = calculate_snr(mock_parameters_ifs, 1.0)

        # Verify results
        filter_name = mock_parameters_ifs["filter_list"][0].name
        assert filter_name in snr
        assert "wavelength" in snr[filter_name]
        assert "snr" in snr[filter_name]
        assert np.array_equal(snr[filter_name]["snr"], np.array([10.0]))
        assert filter_name in validation_variables
        assert validation_variables[filter_name] == {}

        # Verify scene regridding was called
        mock_scene_instance.regrid_spectra.assert_called_once()
