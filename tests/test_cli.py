import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pyEDITH.cli import main, calculate_texp, calculate_snr
from pyEDITH import AstrophysicalScene, Observation, ObservatoryBuilder
from io import StringIO
import tempfile
import os


@pytest.fixture
def mock_args():
    """
    Fixture to create a mock arguments object.
    This simulates the parsed command-line arguments.
    """

    class Args:
        edith = "test.edith"
        verbose = False
        time = 60.0

    return Args()


@pytest.fixture
def mock_parameters():
    """
    Fixture to create a mock parameters dictionary.
    This simulates the parsed configuration from an .edith file.
    """
    return {
        "wavelength": [0.5],
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IMAGER",
        "observatory_preset": "ToyModel",
    }


def test_main_no_args():
    """Test main function with no arguments."""
    with patch("sys.argv", ["edith"]):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main()
            assert "usage: edith" in fake_out.getvalue()


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
def test_main_missing_edith_arg(subcommand, error_message):
    """Test main function with missing --edith argument for each subcommand."""
    with patch("sys.argv", ["edith", subcommand]):
        with pytest.raises(SyntaxError, match=error_message):
            main()


def test_main_snr_missing_time_arg():
    """Test main function with missing --time argument for snr subcommand."""
    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith"]):
        with pytest.raises(
            SyntaxError,
            match="Both --edith and --time arguments are required for snr subfunction.",
        ):
            main()


@patch("pyEDITH.cli.parse_input.read_configuration")
def test_main_etc2snr_no_secondary_params(mock_read_configuration):
    """Test main function when etc2snr has no secondary parameters."""
    mock_read_configuration.return_value = ({}, {})

    with patch("sys.argv", ["edith", "etc2snr", "--edith", "test.edith"]):
        with pytest.raises(
            ValueError, match="The secondary parameters are not specified."
        ):
            main()


@patch("pyEDITH.cli.parse_input.read_configuration")
def test_main_etc2snr_multiple_primary_lambdas(mock_read_configuration):
    """Test main function when etc2snr has multiple primary lambdas."""
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
    """Test main function when etc2snr calculation returns infinite exposure time."""
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
def test_main_etc(mock_calculate_texp, mock_read_configuration, mock_args):
    """
    Test the 'etc' subcommand of the main CLI function.

    This test mocks the configuration reading and exposure time calculation,
    then checks if the main function correctly calls these and prints the result.
    """
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_texp.return_value = (np.array([1.0]), {})

    with patch("sys.argv", ["edith", "etc", "--edith", "test.edith"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([1.0]))


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_snr")
def test_main_snr(mock_calculate_snr, mock_read_configuration, mock_args):
    """
    Test the 'snr' subcommand of the main CLI function.

    This test mocks the configuration reading and SNR calculation,
    then checks if the main function correctly calls these and prints the result.
    """
    mock_read_configuration.return_value = ({}, {})
    mock_calculate_snr.return_value = (np.array([10.0]), {})

    with patch("sys.argv", ["edith", "snr", "--edith", "test.edith", "--time", "60"]):
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_with(np.array([10.0]))


@patch("pyEDITH.cli.parse_input.read_configuration")
@patch("pyEDITH.cli.calculate_texp")
@patch("pyEDITH.cli.calculate_snr")
def test_main_etc2snr(
    mock_calculate_snr, mock_calculate_texp, mock_read_configuration, mock_args
):
    """
    Test the 'etc2snr' subcommand of the main CLI function.

    This test mocks the configuration reading, exposure time calculation, and SNR calculation.
    It then checks if the main function correctly calls these and prints the results.
    """
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


def test_calculate_texp(mock_parameters):
    """
    Test the calculate_texp function.

    This test mocks the Observation, AstrophysicalScene, ObservatoryBuilder, and
    calculate_exposure_time_or_snr functions. It then checks if calculate_texp
    correctly sets up these objects and calls the calculation function.
    """
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.ObservatoryBuilder") as mock_builder,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):

        # Set up mock objects and their return values
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory = MagicMock()
        mock_builder.create_observatory.return_value = mock_observatory

        # Call the function under test
        texp, validation_variables = calculate_texp(mock_parameters, False)

        # Assert the results
        assert np.array_equal(texp, np.array([1.0]))
        assert validation_variables == {}

        # Check that all expected method calls occurred
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_builder.create_observatory.assert_called_once()
        mock_calculate.assert_called_once()

    # modify the mock parameters to test IFS mode regridding in scene
    mock_parameters["spectral_resolution"] = [140, 40]
    mock_parameters["channel_bounds"] = 1.0
    mock_parameters["regrid_wavelength"] = True
    mock_parameters["observing_mode"] = "IFS"

    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.ObservatoryBuilder") as mock_builder,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):

        # Set up mock objects and their return values
        mock_observation_instance = MagicMock()
        mock_observation_instance.exptime = np.array([1.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory = MagicMock()
        mock_builder.create_observatory.return_value = mock_observatory

        # Call the function under test
        texp, validation_variables = calculate_texp(mock_parameters, False)

        # Assert the results
        assert np.array_equal(texp, np.array([1.0]))
        assert validation_variables == {}

        # Check that all expected method calls occurred
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_builder.create_observatory.assert_called_once()
        mock_calculate.assert_called_once()


def test_calculate_snr(mock_parameters):
    """
    Test the calculate_snr function.

    This test mocks the Observation, AstrophysicalScene, ObservatoryBuilder, and
    calculate_exposure_time_or_snr functions. It then checks if calculate_snr
    correctly sets up these objects and calls the calculation function.
    """
    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.ObservatoryBuilder") as mock_builder,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):

        # Set up mock objects and their return values
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory = MagicMock()
        mock_builder.create_observatory.return_value = mock_observatory

        # Call the function under test
        snr, validation_variables = calculate_snr(mock_parameters, 1.0, False)

        # Assert the results
        assert np.array_equal(snr, np.array([10.0]))
        assert validation_variables == {}

        # Check that all expected method calls occurred
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_builder.create_observatory.assert_called_once()
        mock_calculate.assert_called_once()

    # modify the mock parameters to test IFS mode regridding in scene
    mock_parameters["spectral_resolution"] = [140, 40]
    mock_parameters["channel_bounds"] = 1.0
    mock_parameters["regrid_wavelength"] = True
    mock_parameters["observing_mode"] = "IFS"

    with (
        patch("pyEDITH.cli.Observation") as mock_observation,
        patch("pyEDITH.cli.AstrophysicalScene") as mock_scene,
        patch("pyEDITH.cli.ObservatoryBuilder") as mock_builder,
        patch("pyEDITH.cli.calculate_exposure_time_or_snr") as mock_calculate,
    ):

        # Set up mock objects and their return values
        mock_observation_instance = MagicMock()
        mock_observation_instance.fullsnr = np.array([10.0])
        mock_observation_instance.validation_variables = {}
        mock_observation.return_value = mock_observation_instance

        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance

        mock_observatory = MagicMock()
        mock_builder.create_observatory.return_value = mock_observatory

        # Call the function under test
        snr, validation_variables = calculate_snr(mock_parameters, 1.0, False)

        # Assert the results
        assert np.array_equal(snr, np.array([10.0]))
        assert validation_variables == {}

        # Check that all expected method calls occurred
        mock_observation.assert_called_once()
        mock_scene.assert_called_once()
        mock_builder.create_observatory.assert_called_once()
        mock_calculate.assert_called_once()
