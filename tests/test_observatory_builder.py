import pytest
import os
import json
from unittest.mock import patch, mock_open
from pyEDITH.observatory_builder import ObservatoryBuilder
from pyEDITH.observatory import Observatory
from pyEDITH.components import telescopes, coronagraphs, detectors


@pytest.fixture
def mock_registry():
    return {
        "telescopes": {
            "EAC1": {"class": "EACTelescope", "path": ""},
            "ToyModel": {"class": "ToyModelTelescope", "path": None},
        },
        "coronagraphs": {
            "AAVC": {"class": "CoronagraphYIP", "path": "AAVC_coronagraph"},
            "LUVOIR": {"class": "CoronagraphYIP", "path": "usort_offaxis_ovc"},
            "ToyModel": {"class": "ToyModelCoronagraph", "path": None},
        },
        "detectors": {
            "EAC1": {"class": "EACDetector", "path": ""},
            "ToyModel": {"class": "ToyModelDetector", "path": None},
        },
    }


def test_load_registry(mock_registry):
    with patch("json.load", return_value=mock_registry):
        registry = ObservatoryBuilder.load_registry()
    assert "ToyModel" in registry["telescopes"]
    assert "ToyModel" in registry["coronagraphs"]
    assert "ToyModel" in registry["detectors"]


def test_build_component_path():

    # Set environment variables
    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        # Telescopes and empty path
        assert ObservatoryBuilder.build_component_path("telescopes", "") == "/sci_eng/"
        # Coronagraph and full path
        assert (
            ObservatoryBuilder.build_component_path("coronagraphs", "AAVC_coronagraph")
            == "/yip_coro/AAVC_coronagraph"
        )

    # Environment variables not set
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError) as excinfo:
            ObservatoryBuilder.build_component_path("telescopes", "EAC1")
        assert "SCI_ENG_DIR environment variable not set" in str(excinfo.value)

        with pytest.raises(EnvironmentError) as excinfo:
            ObservatoryBuilder.build_component_path("coronagraphs", "AAVC")
        assert "YIP_CORO_DIR environment variable not set" in str(excinfo.value)

    # Test with invalid component type
    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.build_component_path("unknown", "EAC1")
    assert "Unknown component type: unknown" in str(excinfo.value)


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.load_registry")
def test_create_observatory(mock_load_registry, mock_registry):
    mock_load_registry.return_value = mock_registry

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        # Test with preset
        observatory = ObservatoryBuilder.create_observatory("EAC1")
        assert isinstance(observatory, Observatory)

        assert observatory.telescope.path == "/sci_eng/"
        assert observatory.coronagraph.path == "/yip_coro/usort_offaxis_ovc"
        assert observatory.detector.path == "/sci_eng/"

        # Test with custom config
        custom_config = {
            "telescope": "EAC1",
            "coronagraph": "AAVC",
            "detector": "ToyModel",
        }
        observatory = ObservatoryBuilder.create_observatory(custom_config)
        assert isinstance(observatory, Observatory)
        assert observatory.telescope.path == "/sci_eng/"
        assert observatory.coronagraph.path == "/yip_coro/AAVC_coronagraph"
        assert observatory.detector.path == None

        # Test with invalid preset
        with pytest.raises(ValueError) as excinfo:
            ObservatoryBuilder.create_observatory("UNKNOWN")
        assert str(excinfo.value) == "Unknown preset observatory: UNKNOWN"

        # Test with invalid input
        with pytest.raises(ValueError):
            ObservatoryBuilder.create_observatory(123)


@patch("pyEDITH.observatory_builder.ObservatoryBuilder.build_component_path")
def test_create_component(mock_build_path, mock_registry):
    mock_build_path.side_effect = lambda component_type, path: (
        f"/mock/{component_type}/{path}"
    )

    with patch.dict(
        os.environ, {"SCI_ENG_DIR": "/sci_eng", "YIP_CORO_DIR": "/yip_coro"}
    ):
        # Test telescope creation
        telescope = ObservatoryBuilder._create_component(
            "telescopes", "EAC1", mock_registry
        )
        assert isinstance(telescope, telescopes.EACTelescope)
        assert telescope.path == "/mock/telescopes/"
        assert telescope.keyword == "EAC1"

        # Test coronagraph creation
        coronagraph = ObservatoryBuilder._create_component(
            "coronagraphs", "AAVC", mock_registry
        )
        assert isinstance(coronagraph, coronagraphs.CoronagraphYIP)
        assert coronagraph.path == "/mock/coronagraphs/AAVC_coronagraph"
        assert coronagraph.keyword == "AAVC"

        # Test detector creation
        detector = ObservatoryBuilder._create_component(
            "detectors", "EAC1", mock_registry
        )
        assert isinstance(detector, detectors.EACDetector)
        assert detector.path == "/mock/detectors/"
        assert detector.keyword == "EAC1"

        # Test ToyModel component (no path)
        toy_telescope = ObservatoryBuilder._create_component(
            "telescopes", "ToyModel", mock_registry
        )
        assert isinstance(toy_telescope, telescopes.ToyModelTelescope)
        assert toy_telescope.path is None
        assert toy_telescope.keyword == "ToyModel"

        # Test with unknown component
        with pytest.raises(ValueError) as excinfo:
            ObservatoryBuilder._create_component("telescopes", "Unknown", mock_registry)
        assert "Unknown telescopes keyword: Unknown" in str(excinfo.value)

        # Test with unknown class
        mock_registry["telescopes"]["Invalid"] = {"class": "InvalidClass", "path": ""}
        with pytest.raises(ValueError) as excinfo:
            ObservatoryBuilder._create_component("telescopes", "Invalid", mock_registry)
        assert "Unknown component class: InvalidClass" in str(excinfo.value)


def test_configure_observatory():
    observatory = Observatory()
    config = {"some_config": "value"}
    observation = "mock_observation"
    scene = "mock_scene"

    with patch.object(Observatory, "load_configuration") as mock_load_config:
        configured_observatory = ObservatoryBuilder.configure_observatory(
            observatory, config, observation, scene
        )

    mock_load_config.assert_called_once_with(config, observation, scene)
    assert configured_observatory == observatory


def test_add_preset():
    new_preset = {
        "telescope": "NewTelescope",
        "coronagraph": "NewCoronagraph",
        "detector": "NewDetector",
    }
    ObservatoryBuilder.add_preset("NewPreset", new_preset)
    assert "NewPreset" in ObservatoryBuilder.PRESETS
    assert ObservatoryBuilder.PRESETS["NewPreset"] == new_preset

    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.add_preset("EAC1", new_preset)
    assert str(excinfo.value) == "Preset 'EAC1' already exists"


def test_remove_preset():
    ObservatoryBuilder.remove_preset("EAC1")
    assert "EAC1" not in ObservatoryBuilder.PRESETS

    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.remove_preset("NonExistentPreset")
    assert str(excinfo.value) == "Preset 'NonExistentPreset' does not exist"


def test_list_presets():
    presets = ObservatoryBuilder.list_presets()
    assert isinstance(presets, list)
    assert "ToyModel" in presets


def test_get_preset_config():
    config = ObservatoryBuilder.get_preset_config("ToyModel")
    assert isinstance(config, dict)
    assert "telescope" in config
    assert config["telescope"] == "ToyModel"

    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.get_preset_config("NonExistentPreset")
    assert str(excinfo.value) == "Preset 'NonExistentPreset' does not exist"


def test_validate_config():
    valid_config = {"telescope": "EAC1", "coronagraph": "AAVC", "detector": "EAC1"}
    ObservatoryBuilder.validate_config(valid_config)

    invalid_config1 = {"telescope": "EAC1", "coronagraph": "AAVC"}
    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.validate_config(invalid_config1)
    assert str(excinfo.value) == "Missing required configuration key: detector"

    invalid_config2 = {"telescope": "EAC1", "coronagraph": "AAVC", "detector": 123}
    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.validate_config(invalid_config2)
    assert str(excinfo.value) == "Configuration value for detector must be a string"


def test_modify_preset():
    ObservatoryBuilder.modify_preset("ToyModel", coronagraph="AAVC")
    assert ObservatoryBuilder.PRESETS["ToyModel"]["coronagraph"] == "AAVC"

    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.modify_preset("NonExistentPreset", telescope="NewTelescope")
    assert str(excinfo.value) == "Preset 'NonExistentPreset' does not exist"

    with pytest.raises(ValueError) as excinfo:
        ObservatoryBuilder.modify_preset("ToyModel", invalid_key="Value")
    assert str(excinfo.value) == "Invalid configuration key: invalid_key"
