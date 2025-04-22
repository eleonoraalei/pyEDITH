import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock
from pyEDITH.components.detectors import ToyModelDetector, EACDetector
from pyEDITH.units import (
    MAS,
    DIMENSIONLESS,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    WAVELENGTH,
    LENGTH,
    ARCSEC,
    SECOND,
    FRAME,
)


class MockMediator:
    def __init__(self, observing_mode="IMAGER"):
        self.observing_mode = observing_mode

    def get_scene_parameter(self, param):
        if param == "angular_diameter_arcsec":
            return 0.05 * ARCSEC
        return 1.0

    def get_telescope_parameter(self, param):
        if param == "diameter":
            return 8.0 * LENGTH
        return 1.0

    def get_observation_parameter(self, param):
        if param == "wavelength":
            if self.observing_mode == "IFS":
                return np.array([0.5, 0.7, 1.2]) * WAVELENGTH
            elif self.observing_mode == "IMAGER":
                return np.array([0.5]) * WAVELENGTH
        else:
            return 1.0

    def get_coronagraph_parameter(self, param):
        if param == "bandwidth":
            return 0.2
        return 1.0


@pytest.fixture
def mock_instrument():
    mock = MagicMock()

    # Use a smaller wavelength range for manageability
    mock.lam = [0.5, 1.5] * u.um

    # Create shorter arrays for all the optical elements
    array_length = 2
    default_array = np.linspace(0.359, 0.988, array_length)

    mock.__dict__.update(
        {
            "verbose": False,
            "lam": mock.lam,
            "OP_full": [
                "PM",
                "SM",
                "TCA",
                "wave_beamsplitter",
                "pol_beamsplitter",
                "FSM",
                "OAPs_forward",
                "DM1",
                "DM2",
                "Fold",
                "OAPs_back",
                "Apodizer",
                "Focal_Plane_Mask",
                "Lyot_Stop",
                "Field_Stop",
                "filters",
                "Detector",
            ],
            "OP_tele": ["PM", "SM"],
            "OP_inst": [
                "TCA",
                "wave_beamsplitter",
                "pol_beamsplitter",
                "FSM",
                "OAPs_forward",
                "DM1",
                "DM2",
                "Fold",
                "OAPs_back",
                "Apodizer",
                "Focal_Plane_Mask",
                "Lyot_Stop",
                "Field_Stop",
                "filters",
            ],
            "OP_det": ["Detector"],
            "TCA": default_array,
            "wb_tran": np.concatenate([np.zeros(5), np.ones(5)]),
            "wb_refl": np.concatenate([np.ones(5), np.zeros(5)]),
            "wave_beamsplitter": np.ones(array_length),
            "pol_beamsplitter": np.ones(array_length),
            "FSM": default_array,
            "OAPs_forward": default_array,
            "DM1": default_array,
            "DM2": default_array,
            "Fold": default_array,
            "OAPs_back": default_array,
            "Apodizer": np.full(array_length, 0.95),
            "Focal_Plane_Mask": np.linspace(0.91, 0.89, array_length),
            "Lyot_Stop": default_array,
            "Field_Stop": np.linspace(0.91, 0.89, array_length),
            "filters": np.ones(array_length),
            "total_inst_refl": np.full(array_length, 0.7),
        }
    )

    return mock


@pytest.fixture
def mock_detector():
    def _create_mock(detector_type):
        mock = MagicMock()

        mock.lam = np.array([0.2, 0.8, 1.1, 1.6]) * u.um
        mock.verbose = False

        qe_vis = np.array([0.9, 0.9, np.nan, np.nan])
        qe_nir = np.array([np.nan, np.nan, 0.85, 0.85])

        common_dict = {
            "lam": mock.lam,
            "verbose": False,
            "qe_vis": qe_vis,
            "dc_vis": 3e-05,
            "cic_vis": None,
            "qe_nir": qe_nir,
            "dc_nir": 0.0001,
            "cic_nir": None,
        }

        if detector_type == "IMAGER":
            mock.__dict__.update(
                {
                    **common_dict,
                    "rn_vis": 0.1,
                    "rn_nir": 0.3,
                }
            )
        elif detector_type == "IFS":
            mock.__dict__.update(
                {
                    **common_dict,
                    "rn_vis": 0.0,
                    "rn_nir": 0.4,
                }
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        return mock

    return _create_mock


def test_toy_model_detector_init():
    detector = ToyModelDetector()
    assert detector.path is None
    assert detector.keyword is None


def test_toy_model_detector_load_configuration():
    detector = ToyModelDetector()
    parameters = {
        "pixscale_mas": 10,
        "npix_multiplier": [2],
        "DC": [4e-5],
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
    }
    mediator = MockMediator()

    detector.load_configuration(parameters, mediator)

    assert detector.pixscale_mas == 10 * MAS  # User-input
    assert np.all(
        detector.npix_multiplier == [2] * DIMENSIONLESS
    )  # overwriting default
    assert np.all(detector.DC == [4e-5] * DARK_CURRENT)
    assert np.all(detector.RN == [1.0] * READ_NOISE)
    assert np.all(detector.tread == [1100] * READ_TIME)
    assert np.all(detector.CIC == [1.5e-3] * CLOCK_INDUCED_CHARGE)
    assert np.all(detector.QE == [0.9] * QUANTUM_EFFICIENCY)  # Default
    assert np.all(detector.dQE == [0.75] * DIMENSIONLESS)  # Default

    # Test default values
    detector = ToyModelDetector()
    detector.load_configuration({}, mediator)

    assert np.isclose(detector.pixscale_mas, 6.4457752 * MAS)


@pytest.mark.parametrize("observing_mode", ["IMAGER", "IFS"])
@patch("eacy.load_detector")
def test_eac_detector_load_configuration(
    mock_instrument,
    mock_detector,
    observing_mode,
):
    with patch("eacy.load_instrument", return_value=mock_instrument), patch(
        "eacy.load_detector", return_value=mock_detector(observing_mode)
    ):

        detector = EACDetector()
        parameters = {"observing_mode": observing_mode}
        mediator = MockMediator(observing_mode)

        detector.load_configuration(parameters, mediator)

        assert detector.pixscale_mas is not None
        assert np.all(detector.npix_multiplier == 1 * DIMENSIONLESS)
        assert detector.DC.unit == DARK_CURRENT
        assert detector.RN.unit == READ_NOISE
        assert detector.tread.unit == READ_TIME
        assert detector.CIC.unit == CLOCK_INDUCED_CHARGE
        assert detector.QE.unit == QUANTUM_EFFICIENCY
        assert detector.dQE.unit == DIMENSIONLESS

        # Common assertions
        expected_shape = (1,) if observing_mode == "IMAGER" else (3,)
        assert detector.DC.shape == expected_shape
        assert detector.RN.shape == expected_shape
        assert detector.QE.shape == expected_shape

        assert np.allclose(detector.DC[:2].value, 3e-05)
        assert np.allclose(detector.DC[2:].value, 0.0001)

        # Mode-specific assertions
        if observing_mode == "IMAGER":
            assert np.allclose(detector.DC.value, 3e-05)
            assert np.allclose(detector.RN.value, 0.1)

        elif observing_mode == "IFS":
            assert np.allclose(detector.DC[:2].value, 3e-05)
            assert np.allclose(detector.DC[2:].value, 0.0001)
            assert np.allclose(detector.RN[:2].value, 0.0)
            assert np.allclose(detector.RN[2:].value, 0.4)

        # Additional assertions
        assert detector.CIC.shape == expected_shape
        assert np.all(
            detector.CIC.value == 0
        )  # Assuming CIC is set to 0 when None is provided

    # Unsupported observing mode
    detector = EACDetector()
    parameters = {"observing_mode": "test"}
    mediator = MockMediator("test")
    with pytest.raises(KeyError, match="Unsupported observing mode: test"):
        detector.load_configuration(parameters, mediator)


@pytest.mark.parametrize("observing_mode", ["IMAGER", "IFS"])
def test_eac_detector_etc_validation_inputs(observing_mode):
    detector = EACDetector()
    mediator = MockMediator(observing_mode)

    parameters = {
        "observing_mode": observing_mode,
        "t_photon_count_input": 0.7,
        "det_npix_input": 200,
    }
    detector.load_configuration(parameters, mediator)
    assert hasattr(detector, "t_photon_count_input")
    assert hasattr(detector, "det_npix_input")
    assert detector.t_photon_count_input == 0.7 * SECOND / FRAME
    assert detector.det_npix_input == 200 * DIMENSIONLESS


def test_eac_detector_load_configuration_invalid():

    detector = EACDetector()
    parameters = {"observing_mode": "INVALID"}
    mediator = MockMediator("IMAGER")

    with pytest.raises(KeyError, match="Unsupported observing mode: INVALID"):
        detector.load_configuration(parameters, mediator)


def test_detector_validate_configuration():
    detector = ToyModelDetector()
    parameters = {
        "pixscale_mas": 10,
        "npix_multiplier": [2],
        "DC": [4e-5],
        "RN": [1.0],
        "tread": [1100],
        "CIC": [1.5e-3],
        "QE": [0.95],
        "dQE": [0.8],
    }
    mediator = MockMediator()
    detector.load_configuration(parameters, mediator)

    # This should not raise any exception
    detector.validate_configuration()

    # Test missing attribute
    delattr(detector, "pixscale_mas")
    with pytest.raises(
        AttributeError, match="Detector is missing attribute: pixscale_mas"
    ):
        detector.validate_configuration()

    # Restore the attribute and test incorrect type
    setattr(detector, "pixscale_mas", 10)  # Not a Quantity
    with pytest.raises(
        TypeError, match="Detector attribute pixscale_mas should be a Quantity"
    ):
        detector.validate_configuration()

    # Test incorrect units
    setattr(detector, "pixscale_mas", 10 * u.arcsec)
    with pytest.raises(
        ValueError, match="Detector attribute pixscale_mas has incorrect units"
    ):
        detector.validate_configuration()
