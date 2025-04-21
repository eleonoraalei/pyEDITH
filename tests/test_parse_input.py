import pytest
import numpy as np
from astropy import units as u
from pathlib import Path
import tempfile
import os
from pyEDITH.parse_input import (
    parse_input_file,
    parse_parameters,
    read_configuration,
    get_observatory_config,
)
from pyEDITH.units import WAVELENGTH, DIMENSIONLESS, LENGTH


@pytest.fixture
def sample_input_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(
            """
        ; This is a comment
        wavelength = 0.5
        Lstar = 1.0
        distance = 10
        magV = 5.0
        nzodis = 3.0
        observing_mode = 'IMAGER'
        secondary_wavelength = 1.0
        """
        )
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def sample_input_file_error():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".edith") as tmp:
        tmp.write(
            """
        ; This is a comment
        wavelength = [0.5, 0.6]
        Lstar = 1.0
        distance = 10
        magV = 5.0
        nzodis = 3.0
        observing_mode = 'IMAGER'
        secondary_wavelength = 1.0
        """
        )
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


def test_parse_input_file(sample_input_file,sample_input_file_error):
    variables, secondary_variables = parse_input_file(
        sample_input_file, secondary_flag=True
    )

    assert variables["wavelength"] == 0.5
    assert variables["Lstar"] == 1.0
    assert variables["distance"] == 10
    assert variables["magV"] == 5.0
    assert variables["nzodis"] == 3.0
    assert variables["observing_mode"] == "IMAGER"
    assert secondary_variables["wavelength"] == 1.0

    with pytest.raises(KeyError):
        variables, secondary_variables = parse_input_file(
        sample_input_file_error, secondary_flag=True
    )



def test_parse_parameters():
    parameters = {
        "wavelength":  [0.5,0.6,0.7],
        "Lstar": 1.0,
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
    }
    parsed = parse_parameters(parameters)

    assert np.all(parsed["wavelength"] == np.array([0.5,0.6,0.7] ))
    assert parsed["Lstar"] == 1.0
    assert parsed["distance"] == 10
    assert parsed["magV"] == 5.0
    assert parsed["nzodis"] == 3.0
    assert parsed["observing_mode"] == "IFS"
    assert parsed["nlambda"] == 3


def test_read_configuration(sample_input_file):
    parsed_parameters, parsed_secondary_parameters = read_configuration(
        sample_input_file, secondary_flag=True
    )

    assert np.all(parsed_parameters["wavelength"] == np.array([0.5]))
    assert parsed_parameters["Lstar"] == 1.0
    assert parsed_parameters["distance"] == 10
    assert parsed_parameters["magV"] == 5.0
    assert parsed_parameters["nzodis"] == 3.0
    assert parsed_parameters["observing_mode"] == "IMAGER"
    assert parsed_secondary_parameters["wavelength"] == np.array([1.0]) 


def test_get_observatory_config():
    parameters = {"observatory_preset": "EAC1"}
    config = get_observatory_config(parameters)
    assert config == "EAC1"

    parameters = {
        "telescope_type": "EAC1",
        "coronagraph_type": "AAVC",
        "detector_type": "EAC1",
    }
    config = get_observatory_config(parameters)
    assert config == {"telescope": "EAC1", "coronagraph": "AAVC", "detector": "EAC1"}

    with pytest.raises(ValueError):
        get_observatory_config({})



def test_parse_parameters_IFS_mode():
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


def test_parse_parameters_IMAGER_mode():
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": [0.5],
    }
    parsed = parse_parameters(parameters)

    assert parsed["observing_mode"] == "IMAGER"
    assert np.all(parsed["wavelength"] == np.array([0.5]))


if __name__ == "__main__":
    pytest.main()
