import pytest
import numpy as np
from astropy import units as u
from pyEDITH.observation import Observation
from pyEDITH.units import WAVELENGTH, DIMENSIONLESS, LAMBDA_D, TIME, MAGNITUDE


def test_observation_init():
    obs = Observation()
    assert obs.td_limit == 1e20 * TIME


def test_load_configuration(capsys):
    obs = Observation()

    # Test invalid wavelength grid
    parameters = {"observing_mode" : "IFS",
                  "wavelength" : [0.5, 0.5, 0.5],
                  "snr": [7.0, 7.0, 7.0],
                  "regrid_wavelength" : False,
                  "psf_trunc_ratio": 0.3,
                  "CRb_multiplier": 2.0,
                }
    
    obs.load_configuration(parameters)

    captured = capsys.readouterr()
    assert (
        "WARNING: Wavelength grid is not valid. Using default spectral resolution of 140."
        in captured.out
    )

    # Test no photometric_aperture_radius nor psf_trunc_ratio case
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "CRb_multiplier": 2.0,
    }
    with pytest.raises(KeyError):
        obs.load_configuration(parameters)

    # Test photometric_aperture_radius case
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "photometric_aperture_radius": 0.85,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    obs.load_configuration(parameters)

    assert np.all(obs.wavelength == parameters["wavelength"] * WAVELENGTH)
    assert np.all(obs.SNR == parameters["snr"] * DIMENSIONLESS)
    assert (
        obs.photometric_aperture_radius
        == parameters["photometric_aperture_radius"] * LAMBDA_D
    )
    assert obs.CRb_multiplier == parameters["CRb_multiplier"]

    # Test case when both are available
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "photometric_aperture_radius": 0.85,
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    obs.load_configuration(parameters)
    captured = capsys.readouterr()
    assert (
        "Warning: Both 'photometric_aperture_radius' and 'psf_trunc_ratio' provided. Using 'psf_trunc_ratio' and ignoring 'photometric_aperture_radius'."
        in captured.out
    )

    assert np.all(obs.wavelength == parameters["wavelength"] * WAVELENGTH)
    assert np.all(obs.SNR == parameters["snr"] * DIMENSIONLESS)
    assert obs.psf_trunc_ratio == parameters["psf_trunc_ratio"] * DIMENSIONLESS
    assert obs.CRb_multiplier == parameters["CRb_multiplier"]

    # Test case when neither are available
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    with pytest.raises(KeyError):
        obs.load_configuration(parameters)

    # Test psf_trunc_ratio case
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    obs.load_configuration(parameters)

    assert np.all(obs.wavelength == parameters["wavelength"] * WAVELENGTH)
    assert np.all(obs.SNR == parameters["snr"] * DIMENSIONLESS)
    assert obs.psf_trunc_ratio == parameters["psf_trunc_ratio"] * DIMENSIONLESS
    assert obs.CRb_multiplier == parameters["CRb_multiplier"]

    # Test single wavelength input
    single_params = parameters.copy()
    single_params["wavelength"] = [0.5]
    single_params["snr"] = [7.0]
    obs.load_configuration(single_params)
    assert len(obs.wavelength) == 1
    assert len(obs.SNR) == 1

    # Test error handling
    with pytest.raises(KeyError):
        obs.load_configuration({"invalid_key": 0})


    # Test IFS mode: spectral_resolution is not included
    parameters = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": [7.0, 7.0, 7.0],
        "channel_bounds" : [1.],
        "regrid_wavelength": True,
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IFS",
    }
    with pytest.raises(KeyError):
        obs.load_configuration(parameters)

    # Test IFS mode: channel_bounds is not included
    parameters = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": [7.0, 7.0, 7.0],
        "spectral_resolution" : [140, 40],
        "regrid_wavelength": True,
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IFS",
    }
    with pytest.raises(KeyError):
        obs.load_configuration(parameters)


    # Test IFS mode: spectral_resolution and channel_bounds included
    parameters = {
        "wavelength": np.linspace(0.5, 1.7, 1000),
        "snr": [7.0, 7.0, 7.0],
        "spectral_resolution" : [140, 40],
        "channel_bounds" : [1.],
        "regrid_wavelength": True,
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IFS",
    }
    obs.load_configuration(parameters)

    # Test whether the calculated spectral grid is at the correct resolution for the correct spectral channels
    assert np.all(obs.wavelength[obs.wavelength.value < parameters["channel_bounds"]] / obs.delta_wavelength[obs.wavelength.value < parameters["channel_bounds"]] == parameters["spectral_resolution"][0])
    assert np.all(obs.wavelength[obs.wavelength.value >= parameters["channel_bounds"]] / obs.delta_wavelength[obs.wavelength.value >= parameters["channel_bounds"]] == parameters["spectral_resolution"][1])


def test_set_output_arrays():
    obs = Observation()
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "photometric_aperture_radius": 0.85,
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    obs.load_configuration(parameters)
    obs.set_output_arrays()

    assert obs.tp == 0.0 * TIME
    assert obs.exptime.shape == (3,)
    assert obs.fullsnr.shape == (3,)
    assert np.all(obs.exptime == 0.0 * TIME)
    assert np.all(obs.fullsnr == 0.0 * DIMENSIONLESS)


def test_validate_configuration():
    obs = Observation()

    # Missing photometric_aperture_radius and psf_trunc_ratio
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "CRb_multiplier": 2.0,
    }
    with pytest.raises(KeyError):
        obs.load_configuration(parameters)

    with pytest.raises(AttributeError):
        obs.validate_configuration()

    # PSF trunc ratio
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "psf_trunc_ratio": 0.3,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    obs.load_configuration(parameters)

    # Test valid configuration
    obs.validate_configuration()  # This should not raise any exception

    # Photap rad
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "snr": [7.0, 7.0, 7.0],
        "photometric_aperture_radius": 0.7,
        "CRb_multiplier": 2.0,
        "observing_mode" : "IMAGER",
    }
    obs.load_configuration(parameters)

    # Test valid configuration
    obs.validate_configuration()  # This should not raise any exception

    # Test invalid types
    invalid_obs = Observation()
    invalid_obs.load_configuration(parameters)
    invalid_obs.wavelength = "invalid"
    with pytest.raises(TypeError):
        invalid_obs.validate_configuration()

    # Test missing attributes
    missing_attr_obs = Observation()
    missing_attr_obs.load_configuration(parameters)
    delattr(missing_attr_obs, "wavelength")
    with pytest.raises(AttributeError):
        missing_attr_obs.validate_configuration()

    # Test incorrect units
    incorrect_units_obs = Observation()
    incorrect_units_obs.load_configuration(parameters)
    incorrect_units_obs.wavelength = (
        incorrect_units_obs.wavelength.value * MAGNITUDE
    )  # Incorrect unit
    with pytest.raises(ValueError):
        incorrect_units_obs.validate_configuration()
