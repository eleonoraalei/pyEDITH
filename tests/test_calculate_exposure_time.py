import pytest
import numpy as np
from pyEDITH import AstrophysicalScene, Observation, Instrument, Edith
from pyEDITH.exposure_time_calculator import calculate_exposure_time


@pytest.fixture
def setup_edith():
    observation = Observation()
    scene = AstrophysicalScene()
    instrument = Instrument()
    edith = Edith(scene, observation)

    # Set up basic parameters
    observation.nlambd = 1
    observation.lambd = np.array([0.5])  # 500 nm
    observation.SR = np.array([5.0])
    observation.SNR = np.array([7.0])

    scene.ntargs = 1
    scene.mag = np.array([[10.0]])
    scene.F0 = np.array([1e-8])
    scene.Fzodi_list = np.array([[1e-10]])
    scene.Fexozodi_list = np.array([[1e-11]])
    scene.Fbinary_list = np.array([[0.0]])
    scene.sp = np.array([[[1.0]]])
    scene.xp = np.array([[[1.0]]])
    scene.yp = np.array([[[0.0]]])
    scene.Fp0 = np.array([[[1e-10]]])

    instrument.telescope.D = 7.87
    instrument.telescope.Area = 48.0
    instrument.telescope.throughput = np.array([0.36])
    instrument.coronagraph.bandwidth = 0.2
    instrument.coronagraph.IWA = 2.0
    instrument.coronagraph.OWA = 100.0
    instrument.coronagraph.contrast = 1e-10
    instrument.coronagraph.pixscale = 0.25
    instrument.coronagraph.npix = 480
    instrument.coronagraph.xcenter = 240.0
    instrument.coronagraph.ycenter = 240.0

    edith.norbits = 1
    edith.nmeananom = 1

    return observation, scene, instrument, edith


def test_planet_outside_OWA(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.sp = np.array([[[1000.0]]])
    scene.xp = np.array([[[1000.0]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_planet_inside_IWA(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.sp = np.array([[[0.1]]])
    scene.xp = np.array([[[0.1]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_planet_below_noise_floor(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.Fp0 = np.array([[[1e-20]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_exposure_time_exceeds_limit(setup_edith):
    observation, scene, instrument, edith = setup_edith
    edith.td_limit = 1e-6  # Set a very low limit
    scene.Fp0 = np.array([[[1e-15]]])  # Weak planet signal
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_negative_exposure_time(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.coronagraph.contrast = 1e-5  # High contrast
    scene.Fp0 = np.array([[[1e-3]]])  # Very bright planet
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_valid_exposure_time(setup_edith):
    observation, scene, instrument, edith = setup_edith
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_zero_flux_from_planet(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.Fp0 = np.array([[[0.0]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_extremely_bright_planet(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.Fp0 = np.array([[[1e10]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_zero_coronagraph_throughput(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.coronagraph.Tcore = 0.0
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_zero_telescope_area(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.telescope.Area = 0.0
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_zero_bandwidth(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.coronagraph.bandwidth = 0.0
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_planet_at_edge_of_OWA(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.sp = np.array([[[instrument.coronagraph.OWA - 0.01]]])
    scene.xp = np.array([[[instrument.coronagraph.OWA - 0.01]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_planet_at_edge_of_IWA(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.sp = np.array([[[instrument.coronagraph.IWA + 0.01]]])
    scene.xp = np.array([[[instrument.coronagraph.IWA + 0.01]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_multiple_wavelengths(setup_edith):
    observation, scene, instrument, edith = setup_edith
    observation.nlambd = 3
    observation.lambd = np.array([0.5, 1.0, 1.5])
    observation.SR = np.array([5.0, 5.0, 5.0])
    observation.SNR = np.array([7.0, 7.0, 7.0])
    scene.mag = np.array([[10.0, 9.5, 9.0]])
    scene.F0 = np.array([1e-8, 1e-8, 1e-8])
    scene.Fzodi_list = np.array([[1e-10, 1e-10, 1e-10]])
    scene.Fexozodi_list = np.array([[1e-11, 1e-11, 1e-11]])
    scene.Fbinary_list = np.array([[0.0, 0.0, 0.0]])
    instrument.telescope.throughput = np.array([0.36, 0.36, 0.36])

    calculate_exposure_time(observation, scene, instrument, edith)
    assert edith.exptime.shape == (1, 3)
    assert not np.any(np.isinf(edith.exptime))
    assert np.all(edith.exptime > 0)


def test_multiple_targets(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.ntargs = 2
    scene.mag = np.array([[10.0], [11.0]])
    scene.sp = np.array([[[1.0]], [[1.5]]])
    scene.xp = np.array([[[1.0]], [[1.5]]])
    scene.yp = np.array([[[0.0]], [[0.0]]])
    scene.Fp0 = np.array([[[1e-10]], [[1e-11]]])

    calculate_exposure_time(observation, scene, instrument, edith)
    assert edith.exptime.shape == (2, 1)
    assert not np.any(np.isinf(edith.exptime))
    assert np.all(edith.exptime > 0)


def test_extreme_snr_requirement(setup_edith):
    observation, scene, instrument, edith = setup_edith
    observation.SNR = np.array([1000.0])  # Very high SNR requirement
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_very_low_snr_requirement(setup_edith):
    observation, scene, instrument, edith = setup_edith
    observation.SNR = np.array([0.1])  # Very low SNR requirement
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_extreme_contrast(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.coronagraph.contrast = 1e-20  # Extremely high contrast
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_no_contrast(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.coronagraph.contrast = 1.0  # No contrast
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_extreme_zodi(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.Fzodi_list = np.array([[1e-5]])  # Very high zodiacal light
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_extreme_exozodi(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.Fexozodi_list = np.array([[1e-5]])  # Very high exozodiacal light
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_no_background(setup_edith):
    observation, scene, instrument, edith = setup_edith
    scene.Fzodi_list = np.array([[0.0]])
    scene.Fexozodi_list = np.array([[0.0]])
    scene.Fbinary_list = np.array([[0.0]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_extreme_telescope_diameter(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.telescope.D = 100.0  # Very large telescope
    instrument.telescope.Area = (
        np.pi * (instrument.telescope.D / 2) ** 2 * (1.0 - 0.121)
    )
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_tiny_telescope_diameter(setup_edith):
    observation, scene, instrument, edith = setup_edith
    instrument.telescope.D = 0.1  # Very small telescope
    instrument.telescope.Area = (
        np.pi * (instrument.telescope.D / 2) ** 2 * (1.0 - 0.121)
    )
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_extreme_spectral_resolution(setup_edith):
    observation, scene, instrument, edith = setup_edith
    observation.SR = np.array([1e6])  # Very high spectral resolution
    calculate_exposure_time(observation, scene, instrument, edith)
    assert np.isinf(edith.exptime[0, 0])


def test_low_spectral_resolution(setup_edith):
    observation, scene, instrument, edith = setup_edith
    observation.SR = np.array([1.0])  # Very low spectral resolution
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.isinf(edith.exptime[0, 0])
    assert edith.exptime[0, 0] > 0


def test_multiple_orbits(setup_edith):
    observation, scene, instrument, edith = setup_edith
    edith.norbits = 3
    scene.sp = np.array([[[1.0, 1.5, 2.0]]])
    scene.xp = np.array([[[1.0, 1.5, 2.0]]])
    scene.yp = np.array([[[0.0, 0.0, 0.0]]])
    scene.Fp0 = np.array([[[1e-10, 1e-11, 1e-12]]])
    calculate_exposure_time(observation, scene, instrument, edith)
    assert not np.any(np.isinf(edith.exptime))
