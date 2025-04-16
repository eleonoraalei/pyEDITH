import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock
from pyEDITH.exposure_time_calculator import (
    calculate_CRp,
    calculate_CRbs,
    calculate_CRbz,
    calculate_CRbez,
    calculate_CRbbin,
    calculate_CRbth,
    calculate_CRbd,
    calculate_CRnf,
    calculate_t_photon_count,
    calculate_exposure_time_or_snr,
)
from pyEDITH.units import (
    PHOTON_FLUX_DENSITY,
    DIMENSIONLESS,
    LENGTH,
    LAMBDA_D,
    ARCSEC,
    WAVELENGTH,
    TEMPERATURE,
    DARK_CURRENT,
    READ_NOISE,
    READ_TIME,
    CIC,
    QE,
)


# Mock classes for testing
class MockObservation:
    def __init__(self):
        self.wavelength = [500] * u.nm
        self.SNR = [10] * DIMENSIONLESS
        self.nlambd = 1


class MockScene:
    def __init__(self):
        self.F0 = [1e8] * PHOTON_FLUX_DENSITY
        self.Fstar = [1e-8] * DIMENSIONLESS
        self.Fp0 = [1e-10] * DIMENSIONLESS
        self.dist = 10 * u.pc
        self.Fzodi_list = [1e-7] * DIMENSIONLESS
        self.Fexozodi_list = [1e-8] * DIMENSIONLESS
        self.Fbinary_list = [0] * DIMENSIONLESS


class MockObservatory:
    def __init__(self):
        self.telescope = MagicMock()
        self.telescope.Area = 10 * u.m**2
        self.telescope.temperature = 290 * u.K
        self.detector = MagicMock()
        self.detector.DC = [1e-3] * DARK_CURRENT
        self.detector.RN = [2] * READ_NOISE
        self.detector.tread = [100] * READ_TIME
        self.detector.CIC = [1e-3] * CIC
        self.detector.QE = [0.9] * QE
        self.coronagraph = MagicMock()
        self.coronagraph.photap_frac = [0.5] * DIMENSIONLESS
        self.coronagraph.pixscale = 0.1 * LAMBDA_D
        self.coronagraph.nchannels = 1
        self.total_throughput = [0.3] * DIMENSIONLESS


def test_calculate_CRp():
    F0 = 1e8 * PHOTON_FLUX_DENSITY
    Fstar = 1e-8 * DIMENSIONLESS
    Fp0 = 1e-10 * DIMENSIONLESS
    area = 10 * u.m**2
    Upsilon = 0.5 * DIMENSIONLESS
    throughput = 0.3 * DIMENSIONLESS
    dlambda = 100 * u.nm
    nchannels = 1

    result = calculate_CRp(
        F0, Fstar, Fp0, area, Upsilon, throughput, dlambda, nchannels
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_CRbs():
    F0 = 1e8 * PHOTON_FLUX_DENSITY
    Fstar = 1e-8 * DIMENSIONLESS
    Istar = 1e-10 * DIMENSIONLESS
    area = 10 * u.m**2
    pixscale = 0.1 * LAMBDA_D
    throughput = 0.3 * DIMENSIONLESS
    dlambda = 100 * u.nm
    nchannels = 1

    result = calculate_CRbs(
        F0, Fstar, Istar, area, pixscale, throughput, dlambda, nchannels
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_CRbz():
    F0 = 1e8 * PHOTON_FLUX_DENSITY
    Fzodi = 1e-7 * DIMENSIONLESS
    lod_arcsec = 0.1 * u.arcsec
    skytrans = 0.9 * DIMENSIONLESS
    area = 10 * u.m**2
    throughput = 0.3 * DIMENSIONLESS
    dlambda = 100 * u.nm
    nchannels = 1

    result = calculate_CRbz(
        F0, Fzodi, lod_arcsec, skytrans, area, throughput, dlambda, nchannels
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_CRbez():
    F0 = 1e8 * PHOTON_FLUX_DENSITY
    Fexozodi = 1e-8 * DIMENSIONLESS
    lod_arcsec = 0.1 * u.arcsec
    skytrans = 0.9 * DIMENSIONLESS
    area = 10 * u.m**2
    throughput = 0.3 * DIMENSIONLESS
    dlambda = 100 * u.nm
    nchannels = 1
    dist = 10 * u.pc
    sp = 1 * u.arcsec

    result = calculate_CRbez(
        F0,
        Fexozodi,
        lod_arcsec,
        skytrans,
        area,
        throughput,
        dlambda,
        nchannels,
        dist,
        sp,
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_CRbbin():
    F0 = 1e8 * PHOTON_FLUX_DENSITY
    Fbinary = 0 * DIMENSIONLESS
    skytrans = 0.9 * DIMENSIONLESS
    area = 10 * u.m**2
    throughput = 0.3 * DIMENSIONLESS
    dlambda = 100 * u.nm
    nchannels = 1

    result = calculate_CRbbin(
        F0, Fbinary, skytrans, area, throughput, dlambda, nchannels
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value == 0


def test_calculate_CRbth():
    lam = 500 * u.nm
    area = 10 * u.m**2
    dlambda = 100 * u.nm
    temp = 290 * u.K
    lod_rad = 0.1 * u.rad
    emis = 0.1 * DIMENSIONLESS
    QE = 0.9 * QE
    dQE = 0.9 * DIMENSIONLESS

    result = calculate_CRbth(lam, area, dlambda, temp, lod_rad, emis, QE, dQE)
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_CRbd():
    det_npix = 100 * u.pixel
    det_DC = 1e-3 * DARK_CURRENT
    det_RN = 2 * READ_NOISE
    det_tread = 100 * READ_TIME
    det_CIC = 1e-3 * CIC
    t_photon_count = 1 * u.s

    result = calculate_CRbd(
        det_npix, det_DC, det_RN, det_tread, det_CIC, t_photon_count
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_CRnf():
    F0 = 1e8 * PHOTON_FLUX_DENSITY
    Fstar = 1e-8 * DIMENSIONLESS
    area = 10 * u.m**2
    pixscale = 0.1 * LAMBDA_D
    throughput = 0.3 * DIMENSIONLESS
    dlambda = 100 * u.nm
    nchannels = 1
    SNR = 10
    noisefloor = 1e-10 * DIMENSIONLESS

    result = calculate_CRnf(
        F0, Fstar, area, pixscale, throughput, dlambda, nchannels, SNR, noisefloor
    )
    assert result.unit.is_equivalent(u.photon / u.s)
    assert result.value > 0


def test_calculate_t_photon_count():
    det_npix = 100 * u.pixel
    det_CR = 1000 * u.photon / u.s

    result = calculate_t_photon_count(det_npix, det_CR)
    assert result.unit.is_equivalent(u.s)
    assert result.value > 0


@patch("pyEDITH.exposure_time_calculator.calculate_CRp")
@patch("pyEDITH.exposure_time_calculator.calculate_CRbs")
@patch("pyEDITH.exposure_time_calculator.calculate_CRbz")
@patch("pyEDITH.exposure_time_calculator.calculate_CRbez")
@patch("pyEDITH.exposure_time_calculator.calculate_CRbbin")
@patch("pyEDITH.exposure_time_calculator.calculate_CRbth")
@patch("pyEDITH.exposure_time_calculator.calculate_CRbd")
@patch("pyEDITH.exposure_time_calculator.calculate_CRnf")
def test_calculate_exposure_time_or_snr(
    mock_CRnf,
    mock_CRbd,
    mock_CRbth,
    mock_CRbbin,
    mock_CRbez,
    mock_CRbz,
    mock_CRbs,
    mock_CRp,
):
    # Set up mock return values
    mock_CRp.return_value = 10 * u.photon / u.s
    mock_CRbs.return_value = 1 * u.photon / u.s
    mock_CRbz.return_value = 1 * u.photon / u.s
    mock_CRbez.return_value = 1 * u.photon / u.s
    mock_CRbbin.return_value = 0 * u.photon / u.s
    mock_CRbth.return_value = 1 * u.photon / u.s
    mock_CRbd.return_value = 1 * u.photon / u.s
    mock_CRnf.return_value = 0.1 * u.photon / u.s

    observation = MockObservation()
    scene = MockScene()
    observatory = MockObservatory()

    # Test exposure time calculation
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert hasattr(observation, "exptime")
    assert observation.exptime.unit.is_equivalent(u.s)
    assert observation.exptime.value > 0

    # Test SNR calculation
    observation.obstime = 3600 * u.s
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, mode="signal_to_noise"
    )
    assert hasattr(observation, "fullsnr")
    assert observation.fullsnr.unit.is_equivalent(u.dimensionless_unscaled)
    assert observation.fullsnr.value > 0
