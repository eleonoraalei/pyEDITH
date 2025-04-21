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
    TIME,
    FRAME,
    CLOCK_INDUCED_CHARGE,
    QUANTUM_EFFICIENCY,
    ELECTRON,
    PHOTON_COUNT,
    INV_SQUARE_ARCSEC,
    PIXEL,
    ZODI
)
from pyEDITH.components.telescopes import ToyModelTelescope
from pyEDITH.components.coronagraphs import ToyModelCoronagraph
from pyEDITH.components.detectors import ToyModelDetector



# Mock classes for testing
class MockObservation:
    def __init__(self):
        self.td_limit= 1.e+20 *u.s
        self.wavelength=u.Quantity([0.5],u.micron)
        self.SNR = u.Quantity([7],DIMENSIONLESS)
        self.photap_rad = 0.85*LAMBDA_D
        self.CRb_multiplier = 2
        self.nlambd = 1
        self.tp = 0.*u.s 
        self.exptime = u.Quantity([0.],u.s)
        self.fullsnr = u.Quantity([0.],DIMENSIONLESS)



class MockScene:
    def __init__(self):
        self.F0V= 10374.9964895 * u.photon/u.nm/u.s/u.cm**2
        self.Lstar=0.86* u.L_sun
        self.dist = 14.8*u.pc
        self.F0=u.Quantity([12638.83670769],u.photon/u.nm/u.s/u.cm**2)
        self.vmag = 5.84*u.mag 
        self.mag = u.Quantity([6.189576],u.mag)
        self.deltamag=u.Quantity([25.5],u.mag)
        self.min_deltamag=25*u.mag 
        self.Fstar=u.Quantity([0.00334326],DIMENSIONLESS)
        self.Fp0=u.Quantity([6.30957344e-11],DIMENSIONLESS)
        self.Fp0_min = 1.e-10*DIMENSIONLESS
        self.angular_diameter_arcsec=0.01*ARCSEC
        self.nzodis=3*ZODI 
        self.ra = 236.00757737 *u.deg
        self.dec = 2.51516683 *u.deg
        self.separation =  0.0628 *u.arcsec
        self.xp =  0.0628 *u.arcsec
        self.yp =  0. *u.arcsec
        self.M_V = 4.98869142 *u.mag
        self.Fzodi_list=u.Quantity([6.11055505e-10], 1 / u.arcsec**2),
        self.Fexozodi_list=u.Quantity([2.97724302e-09], 1 / u.arcsec**2),
        self.Fbinary_list = u.Quantity([0], DIMENSIONLESS)

class MockObservatory:
    def __init__(self):
        self.observing_mode='IMAGER'
        self.optics_throughput=u.Quantity([0.362],DIMENSIONLESS)
        self.epswarmTrcold= u.Quantity([0.638], DIMENSIONLESS)
        self.total_throughput = u.Quantity([0.23135872], u.electron / u.photon)
        self.telescope = ToyModelTelescope()
        self.telescope.path= None
        self.telescope.keyword= 'ToyModel'
        self.telescope.diameter=7.87 *u.m
        self.telescope.unobscured_area=0.879
        self.telescope.toverhead_fixed= 8381.3 *u.s
        self.telescope.toverhead_multi = 1.1 * DIMENSIONLESS
        self.telescope.telescope_throughput= u.Quantity([0.823],DIMENSIONLESS)
        self.telescope.temperature= 290.*u.K
        self.telescope.Tcontam= 0.95* DIMENSIONLESS
        self.telescope.Area = 42.75906827 * u.m**2
        
        self.detector = ToyModelDetector()
        self.detector.path = None
        self.detector.keyword='ToyModel'
        self.detector.pixscale_mas= 6.55224925*u.mas
        self.detector.npix_multiplier= u.Quantity([1.],DIMENSIONLESS)
        self.detector.DC = u.Quantity([3.e-05] ,DARK_CURRENT)
        self.detector.RN = u.Quantity([0.], READ_NOISE)
        self.detector.tread = u.Quantity([1000.] ,READ_TIME)
        self.detector.CIC=u.Quantity([0.0013] ,CLOCK_INDUCED_CHARGE)
        self.detector.QE = u.Quantity([0.897] ,QUANTUM_EFFICIENCY)
        self.detector.dQE = u.Quantity([0.75],DIMENSIONLESS)
        
        self.coronagraph =ToyModelCoronagraph()
        self.coronagraph.path=None
        self.coronagraph.keyword='ToyModel'
        self.coronagraph.pixscale= 30. *LAMBDA_D
        self.coronagraph.minimum_IWA= 1. *LAMBDA_D
        self.coronagraph.maximum_OWA=60.*LAMBDA_D
        self.coronagraph.contrast=1.05e-13*DIMENSIONLESS
        self.coronagraph.noisefloor_factor= 0.03*DIMENSIONLESS
        self.coronagraph.bandwidth= 0.2
        self.coronagraph.Tcore=0.2968371 *DIMENSIONLESS
        self.coronagraph.TLyot= 0.65*DIMENSIONLESS
        self.coronagraph.nrolls=1
        self.coronagraph.nchannels= 2
        self.coronagraph.coronagraph_throughput= u.Quantity([0.44],DIMENSIONLESS)
        self.coronagraph.coronagraph_spectral_resolution=1.*DIMENSIONLESS
        self.coronagraph.npsfratios= 1
        self.coronagraph.npix =4
        self.coronagraph.xcenter= 2. *PIXEL
        self.coronagraph.ycenter= 2. *PIXEL
        self.coronagraph.r= u.Quantity([[63.63961031, 47.4341649 , 47.4341649 , 63.63961031],
            [47.4341649 , 21.21320344, 21.21320344, 47.4341649 ],
            [47.4341649 , 21.21320344, 21.21320344, 47.4341649 ],
            [63.63961031, 47.4341649 , 47.4341649 , 63.63961031]],LAMBDA_D),
        self.coronagraph.omega_lod= u.Quantity([[[2.26980069],
             [2.26980069],
             [2.26980069],
             [2.26980069]],
 
            [[2.26980069],
             [2.26980069],
             [2.26980069],
             [2.26980069]],
 
            [[2.26980069],
             [2.26980069],
             [2.26980069],
             [2.26980069]],
 
            [[2.26980069],
             [2.26980069],
             [2.26980069],
             [2.26980069]]],LAMBDA_D**2)
        self.coronagraph.skytrans=u.Quantity([[0.65, 0.65, 0.65, 0.65],
            [0.65, 0.65, 0.65, 0.65],
            [0.65, 0.65, 0.65, 0.65],
            [0.65, 0.65, 0.65, 0.65]], DIMENSIONLESS)
        self.coronagraph.photap_frac=u.Quantity([[[0.       ],
             [0.2968371],
             [0.2968371],
             [0.       ]],
 
            [[0.2968371],
             [0.2968371],
             [0.2968371],
             [0.2968371]],
 
            [[0.2968371],
             [0.2968371],
             [0.2968371],
             [0.2968371]],
 
            [[0.       ],
             [0.2968371],
             [0.2968371],
             [0.       ]]], DIMENSIONLESS)
        self.coronagraph.PSFpeak= u.Quantity(0.01625, DIMENSIONLESS)
        self.coronagraph.Istar= u.Quantity([[1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15]], DIMENSIONLESS)
        self.coronagraph.noisefloor= u.Quantity([[5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17]], DIMENSIONLESS)




def test_calculate_CRp():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fstar = 0.005311289818550127 * DIMENSIONLESS
    Fp0 = 1e-9 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    Upsilon = 0.2968371 * DIMENSIONLESS
    throughput = 0.35910000000000003 * ELECTRON/PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2

    result = calculate_CRp(
        F0, Fstar, Fp0, area, Upsilon, throughput, dlambda, nchannels
    )
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value,0.64877874) 



def test_calculate_CRbs():
    F0 = 13400.0  * PHOTON_FLUX_DENSITY
    Fstar = 0.005311289818550127 * DIMENSIONLESS
    Istar = 2.3272595994978797e-14* DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    pixscale = 0.25 * LAMBDA_D
    throughput = 0.35910000000000003 * ELECTRON/PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2

    result = calculate_CRbs(
        F0, Fstar, Istar, area, pixscale, throughput, dlambda, nchannels
    )
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value,0.0008138479 ) 


def test_calculate_CRbz():
    F0 = 13400.0  * PHOTON_FLUX_DENSITY
    Fzodi = 3.5213620474344346e-10 * INV_SQUARE_ARCSEC
    skytrans = 0.4006394155914143* DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON/PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2
    lod_arcsec = 0.013104498490920989 * ARCSEC
    
    result = calculate_CRbz(
        F0, Fzodi, lod_arcsec, skytrans, area, throughput, dlambda, nchannels
    )
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value,0.0099697346 ) 



def test_calculate_CRbez():
    F0 = 13400.0  * PHOTON_FLUX_DENSITY
    Fexozodi = 7.1490465158365465e-09 * INV_SQUARE_ARCSEC
    skytrans = 0.6161309232588068* DIMENSIONLESS
    sp=0.02784705929320709 * ARCSEC
    dist=18.195476531982425 *u.pc
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON/PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2
    lod_arcsec = 0.013104498490920989 * ARCSEC

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
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value,1.2124248  ) 


def test_calculate_CRbbin():
    F0 = 13400.0  * PHOTON_FLUX_DENSITY
    Fbinary = 0. * INV_SQUARE_ARCSEC  # ETC does not really calculate this properly
    skytrans = 0.6161309232588068* DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON/PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2


    result = calculate_CRbbin(
        F0, Fbinary, skytrans, area, throughput, dlambda, nchannels
    )
    assert result.unit==(u.electron / (u.s))
    assert result.value == 0


def test_calculate_CRbth():
    lam = 0.5*WAVELENGTH
    area = 427590.68268120557 * u.cm**2
    dlambda = 100 * u.nm
    temp = 290 * u.K
    lod_rad = 6.353240152477764e-08 * u.rad
    emis = 0.468* DIMENSIONLESS
    QE = 0.675 * QUANTUM_EFFICIENCY
    dQE = 1.0 * DIMENSIONLESS

    result = calculate_CRbth(lam, area, dlambda, temp, lod_rad, emis, QE, dQE)
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value, 2.848015E-30) 


def test_calculate_CRbd():
    det_npix = 9.054697 * PIXEL
    det_DC = 3e-05 * DARK_CURRENT
    det_RN = 2 * READ_NOISE
    det_tread = 1000.0 * READ_TIME
    det_CIC = 1e-3 * CLOCK_INDUCED_CHARGE
    t_photon_count = 13.79303 * TIME/FRAME

    result = calculate_CRbd(
        det_npix, det_DC, det_RN, det_tread, det_CIC, t_photon_count
    )
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value, 0.037146898) 


def test_calculate_CRnf():
    F0 = 13400.0  * PHOTON_FLUX_DENSITY
    Fstar = 0.005311289818550127 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    pixscale = 0.25 * LAMBDA_D
    throughput = 0.35910000000000003 * ELECTRON/PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2
    SNR = 7
    noisefloor= 7.25659425003725e-18 * DIMENSIONLESS

    result = calculate_CRnf(
        F0, Fstar, area, pixscale, throughput, dlambda, nchannels, SNR, noisefloor
    )
    assert result.unit==(u.electron / (u.s))
    assert np.isclose(result.value, 1.7763531E-6) 


def test_calculate_t_photon_count():
    det_npix = 9.054697 * PIXEL
    det_CR = 0.723971066592388 * ELECTRON/TIME

    result = calculate_t_photon_count(det_npix, det_CR)
    assert result.unit==(u.s/FRAME)
    assert np.isclose(result.value,1.8583934 )



def test_calculate_exposure_time_or_snr(capsys
):


    observation = MockObservation()
    scene = MockScene()
    observatory = MockObservatory()
    
   


    # Test exposure time calculation
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert hasattr(observation, "exptime")
    assert observation.exptime.unit==(u.s)
    assert np.isclose(observation.exptime.value,126150.86787119)

    # Test SNR calculation
    observation.obstime = 10 * u.hr
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, mode="signal_to_noise"
    )
    assert hasattr(observation, "fullsnr")
    assert observation.fullsnr.unit==(u.dimensionless_unscaled)
    assert np.isclose(observation.fullsnr.value,3.38987056)


    # checking infinity cases
    observatory.coronagraph.minimum_IWA = 6
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert hasattr(observation, "exptime")
    assert observation.exptime.unit==(u.s)
    assert (observation.exptime[0]) == np.inf

    captured = capsys.readouterr()
    assert "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results." in captured.out

    # restore IWA, increase noisefloor
    observatory.coronagraph.minimum_IWA = 1.
    observatory.coronagraph.noisefloor = 1e10*observatory.coronagraph.noisefloor
    
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert hasattr(observation, "exptime")
    assert observation.exptime.unit==(u.s)
    assert (observation.exptime[0]) == np.inf

    captured = capsys.readouterr()
    assert "WARNING: Count rate of the planet smaller than the noise floor. Hardcoded infinity results." in captured.out
