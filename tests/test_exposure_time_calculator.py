import pytest
import numpy as np
from astropy import units as u
from unittest.mock import patch, MagicMock

from pyEDITH.exposure_time_calculator import *
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
    ZODI,
)
from pyEDITH.components.telescopes import ToyModelTelescope
from pyEDITH.components.coronagraphs import ToyModelCoronagraph
from pyEDITH.components.detectors import ToyModelDetector


# Mock classes for testing
class MockObservation:
    def __init__(self):
        self.td_limit = 1.0e20 * u.s
        self.wavelength = u.Quantity([0.5], u.micron)
        self.SNR = u.Quantity([7], DIMENSIONLESS)
        self.photometric_aperture_radius = 0.85 * LAMBDA_D
        self.CRb_multiplier = 2
        self.nlambd = 1
        self.tp = 0.0 * u.s
        self.exptime = u.Quantity([0.0], u.s)
        self.fullsnr = u.Quantity([0.0], DIMENSIONLESS)
        self.snr_ez = u.Quantity([0.0], DIMENSIONLESS)


class MockScene:
    def __init__(self):
        self.F0V = 10374.9964895 * u.photon / u.nm / u.s / u.cm**2
        self.Lstar = 0.86 * u.L_sun
        self.dist = 14.8 * u.pc
        self.F0 = u.Quantity([12638.83670769], u.photon / u.nm / u.s / u.cm**2)
        self.vmag = 5.84 * u.mag
        self.mag = u.Quantity([6.189576], u.mag)
        self.deltamag = u.Quantity([25.5], u.mag)
        self.min_deltamag = 25 * u.mag
        self.Fs_over_F0 = u.Quantity([0.00334326], DIMENSIONLESS)
        self.Fp_over_Fs = u.Quantity([6.30957344e-11], DIMENSIONLESS)
        self.Fp_min_over_Fs = 1.0e-10 * DIMENSIONLESS
        self.stellar_angular_diameter_arcsec = 0.01 * ARCSEC
        self.nzodis = 3 * ZODI
        self.ra = 236.00757737 * u.deg
        self.dec = 2.51516683 * u.deg
        self.separation = 0.0628 * u.arcsec
        self.xp = 0.0628 * u.arcsec
        self.yp = 0.0 * u.arcsec
        self.M_V = 4.98869142 * u.mag
        self.Fzodi_list = (u.Quantity([6.11055505e-10], 1 / u.arcsec**2),)
        self.Fexozodi_list = (u.Quantity([2.97724302e-09], 1 / u.arcsec**2),)
        self.Fbinary_list = u.Quantity([0], DIMENSIONLESS)


class MockObservatory:
    def __init__(self):
        self.observing_mode = "IMAGER"
        self.optics_throughput = u.Quantity([0.362], DIMENSIONLESS)
        self.epswarmTrcold = u.Quantity([0.638], DIMENSIONLESS)
        self.total_throughput = u.Quantity([0.23135872], u.electron / u.photon)
        self.telescope = ToyModelTelescope()
        self.telescope.path = None
        self.telescope.keyword = "ToyModel"
        self.telescope.diameter = 7.87 * u.m
        self.telescope.unobscured_area = 0.879
        self.telescope.toverhead_fixed = 8381.3 * u.s
        self.telescope.toverhead_multi = 1.1 * DIMENSIONLESS
        self.telescope.telescope_optical_throughput = u.Quantity([0.823], DIMENSIONLESS)
        self.telescope.temperature = 290.0 * u.K
        self.telescope.T_contamination = 0.95 * DIMENSIONLESS
        self.telescope.Area = 42.75906827 * u.m**2

        self.detector = ToyModelDetector()
        self.detector.path = None
        self.detector.keyword = "ToyModel"
        self.detector.pixscale_mas = 6.55224925 * u.mas
        self.detector.npix_multiplier = u.Quantity([1.0], DIMENSIONLESS)
        self.detector.DC = u.Quantity([3.0e-05], DARK_CURRENT)
        self.detector.RN = u.Quantity([0.0], READ_NOISE)
        self.detector.tread = u.Quantity([1000.0], READ_TIME)
        self.detector.CIC = u.Quantity([0.0013], CLOCK_INDUCED_CHARGE)
        self.detector.QE = u.Quantity([0.897], QUANTUM_EFFICIENCY)
        self.detector.dQE = u.Quantity([0.75], DIMENSIONLESS)

        self.coronagraph = ToyModelCoronagraph()
        self.coronagraph.path = None
        self.coronagraph.keyword = "ToyModel"
        self.coronagraph.pixscale = 30.0 * LAMBDA_D
        self.coronagraph.minimum_IWA = 1.0 * LAMBDA_D
        self.coronagraph.maximum_OWA = 60.0 * LAMBDA_D
        self.coronagraph.contrast = 1.05e-13 * DIMENSIONLESS
        self.coronagraph.noisefloor_factor = 0.03 * DIMENSIONLESS
        self.coronagraph.bandwidth = 0.2
        self.coronagraph.Tcore = 0.2968371 * DIMENSIONLESS
        self.coronagraph.TLyot = 0.65 * DIMENSIONLESS
        self.coronagraph.nrolls = 2
        self.coronagraph.nchannels = 2
        self.coronagraph.coronagraph_optical_throughput = u.Quantity(
            [0.44], DIMENSIONLESS
        )
        self.coronagraph.coronagraph_spectral_resolution = 1.0 * DIMENSIONLESS
        self.coronagraph.npsfratios = 1
        self.coronagraph.npix = 4
        self.coronagraph.xcenter = 2.0 * PIXEL
        self.coronagraph.ycenter = 2.0 * PIXEL
        self.coronagraph.r = (
            u.Quantity(
                [
                    [63.63961031, 47.4341649, 47.4341649, 63.63961031],
                    [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                    [47.4341649, 21.21320344, 21.21320344, 47.4341649],
                    [63.63961031, 47.4341649, 47.4341649, 63.63961031],
                ],
                LAMBDA_D,
            ),
        )
        self.coronagraph.omega_lod = u.Quantity(
            [
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
                [[2.26980069], [2.26980069], [2.26980069], [2.26980069]],
            ],
            LAMBDA_D**2,
        )
        self.coronagraph.skytrans = u.Quantity(
            [
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
                [0.65, 0.65, 0.65, 0.65],
            ],
            DIMENSIONLESS,
        )
        self.coronagraph.photometric_aperture_throughput = u.Quantity(
            [
                [[0.0], [0.2968371], [0.2968371], [0.0]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.2968371], [0.2968371], [0.2968371], [0.2968371]],
                [[0.0], [0.2968371], [0.2968371], [0.0]],
            ],
            DIMENSIONLESS,
        )
        self.coronagraph.PSFpeak = u.Quantity(0.01625, DIMENSIONLESS)
        self.coronagraph.Istar = u.Quantity(
            [
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
                [1.70625e-15, 1.70625e-15, 1.70625e-15, 1.70625e-15],
            ],
            DIMENSIONLESS,
        )
        self.coronagraph.noisefloor = u.Quantity(
            [
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
                [5.11875e-17, 5.11875e-17, 5.11875e-17, 5.11875e-17],
            ],
            DIMENSIONLESS,
        )


def test_calculate_CRp():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fs_over_F0 = 0.005311289818550127 * DIMENSIONLESS
    Fp_over_Fs = 1e-9 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    Upsilon = 0.2968371 * DIMENSIONLESS
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2

    result = calculate_CRp(
        F0, Fs_over_F0, Fp_over_Fs, area, Upsilon, throughput, dlambda, nchannels
    )
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 0.64877874)


def test_calculate_CRbs():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fs_over_F0 = 0.005311289818550127 * DIMENSIONLESS
    Istar = 2.3272595994978797e-14 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    pixscale = 0.25 * LAMBDA_D
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2

    result = calculate_CRbs(
        F0, Fs_over_F0, Istar, area, pixscale, throughput, dlambda, nchannels
    )
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 0.0008138479)


def test_calculate_CRbz():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fzodi = 3.5213620474344346e-10 * INV_SQUARE_ARCSEC
    skytrans = 0.4006394155914143 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2
    lod_arcsec = 0.013104498490920989 * ARCSEC

    result = calculate_CRbz(
        F0, Fzodi, lod_arcsec, skytrans, area, throughput, dlambda, nchannels
    )
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 0.0099697346)


def test_calculate_CRbez():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fexozodi = 7.1490465158365465e-09 * INV_SQUARE_ARCSEC
    skytrans = 0.6161309232588068 * DIMENSIONLESS
    sp = 0.02784705929320709 * ARCSEC
    dist = 18.195476531982425 * u.pc
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
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
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 1.2124248)


def test_calculate_CRbbin():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fbinary = 0.0 * INV_SQUARE_ARCSEC  # ETC does not really calculate this properly
    skytrans = 0.6161309232588068 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2

    result = calculate_CRbbin(
        F0, Fbinary, skytrans, area, throughput, dlambda, nchannels
    )
    assert result.unit == (u.electron / (u.s))
    assert result.value == 0


def test_calculate_CRbth():
    lam = 0.5 * WAVELENGTH
    area = 427590.68268120557 * u.cm**2
    dlambda = 100 * u.nm
    temp = 290 * u.K
    lod_rad = 6.353240152477764e-08 * u.rad
    emis = 0.468 * DIMENSIONLESS
    QE = 0.675 * QUANTUM_EFFICIENCY
    dQE = 1.0 * DIMENSIONLESS

    result = calculate_CRbth(lam, area, dlambda, temp, lod_rad, emis, QE, dQE)
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 2.848015e-30)


def test_calculate_CRbd():
    det_npix = 9.054697 * PIXEL
    det_DC = 3e-05 * DARK_CURRENT
    det_RN = 2 * READ_NOISE
    det_tread = 1000.0 * READ_TIME
    det_CIC = 1e-3 * CLOCK_INDUCED_CHARGE
    t_photon_count = 13.79303 * TIME / FRAME

    result = calculate_CRbd(
        det_npix, det_DC, det_RN, det_tread, det_CIC, t_photon_count
    )
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 0.037146898)


def test_calculate_CRnf():
    F0 = 13400.0 * PHOTON_FLUX_DENSITY
    Fs_over_F0 = 0.005311289818550127 * DIMENSIONLESS
    area = 427590.68268120557 * u.cm**2
    pixscale = 0.25 * LAMBDA_D
    throughput = 0.35910000000000003 * ELECTRON / PHOTON_COUNT
    dlambda = 100 * u.nm
    nchannels = 2
    SNR = 7
    noisefloor = 7.25659425003725e-18 * DIMENSIONLESS

    result = calculate_CRnf(
        F0, Fs_over_F0, area, pixscale, throughput, dlambda, nchannels, SNR, noisefloor
    )
    assert result.unit == (u.electron / (u.s))
    assert np.isclose(result.value, 1.7763531e-6)


def test_calculate_t_photon_count():
    det_npix = 9.054697 * PIXEL
    det_CR = 0.723971066592388 * ELECTRON / TIME

    result = calculate_t_photon_count(det_npix, det_CR)
    assert result.unit == (u.s / FRAME)
    assert np.isclose(result.value, 1.8583934)


def test_calculate_exposure_time_or_snr(capsys):

    observation = MockObservation()
    scene = MockScene()
    observatory = MockObservatory()

    # Test exposure time calculation
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert hasattr(observation, "exptime")
    assert observation.exptime.unit == (u.s)
    assert np.isclose(observation.exptime.value, 252301.15315671)

    # Test SNR calculation
    observation.obstime = 10 * u.hr
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, mode="signal_to_noise"
    )
    assert hasattr(observation, "fullsnr")
    assert observation.fullsnr.unit == (u.dimensionless_unscaled)
    assert np.isclose(observation.fullsnr.value, 5.14487031)
    assert np.isclose(observation.snr_ez.value, 31.60907842)

    # Setting values (used for ETC validation)
    observatory.detector.det_npix_input = 100 * DIMENSIONLESS
    observatory.detector.t_photon_count_input = 1.0 * SECOND / FRAME

    # Run calculation with ETC_validation=True
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, ETC_validation=True
    )
    # Check that det_npix and t_photon_count were fixed to input values
    assert np.isclose(observation.validation_variables[0]["det_npix"].value, 100)
    assert np.isclose(observation.validation_variables[0]["t_photon_count"].value, 1.0)

    # INFINITY CASES
    # Case 1: Planet outside OWA
    observatory.coronagraph.maximum_OWA = 0.5 * LAMBDA_D
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert np.isinf(observation.exptime[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results."
        in captured.out
    )
    ## Same for SNR
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, mode="signal_to_noise"
    )
    assert np.isinf(observation.fullsnr[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results."
        in captured.out
    )

    observatory.coronagraph.maximum_OWA = 60.0 * LAMBDA_D  # Reset to original value

    # Case 2: Planet inside IWA
    observatory.coronagraph.minimum_IWA = 10.0 * LAMBDA_D
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert np.isinf(observation.exptime[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results."
        in captured.out
    )
    ## Same for SNR
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, mode="signal_to_noise"
    )
    assert np.isinf(observation.fullsnr[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results."
        in captured.out
    )
    observatory.coronagraph.minimum_IWA = 1.0 * LAMBDA_D  # Reset to original value

    # Case 3: Photometric aperture not large enough
    original_omega_lod = observatory.coronagraph.omega_lod.copy()
    observatory.coronagraph.omega_lod = u.Quantity(
        np.zeros_like(original_omega_lod), LAMBDA_D**2
    )
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert np.isinf(observation.exptime[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Photometric aperture is not large enough. Hardcoded infinity results."
        in captured.out
    )
    ## same for SNR
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose=False, mode="signal_to_noise"
    )
    assert np.isinf(observation.fullsnr[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Photometric aperture is not large enough. Hardcoded infinity results."
        in captured.out
    )
    observatory.coronagraph.omega_lod = original_omega_lod  # Reset to original value

    # Case 4: Count rate of the planet smaller than the noise floor
    original_noisefloor = observatory.coronagraph.noisefloor.copy()
    observatory.coronagraph.noisefloor = u.Quantity(
        np.ones_like(original_noisefloor) * 1e10, DIMENSIONLESS
    )
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert np.isinf(observation.exptime[0])
    captured = capsys.readouterr()
    assert (
        "WARNING: Count rate of the planet smaller than the noise floor. Hardcoded infinity results."
        in captured.out
    )
    observatory.coronagraph.noisefloor = original_noisefloor  # Reset to original value

    # Case 5: Negative exposure time (does not make sense)
    # This case doesn't print a warning, it just sets the exposure time to infinity
    original_toverhead_fixed = observatory.telescope.toverhead_fixed
    observatory.telescope.toverhead_fixed = -1000000 * u.s  # just to make the test fail
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert np.isinf(observation.exptime[0])
    observatory.telescope.toverhead_fixed = (
        original_toverhead_fixed  # Reset to original value
    )

    # Case 6: Exposure time beyond td_limit
    # This case doesn't print a warning, it just sets the exposure time to infinity
    original_td_limit = observation.td_limit
    observation.td_limit = 1 * u.s
    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
    assert np.isinf(observation.exptime[0])
    observation.td_limit = original_td_limit  # Reset to original value

    # Invalid observing mode
    with pytest.raises(
        ValueError, match="Invalid mode. Use 'exposure_time' or 'signal_to_noise'."
    ):
        calculate_exposure_time_or_snr(
            observation, scene, observatory, verbose=False, mode="invalid"
        )

    # Testing verbose output
    with patch(
        "pyEDITH.exposure_time_calculator.utils.print_all_variables"
    ) as mock_print:
        calculate_exposure_time_or_snr(observation, scene, observatory, verbose=True)

        # Assert that print_all_variables was called
        mock_print.assert_called()

        # If you want to be more specific, you can check that it was called with the correct arguments
        args, kwargs = mock_print.call_args
        assert args[0] == observation
        assert args[1] == scene
        assert args[2] == observatory
        # You can add more assertions for other arguments if needed

    # Test that it's not called when verbose is False
    with patch(
        "pyEDITH.exposure_time_calculator.utils.print_all_variables"
    ) as mock_print:
        calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)
        mock_print.assert_not_called()

    # Bandwidth restriction warning
    observation.wavelength = [0.5] * u.um
    observation.nlambd = 1
    observatory.observing_mode = "IMAGER"
    observatory.coronagraph.bandwidth = 1.0
    observatory.coronagraph.coronagraph_spectral_resolution = 10

    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)

    captured = capsys.readouterr()
    assert (
        "WARNING: Bandwidth larger than what the coronagraph allows. Selecting widest possible bandwidth..."
        in captured.out
    )

    # Check if the bandwidth was correctly adjusted
    assert np.isclose(observation.validation_variables[0]["deltalambda_nm"].value, 50)

    # IFS-specific calculations
    # Correct calculation
    observatory.observing_mode = "IFS"
    observation.wavelength = [0.5, 0.6, 0.7] * u.um

    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)

    # Check that deltalambda_nm was calculated correctly
    expected_dlam = np.gradient(observation.wavelength)
    assert np.allclose(
        observation.validation_variables[0]["deltalambda_nm"].value,
        expected_dlam[0].to(u.nm).value,
    )

    # Wrong calculation
    observatory.observing_mode = "IFS"
    observation.wavelength = [0.5, 0.5, 0.5] * u.um  # Invalid grid (no gradient)

    calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)

    captured = capsys.readouterr()
    assert (
        "WARNING: Wavelength grid is not valid. Using default spectral resolution of 140."
        in captured.out
    )

    # Check that default resolution was used
    expected_dlam = observation.wavelength[0] / 140
    assert np.isclose(
        observation.validation_variables[0]["deltalambda_nm"].value,
        expected_dlam.to(u.nm).value,
    )

    # Invalid observing mode
    observatory.observing_mode = "INVALID"

    with pytest.raises(
        ValueError, match="Invalid observation mode. Choose 'IMAGER' or 'IFS'."
    ):
        calculate_exposure_time_or_snr(observation, scene, observatory, verbose=False)


def test_measure_coronagraph_performance_at_IWA():
    # Create a mock coronagraph
    coronagraph = MagicMock()
    coronagraph.photometric_aperture_throughput = (
        np.ones((100, 100, 1)) * 0.5 * DIMENSIONLESS
    )
    coronagraph.Istar = np.ones((100, 100)) * 1e-10 * DIMENSIONLESS
    coronagraph.skytrans = np.ones((100, 100)) * 0.9 * DIMENSIONLESS
    coronagraph.omega_lod = np.ones((100, 100, 1)) * 2 * LAMBDA_D**2
    coronagraph.npix = 100
    coronagraph.xcenter = 50 * PIXEL
    coronagraph.ycenter = 50 * PIXEL

    oneopixscale_arcsec = 10 / u.arcsec  # This is just an example value

    # Run the function
    result = measure_coronagraph_performance_at_IWA(
        coronagraph.photometric_aperture_throughput,
        coronagraph.Istar,
        coronagraph.skytrans,
        coronagraph.omega_lod,
        coronagraph.npix,
        coronagraph.xcenter,
        coronagraph.ycenter,
        oneopixscale_arcsec,
    )

    # Check the results
    assert len(result) == 6
    assert all(isinstance(r, u.Quantity) for r in result)
