import numpy as np
import pytest
import astropy.units as u
from pyEDITH.units import (
    LUMINOSITY,
    DISTANCE,
    MAGNITUDE,
    ARCSEC,
    ZODI,
    DEG,
    WAVELENGTH,
    PHOTON_FLUX_DENSITY,
    DIMENSIONLESS,
    INV_SQUARE_ARCSEC,
    SPECTRAL_FLUX_DENSITY_CGS,
)


from pyEDITH.astrophysical_scene import (
    AstrophysicalScene,
    calc_flux_zero_point,
    calc_exozodi_flux,
    calc_zodi_flux,
)


# Test calc_flux_zero_point function
def test_calc_flux_zero_point(capsys):
    wavelength = 500 * u.nm
    f0 = calc_flux_zero_point(wavelength)
    assert isinstance(f0, u.Quantity)
    assert f0.unit.is_equivalent(u.photon * u.s**-1 * u.cm**-2 * u.Hz**-1)

    # check various output units
    f0_pcgs = calc_flux_zero_point(wavelength, output_unit="pcgs")
    assert f0_pcgs.unit.is_equivalent(u.photon * u.s**-1 * u.cm**-2 * u.Hz**-1)

    f0_cgs = calc_flux_zero_point(wavelength, output_unit="cgs")
    assert f0_cgs.unit.is_equivalent(SPECTRAL_FLUX_DENSITY_CGS)

    f0_jy = calc_flux_zero_point(wavelength, output_unit="jy")
    assert f0_jy.unit.is_equivalent(u.Jy)

    # check perlambd flag
    f0_freq = calc_flux_zero_point(wavelength, output_unit="pcgs", perlambd=False)
    f0_wave = calc_flux_zero_point(wavelength, output_unit="pcgs", perlambd=True)

    assert f0_freq.unit != f0_wave.unit
    assert f0_wave.unit.is_equivalent(PHOTON_FLUX_DENSITY)

    # check perlambd flag cgs version
    f0_freq = calc_flux_zero_point(wavelength, output_unit="cgs", perlambd=False)
    f0_wave = calc_flux_zero_point(wavelength, output_unit="cgs", perlambd=True)

    assert f0_freq.unit != f0_wave.unit
    assert f0_wave.unit.is_equivalent(u.erg / (u.s * u.cm**3))

    # check AB system
    f0_johnson = calc_flux_zero_point(wavelength, output_unit="jy", AB=False)
    f0_ab = calc_flux_zero_point(wavelength, output_unit="jy", AB=True)

    assert np.isclose(f0_ab.value, 3631, rtol=1e-6)
    assert f0_johnson.value != f0_ab.value

    # test invalid output string
    with pytest.raises(ValueError):
        calc_flux_zero_point(wavelength, output_unit="invalid")

    # test error Jy+perlambd
    with pytest.raises(ValueError):
        calc_flux_zero_point(wavelength, output_unit="jy", perlambd=True)

    # test verbose
    calc_flux_zero_point(wavelength, verbose=True)
    captured = capsys.readouterr()
    assert "Flux zero point calculated" in captured.out

    # test array shaped wavelengths
    wavelengths = np.array([0.3, 0.5, 1, 5]) * WAVELENGTH

    f0_array = calc_flux_zero_point(wavelengths, output_unit="pcgs", perlambd=True)
    assert isinstance(f0_array, u.Quantity)
    assert f0_array.shape == wavelengths.shape
    assert f0_array.unit.is_equivalent(PHOTON_FLUX_DENSITY)


def test_calc_exozodi_flux():
    # Basic setup
    M_V = 5 * MAGNITUDE
    vmag = 6 * MAGNITUDE
    nexozodis = 1 * ZODI
    lambd = np.array([0.5, 0.6, 0.7]) * WAVELENGTH
    lambdmag = np.array([6, 5.9, 5.8]) * MAGNITUDE

    # Test basic functionality
    exozodi_flux = calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)
    assert isinstance(exozodi_flux, u.Quantity)
    assert exozodi_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(exozodi_flux) == len(lambd)

    # Test single wavelength (one-dimensional array)
    single_flux = calc_exozodi_flux(
        M_V, vmag, nexozodis, u.Quantity([lambd[0]]), u.Quantity([lambdmag[0]])
    )
    assert isinstance(single_flux, u.Quantity)
    assert single_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(single_flux) == 1

    # Test multiple zodi
    multi_zodi_flux = calc_exozodi_flux(M_V, vmag, 3 * ZODI, lambd, lambdmag)
    assert np.allclose(multi_zodi_flux.value, 3 * exozodi_flux.value)

    # Test zero zodi
    zero_flux = calc_exozodi_flux(M_V, vmag, 0 * ZODI, lambd, lambdmag)
    assert np.all(zero_flux.value == 0)

    # Test input validation
    with pytest.raises(ValueError):
        calc_exozodi_flux(
            M_V, vmag, nexozodis, lambd, lambdmag[:-1]
        )  # Mismatched length

    # Test units
    flux_dimensionless = exozodi_flux.to(
        DIMENSIONLESS, equivalencies=u.dimensionless_angles()
    )
    assert isinstance(flux_dimensionless, u.Quantity)
    assert flux_dimensionless.unit == DIMENSIONLESS

    # Test: real values
    M_V = 4.144 * MAGNITUDE
    vmag = 5.444 * MAGNITUDE
    nexozodis = 3 * ZODI
    lambd = np.array([0.5]) * WAVELENGTH
    lambdmag = np.array([5.687]) * MAGNITUDE

    # Test basic functionality
    exozodi_flux = calc_exozodi_flux(M_V, vmag, nexozodis, lambd, lambdmag)
    assert np.all(exozodi_flux.value > 0)
    assert np.isclose(exozodi_flux.value, 7.149e-09)


def test_astrophysical_scene_initialization():
    scene = AstrophysicalScene()
    assert hasattr(scene, "F0V")
    assert np.isclose(scene.F0V.value, 10374.996)
    assert scene.F0V.unit == PHOTON_FLUX_DENSITY


def test_calc_zodi_flux():
    # Basic setup
    dec = 45 * DEG
    ra = 180 * DEG
    lambd = np.array([0.5, 1]) * WAVELENGTH
    F0 = np.array([13476, 3.82e03]) * PHOTON_FLUX_DENSITY

    # Test basic functionality
    zodi_flux = calc_zodi_flux(dec, ra, lambd, F0)
    assert isinstance(zodi_flux, u.Quantity)
    assert zodi_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(zodi_flux) == len(lambd)

    # Test single wavelength (one-dimensional array)
    single_flux = calc_zodi_flux(dec, ra, u.Quantity([lambd[0]]), u.Quantity([F0[0]]))
    assert isinstance(single_flux, u.Quantity)
    assert single_flux.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert len(single_flux) == 1

    # Test units
    flux_dimensionless = zodi_flux.to(
        DIMENSIONLESS, equivalencies=u.dimensionless_angles()
    )
    assert isinstance(flux_dimensionless, u.Quantity)
    assert flux_dimensionless.unit == DIMENSIONLESS

    # Test starshade mode
    # with pytest.raises(ValueError):
    #     calc_zodi_flux(dec, ra, lambd, F0, starshade=True)  # Missing ss_elongation

    # with pytest.raises(ValueError):
    #     calc_zodi_flux(
    #         dec, ra, lambd, F0, starshade=False, ss_elongation=45 * DEG
    #     )  # Inconsistent starshade mode

    # Test: real values    print(ra, dec, lambd, F0, flux_zodi)

    dec = 79.5648101633 * DEG
    ra = 101.5589542028 * DEG
    lambd = np.array([0.5]) * WAVELENGTH
    F0 = np.array([13400]) * PHOTON_FLUX_DENSITY

    zodi_flux = calc_zodi_flux(dec, ra, lambd, F0)
    assert np.all(zodi_flux.value > 0)
    assert np.isclose(zodi_flux.value, 3.52136205e-10, rtol=1e-4)


def test_astrophysical_scene_load_configuration(capsys):
    scene = AstrophysicalScene()

    # Test with magnitude inputs
    parameters = {
        "wavelength": [0.5, 0.55],
        "Lstar": 1.0,
        "distance": 10,
        "magV": 5.0,
        "mag": [5.1, 5.2],
        "stellar_angular_diameter": 0.001,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }
    scene.load_configuration(parameters)

    assert scene.Lstar == 1.0 * LUMINOSITY
    assert scene.dist == 10 * DISTANCE
    assert scene.vmag == 5.0 * MAGNITUDE
    assert scene.stellar_angular_diameter_arcsec == 0.001 * ARCSEC
    assert scene.nzodis == 3.0 * ZODI
    assert scene.ra == 180.0 * DEG
    assert scene.dec == 0.0 * DEG
    assert scene.separation == 0.1 * ARCSEC
    assert scene.deltamag == 20.0 * MAGNITUDE
    assert scene.min_deltamag == 25.0 * MAGNITUDE
    assert isinstance(scene.mag, u.Quantity)
    assert len(scene.mag) == 2
    assert scene.mag.unit == MAGNITUDE
    assert np.allclose(scene.mag.value, [5.1, 5.2])

    # Test Fs_over_F0 calculation
    assert np.allclose(scene.Fs_over_F0.value, 10 ** (-0.4 * np.array([5.1, 5.2])))

    # Test case where F0 was provided
    parameters["F0"] = 13400
    scene.load_configuration(parameters)
    assert scene.F0 == parameters["F0"] * PHOTON_FLUX_DENSITY

    # Test with flux inputs
    flux_parameters = {
        "wavelength": [0.5, 0.55],
        "Lstar": 0.86,
        "distance": 14.8,
        "Fstar_10pc": [1.128e02, 1.13e02],
        "FstarV_10pc": 1.244e02,
        "Fp/Fs": [6.3e-8, 6.4e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_angular_diameter": 0.01,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
    }
    scene.load_configuration(flux_parameters)

    assert scene.Lstar == flux_parameters["Lstar"] * LUMINOSITY
    assert scene.dist == flux_parameters["distance"] * DISTANCE
    assert np.all(
        scene.Fp_over_Fs == np.array(flux_parameters["Fp/Fs"]) * DIMENSIONLESS
    )
    assert scene.Fp_min_over_Fs == flux_parameters["Fp_min/Fs"] * DIMENSIONLESS
    assert isinstance(scene.Fs_over_F0, u.Quantity)
    assert len(scene.Fs_over_F0) == 2
    assert scene.Fs_over_F0.unit == DIMENSIONLESS
    assert (
        scene.stellar_angular_diameter_arcsec
        == flux_parameters["stellar_angular_diameter"] * ARCSEC
    )
    assert scene.nzodis == flux_parameters["nzodis"] * ZODI
    assert scene.ra == flux_parameters["ra"] * DEG
    assert scene.dec == flux_parameters["dec"] * DEG
    assert scene.separation == flux_parameters["separation"] * ARCSEC

    assert np.allclose(
        scene.Fs_over_F0.value,
        np.array(flux_parameters["Fstar_10pc"])
        * (10 * DISTANCE / scene.dist) ** 2
        / scene.F0.value,
    )
    assert np.all(scene.F0.unit == PHOTON_FLUX_DENSITY)
    assert len(scene.F0) == len(flux_parameters["wavelength"])

    # Check calculated magnitudes
    calculated_vmag = -2.5 * np.log10(
        flux_parameters["FstarV_10pc"]
        * (10 * DISTANCE / scene.dist) ** 2
        / scene.F0V.value
    )
    assert np.isclose(scene.vmag.value, calculated_vmag, rtol=1e-6)

    calculated_mag = -2.5 * np.log10(
        np.array(flux_parameters["Fstar_10pc"])
        * (10 * DISTANCE / scene.dist) ** 2
        / scene.F0.value
    )
    assert np.allclose(scene.mag.value, calculated_mag, rtol=1e-6)

    # Check calculated delta_mag and min_delta_mag
    calculated_delta_mag = -2.5 * np.log10(np.array(flux_parameters["Fp/Fs"]))
    assert np.allclose(scene.deltamag.value, calculated_delta_mag, rtol=1e-6)

    calculated_min_delta_mag = -2.5 * np.log10(flux_parameters["Fp_min/Fs"])
    assert np.isclose(scene.min_deltamag.value, calculated_min_delta_mag, rtol=1e-6)

    # Test error handling for insufficient parameters
    with pytest.raises(KeyError):
        scene.load_configuration({"Lstar": 1.0})  # Missing required parameters

    # Test error handling for mixed magnitude and flux inputs
    mixed_parameters = {
        "wavelength": [0.5, 0.55],
        "Lstar": 1.0,
        "distance": 10,
        "magV": 5.0,
        "Fstar_10pc": [1.128e02, 1.13e02],
        "stellar_angular_diameter": 0.001,
        "nzodis": 3.0,
        "ra": 180.0,
        "dec": 0.0,
        "separation": 0.1,
        "delta_mag": 20.0,
        "delta_mag_min": 25,
    }

    with pytest.raises(ValueError):
        scene.load_configuration(mixed_parameters)

    # Test with single wavelength
    single_wavelength_params = parameters.copy()
    single_wavelength_params["wavelength"] = [0.5]
    single_wavelength_params["mag"] = [5.1]
    scene.load_configuration(single_wavelength_params)
    assert len(scene.mag) == 1
    assert len(scene.Fs_over_F0) == 1

    # FstarV_10pc is missing: if IFS mode, it can be calculated
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "Lstar": 0.86,
        "distance": 10.0,
        "Fstar_10pc": [1.128e02, 1.244e02, 1.13e02],
        "Fp/Fs": [6.3e-8, 6.4e-8, 6.5e-8],
        "Fp_min/Fs": 1e-10,
        "stellar_angular_diameter": 0.01,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "observing_mode": "IFS",
    }

    # Capture warnings
    scene.load_configuration(parameters)
    captured = capsys.readouterr()
    assert (
        "WARNING: `FstarV_10pc` not specified in parameters. Calculating internally..."
        in captured.out
    )

    # The interpolated value at 0.55 um should be close to 1.244e02
    expected_fstarv = 1.244e02 * PHOTON_FLUX_DENSITY
    calculated_fstarv = scene.Fs_over_F0[1] * scene.F0[1]  # At 0.55 um
    assert np.isclose(calculated_fstarv, expected_fstarv, rtol=1e-6)

    # FstarV_10pc is missing: in IMAGER mode, just fail
    parameters = {
        "wavelength": 0.5,
        "Lstar": 0.86,
        "distance": 10.0,
        "Fstar_10pc": 1.128e02,
        "Fp/Fs": 6.3e-8,
        "Fp_min/Fs": 1e-10,
        "stellar_angular_diameter": 0.01,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "observing_mode": "IMAGER",
    }

    # Capture warnings
    with pytest.raises(ValueError, match="FstarV_10pc missing in parameters."):
        scene.load_configuration(parameters)


def test_calculate_zodi_exozodi():
    # Create an AstrophysicalScene instance
    scene = AstrophysicalScene()

    # Set up parameters
    parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "Lstar": 0.86,
        "distance": 14.8,
        "magV": 5.84,
        "mag": [5.687, 5.632, 5.577],
        "stellar_angular_diameter": 0.01,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    # Load configuration
    scene.load_configuration(parameters)

    # Create a mock Observation object
    class MockObservation:
        wavelength = np.array(parameters["wavelength"]) * WAVELENGTH
        nlambd = len(wavelength)

    observation = MockObservation()

    # Calculate zodi and exozodi
    scene.calculate_zodi_exozodi(parameters)

    # Check that Fzodi_list, Fexozodi_list, and Fbinary_list are created
    assert hasattr(scene, "Fzodi_list")
    assert hasattr(scene, "Fexozodi_list")
    assert hasattr(scene, "Fbinary_list")

    # Check that the lists have the correct length
    assert len(scene.Fzodi_list) == observation.nlambd
    assert len(scene.Fexozodi_list) == observation.nlambd
    assert len(scene.Fbinary_list) == observation.nlambd

    # Check that the values are non-negative
    assert np.all(scene.Fzodi_list.value >= 0)
    assert np.all(scene.Fexozodi_list.value >= 0)
    assert np.all(
        scene.Fbinary_list.value == 0
    )  # Binary flux should be zero as per the implementation

    # Check units
    assert scene.Fzodi_list.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert scene.Fexozodi_list.unit.is_equivalent(INV_SQUARE_ARCSEC)
    assert scene.Fbinary_list.unit == DIMENSIONLESS

    # Check that M_V is calculated correctly
    expected_M_V = (
        scene.vmag - 5 * np.log10(scene.dist.value) * MAGNITUDE + 5 * MAGNITUDE
    )
    assert np.isclose(scene.M_V.value, expected_M_V.value)

    # Test with single wavelength
    single_wavelength_params = parameters.copy()
    single_wavelength_params["wavelength"] = [0.5]
    single_wavelength_params["mag"] = [5.1]

    scene.calculate_zodi_exozodi(single_wavelength_params)
    assert len(scene.Fzodi_list) == 1
    assert len(scene.Fexozodi_list) == 1
    assert len(scene.Fbinary_list) == 1

    # Test error handling
    with pytest.raises(KeyError):
        scene.calculate_zodi_exozodi({})


def test_validate_configuration():
    scene = AstrophysicalScene()

    # Set up valid parameters
    valid_parameters = {
        "wavelength": [0.5, 0.55, 0.6],
        "Lstar": 0.86,
        "distance": 14.8,
        "magV": 5.84,
        "mag": [5.687, 5.632, 5.577],
        "stellar_angular_diameter": 0.01,
        "nzodis": 3.0,
        "ra": 236.0075773682300,
        "dec": 02.5151668316500,
        "separation": 0.1,
        "delta_mag": 25.5,
        "delta_mag_min": 25.0,
    }

    # Load valid configuration
    scene.load_configuration(valid_parameters)

    # Mock observation for calculate_zodi_exozodi
    class MockObservation:
        wavelength = np.array(valid_parameters["wavelength"]) * WAVELENGTH
        nlambd = len(wavelength)

    scene.calculate_zodi_exozodi(valid_parameters)

    # Test valid configuration
    try:
        scene.validate_configuration()
    except Exception as e:
        pytest.fail(f"Valid configuration raised an unexpected exception: {e}")

    # Test missing attributes
    for attr in [
        "Lstar",
        "dist",
        "stellar_angular_diameter_arcsec",
        "nzodis",
        "ra",
        "dec",
        "separation",
        "F0V",
        "F0",
        "Fzodi_list",
        "Fexozodi_list",
        "Fbinary_list",
        "Fp_over_Fs",
        "Fs_over_F0",
    ]:
        temp = getattr(scene, attr)
        delattr(scene, attr)
        with pytest.raises(
            AttributeError, match=f"AstrophysicalScene is missing attribute: {attr}"
        ):
            scene.validate_configuration()
        setattr(scene, attr, temp)

    # # Test incorrect types
    incorrect_type_tests = [
        ("Lstar", 1),
        ("dist", 10),
        ("stellar_angular_diameter_arcsec", 0.01),
        ("nzodis", 3),
        ("ra", 236),
        ("dec", 2),
        ("separation", 0.1),
        ("F0V", 1e8),
        ("F0", [1e8, 1e8, 1e8]),
        ("Fzodi_list", [1e-7, 1e-7, 1e-7]),
        ("Fexozodi_list", [1e-8, 1e-8, 1e-8]),
        ("Fbinary_list", [0, 0, 0]),
        ("Fp_over_Fs", [1e-5, 1e-5, 1e-5]),
        ("Fs_over_F0", [1, 1, 1]),
    ]

    for attr, incorrect_value in incorrect_type_tests:
        temp = getattr(scene, attr)
        setattr(scene, attr, incorrect_value)
        with pytest.raises(
            TypeError, match=f"AstrophysicalScene attribute {attr} should be a Quantity"
        ):
            scene.validate_configuration()
        setattr(scene, attr, temp)

    # Test incorrect units
    incorrect_unit_tests = [
        ("Lstar", 1 * u.kg),
        ("dist", 10 * u.km),
        ("stellar_angular_diameter_arcsec", 0.01 * u.rad),
        ("nzodis", 3 * u.m),
        ("ra", 236 * u.rad),
        ("dec", 2 * u.rad),
        ("separation", 0.1 * u.m),
        ("F0V", 1e8 * u.W / (u.m**2)),
        ("F0", [1e8, 1e8, 1e8] * u.W / (u.m**2)),
        ("Fzodi_list", [1e-7, 1e-7, 1e-7] * u.W / (u.m**2)),
        ("Fexozodi_list", [1e-8, 1e-8, 1e-8] * u.W / (u.m**2)),
        ("Fbinary_list", [0, 0, 0] * u.m),
        ("Fp_over_Fs", [1e-5, 1e-5, 1e-5] * u.m),
        ("Fs_over_F0", [1, 1, 1] * u.m),
    ]

    for attr, incorrect_value in incorrect_unit_tests:
        temp = getattr(scene, attr)
        setattr(scene, attr, incorrect_value)
        try:
            scene.validate_configuration()
            pytest.fail(
                f"Expected ValueError for incorrect units of {attr}, but no exception was raised"
            )
        except Exception as e:
            assert isinstance(
                e, ValueError
            ), f"Expected ValueError for {attr}, but got {type(e).__name__}"
            assert "has incorrect units" in str(
                e
            ), f"Unexpected error message for {attr}: {str(e)}"
        finally:
            setattr(scene, attr, temp)

    # Test non-numerical values
    # Test attributes with invalid types
    attributes_to_test = [
        "Lstar",
        "dist",
        "stellar_angular_diameter_arcsec",
        "nzodis",
        "ra",
        "dec",
        "separation",
        "F0V",
        "F0",
        "Fzodi_list",
        "Fexozodi_list",
        "Fbinary_list",
        "Fp_over_Fs",
        "Fs_over_F0",
    ]

    for attr in attributes_to_test:
        original_value = getattr(scene, attr)
        try:
            # Set attribute to an invalid type
            if isinstance(original_value, (u.Quantity, list, np.ndarray)):
                setattr(scene, attr, "invalid string")
            else:
                setattr(scene, attr, ["invalid", "list"])

            with pytest.raises((TypeError, ValueError)):
                scene.validate_configuration()
        finally:
            # Restore the original value
            setattr(scene, attr, original_value)

    # Test missing attributes
    for attr in attributes_to_test:
        if hasattr(scene, attr):
            original_value = getattr(scene, attr)
            delattr(scene, attr)
            with pytest.raises(AttributeError):
                scene.validate_configuration()
            setattr(scene, attr, original_value)
