import pytest
import numpy as np
from astropy import units as u
from pathlib import Path
import tempfile
import os
import pytest
from pyEDITH.parse_input import *
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
        observing_mode = IMAGER
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
        observing_mode = IMAGER
        """
        )
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


def test_parse_input_file(sample_input_file, sample_input_file_error):
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

    with pytest.raises(
        KeyError,
        match="In IMAGER mode you can only use one wavelength at a time. If you are simulating photometry, please run every single wavelength separately. If you want to model a spectrum, please use IFS mode.",
    ):
        variables, secondary_variables = parse_input_file(
            sample_input_file_error, secondary_flag=False
        )

    with pytest.raises(
        KeyError,
        match="Secondary flag is True but no secondary variables found in the input file.",
    ):
        variables, secondary_variables = parse_input_file(
            sample_input_file_error, secondary_flag=True
        )

    # Test for good IFS mode
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".edith"
    ) as ifs_good:
        ifs_good.write(
            """
        observing_mode = 'IFS'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8,1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """
        )
        ifs_good.flush()
        variables, secondary_variables = parse_input_file(
            ifs_good.name, secondary_flag=False
        )
        assert np.all(variables["wavelength"] == [0.5, 0.6, 0.7])
        assert np.all(variables["Fstar_10pc"] == [1e-8, 1e-8, 1e-8])
        assert np.all(variables["Fp/Fs"] == [1e-10, 1e-10, 1e-10])

        assert variables["nlambda"] == 3
    os.unlink(ifs_good.name)

    # Test for IFS with missing keys
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".edith"
    ) as ifs_missing:
        ifs_missing.write(
            """
        observing_mode = 'IFS'
        """
        )
        ifs_missing.flush()
        with pytest.raises(
            ValueError,
            match="Required parameters 'wavelength', 'Fstar_10pc', and 'Fp/Fs' are not provided. Please write them explicitly or provide a spectrum_file path.",
        ):
            parse_input_file(ifs_missing.name, secondary_flag=False)

    os.unlink(ifs_missing.name)

    # Test for IFS mode with mismatched column lengths
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".edith"
    ) as ifs_mismatched:
        ifs_mismatched.write(
            """
        observing_mode = 'IFS'
        wavelength = [0.5, 0.6, 0.7]
        Fstar_10pc = [1e-8, 1e-8]
        Fp/Fs = [1e-10, 1e-10, 1e-10]
        """
        )
        ifs_mismatched.flush()
        with pytest.raises(
            ValueError,
            match="All of wavelength, Fstar_10pc, Fp/Fs must have the same length",
        ):
            parse_input_file(ifs_mismatched.name, secondary_flag=False)
    os.unlink(ifs_mismatched.name)

    # Test for invalid spectrum file
    with pytest.raises(
        FileNotFoundError, match="Spectrum file not found: nonexistent_file.csv"
    ):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".edith"
        ) as invalid_input:
            invalid_input.write(
                """
            observing_mode = "IFS"
            spectrum_file = 'nonexistent_file.csv'
            """
            )
            invalid_input.flush()
            parse_input_file(invalid_input.name, secondary_flag=False)
        os.unlink(invalid_input.name)

    # Test for spectrum file with incorrect number of columns
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as invalid_columns:
        invalid_columns.write("wavelength,Fstar_10pc\n0.5,1e-8")
        invalid_columns.flush()
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".edith"
        ) as invalid_columns_input:
            invalid_columns_input.write(
                f"""
            observing_mode = 'IFS'
            spectrum_file = '{invalid_columns.name}'
            """
            )
            invalid_columns_input.flush()
            with pytest.raises(
                ValueError, match="Spectrum file must contain exactly 3 columns"
            ):
                parse_input_file(invalid_columns_input.name, secondary_flag=False)
        os.unlink(invalid_columns_input.name)
    os.unlink(invalid_columns.name)

    # Test for spectrum file with non-numeric values
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as non_numeric:
        non_numeric.write("wavelength,Fstar_10pc,Fp/Fs\n0.5,1e-8,invalid")
        non_numeric.flush()
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".edith"
        ) as non_numeric_input:
            non_numeric_input.write(
                f"""
            observing_mode = 'IFS'
            spectrum_file = '{non_numeric.name}'
            """
            )
            non_numeric_input.flush()
            with pytest.raises(
                ValueError, match="Column 'Fp/Fs' contains non-numeric values"
            ):
                parse_input_file(non_numeric_input.name, secondary_flag=False)
        os.unlink(non_numeric_input.name)
    os.unlink(non_numeric.name)


def test_parse_parameters(capsys):
    # Test basic functionality with multiple parameters
    parameters = {
        "wavelength": [0.5, 0.6, 0.7],
        "Lstar": 1.0,
        "distance": 10,
        "magV": 5.0,
        "nzodis": 3.0,
        "observing_mode": "IFS",
        "snr": [10, 20, 30],
        "T_optical": 0.8,
        "diameter": 2.4,
        "toverhead_fixed": 300,
        "contrast": 1e-10,
        "nrolls": 3,
        "observatory_preset": "EAC1",
    }
    parsed = parse_parameters(parameters)

    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert parsed["Lstar"] == 1.0
    assert parsed["distance"] == 10
    assert parsed["magV"] == 5.0
    assert parsed["nzodis"] == 3.0
    assert parsed["observing_mode"] == "IFS"
    assert parsed["nlambda"] == 3
    assert np.all(parsed["snr"] == np.array([10, 20, 30]))
    assert np.all(parsed["T_optical"] == np.array([0.8, 0.8, 0.8]))
    assert parsed["diameter"] == 2.4
    assert parsed["toverhead_fixed"] == 300
    assert parsed["contrast"] == 1e-10
    assert parsed["nrolls"] == 3
    assert parsed["observatory_preset"] == "EAC1"

    # Wavelength tests
    # Test wavelength as a scalar
    parsed = parse_parameters({"wavelength": 0.5})
    assert parsed["wavelength"] == np.array([0.5])
    assert isinstance(parsed["wavelength"], np.ndarray)

    # Test wavelength as a list
    parsed = parse_parameters({"wavelength": [0.5, 0.6, 0.7]})
    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]))
    assert isinstance(parsed["wavelength"], np.ndarray)

    # Test wavelength as a scalar quantity
    parsed = parse_parameters({"wavelength": 0.5 * u.um})
    assert parsed["wavelength"] == [0.5] * u.um
    assert isinstance(parsed["wavelength"], u.Quantity)

    # Test wavelength as a list quantity
    parsed = parse_parameters({"wavelength": [0.5, 0.6, 0.7] * u.um})
    assert np.all(parsed["wavelength"] == [0.5, 0.6, 0.7] * u.um)
    assert isinstance(parsed["wavelength"], u.Quantity)

    # Test Case 1: default_len > 1 but value is a pure scalar
    parse_parameters({"wavelength": [0.5, 0.6, 0.7], "snr": 10})
    captured = capsys.readouterr()
    assert (
        "WARNING: snr should be a list of length 3. pyEDITH will create one assuming the input value for all the elements of the list."
        in captured.out
    )
    parsed = parse_parameters({"wavelength": [0.5, 0.6, 0.7], "snr": 10})
    assert np.all(parsed["snr"] == np.array([10, 10, 10]))

    # Test Case 1a: default_len > 1 but value is a Quantity scalar
    parse_parameters(
        {"wavelength": [0.5, 0.6, 0.7], "snr": 10 * u.dimensionless_unscaled}
    )
    captured = capsys.readouterr()
    assert (
        "WARNING: snr should be a list of length 3. pyEDITH will create one assuming the input value for all the elements of the list."
        in captured.out
    )
    parsed = parse_parameters(
        {"wavelength": [0.5, 0.6, 0.7], "snr": 10 * u.dimensionless_unscaled}
    )
    assert np.all(parsed["snr"] == np.array([10, 10, 10]) * u.dimensionless_unscaled)

    # Test Case 2: default_len > 1 but value has a length > 1 and != default_len
    with pytest.raises(
        ValueError,
        match="snr should be a list of length 3, but it has length 2.",
    ):
        parse_parameters({"wavelength": [0.5, 0.6, 0.7], "snr": [10, 20]})

    # Test Case 3: default_len == 1, return a single element array
    parsed = parse_parameters({"wavelength": 0.5, "snr": [10, 20, 30]})
    captured = capsys.readouterr()
    assert (
        "WARNING: snr should be a list of length 1 but you assigned multiple values. pyEDITH will create a list assuming only the first input value."
        in captured.out
    )
    assert np.all(parsed["snr"] == np.array([10]))

    # Test with Quantity input
    parsed = parse_parameters(
        {
            "wavelength": [0.5, 0.6, 0.7] * u.um,
            "snr": [10, 20, 30] * u.dimensionless_unscaled,
        }
    )
    assert np.all(parsed["wavelength"] == np.array([0.5, 0.6, 0.7]) * u.um)
    assert np.all(parsed["snr"] == np.array([10, 20, 30]) * u.dimensionless_unscaled)

    # Test Case 3: default_len == 1, return a single element array with quantity input
    parsed = parse_parameters(
        {"wavelength": 0.5, "snr": [10, 20, 30] * u.dimensionless_unscaled}
    )
    captured = capsys.readouterr()
    assert (
        "WARNING: snr should be a list of length 1 but you assigned multiple values. pyEDITH will create a list assuming only the first input value."
        in captured.out
    )
    assert np.all(parsed["snr"] == u.Quantity([10], u.dimensionless_unscaled))

    # Test when wavelength is not provided but nlambda is
    parsed = parse_parameters({"snr": 10}, nlambda=3)
    assert parsed["nlambda"] == 3
    assert np.all(parsed["snr"] == np.array([10, 10, 10]))

    # Test when both wavelength and nlambda are not provided
    with pytest.raises(
        ValueError,
        match="pyEDITH does not have access to wavelength here, you should provide nlambda as an argument to this function.",
    ):
        parse_parameters({"snr": 10})

    # Test wavelength parameters
    wavelength_params = [
        "snr",
        "T_optical",
        "epswarmTrcold",
        "npix_multiplier",
        "DC",
        "RN",
        "tread",
        "CIC",
        "QE",
        "dQE",
        "IFS_eff",
        "mag",
        "Fstar_10pc",
        "Fp/Fs",
        "delta_mag",
        "F0",
        "det_npix_input",
    ]

    # Test with single wavelength
    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})
        assert np.all(parsed[param] == np.array([1.5]))
        assert isinstance(parsed[param], np.ndarray)

    # Test with multiple wavelengths
    wavelengths = [0.5, 0.6, 0.7]
    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": wavelengths, param: [1.5, 2.5, 3.5]})
        assert np.all(parsed[param] == np.array([1.5, 2.5, 3.5]))
        assert isinstance(parsed[param], np.ndarray)

    # Test with scalar input for multiple wavelengths
    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": wavelengths, param: 1.5})
        assert np.all(parsed[param] == np.array([1.5, 1.5, 1.5]))
        assert isinstance(parsed[param], np.ndarray)

    # Test with Quantity input
    for param in wavelength_params:
        parsed = parse_parameters({"wavelength": wavelengths, param: 1.5 * u.m})
        assert np.all(parsed[param] == np.array([1.5, 1.5, 1.5]) * u.m)
        assert isinstance(parsed[param], u.Quantity)

    # Test with mismatched lengths
    with pytest.raises(ValueError):
        parse_parameters({"wavelength": wavelengths, "snr": [1.5, 2.5]})

    # Test with nlambda provided instead of wavelength
    parsed = parse_parameters({"snr": 1.5}, nlambda=3)
    assert np.all(parsed["snr"] == np.array([1.5, 1.5, 1.5]))
    assert isinstance(parsed["snr"], np.ndarray)
    target_params = [
        "Lstar",
        "distance",
        "magV",
        "FstarV_10pc",
        "stellar_angular_diameter",
        "nzodis",
        "ra",
        "dec",
        "delta_mag_min",
        "Fp_min/Fs",
        "separation",
    ]
    for param in target_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})
        assert parsed[param] == 1.5
        assert isinstance(parsed[param], float)

    # Test scalar parameters
    scalar_params = [
        "photometric_aperture_radius",
        "psf_trunc_ratio",
        "diameter",
        "toverhead_fixed",
        "toverhead_multi",
        "minimum_IWA",
        "maximum_OWA",
        "contrast",
        "noisefloor_factor",
        "bandwidth",
        "Tcore",
        "TLyot",
        "temperature",
        "T_contamination",
        "CRb_multiplier",
        "t_photon_count_input",
    ]
    for param in scalar_params:
        parsed = parse_parameters({"wavelength": 0.5, param: 1.5})
        assert parsed[param] == 1.5
        assert isinstance(parsed[param], float)

    # Test integer parameters
    parsed = parse_parameters({"wavelength": 0.5, "nrolls": 3})
    assert parsed["nrolls"] == 3
    assert isinstance(parsed["nrolls"], int)

    # Test observatory specs
    observatory_specs = [
        "observatory_preset",
        "telescope_type",
        "coronagraph_type",
        "detector_type",
        "observing_mode",
    ]
    for spec in observatory_specs:
        parsed = parse_parameters({"wavelength": 0.5, spec: "TestSpec"})
        assert parsed[spec] == "TestSpec"
        assert isinstance(parsed[spec], str)


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

    # test secondary flag false
    parsed_parameters, parsed_secondary_parameters = read_configuration(
        sample_input_file, secondary_flag=False
    )
    assert parsed_secondary_parameters == {}


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

    # Test for IFS mode with spectrum file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as spectrum_file:
        spectrum_file.write(
            "wavelength,Fstar_10pc,Fp/Fs\n0.5,1e-9,1e-11\n0.6,1e-9,1e-11\n0.7,1e-8,1e-10"
        )
        spectrum_file.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".edith"
        ) as ifs_input:
            ifs_input.write(
                f"""
            observing_mode = 'IFS'
            spectrum_file = '{spectrum_file.name}'
            """
            )
            ifs_input.flush()

            variables, _ = parse_input_file(ifs_input.name, secondary_flag=False)

            assert variables["observing_mode"] == "IFS"
            assert "wavelength" in variables
            assert "Fstar_10pc" in variables
            assert "Fp/Fs" in variables
            assert len(variables["wavelength"]) == 3

            # Check actual values
            np.testing.assert_almost_equal(variables["wavelength"], [0.5, 0.6, 0.7])
            np.testing.assert_almost_equal(variables["Fstar_10pc"], [1e-9, 1e-9, 1e-8])
            np.testing.assert_almost_equal(variables["Fp/Fs"], [1e-11, 1e-11, 1e-10])

        os.unlink(ifs_input.name)
    os.unlink(spectrum_file.name)


def test_parse_parameters_IMAGER_mode():
    parameters = {
        "observing_mode": "IMAGER",
        "wavelength": [0.5],
    }
    parsed = parse_parameters(parameters)

    assert parsed["observing_mode"] == "IMAGER"
    assert np.all(parsed["wavelength"] == np.array([0.5]))
