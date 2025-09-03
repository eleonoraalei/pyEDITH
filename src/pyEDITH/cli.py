from pyEDITH import AstrophysicalScene, Observation, ObservatoryBuilder
from pyEDITH import calculate_exposure_time_or_snr, parse_input
from argparse import ArgumentParser
import numpy as np
import astropy.units as u
import sys

import os

# Declare the environment variable
# os.environ["SCI_ENG_DIR"] = "/path/to/Sci-Eng-Interface/hwo_sci_eng"


def main():
    """
    Main entry point for the py-E.D.I.T.H. command-line interface.

    This function sets up the argument parser and handles the execution
    of different subcommands: etc (Exposure Time Calculator), snr (Signal-to-Noise
    Ratio), and etc2snr (Exposure Time to Signal-to-Noise Ratio).

    The function reads configuration from .edith files and calls the appropriate
    calculation functions based on the subcommand.

    Raises
    ------
    SyntaxError
        If required command-line arguments are missing
    TypeError
        If multiple wavelengths are provided for primary lambda in etc2snr mode
    ValueError
        If secondary parameters are not specified in etc2snr mode or
        if the returned exposure time is infinity
    """

    parser = ArgumentParser(
        description="Available command line arguments for E.D.I.T.H."
    )
    subparsers = parser.add_subparsers(
        dest="subfunction", help="Subfunction to execute"
    )

    parser_a = subparsers.add_parser(
        "etc", help="Exposure Time Calculator for a specific lambda (in .edith file)"
    )
    parser_a.add_argument("--edith", type=str, help="an .edith file")

    parser_b = subparsers.add_parser(
        "snr",
        help="SNR calculator for a specific lambda (in .edith file) and for a given \
            exposure time (argument, in hours)",
    )
    parser_b.add_argument("--edith", type=str, help="an .edith file")
    parser_b.add_argument(
        "--time", type=float, help="the desired observing time in minutes"
    )

    parser_c = subparsers.add_parser(
        "etc2snr",
        help="SNR calculator for a secondary lambda based on an exposure file \
            calculated on a primary lambda (in .edith file)",
    )
    parser_c.add_argument("--edith", type=str, help="an .edith file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    args = parser.parse_args()

    if args.subfunction == "etc":
        if not args.edith:
            raise SyntaxError("--edith argument is required for etc subfunction.")

        parameters, _ = parse_input.read_configuration(args.edith)
        texp, _ = calculate_texp(parameters, args.verbose)
        print(texp)

    elif args.subfunction == "snr":
        if not args.edith or args.time is None:
            raise SyntaxError(
                "Both --edith and --time arguments are required for snr subfunction."
            )

        parameters, _ = parse_input.read_configuration(args.edith)
        texp = args.time
        snr, _ = calculate_snr(parameters, texp, args.verbose)
        print(snr)

    elif args.subfunction == "etc2snr":
        if not args.edith:
            raise SyntaxError("--edith argument is required for etc2snr subfunction.")

        parameters, secondary_parameters = parse_input.read_configuration(
            args.edith, secondary_flag=True
        )

        if not secondary_parameters:
            raise ValueError("The secondary parameters are not specified.")

        if len(parameters["wavelength"]) > 1:
            raise TypeError("Cannot accept multiple lambdas as primary lambda")
        else:
            for key in parameters:
                if key not in secondary_parameters:
                    secondary_parameters[key] = parameters[key]

        print("Calculating texp from primary lambda")
        print(parameters.keys())
        texp, _ = calculate_texp(parameters, args.verbose)
        print("Reference exposure time: ", texp)
        if np.isfinite(texp).all():
            print("Calculating snr on secondary lambda")
            snr, _ = calculate_snr(secondary_parameters, texp, args.verbose)
            print("SNR at the secondary lambda: ", snr)
        else:
            raise ValueError("Returned exposure time is infinity.")
    else:
        parser.print_help()


def calculate_texp(
    parameters: dict, verbose: bool, ETC_validation: bool = False
) -> np.array:
    """
    Calculate exposure time for a planet observation with specified parameters.

    This function initializes Observation, AstrophysicalScene, and Observatory
    objects with the provided parameters, then calculates the required exposure
    time to achieve the specified signal-to-noise ratio at each wavelength.

    Parameters
    ----------
    parameters : dict
        Dictionary containing all input parameters for the calculation
    verbose : bool
        If True, print detailed calculation information
    ETC_validation : bool, optional
        If True, use specific parameter values for validation against the ETC,
        default is False

    Returns
    -------
    tuple
        A tuple containing:

        observation.exptime : numpy.ndarray
            Exposure time in hours for each wavelength

        observation.validation_variables : dict
            Validation variables containing intermediate calculation results

    """

    # Define Observation and load relevant parameters
    observation = Observation()
    observation.load_configuration(parameters)
    observation.set_output_arrays()
    observation.validate_configuration()

    # Define Astrophysical Scene and load relevant parameters,
    # then calculate zodi/exozodi
    scene = AstrophysicalScene()
    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)
    scene.validate_configuration()
    if (
        parameters["observing_mode"] == "IFS"
        and parameters["regrid_wavelength"] is True
    ):
        scene.regrid_spectra(parameters, observation)

    # Create and configure Observatory using ObservatoryBuilder
    observatory_config = parse_input.get_observatory_config(parameters)
    observatory = ObservatoryBuilder.create_observatory(observatory_config)
    ObservatoryBuilder.configure_observatory(
        observatory, parameters, observation, scene
    )
    observatory.validate_configuration()

    # EXPOSURE TIME CALCULATION
    calculate_exposure_time_or_snr(
        observation,
        scene,
        observatory,
        verbose,
        ETC_validation=ETC_validation,
        mode="exposure_time",
    )

    return observation.exptime, observation.validation_variables


def calculate_snr(parameters: dict, reference_texp: float, verbose: bool):
    """
    Calculate signal-to-noise ratio for a given exposure time.

    This function initializes Observation, AstrophysicalScene, and Observatory
    objects with the provided parameters, then calculates the achievable
    signal-to-noise ratio for the specified exposure time at each wavelength.

    Parameters
    ----------
    parameters : dict
        Dictionary containing all input parameters for the calculation
    reference_texp : float
        Reference exposure time in hours
    verbose : bool
        If True, print detailed calculation information

    Returns
    -------
    tuple
        A tuple containing:

        observation.fullsnr : numpy.ndarray
            Signal-to-noise ratio for each wavelength

        observation.validation_variables : dict
            Validation variables containing intermediate calculation results

    """

    # Define Observation and load relevant parameters
    observation = Observation()
    observation.load_configuration(parameters)
    observation.set_output_arrays()

    # Define Astrophysical Scene and load relevant parameters,
    # then calculate zodi/exozodi
    scene = AstrophysicalScene()
    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(parameters)
    if (
        parameters["observing_mode"] == "IFS"
        and parameters["regrid_wavelength"] is True
    ):
        scene.regrid_spectra(parameters, observation)

    # Create and configure Observatory using ObservatoryBuilder
    observatory_config = parse_input.get_observatory_config(parameters)
    observatory = ObservatoryBuilder.create_observatory(observatory_config)
    ObservatoryBuilder.configure_observatory(
        observatory, parameters, observation, scene
    )
    observatory.validate_configuration()

    # SNR CALCULATION
    observation.obstime = reference_texp
    calculate_exposure_time_or_snr(
        observation, scene, observatory, verbose, mode="signal_to_noise"
    )
    # print(istar, coronagraph.type,  edith.exptime[istar][ilambd])

    return observation.fullsnr, observation.validation_variables
