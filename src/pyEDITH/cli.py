from pyEDITH import AstrophysicalScene, Observation, ObservatoryBuilder
from pyEDITH import calculate_exposure_time_or_snr, parse_input
from argparse import ArgumentParser
import numpy as np
import sys

import os

# Declare the environment variable
# os.environ["SCI_ENG_DIR"] = "/path/to/Sci-Eng-Interface/hwo_sci_eng"


def main():
    """
    Main entry point for the E.D.I.T.H. command-line interface.

    This function sets up the argument parser and handles the execution
    of different subcommands: etc (Exposure Time Calculator), snr (Signal-to-Noise
    Ratio), and etc2snr (Exposure Time to Signal-to-Noise Ratio).

    The function reads configuration from .edith files and calls the appropriate
    calculation functions based on the subcommand.

    Raises:
    -------
    UserWarning
        If the primary and secondary parameters are the same in etc2snr mode.
    ValueError
        If the returned exposure time is infinity in etc2snr mode.
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


def calculate_texp(parameters: dict, verbose, ETC_validation=False) -> np.array:
    """
    Calculates the exposure time for a planet observed with a given coronagraph.

    This function uses the exposure_time_calculator.c routine, which has been
    benchmarked using the ESYWG/CDS/CTR Exposure Time Comparison spreadsheet.

    Parameters:
    -----------
    parameters : dict
        A dictionary containing the input parameters for the calculation.

    Returns:
    --------
    np.array
        The exposure time in hours for each star and lambda.

    Raises:
    -------
    KeyError
        If the coronagraph type specified in the parameters is not valid.

    Notes:
    ------
    For now, only the "toy" coronagraph is implemented.
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
    scene.calculate_zodi_exozodi(observation)
    scene.validate_configuration()

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


def calculate_snr(parameters, reference_texp, verbose):
    """
    Calculates the signal-to-noise ratio (SNR) for a given exposure time.

    Parameters:
    -----------
    parameters : dict
        A dictionary containing the input parameters for the calculation.
    reference_texp : float
        The reference exposure time in hours.

    Returns:
    --------
    None
        Prints the calculated SNR.

    Raises:
    -------
    KeyError
        If the coronagraph type specified in the parameters is not valid.
    ValueError
        If the calculation fails to converge within the maximum number of iterations
        or if the calculated exposure time is infinity.
    """

    # Define Observation and load relevant parameters
    observation = Observation()
    observation.load_configuration(parameters)
    observation.set_output_arrays()

    # Define Astrophysical Scene and load relevant parameters,
    # then calculate zodi/exozodi
    scene = AstrophysicalScene()
    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(observation)

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
