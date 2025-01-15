from pyEDITH import AstrophysicalScene, Observation, ToyModel, Edith
from pyEDITH import calculate_exposure_time, calculate_signal_to_noise, parse_input
from argparse import ArgumentParser
import numpy as np
import sys


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

    args = parser.parse_args()

    if args.subfunction == "etc":
        if not args.edith:
            print("Error: --edith argument is required for etc subfunction.")
            parser_a.print_help(sys.stderr)
            sys.exit(1)
        parameters, _ = parse_input.read_configuration(args.edith)
        texp = calculate_texp(parameters)
        print(texp)

    elif args.subfunction == "snr":
        if not args.edith or args.time is None:
            print(
                "Error: Both --edith and --time arguments are required for snr \
                    subfunction."
            )
            parser_b.print_help(sys.stderr)
            sys.exit(1)
        parameters, _ = parse_input.read_configuration(args.edith)
        texp = args.time
        snr = calculate_snr(parameters, texp)
        print(snr)

    elif args.subfunction == "etc2snr":
        if not args.edith:
            print("Error: --edith argument is required for etc2snr subfunction.")
            parser_c.print_help(sys.stderr)
            sys.exit(1)
        parameters, secondary_parameters = parse_input.read_configuration(
            args.edith, secondary_flag=True
        )
        if not secondary_parameters:
            raise ValueError("The secondary parameters are not specified.")

        if len(parameters["lambd"]) > 1:
            raise TypeError("Cannot accept multiple lambdas as primary lambda")
        else:
            for key in parameters:
                if key not in secondary_parameters:
                    secondary_parameters[key] = parameters[key]

        print("Calculating texp from primary lambda")
        texp = calculate_texp(parameters)
        print("Reference exposure time: ", texp)
        if np.isfinite(texp).all():
            pass
            print("Calculating snr on secondary lambda")
            snr = calculate_snr(secondary_parameters, texp)
            print("SNR at the secondary lambda: ", snr)
        else:
            raise ValueError("Returned exposure time is infinity.")
    else:
        parser.print_help()


# def input(args):


#     ### CHECK COMMAND LINE INPUTS
#     # check input file was provided

#     if not Path(args.ayo).exists():
#         raise ValueError('Cannot find this .ayo file. Please check that the \
#                           path/name is correct.')

#     if not Path(args.coro).exists():
#         raise ValueError('Cannot find this .coro file. Please check that the \
#                            path/name is correct.')

#     if not Path(args.targets).exists():
#         raise ValueError('Cannot find this .csv file. Please check that the \
#                            path/name is correct.')

#     ### CONVERT COMMAND LINE INPUTS INTO SINGLE DICT
#     #----- FILE READING -----
#     # with open(self.input_file,'r') as f:
#     #     parameters=yaml.safe_load(f)
#     parameters=parse_input.parse_input_file(args.input_file)
#     return parameters


def calculate_texp(parameters: dict) -> np.array:
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

    # Define Astrophysical Scene and load relevant parameters,
    # then calculate zodi/exozodi
    scene = AstrophysicalScene()
    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(observation)

    # Define Instrument and load relevant parameters
    if parameters["coro_type"] == "toymodel":
        instrument = ToyModel()
        instrument.initialize(parameters)

        # Generate secondary parameters specific to the ToyModel subclass
        instrument.coronagraph.generate_secondary_parameters(observation)
        """TODO implement something like this 
        
        elif parameters["coro_type"] =="EAC1":

            instrument = EAC1()
            instrument.initialize(parameters) #replace EAC1 default parameters if you want
            instrument.coronagraph.generate_secondary_parameters(observation)
        """

    else:
        raise KeyError("The coro_type keyword is not valid.")

    # Define Edith object and load default parameters
    edith = Edith(scene, observation)
    edith.load_default_parameters()

    # EXPOSURE TIME CALCULATION
    calculate_exposure_time(observation, scene, instrument, edith)

    return edith.exptime


def calculate_snr(parameters, reference_texp):
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

    # Define Astrophysical Scene and load relevant parameters,
    # then calculate zodi/exozodi
    scene = AstrophysicalScene()
    scene.load_configuration(parameters)
    scene.calculate_zodi_exozodi(observation)

    # Define Instrument and load relevant parameters
    if parameters["coro_type"] == "toymodel":
        instrument = ToyModel()
        instrument.initialize(parameters)

        # Generate secondary parameters specific to the ToyModel subclass
        instrument.coronagraph.generate_secondary_parameters(observation)
        """TODO implement something like this 
        
        elif parameters["coro_type"] =="EAC1":

            instrument = EAC1()
            instrument.initialize(parameters) #replace EAC1 default parameters if you want
            instrument.coronagraph.generate_secondary_parameters(observation)
        """

    else:
        raise KeyError("The coro_type keyword is not valid.")

    # Define Edith object and load default parameters
    edith = Edith(scene, observation)
    edith.load_default_parameters()

    # SNR CALCULATION
    edith.obstime = reference_texp
    calculate_signal_to_noise(observation, scene, instrument, edith)
    # print(istar, coronagraph.type,  edith.exptime[istar][ilambd])

    return edith.fullsnr
