from abc import ABC, abstractmethod
import numpy as np
from .. import parse_input
import astropy.units as u
from ..units import *


class Detector(ABC):
    """
    A class representing a detector for astronomical observations.

    This class manages detector-specific parameters and configurations
    used in astronomical simulations and observations.

    Attributes
    ----------
    pixscale_mas : float
        Detector pixel scale in milliarcseconds.
    npix_multiplier : ndarray
        Number of detector pixels per image plane "pixel".
    DC : ndarray
        Dark current in counts per pixel per second.
    RN : ndarray
        Read noise in counts per pixel per read.
    tread : ndarray
        Read time in seconds.
    CIC : ndarray
        Clock-induced charge in counts per pixel per photon count.
    dQE: ndarray
        Quantum efficiency of detector
    QE: ndarray
        Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    """

    @abstractmethod
    def load_configuration(self):
        pass

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "pixscale_mas": MAS,
            "npix_multiplier": DIMENSIONLESS,
            "DC": DARK_CURRENT,
            "RN": READ_NOISE,
            "tread": READ_TIME,
            "CIC": CIC,
            "QE": QE,
            "dQE": DIMENSIONLESS,
        }

        for arg, expected_unit in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"Detector is missing attribute: {arg}")
            value = getattr(self, arg)
            if not isinstance(value, u.Quantity):
                raise TypeError(f"Detector attribute {arg} should be a Quantity")
            if not value.unit.is_equivalent(expected_unit):
                raise ValueError(
                    f"Detector attribute {arg} has incorrect units. Expected {expected_unit}, got {value.unit}"
                )


class ToyModelDetector(Detector):
    """
    A toy model detector class that extends the base Detector class.

    This class represents a simplified detector model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "pixscale_mas": None,  # Detector pixel scale in milliarcseconds.
        "npix_multiplier": [1]
        * DIMENSIONLESS,  # Number of detector pixels per image plane "pixel".
        "DC": [3e-5] * DARK_CURRENT,  # Dark current (counts pix^-1 s^-1, nlambd array)
        "RN": [0.0] * READ_NOISE,  # Read noise (counts pix^-1 read^-1, nlambd array)
        "tread": [1000] * READ_TIME,  # Read time (s, nlambd array)
        "CIC": [1.3e-3]
        * CIC,  # Clock-induced charge (counts pix^-1 photon_count^-1, nlambd array)
        "QE": [0.9] * QE,  # Quantum efficiency of detector
        "dQE": [0.75]
        * DIMENSIONLESS,  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    }

    def __init__(self, path=None, keyword=None):
        self.path = path
        self.keyword = keyword

    def load_configuration(self, parameters, mediator) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the Detector object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        mediator: ObservatoryMediator
        Returns
        -------
        None

        """

        # Calculate default detector pixel scale based on telescope
        self.DEFAULT_CONFIG["pixscale_mas"] = (
            0.5
            * lambda_d_to_arcsec(
                1 * LAMBDA_D,
                0.5e-6 * LENGTH,
                mediator.get_telescope_parameter("diameter").to(LENGTH),
            )
        ).to(MAS)

        # Load parameters, use defaults if not provided
        for key, default_value in self.DEFAULT_CONFIG.items():
            if key in parameters:
                # User provided a value
                user_value = parameters[key]
                if isinstance(default_value, u.Quantity):
                    # Ensure the user value has the same unit as the default
                    if isinstance(user_value, u.Quantity):
                        setattr(self, key, user_value.to(default_value.unit))
                    else:
                        setattr(self, key, u.Quantity(user_value, default_value.unit))
                else:
                    # For non-Quantity values (like integers), use as is
                    setattr(self, key, user_value)
            else:
                # Use default value
                setattr(self, key, default_value)

        # # Convert to numpy array when appropriate
        array_params = [
            "npix_multiplier",
            "DC",
            "RN",
            "tread",
            "CIC",
            "QE",
            "dQE",
        ]
        for param in array_params:
            attr_value = getattr(self, param)
            if isinstance(attr_value, u.Quantity):
                # If it's already a Quantity, convert to numpy array while preserving units
                setattr(
                    self,
                    param,
                    u.Quantity(
                        np.array(attr_value.value, dtype=np.float64), attr_value.unit
                    ),
                )
            else:
                # If it's not a Quantity, convert to numpy array without units
                setattr(self, param, np.array(attr_value, dtype=np.float64))


class EACDetector(Detector):
    """
    A toy model detector class that extends the base Detector class.

    This class represents a simplified detector model for use in simulations.
    """

    DEFAULT_CONFIG = {
        "pixscale_mas": None,  # Detector pixel scale in milliarcseconds.
        "npix_multiplier": [1]
        * DIMENSIONLESS,  # Number of detector pixels per image plane "pixel".
        "DC": None,  # Dark current (counts pix^-1 s^-1, nlambd array)
        "RN": None,  # Read noise (counts pix^-1 read^-1, nlambd array)
        "tread": [1000] * READ_TIME,  # Read time (s, nlambd array) # TO ADD TO YAML
        "CIC": [0]
        * CIC,  # Clock-induced charge (counts pix^-1 photon_count^-1, nlambd array) # TO ADD TO YAML
        "QE": None,  # Quantum efficiency of detector
        "dQE": None,  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies
    }

    def __init__(self, path=None, keyword=None):
        self.path = path
        self.keyword = keyword

    def load_configuration(self, parameters, mediator) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the Detector object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        mediator: ObservatoryMediator
        Returns
        -------
        None

        """
        from eacy import load_detector

        # ****** Update Default Config when necessary ******

        detector_params = load_detector(parameters["observing_mode"]).__dict__
        if parameters["observing_mode"] == "IMAGER":
            wavelength_range = [
                mediator.get_observation_parameter("wavelength")
                * (1 - 0.5 * mediator.get_coronagraph_parameter("bandwidth")),
                mediator.get_observation_parameter("wavelength")
                * (1 + 0.5 * mediator.get_coronagraph_parameter("bandwidth")),
            ]

            detector_params = parse_input.average_over_bandpass(
                detector_params, wavelength_range
            )
        elif parameters["observing_mode"] == "IFS":
            detector_params = parse_input.interpolate_over_bandpass(
                detector_params, mediator.get_observation_parameter("wavelength")
            )
        else:
            raise ValueError(
                f"Unsupported observing mode: {parameters['observing_mode']}"
            )

        dc_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)
        dc_arr[mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH] = (
            detector_params["dc_vis"]
        )
        dc_arr[mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH] = (
            detector_params["dc_nir"]
        )
        self.DEFAULT_CONFIG["DC"] = (
            dc_arr * DARK_CURRENT
        )  # Dark current (counts pix^-1 s^-1, nlambd array)

        # self.DEFAULT_CONFIG["DC"] = (
        #     np.array(
        #         [
        #             (
        #                 float(detector_params["dc_vis"])
        #                 if mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
        #                 else float(detector_params["dc_nir"])
        #             )
        #         ]
        #     )
        #     * DARK_CURRENT
        # )
        # Dark current (counts pix^-1 s^-1, nlambd array)

        rn_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)
        rn_arr[mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH] = (
            detector_params["rn_vis"]
        )
        rn_arr[mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH] = (
            detector_params["rn_nir"]
        )
        self.DEFAULT_CONFIG["RN"] = rn_arr * READ_NOISE
        # self.DEFAULT_CONFIG["RN"] = [
        #     (
        #         detector_params["rn_vis"]
        #         if mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
        #         else detector_params["rn_nir"]
        #     )
        # ] * READ_NOISE

        if parameters["observing_mode"] == "IMAGER":
            qe_arr = [
                (
                    detector_params["qe_vis"]
                    if mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
                    else detector_params["qe_nir"]
                )
            ]
        elif parameters["observing_mode"] == "IFS":
            # combine the vis and nir qe arrays into a single array.
            qe_arr = np.empty_like(
                mediator.get_observation_parameter("wavelength").value
            )
            qe_arr[
                mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
            ] = detector_params["qe_vis"][
                mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
            ]
            qe_arr[
                mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH
            ] = detector_params["qe_nir"][
                mediator.get_observation_parameter("wavelength") >= 1 * WAVELENGTH
            ]
            # make sure qe_arr does not contain NaNs
            assert ~np.isnan(np.sum(qe_arr)), "QE array contains NaN values"
        else:
            raise ValueError(
                f"Unsupported observing mode: {parameters['observing_mode']}"
            )

        self.DEFAULT_CONFIG["QE"] = qe_arr * QE
        # self.DEFAULT_CONFIG["QE"] = [
        #     (
        #         detector_params["qe_vis"]
        #         if mediator.get_observation_parameter("wavelength") < 1 * WAVELENGTH
        #         else detector_params["qe_nir"]
        #     )
        # ] * QE

        dQE_arr = np.empty_like(mediator.get_observation_parameter("wavelength").value)

        # for now, hardcoded to 0.75
        dQE_arr.fill(0.75)
        self.DEFAULT_CONFIG["dQE"] = dQE_arr * DIMENSIONLESS
        # self.DEFAULT_CONFIG["dQE"] = [
        #     0.75
        # ] * DIMENSIONLESS  # Effective QE due to degradation, cosmic ray effects, readout inefficiencies ## TO ADD TO YAML

        # Calculate default detector pixel scale based on telescope
        self.DEFAULT_CONFIG["pixscale_mas"] = (
            0.5
            * (0.5e-6 * LENGTH / mediator.get_telescope_parameter("diameter"))
            * (180.0 / np.double(np.pi) * 60.0 * 60.0 * 1000.0)
        ) * MAS

        # fill in tread and CIC to match the length of the wavelength array
        self.DEFAULT_CONFIG["tread"] = (
            np.full_like(
                mediator.get_observation_parameter("wavelength").value,
                self.DEFAULT_CONFIG["tread"][0].value,
                dtype=np.float64,
            )
            * READ_TIME
        )
        self.DEFAULT_CONFIG["CIC"] = (
            np.full_like(
                mediator.get_observation_parameter("wavelength").value,
                self.DEFAULT_CONFIG["CIC"][0].value,
                dtype=np.float64,
            )
            * CIC
        )

        # ***** Load parameters, use defaults if not provided *****
        for key, default_value in self.DEFAULT_CONFIG.items():
            if key in parameters:
                # User provided a value
                user_value = parameters[key]
                if isinstance(default_value, u.Quantity):
                    # Ensure the user value has the same unit as the default
                    # TODO Implement conversion of units from the input file
                    if isinstance(user_value, u.Quantity):
                        setattr(self, key, user_value.to(default_value.unit))
                    else:
                        setattr(self, key, u.Quantity(user_value, default_value.unit))
                else:
                    # For non-Quantity values (like integers), use as is
                    setattr(self, key, user_value)
            else:
                # Use default value
                setattr(self, key, default_value)

        # ***** Convert to numpy array when appropriate *****
        array_params = [
            "npix_multiplier",
            "DC",
            "RN",
            "tread",
            "CIC",
            "QE",
            "dQE",
        ]
        for param in array_params:
            attr_value = getattr(self, param)
            if isinstance(attr_value, u.Quantity):
                # If it's already a Quantity, convert to numpy array while preserving units
                setattr(
                    self,
                    param,
                    u.Quantity(
                        np.array(attr_value.value, dtype=np.float64), attr_value.unit
                    ),
                )
            else:
                # If it's not a Quantity, convert to numpy array without units
                setattr(self, param, np.array(attr_value, dtype=np.float64))

        # Ensure npix_multiplier has the same length as wavelength (the other ones are taken care of by EACy)
        if len(self.npix_multiplier) != len(
            mediator.get_observation_parameter("wavelength")
        ):
            self.npix_multiplier = (
                np.full_like(
                    mediator.get_observation_parameter("wavelength").value,
                    self.npix_multiplier[0],
                    dtype=np.float64,
                )
                * DIMENSIONLESS
            )

        # USED ONLY TO VALIDATE ETCs
        if "t_photon_count_input" in parameters.keys():

            self.t_photon_count_input = (
                parameters["t_photon_count_input"] * SECOND / FRAME
            )
