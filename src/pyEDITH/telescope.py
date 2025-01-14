import numpy as np


class Telescope:
    """
    A class representing a telescope for astronomical observations.

    This class provides methods to initialize and configure a telescope
    for simulating observations of exoplanets and their host stars.

    Attributes
    ----------
    D : float
        Circumscribed diameter of the telescope aperture in meters.
    Area : float
        Effective collecting area of the telescope in square meters.
    toverhead_fixed : float
        Fixed overhead time in seconds.
    toverhead_multi : float
        Multiplicative overhead time.
    throughput : numpy.ndarray
        Array of throughput values (excluding coronagraph core throughput).

    Methods
    -------
    load_configuration(parameters)
        Load configuration parameters for the simulation from a dictionary.
    """

    def __init__(self) -> None:
        """
        Initialize the Edith object with default values for output arrays.
        """
        pass

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary of
        parameters that was read from the input file.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        Returns
        -------
        None
        """

        # -------- INPUTS ---------
        # Telescope & spacecraft parameters
        self.D = parameters[
            "diameter"
        ]  # circumscribed diameter of aperture (m) # scalar
        self.Area = (
            np.single(np.pi) / 4.0 * self.D**2.0 * (1.0 - 0.121)
        )  # effective collecting area of telescope (m^2) # scalar
        self.toverhead_fixed = parameters[
            "toverhead_fixed"
        ]  # fixed overhead time (seconds)
        self.toverhead_multi = parameters[
            "toverhead_multi"
        ]  # multiplicative overhead time
        self.throughput = np.array(
            parameters["throughput"], dtype=np.float64
        )  # throughput not incl. coronagraph core throughput # nlambd array
