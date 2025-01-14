import numpy as np


class Telescope:
    """

    Methods:
    --------
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

        Parameters:
        -----------
        parameters : dict
            A dictionary containing various simulation parameters including:
            - Target star parameters (ntargs, Lstar, dist, vmag, mag,
                                      angdiam_arcsec, nzodis, ra, dec)
            - Planet parameters (sp, deltamag, min_deltamag)
            - Observational parameters (lambd, SR, SNR, throughput, photap_rad)
            - Telescope & spacecraft parameters (D, toverhead_fixed, toverhead_multi)
            - Instrument parameters (IWA, OWA, contrast, noisefloor_factor, bandwidth,
                                     core_throughput, Lyot_transmission)
            - Detector parameters (npix_multiplier, dark_current, read_noise,
                                     read_time, cic)
            - Coronagraph parameters (coro_type, nrolls)
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
