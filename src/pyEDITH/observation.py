import numpy as np


class Observation:
    """

    Methods:
    --------

    """

    def __init__(self) -> None:
        """
        Initialize the Edith object with default values for output arrays.
        """
        pass  # there are no default values, TODO it should fail if not provided

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
        # Observational parameters
        self.lambd = np.array(
            parameters["lambd"], dtype=np.float64
        )  # wavelength # nlambd array #unit: micron
        self.nlambd = len(self.lambd)
        self.SR = np.array(
            parameters["resolution"], dtype=np.float64
        )  # spec res # nlambd array
        self.SNR = np.array(
            parameters["snr"], dtype=np.float64
        )  # signal to noise # nlambd array
        self.photap_rad = parameters["photap_rad"]  # (lambd/D) # scalar
