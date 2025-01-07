import numpy as np

class Detector():
    """
    Methods:
    --------
    """
    def __init__(self) -> None:
        """
        """

    def load_configuration(self,parameters:dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary of parameters that was read from the input file.

        Parameters:
        -----------
        parameters : dict
            A dictionary containing various simulation parameters including:
            - Target star parameters (ntargs, Lstar, dist, vmag, mag, angdiam_arcsec, nzodis, ra, dec)
            - Planet parameters (sp, deltamag, min_deltamag)
            - Observational parameters (lambd, SR, SNR, throughput, photap_rad)
            - Telescope & spacecraft parameters (D, toverhead_fixed, toverhead_multi)
            - Instrument parameters (IWA, OWA, contrast, noisefloor_factor, bandwidth, core_throughput, Lyot_transmission)
            - Detector parameters (npix_multiplier, dark_current, read_noise, read_time, cic)
            - Coronagraph parameters (coro_type, nrolls)
        """

        #-------- INPUTS ---------

        # Detector parameters
        self.det_pixscale_mas = 0.5*(0.5e-6/parameters['diameter']) * (180./np.double(np.pi)*60.*60.*1000.) # detector pixel scale (mas) # scalar
        self.det_npix_multiplier = np.array(parameters['npix_multiplier'], dtype=np.float64) # # of detector pixels per image plane "pixel"# nlambd array, 1 for detections or spectra w/ ERD, ~6*(140/SR) for spectra with IFS
        self.det_DC = np.array(parameters['dark_current'],dtype=np.float64) # dark current (counts pix^-1 s^-1) # nlambd array
        self.det_RN = np.array(parameters['read_noise'],dtype=np.float64) # read noise (counts pix^-1 read^-1) # nlambd array
        self.det_tread = np.array(parameters['read_time'],dtype=np.float64) # read time (s) # nlambd array
        self.det_CIC = np.array(parameters['cic'],dtype=np.float64) # clock induced charge (counts pix^-1 photon_count^-1) # nlambd array


