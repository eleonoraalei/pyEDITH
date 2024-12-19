import numpy as np
import pyEDITH.flux_zodis_calculation as fluxzodi
import pyEDITH.parse_input as io

class Edith():
    """
    A class representing the E.D.I.T.H. (Exoplanet Detection Imaging and Throughput) simulation environment.

    This class handles the configuration, initialization, and calculations for exoplanet detection simulations.

    Methods:
    --------
    __init__()
    load_configuration(parameters)
    load_default_parameters()
    calculate_zodi_exozodi()
    """
    def __init__(self) -> None:
        """
        Initialize the Edith object with default values for output arrays.
        """

        #Initialize some arrays needed for outputs...
        self.besticoro = np.array([0]) 
        self.bestilambd = np.array([0])
        self.avgpsfomega = np.array([0.0])
        self.minpsfomega = np.array([0.0])
        self.maxpsfomega = np.array([0.0])
        self.avgpsftruncratio = np.array([0.0])
        self.minpsftruncratio = np.array([0.0])
        self.maxpsftruncratio = np.array([0.0])
        self.tp = np.array([[[0.0]]],dtype=np.float64) #exposure time of every planet (nmeananom x norbits x ntargs array), used in c function [NOTE: nmeananom = nphases in C code]


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
        # Target star parameters
        self.ntargs=parameters['ntargs']
        self.Lstar = np.array(parameters['Lstar'],dtype=np.float64) # luminosity of star (solar luminosities) # ntargs array
        self.dist = np.array(parameters['distance'],dtype=np.float64)  # distance to star (pc) # ntargs array
        self.vmag = np.array(parameters['magV'],dtype=np.float64) # stellar mag at V band # ntargs array
        self.mag = np.array(parameters['mag'],dtype=np.float64)                     # stellar mag at desired lambd # nlambd x ntargs array
        self.angdiam_arcsec = np.array(parameters['angdiam'],dtype=np.float64)          # angular diameter of star (arcsec) # ntargs array
        self.nzodis = np.array(parameters['nzodis'],dtype=np.float64) # amount of exozodi around target star ("zodis") # ntargs array
        self.ra = np.array(parameters['ra'],dtype=np.float64) # right ascension of target star used to estimate zodi (deg) # ntargs array
        self.dec = np.array(parameters['dec'],dtype=np.float64) # declination of target star used to estimate zodi (deg) # ntargs array

        # Planet parameters
        self.sp = np.array(parameters['sp'],dtype=np.float64) # separation of planet (arcseconds) # nmeananom x norbits x ntargs array
        self.deltamag = np.array(parameters['delta_mag'],dtype=np.float64) # difference in mag between planet and host star # nmeananom x norbits x ntargs array
        self.min_deltamag = np.array(parameters['delta_mag_min'],dtype=np.float64) # brightest planet to resolve w/ photon counting detector evaluated at the IWA, sets the time between counts # ntargs array

        # Observational parameters
        self.lambd = np.array(parameters['lambd'],dtype=np.float64) # wavelength # nlambd array #unit: micron
        self.nlambd = len(self.lambd)
        self.SR = np.array(parameters['resolution'],dtype=np.float64) # spec res # nlambd array
        self.SNR = np.array(parameters['snr'],dtype=np.float64) # signal to noise # nlambd array
        self.throughput = np.array(parameters['throughput'],dtype=np.float64) # throughput not incl. coronagraph core throughput # nlambd array
        self.photap_rad =parameters['photap_rad'] # (lambd/D) # scalar

        # Telescope & spacecraft parameters
        self.D = parameters['diameter'] # circumscribed diameter of aperture (m) # scalar
        self.Area = np.single(np.pi)/4.*self.D**2.*(1.0-0.121) # effective collecting area of telescope (m^2) # scalar
        self.toverhead_fixed = parameters['toverhead_fixed'] # fixed overhead time (seconds)
        self.toverhead_multi = parameters['toverhead_multi']         # multiplicative overhead time

        # Instrument parameters
        self.IWA = parameters['IWA']   # smallest WA to allow (lambd/D) # scalar
        self.OWA = parameters['OWA'] # largest WA to allow (lambd/D) # scalar
        self.contrast = parameters['contrast'] # contrast of coronagraph (uniform over dark hole and unitless) # scalar
        self.noisefloor =parameters['noisefloor_factor']*self.contrast # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar
        self.bandwidth = parameters['bandwidth'] # fractional bandwidth of coronagraph (unitless)
        self.Tcore = parameters['core_throughput']  # core throughput of coronagraph (uniform over dark hole, unitless) # scalar
        self.TLyot = parameters['Lyot_transmission'] # 1.6*Tcore # Lyot transmission of the coronagraph and the factor of 1.6 is just an estimate, used for skytrans

        # Detector parameters
        self.det_pixscale_mas = 0.5*(0.5e-6/self.D) * (180./np.double(np.pi)*60.*60.*1000.) # detector pixel scale (mas) # scalar
        self.det_npix_multiplier = np.array(parameters['npix_multiplier'], dtype=np.float64) # # of detector pixels per image plane "pixel"# nlambd array, 1 for detections or spectra w/ ERD, ~6*(140/SR) for spectra with IFS
        self.det_DC = np.array(parameters['dark_current'],dtype=np.float64) # dark current (counts pix^-1 s^-1) # nlambd array
        self.det_RN = np.array(parameters['read_noise'],dtype=np.float64) # read noise (counts pix^-1 read^-1) # nlambd array
        self.det_tread = np.array(parameters['read_time'],dtype=np.float64) # read time (s) # nlambd array
        self.det_CIC = np.array(parameters['cic'],dtype=np.float64) # clock induced charge (counts pix^-1 photon_count^-1) # nlambd array

        #Coronagraph parameters
        self.coro_type=parameters['coro_type']
        self.nrolls=parameters['nrolls']


        # -------- END OF INPUTS ---------

    def load_default_parameters(self) -> None:
        """
        Load default parameters for the simulation.

        This method sets default values for various simulation parameters that typically don't need to be changed,
        including psf_trunc_ratio, td_limit, nooptimize, and optimize_phase.
        """

        # Misc parameters that probably don't need to be changed
        self.psf_trunc_ratio = np.array([0.3],dtype=np.float64) # array
        self.td_limit = 1e20 # limit placed on exposure times (something really large for this code) # scalar
        self.nooptimize = 0 # do not attempt to optimize exposure times for this code # scalar
        self.optimize_phase = 0 # optimize the phase of the planet (does not work in this code) # scalar

        #Some things specific to this code
        # self.ntargs = 1 #specified in the reading of the input
        # self.nlambd = 1 #specified in the reading of the input
        self.ntot = 1
        self.nmeananom = 1
        self.norbits = 1
        self.xp = self.sp.copy()
        self.yp = self.sp.copy()*0.0  ## FOR NOW IT IS ASSUMED TO BE ON THE X AXIS SO THAT XP = SP (input) and YP = 0

    def calculate_zodi_exozodi(self):
        # calculate zodi and exozodi

        #calculate flux at zero point for the V band and the prescribed lambda
        self.F0V= (fluxzodi.calc_flux_zero_point(lambd=0.55, unit='pcgs',perlambd=True))/1e7 #convert to photons cm^-2 nm^-1 s^-1
        self.F0= (fluxzodi.calc_flux_zero_point(lambd=self.lambd, unit='pcgs',perlambd=True))/1e7 #convert to photons cm^-2 nm^-1 s^-1
        
        self.M_V = self.vmag - 5.0 * np.log10(self.dist) + 5.0 #calculate absolute V band mag of target #TODO in etc?
        self.Fzodi_list = fluxzodi.calc_zodi_flux(self.dec, self.ra, self.lambd, self.F0)
        
        self.Fexozodi_list = fluxzodi.calc_exozodi_flux(self.M_V, self.vmag, self.F0V, self.nzodis, self.lambd, self.mag, self.F0)
        self.Fbinary_list = np.full((self.nlambd,self.ntargs),0.0) #this code ignores stray light from binaries
        self.Fp0 = 10.**(-0.4*self.deltamag)       #flux of planet



