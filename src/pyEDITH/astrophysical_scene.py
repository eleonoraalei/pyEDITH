import numpy as np
from scipy.interpolate import interp1d


def calc_flux_zero_point(
    lambd: np.ndarray,
    unit: str = "",
    perlambd: bool = False,
    AB: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Calculate the flux zero point for given wavelengths.

    This function calculates the flux zero point for a given set of wavelengths.
    By default, it returns Johnson zero points. If the 'AB' flag is set, it will
    return AB flux zero points.

    Parameters:
    -----------
    lambd : np.ndarray
        Wavelengths in microns.
    unit : str, optional
        Output unit. Possible values are:
        - "jy" for output in Jy
        - "cgs" for output in erg s^-1 cm^-2 Hz^-1
        - "pcgs" for output in photons s^-1 cm^-2 Hz^-1
        Default is an empty string.
    perlambd : bool, optional
        If True, convert all output values to cm^-1 instead of Hz^-1.
        Default is False.
    AB : bool, optional
        If True, use AB magnitude system instead of Johnson.
        Default is False.
    verbose : bool, optional
        If True, print additional information.
        Default is False.

    Returns:
    --------
    np.ndarray
        Flux zero points for the given wavelengths in the specified units.

    Raises:
    -------
    ValueError
        If unit is not specified or is invalid, or if incompatible options are selected.

    Notes:
    ------
    The Johnson zero points come from Table A.2 from the Spitzer Telescope Handbook
    (http://irsa.ipac.caltech.edu/data/SPITZER/docs/spitzermission/missionoverview/spitzertelescopehandbook/19/).

    That table is as follows:
    Passband	Effective wavelength (microns)	Johnson Zero point (Jy)
    U	0.36	1823
    B	0.44	4130
    V	0.55	3781
    R	0.71	2941
    I	0.97	2635
    J	1.25	1603
    H	1.60	1075
    K	2.22	667
    L	3.54	288
    M	4.80	170
    N	10.6	36
    O	21.0	9.4

    The AB zero points are simply calculated as 5.5099e6 / lambd
    """

    if unit == "":
        raise ValueError(
            'Must specificy output units: unit="jy" for output in Jy, '
            + 'unit="cgs" for output in erg s^-1 cm^-2 Hz^-1, '
            + 'unit="pcgs" for output in photons s^-1 cm^-2 Hz^-1. '
        )
    if unit not in ["jy", "cgs", "pcgs"]:
        raise ValueError(
            'Invalid unit value. Possible values are: unit="jy" for output in Jy, '
            + 'unit="cgs" for output in erg s^-1 cm^-2 Hz^-1, '
            + 'unit="pcgs" for output in photons s^-1 cm^-2 Hz^-1. '
        )

    if unit == "jy" and perlambd:
        raise ValueError("Cannot set Jy and perlambd")

    # Calculate the zero point
    if AB:
        # AB magnitude system
        f0 = 3.6308e3  # This is the zero point Jy
    else:
        # Johnson magnitude system
        known_lambd = np.array(
            [0.36, 0.44, 0.55, 0.71, 0.97, 1.25, 1.60, 2.22, 3.54, 4.80, 10.6, 21.0]
        )
        known_zeropoint_jy = np.array(
            [1823, 4130, 3781, 2941, 2635, 1603, 1075, 667, 288, 170, 36, 9.4]
        )

        # Now we interpolate to lambd
        # Note that the interpolation is best done in log-log space
        interp = interp1d(
            np.log10(known_lambd),
            np.log10(known_zeropoint_jy),
            kind="cubic",
            fill_value="extrapolate",
        )  # TODO this interpolation is not exactly the same, is that okay?
        logf0 = interp(np.log10(lambd))
        f0 = 10.0**logf0  # undo the logarithm

    # Now change the units from Jy if necessary
    c = 2.998e10  # cm s^-1
    h = 6.62608e-27  # planck constant in cgs
    lambd_cm = lambd / 1e4
    ephoton_cgs = h * c / lambd_cm
    if unit == "cgs":
        f0 = np.double(f0) * 1e-23  # convert to CGS
    if unit == "pcgs":
        f0 = np.double(f0) * 1e-23 / ephoton_cgs  # convert to # photons in CGS
    if perlambd:
        f0 *= c / lambd_cm**2.0

    if unit == "jy":
        unittext = "Jy"
    if unit == "cgs":
        if not perlambd:
            unittext = "erg s^-1 cm^-2 Hz^-1 (CGS per unit frequency)"
        else:
            unittext = "erg s^-1 cm^-3 (CGS per unit wavelength)"
    if unit == "pcgs":
        if not perlambd:
            unittext = "photons s^-1 cm^-2 Hz^-1 (CGS per unit frequency)"
        else:
            unittext = "photons s^-1 cm^-3 (CGS per unit wavelength)"

    if verbose:
        print(
            "Johnson flux zero point calculated at %.2f" % lambd
            + " microns in units of "
            + unittext
        )

    return np.array(f0)


def calc_exozodi_flux(
    M_V: np.ndarray,
    vmag: np.ndarray,
    F0V: float,
    nexozodis: np.ndarray,
    lambd: np.ndarray,
    lambdmag: np.ndarray,
    F0lambd: np.ndarray,
) -> np.ndarray:
    """
    Calculate the exozodiacal light flux for given stellar and observational parameters.

    This function computes the exozodiacal light flux for a set of stars, taking into
    account their absolute and apparent magnitudes, the number of exozodis, and the
    observational wavelengths.

    Parameters:
    -----------
    M_V : np.ndarray
        V band absolute magnitude of stars.
    vmag : np.ndarray
        V band apparent magnitude of stars.
    F0V : float
        V band flux zero point of stars.
    nexozodis : np.ndarray
        Number of exozodis for each star.
    lambd : np.ndarray
        Wavelength in microns (vector of length nlambd).
    lambdmag : np.ndarray
        Apparent magnitude of stars at each lambd (array nstars x nlambd).
    F0lambd : np.ndarray
        Flux zero point at wavelength lambd (vector of length nlambd).

    Returns:
    --------
    np.ndarray
        Exozodi surface brightness in units of photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0.
        This is equivalent to 10^(-0.4*magOmega_EZ).
        - Multiply by F0 to get photons s^-1 cm^-2 arcsec^-2 nm^-1
        - Multiply by energy of photons to get erg s^-1 cm^-2 arcsec^-2 nm^-1
        The output array has dimensions (nlambd, nstars).

    Notes:
    ------
    - The function assumes a V band surface brightness of 22.0 mag arcsec^-2 for 1 zodi.
    - The calculation maintains a constant optical depth regardless of stellar type.
    - The function uses the Sun's V band absolute magnitude (4.83) as a reference.

    Raises:
    -------
    ValueError
        If the input arrays have incompatible dimensions.
    """

    if len(lambd) != len(F0lambd):
        raise ValueError(
            "ERROR. F0lambd and lambd must be vectors of identical length."
        )
    if len(lambd) > 1:
        if len(lambd) != lambdmag.shape[0]:
            raise ValueError(
                "ERROR. lambdmag must have dimensions nstars x nlambd, \
                      where nlambd is the length of lambd."
            )  # TODO check because IDL and PYTHON are flipped

    vmag_1zodi = 22.0  # V band surface brightness of 1 zodi in mag arcsec^-2
    vflux_1zodi = 10.0 ** (
        -0.4 * vmag_1zodi
    )  # V band flux (modulo the F0 factor out front)

    M_V_sun = 4.83  # V band absolute magnitude of Sun

    # Calculate counts s^-1 cm^-2 arcsec^-2 nm^-1 @ V band
    # The following formula maintains a constant optical depth regardless
    # of stellar type
    # NOTE: the older version of the code used the following expression
    # flux_exozodi_Vband = F0V * (double(10.)^(-0.4*(M_V - M_V_sun))
    #                                       / Lstar) * (nexozodis * vflux_1zodi)
    # The above expression has a 1/Lstar in it, which corrects for the
    # 1/r^2 factor based on the EEID.  In the new version of exposure_time_calculator.c,
    # we explicitly divide by 1/r^2, so the new version of the
    # eflux_exozodi_Vband expression does not divide by Lstar.
    flux_exozodi_Vband = (
        F0V * np.double(10.0) ** (-0.4 * (M_V - M_V_sun)) * (nexozodis * vflux_1zodi)
    )  # vector of length nstars

    # Now, multiply by the ratio of the star's counts received at lambd to those
    # received at V band
    nstars = len(vmag)
    nlambd = len(lambd)
    flux_exozodi_lambd = np.full(
        (nlambd, nstars), flux_exozodi_Vband
    )  # congrid([[flux_exozodi_Vband],[flux_exozodi_Vband]],nstars,nlambd)

    # TODO in python it needs to be treated differently if it is 1D or 2D array.
    # Check better solution?
    # if nstars>1:
    for ilambd in np.arange(nlambd):
        flux_exozodi_lambd[ilambd, :] *= (
            F0lambd[ilambd] * np.double(10.0) ** (-0.4 * lambdmag[ilambd, :])
        ) / (F0V * np.double(10.0) ** (-0.4 * vmag))
    # else:
    #   for ilambd in np.arange(nlambd):
    #     flux_exozodi_lambd[ilambd] *= (F0lambd[ilambd] * np.double(10.)**(-0.4*
    #                   lambdmag[ilambd])) / (F0V * np.double(10.)**(-0.4*vmag))

    # Now, because we return just the quantity 10.^(-0.4*magOmega_EZ), we remove the F0
    for ilambd in np.arange(nlambd):
        flux_exozodi_lambd[ilambd, :] /= F0lambd[ilambd]

    return flux_exozodi_lambd  # TODO. Unclear what shape is this supposed to have


def calc_zodi_flux(
    dec: np.ndarray,
    ra: np.ndarray,
    lambd: np.ndarray,
    F0: np.ndarray,
    starshade: bool = False,
    ss_elongation: np.ndarray = np.array([]),
) -> np.ndarray:
    """
    Calculate the zodiacal light flux for given celestial coordinates and wavelengths.

    This function computes the zodiacal light flux based on the target's position in the sky,
    observation wavelengths, and whether a starshade is used. It uses the model from
    Leinert et al. (1998) to calculate the zodiacal light intensity.

    Parameters:
    -----------
    dec : np.ndarray
        Declination of targets in degrees (J2000 equatorial coordinate).
    ra : np.ndarray
        Right ascension of targets in degrees (J2000 equatorial coordinate).
    lambd : np.ndarray
        Wavelengths in microns (vector of length nlambd).
    F0 : np.ndarray
        Flux zero points at wavelengths lambd (vector of length nlambd).
    starshade : bool, optional
        Flag to enable starshade mode (default is False).
    ss_elongation : np.ndarray, optional
        Mean solar elongation for the starshade's observations in degrees (required if starshade=True).

    Returns:
    --------
    np.ndarray
        Zodi surface brightness in units of photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0.
        This is equivalent to 10^(-0.4*magOmega_ZL).
        - Multiply by F0 to get photons s^-1 cm^-2 arcsec^-2 nm^-1
        - Multiply by energy of photons to get erg s^-1 cm^-2 arcsec^-2 nm^-1
        The output array has dimensions (nlambd, nstars).

    Raises:
    -------
    ValueError
        If F0 and lambd have different lengths, or if starshade mode is inconsistent with ss_elongation.

    Notes:
    ------
    - The function uses the zodiacal light model from Leinert et al. (1998).
    - For coronagraph mode, it assumes observations near solar longitude of 135 degrees.
    - Starshade functionality is currently not fully implemented.

    References:
    -----------
    Leinert, C., et al. (1998). The 1997 reference of diffuse night sky brightness.
    Astronomy and Astrophysics Supplement Series, 127(1), 1-99.
    """

    if len(lambd) != len(F0):
        raise ValueError("ERROR. F0 and lambd must be vectors of identical length.")
    if starshade and (len(ss_elongation) == 0):
        raise ValueError(
            "ERROR. You have set the STARSHADE flag.  Must specify SS_ELONGATION \
                  in degrees."
        )
    if not starshade and (len(ss_elongation) > 0):
        raise ValueError(
            "ERROR. You must enable STARSHADE mode if you are setting SS_ELONGATION \
                in degrees."
        )

    # Convert to radians internally
    dec_rad = dec * (np.double(np.pi) / 180.0)
    ra_rad = ra * (np.double(np.pi) / 180.0)

    # First, convert equatorial coordinates to ecliptic coordinates
    # This is nicely summarized in Leinert et al. (1998)
    eps = 23.439  # J2000 obliquity of the ecliptic in degrees
    eps_rad = eps * (np.double(np.pi) / 180.0)  # convert to radians
    sinbeta = np.sin(dec_rad) * np.cos(eps_rad) - np.cos(dec_rad) * np.sin(
        eps_rad
    ) * np.sin(
        ra_rad
    )  # all we need is the sine of the latitude

    # First, convert equatorial coordinates to ecliptic coordinates
    # This is nicely summarized in Leinert et al. (1998)
    eps = 23.439  # J2000 obliquity of the ecliptic in degrees
    eps_rad = eps * (np.double(np.pi) / 180.0)  # convert to radians
    sinbeta = np.sin(dec_rad) * np.cos(eps_rad) - np.cos(dec_rad) * np.sin(
        eps_rad
    ) * np.sin(
        ra_rad
    )  # all we need is the sine of the latitude
    # sinbeta0 = sinbeta  # contains the +/- values of beta
    sinbeta = abs(sinbeta)  # f135 below is symmetric about beta=0
    # sinbeta2 = sinbeta * sinbeta
    # sinbeta3 = sinbeta2 * sinbeta

    # print,'beta = ',asin(sinbeta)*180./pi

    # COMMENTED OUT SINCE THIS IS NOT REALLY USED (EXCEPT FROM ONE SPECIFIC ROW/COLUMN)
    # NOTE: assumes the same degree of observation so far
    # Here are the solar longitude and beta values for Table 17 from
    # Leinert et al. (1998)
    beta_vector = np.array([0.0, 5, 10, 15, 20, 25, 30, 45, 60, 75])
    beta_array = np.full((20, 10), beta_vector)
    beta_array = beta_array[0:18, :]  # TODO Why though?
    sollong_vector = np.array(
        [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            60,
            75,
            90,
            105,
            120,
            135,
            150,
            165,
            180.0,
        ]
    )
    # sollong_array = np.full(
    #    (10, 19), sollong_vector
    # ).T  # rotates so that we get 19 rows x 10 columns, and the rows
    # have increasing values according to sollong_vector

    # Here are the values in Table 17
    table17 = np.array(
        [
            [-1, -1, -1, 3140, 1610, 985, 640, 275, 150, 100],
            [-1, -1, -1, 2940, 1540, 945, 625, 271, 150, 100],
            [-1, -1, 4740, 2470, 1370, 865, 590, 264, 148, 100],
            [11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100],
            [6400, 4480, 2410, 1410, 910, 635, 454, 237, 141, 99],
            [3840, 2830, 1730, 1100, 749, 545, 410, 223, 136, 97],
            [2480, 1870, 1220, 845, 615, 467, 365, 207, 131, 95],
            [1650, 1270, 910, 680, 510, 397, 320, 193, 125, 93],
            [1180, 940, 700, 530, 416, 338, 282, 179, 120, 92],
            [910, 730, 555, 442, 356, 292, 250, 166, 116, 90],
            [505, 442, 352, 292, 243, 209, 183, 134, 104, 86],
            [338, 317, 269, 227, 196, 172, 151, 116, 93, 82],
            [259, 251, 225, 193, 166, 147, 132, 104, 86, 79],
            [212, 210, 197, 170, 150, 133, 119, 96, 82, 77],
            [188, 186, 177, 154, 138, 125, 113, 90, 77, 74],
            [179, 178, 166, 147, 134, 122, 110, 90, 77, 73],
            [179, 178, 165, 148, 137, 127, 116, 96, 79, 72],
            [196, 192, 179, 165, 151, 141, 131, 104, 82, 72],
            [230, 212, 195, 178, 163, 148, 134, 105, 83, 72],
        ]
    )

    # For a coronagraph, we assume star can be observed near sollong ~ 135 degrees
    # The old f135 calculation.  Now we interpolate the above table.
    # I135_o_I90 = 0.69
    # f135 = 1.02331 - 0.565652*sinbeta - 0.883996*sinbeta2 + 0.852900*sinbeta3
    # f = I135_o_I90 * f135
    j = np.argmin(abs(sollong_vector - 135.0))
    k = np.argmin(abs(sollong_vector - 90.0))
    interp = interp1d(
        np.sin(beta_vector * np.pi / 180.0),
        table17[j] / table17[k, 0],
        kind="cubic",
        fill_value="extrapolate",
    )
    f = interp(sinbeta)

    # STARSHADE functionality not enabled #
    # For a starshade, we assume a mean solar elongation of 59 degrees
    # if starshade==True:
    #     # TODO: not sure this works, but it is not used

    #     #First, calculate inclination to point at beta of all targets
    #     inclination = np.arcsin(sinbeta0.copy() / np.sin(ss_elongation*
    #                                           np.pi/180.))*180./np.pi
    #     cossollong = np.cos(ss_elongation*np.pi/180.)/np.cos(
    #                           np.arcsin(sinbeta.copy())) #solar longitude of
    #                                           targets at elongation of ss_elongation
    #     sinsollong = np.cos(inclination*np.pi/180.) * np.sin(ss_elongation*np.pi/180.)
    #                                            / np.cos(np.arcsin(sinbeta0.copy()))
    #     sollong = np.arctan(sinsollong.copy(),cossollong.copy())*180/np.pi
    #     #Now that we have the solar longitude and beta of every target at a
    #     #solar elongation of ss_elongation, we can calculate their zodi values

    #     sollong_indices=sollong.copy()#first, just make some arrays
    #     sollong_indices[sollong<45.]=sollong[sollong<45.]/5
    #     sollong_indices[sollong>=45.]= 9.+(sollong[sollong>=45]-45.)/15.

    #     beta = abs(np.arcsin(sinbeta.copy())*180./np.pi)
    #     beta_indices=sollong.copy()
    #     beta_indices[beta<30.]=beta[beta<30.]/5
    #     beta_indices[beta>=45.]= 6.+(beta[beta>=30.]-30.)/15.

    #     points=np.concatenate(([sollong_indices],[beta_indices]))
    #     f = interpn((sollong_vector,beta_vector), table17/table17[k,0], points.T)
    # TODO: Change to cubic interpolation

    # wavelength dependence
    # The following comes from fits to Table 19 in Leinert et al 1998

    zodi_lambd = np.array(
        [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5, 4.8, 12, 25, 60, 100, 140]
    )  # microns
    zodi_blambd = np.array(
        [
            2.5e-8,
            5.3e-7,
            2.2e-6,
            2.6e-6,
            2.0e-6,
            1.3e-6,
            1.2e-6,
            8.1e-7,
            1.7e-7,
            5.2e-8,
            1.2e-7,
            7.5e-7,
            3.2e-7,
            1.8e-8,
            3.2e-9,
            6.9e-10,
        ]
    )  # W m^-2 sr^-1 micron^-1
    # convert the above to erg s^-1 cm^-2 arcsec^-2 nm^-1
    zodi_blambd *= 1.0e7  # convert W to erg s^-1
    zodi_blambd /= 1.0e4  # convert m^-2 to cm^-2
    zodi_blambd /= 4.25e10  # convert sr^-1 to arcsec^-2
    zodi_blambd /= 1000.0  # convert micron^-1 to nm^-1

    interp = interp1d(np.log10(zodi_lambd), np.log10(zodi_blambd), kind="cubic")
    blambd = interp(np.log10(lambd))
    blambd = 10.0**blambd
    I90fabsfco = blambd

    # Now divide by energy of a photon in erg
    c = 2.998e10  # cm s^-1
    h = 6.62608e-27  # planck constant in cgs
    ephoton = h * c / (lambd / 1e4)
    I90fabsfco /= ephoton
    I90fabsfco /= F0

    # Multiply by the wavelength dependence for each value of lambd
    nstars = len(ra)
    nlambd = len(lambd)

    flux_zodi = np.full((nlambd, nstars), f)
    for ilambd in range(nlambd):
        flux_zodi[ilambd, :] *= I90fabsfco[ilambd]

    return flux_zodi


class AstrophysicalScene:
    """
    A class representing an astrophysical scene for exoplanet observation simulations.

    This class encapsulates various astrophysical parameters and methods to calculate
    zodi and exozodi fluxes for a set of target stars and their potential exoplanets.

    Attributes
    ----------
    ntargs : int
        Number of target stars
    Lstar : ndarray
        Luminosity of stars in solar luminosities
    dist : ndarray
        Distance to stars in parsecs
    vmag : ndarray
        Stellar magnitudes at V band
    mag : ndarray
        Stellar magnitudes at desired wavelengths
    angdiam_arcsec : ndarray
        Angular diameter of stars in arcseconds
    nzodis : ndarray
        Amount of exozodi around target stars in "zodis"
    ra : ndarray
        Right ascension of target stars in degrees
    dec : ndarray
        Declination of target stars in degrees
    sp : ndarray
        Separation of planets in arcseconds
    xp : ndarray
        X-coordinate of planets in arcseconds
    yp : ndarray
        Y-coordinate of planets in arcseconds
    deltamag : ndarray
        Magnitude difference between planets and host stars
    min_deltamag : ndarray
        Brightest planet to resolve at the IWA
    F0V : float
        Flux zero point for V band
    F0 : ndarray
        Flux zero points for prescribed wavelengths
    M_V : ndarray
        Absolute V band magnitudes of target stars
    Fzodi_list : ndarray
        Zodiacal light fluxes
    Fexozodi_list : ndarray
        Exozodiacal light fluxes
    Fbinary_list : ndarray
        Binary star fluxes (currently ignored)
    Fp0 : ndarray
        Flux of planets

    Methods
    -------
    load_configuration(parameters)
        Load configuration parameters for the simulation from a dictionary.
    calculate_zodi_exozodi(observation)
        Calculate zodiacal and exozodiacal light fluxes for the given observation.
    """

    def __init__(self) -> None:
        """
        Initialize the AstrophysicalScene object with default values for output arrays.
        """
        pass
        # there are no default values, TODO it should fail if not provided

    def load_configuration(self, parameters: dict) -> None:
        """
        Load configuration parameters for the simulation from a dictionary.

        This method initializes various attributes of the AstrophysicalScene object
        using the provided parameters dictionary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing simulation parameters including target star
            parameters, planet parameters, and observational parameters.
        Returns
        -------
        None

        Raises
        ------
        KeyError
            If a required parameter is missing from the input dictionary.
        """

        # -------- INPUTS ---------
        # Target star parameters
        self.ntargs = 1

        # luminosity of star (solar luminosities) (ntargs array)
        self.Lstar = np.array(parameters["Lstar"], dtype=np.float64)

        # distance to star (pc) (ntargs array)
        self.dist = np.array(parameters["distance"], dtype=np.float64)

        # stellar mag at V band (ntargs array)
        self.vmag = np.array(parameters["magV"], dtype=np.float64)

        # stellar mag at desired lambd (nlambd x ntargs array)
        self.mag = np.array(parameters["mag"], dtype=np.float64)

        # angular diameter of star (arcsec) (ntargs array)
        self.angdiam_arcsec = np.array(parameters["angdiam"], dtype=np.float64)

        # amount of exozodi around target star ("zodis") (ntargs array)
        self.nzodis = np.array(parameters["nzodis"], dtype=np.float64)

        # right ascension of target star used to estimate zodi (deg) (ntargs array)
        self.ra = np.array(parameters["ra"], dtype=np.float64)

        # declination of target star used to estimate zodi (deg) (ntargs array)
        self.dec = np.array(parameters["dec"], dtype=np.float64)

        # Planet parameters
        # separation of planet (arcseconds) (nmeananom x norbits x ntargs array)
        # NOTE FOR NOW IT IS ASSUMED TO BE ON THE X AXIS
        # SO THAT XP = SP (input) and YP = 0
        self.sp = np.array(parameters["sp"], dtype=np.float64)
        self.xp = self.sp.copy()
        self.yp = self.sp.copy() * 0.0

        # difference in mag between planet and host star
        # (nmeananom x norbits x ntargs array)
        self.deltamag = np.array(parameters["delta_mag"], dtype=np.float64)
        # brightest planet to resolve w/ photon counting detector evaluated at
        # the IWA, sets the time between counts (ntargs array)
        self.min_deltamag = np.array(parameters["delta_mag_min"], dtype=np.float64)

    def calculate_zodi_exozodi(self, observation: object) -> None:
        """
        Calculate zodiacal and exozodiacal light fluxes for the given observation.

        This method computes the flux zero points, zodiacal light fluxes, and
        exozodiacal light fluxes for the target stars based on the provided
        observation parameters.

        Parameters
        ----------
        observation : object
            An object containing observation parameters.
            Must have attributes:
                lambd : array_like
                    Wavelengths for the observation.
                nlambd : int
                    Number of wavelengths.

        Returns
        -------
        None

        """

        # calculate flux at zero point for the V band and the prescribed lambda
        self.F0V = (
            calc_flux_zero_point(lambd=0.55, unit="pcgs", perlambd=True)
        ) / 1e7  # convert to photons cm^-2 nm^-1 s^-1
        self.F0 = (
            calc_flux_zero_point(lambd=observation.lambd, unit="pcgs", perlambd=True)
        ) / 1e7  # convert to photons cm^-2 nm^-1 s^-1

        self.M_V = (
            self.vmag - 5.0 * np.log10(self.dist) + 5.0
        )  # calculate absolute V band mag of target
        self.Fzodi_list = calc_zodi_flux(self.dec, self.ra, observation.lambd, self.F0)

        self.Fexozodi_list = calc_exozodi_flux(
            self.M_V,
            self.vmag,
            self.F0V,
            self.nzodis,
            observation.lambd,
            self.mag,
            self.F0,
        )
        self.Fbinary_list = np.full(
            (observation.nlambd, self.ntargs), 0.0
        )  # this code ignores stray light from binaries
        self.Fp0 = 10.0 ** (-0.4 * self.deltamag)  # flux of planet

        ### TODO deltamag --> Fp/Fs
