import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
from .units import *
from astropy.coordinates import SkyCoord


def calc_flux_zero_point(
    lambd: u.Quantity,
    output_unit: str = "pcgs",
    perlambd: bool = False,
    AB: bool = False,
    verbose: bool = False,
) -> u.Quantity:
    """
    Calculate the flux zero point for given wavelengths.

    This function calculates the flux zero point for a given set of wavelengths.
    By default, it returns Johnson zero points. If the 'AB' flag is set, it will
    return AB flux zero points.

    Parameters:
    -----------
    lambd : astropy.units.Quantity
        Wavelengths. Must be in units of length.
    output_unit : str, optional
        Output unit type. Possible values are:
        - "jy" for output in Jy
        - "cgs" for output in erg s^-1 cm^-2 Hz^-1
        - "pcgs" for output in photons s^-1 cm^-2 Hz^-1
        Default is "pcgs".
    perlambd : bool, optional
        If True, convert all output values to per wavelength instead of per frequency.
        Default is False.
    AB : bool, optional
        If True, use AB magnitude system instead of Johnson.
        Default is False.
    verbose : bool, optional
        If True, print additional information.
        Default is False.

    Returns:
    --------
    astropy.units.Quantity
        Flux zero points for the given wavelengths in the specified units.

    Raises:
    -------
    ValueError
        If output_unit is not specified or is invalid, or if incompatible options are selected.
    """

    if output_unit not in ["jy", "cgs", "pcgs"]:
        raise ValueError(
            'Invalid output_unit value. Possible values are: "jy" for output in Jy, '
            '"cgs" for output in erg s^-1 cm^-2 Hz^-1, '
            '"pcgs" for output in photons s^-1 cm^-2 Hz^-1.'
        )

    if output_unit == "jy" and perlambd:
        raise ValueError("Cannot set Jy and perlambd")

    # Ensure lambd is in the correct units (cm for CGS)
    lambd = lambd.to(u.cm)

    # Convert constants to CGS
    h_cgs = const.h.cgs.value
    c_cgs = const.c.cgs.value

    # Calculate the zero point
    if AB:
        # AB magnitude system
        f0 = 3631 * u.Jy  # AB zero point
    else:
        # Johnson magnitude system
        known_lambd = (
            np.array(
                [0.36, 0.44, 0.55, 0.71, 0.97, 1.25, 1.60, 2.22, 3.54, 4.80, 10.6, 21.0]
            )
            * u.um
        )
        known_zeropoint_jy = (
            np.array([1823, 4130, 3781, 2941, 2635, 1603, 1075, 667, 288, 170, 36, 9.4])
            * u.Jy
        )

        # Interpolation in log-log space
        interp = interp1d(
            np.log10(known_lambd.value),
            np.log10(known_zeropoint_jy.value),
            kind="cubic",
            fill_value="extrapolate",
        )
        logf0 = interp(np.log10(lambd.to(u.um).value))
        f0 = 10.0**logf0 * u.Jy

    # Convert to desired output units
    if output_unit == "cgs":
        # Convert to erg / (s * cm^2 * Hz)
        f0 = f0.to(SPECTRAL_FLUX_DENSITY_CGS)
    elif output_unit == "pcgs":
        # Convert to photons / (s * cm^2 * Hz)
        f0 = f0.to(
            PHOTON_COUNT / (u.cm**2 * u.s * u.Hz),
            equivalencies=u.spectral_density(lambd),
        )

    # Convert to per wavelength if requested
    if perlambd:
        if output_unit == "jy":
            f0 = f0.to(u.Jy / u.cm, equivalencies=u.spectral_density(lambd))
        elif output_unit == "cgs":
            f0 = f0.to(u.erg / (u.s * u.cm**3), equivalencies=u.spectral_density(lambd))
        elif output_unit == "pcgs":
            f0 = f0.to(
                PHOTON_COUNT / (u.s * u.cm**3), equivalencies=u.spectral_density(lambd)
            )

    if verbose:
        print(f"Flux zero point calculated at {lambd} in units of {f0.unit}")

    return f0


def calc_exozodi_flux(
    M_V: u.Quantity,
    vmag: u.Quantity,
    nexozodis: u.Quantity,
    lambd: u.Quantity,
    lambdmag: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodiacal light flux for given stellar and observational parameters.

    This function computes the exozodiacal light flux for a set of stars, taking into
    account their absolute and apparent magnitudes, the number of exozodis, and the
    observational wavelengths.

    Parameters:
    -----------
    M_V : Quantity
        V band absolute magnitude of stars.
    vmag : Quantity
        V band apparent magnitude of stars.
    nexozodis : Quantity
        Number of exozodis for each star.
    lambd : Quantity
        Wavelength in microns (vector of length nlambd).
    lambdmag : Quantity
        Apparent magnitude of stars at each lambd (array nstars x nlambd).
    F0lambd : Quantity
        Flux zero point at wavelength lambd (vector of length nlambd).

    Returns:
    --------
    Quantity
        Exozodi surface brightness in units of photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0.
        This is equivalent to 10^(-0.4*magOmega_EZ).

    """

    if len(lambd) > 1:
        if len(lambd) != lambdmag.shape[0]:
            raise ValueError(
                "ERROR. lambdmag must have dimensions nstars x nlambd, where nlambd is the length of lambd."
            )

    vmag_1zodi = (
        22.0 * SURFACE_BRIGHTNESS
    )  # V band surface brightness of 1 zodi in mag arcsec^-2
    vflux_1zodi = 10.0 ** (
        -0.4 * vmag_1zodi.value
    )  # V band flux (modulo the F0 factor out front) #TODO check that it is correctly interpreted as dimensionless

    M_V_sun = 4.83 * MAGNITUDE  # V band absolute magnitude of Sun

    """
    UNNECESSARY
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
        F0V * 10.0 ** (-0.4 * (M_V - M_V_sun).value) * (nexozodis * vflux_1zodi)
    )  # vector of length nstars
    """
    # Now, multiply by the ratio of the star's counts received at lambd to those
    # received at V band
    nstars = len(vmag)
    nlambd = len(lambd)

    flux_exozodi_lambd = np.zeros((nlambd, nstars)) * DIMENSIONLESS  # modulo factor F0

    # Adjust flux for each wavelength
    for ilambd in range(nlambd):
        flux_exozodi_lambd[ilambd, :] = (
            nexozodis
            * vflux_1zodi
            * 10.0 ** (-0.4 * (M_V - M_V_sun).value)
            * 10.0 ** (-0.4 * (lambdmag[ilambd, :] - vmag).value)
        )
    """
    I simplified the equation. It was originally this:

    flux_exozodi_lambd = np.full(
        (nlambd, nstars), flux_exozodi_Vband
    )  # congrid([[flux_exozodi_Vband],[flux_exozodi_Vband]],nstars,nlambd)

    for ilambd in np.arange(nlambd):
        flux_exozodi_lambd[ilambd, :] *= (
            F0lambd[ilambd] * np.double(10.0) ** (-0.4 * lambdmag[ilambd, :])
        ) / (F0V * np.double(10.0) ** (-0.4 * vmag))
    )

    # Now, because we return just the quantity 10.^(-0.4*magOmega_EZ), we remove the F0
    for ilambd in np.arange(nlambd):
        flux_exozodi_lambd[ilambd, :] /= F0lambd[ilambd]
    """

    return (
        flux_exozodi_lambd * INV_SQUARE_ARCSEC
    )  # CONVERTING FROM FLUX TO SURFACE BRIGHTNESS (modulo F0 that gets multiplied later)


def calc_zodi_flux(
    dec: u.Quantity,
    ra: u.Quantity,
    lambd: u.Quantity,
    F0: u.Quantity,
    starshade: bool = False,
    ss_elongation: u.Quantity = None,
) -> u.Quantity:
    """

    Calculate the zodiacal light flux for given celestial coordinates and wavelengths.

    This function computes the zodiacal light flux based on the target's position in the sky,
    observation wavelengths, and whether a starshade is used. It uses the model from
    Leinert et al. (1998) to calculate the zodiacal light intensity.

    Parameters:
    -----------
    dec : Quantity
        Declination of targets in degrees (J2000 equatorial coordinate).
    ra : Quantity
        Right ascension of targets in degrees (J2000 equatorial coordinate).
    lambd : Quantity
        Wavelengths in microns (vector of length nlambd).
    F0 : Quantity
        Flux zero points at wavelengths lambd (vector of length nlambd).
    starshade : bool, optional
        Flag to enable starshade mode (default is False).
    ss_elongation : Quantity, optional
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

    if starshade and ss_elongation is None:
        raise ValueError(
            "ERROR. You have set the STARSHADE flag. Must specify SS_ELONGATION in degrees."
        )
    if not starshade and ss_elongation is not None:
        raise ValueError(
            "ERROR. You must enable STARSHADE mode if you are setting SS_ELONGATION in degrees."
        )

    # Convert equatorial coordinates to ecliptic coordinates
    coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
    ecl_coords = coords.barycentrictrueecliptic
    beta = ecl_coords.lat.rad

    # all we need is the sine of the latitude
    # Use absolute value of the sin of beta (symmetry about ecliptic plane)
    sinbeta = np.abs(np.sin(beta))

    # SOURCE: Leinert et al. (1998)
    # Define solar longitude and beta values for interpolation
    beta_vector = np.array([0.0, 5, 10, 15, 20, 25, 30, 45, 60, 75]) * u.deg
    sollong_vector = (
        np.array(
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
        * u.deg
    )

    # Table 17 values (assumed to be in some brightness units)
    table17 = (
        np.array(
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
        * SPECTRAL_RADIANCE
    )
    # For coronagraph, assume observations near solar longitude of 135 degrees
    j = np.argmin(np.abs(sollong_vector - 135 * u.deg))
    k = np.argmin(np.abs(sollong_vector - 90 * u.deg))

    # Interpolate to get zodi brightness factor
    from scipy.interpolate import interp1d

    interp = interp1d(
        np.sin(beta_vector),
        table17[j]
        / table17[
            k, 0
        ],  # TODO: Is this where the factor 1e-8 in the original table disappears?
        kind="cubic",
        fill_value="extrapolate",
    )
    f = interp(sinbeta) * DIMENSIONLESS

    # Wavelength dependence (fits to Table 19 in Leinert et al 1998)
    zodi_lambd = (
        np.array(
            [
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
                1.0,
                1.2,
                2.2,
                3.5,
                4.8,
                12,
                25,
                60,
                100,
                140,
            ]
        )
        * WAVELENGTH
    )
    zodi_blambd = (
        np.array(
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
        )
        * SPECTRAL_RADIANCE
    )

    # Convert to erg s^-1 cm^-2 arcsec^-2 nm^-1
    zodi_blambd = zodi_blambd.to(SPECTRAL_RADIANCE_CGS)

    # Interpolate to get zodi brightness at desired wavelengths
    interp = interp1d(
        np.log10(zodi_lambd.value), np.log10(zodi_blambd.value), kind="cubic"
    )
    blambd = 10 ** interp(np.log10(lambd.to(u.um).value)) * zodi_blambd.unit

    # Convert to photon flux
    # I90fabsfco = blambd / (u.h * u.c / lambd)

    I90fabsfco = blambd.to(
        PHOTON_SPECTRAL_RADIANCE,
        equivalencies=u.spectral_density(lambd),
    )
    # Divide by F0
    I90fabsfco = I90fabsfco / F0

    # Calculate final zodi flux
    nstars = len(ra)
    nlambd = len(lambd)
    flux_zodi = u.Quantity(np.zeros((nlambd, nstars)), unit=I90fabsfco.unit)
    for ilambd in range(nlambd):
        flux_zodi[ilambd, :] = f * I90fabsfco[ilambd]

    return flux_zodi  # 1/arcsec^2 (UNITS OF SPECTRAL RADIANCE)


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
        self.Lstar = parameters["Lstar"] * LUMINOSITY

        # distance to star (pc) (ntargs array)
        self.dist = parameters["distance"] * DISTANCE

        # stellar mag at V band (ntargs array)
        self.vmag = parameters["magV"] * MAGNITUDE

        # stellar mag at desired lambd (nlambd x ntargs array)
        self.mag = parameters["mag"] * MAGNITUDE

        # angular diameter of star (arcsec) (ntargs array)
        self.angdiam_arcsec = parameters["angdiam"] * ARCSEC

        # amount of exozodi around target star ("zodis") (ntargs array)
        self.nzodis = parameters["nzodis"] * ZODI

        # right ascension of target star used to estimate zodi (deg) (ntargs array)
        self.ra = parameters["ra"] * DEG

        # declination of target star used to estimate zodi (deg) (ntargs array)
        self.dec = parameters["dec"] * DEG

        # Planet parameters
        # separation of planet (arcseconds) (nmeananom x norbits x ntargs array)
        # NOTE FOR NOW IT IS ASSUMED TO BE ON THE X AXIS
        # SO THAT XP = SP (input) and YP = 0
        self.sp = parameters["sp"] * ARCSEC
        self.xp = self.sp.copy()
        self.yp = self.sp.copy() * 0.0

        # difference in mag between planet and host star
        # (nmeananom x norbits x ntargs array)
        self.deltamag = parameters["delta_mag"] * MAGNITUDE
        # brightest planet to resolve w/ photon counting detector evaluated at
        # the IWA, sets the time between counts (ntargs array)
        self.min_deltamag = parameters["delta_mag_min"] * MAGNITUDE

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

        # TODO done now to preserve the output of calc_flux_zero_point compared
        # to Chris's code. Consider converting to the right units directly.

        self.F0V = (
            calc_flux_zero_point(
                lambd=0.55 * WAVELENGTH, output_unit="pcgs", perlambd=True
            )
        ).to(
            PHOTON_FLUX_DENSITY, equivalencies=u.spectral_density(0.55 * WAVELENGTH)
        )  # convert to photons cm^-2 nm^-1 s^-1

        self.F0 = (
            calc_flux_zero_point(
                lambd=observation.lambd, output_unit="pcgs", perlambd=True
            )
        ).to(PHOTON_FLUX_DENSITY, equivalencies=u.spectral_density(observation.lambd))
        # convert to photons cm^-2 nm^-1 s^-1

        self.M_V = (
            self.vmag - 5.0 * np.log10(self.dist.value) * MAGNITUDE + 5.0 * MAGNITUDE
        )  # calculate absolute V band mag of target

        self.Fzodi_list = calc_zodi_flux(self.dec, self.ra, observation.lambd, self.F0)
        print(self.Fzodi_list)
        self.Fexozodi_list = calc_exozodi_flux(
            self.M_V,
            self.vmag,
            self.nzodis,
            observation.lambd,
            self.mag,
        )  # scalar

        self.Fbinary_list = (
            np.full((observation.nlambd, self.ntargs), 0.0) * DIMENSIONLESS
        )  # this code ignores stray light from binaries

        # flux of planet (dimensionless factor, will be multiplied by F0 internally which gives it flux units)
        self.Fp0 = (10.0 ** (-0.4 * self.deltamag.value)) * DIMENSIONLESS

        ### TODO deltamag --> Fp/Fs

    def validate_configuration(self):
        """
        Check that mandatory variables are there and have the right format.
        There can be other variables, but they are not needed for the calculation.
        """
        expected_args = {
            "ntargs": int,
            "Lstar": LUMINOSITY,
            "dist": DISTANCE,
            "vmag": MAGNITUDE,
            "mag": MAGNITUDE,
            "angdiam_arcsec": ARCSEC,
            "nzodis": ZODI,
            "ra": DEG,
            "dec": DEG,
            "sp": ARCSEC,
            "xp": ARCSEC,
            "yp": ARCSEC,
            "deltamag": MAGNITUDE,
            "min_deltamag": MAGNITUDE,
            "F0V": PHOTON_FLUX_DENSITY,
            "F0": PHOTON_FLUX_DENSITY,
            "M_V": MAGNITUDE,
            "Fzodi_list": INV_SQUARE_ARCSEC,
            "Fexozodi_list": INV_SQUARE_ARCSEC,
            "Fbinary_list": DIMENSIONLESS,
            "Fp0": DIMENSIONLESS,
        }

        for arg, expected_type in expected_args.items():
            if not hasattr(self, arg):
                raise AttributeError(f"AstrophysicalScene is missing attribute: {arg}")

            value = getattr(self, arg)

            if expected_type is int:
                if not isinstance(value, (int, np.integer)):
                    raise TypeError(
                        f"AstrophysicalScene attribute {arg} should be an integer"
                    )
            elif expected_type in ALL_UNITS:
                if not isinstance(value, u.Quantity):
                    raise TypeError(
                        f"AstrophysicalScene attribute {arg} should be a Quantity"
                    )
                if not value.unit.is_equivalent(expected_type):
                    raise ValueError(
                        f"AstrophysicalScene attribute {arg} has incorrect units"
                    )
            else:
                raise ValueError(f"Unexpected type specification for {arg}")

            # Additional check for numerical values
            if isinstance(value, u.Quantity):
                if not np.issubdtype(value.value.dtype, np.number):
                    raise TypeError(
                        f"AstrophysicalScene attribute {arg} should contain numerical values"
                    )
            elif not np.issubdtype(type(value), np.number):
                raise TypeError(
                    f"AstrophysicalScene attribute {arg} should be a number"
                )
