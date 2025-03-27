import astropy.units as u
import sys

# Basic units
LENGTH = u.m
TIME = u.s
MASS = u.kg
TEMPERATURE = u.K
ANGLE = u.rad
WAVELENGTH = u.um
FREQUENCY = u.Hz
ENERGY = u.J
POWER = u.W
PHOTON_COUNT = u.photon

# Astronomical units
DISTANCE = u.pc
LUMINOSITY = u.L_sun
MAGNITUDE = u.mag
FLUX = u.W / u.m**2
FLUX_DENSITY = u.W / (u.m**2 * u.Hz)

# Angular units
ARCSEC = u.arcsec
MAS = u.mas
DEG = u.deg
INV_SQUARE_ARCSEC = u.def_unit("arcsec^-2", u.arcsec**-2)

# Time units
SECOND = u.s
MINUTE = u.min
HOUR = u.hour
DAY = u.day
YEAR = u.year

# Custom units
READ = u.def_unit("read", doc="Detector read")
FRAME = u.def_unit("frame", doc="Detector frame")

ZODI = u.def_unit(
    "zodi", doc="Unit of zodiacal light intensity", represents=u.dimensionless_unscaled
)


# Spectral units
SPECTRAL_FLUX_DENSITY_CGS = u.erg / (u.cm**2 * u.s * u.Hz)
PHOTON_FLUX_DENSITY = PHOTON_COUNT / (u.cm**2 * u.s * u.nm)
PHOTON_SPECTRAL_RADIANCE = PHOTON_COUNT / (u.cm**2 * u.s * u.nm * u.arcsec**2)
SURFACE_BRIGHTNESS = u.mag / u.arcsec**2
SPECTRAL_RADIANCE = u.W / (u.m**2 * u.sr * u.um)
SPECTRAL_RADIANCE_CGS = u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.nm)

# Dimensionless units
DIMENSIONLESS = u.dimensionless_unscaled

# Telescope-specific units
LAMBDA_D = u.def_unit(
    "λ/D",
    doc="Dimensionless unit of angular resolution in terms of wavelength over telescope diameter",
    represents=u.dimensionless_unscaled,
)


PIXEL = u.pix
ELECTRON = u.electron

# Derived units
DARK_CURRENT = ELECTRON / (PIXEL * TIME)
READ_NOISE = ELECTRON / (PIXEL * READ)
READ_TIME = TIME
CIC = ELECTRON / (PIXEL * FRAME)  # from Chris 2019 paper
QE = ELECTRON / PHOTON_COUNT


def lambda_d_to_radians(
    value_lod: u.Quantity, wavelength: u.Quantity, diameter: u.Quantity
) -> u.Quantity:
    """
    Convert a λ/D value to radians.

    Parameters:
    -----------
    value_lod : Quantity
        The λ/D value (dimensionless)
    wavelength : Quantity
        The wavelength
    diameter : Quantity
        The telescope diameter

    Returns:
    --------
    Quantity
        The angle in radians
    """
    return (value_lod * wavelength / diameter).to(
        u.rad, equivalencies=u.dimensionless_angles()
    )


def radians_to_lambda_d(
    angle: u.Quantity, wavelength: u.Quantity, diameter: u.Quantity
) -> u.Quantity:
    """
    Convert an angle in radians to λ/D.

    Parameters:
    -----------
    angle : Quantity
        The angle in radians
    wavelength : Quantity
        The wavelength
    diameter : Quantity
        The telescope diameter

    Returns:
    --------
    Quantity
        The value in λ/D (dimensionless)
    """
    return (angle * diameter / wavelength).to(
        LAMBDA_D, equivalencies=u.dimensionless_angles()
    )


def lambda_d_to_arcsec(
    value_lod: u.Quantity, wavelength: u.Quantity, diameter: u.Quantity
) -> u.Quantity:
    """
    Convert λ/D to arcseconds.

    Parameters:
    -----------
    value_lod : Quantity
        The λ/D value (dimensionless)
    wavelength : Quantity
        Wavelength
    diameter : Quantity
        Telescope diameter

    Returns:
    --------
    Quantity
        Angular size in arcseconds
    """
    return lambda_d_to_radians(value_lod, wavelength, diameter).to(u.arcsec)


def arcsec_to_lambda_d(
    angle: u.Quantity, wavelength: u.Quantity, diameter: u.Quantity
) -> u.Quantity:
    """
    Convert arcseconds to λ/D.

    Parameters:
    -----------
    angle : Quantity
        Angle in arcseconds
    wavelength : Quantity
        Wavelength
    diameter : Quantity
        Telescope diameter

    Returns:
    --------
    Quantity
        Angular size in λ/D
    """
    return radians_to_lambda_d(angle.to(u.rad), wavelength, diameter)


def arcsec_to_au(angle: u.Quantity, distance: u.Quantity) -> u.Quantity:
    """
    Convert an angle in arcseconds to a distance in AU, given a distance in parsecs.

    Parameters:
    angle (Quantity): The angle in arcseconds
    distance (Quantity): The distance in parsecs

    Returns:
    Quantity: The corresponding distance in AU
    """
    return (angle.to(u.radian).value * distance).to(u.au)


# Automatically generate ALL_UNITS list
ALL_UNITS = [
    obj
    for name, obj in sys.modules[__name__].__dict__.items()
    if isinstance(obj, (u.UnitBase, u.CompositeUnit, u.IrreducibleUnit))
    and name.isupper()  # Only include uppercase variables (assuming they are constants)
]
