import numpy as np
from astropy import units as u
from astropy.constants import h, c, k_B

def planck_lambda(wavelength, temperature):
    """
    Calculate blackbody spectral radiance at a given wavelength and temperature.
    
    Parameters:
      wavelength : astropy Quantity
          Wavelength at which to compute the radiance (e.g., 10*u.um).
      temperature : astropy Quantity
          Blackbody temperature (e.g., 5778*u.K).
    
    Returns:
      intensity : astropy Quantity
          Spectral radiance in units of W/(m²·µm·sr).
    """
    # Calculate the exponent (ensuring it is dimensionless)
    exponent = (h * c) / (wavelength * k_B * temperature)
    
    # Compute Planck's law: intensity in W/(m²·m·sr)
    intensity = (2 * h * c**2 / wavelength**5) / (np.exp(exponent.value) - 1)
    
    # Convert from per meter to per micrometer
    return intensity.to(u.W / (u.m**2 * u.um * u.sr))

# Example usage:
wave = 10 * u.um
temp = 5778 * u.K
print(planck_lambda(wave, temp))

