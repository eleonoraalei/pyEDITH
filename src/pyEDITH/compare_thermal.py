import pyEDITH.exposure_time_calculator as pyetc
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# pyedith thermal background

lam = 0.5 * u.um #np.arange(0.5, 2, 1000) * u.um
tele_diam = 6.5*u.m
area = np.pi * (tele_diam/2)**2
dlambda = 0.2*lam # 20% bandpass
temp = 300 * u.K #K
emis = 1.
lod_arcsec = ((lam / tele_diam)*u.radian).to(u.arcsec)

Cbth_pyedith = pyetc.calculate_CRbth(lam, area, dlambda, temp, lod_arcsec, emis)   


print(f"pyEDITH thermal CR: {Cbth_pyedith}")


# AYO thermal
#CRbthermalfactor = Blambda * deltalambda_nm * A_cm * (lod_rad * lod_rad) * epswarmTrcold * QE 
# essentially: BB * wl_bin * tele_area * (lod_rad**2) * eps * QE

def calc_Blambda_chris(temp, lam):
    lam_cm = lam.to(u.cm).value # cm
    h = 6.6261e-27 #erg s
    c = 2.99792458e10 # cm s^-1
    k = 1.380622e-16 # erg K^-1
    
    const1 = 2.0*h*c*c
    const2 = h*c/k
    hc = h*c

    result = const1 / ((lam_cm*lam_cm*lam_cm*lam_cm*lam_cm) * (np.exp(const2 / (lam_cm*temp)) - 1.0))
    
    result /= (hc/lam_cm) # divide by energy of a photon to get photons cm^-3 s^-1 steradian^-1
    result /= 1e7 # convert to cm^-2 nm^-1 s^-1 steradian^-1
    return result


Blam_chris = calc_Blambda_chris(temp.value, lam)
deltalambda_nm = dlambda.to(u.nm).value # convert to nm
A_cm = area.to(u.cm**2).value
lod_rad = lod_arcsec.to(u.radian)
epswarmTrcold = emis
QE = 1.

rad2_to_sr = (1*u.arcsec**2) / (1*u.sr)


CRbthermalfactor = Blam_chris * deltalambda_nm * A_cm * (lod_rad * lod_rad) * epswarmTrcold * QE 


print("AYO thermal CR:", CRbthermalfactor)  


print("Off by a factor of", Cbth_pyedith / CRbthermalfactor)
# Note: AYO does not multiply omega by pi... this is wrong.