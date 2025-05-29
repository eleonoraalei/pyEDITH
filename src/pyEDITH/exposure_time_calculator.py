from typing import Tuple
import numpy as np
from pyEDITH import AstrophysicalScene, Observation, Observatory
from pyEDITH.components.coronagraphs import CoronagraphYIP, ToyModelCoronagraph
import astropy.constants as c
import astropy.units as u
from astropy.modeling import models
from .units import *
from . import utils
import pickle


def calculate_CRp(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    Fp_over_Fs: u.Quantity,
    area: u.Quantity,
    Upsilon: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the planet count rate.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point. [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux. [dimensionless]
    Fp_over_Fs : u.Quantity
        Planet flux relative to star. [dimensionless]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    Upsilon : u.Quantity
        Core throughput of the coronagraph. [dimensionless]
    throughput : u.Quantity
        Throughput of the system (includes QE). [electrons/photons]
    dlambda : u.Quantity
        Bandwidth. [um]
    nchannels : int
        Number of channels.

    Returns
    -------
    u.Quantity
        Planet count rate. [electrons / s]

    Notes
    -----
    PLANET COUNT RATE

    CRp=F_0 * F_{star} * 10^{-0.4 Delta mag_{obs}} A Upsilon T Delta lambda

    which simplifies as

    CRp=F_0*F_{star}*Fp_0 *A Upsilon T Delta lambda

    in AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm * nchannels
    Fs_over_F0 = 10**(-0.4 * magstar)
    CRpfactor = Fs_over_F0 * FATDL
    tempCRpfactor = Fp_over_Fs[iplanetpistartnp] * CRpfactor
    CRp = tempCRpfactor * photometric_aperture_throughput[index2]
    """
    return (
        F0 * Fs_over_F0 * Fp_over_Fs * area * Upsilon * throughput * dlambda * nchannels
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbs(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    Istar: u.Quantity,
    area: u.Quantity,
    pixscale: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the stellar leakage count rate.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point. [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux. [dimensionless]
    Istar : u.Quantity
        Stellar intensity at the given pixel. [dimensionless]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    pixscale : u.Quantity
        Pixel scale of the detector. [lambda/D]
    throughput : u.Quantity
        Throughput of the system (includes QE). [electrons/photons]
    dlambda : u.Quantity
        Bandwidth. [um]
    nchannels : int
        Number of channels.

    Returns
    -------
    u.Quantity
        Stellar leakage count rate. [electrons / s]


    Notes
    -----
    THEORY: STELLAR LEAKAGE

    CRbs = F_0 * 10^{-0.4m_lambda} * zeta * PSF_{peak}
            * Omega * A * T * Deltalambda

    This simplifies as
    CRbs = F_0 * F_{star} *(zeta * PSF_{peak}) * A * Omega * T * Deltalambda

    IN AYO:
    Fs_over_F0 = pow((double) 10., -0.4*magstar);
    FATDL = F0 * A_cm * throughput * deltalambda_nm * nchannels;
    CRbsfactor = Fs_over_F0 * oneopixscale2 * FATDL;
    -- DETECTOR:
       det_CRbs = CRbsfactor * det_Istar;
       +++ NOTE: Later used to calculate photon counting time +++
    -- ETC:
       tempCRbsfactor = CRbsfactor * Istar_interp[index];
       +++ NOTE: Later added add together into tempCRbfactor+++
       +++ THEN: CRb = tempCRbfactor * omega_lod[index2]; +++
    """
    return (
        F0
        * Fs_over_F0
        * Istar
        * area
        * throughput
        * dlambda
        * nchannels
        / (pixscale**2)
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbz(
    F0: u.Quantity,
    Fzodi: u.Quantity,
    lod_arcsec: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the local zodiacal light count rate.

     Parameters
    ----------
    F0 : u.Quantity
        Flux zero point. [photons / (s * cm^2 * nm)]
    Fzodi : u.Quantity
        Zodiacal light flux. [dimensionless]
    lod_arcsec : u.Quantity
        Lambda/D in arcseconds. [arcsec]
    skytrans : u.Quantity
        Sky transmission. [dimensionless]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE). [electrons/photons]
    dlambda : u.Quantity
        Bandwidth. [um]
    nchannels : int
        Number of channels.

    Returns
    -------
    u.Quantity
        Local zodiacal light count rate. [electrons / s]

    Notes
    -----
    THEORY: LOCAL ZODI LEAKAGE

    CRbz=F_0* 10^{-0.4z}* Omega A T Delta lambda

    IN AYO:
    CRbzfactor = Fzodi * lod_arcsec2 * FATDL;
    FATDL = F0 * A_cm * throughput * deltalambda_nm * nchannels;
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    lod_arcsec2 = lod_arcsec * lod_arcsec
    -- DETECTOR:
        det_CRbz = CRbzfactor * det_skytrans;
         +++ NOTE: Later used to calculate photon counting time +++
    -- ETC:
        tempCRbzfactor = CRbzfactor * skytrans[index];
        +++NOTE: Later added together into tempCRbfactor+++
        +++ THEN: CRb = tempCRbfactor * omega_lod[index2]; +++
    """
    return (
        F0 * Fzodi * skytrans * area * throughput * dlambda * nchannels * lod_arcsec**2
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbez(
    F0: u.Quantity,
    Fexozodi: u.Quantity,
    lod_arcsec: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
    dist: u.Quantity,
    sp: u.Quantity,
) -> u.Quantity:
    """
    Calculate the exozodiacal light count rate.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point. [photons / (s * cm^2 * nm)]
    Fexozodi : u.Quantity
        Exozodiacal light flux. [dimensionless]
    lod_arcsec : u.Quantity
        Lambda/D in arcseconds. [arcsec]
    skytrans : u.Quantity
        Sky transmission. [dimensionless]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE). [electrons/photons]
    dlambda : u.Quantity
        Bandwidth. [um]
    nchannels : int
        Number of channels.
    dist : u.Quantity
        Distance to the star. [pc]
    sp : u.Quantity
        Separation of the planet. [arcsec]

    Returns
    -------
    u.Quantity
        Exozodiacal light count rate. [electrons / s]

    Notes
    -----
    THEORY: EXOZODI LEAKAGE
    CRbez=F_0 * n * 10^{-0.4mag_{exozodi}} * Omega * A * T * Delta lambda

    IN AYO:
    CRbezfactor = Fexozodi * lod_arcsec2 * FATDL / (dist*dist);
    FATDL = F0 * A_cm * throughput * deltalambda_nm * nchannels
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    lod_arcsec2 = lod_arcsec * lod_arcsec
    -- DETECTOR:
        det_CRbez = CRbezfactor * det_skytrans / (det_sep*det_sep);
        ++ NOTE: Later used to calculate photon counting time +++
    -- ETC:
        tempCRbezfactor = CRbezfactor * skytrans[index] / (sp[iplanetpistartnp]*sp[iplanetpistartnp]);
        +++NOTE: Later added together into tempCRbfactor+++
        +++ THEN: CRb = tempCRbfactor * omega_lod[index2]; +++

    Chris Stark Mar 2025: The flux from the exozodi gets scaled as (1 AU / sp_AU)^2.
    This is because we define exozodi in terms of a surface density at 1 AU from a
    solar twin, not a surface brightness. So if at 10 pc 1 zodi of exozodi has a
    surface brightness of X at 1 AU, at 2 AU it would have 1/4th the surface brightness
    to account for the 1/r^2 illumination factor. I.e., planets that are more distant
    from their host stars reside in fainter exozodi.

    """
    # Calculate Fexozodi at the separation (scale the value of Fexozodi at 1 AU
    # to the separation in AU)
    scaling_factor = u.AU / arcsec_to_au(sp, dist)
    return (
        F0
        * (Fexozodi * scaling_factor**2)
        * skytrans
        * area
        * throughput
        * dlambda
        * nchannels
        * lod_arcsec**2
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )  # this is to simplify the arcsec^2/arcsec^2 that somehow does not simplify by itself


def calculate_CRbbin(
    F0: u.Quantity,
    Fbinary: u.Quantity,
    skytrans: u.Quantity,
    area: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
) -> u.Quantity:
    """
    Calculate the count rate from neighboring stars.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point. [photons / (s * cm^2 * nm)]
    Fbinary : u.Quantity
        Flux from neighboring stars. [dimensionless]
    skytrans : u.Quantity
        Sky transmission. [dimensionless]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    throughput : u.Quantity
        Throughput of the system (includes QE). [electrons/photons]
    dlambda : u.Quantity
        Bandwidth. [um]
    nchannels : int
        Number of channels.

    Returns
    -------
    u.Quantity
        Count rate from neighboring stars. [electrons / s]
    Notes
    -----
    THEORY: NEIGHBORING STARS LEAKAGE

    CRbbin=F_0* 10^{-0.4mag_binary}* Omega A T Delta lambda

    IN AYO:
    CRbbinfactor = Fbinary * FATDL;
    FATDL = F0 * A_cm * throughput * deltalambda_nm * nchannels
    -- DETECTOR:
        det_CRbbin = CRbbinfactor * det_skytrans;
        ++ NOTE: Later used to calculate photon counting time +++

    -- ETC:
        tempCRbbinfactor = CRbbinfactor * skytrans[index];
        +++ NOTE: Later added together into tempCRbfactor+++
        +++ THEN: CRb = tempCRbfactor * omega_lod[index2]; +++

    """
    return (F0 * Fbinary * skytrans * area * throughput * dlambda * nchannels).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRbth(
    lam: u.Quantity,
    area: u.Quantity,
    dlambda: u.Quantity,
    temp: u.Quantity,
    lod_rad: u.Quantity,
    emis: u.Quantity,
    QE: u.Quantity,
    dQE: u.Quantity,
) -> u.Quantity:
    """
    Calculate background thermal count rate.

     Parameters
    ----------
    lam : u.Quantity
        Wavelength of observation. [um]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    dlambda : u.Quantity
        Bandwidth. [um]
    temp : u.Quantity
        Telescope mirror temperature. [K]
    lod_rad : u.Quantity
        Lambda/D in radians. [rad]
    emis : u.Quantity
        Effective emissivity for the observing system. [dimensionless]
    QE : u.Quantity
        Quantum efficiency. [electron/photon]
    dQE : u.Quantity
        Effective QE due to degradation. [dimensionless factor to multiply to QE]
    Returns
    -------
    u.Quantity
        Count rate from thermal background. [electrons / s]

    Notes
    -----

    IN AYO:
    Blambda = calcBlambda(temperature, lambda);
    CRbthermalfactor = Blambda * deltalambda_nm * A_cm * (lod_rad * lod_rad) * epswarmTrcold * QE;
    -- DETECTOR:
        det_CRbthermal = CRbthermalfactor * det_omega_lod;
        ++ NOTE: Later used to calculate photon counting time +++

    -- ETC:
        tempCRbthermalfactor = CRbthermalfactor;
        +++ NOTE: Later added together into tempCRbfactor+++
        +++ THEN: CRb = tempCRbfactor * omega_lod[index2]; +++
    """
    # Calculate blackbody radiation
    bb = models.BlackBody(
        temperature=temp, scale=1 * u.erg / (u.cm**2 * u.AA * u.s * u.sr)
    )
    Blambda_energy = bb(lam)

    # Convert to photon spectral radiance
    Blambda_photon = (Blambda_energy).to(
        u.photon / (u.cm**2 * u.nm * u.s * u.sr), equivalencies=u.spectral_density(lam)
    )

    # Calculate thermal background count rate
    return (Blambda_photon * dlambda * area * (lod_rad * lod_rad) * emis * QE * dQE).to(
        u.electron / u.s
    )

    # exp_power = c.h * c.c / lam / c.k_B / temp
    # Bsys = 2 * c.h * c.c**2 / (lam**5 * (np.exp(exp_power) - 1))

    # Bsys = Bsys.to(u.W / u.m**2 / u.um) / u.sr

    # # angular area of photometric aperture
    # Omega = np.pi * lod_arcsec**2  # in arcsec**2
    # Omega = Omega.to(u.sr)

    # photon_energy = c.h * c.c / lam
    # photon_energy = photon_energy.to(u.J) / u.photon

    # return (Bsys * emis * Omega * area * dlambda / photon_energy).to(u.photon / u.s)


def calculate_t_photon_count(
    det_npix: u.Quantity,
    det_CR: u.Quantity,
) -> u.Quantity:
    """
    Calculate the photon counting time.

    Parameters
    ----------
    det_npix : u.Quantity
        Number of detector pixels [pix]
    det_CR : u.Quantity
        Detector count rate. [photons / s]


    Returns
    -------
    u.Quantity
        Photon counting time i.e. average time to detect one photon per pixel. [s * pixel / ph ]
    Notes
    -----

    # According to Bernie Rauscher:
    # effective_dark_current = dark_current - f * cic * (1+W_-1[q/e])^-1.
    # If q = 0.99, (1+W_-1[q/e])^-1 = -6.73 such that
    # effective_dark_current = dark_current + f * cic * 6.73,
    # where f is the brightest pixel you care about in counts s^-1

    From Stark 2019:
    q is Geiger efficiency = quantifies the probability that one or fewer
    photons arrive during a frame. [units: photons/frame?]

    t=-1/CRsat*{1+W_-1[-q/e]}

    CRsat = count rate of the brightest pixel for which we wish to achieve a given q.
            -> WE ASSUME THAT IT IS det_CR I.E. THE NOISE AROUND THE IWA
               CALCULATED EARLIER [electron / s]

    If q = 0.99, (1+W_-1[q/e])^-1 = -6.73, so
    t = -1/CRsat / -6.73 = 1/(CRsat*6.73) --> GOAL OF THIS FUNCTION

    Which eventually becomes (in CRd):
    effective_dark_current = dark_current - cic * CRsat (1+W_-1[q/e])^-1 =
                            dark_current + cic/t = dark_current + cic*CRsat*6.73

          UNITS:            [electrons/pix/s] +[electrons/pix/frame]/[s/frame]

    So this means that t should have [s/frame] units
    """
    counts_per_second_per_pixel = det_CR / det_npix  #  electron / s / pix
    # NOTE: I am just extrapolating that 6.73 has units [pix*frame/electron]
    t_photon_count = 1.0 / (
        6.73 * PIXEL * FRAME / ELECTRON * counts_per_second_per_pixel
    )
    return t_photon_count


def calculate_CRbd(
    det_npix: u.Quantity,
    det_DC: u.Quantity,
    det_RN: u.Quantity,
    det_tread: u.Quantity,
    det_CIC: u.Quantity,
    t_photon_count: u.Quantity,
) -> u.Quantity:
    """
    Calculate the detector noise count rate.

    Parameters
    ----------
    det_npix : u.Quantity
        Number of detector pixels [pix]
    det_DC : u.Quantity
        Dark current. [electron / pix / s]
    det_RN : u.Quantity
        Read noise. [electron / pix / read]
    det_tread : u.Quantity
        Read time. [s]
    det_CIC : u.Quantity
        Clock-induced charge. [electron / pix / photon]
    t_photon_count : u.Quantity
        Photon counting time. [pix * s / electron]
    Returns
    -------
    u.Quantity
        Detector noise count rate. [photons / s]

    Notes
    -----
    DETECTOR NOISE

    CRbd = n_{pix}(xi +RN^2/tau_{exposure}+CIC/t_{photon_count})
        = npix (xi+RN^2/tau_{exposure}+ 6.73*CRsat*CIC

    IN AYO:
    CRbdfactor = det_DC + det_RN * det_RN/det_tread + det_CIC / t_photon_count;
    -- DETECTOR:
    N/A (calculated directly in ETC)
    -- ETC:
    det_npix = det_npix_multiplier * (omega_lod[index2] * oneodetpixscale_lod2) * nchannels;
    CRbd = CRbdfactor * det_npix;
    +++NOTE: Later added to CRb+++
    +++ THEN: CRb += CRbd; +++

    """
    # Using the variance of the read noise but keeping the same units as det_RN alone.
    read_noise_variance = det_RN * det_RN.value
    return (
        (det_DC + read_noise_variance / det_tread + det_CIC / t_photon_count) * det_npix
    ).to(
        u.electron / (u.s),
        equivalencies=u.equivalencies.dimensionless_angles(),
    )


def calculate_CRnf(
    F0: u.Quantity,
    Fs_over_F0: u.Quantity,
    area: u.Quantity,
    pixscale: u.Quantity,
    throughput: u.Quantity,
    dlambda: u.Quantity,
    nchannels: int,
    SNR: u.Quantity,
    noisefloor: u.Quantity,
) -> u.Quantity:
    """
    Calculate the noise floor count rate.

    Parameters
    ----------
    F0 : u.Quantity
        Flux zero point. [photons / (s * cm^2 * nm)]
    Fs_over_F0 : u.Quantity
        Stellar flux. [dimensionless]
    area : u.Quantity
        Collecting area of the telescope. [cm^2]
    pixscale : u.Quantity
        Pixel scale of the detector. [lambda/D]
    throughput : u.Quantity
        Throughput of the system. [dimensionless]
    dlambda : u.Quantity
        Bandwidth. [um]
    nchannels : int
        Number of channels.
    SNR : float
        Signal-to-noise ratio.
    noisefloor : u.Quantity
        Noise floor level. [dimensionless]

    Returns
    -------
    u.Quantity
        Noise floor count rate. [photons / s]

    Notes
    -----
    Calculate the count rate of the noise floor.
    This should be the stddev (over the "noise region") of
    the difference of the photometric aperture-integrated
    stellar PSFs. The photometric aperture integration, and
    stddev of that have been calculated prior to the call to
    this function, and was then divided by the number of pixels
    in the photometric aperture. So here we calculate noise
    using the same method as the leaked starlight.
    The "noisefloor" array is equal to

    stddev(integral(Istar1,dphotometric_ap) - integral(Istar2,dphotometric_ap))
                            / (omega/(npix*npix))

    Reminder:
    self.noisefloor =parameters['noisefloor_factor']*self.contrast
    # = 1 sigma systematic noise floor expressed as a contrast
    (uniform over dark hole and unitless) # scalar

    in AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm * nchannels
    CRbsfactor = Fs_over_F0 * oneopixscale2 * FATDL  # for stellar leakage count
    rate calculation
    Fs_over_F0 = 10**(-0.4 * magstar)
        tempCRnffactor = SNR * CRbsfactor * noisefloor_interp[index];


    # NOTE: Since Omega is not used when calculating the detector
    # noise components, this multiplication is done outside the
    # function when needed. i.e.
    # CRnoisefloor = tempCRnffactor * omega_lod[index2];

    """
    return (
        SNR
        * (F0 * Fs_over_F0 * area * throughput * dlambda * nchannels / (pixscale**2))
        * noisefloor
    )


def measure_coronagraph_performance_at_IWA(
    # psf_trunc_ratio: u.Quantity, # commenting out so that we can use either psf_trunc_ratio or photometric_aperture_radius for omega calculation. This is not used anyway in this function.
    photometric_aperture_throughput: u.Quantity,
    Istar_interp: u.Quantity,
    skytrans: u.Quantity,
    omega_lod: u.Quantity,
    npix: int,
    xcenter: u.Quantity,
    ycenter: u.Quantity,
    oneopixscale_arcsec: u.Quantity,
) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    Measure the performance of the coronagraph at the Inner Working Angle (IWA).

    This function determines the IWA and calculates various parameters at that point:

    1. Finds the psf_trunc_ratio closest to 0.3
    2. Determines the IWA by finding where photometric_aperture_throughput falls to half its maximum value
    3. Calculates maximum values of Istar, skytrans, photometric_aperture_throughput, and omega_lod in a 2-pixel annulus at the IWA


     Parameters
    ----------
    psf_trunc_ratio : u.Quantity
        PSF truncation ratios. [dimensionless]
    photometric_aperture_throughput : u.Quantity
        Photometric aperture fractions. [dimensionless]
    Istar_interp : u.Quantity
        Interpolated stellar intensity. [dimensionless]
    skytrans : u.Quantity
        Sky transmission. [dimensionless]
    omega_lod : u.Quantity
        Solid angle of photometric aperture. [lambda/D]^2
    npix : int
        Number of pixels in each dimension.
    xcenter : u.Quantity
        X-coordinate of the center. [pixel]
    ycenter : u.Quantity
        Y-coordinate of the center. [pixel]
    oneopixscale_arcsec : u.Quantity
        Inverse of pixel scale. [1/arcsec]

    Returns
    -------
    Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]
        det_sep_pix: Separation at the IWA. [pixel]
        det_sep: Separation at the IWA. [arcsec]
        det_Istar: Maximum stellar intensity at the IWA. [dimensionless]
        det_skytrans: Maximum sky transmission at the IWA. [dimensionless]
        det_photometric_aperture_throughput: Maximum photometric aperture fraction at the IWA. [dimensionless]
        det_omega_lod: Solid angle corresponding to max photometric_aperture_throughput at the IWA. [lambda/D]^2

    """

    # Find psf_trunc_ratio closest to 0.3
    # Commenting this out for now since this is not used.
    # bestiratio = np.argmin(
    #     np.abs(psf_trunc_ratio - 0.3)
    # )  # NOT USED, in EDITH only one psf_trunc_ratio
    bestiratio = 0  # len = 1 array, so only one index to choose

    # Find maximum photometric_aperture_throughput in first half of image
    maxphotometric_aperture_throughput = np.max(
        photometric_aperture_throughput[: npix // 2, int(ycenter.value), bestiratio]
    )

    # Find IWA = where photometric_aperture_throughput is half the value of the maximum
    row = photometric_aperture_throughput[:, int(ycenter.value), bestiratio]
    iwa_index = np.where(
        row[: int(xcenter.value)] > 0.5 * maxphotometric_aperture_throughput
    )[0][-1]
    det_sep_pix = abs((iwa_index + 0.5) - xcenter.value) * PIXEL
    det_sep = det_sep_pix / oneopixscale_arcsec  # translates to arcsec

    # Calculate max values in 2-pixel annulus at det_sep
    y, x = np.ogrid[:npix, :npix]
    dist_from_center = (
        np.sqrt((x - xcenter.value + 0.5) ** 2 + (y - ycenter.value + 0.5) ** 2)
    ) * PIXEL

    mask = np.abs(dist_from_center.value - det_sep_pix.value) < 2
    # NOTE: We use .value because we want indices

    det_Istar = np.max(Istar_interp[mask]) * DIMENSIONLESS
    det_skytrans = np.max(skytrans[mask]) * DIMENSIONLESS

    photometric_aperture_throughput_masked = photometric_aperture_throughput[
        :, :, bestiratio
    ][mask]
    det_photometric_aperture_throughput = (
        np.max(photometric_aperture_throughput_masked) * DIMENSIONLESS
    )
    det_omega_lod = omega_lod[:, :, bestiratio][mask][
        np.argmax(photometric_aperture_throughput_masked)
    ]

    return (
        det_sep_pix,
        det_sep,
        det_Istar,
        det_skytrans,
        det_photometric_aperture_throughput,
        det_omega_lod,
    )


def calculate_exposure_time_or_snr(
    observation: Observation,
    scene: AstrophysicalScene,
    observatory: Observatory,
    verbose: bool,
    ETC_validation: bool = False,
    mode: str = "exposure_time",
) -> None:
    """
    Calculate the exposure time for each target and wavelength.
    This function calculates the exposure time for each target star and wavelength,
    taking into account various noise sources and coronagraph performance metrics.
    It iterates through targets, wavelengths, orbits, and phases to compute
    the required exposure time for planet detection.

    Parameters
    ----------
    observation : Observation
        Object containing observation parameters.
    scene : AstrophysicalScene
        Object containing scene parameters.
    observatory: Observatory
        Object containing observatory parameters.
    verbose : boolean
        Verbose flag.
    """

    # Check modes
    if mode not in ["exposure_time", "signal_to_noise"]:
        raise ValueError("Invalid mode. Use 'exposure_time' or 'signal_to_noise'.")

    observation.validation_variables = {}
    observation.photon_counts = {
        "CRp": np.empty(observation.nlambd),
        "CRbs": np.empty(observation.nlambd),
        "CRbz": np.empty(observation.nlambd),
        "CRbez": np.empty(observation.nlambd),
        "CRbbin": np.empty(observation.nlambd),
        "CRbth": np.empty(observation.nlambd),
        "CRbd": np.empty(observation.nlambd),
        "CRnf": np.empty(observation.nlambd),
        "CRb": np.empty(observation.nlambd),
    }

    for ilambd in range(observation.nlambd):

        # Take the lesser of the desired bandwidth
        # and what coronagraph allows
        if observatory.observing_mode == "IMAGER":
            deltalambda_nm = (
                np.min(
                    [
                        (observation.wavelength[ilambd].to(u.nm).value)
                        / observatory.coronagraph.coronagraph_spectral_resolution,
                        observatory.coronagraph.bandwidth
                        * (observation.wavelength[ilambd].to(u.nm).value),
                    ]
                )
                * u.nm
            )  # nanometers
            if (
                observatory.coronagraph.bandwidth
                * observation.wavelength[ilambd].to(u.nm).value
                >= observation.wavelength[ilambd].to(u.nm).value
                / observatory.coronagraph.coronagraph_spectral_resolution
            ):
                print(
                    "WARNING: Bandwidth larger than what the coronagraph allows. Selecting widest possible bandwidth..."
                )
        elif observatory.observing_mode == "IFS":
            # the effective bandwidth is the width of the spectral element 
            deltalambda_nm = observation.delta_wavelength[ilambd].to(u.nm)
        else:
            raise ValueError("Invalid observation mode. Choose 'IMAGER' or 'IFS'.")

        # Calculate λ/D (dimensionless)
        lod = 1 * LAMBDA_D

        # Convert to radians
        # NOTE: using LENGTH here ensures that if we change the value of the unit,
        # this is still dimensionless
        lod_rad = lambda_d_to_radians(
            lod,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )

        # Convert to arcseconds
        lod_arcsec = lod_rad.to(u.arcsec)

        area_cm2 = observatory.telescope.Area.to(u.cm**2)

        detpixscale_lod = arcsec_to_lambda_d(
            observatory.detector.pixscale_mas.to(u.arcsec),
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )  # LAMBDA_D units

        stellar_diam_lod = arcsec_to_lambda_d(
            scene.stellar_angular_diameter_arcsec,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )  # LAMBDA_D units

        """
        WE DO NOT INTERPOLATE ANYMORE, INTERPOLATION IS DONE WITHIN YIPPY 
        # Interpolate Istar, noisefloor based on angular diameter
        # of the star (depends on the target). It reduces dimensionality
        # from 3D arrays [npix,npix,angdiam] to 2D arrays [npix,npix].
        # The interpolation is done based on the value of
        # stellar_diam_lod (dependence on istar)

        Istar_interp, noisefloor_interp = interpolate_arrays(
            observatory.coronagraph.Istar,
            observatory.coronagraph.noisefloor,
            observatory.coronagraph.npix,
            observatory.coronagraph.ndiams,
            stellar_diam_lod,
            observatory.coronagraph.angdiams,
        )
        """

        # Measure coronagraph performance at each IWA
        pixscale_rad = observatory.coronagraph.pixscale * lambda_d_to_radians(
            lod,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )  # going from LAMBDA_D to radians

        oneopixscale_arcsec = 1 * PIXEL / pixscale_rad.to(u.arcsec)

        # Measure coronagraph performance at each IWA
        (
            det_sep_pix,
            det_sep,
            det_Istar,
            det_skytrans,
            det_photometric_aperture_throughput,
            det_omega_lod,
        ) = measure_coronagraph_performance_at_IWA(
            # observation.psf_trunc_ratio, # this is no longer used in the function
            observatory.coronagraph.photometric_aperture_throughput,
            observatory.coronagraph.Istar,
            observatory.coronagraph.skytrans,
            observatory.coronagraph.omega_lod,
            observatory.coronagraph.npix,
            observatory.coronagraph.xcenter,
            observatory.coronagraph.ycenter,
            oneopixscale_arcsec,
        )

        if ETC_validation:
            print("Fixing det_npix for validation...")

            det_npix = observatory.detector.det_npix_input * PIXEL
        else:
            #  Calculate det_npix
            det_npix = (
                observatory.detector.npix_multiplier[ilambd]
                * det_omega_lod
                / (detpixscale_lod**2)
                * observatory.coronagraph.nchannels
            ) * PIXEL  # number of pixels in detector

        # Here we calculate detector noise, as it may depend on count rates
        # We don't know the count rates yet, so we make estimates based on
        # values near the IWA

        # Detector noise from signal itself (we budget for 10x
        # the planet count rate for the minimum detectable planet)

        det_CRp = calculate_CRp(
            scene.F0[ilambd],
            scene.Fs_over_F0[ilambd],
            10 * scene.Fp_min_over_Fs,
            area_cm2,
            det_photometric_aperture_throughput,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbs = calculate_CRbs(
            scene.F0[ilambd],
            scene.Fs_over_F0[ilambd],
            det_Istar,
            area_cm2,
            observatory.coronagraph.pixscale,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbz = calculate_CRbz(
            scene.F0[ilambd],
            scene.Fzodi_list[ilambd],
            lod_arcsec,
            det_skytrans,
            area_cm2,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbez = calculate_CRbez(
            scene.F0[ilambd],
            scene.Fexozodi_list[ilambd],
            lod_arcsec,
            det_skytrans,
            area_cm2,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
            scene.dist,
            det_sep,
        )

        det_CRbbin = calculate_CRbbin(
            scene.F0[ilambd],
            scene.Fbinary_list[ilambd],
            det_skytrans,
            area_cm2,
            observatory.total_throughput[ilambd],
            deltalambda_nm,
            observatory.coronagraph.nchannels,
        )

        det_CRbth = (
            calculate_CRbth(
                observation.wavelength[ilambd],
                area_cm2,
                deltalambda_nm,
                observatory.telescope.temperature,
                lod_rad,
                observatory.epswarmTrcold[ilambd],
                observatory.detector.QE[ilambd,],
                observatory.detector.dQE[ilambd,],
            )
            * det_omega_lod
        )

        det_CR = det_CRp + det_CRbs + det_CRbz + det_CRbez + det_CRbbin + det_CRbth

        # Calculate position of the planet in the image
        # (from l/D to pixel)
        ix = (
            scene.xp * oneopixscale_arcsec + observatory.coronagraph.xcenter
        ).value  # this is the "index" of the position in pixel, i.e. the number of the pixel where the planet is
        iy = (
            scene.yp * oneopixscale_arcsec + observatory.coronagraph.ycenter
        ).value  # this is the "index" of the position in pixel, i.e. the number of the pixel where the planet is

        # Calculate separation (from arcsec to l/D)
        sp_lod = arcsec_to_lambda_d(
            scene.separation,
            observation.wavelength[ilambd].to(LENGTH),
            observatory.telescope.diameter.to(LENGTH),
        )

        # If planet is within the boundaries of the observatory.coronagraph
        # simulation and hard IWA/OWA cutoffs...
        if (
            (ix >= 0)  # check that x pixel is positive
            and (
                ix < observatory.coronagraph.npix
            )  # check that it is less than the maximum pixel number
            and (iy >= 0)  # check that the y pixel is positive
            and (
                iy < observatory.coronagraph.npix
            )  # check that it is less than the maximum pixel number
            and (
                sp_lod > observatory.coronagraph.minimum_IWA
            )  # check that the separation in l/D is more than the minimum allowed IWA
            and (
                sp_lod < observatory.coronagraph.maximum_OWA
            )  # check that the separation in l/D is less than the maximum allowed OWA
        ):

            for iratio in np.arange(observatory.coronagraph.npsfratios):
                # First we just calculate CRp and CRnoisefloor
                # to see if CRp > CRnoisefloor

                # PLANET COUNT RATE CRP
                CRp = calculate_CRp(
                    scene.F0[ilambd],
                    scene.Fs_over_F0[ilambd],
                    scene.Fp_over_Fs[ilambd],
                    area_cm2,
                    observatory.coronagraph.photometric_aperture_throughput[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ],
                    observatory.total_throughput[ilambd],
                    deltalambda_nm,
                    observatory.coronagraph.nchannels,
                )
                observation.photon_counts["CRp"][ilambd] = CRp.value

                # NOISE FLOOR CRNF
                if mode == "exposure_time":
                    # NOISE FLOOR CRNF
                    CRnf = calculate_CRnf(
                        scene.F0[ilambd],
                        scene.Fs_over_F0[ilambd],
                        area_cm2,
                        observatory.coronagraph.pixscale,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                        observation.SNR[ilambd],
                        observatory.coronagraph.noisefloor[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                    )

                elif mode == "signal_to_noise":
                    #  NOTE THIS TIME THIS IS JUST THE NOISE
                    # FACTOR RATIO (i.e. we assume SNR =1 so
                    # that we can use it for the snr calculation later)

                    CRnf = calculate_CRnf(
                        scene.F0[ilambd],
                        scene.Fs_over_F0[ilambd],
                        area_cm2,
                        observatory.coronagraph.pixscale,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                        1,
                        observatory.coronagraph.noisefloor[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                    )

                # multiply by omega at that point
                CRnf *= observatory.coronagraph.omega_lod[
                    int(np.floor(iy)), int(np.floor(ix)), iratio
                ]
                observation.photon_counts["CRnf"][ilambd] = CRnf.value

                # NOTE: noisefloor_interp: technically the Y axis
                # is rows and the X axis is columns,
                # that is why they are inverted
                # NOTE: Evaluate if int(round(iy)) is better than
                # np.floor. Kept np.floor for consistency

                # Check if photometric aperture is large enough:
                if (
                    observatory.coronagraph.omega_lod[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ]
                    > detpixscale_lod**2
                ):

                    # (for exposure time mode) Check if it's above the noise floor
                    if mode == "exposure_time" and CRp <= CRnf:
                        print(
                            "WARNING: Count rate of the planet smaller than the noise floor. Hardcoded infinity results."
                        )

                        observation.exptime[ilambd] = np.inf
                        continue  # Skip to next iteration

                    # Calculate the rest of the background noise

                    # NOTE: WHEN CALCULATING THE COUNT RATES,
                    # WE NEED TO MULTIPLY BY OMEGA_LOD i.e.
                    # THE SOLID ANGLE OF THE PHOTOMETRIC APERTURE

                    # Calculate CRbs
                    CRbs = calculate_CRbs(
                        scene.F0[ilambd],
                        scene.Fs_over_F0[ilambd],
                        observatory.coronagraph.Istar[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.coronagraph.pixscale,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                    )
                    observation.photon_counts["CRbs"][ilambd] = (
                        CRbs.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbz
                    CRbz = calculate_CRbz(
                        scene.F0[ilambd],
                        scene.Fzodi_list[ilambd],
                        lod_arcsec,
                        observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                    )
                    observation.photon_counts["CRbz"][ilambd] = (
                        CRbz.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbez
                    CRbez = calculate_CRbez(
                        scene.F0[ilambd],
                        scene.Fexozodi_list[ilambd],
                        lod_arcsec,
                        observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                        scene.dist,
                        scene.separation,
                    )
                    observation.photon_counts["CRbez"][ilambd] = (
                        CRbez.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbbin
                    CRbbin = calculate_CRbbin(
                        scene.F0[ilambd],
                        scene.Fbinary_list[ilambd],
                        observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        area_cm2,
                        observatory.total_throughput[ilambd],
                        deltalambda_nm,
                        observatory.coronagraph.nchannels,
                    )
                    observation.photon_counts["CRbbin"][ilambd] = (
                        CRbbin.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # Calculate CRbd
                    t_photon_count = calculate_t_photon_count(
                        det_npix,
                        det_CR,
                    )
                    if ETC_validation:
                        print("Fixing t_photon_count for validation...")
                        # the ETC validation (Stark+2025) fixed the frame rate
                        # t_photon_count = 1 / (det_CRp.value) * SECOND / FRAME
                        t_photon_count = observatory.detector.t_photon_count_input

                    CRbd = calculate_CRbd(
                        det_npix,
                        observatory.detector.DC[ilambd],
                        observatory.detector.RN[ilambd],
                        observatory.detector.tread[ilambd],
                        observatory.detector.CIC[ilambd],
                        t_photon_count,
                    )

                    observation.photon_counts["CRbd"][ilambd] = CRbd.value

                    CRbth = calculate_CRbth(
                        observation.wavelength[ilambd],
                        area_cm2,
                        deltalambda_nm,
                        observatory.telescope.temperature,
                        lod_rad,
                        observatory.epswarmTrcold[ilambd],
                        observatory.detector.QE[ilambd,],
                        observatory.detector.dQE[ilambd,],
                    )
                    observation.photon_counts["CRbth"][ilambd] = (
                        CRbth.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), iratio
                        ].value
                    )

                    # TOTAL BACKGROUND NOISE
                    CRb = (
                        CRbs + CRbz + CRbez + CRbbin + CRbth
                    ) * observatory.coronagraph.omega_lod[
                        int(np.floor(iy)), int(np.floor(ix)), iratio
                    ]
                    observation.photon_counts["CRb"][ilambd] = CRb.value

                    # Add detector noise
                    CRb += CRbd

                    # EXPOSURE TIME
                    if mode == "exposure_time":
                        # count rate term
                        # NOTE this includes the systematic noise floor
                        # term a la Bijan Nemati
                        cp = (
                            (CRp + observation.CRb_multiplier * CRb)
                            / (CRp * CRp - CRnf * CRnf)
                            * u.electron
                        )

                        # UNITS:
                        # ([electron/s]+[electron/s])/([electron/s]^2+[electron/s]^2) =
                        # [s/electron]

                        # Calculate Exposure time
                        observation.exptime[ilambd] = (
                            observation.SNR[ilambd]
                            * observation.SNR[ilambd]
                            * cp
                            * observatory.telescope.toverhead_multi
                            + observatory.telescope.toverhead_fixed
                        )  # record exposure time with overheads

                        # UNITS:
                        # []^2*[s/electron]*[]+[s] == [s]
                        if observation.exptime[ilambd] < 0:
                            # time is past the systematic
                            # noise floor limit
                            observation.exptime[ilambd] = np.inf

                        if observation.exptime[ilambd] > observation.td_limit:
                            # treat as unobservable
                            # if beyond exposure time limit
                            observation.exptime[ilambd] = np.inf

                        if observatory.coronagraph.nrolls != 1:
                            # multiply by number of required rolls to
                            # achieve 360 deg coverage
                            # (after tlimit enforcement)
                            observation.exptime[
                                ilambd
                            ] *= observatory.coronagraph.nrolls
                    elif mode == "signal_to_noise":

                        # cp not used in this mode. Note: This will make the science time in
                        # validation variables be 0!
                        cp = 0
                        # SIGNAL-TO-NOISE
                        # time term
                        time_factors = (
                            observation.obstime * observatory.coronagraph.nrolls
                            - observatory.telescope.toverhead_fixed
                        ) / (
                            observatory.telescope.toverhead_multi
                            * ((CRp + observation.CRb_multiplier * CRb))
                        )

                        # UNITS:
                        # ([s]*[]-[s])/([electron/s]+[]*[electron/s])
                        # [s]/[electron/s]=[s^2/electron]

                        # Signal-to-noise
                        observation.fullsnr[ilambd] = (
                            np.sqrt(
                                (time_factors * CRp**2)
                                / (1 * ELECTRON + time_factors * CRnf**2)
                            )
                            * DIMENSIONLESS
                        )

                        # UNITS:
                        # ([s^2/electron]*[electron/s]^2)/([electron]+[s^2/electron]*[electron/s]^2)=
                        # [electron]/[electron] = []

                    # Store the variables of interest
                    observation.validation_variables[ilambd] = {
                        "F0": scene.F0[ilambd],
                        "magstar": scene.mag,
                        "Lstar": scene.Lstar,
                        "dist": scene.dist,
                        "D": observatory.telescope.diameter,
                        "A_cm": area_cm2,
                        "wavelength": observation.wavelength[ilambd].to(u.nm),
                        "deltalambda_nm": deltalambda_nm,
                        "snr": observation.SNR[ilambd],
                        "nzodis": scene.nzodis,
                        "toverhead_fixed": observatory.telescope.toverhead_fixed,
                        "toverhead_multi": observatory.telescope.toverhead_multi,
                        "det_DC": observatory.detector.DC[ilambd],
                        "det_RN": observatory.detector.RN[ilambd],
                        "det_CIC": observatory.detector.CIC[ilambd],
                        "det_tread": observatory.detector.tread[ilambd],
                        "det_pixscale_mas": observatory.detector.pixscale_mas,
                        "dQE": observatory.detector.dQE[ilambd],
                        "QE": observatory.detector.QE[ilambd],
                        "T_optical": observatory.optics_throughput[ilambd],
                        "Fs_over_F0": scene.Fs_over_F0[ilambd] * scene.F0[ilambd],
                        "Fp": scene.Fs_over_F0[ilambd]
                        * scene.F0[ilambd]
                        * scene.Fp_over_Fs[ilambd],
                        "Fzodi": scene.Fzodi_list[ilambd] * scene.F0[ilambd],
                        "Fexozodi": scene.Fexozodi_list[ilambd]
                        * scene.F0[ilambd]
                        / (scene.separation**2 * scene.dist**2),
                        "sp_lod": arcsec_to_lambda_d(
                            scene.separation,
                            observation.wavelength[ilambd],
                            observatory.telescope.diameter,
                        ),
                        "omega_lod": observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        # "throughput": observatory.total_throughput[ilambd],
                        "T_core or photometric_aperture_throughput": observatory.coronagraph.photometric_aperture_throughput[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "Istar": observatory.coronagraph.Istar[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        "Istar*oneopixscale2 in (l/D)^-2": observatory.coronagraph.Istar[
                            int(np.floor(iy)), int(np.floor(ix))
                        ]
                        * (1 / observatory.coronagraph.pixscale) ** 2,
                        # "contrast * offset PSF peak *oneopixscale2  in (l/D)^-2 (unused)": 0.025
                        # * observatory.coronagraph.TLyot
                        # * observatory.coronagraph.contrast
                        # * (1 / observatory.coronagraph.pixscale) ** 2,
                        "skytrans": observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ],
                        "skytrans*oneopixscale2  in (l/D)^-2": observatory.coronagraph.skytrans[
                            int(np.floor(iy)), int(np.floor(ix))
                        ]
                        * (1 / observatory.coronagraph.pixscale) ** 2,
                        "det_npix": det_npix,
                        "t_photon_count": t_photon_count,
                        "CRp": CRp,
                        "CRbs": CRbs
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbz": CRbz.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbez": CRbez.value
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbbin": CRbbin
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRbth": CRbth
                        * observatory.coronagraph.omega_lod[
                            int(np.floor(iy)), int(np.floor(ix)), 0
                        ],
                        "CRb": CRb,
                        "CRbd": CRbd,
                        "CRnf": CRnf,
                        "sciencetime": observation.SNR[ilambd]
                        * observation.SNR[ilambd]
                        * cp,
                        "exptime": observation.exptime[ilambd],
                    }

                else:
                    print(
                        "WARNING: Photometric aperture is not large enough. Hardcoded infinity results."
                    )
                    if mode == "exposure_time":
                        observation.exptime[ilambd] = np.inf
                    elif mode == "signal_to_noise":
                        observation.fullsnr[ilambd] = np.inf

        else:
            print(
                "WARNING: Planet outside OWA or inside IWA. Hardcoded infinity results."
            )
            if mode == "exposure_time":
                observation.exptime[ilambd] = np.inf
            elif mode == "signal_to_noise":
                observation.fullsnr[ilambd] = np.inf

        if verbose:
            utils.print_all_variables(
                observation,
                scene,
                observatory,
                deltalambda_nm,
                lod,
                lod_rad,
                lod_arcsec,
                area_cm2,
                detpixscale_lod,
                stellar_diam_lod,
                pixscale_rad,
                oneopixscale_arcsec,
                det_sep_pix,
                det_sep,
                det_Istar,
                det_skytrans,
                det_photometric_aperture_throughput,
                det_omega_lod,
                det_CRp,
                det_CRbs,
                det_CRbz,
                det_CRbez,
                det_CRbbin,
                det_CRbth,
                det_CR,
                ix,
                iy,
                sp_lod,
                CRp,
                CRnf,
                CRbs,
                CRbz,
                CRbez,
                CRbbin,
                t_photon_count,
                CRbd,
                CRbth,
                CRb,
                # cp,
            )
    # Save the photon counts for later analysis
    pickle.dump(observation.photon_counts, open("photon_counts.pk", "wb"))

    return
