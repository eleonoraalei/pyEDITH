from typing import Tuple
import numpy as np
from pyEDITH import AstrophysicalScene, Observation, Observatory


def calculate_CRp(
    F0: float,
    Fstar: float,
    Fp0: float,
    area: float,
    Upsilon: float,
    throughput: float,
    dlambda: float,
) -> float:
    """
    Calculate the planet count rate.

    Parameters
    ----------
    F0 : float
        Flux zero point.
    Fstar : float
        Stellar flux.
    Fp0 : float
        Planet flux relative to star.
    area : float
        Collecting area of the telescope.
    Upsilon : float
        Core throughput of the coronagraph.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.

    Returns
    -------
    float
        Planet count rate.

    Notes
    -----
    PLANET COUNT RATE

    CRp=F_0 * F_{star} * 10^{-0.4 Delta mag_{obs}} A Upsilon T Delta lambda

    which simplifies as

    CRp=F_0*F_{star}*Fp_0 *A Upsilon T Delta lambda

    in AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm
    Fstar = 10**(-0.4 * magstar)
    CRpfactor = Fstar * FATDL
    tempCRpfactor = Fp0[iplanetpistartnp] * CRpfactor
    CRp = tempCRpfactor * photap_frac[index2]
    """
    return F0 * Fstar * Fp0 * area * Upsilon * throughput * dlambda


def calculate_CRbs(
    F0: float,
    Fstar: float,
    Istar: float,
    area: float,
    pixscale: float,
    throughput: float,
    dlambda: float,
) -> float:
    """
    Calculate the stellar leakage count rate.

    Parameters
    ----------
    F0 : float
        Flux zero point.
    Fstar : float
        Stellar flux.
    Istar : float
        Stellar intensity at the given pixel.
    area : float
        Collecting area of the telescope.
    pixscale : float
        Pixel scale of the detector.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.

    Returns
    -------
    float
        Stellar leakage count rate.

    Notes
    -----
    STELLAR LEAKAGE

    CRbs = F_0 * 10^{-0.4m_lambda} * zeta * PSF_{peak}
            * Omega * A * T * Deltalambda

    This simplifies as
    CRbs = F_0 * F_{star} *(zeta * PSF_{peak}) * A * Omega * T * Deltalambda

    in AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbsfactor = Fstar * oneopixscale2 * FATDL  # for stellar leakage
                                                # count rate calculation
    Fstar = 10**(-0.4 * magstar)
    tempCRbsfactor = CRbsfactor * Istar_interp[index]


    # NOTE: Since Omega is not used when calculating the detector
    # noise components, this multiplication is done outside the
    # function when needed. i.e.
    # CRbs = tempCRbsfactor * omega_lod[index2]

    """
    return F0 * Fstar * Istar * area * throughput * dlambda / (pixscale**2)


def calculate_CRbz(
    F0: float,
    Fzodi: float,
    lod_arcsec: float,
    skytrans: float,
    area: float,
    throughput: float,
    dlambda: float,
) -> float:
    """
    Calculate the local zodiacal light count rate.

    Parameters
    ----------
    F0 : float
        Flux zero point.
    Fzodi : float
        Zodiacal light flux.
    lod_arcsec : float
        Lambda/D in arcseconds.
    skytrans : float
        Sky transmission.
    area : float
        Collecting area of the telescope.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.

    Returns
    -------
    float
        Local zodiacal light count rate.

    Notes
    -----

    LOCAL ZODI LEAKAGE

    CRbz=F_0* 10^{-0.4z}* Omega A T Delta lambda

    In AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbzfactor = Fzodi * lod_arcsec2 * FATDL  # count rate for zodi
    tempCRbzfactor = CRbzfactor * skytrans[index]
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    lod_arcsec2 = lod_arcsec * lod_arcsec

    # NOTE: Since Omega is not used when calculating the detector noise
    # components,this multiplication is done outside the function when needed.
    # i.e.
    CRbz = tempCRbzfactor * omega_lod[index2];
    """

    return F0 * Fzodi * skytrans * area * throughput * dlambda * lod_arcsec**2


def calculate_CRbez(
    F0: float,
    Fexozodi: float,
    lod_arcsec: float,
    skytrans: float,
    area: float,
    throughput: float,
    dlambda: float,
    dist: float,
    sp: float,
) -> float:
    """
    Calculate the exozodiacal light count rate.

    Parameters
    ----------
    F0 : float
        Flux zero point.
    Fexozodi : float
        Exozodiacal light flux.
    lod_arcsec : float
        Lambda/D in arcseconds.
    skytrans : float
        Sky transmission.
    area : float
        Collecting area of the telescope.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.
    dist : float
        Distance to the star.
    sp : float
        Separation of the planet.

    Returns
    -------
    float
        Exozodiacal light count rate.

    Notes
    -----
    EXOZODI LEAKAGE
    CRbez=F_0 * n * 10^{-0.4mag_{exozodi}} * Omega * A * T * Delta lambda

    In AYO:

    CRbezfactor = Fexozodi * lod_arcsec2 * FATDL / (dist * dist);
    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    tempCRbezfactor = CRbezfactor * skytrans[index] /
                        (sp[iplanetpistartnp] * sp[iplanetpistartnp]);
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    lod_arcsec2 = lod_arcsec * lod_arcsec

    # NOTE: Since Omega is not used when calculating the detector noise
    # components,this multiplication is done outside the function when needed.
    # i.e.
    CRbez = tempCRbezfactor * omega_lod[index2];

    """
    return (
        F0 * Fexozodi * skytrans * area * throughput * dlambda * lod_arcsec**2
    ) / (dist**2 * sp**2)


def calculate_CRbbin(
    F0: float,
    Fbinary: float,
    skytrans: float,
    area: float,
    throughput: float,
    dlambda: float,
) -> float:
    """
    Calculate the count rate from neighboring stars.

    Parameters
    ----------
    F0 : float
        Flux zero point.
    Fbinary : float
        Flux from neighboring stars.
    skytrans : float
        Sky transmission.
    area : float
        Collecting area of the telescope.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.

    Returns
    -------
    float
        Count rate from neighboring stars.

    Notes
    -----
    NEIGHBORING STARS LEAKAGE

    TBD

    CRbbin=F_0* 10^{-0.4mag_binary}* Omega A T Delta lambda

    In AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbbinfactor = Fbinary * FATDL  # count rate for scattered light from
                                    # nearby stars
    tempCRbbinfactor = CRbbinfactor * skytrans[index]

    # NOTE: Since Omega is not used when calculating the detector noise
    # components, this multiplication is done outside the function when needed.
    # i.e.
    # CRbbin = tempCRbbinfactor * omega_lod[index2];

    """

    return F0 * Fbinary * skytrans * area * throughput * dlambda


def calculate_CRbth(
    lam: float,
    skytrans: float,
    area: float,
    throughput: float,
    dlambda: float,
    temp: float,
    lod_arcsec: float,
    emis=1.0,
) -> float:
    """
    Calculate background thermal count rate

    Parameters
    ----------
    lam : float
        wavelengths of observation
    skytrans : float
        Sky transmission.
    area : float
        Collecting area of the telescope.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.
    temp  : float
        Telescope mirror temperature [K]
    emis : float
        Effective emissivity for the observing system (of order unity)

    Returns
    -------
    float
        Count rate from thermal background
    """

    lam *= u.um
    epower = c.h * c.c / lam / c.k_B / temp
    Bsys = 2 * c.h * c.c**2 / lam**5 / (np.exp(epower) - 1)

    Bsys = Bsys.to(u.W / u.m**2 / u.um) / u.sr

    # angular area of photometric aperture
    Omega = np.pi * (lod_arcsec) ** 2  # in arcsec**2
    Omega = Omega.to(u.sr)

    photon_energy = c.h * c.c / lam
    photon_energy = photon_energy.to(u.J) / u.photon

    return (
        (Bsys * emis * Omega * skytrans * area * throughput * dlambda / photon_energy)
        .to(u.photon / u.s)
        .value
    )


def calculate_t_photon_count(
    lod_arcsec: float,
    det_pixscale_mas: float,
    det_npix_multiplier: float,
    det_omega_lod: float,
    det_CR: float,
) -> float:
    """
    Calculate the photon counting time.

    Parameters
    ----------
    lod_arcsec : float
        Lambda/D in arcseconds.
    det_pixscale_mas : float
        Detector pixel scale in milliarcseconds.
    det_npix_multiplier : float
        Multiplier for number of detector pixels.
    det_omega_lod : float
        Solid angle of the photometric aperture in units of (lambda/D)^2.
    det_CR : float
        Detector count rate.

    Returns
    -------
    float
        Photon counting time.

    Notes
    -----

    # According to Bernie Rauscher:
    # effective_dark_current = dark_current - f * cic * (1+W_-1[q/e])^-1.
    # If q = 0.99, (1+W_-1[q/e])^-1 = -6.73 such that
    # effective_dark_current = dark_current + f * cic * 6.73,
    # where f is the brightest pixel you care about in counts s^-1
    """

    detpixscale_lod = det_pixscale_mas / (lod_arcsec * 1000.0)

    #  this is temporary to estimate the per pixel noise
    det_npix = det_npix_multiplier * det_omega_lod / (detpixscale_lod**2)
    t_photon_count = 1.0 / (6.73 * (det_CR / det_npix))
    return t_photon_count


def calculate_CRbd(
    det_npix_multiplier: float,
    det_DC: float,
    det_RN: float,
    det_tread: float,
    det_CIC: float,
    t_photon_count: float,
    det_omega_lod: float,
    detpixscale_lod: float,
) -> float:
    """
    Calculate the detector noise count rate.

    Parameters
    ----------
    det_npix_multiplier : float
        Multiplier for number of detector pixels.
    det_DC : float
        Dark current.
    det_RN : float
        Read noise.
    det_tread : float
        Read time.
    det_CIC : float
        Clock-induced charge.
    t_photon_count : float
        Photon counting time.
    det_omega_lod : float
        Solid angle of the photometric aperture in units of (lambda/D)^2.
    detpixscale_lod : float
        Detector pixel scale in units of lambda/D.

    Returns
    -------
    float
        Detector noise count rate.

    Notes
    -----
    DETECTOR NOISE

    CRbd = n_{pix}(xi +RN^2/tau_{exposure}+CIC/t_{photon_count})
    """
    # calculate npix
    det_npix = det_npix_multiplier * det_omega_lod / (detpixscale_lod) ** 2
    return (det_DC + det_RN * det_RN / det_tread + det_CIC / t_photon_count) * det_npix


def calculate_CRnf(
    F0: float,
    Fstar: float,
    area: float,
    pixscale: float,
    throughput: float,
    dlambda: float,
    SNR: float,
    noisefloor: float,
) -> float:
    """
    Calculate the noise floor count rate.

    Parameters
    ----------
    F0 : float
        Flux zero point.
    Fstar : float
        Stellar flux.
    area : float
        Collecting area of the telescope.
    pixscale : float
        Pixel scale of the detector.
    throughput : float
        Throughput of the system.
    dlambda : float
        Bandwidth.
    SNR : float
        Signal-to-noise ratio.
    noisefloor : float
        Noise floor level.

    Returns
    -------
    float
        Noise floor count rate.

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

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbsfactor = Fstar * oneopixscale2 * FATDL  # for stellar leakage count
    rate calculation
    Fstar = 10**(-0.4 * magstar)
        tempCRnffactor = SNR * CRbsfactor * noisefloor_interp[index];


    # NOTE: Since Omega is not used when calculating the detector
    # noise components, this multiplication is done outside the
    # function when needed. i.e.
    # CRnoisefloor = tempCRnffactor * omega_lod[index2];

    """
    return SNR * (F0 * Fstar * area * throughput * dlambda / (pixscale**2)) * noisefloor


def interpolate_arrays(
    Istar: np.ndarray,
    noisefloor: np.ndarray,
    npix: int,
    ndiams: int,
    stellar_diam_lod: float,
    angdiams: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate Istar and noisefloor arrays based on stellar diameter.

    Parameters
    ----------
    Istar : np.ndarray
        3D array of star intensities.
    noisefloor : np.ndarray
        3D array of noise floor values.
    npix : int
        Number of pixels in each dimension.
    ndiams : int
        Number of stellar diameters.
    stellar_diam_lod : float
        Stellar diameter in lambda/D units.
    angdiams : np.ndarray
        Array of angular diameters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Interpolated Istar and noisefloor arrays.

    Notes
    -----
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    oneolod_arcsec = 1.0 / lod_arcsec
    stellar_diam_lod = angdiamstar_arcsec * oneolod_arcsec
    # Usage:
    # Assuming Istar and noisefloor are 3D NumPy arrays with shape
    # (npix, npix, ndiams)
    # and angdiams is a 1D NumPy array
    Istar_interp, noisefloor_interp = interpolate_arrays(Istar, noisefloor,
                        npix, ndiams, stellar_diam_lod, angdiams)

    """
    Istar_interp = np.zeros((npix, npix))
    noisefloor_interp = np.zeros((npix, npix))

    k = np.searchsorted(angdiams, stellar_diam_lod)

    if k < ndiams:
        # Interpolation
        weight = (stellar_diam_lod - angdiams[k - 1]) / (angdiams[k] - angdiams[k - 1])
        Istar_interp = (1 - weight) * Istar[:, :, k - 1] + weight * Istar[:, :, k]
        noisefloor_interp = (1 - weight) * noisefloor[
            :, :, k - 1
        ] + weight * noisefloor[:, :, k]
    else:
        # Extrapolation
        weight = (stellar_diam_lod - angdiams[k - 1]) / (
            angdiams[k - 1] - angdiams[k - 2]
        )
        Istar_interp = Istar[:, :, k - 1] + weight * (
            Istar[:, :, k - 1] - Istar[:, :, k - 2]
        )
        noisefloor_interp = noisefloor[:, :, k - 1] + weight * (
            noisefloor[:, :, k - 1] - noisefloor[:, :, k - 2]
        )

    # Ensure non-negative values
    Istar_interp = np.maximum(Istar_interp, 0)
    noisefloor_interp = np.maximum(noisefloor_interp, 0)

    return Istar_interp, noisefloor_interp


def measure_coronagraph_performance(
    psf_trunc_ratio: np.ndarray,
    photap_frac: np.ndarray,
    Istar_interp: np.ndarray,
    skytrans: np.ndarray,
    omega_lod: np.ndarray,
    npix: int,
    xcenter: float,
    ycenter: float,
    oneopixscale_arcsec: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Measure the performance of the coronagraph.

    Parameters
    ----------
    psf_trunc_ratio : np.ndarray
        PSF truncation ratios.
    photap_frac : np.ndarray
        Photometric aperture fractions.
    Istar_interp : np.ndarray
        Interpolated stellar intensity.
    skytrans : np.ndarray
        Sky transmission.
    omega_lod : np.ndarray
        Solid angle of photometric aperture in (lambda/D)^2.
    npix : int
        Number of pixels in each dimension.
    xcenter : float
        X-coordinate of the center.
    ycenter : float
        Y-coordinate of the center.
    oneopixscale_arcsec : float
        Inverse of pixel scale in arcseconds.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        det_sep_pix, det_sep, det_Istar, det_skytrans, det_photap_frac, det_omega_lod

    Notes
    -----
    This function measures various performance metrics of the coronagraph,
    including the detection separation, stellar intensity, sky transmission,
    photometric aperture fraction, and solid angle at the detection point.
    """

    # Find psf_trunc_ratio closest to 0.3
    bestiratio = np.argmin(np.abs(psf_trunc_ratio - 0.3))

    # Find maximum photap_frac in first half of image
    maxphotap_frac = np.max(photap_frac[: npix // 2, int(ycenter), bestiratio])

    # Find IWA
    row = photap_frac[:, int(ycenter), bestiratio]
    iwa_index = np.where(row[: int(xcenter)] > 0.5 * maxphotap_frac)[0][-1]
    det_sep_pix = abs((iwa_index + 0.5) - xcenter)
    det_sep = det_sep_pix / oneopixscale_arcsec

    # Calculate max values in 2-pixel annulus at det_sep
    y, x = np.ogrid[:npix, :npix]
    dist_from_center = np.sqrt((x - xcenter + 0.5) ** 2 + (y - ycenter + 0.5) ** 2)
    mask = np.abs(dist_from_center - det_sep_pix) < 2

    det_Istar = np.max(Istar_interp[mask])
    det_skytrans = np.max(skytrans[mask])

    photap_frac_masked = photap_frac[:, :, bestiratio][mask]
    det_photap_frac = np.max(photap_frac_masked)
    det_omega_lod = omega_lod[:, :, bestiratio][mask][np.argmax(photap_frac_masked)]

    return det_sep_pix, det_sep, det_Istar, det_skytrans, det_photap_frac, det_omega_lod


def calculate_total_throughput(observatory):
    """
    This function calculates the optical (telescope + instrument path) + detector
    throughput, which is used as multiplicative factor when calculating the noise terms.
    """
    return (
        observatory.telescope.telescope_throughput
        * observatory.coronagraph.coronagraph_throughput
        * observatory.detector.dQE
        * observatory.detector.QE
    )


def calculate_exposure_time(
    observation: Observation,
    scene: AstrophysicalScene,
    observatory: Observatory,
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

    """

    # Calculate optical+detector throughput (nlambd array)
    throughput = calculate_total_throughput(observatory)

    # print("Observation inputs:")
    # print(f"nlambd: {observation.nlambd}")
    # print(f"lambd[0]: {observation.lambd[0]}")
    # print(f"SR[0]: {observation.SR[0]}")
    # print(f"SNR[0]: {observation.SNR[0]}")

    # print("\nScene inputs:")
    # print(f"ntargs: {scene.ntargs}")
    # print(f"mag[0, 0]: {scene.mag[0, 0]}")
    # print(f"F0[0]: {scene.F0[0]}")
    # print(f"Fzodi_list[0, 0]: {scene.Fzodi_list[0, 0]}")
    # print(f"Fexozodi_list[0, 0]: {scene.Fexozodi_list[0, 0]}")
    # print(f"Fbinary_list[0, 0]: {scene.Fbinary_list[0, 0]}")
    # print(f"angdiam_arcsec[0]: {scene.angdiam_arcsec[0]}")
    # print(f"dist[0]: {scene.dist[0]}")
    # print(f"xp[0, 0, 0]: {scene.xp[0, 0, 0]}")
    # print(f"yp[0, 0, 0]: {scene.yp[0, 0, 0]}")
    # print(f"sp[0, 0, 0]: {scene.sp[0, 0, 0]}")
    # print(f"Fp0[0, 0, 0]: {scene.Fp0[0, 0, 0]}")
    # print(f"min_deltamag[0]: {scene.min_deltamag[0]}")

    # print("\nInstrument inputs:")
    # print(f"telescope.D: {observatory.telescope.diameter}")
    # print(f"telescope.Area: {observatory.telescope.Area}")
    # print(f"telescope.throughput[0]: {throughput[0]}")
    # print(f"telescope.toverhead_multi: {observatory.telescope.toverhead_multi}")
    # print(f"telescope.toverhead_fixed: {observatory.telescope.toverhead_fixed}")
    # print(f"coronagraph.bandwidth: {observatory.coronagraph.bandwidth}")
    # print(f"coronagraph.pixscale: {observatory.coronagraph.pixscale}")
    # print(f"coronagraph.IWA: {observatory.coronagraph.minimum_IWA}")
    # print(f"coronagraph.OWA: {observatory.coronagraph.maximum_OWA}")
    # print(f"coronagraph.npix: {observatory.coronagraph.npix}")
    # print(f"coronagraph.ndiams: {observatory.coronagraph.ndiams}")
    # print(f"coronagraph.xcenter: {observatory.coronagraph.xcenter}")
    # print(f"coronagraph.ycenter: {observatory.coronagraph.ycenter}")
    # print(f"coronagraph.nrolls: {observatory.coronagraph.nrolls}")
    # print(f"coronagraph.npsfratios: {observatory.coronagraph.npsfratios}")
    # print(f"coronagraph.Istar[0, 0, 0]: {observatory.coronagraph.Istar[0, 0, 0]}")
    # print(
    #     f"coronagraph.noisefloor[0, 0, 0]: {observatory.coronagraph.noisefloor[0, 0, 0]}"
    # )
    # print(f"coronagraph.angdiams[0]: {observatory.coronagraph.angdiams[0]}")
    # print(
    #     f"coronagraph.psf_trunc_ratio[0]: {observatory.coronagraph.psf_trunc_ratio[0]}"
    # )
    # print(
    #     f"coronagraph.photap_frac[0, 0, 0]: {observatory.coronagraph.photap_frac[0, 0, 0]}"
    # )
    # print(f"coronagraph.skytrans[0, 0]: {observatory.coronagraph.skytrans[0, 0]}")
    # print(
    #     f"coronagraph.omega_lod[0, 0, 0]: {observatory.coronagraph.omega_lod[0, 0, 0]}"
    # )

    # print(f"detector.det_pixscale_mas: {observatory.detector.pixscale_mas}")
    # print(f"detector.det_npix_multiplier[0]: {observatory.detector.npix_multiplier[0]}")
    # print(f"detector.det_DC[0]: {observatory.detector.DC[0]}")
    # print(f"detector.det_RN[0]: {observatory.detector.RN[0]}")
    # print(f"detector.det_tread[0]: {observatory.detector.tread[0]}")
    # print(f"detector.det_CIC[0]: {observatory.detector.CIC[0]}")

    # print("\nEDITH inputs:")
    # print(f"norbits: {observation.norbits}")
    # print(f"nmeananom: {observation.nmeananom}")
    # print(f"td_limit: {observation.td_limit}")

    for istar in range(scene.ntargs):  # set to 1
        for ilambd in range(observation.nlambd):  # set to 1

            # Calculate useful quantities
            Fstar = 10 ** (-0.4 * scene.mag[istar, ilambd])

            # Take the lesser of the desired bandwidth
            # and what coronagraph allows
            deltalambda_nm = np.min(
                [
                    (observation.lambd[ilambd] * 1000.0) / observation.SR[ilambd],
                    observatory.coronagraph.bandwidth
                    * (observation.lambd[ilambd] * 1000.0),
                ]
            )

            lod_arcsec = (
                observation.lambd[ilambd] * 1e-6 / observatory.telescope.diameter
            ) * 206264.806

            area_cm2 = observatory.telescope.Area * 100 * 100

            stellar_diam_lod = scene.angdiam_arcsec[istar] / lod_arcsec

            detpixscale_lod = observatory.detector.pixscale_mas / (lod_arcsec * 1000.0)

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

            # Measure coronagraph performance at each IWA
            pixscale_rad = observatory.coronagraph.pixscale * (
                observation.lambd[ilambd] * 1e-6 / observatory.telescope.diameter
            )
            oneopixscale_arcsec = 1.0 / (pixscale_rad * 206264.806)

            # Measure coronagraph performance at each IWA
            (
                det_sep_pix,
                det_sep,
                det_Istar,
                det_skytrans,
                det_photap_frac,
                det_omega_lod,
            ) = measure_coronagraph_performance(
                observatory.coronagraph.psf_trunc_ratio,
                observatory.coronagraph.photap_frac,
                Istar_interp,
                observatory.coronagraph.skytrans,
                observatory.coronagraph.omega_lod,
                observatory.coronagraph.npix,
                observatory.coronagraph.xcenter,
                observatory.coronagraph.ycenter,
                oneopixscale_arcsec,
            )

            # Here we calculate detector noise, as it may depend on count rates
            # We don't know the count rates yet, so we make estimates based on
            # values near the IWA

            # Detector noise from signal itself (we budget for 10x
            # the planet count rate for the minimum detectable planet)

            det_CRp = calculate_CRp(
                scene.F0[ilambd],
                Fstar,
                10 * 10 ** (-0.4 * scene.min_deltamag[istar]),
                area_cm2,
                det_photap_frac,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CRbs = calculate_CRbs(
                scene.F0[ilambd],
                Fstar,
                det_Istar,
                area_cm2,
                observatory.coronagraph.pixscale,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CRbz = calculate_CRbz(
                scene.F0[ilambd],
                scene.Fzodi_list[istar, ilambd],
                lod_arcsec,
                det_skytrans,
                area_cm2,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CRbez = calculate_CRbez(
                scene.F0[ilambd],
                scene.Fexozodi_list[istar, ilambd],
                lod_arcsec,
                det_skytrans,
                area_cm2,
                throughput[ilambd],
                deltalambda_nm,
                scene.dist[istar],
                det_sep,
            )
            det_CRbbin = calculate_CRbbin(
                scene.F0[ilambd],
                scene.Fbinary_list[istar, ilambd],
                det_skytrans,
                area_cm2,
                throughput[ilambd],
                deltalambda_nm,
            )
            det_CRbth = 0.0
            # calculate_CRbth(
            #     observation.lambd[
            #         ilambd
            #     ],  ### this is the wavelength of observation; is this the right variable name??
            #     det_skytrans,
            #     area_cm2,
            #     throughput[ilambd],
            #     deltalambda_nm,
            #     300,  # temperature (should be a variable eventually)
            #     lod_arcsec,
            # )

            det_CR = det_CRp + det_CRbs + det_CRbz + det_CRbez + det_CRbbin + det_CRbth

            for iorbit in np.arange(observation.norbits):

                for iphase in np.arange(observation.nmeananom):

                    # Calculate position of the planet in the image
                    # (from l/D to pixel)
                    ix = (
                        scene.xp[iphase, iorbit, istar] * oneopixscale_arcsec
                        + observatory.coronagraph.xcenter
                    )
                    iy = (
                        scene.yp[iphase, iorbit, istar] * oneopixscale_arcsec
                        + observatory.coronagraph.ycenter
                    )

                    # Calculate separation in arcsec
                    sp_lod = scene.sp[iphase, iorbit, istar] / lod_arcsec

                    # If planet is within the boundaries of the observatory.coronagraph
                    # simulation and hard IWA/OWA cutoffs...
                    if (
                        (ix >= 0)
                        and (ix < observatory.coronagraph.npix)
                        and (iy >= 0)
                        and (iy < observatory.coronagraph.npix)
                        #                        and (sp_lod > observatory.coronagraph.minimum_IWA)
                        #                        and (sp_lod < observatory.coronagraph.maximum_OWA)
                    ):

                        for iratio in np.arange(observatory.coronagraph.npsfratios):
                            # First we just calculate CRp and CRnoisefloor
                            # to see if CRp > CRnoisefloor

                            # PLANET COUNT RATE CRP
                            CRp = calculate_CRp(
                                scene.F0[ilambd],
                                Fstar,
                                scene.Fp0[iphase, iorbit, istar],
                                area_cm2,
                                observatory.coronagraph.photap_frac[
                                    int(np.floor(iy)), int(np.floor(ix)), iratio
                                ],
                                throughput[ilambd],
                                deltalambda_nm,
                            )

                            # NOISE FLOOR CRNF
                            CRnf = calculate_CRnf(
                                scene.F0[ilambd],
                                Fstar,
                                area_cm2,
                                observatory.coronagraph.pixscale,
                                throughput[ilambd],
                                deltalambda_nm,
                                observation.SNR[ilambd],
                                noisefloor_interp[int(np.floor(iy)), int(np.floor(ix))],
                            )

                            # multiply by omega at that point
                            CRnf *= observatory.coronagraph.omega_lod[
                                int(np.floor(iy)), int(np.floor(ix)), iratio
                            ]
                            # NOTE: noisefloor_interp: technically the Y axis
                            # is rows and the X axis is columns,
                            # that is why they are inverted
                            # NOTE: Evaluate if int(round(iy)) is better than
                            # np.floor. Kept np.floor for consistency

                            # Check if it's above the noise floor and
                            # calculate exposure time if conditions are met
                            if (
                                CRp > CRnf
                                and observatory.coronagraph.omega_lod[
                                    int(np.floor(iy)), int(np.floor(ix)), iratio
                                ]
                                > detpixscale_lod**2
                            ):

                                # Calculate the rest of the background noise

                                # NOTE: WHEN CALCULATING THE COUNT RATES,
                                # WE NEED TO MULTIPLY BY OMEGA_LOD i.e.
                                # THE SOLID ANGLE OF THE PHOTOMETRIC APERTURE

                                # Calculate CRbs
                                CRbs = calculate_CRbs(
                                    scene.F0[ilambd],
                                    Fstar,
                                    Istar_interp[int(np.floor(iy)), int(np.floor(ix))],
                                    area_cm2,
                                    observatory.coronagraph.pixscale,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                )

                                # Calculate CRbz
                                CRbz = calculate_CRbz(
                                    scene.F0[ilambd],
                                    scene.Fzodi_list[istar, ilambd],
                                    lod_arcsec,
                                    observatory.coronagraph.skytrans[
                                        int(np.floor(iy)), int(np.floor(ix))
                                    ],
                                    area_cm2,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                )

                                # Calculate CRbez
                                CRbez = calculate_CRbez(
                                    scene.F0[ilambd],
                                    scene.Fexozodi_list[istar, ilambd],
                                    lod_arcsec,
                                    observatory.coronagraph.skytrans[
                                        int(np.floor(iy)), int(np.floor(ix))
                                    ],
                                    area_cm2,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                    scene.dist[istar],
                                    scene.sp[iphase, iorbit, istar],
                                )

                                # Calculate CRbbin
                                CRbbin = calculate_CRbbin(
                                    scene.F0[ilambd],
                                    scene.Fbinary_list[istar, ilambd],
                                    observatory.coronagraph.skytrans[
                                        int(np.floor(iy)), int(np.floor(ix))
                                    ],
                                    area_cm2,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                )

                                # Calculate CRbd
                                t_photon_count = calculate_t_photon_count(
                                    lod_arcsec,
                                    observatory.detector.pixscale_mas,
                                    observatory.detector.npix_multiplier[ilambd],
                                    det_omega_lod,
                                    det_CR,
                                )

                                CRbd = calculate_CRbd(
                                    observatory.detector.npix_multiplier[ilambd],
                                    observatory.detector.DC[ilambd],
                                    observatory.detector.RN[ilambd],
                                    observatory.detector.tread[ilambd],
                                    observatory.detector.CIC[ilambd],
                                    t_photon_count,
                                    det_omega_lod,
                                    detpixscale_lod,
                                )
                                CRbth = 0
                                # calculate_CRbth(
                                #     observation.lambd[
                                #         ilambd
                                #     ],  ### this is the wavelength of observation; is this the right variable name??
                                #     det_skytrans,
                                #     area_cm2,
                                #     throughput[ilambd],
                                #     deltalambda_nm,
                                #     300,  # temperature (should be a variable eventually)
                                #     lod_arcsec,
                                # )

                                # TOTAL BACKGROUND NOISE
                                CRb = (
                                    CRbs + CRbz + CRbez + CRbbin + CRbth
                                ) * observatory.coronagraph.omega_lod[
                                    int(np.floor(iy)), int(np.floor(ix)), iratio
                                ]
                                # Add detector noise
                                CRb += CRbd

                                # EXPOSURE TIME
                                # count rate term
                                # NOTE this includes the systematic noise floor
                                # term a la Bijan Nemati
                                cp = (CRp + 2 * CRb) / (CRp * CRp - CRnf * CRnf)

                                # Calculate Exposure time
                                observation.exptime[istar, ilambd] = (
                                    observation.SNR[ilambd]
                                    * observation.SNR[ilambd]
                                    * cp
                                    * observatory.telescope.toverhead_multi
                                    + observatory.telescope.toverhead_fixed
                                )  # record exposure time with overheads

                                if observation.exptime[istar, ilambd] < 0:
                                    # time is past the systematic
                                    # noise floor limit
                                    observation.exptime[istar, ilambd] = np.inf

                                if (
                                    observation.exptime[istar, ilambd]
                                    > observation.td_limit
                                ):
                                    # treat as unobservable
                                    # if beyond exposure time limit
                                    observation.exptime[istar, ilambd] = np.inf

                                if observatory.coronagraph.nrolls != 1:
                                    # multiply by number of required rolls to
                                    # achieve 360 deg coverage
                                    # (after tlimit enforcement)
                                    observation.exptime[
                                        istar, ilambd
                                    ] *= observatory.coronagraph.nrolls
                            else:
                                # It's below the systematic noise floor...
                                observation.exptime[istar, ilambd] = np.inf
                    else:
                        # outside of the input contrast map or hard
                        # IWA/OWA cutoffs
                        observation.exptime[istar, ilambd] = np.inf

                    # print("Interesting variables for exposure time calculation:")
                    # print(f"CRp: {CRp:.6e}")
                    # print(f"CRb: {CRb:.6e}")
                    # print(f"SNRCRpfloor: {CRnf:.6e}")
                    # print(f"SNR: {observation.SNR[ilambd]:.6e}")
                    # print(
                    #     f"toverhead_multi: {observatory.telescope.toverhead_multi:.6e}"
                    # )
                    # print(
                    #     f"toverhead_fixed: {observatory.telescope.toverhead_fixed:.6e}"
                    # )
                    # # Print calculated values
                    # print(f"Calculated cp: {cp:.6e}")

                    # print("Useful quantities:")
                    # print(
                    #     f"det_npix: {(observatory.detector.npix_multiplier[ilambd] * det_omega_lod /(detpixscale_lod**2)):.6e}"
                    # )
                    # print(
                    #     f"det_npix_multiplier: {observatory.detector.npix_multiplier[ilambd]:.6e}"
                    # )
                    # print(
                    #     f"omega_lod[iratio, iy, ix]: {observatory.coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio]:.6e}"
                    # )
                    # print(f"oneodetpixscale_lod2: {1/detpixscale_lod**2:.6e}")
                    # print(
                    #     f"CRbdfactor: {CRbd/(observatory.detector.npix_multiplier[ilambd]*det_omega_lod/(detpixscale_lod)**2):.6e}"
                    # )
                    # print(f"CRbd: {CRbd:.6e}")
                    # print(f"det_DC: {observatory.detector.DC[ilambd]:.6e}")
                    # print(f"det_RN: {observatory.detector.RN[ilambd]:.6e}")
                    # print(f"det_CIC: {observatory.detector.CIC[ilambd]:.6e}")
                    # print(f"det_tread: {observatory.detector.tread[ilambd]:.6e}")
                    # print(f"t_photon_count: {t_photon_count:.6e}")
                    # print(f"det_CR: {det_CR:.6e}")
                    # print(f"det_sep: {det_sep:.6e}")
                    # print(f"det_sep_pix: {det_sep_pix:.6e}")
                    # print(f"det_Istar: {det_Istar:.6e}")
                    # print(f"det_skytrans: {det_skytrans:.6e}")
                    # print(f"det_photap_frac: {det_photap_frac:.6e}")
                    # print(f"det_omega_lod: {det_omega_lod:.6e}")
                    # print("Interesting Variables:")
                    # print(f"Fstar: {Fstar:.6e}")
                    # print(f"deltalambda_nm: {deltalambda_nm:.6e}")
                    # print(f"lod_arcsec: {lod_arcsec:.6e}")
                    # print(f"area_cm2: {area_cm2:.6e}")
                    # print(f"stellar_diam_lod: {stellar_diam_lod:.6e}")
                    # print(f"pixscale_rad: {pixscale_rad:.6e}")
                    # print(f"oneopixscale_arcsec: {oneopixscale_arcsec:.6e}")

                    # print("\nCoronagraph Performance:")
                    # print(f"det_sep_pix: {det_sep_pix:.6e}")
                    # print(f"det_sep: {det_sep:.6e}")
                    # print(f"det_Istar: {det_Istar:.6e}")
                    # print(f"det_skytrans: {det_skytrans:.6e}")
                    # print(f"det_photap_frac: {det_photap_frac:.6e}")
                    # print(f"det_omega_lod: {det_omega_lod:.6e}")

                    # print("\nDetector Noise Estimates:")
                    # print(f"det_CRp: {det_CRp:.6e}")
                    # print(f"det_CRbs: {det_CRbs:.6e}")
                    # print(f"det_CRbz: {det_CRbz:.6e}")
                    # print(f"det_CRbez: {det_CRbez:.6e}")
                    # print(f"det_CRbbin: {det_CRbbin:.6e}")

                    # print("\nNoise floor:")
                    # print(f"CRnf: {CRnf:.6e}")
                    # print("\nCount Rates:")
                    # print(
                    #     f"omega_lod: {observatory.coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio]:.6e}"
                    # )

                    # print(
                    #     f"CRp: {CRp:.6e}"
                    # )  # photap_frac dimensions (npix,npix,len(photap_frac))
                    # print(
                    #     f"CRbs/omega_lod: {CRbs:.6e}"
                    # )  # Istar_interp dimensions (npix,npix,len(angdiams))
                    # print(
                    #     f"CRbz/omega_lod: {CRbz:.6e}"
                    # )  # skytrans dimensions (npix,npix)
                    # print(
                    #     f"CRbez/omega_lod: {CRbez:.6e}"
                    # )  # skytrans dimensions (npix,npix) x sp dimensions (nmeananom, norbits, ntargs)
                    # print(
                    #     f"CRbbin/omega_lod: {CRbbin:.6e}"
                    # )  # skytrans dimensions (npix,npix)
                    # print(
                    #     f"CRbbd/omega_lod: {CRbd:.6e}"
                    # )  # skytrans dimensions (npix,npix)
                    # print(f"t_photon_count:{t_photon_count:.6e}")
                    # print(f"CRbd:{CRbd:.6e}")
                    # print(f"Total Background Noise (CRb+CRbd): {CRb:.6e}")

        # NOTE FOR FUTURE DEVELOPMENT
        # The nmeananom, norbits, npsfratios loops are not stored in the
        # exptime matrix.
        # This is not a problem right now since these are "fake" loops as of
        # now (nmeananom, norbits, npsfratios all are 1).
        # But this might change in the future.
        return


def calculate_signal_to_noise(
    observation: Observation,
    scene: AstrophysicalScene,
    observatory: Observatory,
) -> None:
    """
    Calculate the signal-to-noise ratio for each target and wavelength.

    Parameters
    ----------
    observation : Observation
        Object containing observation parameters.
    scene : AstrophysicalScene
        Object containing scene parameters.
    observatory: Observatory
        Object containing observatory parameters.

    Returns
    -------
    None
        Results are stored in the observation object.

    Notes
    -----
    This function calculates the signal-to-noise ratio for each target star and wavelength,
    taking into account various noise sources and coronagraph performance metrics.
    It iterates through targets, wavelengths, orbits, and phases to compute
    the achieved signal-to-noise ratio for planet detection given a fixed observation time.
    """

    # Calculate optical+detector throughput (nlambd array)
    throughput = calculate_total_throughput(observatory)

    for istar in range(scene.ntargs):  # set to 1
        for ilambd in range(observation.nlambd):  # set to 1

            # Calculate useful quantities
            Fstar = 10 ** (-0.4 * scene.mag[istar, ilambd])

            # Take the lesser of the desired bandwidth and what
            # coronagraph allows
            deltalambda_nm = np.min(
                [
                    (observation.lambd[ilambd] * 1000.0) / observation.SR[ilambd],
                    observatory.coronagraph.bandwidth
                    * (observation.lambd[ilambd] * 1000.0),
                ]
            )
            lod_arcsec = (
                observation.lambd[ilambd] * 1e-6 / observatory.telescope.diameter
            ) * 206264.806
            area_cm2 = observatory.telescope.Area * 100 * 100
            stellar_diam_lod = scene.angdiam_arcsec[istar] / lod_arcsec
            detpixscale_lod = observatory.detector.pixscale_mas / (lod_arcsec * 1000.0)

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

            # Measure observatory.coronagraph performance at each IWA
            pixscale_rad = observatory.coronagraph.pixscale * (
                observation.lambd[ilambd] * 1e-6 / observatory.telescope.diameter
            )
            oneopixscale_arcsec = 1.0 / (pixscale_rad * 206264.806)

            # Measure observatory.coronagraph performance at each IWA
            (
                det_sep_pix,
                det_sep,
                det_Istar,
                det_skytrans,
                det_photap_frac,
                det_omega_lod,
            ) = measure_coronagraph_performance(
                observatory.coronagraph.psf_trunc_ratio,
                observatory.coronagraph.photap_frac,
                Istar_interp,
                observatory.coronagraph.skytrans,
                observatory.coronagraph.omega_lod,
                observatory.coronagraph.npix,
                observatory.coronagraph.xcenter,
                observatory.coronagraph.ycenter,
                oneopixscale_arcsec,
            )

            # Here we calculate detector noise, as it may depend on count rates
            # We don't know the count rates yet, so we make estimates based on
            # values near the IWA

            # Detector noise from signal itself (we budget for 10x the planet
            # count rate for the minimum detectable planet)
            det_CRp = calculate_CRp(
                scene.F0[ilambd],
                Fstar,
                10 * 10 ** (-0.4 * scene.min_deltamag[istar]),
                area_cm2,
                det_photap_frac,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CRbs = calculate_CRbs(
                scene.F0[ilambd],
                Fstar,
                det_Istar,
                area_cm2,
                observatory.coronagraph.pixscale,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CRbz = calculate_CRbz(
                scene.F0[ilambd],
                scene.Fzodi_list[istar, ilambd],
                lod_arcsec,
                det_skytrans,
                area_cm2,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CRbez = calculate_CRbez(
                scene.F0[ilambd],
                scene.Fexozodi_list[istar, ilambd],
                lod_arcsec,
                det_skytrans,
                area_cm2,
                throughput[ilambd],
                deltalambda_nm,
                scene.dist[istar],
                det_sep,
            )
            det_CRbbin = calculate_CRbbin(
                scene.F0[ilambd],
                scene.Fbinary_list[istar, ilambd],
                det_skytrans,
                area_cm2,
                throughput[ilambd],
                deltalambda_nm,
            )

            det_CR = det_CRp + det_CRbs + det_CRbz + det_CRbez + det_CRbbin
            # TODO ADD QE MULTIPLICATIVELY TO DET_CR
            for iorbit in np.arange(observation.norbits):
                for iphase in np.arange(observation.nmeananom):

                    # Calculate position of the planet in the image
                    # (from l/D to pixel)
                    ix = (
                        scene.xp[iphase, iorbit, istar] * oneopixscale_arcsec
                        + observatory.coronagraph.xcenter
                    )
                    iy = (
                        scene.yp[iphase, iorbit, istar] * oneopixscale_arcsec
                        + observatory.coronagraph.ycenter
                    )
                    # Calculate separation in arcsec
                    sp_lod = scene.sp[iphase, iorbit, istar] / lod_arcsec

                    # if planet is within the boundaries of the observatory.coronagraph
                    # simulation and hard IWA/OWA cutoffs...
                    if (
                        (ix >= 0)
                        and (ix < observatory.coronagraph.npix)
                        and (iy >= 0)
                        and (iy < observatory.coronagraph.npix)
                        #                        and (sp_lod > observatory.coronagraph.minimum_IWA)
                        #                        and (sp_lod < observatory.coronagraph.maximum_OWA)
                    ):

                        for iratio in np.arange(observatory.coronagraph.npsfratios):
                            # First we just calculate CRp and CRnoisefloor to
                            # see if CRp > CRnoisefloor

                            # PLANET COUNT RATE CRP
                            CRp = calculate_CRp(
                                scene.F0[ilambd],
                                Fstar,
                                scene.Fp0[iphase, iorbit, istar],
                                area_cm2,
                                observatory.coronagraph.photap_frac[
                                    int(np.floor(iy)), int(np.floor(ix)), iratio
                                ],
                                throughput[ilambd],
                                deltalambda_nm,
                            )

                            # NOISE FLOOR CRNF
                            #  NOTE THIS TIME THIS IS JUST THE NOISE
                            # FACTOR RATIO (i.e. we assume SNR =1 so
                            # that we can use it for the snr calculation later)

                            CRnf_factor = calculate_CRnf(
                                scene.F0[ilambd],
                                Fstar,
                                area_cm2,
                                observatory.coronagraph.pixscale,
                                throughput[ilambd],
                                deltalambda_nm,
                                1,
                                noisefloor_interp[int(np.floor(iy)), int(np.floor(ix))],
                            )

                            # multiply by omega at that point
                            CRnf_factor *= observatory.coronagraph.omega_lod[
                                int(np.floor(iy)), int(np.floor(ix)), iratio
                            ]
                            # NOTE: noisefloor_interp: technically the Y axis
                            # is rows and the X axis is columns,
                            # that is why they are inverted
                            # NOTE: Evaluate if int(round(iy) is better than
                            # np.floor. Kept np.floor for consistency

                            # Check and calculate exposure time if
                            # conditions are met
                            if (
                                observatory.coronagraph.omega_lod[
                                    int(np.floor(iy)), int(np.floor(ix)), iratio
                                ]
                                > detpixscale_lod**2
                            ):
                                # CALCULATE THE REST OF THE BACKGROUND NOISE

                                # ## WHEN CALCULATING THE COUNT RATES, WE NEED
                                # TO MULTIPLY BY OMEGA_LOD i.e.
                                # # THE SOLID ANGLE OF THE PHOTOMETRIC APERTURE
                                # Calculate CRbs
                                CRbs = calculate_CRbs(
                                    scene.F0[ilambd],
                                    Fstar,
                                    Istar_interp[int(np.floor(iy)), int(np.floor(ix))],
                                    area_cm2,
                                    observatory.coronagraph.pixscale,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                )

                                # Calculate CRbz
                                CRbz = calculate_CRbz(
                                    scene.F0[ilambd],
                                    scene.Fzodi_list[istar, ilambd],
                                    lod_arcsec,
                                    observatory.coronagraph.skytrans[
                                        int(np.floor(iy)), int(np.floor(ix))
                                    ],
                                    area_cm2,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                )

                                # Calculate CRbez
                                CRbez = calculate_CRbez(
                                    scene.F0[ilambd],
                                    scene.Fexozodi_list[istar, ilambd],
                                    lod_arcsec,
                                    observatory.coronagraph.skytrans[
                                        int(np.floor(iy)), int(np.floor(ix))
                                    ],
                                    area_cm2,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                    scene.dist[istar],
                                    scene.sp[iphase, iorbit, istar],
                                )

                                # Calculate CRbbin
                                CRbbin = calculate_CRbbin(
                                    scene.F0[ilambd],
                                    scene.Fbinary_list[istar, ilambd],
                                    observatory.coronagraph.skytrans[
                                        int(np.floor(iy)), int(np.floor(ix))
                                    ],
                                    area_cm2,
                                    throughput[ilambd],
                                    deltalambda_nm,
                                )

                                # Calculate CRbd
                                t_photon_count = calculate_t_photon_count(
                                    lod_arcsec,
                                    observatory.detector.pixscale_mas,
                                    observatory.detector.npix_multiplier[ilambd],
                                    det_omega_lod,
                                    det_CR,
                                )

                                CRbd = calculate_CRbd(
                                    observatory.detector.npix_multiplier[ilambd],
                                    observatory.detector.DC[ilambd],
                                    observatory.detector.RN[ilambd],
                                    observatory.detector.tread[ilambd],
                                    observatory.detector.CIC[ilambd],
                                    t_photon_count,
                                    det_omega_lod,
                                    detpixscale_lod,
                                )

                                # TOTAL BACKGROUND NOISE
                                CRb = (
                                    CRbs + CRbz + CRbez + CRbbin
                                ) * observatory.coronagraph.omega_lod[
                                    int(np.floor(iy)), int(np.floor(ix)), iratio
                                ]
                                # Add detector noise
                                CRb += CRbd

                                # SIGNAL-TO-NOISE
                                # time term
                                time_factors = (
                                    observation.obstime * observatory.coronagraph.nrolls
                                    - observatory.telescope.toverhead_fixed
                                ) / (
                                    observatory.telescope.toverhead_multi
                                    * ((CRp + 2 * CRb))
                                )

                                # Signal-to-noise
                                observation.fullsnr[istar, ilambd] = np.sqrt(
                                    (time_factors * CRp**2)
                                    / (1 + time_factors * CRnf_factor**2)
                                )

                                if observation.fullsnr[istar, ilambd] < 0:
                                    # time is past the systematic noise
                                    # floor limit
                                    observation.fullsnr[istar, ilambd] = 0
                                if observation.fullsnr[istar, ilambd] > 100:
                                    # treat as unobservable if beyond
                                    # exposure time limit
                                    observation.fullsnr[istar, ilambd] = 100

                            else:
                                # It's below the systematic noise floor...
                                observation.fullsnr[istar, ilambd] = np.inf
                    else:
                        # outside of the input contrast map or hard
                        # IWA/OWA cutoffs
                        observation.fullsnr[istar, ilambd] = np.inf

        # NOTE FOR FUTURE DEVELOPMENT
        # The nmeananom, norbits, npsfratios loops are not stored in the
        # fullsnr matrix. This is not a problem right now since these are
        # "fake" loops as of now (nmeananom, norbits, npsfratios all are 1).
        # But this might change in the future.
        return
