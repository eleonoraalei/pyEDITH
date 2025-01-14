import numpy as np

def calculate_CRp(F0, Fstar, Fp0, area,Upsilon, throughput,dlambda):
    '''
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
    '''
    return F0*Fstar*Fp0*area*Upsilon*throughput*dlambda


def calculate_CRbs(F0,Fstar,Istar,area,pixscale,throughput,dlambda):
    '''
    STELLAR LEAKAGE

    CRbs=F_0 * 10^{-0.4m_lambda} * zeta * PSF_{peak} * Omega * A * T * Deltalambda

    This simplifies as
    CRbs=F_0 * F_{star} *(zeta * PSF_{peak}) * A * Omega * T * Deltalambda

    in AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbsfactor = Fstar * oneopixscale2 * FATDL  # for stellar leakage count rate calculation
    Fstar = 10**(-0.4 * magstar)
    tempCRbsfactor = CRbsfactor * Istar_interp[index]
    

    # NOTE: Since Omega is not used when calculating the detector noise components, 
    # this multiplication is done outside the function when needed.
    # i.e.
    # CRbs = tempCRbsfactor * omega_lod[index2]

    '''
    return F0*Fstar*Istar*area*throughput*dlambda/(pixscale**2)


def calculate_CRbz(F0,Fzodi,lod_arcsec, skytrans,area, throughput, dlambda):
    '''
    LOCAL ZODI LEAKAGE

    CRbz=F_0* 10^{-0.4z}* Omega A T Delta lambda

    In AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbzfactor = Fzodi * lod_arcsec2 * FATDL  # count rate for zodi
    tempCRbzfactor = CRbzfactor * skytrans[index]
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    lod_arcsec2 = lod_arcsec * lod_arcsec

    # NOTE: Since Omega is not used when calculating the detector noise components, 
    # this multiplication is done outside the function when needed.
    # i.e.
    CRbz = tempCRbzfactor * omega_lod[index2];
    '''

    return (F0 * Fzodi * skytrans  * area * throughput * dlambda * lod_arcsec**2)


def calculate_CRbez(F0, Fexozodi, lod_arcsec, skytrans, area, throughput, dlambda, dist, sp):
    """
    EXOZODI LEAKAGE
    CRbez=F_0 * n * 10^{-0.4mag_{exozodi}} * Omega * A * T * Delta lambda

    In AYO:

    CRbezfactor = Fexozodi * lod_arcsec2 * FATDL / (dist * dist);
    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    tempCRbezfactor = CRbezfactor * skytrans[index] / (sp[iplanetpistartnp] * sp[iplanetpistartnp]);
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    lod_arcsec2 = lod_arcsec * lod_arcsec

    # NOTE: Since Omega is not used when calculating the detector noise components, 
    # this multiplication is done outside the function when needed.
    # i.e.
    CRbez = tempCRbezfactor * omega_lod[index2];

    """
    return (F0 * Fexozodi * skytrans  * area * throughput * dlambda * lod_arcsec**2) / (dist**2 * sp**2)


def calculate_CRbbin(F0,Fbinary, skytrans,area, throughput, dlambda):
    '''
    NEIGHBORING STARS LEAKAGE

    TBD

    CRbbin=F_0* 10^{-0.4mag_binary}* Omega A T Delta lambda

    In AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbbinfactor = Fbinary * FATDL  # count rate for scattered light from nearby stars
    tempCRbbinfactor = CRbbinfactor * skytrans[index]
    
    # NOTE: Since Omega is not used when calculating the detector noise components, 
    # this multiplication is done outside the function when needed.
    # i.e.
    # CRbbin = tempCRbbinfactor * omega_lod[index2];

    '''

    return (F0 * Fbinary * skytrans * area * throughput * dlambda )


def calculate_t_photon_count(lod_arcsec,det_pixscale_mas,det_npix_multiplier,det_omega_lod,det_CR):
    '''
    Calculates time 

    # According to Bernie Rauscher:
    # effective_dark_current = dark_current - f * cic * (1+W_-1[q/e])^-1.
    # If q = 0.99, (1+W_-1[q/e])^-1 = -6.73 such that
    # effective_dark_current = dark_current + f * cic * 6.73,
    # where f is the brightest pixel you care about in counts s^-1
    '''
    
    detpixscale_lod = det_pixscale_mas / (lod_arcsec * 1000.)
    det_npix = det_npix_multiplier * det_omega_lod /(detpixscale_lod**2)#  this is temporary to estimate the per pixel noise
    t_photon_count = 1. / (6.73 * (det_CR / det_npix))
    return t_photon_count


def calculate_CRbd(det_npix_multiplier,det_DC,det_RN,det_tread,det_CIC,t_photon_count,det_omega_lod,detpixscale_lod):
    '''
    DETECTOR NOISE

    CRbd = n_{pix}(xi +RN^2/tau_{exposure}+CIC/t_{photon_count})
    '''
    # calculate npix
    det_npix = det_npix_multiplier*det_omega_lod/(detpixscale_lod)**2
    return (det_DC + det_RN * det_RN / det_tread + det_CIC / t_photon_count) *det_npix


def calculate_CRnf(F0,Fstar,area,pixscale,throughput,dlambda,SNR,noisefloor):
    '''
    NOISE FLOOR
    Calculate the count rate of the noise floor.
    This should be the stddev (over the "noise region") of
    the difference of the photometric aperture-integrated
    stellar PSFs. The photometric aperture integration, and
    stddev of that have been calculated prior to the call to
    this function, and was then divided by the number of pixels
    in the photometric aperture. So here we calculate noise
    using the same method as the leaked starlight.
    The "noisefloor" array is equal to
    stddev(integral(Istar1,dphotometric_ap) - integral(Istar2,dphotometric_ap)) / (omega/(npix*npix))
    
    Reminder:
    self.noisefloor =parameters['noisefloor_factor']*self.contrast # 1 sigma systematic noise floor expressed as a contrast (uniform over dark hole and unitless) # scalar

    in AYO:

    FATDL = F0 * A_cm * throughput * deltalambda_nm;
    CRbsfactor = Fstar * oneopixscale2 * FATDL  # for stellar leakage count rate calculation
    Fstar = 10**(-0.4 * magstar)
    

    # NOTE: Since Omega is not used when calculating the detector noise components, 
    # this multiplication is done outside the function when needed.
    # i.e.
    # CRnoisefloor = tempCRnffactor * omega_lod[index2];

    '''    
    return SNR * (F0*Fstar*area*throughput*dlambda/(pixscale**2)) * noisefloor

def interpolate_arrays(Istar, noisefloor, npix, ndiams, stellar_diam_lod, angdiams):
    '''
    lod_arcsec = (lambda_ * 1e-6 / D) * 206264.806
    oneolod_arcsec = 1.0 / lod_arcsec
    stellar_diam_lod = angdiamstar_arcsec * oneolod_arcsec
    # Usage:
    # Assuming Istar and noisefloor are 3D NumPy arrays with shape (npix, npix, ndiams)
    # and angdiams is a 1D NumPy array
    Istar_interp, noisefloor_interp = interpolate_arrays(Istar, noisefloor, npix, ndiams, stellar_diam_lod, angdiams)

    '''
    Istar_interp = np.zeros((npix,npix))
    noisefloor_interp = np.zeros((npix,npix))

    k = np.searchsorted(angdiams, stellar_diam_lod)
    
    if k < ndiams:
        # Interpolation
        weight = (stellar_diam_lod - angdiams[k-1]) / (angdiams[k] - angdiams[k-1])
        Istar_interp = (1 - weight) * Istar[:, :, k-1] + weight * Istar[:, :, k]
        noisefloor_interp = (1 - weight) * noisefloor[:, :, k-1] + weight * noisefloor[:, :, k]
    else:
        # Extrapolation
        weight = (stellar_diam_lod - angdiams[k-1]) / (angdiams[k-1] - angdiams[k-2])
        Istar_interp = Istar[:, :, k-1] + weight * (Istar[:, :, k-1] - Istar[:, :, k-2])
        noisefloor_interp = noisefloor[:, :, k-1] + weight * (noisefloor[:, :, k-1] - noisefloor[:, :, k-2])

    # Ensure non-negative values
    Istar_interp = np.maximum(Istar_interp, 0)
    noisefloor_interp = np.maximum(noisefloor_interp, 0)

    return Istar_interp, noisefloor_interp


def measure_coronagraph_performance(psf_trunc_ratio, photap_frac, Istar_interp, skytrans, omega_lod,
                                    npix, xcenter, ycenter, oneopixscale_arcsec):
    '''
    # Usage:
    # Assuming all input arrays are NumPy arrays with appropriate shapes
    det_sep, det_Istar, det_skytrans, det_photap_frac, det_omega_lod = measure_coronagraph_performance(
        psf_trunc_ratio, photap_frac, Istar_interp, skytrans, omega_lod, npix, xcenter, ycenter, oneopixscale_arcsec
    )
    '''


    # Find psf_trunc_ratio closest to 0.3
    bestiratio = np.argmin(np.abs(psf_trunc_ratio - 0.3))

    # Find maximum photap_frac in first half of image
    maxphotap_frac = np.max(photap_frac[:npix//2, int(ycenter),bestiratio])

    # Find IWA
    row = photap_frac[:, int(ycenter),bestiratio]
    iwa_index = np.where(row[:int(xcenter)] > 0.5 * maxphotap_frac)[0][-1]
    det_sep_pix = abs((iwa_index + 0.5) - xcenter)
    det_sep = det_sep_pix / oneopixscale_arcsec

    # Calculate max values in 2-pixel annulus at det_sep
    y, x = np.ogrid[:npix, :npix]
    dist_from_center = np.sqrt((x - xcenter + 0.5)**2 + (y - ycenter + 0.5)**2)
    mask = np.abs(dist_from_center - det_sep_pix) < 2

    det_Istar = np.max(Istar_interp[mask])
    det_skytrans = np.max(skytrans[mask])

    photap_frac_masked = photap_frac[:,:,bestiratio][mask]
    det_photap_frac = np.max(photap_frac_masked)
    det_omega_lod = omega_lod[:,:,bestiratio][mask][np.argmax(photap_frac_masked)]

    return det_sep_pix,det_sep, det_Istar, det_skytrans, det_photap_frac, det_omega_lod



def calculate_exposure_time(observation,scene,telescope, coronagraph,detector,edith):
    ''''''

    for istar in range(scene.ntargs): #set to 1
        for ilambd in range(observation.nlambd): #set to 1

            # Calculate useful quantities
            Fstar = 10**(-0.4 * scene.mag[istar,ilambd])
            deltalambda_nm = np.min([(observation.lambd[ilambd] * 1000.0) / observation.SR[ilambd], coronagraph.bandwidth * (observation.lambd[ilambd] * 1000.0)]) # take the lesser of the desired bandwidth and what coronagraph allows
            lod_arcsec = (observation.lambd[ilambd] * 1e-6 / telescope.D) * 206264.806
            area_cm2=telescope.Area*100*100
            stellar_diam_lod = scene.angdiam_arcsec[istar] /lod_arcsec
            detpixscale_lod = detector.det_pixscale_mas / (lod_arcsec * 1000.)


            # Interpolate Istar, noisefloor based on angular diameter of the star (depends on the target) 
            # (reduces dimensionality from 3D arrays [npix,npix,angdiam] to 2D arrays [npix,npix].)
            # The interpolation is done based on the value of stellar_diam_lod (dependence on istar)
            Istar_interp,noisefloor_interp=interpolate_arrays(
                coronagraph.Istar, coronagraph.noisefloor, coronagraph.npix, 
                coronagraph.ndiams, stellar_diam_lod, coronagraph.angdiams)

            # Measure coronagraph performance at each IWA
            pixscale_rad = coronagraph.pixscale * (observation.lambd[ilambd] * 1e-6 / telescope.D)
            oneopixscale_arcsec = 1.0 / (pixscale_rad * 206264.806)

            # Measure coronagraph performance at each IWA
            det_sep_pix,det_sep, det_Istar, det_skytrans, det_photap_frac, det_omega_lod = measure_coronagraph_performance(
                coronagraph.psf_trunc_ratio, coronagraph.photap_frac, Istar_interp, coronagraph.skytrans, coronagraph.omega_lod,
                coronagraph.npix, coronagraph.xcenter, coronagraph.ycenter,oneopixscale_arcsec
            )

            # Here we calculate detector noise, as it may depend on count rates
            # We don't know the count rates yet, so we make estimates based on
            # values near the IWA

            # Detector noise from signal itself (we budget for 10x the planet count rate for the minimum detectable planet)
            det_CRp = calculate_CRp(scene.F0[ilambd], Fstar, 10 * 10**(-0.4 * scene.min_deltamag[istar]), area_cm2, 
                                    det_photap_frac, telescope.throughput[ilambd], deltalambda_nm)

            det_CRbs = calculate_CRbs(scene.F0[ilambd], Fstar, det_Istar, area_cm2, 
                                    coronagraph.pixscale, 
                                        telescope.throughput[ilambd], deltalambda_nm)

            det_CRbz = calculate_CRbz(scene.F0[ilambd], scene.Fzodi_list[istar,ilambd], 
                                    lod_arcsec, det_skytrans, area_cm2,
                                        telescope.throughput[ilambd], deltalambda_nm)
            
            det_CRbez = calculate_CRbez(scene.F0[ilambd], scene.Fexozodi_list[istar,ilambd], 
                                        lod_arcsec, det_skytrans, area_cm2, 
                                        telescope.throughput[ilambd], deltalambda_nm, 
                                        scene.dist[istar], det_sep)
            det_CRbbin = calculate_CRbbin(scene.F0[ilambd], scene.Fbinary_list[istar,ilambd], 
                                        det_skytrans, area_cm2,
                                            telescope.throughput[ilambd], deltalambda_nm)

            det_CR = det_CRp + det_CRbs + det_CRbz + det_CRbez + det_CRbbin

            for iorbit in np.arange(edith.norbits):

                for iphase in np.arange(edith.nmeananom):  
                    #calculate position of the planet in the image (from l/D to pixel)          
                    ix = scene.xp[iphase,iorbit,istar] * oneopixscale_arcsec + coronagraph.xcenter
                    iy = scene.yp[iphase,iorbit,istar] * oneopixscale_arcsec + coronagraph.ycenter
                    #calculate separation in arcsec
                    sp_lod = scene.sp[iphase,iorbit,istar] /lod_arcsec

                    # if planet is within the boundaries of the coronagraph simulation and hard IWA/OWA cutoffs...
                    if (ix >= 0) and (ix < coronagraph.npix) and (iy >= 0) and (iy < coronagraph.npix) and (sp_lod > coronagraph.IWA) and (sp_lod < coronagraph.OWA):

                        # besttp_v_ratio = float('inf')
                        # bestpsfomega = 0.0
                        # bestpsftruncratio = 0.0

                        for iratio in np.arange(coronagraph.npsfratios):
                            # First we just calculate CRp and CRnoisefloor to see if CRp > CRnoisefloor

                            # PLANET COUNT RATE CRP
                            CRp=calculate_CRp(scene.F0[ilambd], Fstar, scene.Fp0[iphase,iorbit,istar], area_cm2,
                                                coronagraph.photap_frac[int(np.floor(iy)), int(np.floor(ix)),iratio], telescope.throughput[ilambd],deltalambda_nm)

        
                            # NOISE FLOOR CRNF
                            CRnf = calculate_CRnf(scene.F0[ilambd], Fstar, area_cm2, 
                                coronagraph.pixscale, 
                                    telescope.throughput[ilambd], deltalambda_nm,observation.SNR[ilambd],
                                    noisefloor_interp[int(np.floor(iy)), int(np.floor(ix))])                      
                        
                            #multiply by omega at that point
                            CRnf *=coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio]
                            # NOTE: noisefloor_interp: technically the Y axis is rows and the X axis is columns, 
                            # that is why they are inverted
                            # NOTE: Evaluate if int(round(iy) is better than np.floor. Kept np.floor for consistency


                            # Check if it's above the noise floor and calculate exposure time if conditions are met
                            if CRp > CRnf and coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio] > detpixscale_lod**2:
                                # CALCULATE THE REST OF THE BACKGROUND NOISE
                                
                                # ## WHEN CALCULATING THE COUNT RATES, WE NEED TO MULTIPLY BY OMEGA_LOD i.e. 
                                # # THE SOLID ANGLE OF THE PHOTOMETRIC APERTURE
                                # Calculate CRbs
                                CRbs= calculate_CRbs(scene.F0[ilambd],Fstar,Istar_interp[int(np.floor(iy)), int(np.floor(ix))], area_cm2, 
                                                            coronagraph.pixscale, telescope.throughput[ilambd], deltalambda_nm)

                                # Calculate CRbz
                                CRbz=calculate_CRbz(scene.F0[ilambd],scene.Fzodi_list[istar,ilambd],lod_arcsec, 
                                                    coronagraph.skytrans[int(np.floor(iy)), int(np.floor(ix))], 
                                                    area_cm2,  telescope.throughput[ilambd], deltalambda_nm)


                                # Calculate CRbez
                                CRbez= calculate_CRbez(scene.F0[ilambd],scene.Fexozodi_list[istar,ilambd],lod_arcsec, 
                                                    coronagraph.skytrans[int(np.floor(iy)), int(np.floor(ix))], 
                                                        area_cm2,  telescope.throughput[ilambd], deltalambda_nm,
                                                        scene.dist[istar],scene.sp[iphase,iorbit,istar])


                                # Calculate CRbbin
                                CRbbin= calculate_CRbbin(scene.F0[ilambd],scene.Fbinary_list[istar,ilambd], 
                                                        coronagraph.skytrans[int(np.floor(iy)), int(np.floor(ix))],
                                                            area_cm2,  telescope.throughput[ilambd], deltalambda_nm)

                                # Calculate CRbd
                                t_photon_count=calculate_t_photon_count(lod_arcsec,detector.det_pixscale_mas,
                                                                        detector.det_npix_multiplier[ilambd],det_omega_lod,det_CR)

                                CRbd=calculate_CRbd(detector.det_npix_multiplier[ilambd],detector.det_DC[ilambd],detector.det_RN[ilambd],
                                                    detector.det_tread[ilambd],detector.det_CIC[ilambd],t_photon_count,det_omega_lod,detpixscale_lod)
                                
                                # TOTAL BACKGROUND NOISE
                                CRb = (CRbs+CRbz+CRbez+CRbbin)* coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio]
                                #Add detector noise
                                CRb += CRbd        

                                # EXPOSURE TIME
                                # count rate term
                                cp = ((CRp + 2*CRb) / (CRp * CRp - CRnf * CRnf)) # this includes the systematic noise floor term a la Bijan Nemati
                                #Exposure time
                                edith.exptime[istar,ilambd] = observation.SNR[ilambd] * observation.SNR[ilambd] * cp * telescope.toverhead_multi + telescope.toverhead_fixed          # record exposure time with overheads
                                
                                if edith.exptime[istar,ilambd] < 0:
                                    # time is past the systematic noise floor limit
                                    edith.exptime[istar,ilambd]=np.inf 
                                if edith.exptime[istar,ilambd] > edith.td_limit:
                                    #treat as unobservable if beyond exposure time limit
                                    edith.exptime[istar,ilambd]=np.inf 

                                if (coronagraph.nrolls != 1):
                                    # multiply by number of required rolls to achieve 360 deg coverage (after tlimit enforcement)
                                    edith.exptime[istar,ilambd] *= coronagraph.nrolls
                            else:
                                #It's below the systematic noise floor...
                                edith.exptime[istar,ilambd]= np.inf 
                    else:
                        #outside of the input contrast map or hard IWA/OWA cutoffs
                        edith.exptime[istar,ilambd]=np.inf
            
        # NOTE FOR FUTURE DEVELOPMENT
        # The nmeananom, norbits, npsfratios loops are not stored in the exptimematrix. 
        # This is not a problem right now since these are "fake" loops as of now (nmeananom, norbits, npsfratios all are 1).
        # But this might change in the future.
        return





def calculate_signal_to_noise(observation,scene,telescope, coronagraph,detector,edith):
    ''' '''

    for istar in range(scene.ntargs): #set to 1
        for ilambd in range(observation.nlambd): #set to 1

            # Calculate useful quantities
            Fstar = 10**(-0.4 * scene.mag[istar,ilambd])
            deltalambda_nm = np.min([(observation.lambd[ilambd] * 1000.0) / observation.SR[ilambd], coronagraph.bandwidth * (observation.lambd[ilambd] * 1000.0)]) # take the lesser of the desired bandwidth and what coronagraph allows
            lod_arcsec = (observation.lambd[ilambd] * 1e-6 / telescope.D) * 206264.806
            area_cm2=telescope.Area*100*100
            stellar_diam_lod = scene.angdiam_arcsec[istar] /lod_arcsec
            detpixscale_lod = detector.det_pixscale_mas / (lod_arcsec * 1000.)

            # Interpolate Istar, noisefloor based on angular diameter of the star (depends on the target) 
            # (reduces dimensionality from 3D arrays [npix,npix,angdiam] to 2D arrays [npix,npix].)
            # The interpolation is done based on the value of stellar_diam_lod (dependence on istar)
            Istar_interp,noisefloor_interp=interpolate_arrays(
                coronagraph.Istar, coronagraph.noisefloor, coronagraph.npix, 
                coronagraph.ndiams, stellar_diam_lod, coronagraph.angdiams)

            # Measure coronagraph performance at each IWA
            pixscale_rad = coronagraph.pixscale * (observation.lambd[ilambd] * 1e-6 / telescope.D)
            oneopixscale_arcsec = 1.0 / (pixscale_rad * 206264.806)

            # Measure coronagraph performance at each IWA
            det_sep_pix,det_sep, det_Istar, det_skytrans, det_photap_frac, det_omega_lod = measure_coronagraph_performance(
                coronagraph.psf_trunc_ratio, coronagraph.photap_frac, Istar_interp, coronagraph.skytrans, coronagraph.omega_lod,
                coronagraph.npix, coronagraph.xcenter, coronagraph.ycenter,oneopixscale_arcsec
            )

            # Here we calculate detector noise, as it may depend on count rates
            # We don't know the count rates yet, so we make estimates based on
            # values near the IWA

            # Detector noise from signal itself (we budget for 10x the planet count rate for the minimum detectable planet)
            det_CRp = calculate_CRp(scene.F0[ilambd], Fstar, 10 * 10**(-0.4 * scene.min_deltamag[istar]), area_cm2, 
                                    det_photap_frac, telescope.throughput[ilambd], deltalambda_nm)

            det_CRbs = calculate_CRbs(scene.F0[ilambd], Fstar, det_Istar, area_cm2, 
                                    coronagraph.pixscale, 
                                        telescope.throughput[ilambd], deltalambda_nm)

            det_CRbz = calculate_CRbz(scene.F0[ilambd], scene.Fzodi_list[istar,ilambd], 
                                    lod_arcsec, det_skytrans, area_cm2,
                                        telescope.throughput[ilambd], deltalambda_nm)
            
            det_CRbez = calculate_CRbez(scene.F0[ilambd], scene.Fexozodi_list[istar,ilambd], 
                                        lod_arcsec, det_skytrans, area_cm2, 
                                        telescope.throughput[ilambd], deltalambda_nm, 
                                        scene.dist[istar], det_sep)
            det_CRbbin = calculate_CRbbin(scene.F0[ilambd], scene.Fbinary_list[istar,ilambd], 
                                        det_skytrans, area_cm2,
                                            telescope.throughput[ilambd], deltalambda_nm)

            det_CR = det_CRp + det_CRbs + det_CRbz + det_CRbez + det_CRbbin


            for iorbit in np.arange(edith.norbits):
                for iphase in np.arange(edith.nmeananom):  
                    
                    #calculate position of the planet in the image (from l/D to pixel)          
                    ix = scene.xp[iphase,iorbit,istar] * oneopixscale_arcsec + coronagraph.xcenter
                    iy = scene.yp[iphase,iorbit,istar] * oneopixscale_arcsec + coronagraph.ycenter
                    #calculate separation in arcsec
                    sp_lod = scene.sp[iphase,iorbit,istar] /lod_arcsec

                    # if planet is within the boundaries of the coronagraph simulation and hard IWA/OWA cutoffs...
                    if (ix >= 0) and (ix < coronagraph.npix) and (iy >= 0) and (iy < coronagraph.npix) and (sp_lod > coronagraph.IWA) and (sp_lod < coronagraph.OWA):

                        for iratio in np.arange(coronagraph.npsfratios):
                            # First we just calculate CRp and CRnoisefloor to see if CRp > CRnoisefloor

                            # PLANET COUNT RATE CRP
                            CRp=calculate_CRp(scene.F0[ilambd], Fstar, scene.Fp0[iphase,iorbit,istar], area_cm2,
                                                coronagraph.photap_frac[int(np.floor(iy)), int(np.floor(ix)),iratio], telescope.throughput[ilambd],deltalambda_nm)

        
                            # NOISE FLOOR CRNF NOTE THIS TIME THIS IS JUST THE NOISE FACTOR RATIO (i.e. we assume SNR =1 so that we can use it for the snr calculation later)
                            CRnf_factor = calculate_CRnf(scene.F0[ilambd], Fstar, area_cm2, 
                                coronagraph.pixscale, 
                                    telescope.throughput[ilambd], deltalambda_nm,1,
                                    noisefloor_interp[int(np.floor(iy)), int(np.floor(ix))])                      
                        
                            #multiply by omega at that point
                            CRnf_factor *=coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio]
                            # NOTE: noisefloor_interp: technically the Y axis is rows and the X axis is columns, 
                            # that is why they are inverted
                            # NOTE: Evaluate if int(round(iy) is better than np.floor. Kept np.floor for consistency


                            # Check and calculate exposure time if conditions are met
                            if coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio] > detpixscale_lod**2:
                                # CALCULATE THE REST OF THE BACKGROUND NOISE
                                
                                # ## WHEN CALCULATING THE COUNT RATES, WE NEED TO MULTIPLY BY OMEGA_LOD i.e. 
                                # # THE SOLID ANGLE OF THE PHOTOMETRIC APERTURE
                                # Calculate CRbs
                                CRbs= calculate_CRbs(scene.F0[ilambd],Fstar,Istar_interp[int(np.floor(iy)), int(np.floor(ix))], area_cm2, 
                                                            coronagraph.pixscale, telescope.throughput[ilambd], deltalambda_nm)

                                # Calculate CRbz
                                CRbz=calculate_CRbz(scene.F0[ilambd],scene.Fzodi_list[istar,ilambd],lod_arcsec, 
                                                    coronagraph.skytrans[int(np.floor(iy)), int(np.floor(ix))], 
                                                    area_cm2,  telescope.throughput[ilambd], deltalambda_nm)


                                # Calculate CRbez
                                CRbez= calculate_CRbez(scene.F0[ilambd],scene.Fexozodi_list[istar,ilambd],lod_arcsec, 
                                                    coronagraph.skytrans[int(np.floor(iy)), int(np.floor(ix))], 
                                                        area_cm2,  telescope.throughput[ilambd], deltalambda_nm,
                                                        scene.dist[istar],scene.sp[iphase,iorbit,istar])


                                # Calculate CRbbin
                                CRbbin= calculate_CRbbin(scene.F0[ilambd],scene.Fbinary_list[istar,ilambd], 
                                                        coronagraph.skytrans[int(np.floor(iy)), int(np.floor(ix))],
                                                            area_cm2,  telescope.throughput[ilambd], deltalambda_nm)

                                # Calculate CRbd
                                t_photon_count=calculate_t_photon_count(lod_arcsec,detector.det_pixscale_mas,
                                                                        detector.det_npix_multiplier[ilambd],det_omega_lod,det_CR)

                                CRbd=calculate_CRbd(detector.det_npix_multiplier[ilambd],detector.det_DC[ilambd],detector.det_RN[ilambd],
                                                    detector.det_tread[ilambd],detector.det_CIC[ilambd],t_photon_count,det_omega_lod,detpixscale_lod)
                                
                                # TOTAL BACKGROUND NOISE
                                CRb = (CRbs+CRbz+CRbez+CRbbin)* coronagraph.omega_lod[int(np.floor(iy)), int(np.floor(ix)),iratio]
                                #Add detector noise
                                CRb += CRbd        

                                # SIGNAL-TO-NOISE
                                # time term
                                time_factors= (edith.obstime* coronagraph.nrolls - telescope.toverhead_fixed)/(telescope.toverhead_multi*((CRp + 2*CRb)))

                                # Signal-to-noise
                                edith.fullsnr[istar,ilambd]=np.sqrt((time_factors * CRp**2)/(1 + time_factors * CRnf_factor**2))

                                if edith.fullsnr[istar,ilambd] < 0:
                                    # time is past the systematic noise floor limit
                                    edith.fullsnr[istar,ilambd]=0
                                if edith.fullsnr[istar,ilambd] > 100:
                                    #treat as unobservable if beyond exposure time limit
                                    edith.fullsnr[istar,ilambd]=100

                            else:
                                #It's below the systematic noise floor...
                                edith.fullsnr[istar,ilambd] = np.inf 
                    else:
                        #outside of the input contrast map or hard IWA/OWA cutoffs
                        edith.fullsnr[istar,ilambd]=np.inf
            
        # NOTE FOR FUTURE DEVELOPMENT
        # The nmeananom, norbits, npsfratios loops are not stored in the fullsnr matrix. 
        # This is not a problem right now since these are "fake" loops as of now (nmeananom, norbits, npsfratios all are 1).
        # But this might change in the future.
        return




