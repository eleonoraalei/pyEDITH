from typing import Union
import numpy as np 
from scipy.interpolate import interp1d,interpn

# def calcBnu(temp:float, lambd: Union[float,np.array]) -> Union[float,np.array]:
#     # TODO UNUSED????
#     # Calculates the Planck Law in CGS units
#     #
#     # Input:
#     # temp = temperature in Kelvin
#     # lambd = wavelength in microns
#     #
#     # Output:
#     # result = Bnu in erg cm^-2 steradian^-1
#     #
#     # Created by Chris Stark
#     # University of Maryland Physics Dept/Goddard Space Flight Center
#     # starkc@umd.edu
#     # Last updated 08/09/07 by Chris Stark

#     const1 = 3.972895E-4 #const1 = 2 * h * c * (1E6)^3 (in CGS w/ correction for microns)
#     const2 = 1.438769E4 #const2 = h * c * 1E6 / k (in CGS w/ correction for microns)
#     return const1 / (lambd^3.0 * (np.exp(const2 / (lambd * temp)) - 1))
  
# def calcblambd(temp:float, lambd: Union[float,np.array]) -> Union[float,np.array]:
#     # TODO UNUSED????
#     # Calculates the Planck Law (per unit wavelength) in CGS units
#     #
#     # Input:
#     # temp = temperature in Kelvin
#     # lambd = wavelength in microns
#     #
#     # Output:
#     # result = Bnu in erg cm^-3 s^-1 steradian^-1
#     #
#     # Created by Chris Stark
#     # Goddard Space Flight Center
#     # christopher.c.stark@nasa.gov
#     # Last updated 01/28/15 by Chris Stark

#     const1 = 1.1910439E15 #const1 = 2 * h * c^2 * (1E4)^5 (in CGS w/ correction for microns)
#     const2 = 1.438769E4 #const2 = h * c * 1E4 / k (in CGS w/ correction for microns)
#     return const1 / (lambd^5.0 * (np.exp(const2 / (lambd * temp)) - 1))



def calc_flux_zero_point(lambd: np.array, unit: str = '',
                         perlambd:bool=False,AB:bool=False,verbose:bool=False) -> np.array:
    '''
    This code calculates the flux zero point for a given wavelength

    By default, it returns Johnson zero points
    By setting the /ab keyword, it will return AB flux zero point

    The Johnson zero points come from Table A.2 from the Spitzer Telescope Handbook at
    http://irsa.ipac.caltech.edu/data/SPITZER/docs/spitzermission/missionoverview/spitzertelescopehandbook/19/
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
    3.6308e-20

    INPUT
    lambd (microns)
    jy: flag to make the output value in Jy
    cgs: flag to make the output value in erg s^-1 cm^-2 Hz^-1
    pcgs: flag to make the output value in photons s^-1 cm^-2 Hz^-1
    perlambd: flag to convert all output values to cm^-1 instead of Hz^-1


    #OUTPUT
    #f0_jy (flux zero point in Jy)
    #f0_photons_cgs (flux zero point in photons s^-1 cm^-2 Hz^-1)

    '''
    # TODO Check change, it should be now unnecessary because now False by default
    # if n_elements(jy) eq 0 then jy = 0
    # if n_elements(cgs) eq 0 then cgs = 0
    # if n_elements(pcgs) eq 0 then pcgs = 0
    # if n_elements(perlambd) eq 0 then perlambd = 0
    # if n_elements(verbose) eq 0 then verbose = 0
    if unit =='':
        raise ValueError('Must specificy output units: unit="jy" for output in Jy, '+
                         'unit="cgs" for output in erg s^-1 cm^-2 Hz^-1, '+
                         'unit="pcgs" for output in photons s^-1 cm^-2 Hz^-1. ')
    if unit not in ['jy','cgs','pcgs']:
        raise ValueError('Invalid unit value. Possible values are: unit="jy" for output in Jy, '+
                         'unit="cgs" for output in erg s^-1 cm^-2 Hz^-1, '+
                         'unit="pcgs" for output in photons s^-1 cm^-2 Hz^-1. ')
    # TODO: Check: should no longer be necessary.
    # if (jy + cgs + pcgs) gt 1 then stop,'Must specificy only one unit flag.'
    if unit=='jy' and perlambd==True:
        raise ValueError('Cannot set Jy and perlambd') 


    #Calculate the zero point
    if AB:
        #AB magnitude system
        f0 = 3.6308e3 #This is the zero point Jy
    else:
        #Johnson magnitude system
        known_lambd = np.array([0.36,0.44,0.55,0.71,0.97,1.25,1.60,2.22,3.54,4.80,10.6,21.0])
        known_zeropoint_jy = np.array([1823,4130,3781,2941,2635,1603,1075,667,288,170,36,9.4])
        '''
        # CREATED FINER GRID TO IMPROVE PYTHON INTERPOLATION

        known_lambd=np.array([0.360000,0.375094,0.390821,0.407207,0.424280, 0.44,
                              0.442069,0.460604,0.479916,0.500038,0.521003,
                              0.542848,0.55, 0.565608,0.589323,0.614031,0.639776,
                              0.666601,0.694550,0.71, 0.723670,0.754012,0.785626,
                              0.818565,0.852886,0.888645,0.925904,0.964725, 0.97,
                              1.00517, 1.04732, 1.09123, 1.13698, 1.18465, 
                              1.23432, 1.25, 1.28608, 1.34000, 1.39618, 1.45472, 
                              1.51571, 1.57926,1.60, 1.64548, 1.71447, 1.78635,
                              1.86125, 1.93929, 2.02060, 2.10531, 2.19358, 2.22,
                              2.28556, 2.38138, 2.48123, 2.58526, 2.69366, 
                              2.80659, 2.92427, 3.04688, 3.17462, 3.30773,
                              3.44641, 3.54, 3.59091, 3.74147, 3.89834, 4.06179,
                              4.23209, 4.40953, 4.59442, 4.78705,4.8, 4.98776, 
                              5.19688, 5.41477, 5.64180, 5.87835, 6.12481, 
                              6.38161, 6.64918, 6.92796, 7.21844, 7.52109, 
                              7.83643, 8.16499, 8.50733, 8.86402, 9.23567, 
                              9.62290, 10.0264, 10.4467, 10.6, 10.8848, 11.3411, 
                              11.8166, 12.3121, 12.8283, 13.3661, 13.9266,
                              14.5105, 15.1189, 15.7528, 16.4132, 17.1014, 
                              17.8184, 18.5655, 19.3439, 20.1549, 21.0000])

        known_zeropoint_jy=np.array([1823.00,2309.47,2823.45,3331.13,3792.66,4130.,
                                     4167.16,4418.53,4521.24,4464.58,4254.47,
                                     3912.49,3781., 3707.21,3588.95,3460.09,3322.04,
                                     3176.30,3024.39,2941.,2903.20,2831.21,2771.39,
                                     2723.06,2685.63,2658.69,2641.93,2635.16,2635.,
                                     2512.17,2356.43,2189.25,2014.52,1836.04,
                                     1657.41,1603.,1524.38,1420.41,1326.57,1241.78,
                                     1165.10,1095.66,1075.,1029.55,967.160,909.412,
                                     855.923,806.341,760.353,717.667,678.019, 667.,
                                     636.448,594.955,555.360,517.645,481.789,
                                     447.766,415.540,385.073,356.321,329.236,
                                     303.768,288.,280.864,261.361,243.286,226.529,
                                     210.989,196.574,183.199,170.785,170., 158.639,
                                     147.218,136.524,126.518,117.162,108.423,
                                     100.265,92.6557,85.5639,78.9595,72.8138,
                                     67.0993,61.7899,56.8608,52.2882,48.0495,
                                     44.1235,40.4899,37.1294,36., 34.1755,31.5292,
                                     29.0876,26.8347,24.7561,22.8382,21.0686,
                                     19.4360,17.9297,16.5399,15.2577,14.0748,
                                     12.9834,11.9765,11.0476,10.1906,9.39999])
        '''
        #Now we interpolate to lambd
        #Note that the interpolation is best done in log-log space
        interp=interp1d(np.log10(known_lambd),np.log10(known_zeropoint_jy), kind='cubic',fill_value='extrapolate') # TODO this interpolation is not exactly the same, is that okay?
        logf0=interp(np.log10(lambd))
        f0 = 10.**logf0 #undo the logarithm



    #Now change the units from Jy if necessary
    c = 2.998e10                  #cm s^-1
    h = 6.62608e-27               #planck constant in cgs
    lambd_cm = lambd / 1e4
    ephoton_cgs = h * c/ lambd_cm
    if unit=='cgs': f0 = np.double(f0)*1e-23 #convert to CGS
    if unit=='pcgs': f0 = np.double(f0)*1e-23 / ephoton_cgs #convert to # photons in CGS
    if perlambd: f0 *= (c / lambd_cm**2.)


    if unit=='jy': unittext = 'Jy'
    if unit=='cgs':
        if not perlambd: unittext = 'erg s^-1 cm^-2 Hz^-1 (CGS per unit frequency)'
        else: unittext = 'erg s^-1 cm^-3 (CGS per unit wavelength)'
    if unit=='pcgs':
        if not perlambd: unittext = 'photons s^-1 cm^-2 Hz^-1 (CGS per unit frequency)'
        else: unittext = 'photons s^-1 cm^-3 (CGS per unit wavelength)'

    if verbose: print('Johnson flux zero point calculated at %.2f' % lambd
                      +' microns in units of '+unittext)

    return np.array(f0)

def calc_exozodi_flux(M_V, vmag, F0V, nexozodis, lambd: np.array, lambdmag, F0lambd):

    #Inputs
    #M_V  = V band absolute magnitude of stars
    #vmag = V band apparent magnitude of stars
    #F0V  = V band flux zero point of stars
    #Lstar = bolometric luminosity of stars
    #lambd = wavelength in microns (vector of length nlambd)
    #lambdmag = apparent magnitude of stars at each lambd (array nstars x nlambd)
    #F0lambd = flux zero point at wavelength lambd (vector of length nlambd)

    #Output
    #exozodi surface brightness photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0
    #i.e., it returns 10^(-0.4*magOmega_EZ)
    # - multiply by F0 to get photons s^-1 cm^-2 arcsec^-2 nm^-1
    # - multiply by energy of photons to get erg s^-1 cm^-2 arcsec^-2 nm^-1

    if len(lambd) != len(F0lambd): raise ValueError('ERROR. F0lambd and lambd must be vectors of identical length.')
    if len(lambd)>1:
        if len(lambd)!=lambdmag.shape[0]: raise ValueError('ERROR. lambdmag must have dimensions nstars x nlambd, where nlambd is the length of lambd.') #TODO check because IDL and PYTHON are flipped

    vmag_1zodi = 22. #V band surface brightness of 1 zodi in mag arcsec^-2
    vflux_1zodi = 10.**(-0.4*vmag_1zodi) #V band flux (modulo the F0 factor out front)

    M_V_sun = 4.83 #V band absolute magnitude of Sun

    #Calculate counts s^-1 cm^-2 arcsec^-2 nm^-1 @ V band
    #The following formula maintains a constant optical depth regardless
    #of stellar type
    #NOTE: the older version of the code used the following expression
    #flux_exozodi_Vband = F0V * (double(10.)^(-0.4*(M_V - M_V_sun)) / Lstar) * (nexozodis * vflux_1zodi)
    #The above expression has a 1/Lstar in it, which corrects for the
    #1/r^2 factor based on the EEID.  In the new version of exposure_time_calculator.c,
    #we explicitly divide by 1/r^2, so the new version of the
    #eflux_exozodi_Vband expression does not divide by Lstar.
    flux_exozodi_Vband = F0V * np.double(10.)**(-0.4*(M_V - M_V_sun)) * (nexozodis * vflux_1zodi) #vector of length nstars

    #Now, multiply by the ratio of the star's counts received at lambd to those received at V band
    nstars = len(vmag)
    nlambd = len(lambd)
    flux_exozodi_lambd = np.full((nlambd,nstars), flux_exozodi_Vband) #congrid([[flux_exozodi_Vband],[flux_exozodi_Vband]],nstars,nlambd)
    
    # TODO in python it needs to be treated differently if it is 1D or 2D array. Check better solution?
    # if nstars>1:
    for ilambd in np.arange(nlambd):
        flux_exozodi_lambd[ilambd,:] *= (F0lambd[ilambd] * np.double(10.)**(-0.4*lambdmag[ilambd,:])) / (F0V * np.double(10.)**(-0.4*vmag))
    # else:
    #   for ilambd in np.arange(nlambd):
    #     flux_exozodi_lambd[ilambd] *= (F0lambd[ilambd] * np.double(10.)**(-0.4*lambdmag[ilambd])) / (F0V * np.double(10.)**(-0.4*vmag))
  
    #Now, because we return just the quantity 10.^(-0.4*magOmega_EZ), we remove the F0...
    for ilambd in np.arange(nlambd):
        flux_exozodi_lambd[ilambd,:] /= F0lambd[ilambd]
    
    return flux_exozodi_lambd #TODO. Unclear what shape is this supposed to have

def calc_zodi_flux(dec:np.array, ra:np.array, lambd:np.array, F0:np.array, starshade:bool = False, ss_elongation:np.array = np.array([])):
    
    #Inputs
    #Dec = declination of target in degrees (J2000 equatorial coordinate)
    #RA = right ascension of target in degrees (J2000 equatorial coordinate)
    #lambd = wavelength in microns (vector of length nlambd)
    #starshade = setting this flag turns on starshade mode
    #ss_elongation = mean solar elongation for the starshade's observations (degrees)

    #Output
    #zodi surface brightness photons s^-1 cm^-2 arcsec^-2 nm^-1 / F0
    #i.e., it returns 10^(-0.4*magOmega_ZL)
    # - multiply by F0 to get photons s^-1 cm^-2 arcsec^-2 nm^-1
    # - multiply by energy of photons to get erg s^-1 cm^-2 arcsec^-2 nm^-1
    #output has dimensions nstars x nlambd


    if len(lambd) != len(F0): raise ValueError('ERROR. F0 and lambd must be vectors of identical length.')
    if (starshade == True) and (len(ss_elongation) == 0): raise ValueError('ERROR. You have set the STARSHADE flag.  Must specify SS_ELONGATION in degrees.')
    if (starshade == False) and (len(ss_elongation) > 0): raise ValueError('ERROR. You must enable STARSHADE mode if you are setting SS_ELONGATION in degrees.')

    #Convert to radians internally
    dec_rad = dec*(np.double(np.pi)/180.)
    ra_rad = ra*(np.double(np.pi)/180.)

    #First, convert equatorial coordinates to ecliptic coordinates
    #This is nicely summarized in Leinert et al. (1998)
    eps = 23.439 #J2000 obliquity of the ecliptic in degrees
    eps_rad = eps*(np.double(np.pi)/180.) #convert to radians
    sinbeta = np.sin(dec_rad) * np.cos(eps_rad) - np.cos(dec_rad) * np.sin(eps_rad) * np.sin(ra_rad) #all we need is the sine of the latitude
    sinbeta0 = sinbeta              #contains the +/- values of beta
    sinbeta = abs(sinbeta)          #f135 below is symmetric about beta=0
    sinbeta2 = sinbeta*sinbeta
    sinbeta3 = sinbeta2*sinbeta

    #print,'beta = ',asin(sinbeta)*180./pi

    # COMMENTED OUT SINCE THIS IS NOT REALLY USED (EXCEPT FROM ONE SPECIFIC ROW/COLUMN) NOTE: assumes the same degree of observation so far
    #Here are the solar longitude and beta values for Table 17 from
    #Leinert et al. (1998)
    beta_vector = np.array([0.,5,10,15,20,25,30,45,60,75])
    beta_array = np.full((20,10),beta_vector)
    beta_array = beta_array[0:18,:] # TODO Why though?
    sollong_vector = np.array([0,5,10,15,20,25,30,35,40,45,60,75,90,105,120,135,150,165,180.])
    sollong_array = np.full((10,19),sollong_vector).T # rotates so that we get 19 rows x 10 columns, and the rows have increasing values according to sollong_vector

    #Here are the values in Table 17
    table17=np.array([
    [-1,  	-1, 	-1,	    3140,	1610,	985,	640,	275,	150,	100],
    [-1,	-1,	    -1,	    2940,	1540,	945,	625,	271,	150,	100],
    [-1,	-1,	    4740,	2470,	1370,	865,	590,	264,	148,	100],
    [11500,	6780,	3440,	1860,	1110,	755,	525,	251,	146,	100],
    [6400,	4480,	2410,	1410,	910,	635,	454,	237,	141,	99],
    [3840,	2830,	1730,	1100,	749,	545,	410,	223,	136,	97],
    [2480,	1870,	1220,	845,	615,	467,	365,	207,	131,	95],
    [1650,	1270,	910,	680,	510,	397,	320,	193,	125,	93],
    [1180,	940,	700,	530,	416,	338,	282,	179,	120,	92],
    [910,	730,	555,	442,	356,	292,	250,	166,	116,	90],
    [505,	442,	352,	292,	243,	209,	183,	134,	104,	86],
    [338,	317,	269,	227,	196,	172,	151,	116,	93,	    82],
    [259,	251,	225,	193,	166,	147,	132,	104,	86,	79],
    [212,	210,	197,	170,	150,	133,	119,	96,	82,	77],
    [188,	186,	177,	154,	138,	125,	113,	90,	77,	74],
    [179,	178,	166,	147,	134,	122,	110,	90,	77,	73],
    [179,	178,	165,	148,	137,	127,	116,	96,	79,	72],
    [196,	192,	179,	165,	151,	141,	131,	104,	82,	72],
    [230,	212,	195,	178,	163,	148,	134,	105,	83,	72]])


    #For a coronagraph, we assume star can be observed near sollong ~ 135 degrees
    #The old f135 calculation.  Now we interpolate the above table.
    #I135_o_I90 = 0.69
    #f135 = 1.02331 - 0.565652*sinbeta - 0.883996*sinbeta2 + 0.852900*sinbeta3
    #f = I135_o_I90 * f135
    j = np.argmin(abs(sollong_vector-135.))
    k = np.argmin(abs(sollong_vector-90.))
    interp =interp1d(np.sin(beta_vector*np.pi/180.),table17[j]/table17[k,0],kind='cubic',fill_value='extrapolate') 
    f =interp(sinbeta)

    '''
    # REPLACED WITH A SIMPLIFIED VERSION THAT YIELDS THE SAME RESULT (I just copy the single row/column instead)

    # #### np.sin(beta_vector*np.pi/180)
    # x= [0.,0.08715574,0.17364818, 0.25881905, 0.34202014, 0.42261826,0.5,0.70710678,0.8660254,0.96592583]
    
    # #### table17[j]/table17[k,0] where j and k correspond to the indices of the row at 135. and the column at 90 (see above)
    # y=[0.69111969,0.68725869,0.64092664,0.56756757,0.51737452,0.47104247,0.42471042,0.34749035,0.2972973, 0.28185328]
   
    # interp =interp1d(x,y,fill_value='extrapolate') 
    # f =interp(sinbeta)
    '''

    #### STARSHADE functionality not enabled ####
    #For a starshade, we assume a mean solar elongation of 59 degrees
    # if starshade==True: 
    #     # TODO: not sure this works, but it is not used

    #     #First, calculate inclination to point at beta of all targets
    #     inclination = np.arcsin(sinbeta0.copy() / np.sin(ss_elongation*np.pi/180.))*180./np.pi
    #     cossollong = np.cos(ss_elongation*np.pi/180.)/np.cos(np.arcsin(sinbeta.copy())) #solar longitude of targets at elongation of ss_elongation
    #     sinsollong = np.cos(inclination*np.pi/180.) * np.sin(ss_elongation*np.pi/180.) / np.cos(np.arcsin(sinbeta0.copy()))
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
    

    ''' zodi_lambd=np.array([  0.2     ,   0.213682,   0.2283  ,   0.243919,   0.260605,
         0.278434,   0.297482,   0.317833,   0.339576,   0.362807,
         0.387627,   0.414145,   0.442477,   0.472747,   0.505088,
         0.539642,   0.576559,   0.616002,   0.658143,   0.703168,
         0.751272,   0.802667,   0.857579,   0.916247,   0.978928,
         1.0459  ,   1.11745 ,   1.19389 ,   1.27557 ,   1.36283 ,
         1.45607 ,   1.55568 ,   1.6621  ,   1.77581 ,   1.89729 ,
         2.02709 ,   2.16576 ,   2.31393 ,   2.47222 ,   2.64135 ,
         2.82205 ,   3.01511 ,   3.22138 ,   3.44175 ,   3.67721 ,
         3.92877 ,   4.19754 ,   4.4847  ,   4.7915  ,   5.11929 ,
         5.46951 ,   5.84368 ,   6.24345 ,   6.67057 ,   7.12692 ,
         7.61448 ,   8.13539 ,   8.69194 ,   9.28656 ,   9.92187 ,
        10.6006  ,  11.3258  ,  12.1006  ,  12.9285  ,  13.8129  ,
        14.7579  ,  15.7675  ,  16.8461  ,  17.9986  ,  19.2299  ,
        20.5454  ,  21.951   ,  23.4527  ,  25.0571  ,  26.7713  ,
        28.6027  ,  30.5595  ,  32.65    ,  34.8837  ,  37.2701  ,
        39.8198  ,  42.5439  ,  45.4544  ,  48.564   ,  51.8863  ,
        55.4359  ,  59.2283  ,  63.2802  ,  67.6092  ,  72.2345  ,
        77.176   ,  82.4558  ,  88.0966  ,  94.1234  , 100.563   ,
       107.442   , 114.792   , 122.645   , 131.036   , 140.      ])
    zodi_blambd=np.array([5.88232e-19, 1.05285e-18, 1.82390e-18, 3.05813e-18, 4.96284e-18,
       7.79505e-18, 1.18503e-17, 1.74365e-17, 2.48317e-17, 3.42271e-17,
       4.56621e-17, 5.60664e-17, 6.17846e-17, 6.33569e-17, 6.12456e-17,
       6.08544e-17, 5.90367e-17, 5.59199e-17, 5.17153e-17, 4.67789e-17,
       4.25436e-17, 3.81568e-17, 3.37492e-17, 3.00493e-17, 2.85588e-17,
       2.64107e-17, 2.30944e-17, 1.93543e-17, 1.65760e-17, 1.41858e-17,
       1.20838e-17, 1.02453e-17, 8.64611e-18, 7.26259e-18, 6.07207e-18,
       5.05308e-18, 4.18551e-18, 3.51491e-18, 2.96757e-18, 2.50595e-18,
       2.11657e-18, 1.78805e-18, 1.51082e-18, 1.27683e-18, 1.27737e-18,
       1.42399e-18, 1.68283e-18, 2.10823e-18, 2.79991e-18, 3.30562e-18,
       3.86924e-18, 4.50812e-18, 5.22842e-18, 6.03593e-18, 6.93617e-18,
       7.93407e-18, 9.03390e-18, 1.02389e-17, 1.15514e-17, 1.29722e-17,
       1.45009e-17, 1.61354e-17, 1.76808e-17, 1.77822e-17, 1.75868e-17,
       1.71043e-17, 1.63582e-17, 1.53846e-17, 1.42282e-17, 1.29399e-17,
       1.15725e-17, 1.01775e-17, 8.80174e-18, 7.49285e-18, 6.46752e-18,
       5.51824e-18, 4.65414e-18, 3.88015e-18, 3.19767e-18, 2.60492e-18,
       2.09762e-18, 1.66968e-18, 1.31376e-18, 1.02181e-18, 7.85588e-19,
       5.97034e-19, 4.48511e-19, 3.54356e-19, 2.83744e-19, 2.27068e-19,
       1.81605e-19, 1.45158e-19, 1.15957e-19, 9.25754e-20, 7.35820e-20,
       5.57304e-20, 4.16982e-20, 3.08216e-20, 2.25062e-20, 1.62353e-20])
    '''
    # wavelength dependence
    # The following comes from fits to Table 19 in Leinert et al 1998
    ## COMMENTED TO INCREASE RESOLUTION OF THE GRID (the larger grid already does all conversions)
    zodi_lambd = np.array([0.2,0.3,0.4,0.5,0.7,0.9,1.0,1.2,2.2,3.5,4.8,12,25,60,100,140])  #microns
    zodi_blambd = np.array([2.5e-8,5.3e-7,2.2e-6,2.6e-6,2.0e-6,1.3e-6,1.2e-6,8.1e-7,1.7e-7,5.2e-8,1.2e-7,7.5e-7,3.2e-7,1.8e-8,3.2e-9,6.9e-10]) #W m^-2 sr^-1 micron^-1
    #convert the above to erg s^-1 cm^-2 arcsec^-2 nm^-1
    zodi_blambd *= 1.e7 #convert W to erg s^-1
    zodi_blambd /= 1.e4 #convert m^-2 to cm^-2
    zodi_blambd /= 4.25e10 #convert sr^-1 to arcsec^-2
    zodi_blambd /= 1000.   #convert micron^-1 to nm^-1


    interp= interp1d(np.log10(zodi_lambd),np.log10(zodi_blambd), kind='cubic')
    blambd=interp(np.log10(lambd))
    blambd = 10.**blambd
    I90fabsfco = blambd

    #Now divide by energy of a photon in erg
    c = 2.998e10 #cm s^-1
    h = 6.62608e-27 #planck constant in cgs
    ephoton = h * c/(lambd/1e4)
    I90fabsfco /= ephoton
    I90fabsfco /= F0

    #Multiply by the wavelength dependence for each value of lambd
    nstars = len(ra)
    nlambd = len(lambd)

    flux_zodi = np.full((nlambd,nstars),f)
    for ilambd in range(nlambd):
        flux_zodi[ilambd,:] *= I90fabsfco[ilambd]

    mag_zodi = -2.5*np.log10(flux_zodi)


    return flux_zodi


