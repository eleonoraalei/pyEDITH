# pyEDITH
Exposure Direct Imaging Timer for HWO (Python Version)


## Installation
Clone the pyEDITH repository:

```
git clone https://github.com/eleonoraalei/pyEDITH.git
cd pyEDITH
```

Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the package:
```
pip install -e .
```
Set up environment variables: Add the following lines to your `.bashrc` or `.zshrc` file:

```
export SCI_ENG_DIR="/path/to/Sci-Eng-Interface/hwo_sci_eng"
export YIP_CORO_DIR="/path/to/yips"
```
Replace the paths with the actual paths on your system.



## Running pyEDITH via Terminal

pyEDITH provides a command-line interface with three main functionalities: `etc` (Exposure Time Calculator), `snr` (Signal-to-Noise Ratio), and `etc2snr` (Exposure Time to Signal-to-Noise Ratio). You will need to compile a configuration file that the code will read. You can find some examples in `inputs/`.

- Exposure Time Calculation:
```
pyedith etc --edith path/to/your/config.edith
```

- SNR Calculation:

```
pyedith snr --edith path/to/your/config.edith --time 100000
```
Here, 100000 is the exposure time in seconds.

- ETC to SNR Calculation:
```
pyedith etc2snr --edith path/to/your/config.edith
```
In this case, the config file will need to also have a secondary set of parameters (see e.g. `inputs/input_secondarybandpass.edith`)

Add the -v flag to any command for verbose output:

```
pyedith etc --edith path/to/your/config.edith -v
```


## Running pyEDITH from Python

This mode offers much more flexibility to run the ETC. We refer to our tutorials in `tutorials/` for details on this mode.


# Glossary

<!---
| Variable Name           | Unit                         | Length                   | Meaning                                           |     |
| ----------------------- | ---------------------------- | ------------------------ | ------------------------------------------------- | --- |
| Istar                   | Dimensionless                | [npix, npix]             | Star intensity distribution (on-axis PSF)         |     |
| noisefloor              | Dimensionless                | [npix, npix]             | Noise floor of the coronagraph                    |     |
| photometric_aperture_throughput             | Dimensionless                | [npix, npix, npsfratios] | Photometric aperture throughput                     |     |
| omega_lod               | (λ/D)²                       | [npix, npix, npsfratios] | Solid angle of the photometric aperture           |     |
| skytrans                | Dimensionless                | [npix, npix]             | Sky transmission                                  |     |
| pixscale                | λ/D                          | Scalar                   | Pixel scale of the coronagraph                    |     |
| npix                    | Integer                      | Scalar                   | Number of pixels in the image                     |     |
| xcenter                 | Pixel                        | Scalar                   | X-coordinate of the image center                  |     |
| ycenter                 | Pixel                        | Scalar                   | Y-coordinate of the image center                  |     |
| bandwidth               | Dimensionless                | Scalar                   | Fractional bandwidth of coronagraph               |     |
| npsfratios              | Integer                      | Scalar                   | Number of PSF ratios                              |     |
| nrolls                  | Integer                      | Scalar                   | Number of roll angles                             |     |
| nchannels               | Integer                      | Scalar                   | Number of channels                                |     |
| minimum_IWA             | λ/D                          | Scalar                   | Minimum Inner Working Angle                       |     |
| maximum_OWA             | λ/D                          | Scalar                   | Maximum Outer Working Angle                       |     |
| coronagraph_optical_throughput  | Dimensionless                | [nlambda]                | Throughput for all coronagraph optics             |     |
| diameter                | Length                       | Scalar                   | Circumscribed diameter of telescope aperture      |     |
| Area                    | Length²                      | Scalar                   | Effective collecting area of telescope            |     |
| unobscured_area         | Dimensionless                | Scalar                   | Unobscured area percentage                        |     |
| toverhead_fixed         | Time                         | Scalar                   | Fixed overhead time                               |     |
| toverhead_multi         | Dimensionless                | Scalar                   | Multiplicative overhead time                      |     |
| telescope_optical_throughput    | Dimensionless                | [nlambda]                | Optical throughput of telescope                   |     |
| temperature             | Temperature                  | Scalar                   | Temperature of the warm optics                    |     |
| T_contamination                 | Dimensionless                | Scalar                   | Effective throughput factor for contamination     |     |
| pixscale_mas            | Milliarcsecond               | Scalar                   | Detector pixel scale                              |     |
| npix_multiplier         | Dimensionless                | [nlambda]                | Number of detector pixels per image plane "pixel" |     |
| DC                      | Electron / (Pixel * Second)  | [nlambda]                | Dark current                                      |     |
| RN                      | Electron / (Pixel * Read)    | [nlambda]                | Read noise                                        |     |
| tread                   | Second                       | [nlambda]                | Read time                                         |     |
| CIC                     | Electron / (Pixel * Photon)  | [nlambda]                | Clock-induced charge                              |     |
| QE                      | Electron / Photon            | [nlambda]                | Quantum efficiency of detector                    |     |
| dQE                     | Dimensionless                | [nlambda]                | Effective QE due to degradation                   |     |
| wavelength              | Length                       | [nlambda]                | Observation wavelengths                           |     |
| SNR                     | Dimensionless                | [nlambda]                | Signal-to-noise ratio                             |     |
| photometric_aperture_radius              | λ/D                          | Scalar                   | Photometric aperture radius                       |     |
| psf_trunc_ratio         | Dimensionless                | Scalar                   | PSF truncation ratio                              |     |
| CRb_multiplier          | Dimensionless                | Scalar                   | Factor to multiply to remove background           |     |
| td_limit                | Time                         | Scalar                   | Limit placed on exposure times                    |     |
| nooptimize              | Integer                      | Scalar                   | Flag to disable exposure time optimization        |     |
| optimize_phase          | Integer                      | Scalar                   | Flag to optimize planet phase (non-functional)    |     |
| ntot                    | Integer                      | Scalar                   | Meaning not explicitly defined in code            |     |
| nmeananom               | Integer                      | Scalar                   | Number of mean anomalies                          |     |
| norbits                 | Integer                      | Scalar                   | Number of orbits                                  |     |
| Lstar                   | Solar Luminosity             | Scalar                   | Luminosity of star                                |     |
| dist                    | Length                       | Scalar                   | Distance to star                                  |     |
| vmag                    | Magnitude                    | Scalar                   | Stellar magnitude at V band                       |     |
| mag                     | Magnitude                    | [nlambda]                | Stellar magnitude at desired wavelengths          |     |
| stellar_angular_diameter_arcsec | Arcsecond                    | Scalar                   | Angular diameter of star                          |     |
| nzodis                  | Zodi                         | Scalar                   | Amount of exozodi around target star              |     |
| ra                      | Degree                       | Scalar                   | Right ascension of target star                    |     |
| dec                     | Degree                       | Scalar                   | Declination of target star                        |     |
| semimajor_axis              | Astronomical Units                    | Scalar                   | Semimajor axis of the planet's orbit (used to calculate separation; assumes face-on orbit)                           |     |

| separation              | Arcsecond                    | Scalar                   | Separation of planet                              |     |
| deltamag                | Magnitude                    | Scalar                   | Magnitude difference between planet and host star |     |
| min_deltamag            | Magnitude                    | Scalar                   | Brightest planet to resolve at the IWA            |     |
| F0V                     | Photon / (Second * cm² * nm) | Scalar                   | Flux zero point for V band                        |     |
| F0                      | Photon / (Second * cm² * nm) | [nlambda]                | Flux zero points for prescribed wavelengths       |     |
| M_V                     | Magnitude                    | Scalar                   | Absolute V band magnitude of target star          |     |
| Fzodi_list              | Dimensionless                | [nlambda]                | Zodiacal light fluxes                             |     |
| Fexozodi_list           | Dimensionless                | [nlambda]                | Exozodiacal light fluxes                          |     |
| Fbinary_list            | Dimensionless                | [nlambda]                | Binary star fluxes                                |     |
| Fp_over_Fs                     | Dimensionless                | Scalar                   | Flux of planet relative to star                   |     |
|                         |                              |                          |                                                   |     |
|                         |                              |                          |                                                   |     |
|                         |                              |                          |                                                   |     |
-->

## Within `coronagraphs.py`
| Variable Name                   | Length                   | Unit          | Meaning                                                      | User Editable |
| ------------------------------- | ------------------------ | ------------- | ------------------------------------------------------------ | ------------- |
| Istar                           | [npix, npix]             | Dimensionless | Star intensity distribution (on-axis PSF)                                 | No            |
| noisefloor                      | [npix, npix]             | Dimensionless | Noise floor of the coronagraph                               | No            |
| photometric_aperture_throughput                     | [npix, npix, npsfratios] | Dimensionless | fraction of light entering the coronagraph that ends up within the photometric core of the off-axis (planet) PSF assuming perfectly reflecting/transmitting optics, where the core is the solid angle area `Omega` and is set by either `psf_trunc_ratio` or `photometric_aperture_radius`.                                 | No            |
| omega_lod                       | [npix, npix, npsfratios] | (λ/D)²        | Solid angle of the photometric aperture                      | No            |
| skytrans                        | [npix, npix]             | Dimensionless | Sky transmission; the coronagraph’s performance when observing an infinitely extended source                                           | No            |
| pixscale                        | Scalar                   | λ/D           | Pixel scale of the coronagraph model                              | No            |
| npix                            | Scalar                   | Dimensionless       | length of one side of the coronagraph model images (assuming a square)                               | No            |
| xcenter                         | Scalar                   | Pixel         | X-coordinate of the image center                             | No            |
| ycenter                         | Scalar                   | Pixel         | Y-coordinate of the image center                             | No            |
| bandwidth                       | Scalar                   | Dimensionless | Fractional bandwidth of coronagraph                          | Yes           |
| stellar_radius                | Scalar        | R_sun           | stellar radius in solar radii                                                   | Yes            |
| stellar_angular_diameter_arcsec                | Scalar        | arcsec           | angular diameter of the star                                                   | No            |
| npsfratios                      | Scalar                   | Dimensionless       | Number of PSF truncation ratios   (default 1)                                      | No            |
| nrolls                          | Scalar                   | Dimensionless       | Number of roll angles performed                                       | Yes           |
| nchannels                       | Scalar                   | Dimensionless       | Number of channels in coronagraph                                           | Yes           |
| minimum_IWA                     | Scalar                   | λ/D           | Minimum Inner Working Angle                                  | Yes           |
| maximum_OWA                     | Scalar                   | λ/D           | Maximum Outer Working Angle                                  | Yes           |
| coronagraph_optical_throughput          | [nlambda]                | Dimensionless | Throughput for all coronagraph optics                        | Yes           |
| coronagraph_spectral_resolution | Scalar                   | Dimensionless | Spectral resolution of the coronagraph                       | Yes           |
| contrast                        | Scalar                   | Dimensionless | Noise floor contrast of coronagraph                          | Yes           |
| noisefloor_factor               | Scalar                   | Dimensionless | Systematic noise floor factor                                | Yes           |
| noisefloor_PPF               | Scalar                   | Dimensionless | Noise floor post-processing factor                                | Yes           |
| Tcore                           | Scalar                   | Dimensionless | Core throughput of coronagraph (used in ToyModel only, or if photometric_aperture_radius is specified for omega_lod calculation)       | Yes           |
| TLyot                           | Scalar                   | Dimensionless | Lyot transmission of the coronagraph (used in ToyModel only) | Yes           |
| PSFpeak                         | Scalar                   | Dimensionless | Peak value of the off-axis PSF                                        | No            |

### A note on calculating `omega_lod`:
The photometric aperture `omega_lod` can be calculated via two methods, and the user should specify 
either the `psf_trunc_ratio` or `photometric_aperture_radius` parameters to do so.
- photometric_aperture_radius` simply sets a radius for the photometric aperture, such that `omega_lod = \pi * (photometric_aperture_radius * (lambda/D))^2`, 
where `omega_lod` is the solid angle of the photometric aperture. 
- In contrast, `psf_trunc_ratio` is a more complex way of calculating the photometric aperture solid angle `omega_lod`, necessary because the off-axis PSF is not always going to be a perfect circle, and can be misshapen. In principle, this method takes an off-axis PSF and calculates `omega_lod` as all pixels in the PSF that are above the threshold `psf_trunc_ratio * max(off-axis PSF)`, accounting for imperfect PSF shapes. Note: If the off-axis PSF shape is a perfect airy disk, then `psf_trunc_ratio` is simply `1 - photometric_aperture_radius`. 
- Finally, `photometric_aperture_throughput` is an entirely different, but related, parameter, not to be confused with the two parameters above. This parameter is essentially the core throughput of the off-axis PSF. In other words, this is the fraction of light entering the coronagraph that ends up within the photometric core of the off-axis (planet) PSF assuming perfectly reflecting/transmitting optics, where the core is the solid angle area `omega_lod` and is set by either `psf_trunc_ratio` or `photometric_aperture_radius`. 


## Within `telescopes.py`
| Variable Name        | Length    | Unit          | Meaning                                       | User Editable |
| -------------------- | --------- | ------------- | --------------------------------------------- | ------------- |
| diameter             | Scalar    | m      | Circumscribed diameter of telescope aperture  | Yes           |
| Area                 | Scalar    | m^2     | Effective collecting area of telescope        | No            |
| unobscured_area      | Scalar    | Dimensionless | Unobscured area percentage                    | Yes           |
| toverhead_fixed      | Scalar    | Time          | Fixed overhead time                           | Yes           |
| toverhead_multi      | Scalar    | Dimensionless | Multiplicative overhead time                  | Yes           |
| telescope_optical_throughput | [nlambda] | Dimensionless | Optical throughput of telescope               | Yes           |
| temperature          | Scalar    | Temperature   | Temperature of the warm optics                | Yes           |
| T_contamination              | Scalar    | Dimensionless | Effective throughput factor for contamination | Yes           |

## Within `detectors.py`
| Variable Name   | Length    | Unit                        | Meaning                                           | User Editable |
| --------------- | --------- | --------------------------- | ------------------------------------------------- | ------------- |
| pixscale_mas    | Scalar    | Milliarcsecond              | Detector pixel scale                              | Yes           |
| npix_multiplier | [nlambda] | Dimensionless               | Number of detector pixels per image plane "pixel" | Yes           |
| DC              | [nlambda] | Electron / (Pixel * Second) | Dark current                                      | Yes           |
| RN              | [nlambda] | Electron / (Pixel * Read)   | Read noise                                        | Yes           |
| tread           | [nlambda] | Second                      | Read time                                         | Yes           |
| CIC             | [nlambda] | Electron / (Pixel * Photon) | Clock-induced charge                              | Yes           |
| QE              | [nlambda] | Electron / Photon           | Quantum efficiency of detector                    | Yes           |
| dQE             | [nlambda] | Dimensionless               | Effective QE due to degradation                   | Yes           |


## Within `observation.py`
| Variable Name   | Length    | Unit          | Meaning                                 | User Editable |
| --------------- | --------- | ------------- | --------------------------------------- | ------------- |
| wavelength      | [nlambda] | um      | Observation wavelengths                 | Yes           |
| SNR             | [nlambda] | Dimensionless | Signal-to-noise ratio                   | Yes           |
| photometric_aperture_radius      | Scalar    | λ/D           | Photometric aperture radius             | Yes           |
| psf_trunc_ratio | Scalar    | Dimensionless | truncate the off-axis PSF at a threshold (thresh = psf_trunc_ratio * max(off-axis PSF))             | Yes           |
| CRb_multiplier  | Scalar    | Dimensionless | Factor to multiply assuming differential imaging to remove background | Yes           |
| td_limit        | Scalar    | s        | Limit placed on exposure times          | No            |
| exptime         | [nlambda] | s        | Exposure time for each wavelength       | No            |
| fullsnr         | [nlambda] | Dimensionless | Calculated SNR for each wavelength      | No            |

## Within `astrophysical_scene.py`
| Variable Name           | Length    | Unit                         | Meaning                                           | User Editable |
| ----------------------- | --------- | ---------------------------- | ------------------------------------------------- | ------------- |
| Lstar                   | Scalar    | Solar Luminosity             | Luminosity of star                                | Yes           |
| dist                    | Scalar    | pc                       | Distance to star                                  | Yes           |
| vmag                    | Scalar    | Magnitude                    | Stellar magnitude at V band                       | Yes           |
| mag                     | [nlambda] | Magnitude                    | Stellar magnitude at desired wavelengths          | Yes           |
| stellar_angular_diameter_arcsec | Scalar    | Arcsecond                    | Angular diameter of star                          | No           |
| nzodis                  | Scalar    | Zodi                         | Amount of exozodi around target star              | Yes           |
| ra                      | Scalar    | Degree                       | Right ascension of target star                    | Yes           |
| dec                     | Scalar    | Degree                       | Declination of target star                        | Yes           |
| semimajor_axis              | Scalar | Astronomical Units                    | Scalar                   | Semimajor axis of the planet's orbit (used to calculate separation; assumes face-on orbit)                           |  Yes   |
| separation              | Scalar    | Arcsecond                    | Separation of planet                              | Yes           |
| xp                      | Scalar    | Arcsecond                    | X-coordinate of planet (defaults to zero for now)                           | No            |
| yp                      | Scalar    | Arcsecond                    | Y-coordinate of planet                            | No            |
| deltamag                | [nlambda] | Magnitude                    | Magnitude difference between planet and host star | Yes           |
| min_deltamag            | [nlambda] | Magnitude                    | Brightest planet to resolve at the IWA            | Yes           |
| F0V                     | Scalar    | Photon / (s * cm² * nm) | Flux zero point for V band                        | Yes            |
| F0                      | [nlambda] | Photon / (s * cm² * nm) | Flux zero points for prescribed wavelengths       | Yes           |
| M_V                     | Scalar    | Magnitude                    | Absolute V band magnitude of target star          | No            |
| Fzodi_list              | [nlambda] | Dimensionless                | Zodiacal light fluxes                             | No            |
| Fexozodi_list           | [nlambda] | Dimensionless                | Exozodiacal light fluxes                          | No            |
| Fbinary_list            | [nlambda] | Dimensionless                | Binary star fluxes                                | No            |
| Fp_over_Fs                     | [nlambda] | Dimensionless                | Flux of planet relative to star                   | Yes           |
| Fs_over_F0                   | [nlambda] | Dimensionless                | Stellar flux relative to F0                       | No            |

## Within `observatory.py`
| Variable Name | Length | Unit | Meaning | User Editable |
|---------------|--------|------|---------|---------------|
| optics_throughput | [nlambda] | Dimensionless | Optical throughput of the entire system | Yes* |
| epswarmTrcold | [nlambda] | Dimensionless | Warm emissivity * cold transmission factor | Yes* |
| total_throughput | [nlambda] | Dimensionless | Total throughput including optics and detector | No |
| observing_mode | Scalar | String | Observing mode (e.g., 'IMAGER' or 'IFS') | Yes |

## Within `parse_input.py`
| Variable Name      | Length | Unit    | Meaning                                  | User Editable |
| ------------------ | ------ | ------- | ---------------------------------------- | ------------- |
| secondary_flag     | Scalar | Boolean | Flag for secondary variables             | Yes           |
| observatory_preset | Scalar | String  | Preset configuration for the observatory | Yes           |
| telescope_type     | Scalar | String  | Type of telescope to use                 | Yes           |
| coronagraph_type   | Scalar | String  | Type of coronagraph to use               | Yes           |
| detector_type      | Scalar | String  | Type of detector to use                  | Yes           |
| observing_mode     | Scalar | String  | Observing mode (e.g., 'IMAGER' or 'IFS') | Yes           |

## Within `exposure_time_calculator.py`
| Variable Name    | Length | Unit          | Meaning                                             | User Editable |
| ---------------- | ------ | ------------- | --------------------------------------------------- | ------------- |
| deltalambda_nm   | Scalar | nm            | Bandwidth for each wavelength                       | No            |
| lod              | Scalar | Dimensionless | λ/D (wavelength / telescope diameter)               | No            |
| lod_rad          | Scalar | Radian        | λ/D in radians                                      | No            |
| lod_arcsec       | Scalar | Arcsecond     | λ/D in arcseconds                                   | No            |
| area_cm2         | Scalar | cm²           | Telescope collecting area                           | No            |
| detpixscale_lod  | Scalar | λ/D           | Detector pixel scale in λ/D units                   | No            |
| stellar_diam_lod | Scalar | λ/D           | Stellar diameter in λ/D units                       | No            |
| det_sep_pix      | Scalar | Pixel         | Separation at IWA in pixels                         | No            |
| det_sep          | Scalar | Arcsecond     | Separation at IWA in arcseconds                     | No            |
| det_Istar        | Scalar | Dimensionless | Max stellar intensity at IWA                        | No            |
| det_skytrans     | Scalar | Dimensionless | Max sky transmission at IWA                         | No            |
| det_photometric_aperture_throughput  | Scalar | Dimensionless | Max photometric aperture fraction at IWA            | No            |
| det_omega_lod    | Scalar | (λ/D)²        | Solid angle corresponding to max photometric_aperture_throughput at IWA | No            |
| det_npix         | Scalar | Pixel         | Number of pixels in detector                        | No            |
| CRp              | Scalar | Electron/s    | Planet count rate                                   | No            |
| CRbs             | Scalar | Electron/s    | Stellar leakage count rate                          | No            |
| CRbz             | Scalar | Electron/s    | Local zodiacal light count rate                     | No            |
| CRbez            | Scalar | Electron/s    | Exozodiacal light count rate                        | No            |
| CRbbin           | Scalar | Electron/s    | Binary star count rate                              | No            |
| CRbth            | Scalar | Electron/s    | Thermal background count rate                       | No            |
| CRbd             | Scalar | Electron/s    | Detector noise count rate                           | No            |
| CRnf             | Scalar | Electron/s    | Noise floor count rate                              | No            |
| CRb              | Scalar | Electron/s    | Total background count rate                         | No            |
| t_photon_count   | Scalar | s             | Photon counting time                                | No            |

