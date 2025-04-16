# pyEDITH
Exposure Direct Imaging Timer for HWO (Python Version)


## Installation
Clone the pyEDITH repository:

```
git clone https://github.com/your-repo-url/pyEDITH.git
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


## Glossary


| Variable Name           | Unit                         | Length                   | Meaning                                           |     |
| ----------------------- | ---------------------------- | ------------------------ | ------------------------------------------------- | --- |
| Istar                   | Dimensionless                | [npix, npix]             | Star intensity distribution                       |     |
| noisefloor              | Dimensionless                | [npix, npix]             | Noise floor of the coronagraph                    |     |
| photap_frac             | Dimensionless                | [npix, npix, npsfratios] | Photometric aperture fraction                     |     |
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
| coronagraph_throughput  | Dimensionless                | [nlambda]                | Throughput for all coronagraph optics             |     |
| diameter                | Length                       | Scalar                   | Circumscribed diameter of telescope aperture      |     |
| Area                    | Length²                      | Scalar                   | Effective collecting area of telescope            |     |
| unobscured_area         | Dimensionless                | Scalar                   | Unobscured area percentage                        |     |
| toverhead_fixed         | Time                         | Scalar                   | Fixed overhead time                               |     |
| toverhead_multi         | Dimensionless                | Scalar                   | Multiplicative overhead time                      |     |
| telescope_throughput    | Dimensionless                | [nlambda]                | Optical throughput of telescope                   |     |
| temperature             | Temperature                  | Scalar                   | Temperature of the warm optics                    |     |
| Tcontam                 | Dimensionless                | Scalar                   | Effective throughput factor for contamination     |     |
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
| photap_rad              | λ/D                          | Scalar                   | Photometric aperture radius                       |     |
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
| angular_diameter_arcsec | Arcsecond                    | Scalar                   | Angular diameter of star                          |     |
| nzodis                  | Zodi                         | Scalar                   | Amount of exozodi around target star              |     |
| ra                      | Degree                       | Scalar                   | Right ascension of target star                    |     |
| dec                     | Degree                       | Scalar                   | Declination of target star                        |     |
| separation              | Arcsecond                    | Scalar                   | Separation of planet                              |     |
| deltamag                | Magnitude                    | Scalar                   | Magnitude difference between planet and host star |     |
| min_deltamag            | Magnitude                    | Scalar                   | Brightest planet to resolve at the IWA            |     |
| F0V                     | Photon / (Second * cm² * nm) | Scalar                   | Flux zero point for V band                        |     |
| F0                      | Photon / (Second * cm² * nm) | [nlambda]                | Flux zero points for prescribed wavelengths       |     |
| M_V                     | Magnitude                    | Scalar                   | Absolute V band magnitude of target star          |     |
| Fzodi_list              | Dimensionless                | [nlambda]                | Zodiacal light fluxes                             |     |
| Fexozodi_list           | Dimensionless                | [nlambda]                | Exozodiacal light fluxes                          |     |
| Fbinary_list            | Dimensionless                | [nlambda]                | Binary star fluxes                                |     |
| Fp0                     | Dimensionless                | Scalar                   | Flux of planet relative to star                   |     |
|                         |                              |                          |                                                   |     |
|                         |                              |                          |                                                   |     |
|                         |                              |                          |                                                   |     |


| Variable Name                   | Length                   | Unit          | Meaning                                                      | User Editable |
| ------------------------------- | ------------------------ | ------------- | ------------------------------------------------------------ | ------------- |
| Istar                           | [npix, npix]             | Dimensionless | Star intensity distribution                                  | No            |
| noisefloor                      | [npix, npix]             | Dimensionless | Noise floor of the coronagraph                               | No            |
| photap_frac                     | [npix, npix, npsfratios] | Dimensionless | Photometric aperture fraction                                | No            |
| omega_lod                       | [npix, npix, npsfratios] | (λ/D)²        | Solid angle of the photometric aperture                      | No            |
| skytrans                        | [npix, npix]             | Dimensionless | Sky transmission                                             | No            |
| pixscale                        | Scalar                   | λ/D           | Pixel scale of the coronagraph                               | No            |
| npix                            | Scalar                   | Integer       | Number of pixels in the image                                | No            |
| xcenter                         | Scalar                   | Pixel         | X-coordinate of the image center                             | No            |
| ycenter                         | Scalar                   | Pixel         | Y-coordinate of the image center                             | No            |
| bandwidth                       | Scalar                   | Dimensionless | Fractional bandwidth of coronagraph                          | Yes           |
| angular_diameter                | [ntargs]**CHECK**        | λ/D           | NO MEANING                                                   | No            |
| npsfratios                      | Scalar                   | Integer       | Number of PSF ratios                                         | No            |
| nrolls                          | Scalar                   | Integer       | Number of roll angles                                        | Yes           |
| nchannels                       | Scalar                   | Integer       | Number of channels                                           | Yes           |
| minimum_IWA                     | Scalar                   | λ/D           | Minimum Inner Working Angle                                  | Yes           |
| maximum_OWA                     | Scalar                   | λ/D           | Maximum Outer Working Angle                                  | Yes           |
| coronagraph_throughput          | [nlambda]                | Dimensionless | Throughput for all coronagraph optics                        | Yes           |
| coronagraph_spectral_resolution | Scalar                   | Dimensionless | Spectral resolution of the coronagraph                       | Yes           |
| contrast                        | Scalar                   | Dimensionless | Noise floor contrast of coronagraph                          | Yes           |
| noisefloor_factor               | Scalar                   | Dimensionless | Systematic noise floor factor                                | Yes           |
| Tcore                           | Scalar                   | Dimensionless | Core throughput of coronagraph (used in ToyModel only)       | Yes           |
| TLyot                           | Scalar                   | Dimensionless | Lyot transmission of the coronagraph (used in ToyModel only) | Yes           |
| PSFpeak                         | Scalar                   | Dimensionless | Peak value of the PSF                                        | No            |

| Variable Name        | Length    | Unit          | Meaning                                       | User Editable |
| -------------------- | --------- | ------------- | --------------------------------------------- | ------------- |
| diameter             | Scalar    | Length        | Circumscribed diameter of telescope aperture  | Yes           |
| Area                 | Scalar    | Length²       | Effective collecting area of telescope        | No            |
| unobscured_area      | Scalar    | Dimensionless | Unobscured area percentage                    | Yes           |
| toverhead_fixed      | Scalar    | Time          | Fixed overhead time                           | Yes           |
| toverhead_multi      | Scalar    | Dimensionless | Multiplicative overhead time                  | Yes           |
| telescope_throughput | [nlambda] | Dimensionless | Optical throughput of telescope               | Yes           |
| temperature          | Scalar    | Temperature   | Temperature of the warm optics                | Yes           |
| Tcontam              | Scalar    | Dimensionless | Effective throughput factor for contamination | Yes           |


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

| Variable Name   | Length    | Unit          | Meaning                                 | User Editable |
| --------------- | --------- | ------------- | --------------------------------------- | ------------- |
| wavelength      | [nlambda] | Length        | Observation wavelengths                 | Yes           |
| SNR             | [nlambda] | Dimensionless | Signal-to-noise ratio                   | Yes           |
| photap_rad      | Scalar    | λ/D           | Photometric aperture radius             | Yes           |
| psf_trunc_ratio | Scalar    | Dimensionless | PSF truncation ratio                    | Yes           |
| CRb_multiplier  | Scalar    | Dimensionless | Factor to multiply to remove background | Yes           |
| td_limit        | Scalar    | Time          | Limit placed on exposure times          | No            |
| exptime         | [nlambda] | Time          | Exposure time for each wavelength       | No            |
| fullsnr         | [nlambda] | Dimensionless | Calculated SNR for each wavelength      | No            |

| Variable Name           | Length    | Unit                         | Meaning                                           | User Editable |
| ----------------------- | --------- | ---------------------------- | ------------------------------------------------- | ------------- |
| Lstar                   | Scalar    | Solar Luminosity             | Luminosity of star                                | Yes           |
| dist                    | Scalar    | Length                       | Distance to star                                  | Yes           |
| vmag                    | Scalar    | Magnitude                    | Stellar magnitude at V band                       | Yes           |
| mag                     | [nlambda] | Magnitude                    | Stellar magnitude at desired wavelengths          | Yes           |
| angular_diameter_arcsec | Scalar    | Arcsecond                    | Angular diameter of star                          | Yes           |
| nzodis                  | Scalar    | Zodi                         | Amount of exozodi around target star              | Yes           |
| ra                      | Scalar    | Degree                       | Right ascension of target star                    | Yes           |
| dec                     | Scalar    | Degree                       | Declination of target star                        | Yes           |
| separation              | Scalar    | Arcsecond                    | Separation of planet                              | Yes           |
| xp                      | Scalar    | Arcsecond                    | X-coordinate of planet                            | No            |
| yp                      | Scalar    | Arcsecond                    | Y-coordinate of planet                            | No            |
| deltamag                | [nlambda] | Magnitude                    | Magnitude difference between planet and host star | Yes           |
| min_deltamag            | [nlambda] | Magnitude                    | Brightest planet to resolve at the IWA            | Yes           |
| F0V                     | Scalar    | Photon / (Second * cm² * nm) | Flux zero point for V band                        | No            |
| F0                      | [nlambda] | Photon / (Second * cm² * nm) | Flux zero points for prescribed wavelengths       | Yes           |
| M_V                     | Scalar    | Magnitude                    | Absolute V band magnitude of target star          | No            |
| Fzodi_list              | [nlambda] | Dimensionless                | Zodiacal light fluxes                             | No            |
| Fexozodi_list           | [nlambda] | Dimensionless                | Exozodiacal light fluxes                          | No            |
| Fbinary_list            | [nlambda] | Dimensionless                | Binary star fluxes                                | No            |
| Fp0                     | [nlambda] | Dimensionless                | Flux of planet relative to star                   | Yes           |
| Fstar                   | [nlambda] | Dimensionless                | Stellar flux relative to F0                       | No            |

| Variable Name | Length | Unit | Meaning | User Editable |
|---------------|--------|------|---------|---------------|
| optics_throughput | [nlambda] | Dimensionless | Optical throughput of the entire system | Yes* |
| epswarmTrcold | [nlambda] | Dimensionless | Warm emissivity * cold transmission factor | Yes* |
| total_throughput | [nlambda] | Dimensionless | Total throughput including optics and detector | No |
| observing_mode | Scalar | String | Observing mode (e.g., 'IMAGER' or 'IFS') | Yes |

| Variable Name      | Length | Unit    | Meaning                                  | User Editable |
| ------------------ | ------ | ------- | ---------------------------------------- | ------------- |
| secondary_flag     | Scalar | Boolean | Flag for secondary variables             | Yes           |
| observatory_preset | Scalar | String  | Preset configuration for the observatory | Yes           |
| telescope_type     | Scalar | String  | Type of telescope to use                 | Yes           |
| coronagraph_type   | Scalar | String  | Type of coronagraph to use               | Yes           |
| detector_type      | Scalar | String  | Type of detector to use                  | Yes           |
| observing_mode     | Scalar | String  | Observing mode (e.g., 'IMAGER' or 'IFS') | Yes           |

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
| det_photap_frac  | Scalar | Dimensionless | Max photometric aperture fraction at IWA            | No            |
| det_omega_lod    | Scalar | (λ/D)²        | Solid angle corresponding to max photap_frac at IWA | No            |
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

