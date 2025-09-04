---
title: '`pyEDITH`: A coronagraphic exposure time calculator for the Habitable Worlds Observatory'
tags:
  - Python
  - astronomy
  - coronagraphy
  - exoplanets
  - space telescopes
authors:
  - name: Eleonora Alei
    orcid: 0000-0002-0006-1175
    equal-contrib: true
    affiliation: "1"
  - name: Miles H. Currie
    orcid: 0000-0003-3429-4142
    equal-contrib: true
    corresponding: true
    affiliation: "1"
  - name: Christopher C. Stark
    affiliation: "1"
  - name: Aki Roberge
    orcid: 0000-0002-2989-3725
    affiliation: "1"
  - name: Avi M. Mandell
    orcid: 0000-0002-8119-3355
    affiliation: "1"
affiliations:
 - name: NASA Goddard Space Flight Center
   index: 1
date: 17 July 2025
bibliography: paper.bib
---

# Summary

`pyEDITH` is a Python-based coronagraphic exposure time calculator built for the recommended NASA flagship mission, the Habitable Worlds Observatory (HWO), tasked with searching for signs of habitability and life in dozens of nearby exoplanet systems. `pyEDITH` is designed to simulate wavelength-dependent exposure times and signal-to-noise ratios (S/N) for both photometric and spectroscopic direct imaging observations. `pyEDITH` considers realistic engineering specifications, and allows the user to provide target system information, to calculate synthetic HWO observations of Earth-like exoplanets. We present a schematic of the `pyEDITH` framework in \autoref{fig:diagram}.

![A schematic of the `pyEDITH` components and their relationships, and the data flow from inputs to final calculations.\label{fig:diagram}](pyedith_workflow.pdf)

# Statement of need

`pyEDITH` is a Python package for developing the exoplanet detection and characterization capabilities of the Habitable Worlds Observatory mission. `pyEDITH` implements the Altruistic Yield Optimizer [@stark2014ayo] framework in Python, enabling easier integration with modern astronomical workflows and lowering the technical barrier for users to adopt the tool for their specific needs. `pyEDITH` was designed to be used by the scientific community at all skill levels for understanding the capabilities and limitations of different HWO architectures for exoplanet detection and characterization. The `pyEDITH` package includes API documentation[^1], tutorial notebooks, and has been used by forthcoming scientific publications [@Currie2025exozodi; @Alei2025]. 

[^1]: https://pyedith.readthedocs.io/en/latest/

# Mathematical formalism

The core functionality of `pyEDITH` is to calculate exposure times considering the signal and background sources associated with coronagraphic exoplanet observations. Critically, this includes treatment for the suppression of stellar light relative to the target exoplanet, the key capability enabled by a coronagraph. Mathematically, exposure times are calculated as:

$$\tau=(\mathrm{S/N})^2 \left(\frac{\mathrm{CR}_\mathrm{p}+\alpha\ \mathrm{CR}_\mathrm{b}}{\mathrm{CR}_\mathrm{p}^2 - (\mathrm{S/N})^2\ \mathrm{CR}_\mathrm{nf}^2}\right) \tau_\mathrm{multi}+\tau_\mathrm{static}$$

where S/N is the desired signal-to-noise ratio, $\mathrm{CR}_\mathrm{p}$, $\mathrm{CR}_\mathrm{b}$, and $\mathrm{CR}_\mathrm{nf}$ are the photon count rates of the planet, background, and noise floor, respectively. The background term $\mathrm{CR}_\mathrm{b}$ includes all background noise terms, and is described in detail below. $\alpha$ parameterizes the PSF-subtraction method ($\alpha=2$ for angular differential imaging), and $\tau_\mathrm{multi}$ and $\tau_\mathrm{static}$ are multiplicative and fixed overhead times, respectively, accounting for telescope slew/settling time and achieving the required coronagraphic contrast ratio. Importantly, `pyEDITH` has functionality to calculate S/N given a desired exposure time by inverting the equation above:

$$\mathrm{S/N} = \mathrm{CR}_\mathrm{p} \left[\mathrm{CR}_\mathrm{nf}^2 + (\mathrm{CR}_\mathrm{p} + \alpha\mathrm{CR}_\mathrm{b})\left(\frac{\tau_\mathrm{multi}}{\tau - \tau_\mathrm{static}}\right)\right]^{-0.5}$$

## Planetary Signal
The count rate of the planetary target is given by:
    $$\mathrm{CR}_\mathrm{p} = F_\mathrm{p}\ A\ \Upsilon\ T\ \Delta \lambda$$
where $F_\mathrm{p}$ is the planet flux [$\frac{\mathrm{photon}}{\mathrm{cm} \cdot \mathrm{s} \cdot \mathrm{nm}}$] at the telescope before it proceeds through the instrumentation, $A$ is the collecting area [cm$^2$], $\Upsilon$ is the fraction of light entering the coronagraph that is within the photometric core of the off-axis (planetary) PSF assuming perfectly transmitting/reflecting optics, $T$ is the total throughput of the optics, and $\Delta\lambda$ is the wavelength bin width [nm].

## Background (noise) count rates
The background count rate is composed of stellar leakage ($\mathrm{CR}_{\mathrm{b},*}$), zodiacal/exozodiacal light ($\mathrm{CR}_{\mathrm{b},\mathrm{zodi}}$ and $\mathrm{CR}_{\mathrm{b},\mathrm{exozodi}}$, respectively), observatory thermal radiation ($\mathrm{CR}_{\mathrm{b},\mathrm{thermal}}$), and detector noise ($\mathrm{CR}_{\mathrm{b},\mathrm{detector}}$); the equations for all count rates are given below.

The background count rate is given by:
$$\mathrm{CR}_\mathrm{b}= \mathrm{CR}_{\mathrm{b},*}+\mathrm{CR}_{\mathrm{b},\mathrm{zodi}}+\mathrm{CR}_{\mathrm{b},\mathrm{exozodi}}+\mathrm{CR}_{\mathrm{b},\mathrm{thermal}}+\mathrm{CR}_{\mathrm{b},\mathrm{detector}}$$

### Stellar leakage
Coronagraphs cannot block all host star light, and so the stellar leakage term is given by:
$$\mathrm{CR}_{\mathrm{b},*} = F_{*} \ \zeta\ \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$
where $F_*$ is the stellar flux [$\frac{\mathrm{photon}}{\mathrm{cm} \cdot \mathrm{s} \cdot \mathrm{nm}}$], $\zeta$ is the contrast suppression factor of the coronagraph, and $\Omega$ is the photometric aperture.

### Zodiacal dust
The count rate of the solar system zodiacal dust, assumed to be a gray scatterer, is given by:
$$\mathrm{CR}_{\mathrm{b},\mathrm{zodi}}= F_0\ 10^{-0.4z} \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$
where $F_0$ is the zero-point flux [$\frac{\mathrm{photon}}{\mathrm{cm} \cdot \mathrm{s} \cdot \mathrm{nm}}$] and $z$ is the surface brightness of the zodi [$\mathrm{mag}/\mathrm{arcsec}^2$], scaled by the zodi optical depth integrated along the target line of sight [@leinert1998]. 

### Exozodiacal dust
The count rate of habitable zone dust in exoplanet systems analogous to the zodiacal light, and is given by:
$$\mathrm{CR}_{\mathrm{b},\mathrm{exozodi}}= F_0\ n 10^{-0.4x} \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$ 
where $x$ is the surface brightness in the V band of the exozodi, assumed to be $22\ \mathrm{mag}/\mathrm{arcsec}^2$ and a gray scatterer. $n$ is the exozodi multiplier, which controls the density of exozodiacal dust in a system as a multiple of zodiacal dust.

### Thermal background
The thermal emission of the observatory is given by:
$$\mathrm{CR}_{\mathrm{b},\mathrm{thermal}}= \frac{B_\lambda}{E_\mathrm{photon}} \ \varepsilon_\mathrm{warm}\ T_\mathrm{cold}\ \mathrm{QE}\ \Omega\ A\ \Delta \lambda$$
where $B_\lambda$ is the blackbody function per unit wavelength; $E_\mathrm{photon}$ is the energy of the photon; $\varepsilon_\mathrm{warm}$ is the effective emissivity of all warm optics; $T_\mathrm{cold}$ is the transmission/reflectivity of all cold optics; $\mathrm{QE}$ is the detector's quantum efficiency. 

### Detector noise
Noise from the detector is given by:
$$\mathrm{CR}_{\mathrm{b},\mathrm{detector}}= N_\mathrm{pix} \left(\mathrm{DC}+\frac{\mathrm{RN}}{t_\mathrm{read}}+\frac{\mathrm{CIC}}{t_\mathrm{count}}\right)$$
where $N_\mathrm{pix}$ is the number of detector pixels; $\mathrm{DC}$ is the dark current [$e^-$/pix/s], $\mathrm{RN}$ is the read noise [$e^-$/pix/read], $t_\mathrm{read}$ is the read time [s], $\mathrm{CIC}$ is the clock-induced-charge [$e^-/\mathrm{pix}/\mathrm{photon}$]; and $t_\mathrm{count}$ the photon counting time [s]. 

### Noise floor
The noise floor count rate simulates imperfect coronagraphic speckle subtraction and is given by:
$$\mathrm{CR}_\mathrm{nf}=\mathrm{NF}\ F_*\ \frac{\Omega}{\mathrm{pixscale}^2}\ A\ T\ \Delta\lambda$$
where $\mathrm{NF} = {\mathrm{PSF}_*}/\mathrm{PPF}$. $\mathrm{PSF}_*$ is the on-axis coronagraphic response function, $\mathrm{PPF}$ is an assumed post-processing factor (nominally 30 for HWO, assuming $10^{-10}$ contrast is achieved), and *pixscale* is the pixel scale of the detector.

# Imaging mode
In photometry mode, pyEDITH can calculate the exposure time 𝜏 needed to reach the S/N required for initial exoplanet detection surveys in any bandpass for user-defined exoplanet systems (\autoref{fig:img}). This enables trade studies to determine the set of filters that will maximize the photometric science return of HWO. As a consequence, pyEDITH also allows the study of multi-bandpass photometric strategies, as described in Alei et al. 2025.

![Example of photometric exposure times required to achieve SNR=7 in  different bandpasses and bandwidths, assuming an underlying Earth-like spectrum. \label{fig:img}](img_demo.pdf)


# Spectroscopy mode

The spectroscopy mode of `pyEDITH` calculates exposure time and S/N as a function of wavelength given user-defined models of the host star and exoplanet reflectance spectra. The user can define spectral channels and their corresponding resolutions, enabling maximum flexibility for spectroscopic instrumentation trade studies. Finally, `pyEDITH` can synthesize noisy exoplanet observational data to use in precursor data analysis studies (\autoref{fig:spec}). 

![Example of `pyEDITH`-synthesized HWO data (red) of an Earth-like exoplanet (upper) and the calculated S/N as a function of wavelength (lower).  \label{fig:spec}](spec_demo.pdf)

# Acknowledgements

We acknowledge helpful input from Adric Riedel, Andrew Myers, and Jason Tumlinson at the Space Telescope Science Institute and the broader Habitable Worlds Observatory community. We also thank our beta testers for their valuable feedback. The work of M.H.C. and E.A. was supported by appointments to the NASA Postdoctoral Program at the NASA Goddard Space Flight Center, administered by Oak Ridge Associated Universities under contract with NASA.

# References
