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
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1"
  - name: Miles H. Currie
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: "1"
  - name: Christopher C. Stark
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Aki Roberge
    affiliation: "1"
  - name: Avi M. Mandell
    affiliation: "1"
affiliations:
 - name: NASA Goddard Space Flight Center
   index: 1
date: 17 July 2025
bibliography: paper.bib
---

# Summary

`pyEDITH` is a Python-based coronagraphic exposure time calculator built for NASA's next flagship mission, the Habitable Worlds Observatory (HWO), tasked with searching for signs of habitability and life in dozens of nearby exoplanet systems. `pyEDITH` is designed to simulate wavelength-dependent exposure times and signal-to-noise ratios (S/N) for both photometric and spectroscopic direct imaging observations. `pyEDITH` considers realistic engineering specifications, and allows the user to provide target system information, as well as alter observatory parameters, to calculate synthetic HWO observations of Earth-like exoplanets. We present a schematic of the `pyEDITH` framework in \autoref{fig:diagram}.

![A schematic of the `pyEDITH` components and their relationships, and the data flow from inputs to final calculations.\label{fig:diagram}](pyedith_workflow.pdf)

# Statement of need

`pyEDITH` is a Python package for developing the exoplanet detection and characterization capabilities of the Habitable Worlds Observatory mission. `pyEDITH` has heritage from the Altruistic Yield Optimizer [@stark2014ayo], used for robust and fast calculations. Implementing this framework in Python enables easier integration with modern astronomical workflows and lowers the technical barrier for users to adopt the tool for their specific needs. `pyEDITH` was designed to be used by the scientific community at all skill levels for understanding the capabilities and limitations of different HWO architectures for exoplanet detection and characterization. `pyEDITH` includes API documentation and tutorial notebooks, and has already been used by several forthcoming scientific publications [@Currie2025exozodi; @Alei2025]. It is endorsed by the HWO Project Office at NASA's Goddard Space Flight Center, and through its flexible and user-friendly design, it will enable efficient HWO mission design studies and inform the development of more advanced observer planning tools when HWO launches.

# Mathematical formalism

The core functionality of `pyEDITH` is to calculate exposure times considering the signal and background sources associated with coronagraphic exoplanet observations. Critically, this includes treatment for the suppression of stellar light relative to the target exoplanet, the key capability enabled by a coronagraph. Mathematically, exposure times are calculated as:

$$\tau=(\mathrm{S/N})^2 \left(\frac{\mathrm{CR}_p+\alpha\ \mathrm{CR}_b}{\mathrm{CR}_p^2 - (\mathrm{S/N})^2\ \mathrm{CR}_\mathrm{nf}^2}\right) \tau_\mathrm{multi}+\tau_\mathrm{static}$$

where S/N is the desired signal-to-noise ratio, $\mathrm{CR}_p$, $\mathrm{CR}_b$, and $\mathrm{CR}_\mathrm{nf}$ are the photon count rates of the planet, background, and noise floor, respectively. The background term $\mathrm{CR}_b$ includes all background noise terms, and is described in detail below. $\alpha$ parameterizes the PSF-subtraction method ($\alpha=2$ for angular differential imaging), and $\tau_\mathrm{multi}$ and $\tau_\mathrm{static}$ are multiplicative and fixed overhead times, respectively, accounting for telescope slew/settling time and achieving the required coronagraphic contrast ratio. Importantly, `pyEDITH` has functionality to calculate S/N given a desired exposure time by inverting the equation above:

$$\mathrm{S/N} = \mathrm{CR}_p \left[\mathrm{CR}_\mathrm{nf}^2 + (\mathrm{CR}_\mathrm{p} + \alpha\mathrm{CR}_\mathrm{b})\left(\frac{\tau_\mathrm{multi}}{\tau - \tau_\mathrm{static}}\right)\right]^{-0.5}$$

## Planetary Signal
The count rate of the planetary target is given by:
    $$\mathrm{CR}_p = F_p\ A\ \Upsilon\ T\ \Delta \lambda$$
where $F_p$ is the planet flux [$\frac{\mathrm{photon}}{\mathrm{cm} \mathrm{s} \mathrm{nm}}$] at the telescope, and before it proceeds through the observatory, $A$ is the collecting area [cm$^2$], $\Upsilon$ is the fraction of light entering the coronagraph that is within the photometric core of the off-axis (planetary) PSF assuming perfectly transmitting/reflecting optics, $T$ is the optics throughput, $\Delta\lambda$ is the wavelength bin width [nm].

## Background (noise) count rates
The background count rate is composed of stellar leakage ($\mathrm{CR}_{b,*}$), zodiacal/exozodiacal light ($\mathrm{CR}_{b,\mathrm{zodi}}$ and $\mathrm{CR}_{b,\mathrm{exozodi}}$, respectively), observatory thermal radiation ($\mathrm{CR}_{b,\mathrm{thermal}}$), and detector noise ($\mathrm{CR}_{b,\mathrm{detector}}$); the equations for all count rates are given below.

The background count rate is given by:
$$\mathrm{CR}_b= \mathrm{CR}_{b,*}+\mathrm{CR}_{b,\mathrm{zodi}}+\mathrm{CR}_{b,\mathrm{exozodi}}+\mathrm{CR}_{b,\mathrm{thermal}}+\mathrm{CR}_{b,\mathrm{detector}}$$

### Stellar leakage
Coronagraphs cannot block all host star light, and so the stellar leakage term is given by:
$$\mathrm{CR}_{b,*} = F_{*} \ \zeta\ \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$
where $F_*$ is the stellar flux [$\frac{\mathrm{photon}}{\mathrm{cm} \mathrm{s} \mathrm{nm}}$], $\zeta$ is the contrast suppression factor of the coronagraph, and $\Omega$ is the photometric aperture.

### Zodiacal dust
The count rate of the solar system zodiacal dust is given by:
$$\mathrm{CR}_{b,\mathrm{zodi}}= F_0\ 10^{-0.4z} \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$
where $F_0$ is the zero-point flux [$\frac{\mathrm{photon}}{\mathrm{cm} \mathrm{s} \mathrm{nm}}$] and $z$ is the surface brightness of the zodi, nominally $23\ mag/arcsec^2$. The zodiacal dust is assumed to be a gray scatterer of stellar light. 

### Exozodiacal dust
The count rate of habitable zone dust in exoplanet systems is given by:
$$\mathrm{CR}_{b,\mathrm{exozodi}}= F_0\ n 10^{-0.4x} \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$ 
where $x$ is the surface brightness in the V band of the exozodi, assumed to be $22\ mag/arcsec^2$ and a gray scatterer. $n$ is the exozodi multiplier, which controls the density of exozodiacal dust in a system as a multiple of zodiacal dust.

<!-- ### Stellar binary companions
The countrate accounting for flux leakage from stellar binary companions is given by:
$$\mathrm{CR}_{b,\mathrm{binary}}= F_0\ 10^{-0.4m}\ \Omega\ A\ \Upsilon\ T\ \Delta \lambda$$ 
where $m$ is the magnitude of the binary star seen from the planet -->

### Thermal background
The thermal emission of the observatory is given by:
$$\mathrm{CR}_{b,\mathrm{thermal}}= \frac{B_\lambda}{E_\mathrm{photon}} \ \varepsilon_\mathrm{warm}\ T_\mathrm{cold}\ \mathrm{QE}\ \Omega\ A\ \Delta \lambda$$
where $B_\lambda$ is the blackbody function per unit wavelength; $E_\mathrm{photon}$ is the energy of the photon; $\varepsilon_\mathrm{warm}$ is the effective emissivity of all warm optics; $T_\mathrm{cold}$ is the transmission/reflectivity of all cold optics; $QE$ is the detector's quantum efficiency. 

### Detector noise
Noise from the detector is given by:
$$\mathrm{CR}_{b,\mathrm{detector}}= N_\mathrm{pix} \left(\mathrm{DC}+\frac{\mathrm{RN}}{t_\mathrm{read}}+\frac{\mathrm{CIC}}{t_\mathrm{count}}\right)$$
where $N_\mathrm{pix}$ is the number of detector pixels; $\mathrm{DC}$ is the dark current [$e^-$/pix/s], $\mathrm{RN}$ is the read noise [$e^-$/pix/read], $t_\mathrm{read}$ is the read time [s], $\mathrm{CIC}$ is the clock-induced-charge [$e^-/\mathrm{pix}/\mathrm{photon}$]; and $t_\mathrm{count}$ the photon counting time [s]. 

### Noise floor
The noise floor count rate simulating imperfect coronagraphic speckle subtraction is given by:
$$\mathrm{CR}_\mathrm{nf}=\mathrm{NF}\ F_*\ \frac{\Omega}{\mathrm{pixscale}^2}\ A\ T\ \Delta\lambda$$
where $\mathrm{NF} = {\mathrm{PSF}_*}/\mathrm{PPF}$. $\mathrm{PSF}_*$ is the on-axis coronagraphic response function, $\mathrm{PPF}$ is an assumed post-processing factor (nominally 30 for HWO, assuming $10^{-10} contrast is achieved), and pixscale is the pixel scale of the detector.

# Imaging mode

In photometry mode, `pyEDITH` can calculate the exposure time $\tau$ needed to reach the S/N required for initial exoplanet detection surveys in the V-band (0.5 μm). Given the calculated time to exoplanet discovery, `pyEDITH` simultaneously calculates the corresponding S/N for other strategic bandpasses, which can be defined by the user. These multi-bandpass calculations are designed to flexibly explore different filter choices (including bandwidth) in the UV–NIR, enabling trade studies to determine the set of filters that maximizes the science return. See \autoref{fig:img} for an example demonstration of `pyEDITH`'s imaging mode. 

![Example of photometric exposure times generated for different bandpasses/bandwidths. \label{fig:img}](img_demo.pdf)


# Spectroscopy mode

The spectroscopy mode in `pyEDITH` calculates exposure time and S/N as a function of wavelength. `pyEDITH` allows the user to input models of the host star and exoplanet reflectance spectra, then calculates exposure times and S/N for the user-defined wavelength grid. Alternatively, the user can define unlimited spectral channels and their corresponding resolutions, prompting `pyEDITH` to run calculations for an internal wavelength grid, enabling maximum flexibility for spectroscopic instrumentation trade studies. Finally, `pyEDITH` can synthesize noisy exoplanet observational data to use in precursor data analysis studies. See \autoref{fig:spec} for an example demonstration of `pyEDITH`'s spectroscopy mode. 

![Example of `pyEDITH`-synthesized HWO data (red) of an Earth-like exoplanet (upper) and the calculated S/N as a function of wavelength (lower).  \label{fig:spec}](spec_demo.pdf)

# Documentation and Validation

Documentation for `pyEDITH` is available in the GitHub repository, and includes an API and tutorial notebooks.  We validate `pyEDITH` against other available HWO exposure time calculators, and the results of this analysis are also found in the accompanying documentation.  

<!-- following the prescription of @stark2025validation. Reproducing the validation scenarios in @stark2025validation by fixing our inputs, we compare the outputs of `pyEDITH` to the four other exposure time calculators and find agreement to within a few percent for most common parameters. Minor differences in reported zero-point fluxes can cause discrepancies up to 10% in some parameters; however, these differences also appear in the @stark2025validation comparison. -->

# Acknowledgements

We acknowledge helpful input from Adric Riedel, Andrew Myers, and Jason Tumlinson at the Space Telescope Science Institute and the broader Habitable Worlds Observatory community. We also thank our beta testers for their valuable feedback. The work of M.H.C. and E.A. was supported by appointments to the NASA Postdoctoral Program at the NASA Goddard Space Flight Center, administered by Oak Ridge Associated Universities under contract with NASA.

# References
