---
title: '`pyEDITH`: A coronagraphic exposure time calculator for the Habitable Worlds Observatory'
tags:
  - Python
  - astronomy
  - coronagraphy
  - exoplanets
  - space telescopes
authors:
  - name: Miles H. Currie
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: "1"
  - name: Eleonora Alei
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1"
  - name: Christopher C. Stark
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Aki Roberge
    affiliation: "1"
  - name: Avi Mandell
    affiliation: "1"
affiliations:
 - name: NASA Goddard Space Flight Center
   index: 1
date: 17 July 2025
bibliography: paper.bib
---

# Summary

`pyEDITH` is a Python-based coronagraphic exposure time calculator built for NASA's next flagship mission, the Habitable Worlds Observatory (HWO), tasked with searching for signs of habitability and life in dozens of nearby exoplanet systems. `pyEDITH` is designed to simulate wavelength-dependent exposure times and signal-to-noise ratios (S/N) for both photometric and spectroscopic direct imaging observations. `pyEDITH` considers realistic engineering specifications, and allows the user to provide target system information, as well as alter observatory parameters, to calculate synthetic HWO observations of Earth-like exoplanets. We present a schematic of the `pyEDITH` framework in \autoref{fig:diagram}.

![A schematic of the `pyEDITH` components and their relationshops, and the data flow from inputs to final calculations.\label{fig:diagram}](pyedith_workflow.pdf)

# Statement of need

`pyEDITH` is a Python package for developing the exoplanet detection and characterization capabilities of the Habitable Worlds Observatory mission. `pyEDITH` has heritage from the Altruistic Yield Optimizer [@stark2014ayo], used for robust and fast calculations, and implementing this framework in Python enables easire integration with modern astronomical workflows and lowers the technical barrier for users to adopt the tool for their specific needs. `pyEDITH` was designed to be used by astronomers and engineers, including students, for understanding the capabilities and limitations of different HWO architectures for exoplanet detection and characterization. It has already been used by several forthcoming scientific publications [@Currie2025exozodi; @Alei2025], and is endorsed by the HWO Project Office at NASA's Goddard Space Flight Center. `pyEDITH` is the most sophisticated noise model for HWO to date, and through its flexible and user-friendly design, it will enable efficient HWO mission design studies and inform the development of more advanced observer planning tools when HWO launches.

# Mathematical formalism

The core functionality of `pyEDITH` is to calculate exposure times considering the signal and background sources associated with coronagraphic exoplanet observations. Critically, this includes treatment for the suppression of stellar light relative to the target exoplanet, the key capability enabled by a coronagraph. Mathematically, exposure times are calculated as:

$$\tau=\left[ (\mathrm{S/N})^2 \left(\frac{\mathrm{CR}_p+\alpha\ \mathrm{CR}_b}{\mathrm{CR}_p^2 - (\mathrm{S/N})^2\ \mathrm{CR}_\mathrm{nf}^2}\right) \tau_\mathrm{multi}+\tau_\mathrm{static} \right]$$

where S/N is the desired signal-to-noise ratio, $\mathrm{CR}_p$, $\mathrm{CR}_b$, and $\mathrm{CR}_\mathrm{nf}$ are the photon count rates of the planet, background, and noise floor, respectively. The background $\mathrm{CR}_b$ includes stellar leakage, zodiacal/exozodiacal light, observatory thermal radiation, and detector noise. $\alpha$ parameterizes the PSF-subtraction method ($\alpha=2$ for angular differential imaging), and $\tau_\mathrm{multi}$ and $\tau_\mathrm{static}$ are multiplicative and fixed overhead times, respectively, accounting for telescope slew/settling time and achieving the required coronagraphic contrast ratio. Importantly, `pyEDITH` has functionality to calculate S/N given a desired exposure time by inverting the equation above. `pyEDITH` can be used to calculate photometric observations in imaging mode and spectroscopic observations in spectroscopy mode, briefly described below.

# Imaging mode

In photometry mode, `pyEDITH` can calculate the exposure time $\tau$ needed to reach the S/N required for initial exoplanet detection surveys in the V-band (0.5 μm). Given the calculated time to exoplanet discovery, `pyEDITH` simultaneously calculates the corresponding S/N for other strategic bandpasses, which can be defined by the user. These multi-bandpass calculations are designed to flexibly explore different filter choices (including bandwidth) in the UV–NIR, enabling trade studies to determine the set of filters that maximizes the science return. See Figure \autoref{fig:img} for an example demonstration of `pyEDITH`'s imaging mode. 

![Example of photometric exposure times generated for different bandpasses/bandwidths. \label{fig:img}](img_demo.pdf)


# Spectroscopy mode

The spectroscopy mode in `pyEDITH` calculates exposure time and S/N as a function of wavelength. `pyEDITH` allows the user to input models of the host star and exoplanet reflectance spectra, then calculates exposure times and S/N for the user-defined wavelength grid. Alternatively, the user can define unlimited spectral channels and their corresponding resolutions, prompting `pyEDITH` to run calculations for an internal wavelength grid, enabling maximum flexibility for spectroscopic instrumentation trade studies. Finally, `pyEDITH` can synthesize noisy exoplanet observational data to use in precursor data analysis studies. See Figure \autoref{fig:spec} for an example demonstration of `pyEDITH`'s spectroscopy mode. 

![Example of `pyEDITH`-synthesized HWO data (red) of an Earth-like exoplanet (upper) and the calculated S/N as a function of wavelength (lower).  \label{fig:spec}](spec_demo.pdf)

# Validation

We validate `pyEDITH` against other available exposure time calculators, following the prescription of @stark2025validation. Reproducing the validation scenarios in @stark2025validation by fixing our inputs, we compare the outputs of `pyEDITH` to the four other exposure time calculators and find agreement to within a few percent for most common parameters. Minor differences in reported zero-point fluxes can cause discrepancies up to 10% in some parameters; however, these differences also appear in the @stark2025validation comparison.

# Acknowledgements

We acknowledge contributions from the broader Habitable Worlds Observatory community and thank our beta testers for their valuable feedback. The work of M.H.C. and E.A. was supported by appointments to the NASA Postdoctoral Program at the NASA Goddard Space Flight Center, administered by Oak Ridge Associated Universities under contract with NASA.

# References
