# Welcome to pyEDITH's documentation!



`pyEDITH` is a Python-based coronagraphic exposure time calculator built for NASA's next flagship mission, the Habitable Worlds Observatory (HWO), tasked with searching for signs of habitability and life in dozens of nearby exoplanet systems. `pyEDITH` is designed to simulate wavelength-dependent exposure times and signal-to-noise ratios (S/N) for both photometric and spectroscopic direct imaging observations. `pyEDITH` considers realistic engineering specifications, and allows the user to provide target system information, as well as alter observatory parameters, to calculate synthetic HWO observations of Earth-like exoplanets. We present a schematic of the `pyEDITH` framework below:

![A schematic of the `pyEDITH` components and their relationships, and the data flow from inputs to final calculations.\label{fig:diagram}](_static/pyedith_workflow.pdf)

# Statement of need

`pyEDITH` is a Python package for developing the exoplanet detection and characterization capabilities of the Habitable Worlds Observatory mission. `pyEDITH` has heritage from the Altruistic Yield Optimizer ([Stark et al. 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...795..122S/abstract)), used for robust and fast calculations, and implementing this framework in Python enables easier integration with modern astronomical workflows and lowers the technical barrier for users to adopt the tool for their specific needs. `pyEDITH` was designed to be used by astronomers and engineers, including students, for understanding the capabilities and limitations of different HWO architectures for exoplanet detection and characterization. It has already been used by several forthcoming scientific publications ([Currie et al. 2025](DUMMY URL), [Alei et al. 2025](DUMMY URL)), and is endorsed by the HWO Project Office at NASA's Goddard Space Flight Center. `pyEDITH` is the most sophisticated noise model for HWO to date, and through its flexible and user-friendly design, it will enable efficient HWO mission design studies and inform the development of more advanced observer planning tools when HWO launches.

# How to cite

If you want to use `pyEDITH`, please cite the relevant [JOSS paper](DUMMY LINK). Since this is a paper with two equal-contribution first authors, we recommend the usage of the following notations:


```
\bibitem{pyedith}
Alei, E. and Currie, M. H. et al. (2025),.....
```


```
@article{key,
  author = {Alei, E.{\textsuperscript{*}} and Currie. M. H.{\textsuperscript{*}} and Christopher C. Stark and Aki Roberge and Avi M. Mandell},
  title = {Title of the Article},
  journal = {Journal Name},
  year = {Year},
  volume = {Volume},
  number = {Number},
  pages = {Pages},
  doi = {DOI},
  note = {{\textsuperscript{*}}These authors contributed equally to this work.}
}
```


```{toctree}
:maxdepth: 1
:caption: Contents:

installation
run_pyedith
imaging_tutorial
spectroscopy_tutorial
api
glossary
```
