# Run `pyEDITH`

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

This mode offers much more flexibility to run the ETC. We refer to our tutorials (up next in this documentation, or in the `tutorials/` folder).
