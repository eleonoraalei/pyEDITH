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

