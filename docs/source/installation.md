# Installation

0. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

1. Clone the [Sci-Eng-Interface repository](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface/tree/main). This repository contains all the relevant and updated telescope and detector parameters.

```
git clone https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface/tree/main
```

2. Create a folder containing the "Yield Input Packages" for your preferred coronagraph. Some of these files will be available soon on the Sci-Eng-Interface GitHub, but if you need one in the meantime, [send us an email](mailto:eleonora.alei@nasa.gov).

```
mkdir /path/to/yip/folder
```

2. Clone the pyEDITH repository:

```
git clone https://github.com/eleonoraalei/pyEDITH.git
cd pyEDITH
```

3. Install the package:
```
pip install -e .
```

4. Set up environment variables: Add the following lines to your `.bashrc` or `.zshrc` file:

```
export SCI_ENG_DIR="/path/to/Sci-Eng-Interface/hwo_sci_eng"
export YIP_CORO_DIR="/path/to/yip/folder"
```
Replace the paths with the actual paths on your system.

