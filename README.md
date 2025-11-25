#
[![Language](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3+-blue.svg?style=flat-square)](https://github.com/ticoi/ticoi/blob/main/LICENSE)


Codes to compute shadow maps according to a DEM using the horayzon libray.
This could be done for one date, see an example [here](example/example_shadow_for_one_date.py).
It's also possible to compute the number of days under shadow extent and border, see an example [for large DEM](example/example_large_DEM.py) and [small DEM](example/example_small_DEM.py).

If you use this package, please cite:


## Get started

### INSTALLATION

To clone the git repository and set up the conda environment:

```
git clone git@github.com:LauraneC/SErrVel.git
cd SErrVel
mamba env create -f environment.yml -n SErrVel
mamba activate SErrVel
pip install -e .
git clone https://github.com/ChristianSteger/HORAYZON.git
cd HORAYZON
pip install -e .
```