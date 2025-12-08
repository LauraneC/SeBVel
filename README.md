#
[![Language](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3+-blue.svg?style=flat-square)](https://github.com/ticoi/ticoi/blob/main/LICENSE)

This code enable you to assess seasonal bias in velocity or displacement maps derived from optical images.

First, you can simulate shadow maps according to a given DEM:
* You can compute shadow mask for a given date and hour, see an example [here](example/shadow_maps/example_shadow_for_one_date.py).
* You can compute the number of days under shadow extents and borders, see an example [for large DEM](example/shadow_maps/example_large_DEM.py) and [small DEM](example/shadow_maps/example_small_DEM.py).

Second, we can comput velocity over static areas:
* , see an example [here](example/static_areas/stats_in_stable_areas.py)


If you use this package, please cite:

## Get started

### INSTALLATION

Clone the git repository and set up the conda environment:

```
git clone git@github.com:LauraneC/SeBVel.git
cd SeBVel
mamba env create -f environment.yml -n SeBVel
mamba activate SeBVel
pip install -e .
git clone https://github.com/ChristianSteger/HORAYZON.git
cd HORAYZON
pip install -e .
```

### REFERENCES

This code use other libraries:
* the horayzon libray:
Steger, C. R., Steger, B. and Schär, C. (2022): HORAYZON v1.2: an efficient and flexible ray-tracing algorithm to compute horizon and sky view factor, Geosci. Model Dev., 15, 6817–6840, https://doi.org/10.5194/gmd-15-6817-2022
https://doi.org/10.5281/zenodo.7013764
* the TICOI librairy:
Charrier, L., Dehecq, A., Guo, L., Brun, F., Millan, R., Lioret, N., Copland, L., Maier, N., Dow, C., and Halas, P.: TICOI: an operational Python package to generate regular glacier velocity time series, The Cryosphere, 19, 4555–4583, https://doi.org/10.5194/tc-19-4555-2025, 2025.
