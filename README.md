#
[![Language](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3+-blue.svg?style=flat-square)](https://github.com/ticoi/ticoi/blob/main/LICENSE)

This code enable you to assess seasonal bias in velocity or displacement maps derived from optical images.

First, you can simulate shadow maps according to a given DEM:
* you can compute shadow mask for a given date and hour, see an example [here](example/1-mapping_shadows/shadow_map_for_one_date.py).
* you can compute the number of days under shadow extents and borders, see an example [for large DEM](example/1-mapping_shadows/shadow_map_large_DEM.py) and [small DEM](example/1-mapping_shadows/shadow_map_large_DEM.py).

Second, you can assess the seasonal amplitude of velocity over static areas, which is a proxy of the seasonal bias:
* see an example for the ITS_LIVE dataset [here](example/2-compute_seasonal_bias/Seasonality_Alps_ITS_LIVE.py)

Third, you can estimate the expected seasonal bias over glaciers, by interpolating the seasonal bias over static areas according to the slope and aspect:
* see an example [here](example/3-estimation_of_seasonal_bias_over_glaciers/interpolation_to_glacier_areas.ipynb)


If you use this package, please cite:

Charrier et al (submitted early December 2025)

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
