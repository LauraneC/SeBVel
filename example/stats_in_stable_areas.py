from ticoi.cube_data_classxr import CubeDataClass
from ticoi.example import get_path
import glob


cube_folder = "/home/charriel/Documents/PostdocIGE/Seasonality/Cubes_Alps/cubes_S2_non_filter/"
# cube_folder = '/workdir2/c2h/charriel/Data/Cubes_Alps/cubes_filtered/'
cube_name_list = glob.glob(f'{cube_folder}/*.nc')
# path_save = f'/workdir2/c2h/charriel/Results/Seasonality/IGE_filtered/'  # Path where to store the results
mask_file = f'/home/charriel/Documents/PostdocIGE/Form@ter/Metadata/RGI/Alpes_RGI7.shp'

cube = CubeDataClass()
cube.load(cube_name_list[0], pick_date=["2017-01-01", "2017-03-30"])

import geopandas as gpd
gdf = gpd.read_file(mask_file)
gdf = gdf.to_crs(cube.ds.rio.crs)
cube.ds = cube.ds.rio.write_crs()
masked = cube.ds.rio.clip(gdf.geometry, gdf.crs, drop=False, all_touched=True, invert=True)
vv_mean = masked["vx"].mean(axis=2)

cube.mask_cube(mask=mask_file)
# vv_mean = np.nanmean(cube.ds["vx"],axis=2)
vv_mean = cube.ds["vx"].mean(axis=2)

print(cube)


import matplotlib.pyplot as plt
import numpy as np
plt.imshow(vv_mean)
plt.show()


lon = cube.ds['x'].values
lat = cube.ds['y'].values

# Determine correct extent and origin
extent = [lon.min(), lon.max(), lat.min(), lat.max()]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(9, 9))

# Overlay shapefile
gdf = gdf.to_crs(epsg=32632)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

# Plot raster
img1 = ax.imshow(vv_mean, cmap='seismic', vmin=-50, vmax=50, extent=extent)


# Add colorbar
plt.subplots_adjust(bottom=0.1)
cbar = fig.colorbar(img1, ax=ax, orientation='horizontal', fraction=0.03, pad=0.07, extend='both')
cbar.set_label('Difference between TICOI and TICOI_detect_temp [m/y]', fontsize=12)

plt.show()

# for cube_name_el in cube_name_list:
#     print(cube_name_el)
#
#     stable_area = get_path("Argentiere_static")
#
#     cube = CubeDataClass()
#     cube.load(cube_name_el, pick_date=["2017-01-01", "2017-03-30"],mask=mask_file)
#     # Compute normalized median absolute deviation
#     nmad = cube.compute_nmad(shapefile_path=stable_area, return_as="dataframe",invert=True)
#     # Compte median over stable areas
#     med = cube.compute_med_static_areas(shapefile_path=stable_area, return_as="dataframe",invert=True)
