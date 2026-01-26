import glob
from tqdm import tqdm
import geopandas as gpd
from ticoi.cube_data_classxr import CubeDataClass
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2-updated-october2024/N40E000/ITS_LIVE_vel_EPSG32632_G0120_X350000_Y5050000.zarr'
cube_IGE_path = "/home/charriel/Documents/PostdocIGE/Seasonality/Cubes_Alps/cubes_S2_non_filter_2023"
mask_file = f'/home/charriel/Documents/PostdocIGE/Form@ter/Metadata/RGI/Alpes_RGI7.shp'

cube_list = glob.glob(cube_IGE_path + "/*.nc")
list_mask = []

for cube_name_IGE in tqdm(cube_list):
    print(cube_name_IGE)
    cube_grid = gpd.read_file('/home/charriel/Documents/PostdocIGE/Seasonality/Cubes_Alps/cube_grid.shp')
    cube_samp = cube_grid[cube_grid['NAME'] == f'{cube_name_IGE.split("/")[-1].split("_IGE")[0]}_post0050_yr*']
    subset = cube_samp.bounds['minx'], cube_samp.bounds['maxx'], cube_samp.bounds['maxy'], cube_samp.bounds[
        'miny']  # subset = [330768.1836587775, 334412.3736587863, 5083194.342253393, 5079918.252253384]
    print(subset)
    cube = CubeDataClass()
    cube.load(cube_name_its, subset=subset,proj="EPSG:32632", pick_sensor= ["Sentinel-2"])
    print(np.unique(cube.ds.sensor.values))
    print(cube.nx,cube.ny)
    mask = cube.mask_cube(mask_file, invert=True)
    list_mask.append(mask)

t = xr.concat(list_mask,dim="mid_date")
print('Concat done')
t_mean_time = t.mean(dim=["x", "y"], skipna=True)
print('Mean done')

fig, ax = plt.subplots(2,1)
ax[0].plot(t_mean_time.mid_date,t_mean_time['vx'],linestyle='',marker='o',color='b',markersize=2)
# ax[0].set_ylim(-100,100)
ax[0].set_ylabel("vx [m/y]")
ax[1].plot(t_mean_time.mid_date,t_mean_time['vy'],linestyle='',marker='o',color='b',markersize=2)
# ax[1].set_ylim(-150,150)
ax[1].set_ylabel("vy [m/y]")
plt.show()
fig.savefig('moving_areas_in_time_its.png',dpi=300)


import pandas as pd
# Convert mid_date to datetime if not already
# Ensure mid_date is datetime and sorted
t_mean_time = t_mean_time.assign_coords(mid_date=pd.to_datetime(t_mean_time.mid_date.values))
t_mean_time = t_mean_time.sortby("mid_date")

# --- Rolling median (index-based window) ---
window_size = 3  # number of time steps
t_mean_time_rolled = t_mean_time.rolling(mid_date=window_size, center=True).construct("window_dim").median(dim="window_dim")

# --- Plot ---
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax[0].plot(t_mean_time.mid_date, t_mean_time['vx'], 'o', color='lightgray', markersize=2, label='Image-pair velocities')
ax[0].plot(t_mean_time_rolled.mid_date, t_mean_time_rolled['vx'], color='b', label=f'{window_size}-month rolling median')
ax[0].set_ylabel("vx [m/y]")
ax[0].legend()

ax[1].plot(t_mean_time.mid_date, t_mean_time['vy'], 'o', color='lightgray', markersize=2, label='Image-pair velocities')
ax[1].plot(t_mean_time_rolled.mid_date, t_mean_time_rolled['vy'], color='b', label=f'{window_size}-month rolling median')
ax[1].set_ylabel("vy [m/y]")
ax[1].legend()

plt.xlabel("Date")
plt.tight_layout()
plt.show()
fig.savefig('static_areas_in_time_its_rolled.png', dpi=300)