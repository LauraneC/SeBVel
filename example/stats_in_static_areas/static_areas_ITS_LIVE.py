import glob
from tqdm import tqdm
import geopandas as gpd
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.utils import find_granule_by_point
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

mask_file = f'/home/charrierl/Documents/Seasonality/Greenland/Russel_sector_for_seasonality.gpkg'

# cube_list = glob.glob(cube_IGE_path + "/*.nc")
list_mask = []

mask = gpd.read_file(mask_file)
list_coordinates = [[mask.bounds.minx,mask.bounds.miny],[mask.bounds.maxx,mask.bounds.miny],[mask.bounds.minx,mask.bounds.maxy],[mask.bounds.maxx,mask.bounds.maxy]]
cube_list = [find_granule_by_point(list_coordinates[j]) for j in range(len(list_coordinates))]
print(cube_list)

for cube_name in tqdm(cube_list) :
    if len(cube_name) != 0:
        cube = CubeDataClass()
        cube.load(cube_name[0], pick_sensor= ["Sentinel-2"])
        # print(np.unique(cube.ds.sensor.values))
        # print(cube.nx,cube.ny)
        mask = cube.mask_cube(mask_file, invert=True)
        list_mask.append(cube)

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