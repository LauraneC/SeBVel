from matplotlib import pyplot as plt
from ticoi.cube_data_classxr import CubeDataClass
import glob
import xarray as xr
import numpy as np

cube_folder = "/home/charriel/Documents/PostdocIGE/Seasonality/Cubes_Alps/cubes_S2_non_filter_2023/"
cube_name_list = glob.glob(f'{cube_folder}/*.nc')
mask_file = f'/home/charriel/Documents/PostdocIGE/Form@ter/Metadata/RGI/Alpes_RGI7.shp'

list_mask = []
cube = CubeDataClass()
cube.load(cube_name_list[0], chunks= {})
list_mask.append(cube.mask_cube(mask_file, invert=True))
for cube_name_el in cube_name_list[1:]:
    print(cube_name_el)
    cube = CubeDataClass()
    cube.load(cube_name_el, chunks= {})
    mask = cube.mask_cube(mask_file, invert=True)
    list_mask.append(mask)

t = xr.concat(list_mask,dim="mid_date")
print('Concat done')
t_mean_time = t.mean(dim=["x", "y"], skipna=True)
print('Mean done')

fig, ax = plt.subplots(2,1)
ax[0].plot(t_mean_time.mid_date,t_mean_time['vx'],linestyle='',marker='o',color='b',markersize=2)
ax[0].set_ylim(-100,100)
ax[0].set_ylabel("vx [m/y]")
ax[1].plot(t_mean_time.mid_date,t_mean_time['vy'],linestyle='',marker='o',color='b',markersize=2)
ax[1].set_ylim(-150,150)
ax[1].set_ylabel("vy [m/y]")
plt.show()
fig.savefig('static_areas_in_time.png',dpi=300)

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
ax[0].set_ylim(-100,100)

ax[1].plot(t_mean_time.mid_date, t_mean_time['vy'], 'o', color='lightgray', markersize=2, label='Image-pair velocities')
ax[1].plot(t_mean_time_rolled.mid_date, t_mean_time_rolled['vy'], color='b', label=f'{window_size}-month rolling median')
ax[1].set_ylabel("vy [m/y]")
ax[1].legend()
ax[1].set_ylim(-150,150)

plt.xlabel("Date")
plt.tight_layout()
plt.show()
fig.savefig('static_areas_in_time_ige_rolled.png', dpi=300)

