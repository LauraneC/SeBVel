from ticoi.cube_data_classxr import CubeDataClass
from ticoi.seasonality_functions import match_sine
from ticoi.utils import find_granule_by_point
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

from dask.diagnostics import ProgressBar


glacier = "Kask"

#Russel
if glacier == "Russel":
    area_file = "/Users/charriel/Documents/Seasonality/Russel/Russel_sector_for_seasonality.gpkg"
    mask_file = f'/Users/charriel/Documents/Seasonality/Russel/Stables_areas_Russel_from_PROMICE.gpkg'
    path_save = "/Users/charriel/Documents/Seasonality/Russel/"
    i,j = -49.794, 67.124

if glacier == "Kask":
    area_file = "/Users/charriel/Documents/Seasonality/Yukon/Stables_areas_Kask.gpkg"
    mask_file = f'/Users/charriel/Documents/Seasonality/Yukon/Stables_areas_Kask.gpkg'
    path_save = "/Users/charriel/Documents/Seasonality/Yukon/"
    i,j = -138.7574, 60.7723

if glacier == "Russel": label = "terminus of Issunguata Sermia"
elif glacier == "Kask": label = "tongue of Kaskawulsh"


def seasonal_sine_model(doy, A, maxday, C):
    omega = 2 * np.pi / 365.25
    return C + A * np.cos(omega * (doy - maxday))

def match_sine_new(
    d,
    value_col="vx",
    impose_frequency=True
):
    """
    Fit a 1-year sinusoid with offset to a time series.

    Parameters
    ----------
    d : pandas.DataFrame
        Must contain datetime column and data column
    time_col : str
        Column with datetime
    value_col : str
        Column to fit
    impose_frequency : bool
        (kept for compatibility, frequency is fixed to 1 year)

    Returns
    -------
    A : float
        Amplitude
    maxday : float
        Day-of-year of maximum
    C : float
        Offset
    pcov : 2D array
        Covariance matrix
    """

    # --- prepare data ---
    dates = pd.to_datetime(d.index)
    if isinstance(dates,pd.DatetimeIndex): doy = dates.dayofyear.values
    else: doy = dates.dt.dayofyear.values
    y = d[value_col].values

    mask = np.isfinite(doy) & np.isfinite(y)
    doy = doy[mask]
    y = y[mask]

    # --- initial guesses ---
    C0 = np.nanmean(y)
    A0 = 0.5 * (np.nanmax(y) - np.nanmin(y))
    maxday0 = doy[np.nanargmax(y)]

    p0 = [A0, maxday0, C0]

    # --- fit ---
    popt, pcov = curve_fit(
        seasonal_sine_model,
        doy,
        y,
        p0=p0
    )

    A, maxday, C = popt

    return A, maxday, C, pcov

def evaluate_fitted_sine(dates, A, maxday, C):
    dates = pd.to_datetime(dates)
    if isinstance(dates,pd.DatetimeIndex): doy = dates.dayofyear.values
    else: doy = dates.dt.dayofyear.values
    return seasonal_sine_model(doy, A, maxday, C)

# %%========================================================================= #
#                                   CUBE LOADING                               #
# =========================================================================%% #
list_mask = []

mask = gpd.read_file(area_file)
list_coordinates = [[mask.bounds.minx[0],mask.bounds.miny[0]],[mask.bounds.maxx[0],mask.bounds.miny[0]],[mask.bounds.minx[0],mask.bounds.maxy[0]],[mask.bounds.maxx[0],mask.bounds.maxy[0]]]
cube_list = np.unique([find_granule_by_point([k[0],k[1]]) for k in list_coordinates])
print(cube_list)

for cube_name in tqdm(cube_list) :
    if len(cube_name) != 0:
        cube = CubeDataClass()
        cube.load(cube_name, pick_sensor= ["Sentinel-2"],mask=area_file,pick_date = ["2016-01-01","2025-12-31"])
        print(cube.nx, cube.ny,cube.nz)
        if glacier == "Russel":cube.mask_cube(mask_file, invert=True)
        # cube_mean = cube.ds.mean(dim=["x", "y"], skipna=True).compute(scheduler='threads', num_workers=4)
        list_mask.append(cube.ds)

# %%========================================================================= #
#                           AVERAGE VELOCITY OVER CUBE                        #
# =========================================================================%% #

t = xr.concat(list_mask,dim="mid_date")
print('Concat done')
t = t.chunk({"x": -1, "y": -1, "mid_date": 1000})
with ProgressBar():
    t_mean_time = t.mean(dim=["x", "y"], skipna=True).compute(scheduler='threads', num_workers=4)
print('Mean done')

### Compute velocity magnitude and save nc
t_mean_time["vv"] = np.sqrt(t_mean_time["vx"]**2 + t_mean_time["vy"]**2)
t_mean_time[["vx","vy","vv","date1","date2","mapping","spatial_ref","mid_date"]].to_netcdf(f"{path_save}/tmeantime.nc",engine="h5netcdf")

# %%========================================================================= #
#                           PLOT STATIC AREAS                                 #
# =========================================================================%% #

# Plot of static areas
fig, ax = plt.subplots()
ax.plot(t_mean_time.mid_date,t_mean_time['vv'],linestyle='',marker='o',color='b',markersize=2,label="static areas")
ax.legend()
ax.set_ylabel("vv [m/y]")
plt.show()

#Superimpose with velocities on moving areas, and a given month
url_ls = find_granule_by_point([i, j])
cube = CubeDataClass()
cube.load(url_ls, pick_sensor= ["Sentinel-2"],pick_date = ["2016-01-01","2025-12-31"])
pix = cube.load_pixel(i,j,output_format="df",visual=True)[0]
pix["vv"] = np.sqrt(pix["vx"]**2 + pix["vy"]**2)

fig, ax = plt.subplots(2,1)
ax[0].plot(t_mean_time.mid_date,t_mean_time['vv'],linestyle='',marker='o',color='b',markersize=2,label="static areas")
ax[0].legend()
ax[0].set_ylim(0,500)
ax[0].set_ylabel("vv [m/y]")
ax[1].plot(pix.index,pix['vv'],linestyle='',marker='o',color='orange',markersize=2,label=label)
ax[1].set_ylim(0,500)
ax[1].set_ylabel("vv [m/y]")
ax[1].legend()
x_min = min(t_mean_time.mid_date.min().values, pix.index.min())
x_max = max(t_mean_time.mid_date.max().values, pix.index.max())
# Extract years from the date range
start_year = pd.Timestamp(x_min).year
end_year = pd.Timestamp(x_max).year
# Create June 01 dates for each year in the range
july_30_dates = [pd.Timestamp(f'{year}-06-01') for year in range(start_year, end_year + 1)]
# Add vertical lines to both subplots
for date in july_30_dates:
    ax[0].axvline(x=date, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax[1].axvline(x=date, color='grey', linestyle='--', linewidth=1, alpha=0.7)
plt.show()
fig.savefig(f'{path_save}/moving_areas_in_time_its_vv_superimpose.png',dpi=300)

# %%========================================================================= #
#                           PLOT SINUSOID ON TOP                              #
# =========================================================================%% #
# plot sinusoidal function on top
variable = "vv"
baseline_treshold = 180
df = t_mean_time[["vx","vy","vv","date1","date2","temporal_baseline","mid_date"]].to_dataframe()
df_smallb = df[df["temporal_baseline"]<baseline_treshold]
A_static, maxday_static, C, pcov = match_sine_new(
    d=df_smallb,
    value_col=variable,
)
vx_fit_static = evaluate_fitted_sine(
    df_smallb.index,
    A_static, maxday_static, C
)
print(A_static,maxday_static)
f,A_moving,maxday_moving = match_sine(d=df_smallb,impose_frequency=True,raw_seasonality=False,variable=variable)
print(f"Nathan code {A_moving,maxday_moving}")

pix_smallb = pix[pix["temporal_baseline"]<baseline_treshold]
A_moving, maxday_moving, C, pcov = match_sine_new(
    d=pix_smallb,
    value_col=variable,
)
vx_fit_moving = evaluate_fitted_sine(
    pix_smallb.index,
    A_moving, maxday_moving, C
)
print(A_moving,maxday_moving)

f,A_moving,maxday_moving = match_sine(d=pix_smallb,impose_frequency=True,variable=variable)
print(f"Nathan code {A_moving,maxday_moving}")


fig, ax = plt.subplots(2,1)
ax[0].plot(df_smallb.index,df_smallb[variable],linestyle='',marker='o',color='b',markersize=2,label="static areas")
ax[0].plot(
    df_smallb.index,
    vx_fit_static,
    color='k',
    linewidth=2,
    label="seasonal sine fit"
)

ax[0].legend()
ax[0].set_ylim(-300,500)
ax[0].set_ylabel(f"{variable} [m/y]")
ax[1].plot(pix_smallb.index,pix_smallb[variable],linestyle='',marker='o',color='orange',markersize=2,label=label)
ax[1].plot(
    pix_smallb.index,
    vx_fit_moving,
    color='k',
    linewidth=2,
    label="seasonal sine fit"
)
ax[1].set_ylim(-300,500)
ax[1].set_ylabel(f"{variable} [m/y]")
ax[1].legend()
x_min = min(t_mean_time.mid_date.min().values, pix.index.min())
x_max = max(t_mean_time.mid_date.max().values, pix.index.max())
# Extract years from the date range
start_year = pd.Timestamp(x_min).year
end_year = pd.Timestamp(x_max).year
# Create June 01 dates for each year in the range
july_30_dates = [pd.Timestamp(f'{year}-06-01') for year in range(start_year, end_year + 1)]
# Add vertical lines to both subplots
for date in july_30_dates:
    ax[0].axvline(x=date, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax[1].axvline(x=date, color='grey', linestyle='--', linewidth=1, alpha=0.7)
plt.show()
fig.savefig(f'{path_save}/moving_areas_in_time_its_{variable}_superimpose.png',dpi=300)

print("Start rollling mean")

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



fig, ax = plt.subplots(2,1)
ax[0].plot(t_mean_time.mid_date, t_mean_time['vv'], 'o', color='lightgray', markersize=2, label='Image-pair velocities')
ax[0].plot(t_mean_time_rolled.mid_date,t_mean_time_rolled['vv'],linestyle='',marker='o',color='b',markersize=2,label="static areas")
ax[0].legend()
ax[0].set_ylim(0,500)
ax[0].set_ylabel("vv [m/y]")
ax[1].plot(pix.index,pix['vv'],linestyle='',marker='o',color='orange',markersize=2,label="terminus of Issunguata Sermia")
ax[1].set_ylim(0,500)
ax[1].set_ylabel("vv [m/y]")
ax[1].legend()
plt.show()
fig.savefig(f'{path_save}/moving_areas_in_time_its_vv_superimpose_rolled.png',dpi=300)



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

