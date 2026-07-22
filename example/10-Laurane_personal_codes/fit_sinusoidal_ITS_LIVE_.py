import matplotlib
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.seasonality_functions import match_sine
from ticoi.utils import find_granule_by_point

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from SeBVel.ige_acess import cube_name_from_shp_points
from dask.diagnostics import ProgressBar


glacier = "Kask"
dataset = "ITS_LIVE"

#Russel
if glacier == "Russel":
    path_save = "/Users/charriel/Documents/Seasonality/Russel/"
    i,j = -49.794, 67.124

if glacier == "Kask":
    path_save = "/Users/charriel/Documents/Seasonality/Yukon/"
    i,j = -138.7574, 60.7723

if glacier == "stable_areas":
    path_save = "/Users/charriel/Documents/Seasonality/"
    i,j = -50.348547,67.339909

elif glacier == "Alps":
    path_save = "/Users/charriel/Documents/Seasonality/Alps/"
    i,j = 6.86759, 45.83930

if glacier == "Russel": label = "terminus of Issunguata Sermia"
elif glacier == "Kask": label = "tongue of Kaskawulsh"

if dataset == "IGE_non_filtered":
    prefix = '_all.nc'
    folder_cube = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cubes_S2_non_filter/"

catalog_cubeige = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cube_grid.shp"
mask_shp = "/Users/charriel/Documents/Seasonality/Alps/shadows_projection/shadow_points_example/example_shadow_point_for_paper.gpkg"

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


if dataset == "IGE_non_filtered":
    points = cube_name_from_shp_points(mask_shp, catalog_cubeige)
    cube_name = f"{folder_cube}{str(points['cube'][0][0]).split('_post0050_yr*')[0]}{prefix}"
if dataset== "ITS_LIVE": cube_name = find_granule_by_point([i, j])

cube = CubeDataClass()
#cube.load(url_ls, pick_sensor= ["Sentinel-2"],pick_date = ["2016-01-01","2025-12-31"])
cube.load(cube_name,pick_date = ["2016-01-01","2025-12-31"],pick_sensor = ["Sentinel-2"])

pix = cube.load_pixel(i,j,output_format="df",visual=True)[0]
pix["vv"] = np.sqrt(pix["vx"]**2 + pix["vy"]**2)

baseline_treshold = 180
pix_smallb = pix[pix["temporal_baseline"]<baseline_treshold]


A_moving_vx, maxday_moving, C, pcov = match_sine_new(
    d=pix_smallb,
    value_col="vx",
)
vx_fit_vx = evaluate_fitted_sine(
    pix_smallb.index,
    A_moving_vx, maxday_moving, C
)
print(A_moving_vx,maxday_moving)
f,A_moving,maxday_moving = match_sine(d=pix_smallb,impose_frequency=True,variable="vx")
print(f"Nathan code {A_moving,maxday_moving}")

A_moving_vy, maxday_moving, C, pcov = match_sine_new(
    d=pix_smallb,
    value_col="vy",
)
print(A_moving_vy,maxday_moving)

vx_fit_vy = evaluate_fitted_sine(
    pix_smallb.index,
    A_moving_vy, maxday_moving, C
)
f,A_moving,maxday_moving = match_sine(d=pix_smallb,impose_frequency=True,variable="vy")
print(f"Nathan code {A_moving,maxday_moving}")

A_moving_vv, maxday_moving, C, pcov = match_sine_new(
    d=pix_smallb,
    value_col="vv",
)
vv_fit_vy = evaluate_fitted_sine(
    pix_smallb.index,
    A_moving_vv, maxday_moving, C
)

fig, ax = plt.subplots(3,1)
ax[0].plot(pix_smallb.index,pix_smallb["vx"],linestyle='',marker='o',color='orange',markersize=2)
ax[0].plot(
    pix_smallb.index,
    vx_fit_vx,
    color='k',
    linewidth=2,
    label=f"seasonal sine fit with an amplitude of {np.round(abs(A_moving_vx),1)} m/y"
)
ax[0].legend()
ax[0].set_ylabel("vx [m/y]")
ax[1].plot(pix_smallb.index,pix_smallb["vy"],linestyle='',marker='o',color='orange',markersize=2)
ax[1].plot(
    pix_smallb.index,
    vx_fit_vy,
    color='k',
    linewidth=2,
    label=f"seasonal sine fit with an amplitude of {np.round(abs(A_moving_vy),1)} m/y"
)
ax[1].set_ylabel("vy [m/y]")
ax[1].legend()
ax[2].plot(pix_smallb.index,pix_smallb["vv"],linestyle='',marker='o',color='orange',markersize=2)
ax[2].plot(
    pix_smallb.index,
    vv_fit_vy,
    color='k',
    linewidth=2,
    label=f"seasonal sine fit with an amplitude of {np.round(abs(A_moving_vv),1)} m/y"
)
ax[2].set_ylabel("vy [m/y]")
ax[2].legend()
plt.show()
fig.savefig(f'{path_save}/static_areas_in_time_its{baseline_treshold}.png',dpi=300)