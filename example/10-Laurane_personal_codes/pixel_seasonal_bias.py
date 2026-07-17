""" Script to plot the seasonal signal on top of raw image-pair velocities. The seasonal signal is computed as
The seasonal amplitude is computed using a weighted rolling median or mean applied on the image-pair velocities stacked according to their respective day of the year. The window size of the rolling can be specified. The weights are defined by an exponential decay based on the temporal baselines: $exp(-\Delta_t /L)$, with $\Delta_t$ the temporal baseline and $L$ an empirical value set to 90  to provide higher weights to shorter temporal baselines (which are closer to the instantaneous velocity)."""


from ticoi.cube_data_classxr import CubeDataClass
from ticoi.utils import find_granule_by_point
import matplotlib.pyplot as plt
from SeBVel.ige_acess import cube_name_from_shp_points
from SeBVel.seaso_amp import rolling_circular_pixel, doy_to_month
import pandas as pd
import numpy as np


ROLLING_METHOD = "median"   # mean or median
WINDOW = 30 # window of rolling median or mean
BASELINE_THRESHOLD = 500 # maximal temporal baselines
WEIGHT_MODE = "exp" #exp, inv or None used to weight the temporal baselines
MIN_POINTS = 5 # minimal number of points needed to derive seasonal statistics in the time-series
SAVE_FIG = True
PLOT_ONE_COMP = "vy"

glacier = "Kask"
#dataset = "IGE_non_filtered"
dataset = "ITS_LIVE"

#Russel
if glacier == "Russel":
    path_save = "/Users/charriel/Documents/Seasonality/Russel/"
    i,j = -49.7944, 67.1338

if glacier == "Kask":
    path_save = "/Users/charriel/Documents/Seasonality/Yukon/"
    i,j = -138.7574, 60.7723

if glacier == "stable_areas":
    path_save = "/Users/charriel/Documents/Seasonality/"
    i,j = -50.348547,67.339909

elif glacier == "Alps_exampleTC":
    path_save = "/Users/charriel/Documents/Seasonality/Alps/"
    i,j = 6.86759, 45.83930
    mask_shp = "/Users/charriel/Documents/Seasonality/Alps/shadows_projection/shadow_points_example/example_shadow_point_for_paper.gpkg"

elif glacier == "Alps_no_shadow":
    path_save = "/Users/charriel/Documents/Seasonality/Alps/"
    i,j = 7.03004, 45.867817
    mask_shp = "/Users/charriel/Documents/Seasonality/Alps/point_with_no_shadow.gpkg"


if glacier == "Russel": label = "terminus of Issunguata Sermia"
elif glacier == "Kask": label = "tongue of Kaskawulsh"

if dataset == "IGE_non_filtered":
    prefix = '_all.nc'
    folder_cube = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cubes_S2_non_filter/"

catalog_cubeige = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cube_grid.shp"

if dataset == "IGE_non_filtered":
    points = cube_name_from_shp_points(mask_shp, catalog_cubeige)
    cube_name = f"{folder_cube}{str(points['cube'][0][0]).split('_post0050_yr*')[0]}{prefix}"
if dataset== "ITS_LIVE":
    cube_name = find_granule_by_point([i, j])

# %%========================================================================= #
#                                   CUBE LOADING                               #
# =========================================================================%% #


cube = CubeDataClass()
cube.load(cube_name,pick_date = ["2016-01-01","2025-12-31"],pick_sensor = ["Sentinel-2"],pick_temp_bas = [0 ,BASELINE_THRESHOLD])

# %%========================================================================= #
#                                   PIXEL LOADING                               #
# =========================================================================%% #


pix = cube.load_pixel(i,j,output_format="df",visual=True)[0]
pix["vv"] = np.sqrt(pix["vx"]**2 + pix["vy"]**2)
pix["doy"] = pix.index.dayofyear

# %%========================================================================= #
#                                   ROLLING MEAN COMPUTATION                               #
# =========================================================================%% #

# Use the new circular rolling function directly on raw timeseries
rolled_pix = rolling_circular_pixel(
    pix,
    value_cols=("vx","vy","vv"),
    window=WINDOW,
    min_points= MIN_POINTS,
    method=ROLLING_METHOD,weights_col="temporal_baseline",weight_mode=WEIGHT_MODE,
)

# %%========================================================================= #
#                                   STATS COMPUTATION                              #
# =========================================================================%% #

stats_summary = {}
for comp in ["vx","vy","vv"]:
    s = rolled_pix[f"{comp}_roll"]

    vmax = s.max()
    vmin = s.min()
    mean = s.mean()
    med = s.median()
    amplitude = vmax - vmin
    amplitude2 = (abs(vmax - med) + abs(vmin - med))
    date_max = rolled_pix.loc[s.idxmax(), "doy"]
    if isinstance(date_max,pd.Series): date_max = date_max[0]
    date_min = rolled_pix.loc[s.idxmin(), "doy"]
    # if len(date_min) > 0: date_min = date_min[0]

    stats_summary[comp] = dict(
        max=vmax,
        min=vmin,
        mean=mean,
        amplitude=amplitude,
        amplitude2=amplitude2,
        date_max=date_max,
        date_min=date_min
    )

# %%========================================================================= #
#                                   FIGURES                             #
# =========================================================================%% #


fig, ax = plt.subplots(3, 1, figsize=(5, 9), sharex=True)

components = ["vx", "vy", "vv"]

for i, comp in enumerate(components):
    #y limits
    if comp == "vv": ax[i].set_ylim(0,pix[comp].mean()+2*pix[comp].std())
    else: ax[i].set_ylim(pix[comp].mean()-2*pix[comp].std(),pix[comp].mean()+2*pix[comp].std())

    # --- stacked scatter ---
    ax[i].plot(
        pix["doy"],
        pix[comp],
        linestyle="",
        marker="o",
        color="orange",
        markersize=2,
        alpha=0.5,
    )

    # --- rolling curve ---
    ax[i].plot(
        rolled_pix["doy"],
        rolled_pix[f"{comp}_roll"],
        color="blue",
        linestyle="",
        marker="o",
        markersize=2,
        label=f"{WINDOW}-day {ROLLING_METHOD}",
    )

    # --- stats textbox ---
    st = stats_summary[comp]

    txt = (
        f"Max: {st['max']:.1f} m/y\n"
        f"Min: {st['min']:.1f} m/y\n"
        f"Max−Min: {st['amplitude']:.1f} m/y\n")

    ax[i].text(
        0.02, 0.98,
        txt,
        transform=ax[i].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    ax[i].set_ylabel(f"{comp} [m/y]")
    ax[i].legend()

plt.tight_layout()
fig.savefig(f"{path_save}/pixel_seasonal_bias_amplitude_{i,j}.png")
plt.show()


fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, comp in enumerate(components):

    #y limits
    if comp == "vv": ax[i].set_ylim(0,pix[comp].mean()+2*pix[comp].std())
    else: ax[i].set_ylim(pix[comp].mean()-2*pix[comp].std(),pix[comp].mean()+2*pix[comp].std())

    # raw points
    ax[i].plot(
        pix.index,
        pix[comp],
        linestyle="",
        marker="o",
        color="orange",
        markersize=2,
    )

    # # baseline errorbars
    # ax[i].errorbar(
    #     pix.index,
    #     pix[comp],
    #     xerr=half_baseline_td,
    #     linestyle="",
    #     color="orange",
    #     alpha=0.4
    # )

    #Date of the maximum
    st = stats_summary[comp]

    max_month = doy_to_month(st["date_max"])
    years = np.unique(pix.index.year)

    for k, y in enumerate(years):
        date_max_dt = pd.to_datetime(
            y * 1000 + int(st["date_max"]),
            format="%Y%j"
        )

        ax[i].axvline(
            date_max_dt,
            linestyle="--",
            linewidth=1,
            color="k",
            alpha=0.6,
            label=f"Seasonal peak ({max_month})" if k == 0 else None
        )
    #Value of the min and max
    ax[i].axhline(
        st["max"],
        linestyle="-",
        linewidth=1.5,
        color="red",
        alpha=0.9,
        label=f"Seasonal amplitude {st['amplitude']:.1f} m/y"
    )

    ax[i].axhline(
        st["min"],
        linestyle="-",
        linewidth=1.5,
        color="red",
        alpha=0.9
    )

    ax[i].set_ylabel(f"{comp} [m/y]")
    ax[i].legend()

plt.tight_layout()
fig.savefig(f"{path_save}/pixel_seasonal_bias_{i,j}.png")
plt.show()

if PLOT_ONE_COMP is not None:
    comp = PLOT_ONE_COMP
    fig, ax = plt.subplots(figsize=(6, 3), sharex=True)
    ax.set_ylim(pix[comp].mean()-2*pix[comp].std(),pix[comp].mean()+2*pix[comp].std())

    # raw points
    ax.plot(
        pix.index,
        pix[comp],
        linestyle="",
        marker="o",
        color="orange",
        markersize=2,
    )
    #Date of the maximum
    st = stats_summary[comp]

    max_month = doy_to_month(st["date_max"])
    years = np.unique(pix.index.year)

    for k, y in enumerate(years):
        date_max_dt = pd.to_datetime(
            y * 1000 + int(st["date_max"]),
            format="%Y%j"
        )

        ax.axvline(
            date_max_dt,
            linestyle="--",
            linewidth=1,
            color="k",
            alpha=0.6,
            label=f"Seasonal peak: {st['date_max']} ({max_month})" if k == 0 else None
        )
    #Value of the min and max
    ax.axhline(
        st["max"],
        linestyle="-",
        linewidth=1.5,
        color="red",
        alpha=0.9,
        label=f"Seasonal amplitude {st['amplitude']:.1f} m/y"
    )

    ax.axhline(
        st["min"],
        linestyle="-",
        linewidth=1.5,
        color="red",
        alpha=0.9
    )

    ax.set_ylabel(f"{comp} [m/y]")
    ax.legend()

    plt.tight_layout()
    fig.savefig(f"{path_save}/pixel_seasonal_bias_{i,j}_{PLOT_ONE_COMP}.png")
    plt.show()
