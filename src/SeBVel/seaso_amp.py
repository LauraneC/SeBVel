import ticoi.cube_data_classxr
from numba import prange
import numpy as np
import pandas as pd
import time
import rasterio
from numba import njit
from typing import Literal
import xarray as xr
from datetime import datetime, timedelta
import geoutils as gu
from geoutils import raster
import glob
import geopandas as gpd
import rasterio.mask
import matplotlib.pyplot as plt
from shapely.geometry import mapping

MethodRolling = Literal["median", "mean", "weighted_median"]

@njit
def weighted_median(values:np.ndarray, weights:np.ndarray)->float:
    """
    Compute the weighted median of a list of values.
    :param values: data to compute weighted median
    :param weights: weights
    :return: weighted median
    """

    # Remove NaNs
    n = len(values)
    valid_v = []
    valid_w = []

    for i in range(n):
        v = values[i]
        if not np.isnan(v):
            valid_v.append(v)
            valid_w.append(weights[i])

    if len(valid_v) == 0:
        return np.nan

    v = np.array(valid_v)
    w = np.array(valid_w)

    # Sort by value
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]

    # Normalize weights
    w_sum = np.sum(w)
    if w_sum == 0:
        return np.nan
    w = w / w_sum

    # Cumulative sum
    cw = np.cumsum(w)

    # Median position
    for i in range(len(cw)): #take 50% of w*v
        if cw[i] >= 0.5:
            return v[i]

    return v[-1]

@njit(parallel=True)
def rolling_weighted_median_circular(data:np.array, doy:np.array, window:int, min_points:int,weights:np.array)->np.array:
    """
    Compute a circular rolling weighted median over a window of data.
    :param data: data to roll over
    :param doy: day of the year
    :param window: window size
    :param min_points: minimum number of points to consider
    :return: rolling weighted median
    """

    nt, ny, nx = data.shape
    nd = 366
    half = window // 2

    out = np.full((nd, ny, nx), np.nan, dtype=np.float32)

    for y in prange(ny):
        for x in range(nx):

            ts = data[:, y, x]

            for d in range(1, nd + 1):

                buf_v = []
                buf_w = []

                for t in range(nt):

                    dd = doy[t] - d
                    if dd > 183:
                        dd -= 366
                    elif dd < -183:
                        dd += 366

                    if abs(dd) <= half:
                        v = ts[t]
                        w = weights[t]

                        if not np.isnan(v) and not np.isnan(w):
                            buf_v.append(v)
                            buf_w.append(w)

                if len(buf_v) >= min_points:
                    out[d-1, y, x] = weighted_median(
                        np.array(buf_v),
                        np.array(buf_w)
                    )

    return out

@njit(parallel=True)
def compute_stats(arr:np.array,min_points:int=10)->(float,float,float,float):
    """
    Compute the seasonal amplitude (max - min) and the date of the maximal value of a given time series.
    :param arr: time series data
    :param min_points: minimum number of points needed to compute the statics
    :return: date of maximal value, amplitude
    """

    nd, ny, nx = arr.shape

    vmax = np.full((ny, nx), np.nan)
    vmin = np.full((ny, nx), np.nan)
    datemax = np.full((ny, nx), np.nan)
    vamp = np.full((ny, nx), np.nan)


    for y in prange(ny):
        for x in range(nx):

            series = arr[:, y, x]

            valid = series[~np.isnan(series)] #remove nan

            if len(valid) < min_points:#check if the series is empty or almost
                continue

            vmax[y, x] = np.max(valid)
            vmin[y, x] = np.min(valid)
            datemax[y, x] = np.argmax(valid)
            vamp[y, x] = vmax[y, x]-vmin[y, x]

    return datemax, vamp

def seasonality_numpy_numba(vx:np.ndarray, vy:np.ndarray, doy:np,
                           window:int=30,weights:np.ndarray| None=None,
                           min_points_stats:int=5,min_points_rolling:int=10,verbose:bool=False)-> dict:
    """
    Apply a rolling mean, median, or weighted median to a given time series.
    Then, compute the amplitude (as the difference between the minimal and maximal value of the velocity), and the peak timing (as the date of the maximal velocity)
    :param vx: velocity component values along x-direction
    :param vy: velocity component values along y-direction
    :param doy: day of the year
    :param window: window size of the rolling median
    :param weights: weights used to compute rolling median
    :param min_points_stats: minimum number of points needed to compute the statics
    :param min_points_rolling: minimum number of points needed to compute the rolling
    :return: amplitude and date of maximal velocity for velocity component values along x-direction, velocity component values along y-direction and the velocity magnitude
    """

    vv = np.sqrt(vx**2 + vy**2)

    # --- Rolling ---
    if verbose:
        print("Start compute rolling...")
        start = time.time()

    vx_roll = rolling_weighted_median_circular(
        data=vx, weights=weights, doy=doy, window=window, min_points=min_points_rolling
    )
    vy_roll = rolling_weighted_median_circular(
        data=vy, weights=weights, doy=doy, window=window, min_points=min_points_rolling
    )
    vv_roll = rolling_weighted_median_circular(
        data=vv, weights=weights, doy=doy, window=window, min_points=min_points_rolling
    )

    if verbose: print(f"Rolling took {time.time()-start} seconds...")

    # --- Stats ---
    if verbose:
        print("Start computing stats...")
        start = time.time()
    vx_stats = compute_stats(arr=vx_roll,min_points=min_points_stats)
    vy_stats = compute_stats(arr=vy_roll,min_points=min_points_stats)
    vv_stats = compute_stats(arr=vv_roll,min_points=min_points_stats)
    if verbose: print(f"Stats took {time.time()-start} seconds...")

    return {
        "vx": vx_stats,
        "vy": vy_stats,
        "vv": vv_stats,
    }

def save_geotiff(ds:xr.Dataset,stats:dict,path_save:str,save_name:str):
    """
    Save statistics in geotiff format
    :param ds: cube dataarray with the time series inside
    :param stats: dictionary with statistics
    :param path_save: path where to save the data
    :param save_name: filename of the geotiff
    """
    list_var_to_save = ["vx", "vy", "vv"]

    transform = ds.rio.transform()
    crs = ds.rio.crs

    for var_name in list_var_to_save:
        data_datemax = stats[var_name][0]  # take amplitude only for saving  # NumPy array
        data_amp = stats[var_name][1]  # take amplitude only for saving  # NumPy array

        output_file = f"{path_save}/{save_name}{var_name}.tif"

        ny, nx = data_amp.shape

        with rasterio.open(
                output_file,
                "w",
                driver="GTiff",
                height=ny,
                width=nx,
                dtype=data_amp.dtype,
                count=2,
                crs=crs,
                transform=transform,
                compress="LZW",
        ) as dst:
            dst.write(data_amp.astype("float32"), 1)
            dst.write(data_datemax.astype("float32"), 2)

            dst.set_band_description(1, "Amplitude")
            dst.set_band_description(2, "DOY_of_maximum")

def process(ds: xr.Dataset,ix:int=None,iy:int=None,window:int=30,min_points_rolling:int=5,min_points_stats:int=10,L:int=90,save_name:str=None,path_save:str=None,baseline_treshold:int=500,verbose:bool=False):
    """
    Compute a rolling weighted circular median over a given time series stacked according to the DOY.
    Then, estimate some statistics (date of the maximum and amplitude).
    Save the date of the maximum and the amplitude as geotiff
    :param ds: data array with velocity values
    :param ix: index of split along x
    :param iy: index of split along y
    :param window: window size of the rolling median
    :param min_points_rolling: minimum number of points to consider for the statistics
    :param min_points_stats: minimum number of points to consider for the statistics
    :param L: length of the rolling median
    :param save_name: name of the output file
    :param path_save: path where to save the data
    :param verbose: if True display some messages
    :return:
    """

    if verbose:
        print("Start Loading a variables into memory...")
        start = time.time()

    vx = ds["vx"].values
    vy = ds["vy"].values
    dates = ds["mid_date"].values
    doy = pd.to_datetime(dates).dayofyear.values

    #Compute the weights
    baseline = ds["temporal_baseline"].values
    weights = np.exp(-baseline / L)

    if verbose:
        print(f"Loading took {time.time() - start} seconds...")
        print("Computing stats...")

    # Compute seasonal amplitudes and peak timings
    start = time.time()
    stats = seasonality_numpy_numba(
        vx=vx, vy=vy, doy=doy,
        window=window,
        min_points_stats=min_points_stats,min_points_rolling=min_points_rolling,weights=weights,verbose=verbose)

    if verbose:
        print(f"Computing stats took {time.time() - start} seconds...")
        print("Saving tiff...")
        start = time.time()

    # Save the results as geotiff
    if path_save is not None:
        if ix!= None: save_name_i = f"{save_name}_amplitude_datamax_bastresh{baseline_treshold}_{ix}_{iy}"
        else: save_name_i = f"{save_name}_amplitude_datamax_bastresh_{baseline_treshold}"
        save_geotiff(ds=ds, stats=stats, path_save=path_save, save_name=save_name_i)
        if verbose:
           print(f"Saving tiff took {time.time() - start} seconds...")

def process_split(cube:ticoi.cube_data_classxr.CubeDataClass,ix:int, iy:int,x_step:int,y_step:int,min_points_rolling:int=5,nx_split:int=1,ny_split:int=1,path_save=None,verbose:bool=False):
    """
    Spliting the computation of seasonal amplitude and peak timing into several tiles
    :param cube:
    :param ix: index of split along x
    :param iy: index of split along y
    :param x_step:
    :param y_step:
    :param min_points_rolling: minimum number of points to consider for the statistics
    :param nx_split:
    :param ny_split:
    :param path_save: path where to save the data
    :param verbose: if True display some messages
    :return:
    """

    if verbose:print(f"Processing tile ({ix}, {iy})")

    x0 = ix * x_step
    x1 = (ix + 1) * x_step if ix < nx_split - 1 else cube.nx

    y0 = iy * y_step
    y1 = (iy + 1) * y_step if iy < ny_split - 1 else cube.ny

    ds = cube.ds.isel(
        x=slice(x0, x1),
        y=slice(y0, y1)
    )
    process(ds=ds, ix=ix, iy=iy,min_points_rolling=min_points_rolling,path_save=path_save,verbose=verbose)

def rolling_circular_pixel(
    df,
    value_cols=("vx", "vy", "vv"),
    window=30,
    min_points=5,    # None | "inverse" | "exp"
    L=180.0
):
    """
    Circular rolling statistics for a single pixel.

    For weighted median, uses rolling_weighted_median_circular()
    internally and maps the 366-day result back to each observation's DOY.
    """

    df_roll = df.copy()

    doy = df["doy"].to_numpy(dtype=np.int32)

    # --------------------------------------------------
    # Build weights
    # --------------------------------------------------
    baseline = df["temporal_baseline"].to_numpy(dtype=np.float64)
    weights_all = np.exp(-baseline / L)

    # --------------------------------------------------
    # Process each variable
    # --------------------------------------------------
    for col in value_cols:

        values = df[col].to_numpy(dtype=np.float64)

        # ==================================================
        # WEIGHTED MEDIAN
        # ==================================================
        # rolling_weighted_median_circular expects:
        # (nt, ny, nx)
        # Here ny=nx=1 because this dataframe represents one pixel.
        data_3d = values[:, None, None]

        daily_roll = rolling_weighted_median_circular(
            data=data_3d,
            doy=doy,
            window=window,
            min_points=min_points,
            weights=weights_all
        )

        # daily_roll shape = (366, 1, 1)
        daily_roll = daily_roll[:, 0, 0]

        # Map the daily climatology back to the
        # acquisition DOYs.
        rolled_vals = daily_roll[doy - 1]

        df_roll[f"{col}_roll"] = rolled_vals

    return df_roll

def doy_to_month(doy:int):
    """
    Convert doy to month (in string)
    :param doy: day of the year
    :return: month
    """
    date = datetime(2023, 1, 1) + timedelta(int(doy) - 1)
    return date.strftime("%b")  # abbreviated month (Jan, Feb, …)


def concatenate_geotiff(path_tiff,type_list:list = ['phase', 'amplitude'],vexist:bool=True):

    for type in type_list:
        if type == 'amplitude':
            band = 1
        elif type == 'phase':
            band = 2

        vx_list = glob.glob(f'{path_tiff}/*vx.ti*')
        vy_list = glob.glob(f'{path_tiff}/*vy.ti*')
        if vexist:     vv_list = glob.glob(f'{path_tiff}/*vv.ti*')


        rast_list = [gu.Raster(raster_i, bands=band) for raster_i in vx_list]
        if len(rast_list)>1: amplitudex = raster.merge_rasters(rast_list)
        else: amplitudex = rast_list[0]

        rast_list = [gu.Raster(raster_i, bands=band) for raster_i in vy_list]
        if len(rast_list)>1:amplitudey = gu.raster.merge_rasters(rast_list)
        else: amplitudey = rast_list[0]


        if vexist:
            rast_list = [gu.Raster(raster_i, bands=band) for raster_i in vv_list]
            if len(rast_list)>1:amplitudev = gu.raster.merge_rasters(rast_list)
            else: amplitudev = rast_list[0]

        amplitudex.save(f"{path_tiff}/{type}x.tif")
        amplitudey.save(f"{path_tiff}/{type}y.tif")
        if vexist: amplitudev.save(f"{path_tiff}/{type}v.tif")


def mask_geotiff(path_tiff,shapefile_mask,var_list = ['amplitudex', 'amplitudey', 'amplitudev']):

    for variable in var_list:

        raster_path = f"{path_tiff}/{variable}.tif"  # change to x or y
        output_raster_path = f"{path_tiff}/{variable}_cropped_massif.tif"  # change to x or y

        # Read the shapefile
        shapefile = gpd.read_file(shapefile_mask)
        # Remove empty geometries
        shapefile = shapefile[~shapefile.geometry.is_empty]

        if shapefile.empty:
            raise ValueError("The shapefile contains only empty geometries. Cannot proceed with masking.")

        # Convert geometries to GeoJSON-like format
        geometries = [mapping(geom) for geom in shapefile.geometry]

        # Read the raster file
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True)
            out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Save the masked raster
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)
