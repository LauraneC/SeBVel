from ticoi.cube_data_classxr import CubeDataClass
from ticoi.utils import find_granule_by_point
import xarray as xr
import glob
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import rasterio
from pyproj import CRS


glacier = "Alps"
dataset = "IGE_filtered"

#Russel
if glacier == "Russel":
    area_file = "/Users/charriel/Documents/Seasonality/Russel/Russel_sector_for_seasonality.gpkg"
    path_save = "/Users/charriel/Documents/Seasonality/Russel/"
    result_fn = "Russel_static_areas"  # Name of the netCDF file to be created (if save is True)

if glacier == "Kask":
    area_file = "/Users/charriel/Documents/Seasonality/Yukon/Stables_areas_Kask.gpkg"
    mask_file = f'/Users/charriel/Documents/Seasonality/Yukon/Stables_areas_Kask.gpkg'
    path_save = "/Users/charriel/Documents/Seasonality/Yukon/"
    result_fn = "Kask_static_areas"  # Name of the netCDF file to be created (if save is True)

elif glacier == "Alps":
    area_file = "/Users/charriel/Documents/Seasonality/Alps/metadata/AOI_laurane_seasonality4326.gpkg"
    path_save = "/Users/charriel/Documents/Seasonality/Alps/"

if dataset == "ITS_LIVE":
    mask = gpd.read_file(area_file)
    list_coordinates = [[mask.bounds.minx[0],mask.bounds.miny[0]],[mask.bounds.maxx[0],mask.bounds.miny[0]],[mask.bounds.minx[0],mask.bounds.maxy[0]],[mask.bounds.maxx[0],mask.bounds.maxy[0]]]
    cube_list = np.unique([find_granule_by_point([k[0],k[1]]) for k in list_coordinates])
    path_save += "ITS_LIVE/"


elif dataset == "IGE_filtered":
    cube_IGE_path = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cubes_filtered"
    cube_list = glob.glob(cube_IGE_path + "/*.nc")
    path_save += "IGE_filtered/Displacement/"

elif dataset == "IGE_non_filtered":
    cube_IGE_path = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cubes_S2_non_filter"
    cube_list = glob.glob(cube_IGE_path + "/*.nc")
    path_save += "IGE_non_filtered/"

print(cube_list)

for cube_name in tqdm(cube_list):

    save_name = cube_name.split('/')[-1].split('.')[0]
    print(save_name)

    cube = CubeDataClass()
    cube.load(cube_name, pick_sensor= ["Sentinel-2"],pick_date = ["2016-01-01","2025-12-31"],mask=area_file)
    cube.ds["vx"] = cube.ds["vx"] * cube.ds["temporal_baseline"]/365
    cube.ds["vy"] = cube.ds["vy"] * cube.ds["temporal_baseline"]/365

    print(cube.nx, cube.ny)

    omega = 2 * np.pi / 365.25 # to impose one year frequency
    mid_date = cube.ds.mid_date
    doy = xr.DataArray(
        mid_date.dt.dayofyear,
        dims="mid_date",
        coords={"mid_date ": mid_date }
    )# convert mid_date to day

    cos_t = np.cos(omega * doy)
    sin_t = np.sin(omega * doy)

    def fit_sine_linear(y, cos_t, sin_t):
        mask = np.isfinite(y)#check if infinite or nan
        if mask.sum() < 10:
            return np.nan, np.nan, np.nan

        X = np.stack([
            np.ones(mask.sum()),
            cos_t[mask],
            sin_t[mask]
        ], axis=1)

        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        C, B, D = beta

        A = np.sqrt(B**2 + D**2)
        maxday = (np.arctan2(D, B) / omega) % 365.25

        return A, maxday, C

    def apply_fit(da):
        return xr.apply_ufunc(
            fit_sine_linear,
            da,
            cos_t,
            sin_t,
            input_core_dims=[["mid_date"], ["mid_date"], ["mid_date"]],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float],
        )

    cube.ds["vv"] = np.sqrt(cube.ds.vx**2 + cube.ds.vy**2)
    A_vx, maxday_vx, C_vx = apply_fit(cube.ds.vx)
    A_vy, maxday_vy, C_vy = apply_fit(cube.ds.vy)
    A_vv, maxday_vv, C_vv = apply_fit(cube.ds.vv)



    with rasterio.open(
            f"{path_save}/{save_name}_amplitude_dx.tiff",
            "w",
            driver="GTiff",
            height=A_vx.shape[0],
            width=A_vx.shape[1],
            count=1,
            dtype=str(A_vx.dtype),
            crs=CRS.from_proj4(cube.ds.proj4),
            transform=cube.ds.rio.transform(),
    ) as dst:
        dst.write(A_vx, 1)

    with rasterio.open(
            f"{path_save}/{save_name}_amplitude_dy.tiff",
            "w",
            driver="GTiff",
            height=A_vy.shape[0],
            width=A_vy.shape[1],
            count=1,
            dtype=str(A_vy.dtype),
            crs=CRS.from_proj4(cube.ds.proj4),
            transform=cube.ds.rio.transform(),
    ) as dst:
        dst.write(A_vy, 1)

    with rasterio.open(
            f"{path_save}/{save_name}_amplitude_dd.tiff",
            "w",
            driver="GTiff",
            height=A_vv.shape[0],
            width=A_vv.shape[1],
            count=1,
            dtype=str(A_vv.dtype),
            crs=CRS.from_proj4(cube.ds.proj4),
            transform=cube.ds.rio.transform(),
    ) as dst:
        dst.write(A_vv, 1)