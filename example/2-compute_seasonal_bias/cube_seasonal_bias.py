from ticoi.cube_data_classxr import CubeDataClass
from ticoi.utils import find_granule_by_point
from SeBVel.seaso_amp import process,process_split
import geopandas as gpd
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import time
import os
from itertools import product
import numpy as np

os.system("export OMP_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export NUMBA_NUM_THREADS=4")


glacier = "Kask"
dataset = "ITS_LIVE"
WINDOW = 30
ROLLING_METHOD = "weighted_median"   # or "mean", "median"
MIN_POINTS = 5
BASELINE_THRESHOLD = 500
MAXSIZEGB = 5
NCORES = 4
PICK_SENSOR = ["Sentinel-2"]
VERBOSE = True

area_file = "/Users/charriel/Documents/Seasonality/Russel/alps_crop.gpkg"
path_save = "/Users/charriel/Documents/Seasonality/Alps/"

if dataset == "ITS_LIVE":
    if glacier== "Alps": cube_list = ["s3://its-live-data/datacubes/v2-updated-october2024/N40E000/ITS_LIVE_vel_EPSG32632_G0120_X350000_Y5050000.zarr"]
    else:
        mask = gpd.read_file(area_file)
        list_coordinates = [[mask.bounds.minx[0], mask.bounds.miny[0]], [mask.bounds.maxx[0], mask.bounds.miny[0]],
                            [mask.bounds.minx[0], mask.bounds.maxy[0]], [mask.bounds.maxx[0], mask.bounds.maxy[0]]]
        cube_list = np.unique([find_granule_by_point([k[0], k[1]]) for k in list_coordinates])
    path_save += "ITS_LIVE/"

elif dataset == "IGE_filtered":
    cube_IGE_path = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cubes_filtered"
    cube_list = glob.glob(cube_IGE_path + "/*.nc")
    path_save += "IGE_filtered/"

elif dataset == "IGE_non_filtered":
    cube_IGE_path = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cubes_S2_non_filter"
    cube_list = glob.glob(cube_IGE_path + "/*.nc")
    path_save += "IGE_non_filtered/"

print(cube_list)
catalog_cubeige = "/Users/charriel/Documents/Seasonality/Alps/Cubes_Alps/cube_grid.shp"


for cube_name in tqdm(cube_list):

    save_name = cube_name.split('/')[-1].split('.')[0]
    if VERBOSE: print(save_name)

    cube = CubeDataClass()
    cube.load(cube_name,pick_date = ["2016-01-01" ,"2025-12-31"] ,mask=area_file ,pick_temp_bas = [0 ,BASELINE_THRESHOLD],pick_sensor=PICK_SENSOR)
    if VERBOSE:
        print(f"Sensors list {np.unique(cube.sensor_())}")
        print(f"Cube dimensions {cube.nx, cube.ny}")


    size_gb = cube.ds.nbytes / (1024 ** 3)# Convert to GB
    if VERBOSE:print(f"{size_gb:.3f} GB")
    if size_gb < MAXSIZEGB:
        if VERBOSE:print("No need to split the process")
        if VERBOSE:
            print("Process start...")
            start_t = time.time()
        process(cube.ds,method_rolling=ROLLING_METHOD,min_points_rolling=MIN_POINTS,window=WINDOW,save_name=save_name,path_save=path_save,verbose=False)
        if VERBOSE:print(f"Process took {time.time() - start_t} seconds...")

    else:
        nb_split = int(size_gb / MAXSIZEGB)
        if VERBOSE:print(f"Not enough disk space, splitting the process. Spliting in {nb_split}")
        x_step = int(cube.nx / nb_split)
        y_step = int(cube.ny / nb_split)
        nx_split = int(np.sqrt(nb_split))
        ny_split = nx_split

        if VERBOSE:
            print("Process start...")
            start_t = time.time()

        Parallel(n_jobs=NCORES)(
            delayed(process_split)(cube,ix, iy,x_step,y_step,method_rolling=ROLLING_METHOD,min_points_rolling=MIN_POINTS,nx_split=nx_split,ny_split=ny_split,verbose=VERBOSE)
            for ix, iy in product(range(nx_split), range(ny_split))
        )
        if VERBOSE:print(f"Process took {time.time() - start_t} seconds...")
