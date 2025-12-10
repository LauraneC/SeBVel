#!/usr/bin/env python3

"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method to compute entire Data cubes.
An additional seasonality analysis is implemented, by matching a sinus to TICOI results for each pixel of the considered cube/subset,
thus generating maps with the amplitude of the best matching sinus, the position of its first maximum and an index comparing its amplitude
to the local variations of the raw Data.s

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from osgeo import gdal, osr
from tqdm import tqdm

from ticoi.core import process, process_blocks_refine, save_cube_parameters
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.seasonality_functions import match_sine, AtoVar
import glob
import nest_asyncio

nest_asyncio.apply()

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------- Choose TICOI cube processing method ----------------- ##
# Choose the TICOI cube processing method you want to use :
#    - 'block_process' (recommended) : This implementation divides the Data in smaller Data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than every other TICOI processing methods.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal,
# there should be no problem, but if you try to launch it from some IDE (like Spyder), think of specifying to your IDE to launch it
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the Data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of Data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.
#   - 'load' : The  TICOI cube was already calculated before, load it by giving the cubes to be loaded in a dictionary like {name: path} (at least
# 'raw' and 'interp' must be given)

TICOI_process = "load"

save = True  # If True, save TICOI results to a netCDF file
## ------------------------------ Data selection --------------------------- ##

cube_folder = '/home/charriel/Documents/PostdocIGE/Seasonality/Cubes_Alps/cubes_S2_non_filter/'# Path where to store the results
path_save = f'/home/charriel/Documents/Seasonality/'  # Path where to store the results
cube_name_list = glob.glob(f'{cube_folder}/*.nc')

for cube_name_el in cube_name_list:
    print(cube_name_el)
    cube_name = {
        "raw": cube_name_el}
    flag_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_flags.nc'  # Path to flags file
    # mask_file = f'/workdir2/c2h/charriel/Data/RGI60_MtBlanc/RGI60_MtBlanc_UTM32N.shp'
    mask_file = None
    result_fn = f"{cube_name_el.split('/')[-1].split('.nc')[0]}"  # Name of the netCDF file to be created (if save is True)

    proj = "EPSG:32632"  # EPSG system of the given coordinates

    # Divide the Data in several areas where different methods should be used
    assign_flag = False
    if not assign_flag:
        flag_file = None

    # Regularization method.s to be used (for each flag if flag is not None)
    # regu = {0: 1, 1: "1accelnotnull"}  # With flag (0: stable ground, 1: glaciers)
    regu = "1accelnotnull"
    # regu = '1accelnotnull' # Without flag
    # Regularization coefficient.s to be used (for each flag if flag is not None)
    # coef = {0: 500, 1: 200}  # With flag (0: stable ground, 1: glaciers)
    coef = 500
    # coef = 200 # Without flag
    solver = "LSMR_ini"  # Solver for the inversion
    delete_outliers = {
        "mz_score": 3.5
    }
    # subset = [330768.1836587775, 334412.3736587863, 5083194.342253393, 5079918.252253384]
    subset = None

    # subset = None
    ## ---------------------------- Loading parameters ------------------------- ##
    load_kwargs = {
        "chunks": {},
        "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
        "subset": subset,  # Subset of the Data to be loaded ([xmin, xmax, ymin, ymax] or None)
        "buffer": None,  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
        "pick_date": None,  # Select dates ([min, max] or None to select all)
        "pick_sensor": None,  # Select sensors (None to select all)
        "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
        "proj": proj,  # EPSG system of the given coordinates
        "mask": mask_file,  # Path to mask file (.shp file) to mask some of the Data on cube
        "verbose": False,  # Print information throughout the loading process
    }

    ## ----------------------- Data preparation parameters --------------------- ##
    preData_kwargs = {
        "smooth_method": "savgol",
        # Smoothing method to be used to smooth the Data in time ('gaussian', 'median', 'emwa', 'savgol')
        "s_win": 3,  # Size of the spatial window
        "t_win": 90,  # Time window size for 'ewma' smoothing
        "sigma": 3,  # Standard deviation for 'gaussian' filter
        "order": 3,  # Order of the smoothing function
        "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
        "delete_outliers": delete_outliers,
        # Delete Data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
        "flag": flag_file,  # Divide the Data in several areas where different methods should be used
        "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None)
        "solver": solver,  # Solver for the inversion
        "proj": proj,  # EPSG system of the given coordinates
        "velo_or_disp": "velo",
        # Type of Data contained in the Data cube ('disp' for displacements, and 'velo' for velocities)
        "verbose": True,  # Print information throughout the filtering process
    }

    ## ---------------- Inversion and interpolation parameters ----------------- ##
    inversion_kwargs = {
        "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None)
        "coef": coef,  # Regularization coefficient.s to be used (for each flag if flag is not None)
        "solver": solver,  # Solver for the inversion
        "flag": flag_file,  # Divide the Data in several areas where different methods should be used
        "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
        "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
        "proj": proj,  # EPSG system of the given coordinates
        "interpolation_load_pixel": "nearest",
        # Interpolation method used to load the pixel when it is not in the dataset
        "iteration": True,  # Allow the inversion process to make several iterations
        "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
        "threshold_it": 0.1,
        # Threshold to test the stability of the results between each iteration, used to stop the process
        "apriori_weight": False,  # If True, use apriori weights
        "detect_temporal_decorrelation": True,
        # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
        "linear_operator": None,  # Perform the inversion using this specific linear operator
        "interval_output": 30,
        "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
        "redundancy": 30,  # Redundancy in the interpolated time series in number of days, no redundancy if None
        "result_quality": "X_contribution",
        # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
        "visual": False,  # Plot results along the way
        "path_save": path_save,  # Path where to store the results
        "verbose": False,  # Print information throughout TICOI processing
    }

    smooth_res = False  # Smooth TICOI results (to limit the noise)
    smooth_window_size = 3  # Size of the window for the average filter used to smooth the cube
    smooth_filt = (
        None  # Specify here the filter you want to use to smooth the cube (if None, an average filter will be used)
    )

    ## ----------------------- Parallelization parameters ---------------------- ##
    nb_cpu = 12  # Number of CPU to be used for parallelization
    block_size = 0.1  # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

    ## ------------------- Parameters for seasonality analysis ----------------- ##
    # Is the periodicity frequency imposed to 1/365.25 (one year seasonality) ?
    impose_frequency = True
    # Add several sinus at different freqs (1/365.25 and harmonics (2/365.25, 3/365.25...) if impose_frequency is True)
    #   (only available for impose_frequency = True for now)
    several_freq = 1
    # Compute also the best matching sinus to raw Data, for comparison
    raw_seasonality = True
    # Filter to use in the first place
    # 'highpass' : apply a bandpass filter between low frequencies (reject variations over several years (> 1.5 y))
    # and the Nyquist frequency to ensure Shanon theorem
    # 'lowpass' : or apply a lowpass filter only (to Nyquist frequency) : risk of tackling an interannual trend (long period)
    filt = None
    # Method used to compute local variations
    # 'rolling_7d' : median of the std of the Data centered in +- 3 days around each central date
    # 'uniform_7d' : median of the std of the Data centered in +- 3 days around dates constantly distributed every redundnacy
    # days -- BEST
    # 'uniform_all' : median of the std of each Data covering the dates, which are constantly distributed every redundancy days
    # 'residu' : standard deviation of the Data previously subtracted by TICOI results (ground truth) = standard deviation of the "noise"
    local_var_method = "uniform_7d"
    variable_list = ['vx', 'vy']
    # variable_list = ['direction']

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    # %%========================================================================= #
    #                                 DATA LOADING                                #
    # =========================================================================%% #

    start, stop = [], []
    start.append(time.time())

    # Load the cube.s
    cube = CubeDataClass()
    cube.load(cube_name if TICOI_process != "load" else cube_name["raw"], **load_kwargs)

    # Load raw Data at pixels if required
    print("[Data loading] Loading raw Data...")
    data_raw = process_blocks_refine(
        cube, nb_cpu=nb_cpu, block_size=block_size, returned=["raw"], inversion_kwargs=inversion_kwargs
    )
    data_raw = [
        pd.DataFrame(
            data={
                "date1": raw[0][0][:, 0],
                "date2": raw[0][0][:, 1],
                "vx": raw[0][1][:, 0],
                "vy": raw[0][1][:, 1],
                "errorx": raw[0][1][:, 2],
                "errory": raw[0][1][:, 3],
                "temporal_baseline": raw[0][1][:, 4],
            }
        )
        for raw in data_raw
    ]

    # Prepare interpolation dates
    first_date_interpol, last_date_interpol = cube.prepare_interpolation_date()
    inversion_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})

    stop.append(time.time())
    print(f"[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ")
    print(f"[Data loading] Data loading took {round(stop[-1] - start[-1], 3)} s")

    # %%========================================================================= #
    #                               PERIODICITY MAPS                              #
    # =========================================================================%% #
    # Match a sinus to the Data (frequency which can be fixed to 1/365.25, amplitude, phase which gives the date of the f)

    start.append(time.time())

    driver = gdal.GetDriverByName("GTiff")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(proj.split(":")[1]))

    if type(cube_name) == dict and 'interp' not in cube_name.keys():
        result = data_raw
        raw_seasonality = False

    # Remove pixels with no Data
    empty = list(
        filter(
            bool,
            [
                d if not (result[d].empty and result[d][result[d]["vx"] == 0].shape[0] == 0) else False
                for d in range(len(result))
            ],
        )
    )
    positions = np.array(list(itertools.product(cube.ds["x"].values, cube.ds["y"].values)))[empty, :]
    useful_result = [result[i] for i in empty]
    useful_data_raw = [data_raw[i] for i in empty]

    # Coordinates information
    resolution = int(cube.ds["x"].values[1] - cube.ds["x"].values[0])
    long_data = (positions[:, 0] - np.min(cube.ds["x"].values)).astype(int) // resolution
    lat_data = (positions[:, 1] - np.min(cube.ds["y"].values)).astype(int) // resolution

    # Format raw displacements to velocities
    for raw in data_raw:
        raw["vx"] = raw["vx"] * preData_kwargs["unit"] / raw["temporal_baseline"]
        raw["vy"] = raw["vy"] * preData_kwargs["unit"] / raw["temporal_baseline"]
        raw["vv"] = np.sqrt(raw["vx"] ** 2 + raw["vy"] ** 2)
        raw.index = raw["date1"] + (raw["date2"] - raw["date1"]) // 2

    ##  Best matching sinus map (amplitude and phase, and period if not fixed)
    print("[Fourier analysis] Computing periodicity maps...")
    if not impose_frequency:
        period_map = np.empty([cube.nx, cube.ny])
        period_map[:, :] = np.nan
    amplitude_map = np.empty([cube.nx, cube.ny])
    amplitude_map[:, :] = np.nan
    AtoVar_map = np.empty([cube.nx, cube.ny])
    AtoVar_map[:, :] = np.nan
    peak_map = np.empty([cube.nx, cube.ny])
    peak_map[:, :] = np.nan
    if raw_seasonality:
        amplitude_raw_map = np.empty([cube.nx, cube.ny])
        amplitude_raw_map[:, :] = np.nan
        peak_raw_map = np.empty([cube.nx, cube.ny])
        peak_raw_map[:, :] = np.nan

    for variable in variable_list:
        result_tqdm = tqdm(zip(useful_result, useful_data_raw), total=len(useful_result), mininterval=0.5)
        match_res = np.array(
            Parallel(n_jobs=nb_cpu, verbose=0)(
                delayed(match_sine)(d, filt=filt, impose_frequency=impose_frequency, raw_seasonality=raw_seasonality,
                                    d_raw=raw, variable=variable)
                for d, raw in result_tqdm
            )
        )
        if not impose_frequency:
            period = np.abs(match_res[:, 0])
            period_map[long_data, lat_data] = np.sign(period - 365) * (
                    1 - np.minimum(period, 365) / np.maximum(period, 365))
        amplitude_map[long_data, lat_data] = np.abs(match_res[:, 1])
        peak_map[long_data, lat_data] = match_res[:, 2]
        raw_tqdm = tqdm(zip(match_res[:, 1], useful_data_raw, useful_result), total=len(useful_data_raw),
                        mininterval=0.5)
        AtoVar_map[long_data, lat_data] = Parallel(n_jobs=nb_cpu, verbose=0)(
            delayed(AtoVar)(A, raw, dataf_lp, local_var_method) for A, raw, dataf_lp in raw_tqdm
        )
        if raw_seasonality:
            amplitude_raw_map[long_data, lat_data] = np.abs(match_res[:, 3])
            peak_raw_map[long_data, lat_data] = match_res[:, 4]

        # Save the maps to a .tiff file with two bands (one for period, and one for amplitude)
        if len(variable_list) > 1 and variable != variable_list[0]:
            driver = gdal.GetDriverByName("GTiff")
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(proj.split(":")[1]))

        if impose_frequency:
            tiff = driver.Create(
                f"{path_save}/{list(cube_name.values())[0].split('/')[-1]}matching_sine_map_{filt}_{local_var_method}_{variable}.tiff",
                amplitude_map.shape[0],
                amplitude_map.shape[1],
                3 if not raw_seasonality else 5,
                gdal.GDT_Float32,
            )
            tiff.SetGeoTransform(
                [np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
            # Add descriptions and write Data for all 3 bands
            band1 = tiff.GetRasterBand(1)
            band1.SetDescription("Amplitude Map")  # Set description for band 1
            band1.WriteArray(np.flip(amplitude_map.T, axis=0))  # Write amplitude map Data to band 1

            band2 = tiff.GetRasterBand(2)
            band2.SetDescription("Peak Map")  # Set description for band 2
            band2.WriteArray(np.flip(peak_map.T, axis=0))  # Write peak map Data to band 2

            band3 = tiff.GetRasterBand(3)
            band3.SetDescription("AtoVar Map")  # Set description for band 3
            band3.WriteArray(np.flip(AtoVar_map.T, axis=0))  # Write A-to-Var map Data to band 3

            if raw_seasonality:
                tiff.GetRasterBand(4).WriteArray(np.flip(amplitude_raw_map.T, axis=0))
                tiff.GetRasterBand(5).WriteArray(np.flip(peak_raw_map.T, axis=0))
        else:
            tiff = driver.Create(
                f"{path_save}/{cube_name.split('/')[-1]}matching_sine_map_{filt}_{local_var_method}_{variable}.tiff",
                period_map.shape[0],
                period_map.shape[1],
                4,
                gdal.GDT_Float32,
            )
            # Add descriptions and write Data for all 4 bands
            band1 = tiff.GetRasterBand(1)
            band1.SetDescription("Period Map")  # Set description for band 1
            band1.WriteArray(np.flip(period_map.T, axis=0))  # Write period map Data to band 1

            band2 = tiff.GetRasterBand(2)
            band2.SetDescription("Amplitude Map")  # Set description for band 2
            band2.WriteArray(np.flip(amplitude_map.T, axis=0))  # Write amplitude map Data to band 2

            band3 = tiff.GetRasterBand(3)
            band3.SetDescription("Peak Map")  # Set description for band 3
            band3.WriteArray(np.flip(peak_map.T, axis=0))  # Write peak map Data to band 3

            band4 = tiff.GetRasterBand(4)
            band4.SetDescription("AtoVar Map")  # Set description for band 4
            band4.WriteArray(np.flip(AtoVar_map.T, axis=0))  # Write A-to-Var map Data to band 4

        tiff.SetProjection(srs.ExportToWkt())

        # Needed to effectively save the .tiff file
        tiff = None
        driver = None

        stop.append(time.time())
        del band1, band2, band3
        if raw_seasonality: del amplitude_raw_map, peak_raw_map

    print(f"[Fourier analysis] Computing periodicity maps took {round(stop[-1] - start[-1], 0)} s")
    print(f"[Overall] Overall processing took {round(stop[-1] - start[0], 0)} s")
