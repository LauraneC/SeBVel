# authors: L. Charrier & N. Lioret
# if you use this code, please cite Steger, C. R., Steger, B., & Schär, C. (2022). HORAYZON v1. 2: an efficient and flexible ray-tracing algorithm to compute horizon and sky view factor. Geoscientific Model Development, 15(17), 6817-6840.
#and L.Charrier Error in Seasonal velocity (article soon submited)

import time
import warnings

import datetime as dt
import matplotlib.pyplot as plt

from src.SErrVel.shadow import Shadow

warnings.filterwarnings('ignore')


domain = {"lon_min": 6.80, "lon_max": 6.81,
          "lat_min": 45.833, "lat_max": 45.849} # Domain boundaries [degree]
dist_search = 5.0  # Search distance for terrain shading [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Paths and file names
repo = "/home/charriel/Documents//"
file_dem = "/home/charrierl/Documents/Data/DEMs/Copernicus30m_Seaso/DEM_30m.tif" # Rough DEM to degrade the precision of the DEM on glacier surfaces (limit crevasses impact)
# file_ortho = repo + "20241002_MdG_ortho_0.5m_001_shift_H-V.tif" #to plot shadow over an ortho-rectified image

# Settings
global_settings = {"SMALL_SHADOW_LIMIT": 10,
                   "SHADOW_MAX_VALUE": 50,
                   "LIGHT_SHADOW_MIN_DIFF": 10,
                   "WEAKLY_LINKED_SHADOW_DISK_RADIUS": 5,
                   "SUN_WIDTH": 0.5,
                   "DEM_RESOLUTION": 4,
                   "ORTHO_RESOLUTION": 0.5}

# Acquisition date (/!\ might be different from data submission date)
date = dt.datetime(2024, 10, 2, 10, 38, tzinfo=dt.timezone.utc)

# Parallelization
n_jobs = 12

start = time.time()

shadow = Shadow(file_dem=file_dem, settings=global_settings, domain=domain, dist_search=dist_search, ellps=ellps, verbose=True)
# shadow.load_ortho(file_ortho) #to plot shadow over an ortho-rectified image
shadow.cast_shadow(date, preprocess=True,
                   pipeline=[],
                   plot_along_pipeline=True)

shadow.plot_shadow(background='dem')
plt.show()