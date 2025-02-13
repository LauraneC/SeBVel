import time
import warnings
from core import splitted_process

warnings.filterwarnings('ignore')

### PARAMATERS

# Paths and file names
repo = "/home/charriel/Documents/Seasonality/shadow_map/" # Main repository
file_dem = "/home/charriel/Documents/Seasonality/metadata/DEM/Copernicus10m/DEM_Copernicus10m_MontBlanc.tif" # Rough DEM to degrade the precision of the DEM on glacier surfaces (limit crevasses impact)
file_ortho = repo + "20241002_MdG_ortho_0.5m_001_shift_H-V.tif" # Orthoimage (for illustration only when computing shadow maps)
file_rgi = "/home/charriel/Documents/Seasonality/shadow_map/RGI60_MtBlanc/RGI60_MtBlanc_UTM32N.shp" # RGI file (glaciers inventory)
file_rough_dem = "/home/charriel/Documents/Seasonality/metadata/DEM/Copernicus30m/DEM_30m_cropped.tif" # Rough DEM to degrade the precision of the DEM on glacier surfaces (limit crevasses impact)
path_save = "/home/charriel/Documents/Seasonality/shadow_map/shadow_maps/"

# Boundary parameters
domain = None

# Settings
filter_small_shadows = False # Remove shadows with less than SMALL_SHADOW_LIMIT pixels (in global_settings)
global_settings = {"SMALL_SHADOW_LIMIT": 500} # Global settings (only SMALL_SHADOW_LIMIT is used here)
nb_cpus = 8

# Acquisition date (/!\ might be different from data submission date)
year = 2024
hour = "10:30" # Same hour each day

#nb of subsets used to split the process
nb_split = 5

make_plot=False #make plot
show=False#display it
save=True#save plots and geotiff files

### MAIN

start_time = time.time()
splitted_process(file_dem,
                 file_ortho,file_rgi,rough_dem=file_rough_dem,year=year,hour=hour,nb_split=nb_split,global_settings=global_settings ,domain=domain,
                 filter_small_shadows=filter_small_shadows,contours = 160,nb_cpus=nb_cpus,path_save=path_save,save=save,make_plot=make_plot,show=show,verbose=False)
print(f"Overall processing took {round(time.time() - start_time, 2)} s")
