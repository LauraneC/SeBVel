import time
import warnings

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from shadow import Shadow

warnings.filterwarnings('ignore')

# Paths and file names
repo = "/home/charriel/Documents/Seasonality/shadow_map/" # Main repository
file_dem = repo + "20241002_MdG_MtBlanc_DEM_coreg.tif" # Main DEM
file_ortho = repo + "20241002_MdG_ortho_0.5m_001_shift_H-V.tif" # Orthoimage (for illustration only when computing shadow maps)
file_rgi = "/home/charriel/Documents/Seasonality/shadow_map/RGI60_MtBlanc/RGI60_MtBlanc_UTM32N.shp" # RGI file (glaciers inventory)
file_rough_dem = "/home/charriel/Documents/Form@ter/Metadata/DEM/30m/output_COP30.tif" # Rough DEM to degrade the precision of the DEM on glacier surfaces (limit crevasses impact)

# Boundary parameters
domain = None
domain = {"lon_min": 6.825, "lon_max": 6.915,
          "lat_min": 45.825, "lat_max": 45.895} # Domain boundaries [degree]
dist_search = 1.0  # Search distance for terrain shading [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Settings
filter_small_shadows = False # Remove shadows with less than SMALL_SHADOW_LIMIT pixels (in global_settings)
global_settings = {"SMALL_SHADOW_LIMIT": 500} # Global settings (only SMALL_SHADOW_LIMIT is used here)

# Acquisition date (/!\ might be different from data submission date)
year = 2024
hour = "10:30" # Same hour each day

start_time = time.time()

# Example 1 - Casting a shadow
date = dt.datetime(2024, 10, 2, 10, 30, tzinfo=dt.timezone.utc) # Define the date
# Create shadow object and load DEM and orthoimage in the given domain. If specified, file_rgi and rough_dem allow to degrade DEM's quality in glacier areas to
# limit the impact of crevasses on the shadow maps (put None if you don't want to degrade it)
shadow = Shadow(file_dem=file_dem, file_ortho=file_ortho, settings=global_settings, domain=domain, dist_search=dist_search, ellps=ellps, verbose=True,
                file_rgi=file_rgi, rough_dem=file_rough_dem)
# shadow.cast_shadow(date, preprocess=filter_small_shadows, clean=False) # Cast the shadows
# shadow.plot_shadow(background='ortho', plot_mode='imshow') # Plot the shadows above the orthoimage using plt.imshow
# plt.show()

# Example 2 - Generating a shadow map
# Define the dates when you want to cast shadows (here every 10 days)
dates = [dt.datetime(year, 1, 1, int(hour.split(':')[0]), int(hour.split(':')[1]), tzinfo=dt.timezone.utc) + dt.timedelta(ndays) for ndays in
                range(10, 366 if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0) else 365, 10)]

# Compute shadow map on those dates
# parallelize allows parallelization, contours=0 implies that we compute for each pixel the number of days when it is shadowed (put contours={int}
# to compute the it around the contours only)
shadow_map = shadow.nday_shadow_map(dates, parallelize=8, contours=0, preprocess=filter_small_shadows)
shadow_map = (shadow_map * 365 / (len(dates)+1)).astype(np.uint16) # Convert it to a 365 days count

# Plot the shadow map
shadow.plot_shadow_map(shadow_map, background='ortho', plot_mode='imshow', alpha=0.5, cbar_label="Nb days under shadows",savefig=f'{repo}/shadow_map.png')
plt.show()

print(f"Overall processing took {round(time.time() - start_time, 2)} s")

shadow_map = shadow.nday_shadow_map(dates, parallelize=8, contours=160, preprocess=filter_small_shadows)
shadow_map = (shadow_map * 365 / (len(dates)+1)).astype(np.uint16) # Convert it to a 365 days count
shadow.plot_shadow_map(shadow_map, background='ortho', plot_mode='imshow', alpha=0.5, cbar_label="Nb days under shadow borders",savefig=f'{repo}/shadow_borders_map.png')

plt.show()