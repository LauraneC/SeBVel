import time
import warnings

import datetime as dt
import matplotlib.pyplot as plt

from src.SErrVel.shadow import Shadow

warnings.filterwarnings('ignore')

# domain = {"lon_min": 6.835, "lon_max": 6.851,
#           "lat_min": 45.857, "lat_max": 45.867} # Domain boundaries [degree]
# domain = {"lon_min": 6.808, "lon_max": 6.828,
#           "lat_min": 45.834, "lat_max": 45.848} # Domain boundaries [degree]
domain = {"lon_min": 6.80, "lon_max": 6.83,
          "lat_min": 45.833, "lat_max": 45.859} # Domain boundaries [degree]
dist_search = 5.0  # Search distance for terrain shading [kilometre]
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# Paths and file names
repo = "/home/charriel/Documents/Scripts_dossier/shadow/"
file_dem = repo + "20241002_MdG_MtBlanc_DEM_coreg.tif"
file_ortho = repo + "20241002_MdG_ortho_0.5m_001_shift_H-V.tif"
# repo = "/home/lioretn/Documents/shadow_removal_v0/data/2018-09-08/"
# file_dem = repo + 'MONT-BLANC_2018-09-08_DEM_4m.tif'
# file_ortho = repo + 'MONT-BLANC_2018-09-08_ortho_0.5m_002_mapproj.tif'
# file_shadow_out = repo + "shadow_Mont-Blanc.tif"

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
# date = dt.datetime(2018, 9, 8, 10, 33, tzinfo=dt.timezone.utc)

# Parallelization
n_jobs = 12

start = time.time()

shadow = Shadow(file_dem=file_dem, settings=global_settings, domain=domain, dist_search=dist_search, ellps=ellps, verbose=True)
# shadow.load_ortho(file_ortho)
shadow.cast_shadow(date, preprocess=True,
                   pipeline=[],
                   plot_along_pipeline=True)

shadow.plot_shadow(background='dem')
plt.show()
# shadow.plot_shadow()
# plt.show()
# exit()

# shadow.plot_ortho()
# shadow.plot_shadow(background='ortho')

# shadow.plot_ortho()
# shadow.cast_shadow(date, preprocess=True)
# shadow.plot_shadows()
# plt.show()

# shadow.improve_shadow(pipeline=[(shadow.regional_flood, {"parallelize": True}, "Regionaly flooded shadows")],
#                                 #(shadow.refine_contours, {"parallelize": 12, "contours_buffer_size": 15, "win_size": 150}, "Contour refined shadows")],
#                       plot_along_pipeline=True)
# plt.show()

#%%
# shadow.remove_shadow_block(parallelize=False, contours_method="nb_pixel")

# #%%
# # shadow.plot_shadow_contours(background=['shadow', 'ortho'])
# shadow.plot_shadow_free()

# plt.show()

#%%
# shadow.interpolation_in_contours(interpolation_method='linear')
# shadow.plot_shadow_free()

#%%
# shadow.gaussian_filtering_contours(first_ortho_sigma=3, second_ortho_buffer=2, second_ortho_sigma=2)

# shadow.plot_ortho(savefig='Bionassay_ortho.png')
# shadow.plot_shadow(savefig='Bionassay_shadow.png')
# shadow.plot_shadow_free(savefig='Bionassay_shadow-free.png')

# fig, ax = plt.subplots(nrows=2, figsize=(20, 12))
# shadow.plot_grad_ortho(ax=ax[0])
# shadow.plot_grad_shadow_free(ax=ax[1])

print(f"Overall processing took {round(time.time() - start, 0)} s")

plt.show()