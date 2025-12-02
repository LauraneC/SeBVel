import glob
import os

import datetime as dt
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np

from src.SErrVel.shadow import Shadow
from rasterio.merge import merge
from pyproj import Transformer


def split_domain(file_dem:str, nb_split:int, iteration:int, domain:dict|None=None):
    """
    Get the spatial coordinates of the spatial domain for a given block during the iteratition process
    :param file_dem: filename of the DEM
    :param nb_split: number of blocks to split the domain into
    :param iteration: number of iteration at which the process is
    :param domain: already loaded spatial domain
    :return:
    """
    with rio.open(file_dem) as src:  # get the boundary of the DEM
        if domain is None:
            domain = {"lon_min": src.bounds.left, "lon_max": src.bounds.right,
                      "lat_min": src.bounds.bottom, "lat_max": src.bounds.top}
        else:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            (domain["lon_min"], domain["lat_min"]) = transformer.transform(domain["lon_min"], domain["lat_min"])
            (domain["lon_max"], domain["lat_max"]) = transformer.transform(domain["lon_max"], domain["lat_max"])          

        if abs(domain["lon_max"] - domain["lon_min"]) > abs(domain["lat_max"] - domain["lat_min"]):
            width_split = (domain["lon_max"] - domain["lon_min"]) / nb_split
            domain["lon_min"] = domain["lon_min"] + iteration * width_split
            domain["lon_max"] = domain["lon_min"] + (iteration + 1) * width_split
        else:
            width_split = (domain["lat_max"] - domain["lat_min"]) / nb_split
            domain["lat_min"] = domain["lat_min"] + iteration * width_split
            domain["lat_max"] = domain["lat_min"] + (iteration + 1) * width_split
            
        if src.crs != "EPSG:4326":
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            (domain["lon_min"], domain["lat_min"]) = transformer.transform(domain["lon_min"], domain["lat_min"])
            (domain["lon_max"], domain["lat_max"]) = transformer.transform(domain["lon_max"], domain["lat_max"])

    return domain


def merge_geotiff(files:str, output_path:str):
    """
    Merge geotiff files into a single file
    :param files: path of the files to merge
    :param output_path: name of the output file
    """
    tiff_files = glob.glob(files)
    # Open all GeoTIFFs
    src_files_to_mosaic = [rio.open(fp) for fp in tiff_files]

    # Merge the rasters
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Save the merged raster
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    with rio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all input files
    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged GeoTIFF saved to {output_path}")


def splitted_process(file_dem: str | None = None,
                     file_ortho: str | None = None, file_rgi: str | None = None, rough_dem: str | None = None,
                     year: int = 2021, hour: str = "10:30",time_span:int=10, nb_split: int = 5, global_settings: dict = {},
                     dist_search=1.0, ellps: str = "WGS84", domain: dict | None = None,
                     filter_small_shadows: bool = False, contours: int = 160, nb_cpus: int = 8,
                     path_save: str | None = None, save: bool = True, make_plot: bool = True, show: bool = False,
                     verbose: bool = False):
    # Define the dates when you want to cast shadows (here every 10 days)
    dates = [dt.datetime(year, 1, 1, int(hour.split(':')[0]), int(hour.split(':')[1]),
                         tzinfo=dt.timezone.utc) + dt.timedelta(ndays) for ndays in
             range(time_span , 366 if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0) else 365,time_span )]

    for iteration_split in range(nb_split):
        if verbose: print(f"Split iteration {iteration_split}")
        domain = split_domain(file_dem, nb_split,
                              iteration_split)  # split the domain in nb_splits subset, and select the subset iteration_split
        shadow = Shadow(file_dem=file_dem, file_ortho=file_ortho, settings=global_settings, domain=domain,
                        dist_search=dist_search, ellps=ellps, verbose=verbose,
                        file_rgi=file_rgi,
                        rough_dem=rough_dem)  # initialize the object shadow with global settings, domain (=the subset), ellps ( Earth's surface approximation)
        # if file_rgi, rough_dem are not None, the DEM is replaced by rough_dem over glaciers

        # Compute the number of days covered by shadows
        shadow_map = shadow.nday_shadow_map(dates, parallelize=nb_cpus, contours=0, preprocess=filter_small_shadows)
        shadow_map = (shadow_map * (365 / (len(dates) + 1))).astype(np.uint16)  # Convert it to a 365 days count
        if save: shadow.write_shadow_map(shadow_map,
                                         output_file=f'{path_save}/shadow_map{iteration_split}.tif')  # save it as geotiff
        # Plot the shadow map
        if make_plot:  # make the plot
            save_fig = f'{path_save}/shadow_map{iteration_split}.png' if save else None
            shadow.plot_shadow_map(shadow_map, background='ortho', plot_mode='imshow', alpha=0.5,
                                   cbar_label="Nb days under shadows",
                                   savefig=save_fig)
        if show: plt.show()  # display it

        # Compute the numbers of days where each area is included in the border of the shadow
        shadow_map = shadow.nday_shadow_map(dates, parallelize=8, contours=contours, preprocess=filter_small_shadows)
        shadow_map = (shadow_map * (365 / (len(dates) + 1))).astype(np.uint16)  # Convert it to a 365 days count
        if save: shadow.write_shadow_map(shadow_map,
                                         output_file=f'{path_save}/shadow_map_border{iteration_split}.tif')
        if make_plot:
            save_fig = f'{path_save}/shadow_borders_map{iteration_split}.png' if save else None
            shadow.plot_shadow_map(shadow_map, background='ortho', plot_mode='imshow', alpha=0.5,
                                   cbar_label="Nb days under shadow borders",
                                   savefig=save_fig)
        if show: plt.show()
        del shadow_map, shadow, domain

    if verbose: print(f"Process of {nb_split} iterations complete")

    merge_geotiff(f'{path_save}shadow_map_border*tif', f'/{path_save}/shadow_map_border_merged.tif')
    merge_geotiff(f'{path_save}/shadow_map*tif', f'{path_save}/shadow_map_merged.tif')

    if verbose: print(f"Geotiff merged")