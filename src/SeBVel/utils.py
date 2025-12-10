import os
import xdem

import numpy as np
import geopandas as gpd
import rasterio as rio
import rasterio.mask as rio_mask
from rasterio.merge import merge

from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from scipy.ndimage import sobel
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.filters.rank import gradient
from joblib import dump, load


def reproject_rio(file: str, 
                  dst_crs: str = 'EPSG:4326') -> str:
    
    '''
    Reproject an image (stored in {file}) to a a CRS {dst_crs} using RasterIO and store the result in a file with extension .{ext}
    
    Parameters
    ----------
      --> file [str]: Path to the file containing the image to reproject
      --> dst_crs [str] (optional): CRS to which the image must be reprojected, default is 'EPSG:4326'

    Returns
    -------
      --> file [str]: Path to the file containing the reprojected image
    '''

    with rio.open(file) as src:
        if src.crs == dst_crs:
            return file

        # Image properties
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        ext = file.split('.')[-1]
        file = file.split(f".{ext}")[0] + f"_{dst_crs.split(':')[0]}{dst_crs.split(':')[1]}.{ext}"
        with rio.open(file, 'w', **kwargs) as dst:
            # Reproject the image and save the result
            reproject(
                source=rio.band(src, 1),
                destination=rio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
        
        return file

def load_domain(file: str, 
                domain_outer: dict, 
                use_xdem: bool = False, 
                reprojection: str | None = None,
                mask: str | None = None,
                verbose: bool = False) -> list[np.array, np.array, np.ndarray]:
    
    '''
    Load the image contained in {file} and crop it to {domain_outer}, eventually padding it if the required domain goes beyond the borders of the image
    
    Parameters
    ----------
      --> file: Path to the file containing the image to load
      --> domain_outer [dict]: Dictionnary of type {'lon_min': .., 'lon_max': ..,
                                                    'lat_min': .., 'lat_max': ..} specifying the borders of the domain to be loaded within the image
      --> use_xdem [bool] (optional): To load a DEM, one should prefer using xdem instead of rasterio, put use_xdem=True if so, default is False
      --> reprojection [str | None] (optional): If [str], reproject the image to the given CRS before loading it, default is None
      --> mask [str] (optional): If [str], mask the image with the given mask before loading
      --> verbose [bool] (optional): Print information along the way, default is False

    Returns
    -------
      --> lon [np.array]: Longitudinal position of the pixels within the image {img}
      --> lat [np.array]: Latitudinal position of the pixels within the image {img}
      --> img [np.ndarray]: 2-dimensional Numpy array containing the croped (and potentially padded) image
    '''

    if use_xdem: # To load a DEM only (better)
        dem = xdem.DEM(file)
        if reprojection is not None:
            if verbose: print(f"Reprojecting DEM to {reprojection}...")
            dem = dem.reproject(crs=reprojection)
            ext = file.split('.')[-1]
            file = file.split(f'.{ext}')[0] + f"_{reprojection.split(':')[0]}{reprojection.split(':')[1]}.{ext}"
            dem.save(file)
        raster_size_y, raster_size_x = dem.data.shape
        lon_ulc, lat_ulc = dem.transform[2], dem.transform[5]
        d_lon, d_lat = dem.transform[0], dem.transform[4]
        img = dem.data.astype(np.float32)

    else: # To load images
        if reprojection is not None: # Reproject the image to the given CRS
            if verbose: print(f"Reprojecting image to {reprojection}...")
            file = reproject_rio(file)

        # Load the image and retrieve informations (longitude, latitude, image size and data)
        if verbose: print("Loading image...")
        with rio.open(file) as src:
            raster_size_x, raster_size_y = src.width, src.height
            bounds = src.bounds
            lon_ulc, lat_ulc = bounds.left, bounds.top
            d_lon, d_lat = (bounds.right - bounds.left) / raster_size_x, (bounds.bottom - bounds.top) / raster_size_y
            img = src.read(1)
            del bounds

    # Create a mask from a shapefile
    if mask is not None:
        with rio.open(file, 'r') as src:
          mask = gpd.read_file(mask).to_crs(src.crs)['geometry']
          mask = (rio_mask.mask(src, list(mask), crop=False)[0][0,:,:] > 0).astype(np.bool_)

    # Padding when domain_outer is not completly in the area covered by the image
    diff_lon = [lon_ulc - domain_outer['lon_min'], domain_outer['lon_max'] - (lon_ulc + d_lon * raster_size_x)]
    pad_lon = [int(max(diff_lon[0] / d_lon, 0)), int(max(diff_lon[1] / d_lon, 0))]
    diff_lat = [lat_ulc - domain_outer['lat_max'], domain_outer['lat_min'] - (lat_ulc + d_lat * raster_size_y)]
    pad_lat = [int(max(diff_lat[0] / d_lat, 0)), int(max(diff_lat[1] / d_lat, 0))]

    # Compute latitude and longitude of original DEM image
    img = np.pad(img, ((pad_lat[0]+1, pad_lat[1]+1), (pad_lon[0]+1, pad_lon[1]+1)), mode='constant', constant_values=-1)
    if mask is not None: mask = np.pad(mask, ((pad_lat[0]+1, pad_lat[1]+1), (pad_lon[0]+1, pad_lon[1]+1)), mode='constant', constant_values=0)
    lon_edge = np.linspace(lon_ulc - d_lon * (pad_lon[0] + 1), lon_ulc + d_lon * (raster_size_x + pad_lon[1] + 1), img.shape[1] + 1)
    lat_edge = np.linspace(lat_ulc - d_lat * (pad_lat[0] + 1), lat_ulc + d_lat * (raster_size_y + pad_lat[1] + 1), img.shape[0] + 1)
    lon = lon_edge[:-1] + np.diff(lon_edge)/2.0
    lat = lat_edge[:-1] + np.diff(lat_edge)/2.0

    # Crop relevant domain
    if any([domain_outer["lon_min"] < lon_edge.min(),
            domain_outer["lon_max"] > lon_edge.max(),
            domain_outer["lat_min"] < lat_edge.min(),
            domain_outer["lat_max"] > lat_edge.max()]):
        raise ValueError("Provided tile does not cover domain")
    slice_lon = slice(np.where(lon_edge <= domain_outer["lon_min"])[0][-1],
                      np.where(lon_edge >= domain_outer["lon_max"])[0][0])
    slice_lat = slice(np.where(lat_edge >= domain_outer["lat_max"])[0][-1],
                      np.where(lat_edge <= domain_outer["lat_min"])[0][0])

    img = img[slice_lat, slice_lon].astype(np.float32)
    if mask is not None: mask = mask[slice_lat, slice_lon]
    lon, lat = lon[slice_lon], lat[slice_lat]

    return lon, lat, img, mask

def compute_grad(data: np.ndarray, 
                 method: str = 'sobel', 
                 sigma: int = 1, 
                 disk_size: int = 2) -> np.ndarray:
    
    '''
    Compute the gradient of an image {data} using the specified {method}

    Parameters
    ----------
      --> data [np.ndarray]: The image on which the gradient must be calculated
      --> method [str] (optional): The method to use for computing the gradient among ['sobel', 'gradient'], default is 'sobel'
      --> sigma [int] (optional): For 'sobel' method, the sigma value used (see scipy.ndimage.sobel function), default is 1
      --> disk_size [int] (optional): For 'gradient' method, the radius of the disk to be used (see skimage.filters.rank.gradient function), default is 2

    Returns
    -------
      --> [np.ndarray]: Gradient of the image 
    '''

    if method == 'sobel':
        sobel_h = sobel(gaussian(data, sigma=sigma), 0)
        sobel_v = sobel(gaussian(data, sigma=sigma), 1)
        return np.sqrt(sobel_h**2 + sobel_v**2)
    elif method == 'gradient':
        return gradient(data, footprint=disk(disk_size))
    else:
        raise ValueError("Choose a method among : ['sobel', 'gradient']")   
    
def generate_memmap(array: np.ndarray, 
                    name: str, 
                    folder: str = './joblib_memmap'):

    '''
    Share a numpy {array} between processes in joblib, saving it as a memmap file in memory (shared memory). For parallelization purposes.

    Parameters
    ----------
      --> array [np.ndarray]: Numpy array to share between processes
      --> name [str]: Name of the numpy array (file name)
      --> folder[str] (optional): Folder where memmap file must be saved. Default is './joblib_memmap'
    '''

    if not os.path.exists(folder):
        os.mkdir(folder)

    # Create file and load it again in shared memory
    array_filename_memmap = os.path.join(folder, f"{name}_memmap")
    dump(array, array_filename_memmap)

    return load(array_filename_memmap, mmap_mode='w+')



