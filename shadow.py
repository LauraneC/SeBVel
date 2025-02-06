import os
import time
import xdem
import glob

import datetime as dt
import numpy as np
import horayzon as hray
import rasterio as rio
import rasterio.mask as rio_mask
import skyfield.api as skyapi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import skimage.draw as skdraw
import geopandas as gpd

from tqdm import tqdm
from utils import load_domain, compute_grad, generate_memmap
from pyproj import Transformer
from skimage.morphology import disk, binary_closing, binary_opening, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.segmentation import flood
from skimage.filters import gaussian
from scipy.interpolate import griddata, interpn
from scipy.ndimage import median_filter
from skimage.exposure import histogram
from joblib import Parallel, delayed

class Shadow:
    '''
    Object to manage and remove shadows in a satellite image using a Digital Elevation Model.
    '''

    #%% ---------------------------------------------------------------------------
    #region                           INITIALIZATION
    # -----------------------------------------------------------------------------

    def __init__(self, 
                 file_dem: str | None = None, 
                 file_ortho: str | None = None,
                 settings: dict | None = None,
                 domain: dict | None = None,
                 verbose: bool = False, 
                 **kwargs) -> None:
        
        '''
        Constructor of the shadow class

        Parameters
        ----------

        Returns
        -------
        '''

        # Parameter the plots
        mpl.style.use("classic")
        mpl.rcParams['image.interpolation'] = 'none'
        # Change latex fonts
        mpl.rcParams["mathtext.fontset"] = "custom"
        # Custom mathtext font (set default to Bitstream Vera Sans)
        mpl.rcParams["mathtext.default"] = "rm"
        mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

        # PUBLIC attributes
        self.file_dem = None
        self.elevation_pad = None
        self.lon_pad = None
        self.lat_pad = None
        self.mask_outliers_pad = None

        self.elevation = None
        self.lon = None
        self.lat = None
        self.mask_outliers = None

        self.file_ortho = None
        self.ortho = None
        self.lon_ortho = None
        self.lat_ortho = None
        
        self.shadow = None
        self.grad = None

        self.domain = None
        self.domain_outer = None

        self.shadow_free = None
        self.illumination = None

        self.verbose = verbose

        # Other attributes
        self.ellps = None
        self.sun_position = None
        
        # CONSTANT attributes (defined from a dictionary or a setting file, or default)
        if not isinstance(settings, dict):
            settings = {}
        self.SMALL_SHADOW_LIMIT = settings["SMALL_SHADOW_LIMIT"] if "SMALL_SHADOW_LIMIT" in settings.keys() else 50
        self.SHADOW_MAX_VALUE = settings["SHADOW_MAX_VALUE"] if "SHADOW_MAX_VALUE" in settings.keys() else 100
        self.LIGHT_SHADOW_MIN_DIFF = settings["LIGHT_SHADOW_MIN_DIFF"] if "LIGHT_SHADOW_MIN_DIFF" in settings.keys() else 10
        self.WEAKLY_LINKED_SHADOW_DISK_RADIUS = settings["WEAKLY_LINKED_SHADOW_DISK_RADIUS"] if "WEAKLY_LINKED_SHADOW_DISK_RADIUS" in settings.keys() else 5
        self.SUN_WIDTH = settings["SUN_WIDTH"] if "SUN_WIDTH" in settings.keys() else 0.5
        self.DEM_RESOLUTION = None # Defined in load_dem()
        self.ORTHO_RESOLUTION = None # Defined in load_ortho()
        self.DEM_ORTHO_FACTOR = None # Defined in load_ortho()

        # Load the Digital Elevation Model
        if file_dem is not None:
            self.load_dem(file_dem, domain=domain, **kwargs)
        
        # Load the Orthoimage
        if file_ortho is not None:
            self.load_ortho(file_ortho)

    def load_dem(self, 
                 file_dem: str, 
                 domain: dict | None = None, 
                 dist_search: float = 1.0, 
                 ellps: str = 'WGS84',
                 file_rgi: str | None = None,
                 rough_dem: int | float | str = 16):
        
        '''
        Load the Digital Elevation Model (DEM) which is gonna be used to cast the shadows. The image is cropped to a given {domain}.

        Parameters
        ----------
          --> file_dem [str]: Path to the file containing the DEM
          --> domain [dict | None] (optional): If not None, crops the image to this domain (and pads it to reach Horayzon requirements, 
                see utils.load_domain function). If None, load the whole image (/!\ takes a lot of memory). Default is None.
          --> dist_search [float] (optional): Maximum shadow size (in km) for Horayzon, it has an impact on the size of the padding when loading
                the image, default is 0.5
          --> ellps [str] (optional): Reference ellipsoid to compute the domain outer, default is 'WGS84'
          --> file_rgi [str | None] (optional): Path to the RGI shapefile. If {rough_dem} is provided, the DEM (from {file_dem}) is replaced by a coarser
                DEM (from {rough_dem}) on glaciers. If not, glacier areas are smoothened using a Gaussian filter with {sigma}. It can be used to avoid
                casting shadows from crevasses or seracs. Default is None.
          --> rough_dem [int | float | str] (optional): If str, path to a coarser DEM than the one given in {file_dem}. If int, compute a coarser
                DEM by degrading the {file_dem} DEM to {rough_dem} resolution. Default is 16m.
        '''

        assert os.path.exists(file_dem), f"Impossible to load DEM : {file_dem} not found"

        self.file_dem = file_dem

        # Load the whole DEM
        if domain is None:
            with rio.open(file_dem) as src:
                domain = {"lon_min": src.bounds.left, "lon_max": src.bounds.right,
                          "lat_min": src.bounds.bottom, "lat_max": src.bounds.top}
                if src.crs != "EPSG:4326":
                    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    (domain["lon_min"], domain["lat_min"]) = transformer.transform(domain["lon_min"], domain["lat_min"])
                    (domain["lon_max"], domain["lat_max"]) = transformer.transform(domain["lon_max"], domain["lat_max"])
        
        # Compute the boundaries of the domain on a curved surface (Earth surface)
        self.domain = domain
        self.ellps = ellps
        self.domain_outer = hray.domain.curved_grid(self.domain, dist_search, ellps)
        if self.verbose: print(f"Domain outer : {self.domain_outer}")

        # Compute DEM resolution
        with rio.open(file_dem, 'r') as src:
            self.DEM_RESOLUTION = (src.bounds.right - src.bounds.left) / src.width

        # Load domain in the image
        self.lon_pad, self.lat_pad, self.elevation_pad, mask = load_domain(file_dem, domain_outer=self.domain_outer, use_xdem=True, 
                                                                           reprojection='EPSG:4326', mask=file_rgi, verbose=self.verbose)

        # Mask outliers
        self.mask_outliers_pad = (self.elevation_pad < 0.0) | np.isnan(self.elevation_pad)
        self.elevation_pad[self.mask_outliers_pad] = -1.0

        if mask is not None or self.verbose: 
            slice_in = (slice(np.where(self.lat_pad >= domain["lat_max"])[0][-1],
                              np.where(self.lat_pad <= domain["lat_min"])[0][0] + 1),
                        slice(np.where(self.lon_pad <= domain["lon_min"])[0][-1],
                              np.where(self.lon_pad >= domain["lon_max"])[0][0] + 1))
            if self.verbose: print("Inner domain size in DEM : " + str(self.elevation_pad[slice_in].shape))

            # Replace
            if rough_dem is not None:
                if isinstance(rough_dem, int) or isinstance(rough_dem, float): # Degrade the precise DEM to get a coarser 
                    #TODO directly compute a coarser DEM from the {file_dem} DEM
                    # Difficulties : manage missing data (interpolation ?), manage the padding
                    pass

                else: # Load a coarser DEM
                    _, _, rough_dem, _ = load_domain(rough_dem, domain_outer=self.domain_outer, use_xdem=True, 
                                                     reprojection='EPSG:4326', verbose=self.verbose)
                    rough_dem[(rough_dem < 0) | np.isnan(rough_dem)] = 0
                    rough_dem = resize(rough_dem, self.elevation_pad.shape)

                self.elevation_pad = np.where(mask & (rough_dem > 0), rough_dem, self.elevation_pad) #replace the elevation pad values by the rough dem ones, in the mask area

    def load_ortho(self, 
                   file_ortho: str) -> None:
        
        '''
        Load an orthoimage and crop it to the domain outer (self.domain_outer, see self.load_dem method).
          /!\ The DEM must be loaded before the orthoimage (using self.load_dem method) 

        Parameters
        ----------
          --> file_ortho [str]: Path to the file containing the orthoimage to be loaded
        '''

        assert self.domain_outer != None, "You must load the dem (using load_dem() method) before the orthoimage"
        assert os.path.exists(file_ortho), f"Impossible to load orthoimage : {file_ortho} not found"

        self.file_ortho = file_ortho

        # Compute ortho resolution and multiplication factor between DEM and ortho resolutions
        with rio.open(file_ortho, 'r') as src:
            self.ORTHO_RESOLUTION = (src.bounds.right - src.bounds.left) / src.width
            self.DEM_ORTHO_FACTOR = self.DEM_RESOLUTION / self.ORTHO_RESOLUTION

        # Load domain in the image
        if os.path.isfile(file_ortho.split('.tif')[0] + "_EPSG4326.tif"): # Reprojection was already done
            self.lon_ortho, self.lat_ortho, self.ortho, _ = load_domain(file_ortho.split('.tif')[0] + "_EPSG4326.tif", self.domain_outer, verbose=self.verbose)
        else:
            self.lon_ortho, self.lat_ortho, self.ortho, _ = load_domain(file_ortho, self.domain_outer, reprojection='EPSG:4326', verbose=self.verbose)

        # Compute indices of inner domain
        slice_in_ortho = (slice(np.where(self.lat_ortho >= self.domain["lat_max"])[0][-1],
                                np.where(self.lat_ortho <= self.domain["lat_min"])[0][0] + 1),
                          slice(np.where(self.lon_ortho <= self.domain["lon_min"])[0][-1],
                                np.where(self.lon_ortho >= self.domain["lon_max"])[0][0] + 1))
        self.ortho = self.ortho[slice_in_ortho].astype(np.uint8)
        self.lon_ortho = self.lon_ortho[slice_in_ortho[1]]
        self.lat_ortho = self.lat_ortho[slice_in_ortho[0]]

        if self.verbose: print("Inner domain size in ortho: " + str(self.ortho.shape))

    #endregion
    #%% ---------------------------------------------------------------------------
    #region                              CAST SHADOW
    # -----------------------------------------------------------------------------

    def cast_shadow(self,
                    date: dt.datetime,
                    preprocess: bool = True,
                    pipeline: list = [],
                    plot_along_pipeline: bool = False,
                    resize_to_ortho: bool = True,
                    clean: bool = False,
                    **kwargs):
        '''
        Calculate the shadows casted by the terrain at a given {date} using Horayzon package. Shadows can thus be improved using a shadow improvement
        pipeline and the self.improve_shadow method.
        Parameters
        ----------
          --> date [dt.datetime]: Date (day of the year and hour) at which shadows must be casted using Horayzon package
          --> preprocess [bool] (optional): Apply a preprocessing step to casted shadows (e.g. separate weakly linked shadows and remove small shadows),
                default is True
          --> pipeline [list] (optional): List of dictionaries giving the shadow improvement pipeline (see self.improve_shadow method for more informations).
                An orthoimage must be provided to apply an improvement pipeline. Default is an empty list (no improvement step)
          --> plot_along_pipeline [bool] (optional): Plot the shadow at each step of the shadow improvement pipeline (see self.improve_shadow method),
                default is False
          --> resize_to_ortho [bool] (optional): If True, resize the casted shadow to the orthoimage size.
          --> clean [bool] (optional): Set some variables to None (those which will not be used in future processings if you don't call cast_shadow again).
                The goal is to save memory. Here elevation_pad, lon_pad and lat_pad attributes. Default is False.
          --> **kwargs (optional): Plot parameters if plot_along_pipeline=True (see self.plot_shadow method)
        '''
        # Compute (again) indices of inner domain
        slice_in = (slice(np.where(self.lat_pad >= self.domain["lat_max"])[0][-1],
                          np.where(self.lat_pad <= self.domain["lat_min"])[0][0] + 1),
                    slice(np.where(self.lon_pad <= self.domain["lon_min"])[0][-1],
                          np.where(self.lon_pad >= self.domain["lon_max"])[0][0] + 1))
        offset_0 = slice_in[0].start
        offset_1 = slice_in[1].start
        # Compute ECEF coordinates
        x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(self.lon_pad, self.lat_pad), self.elevation_pad, ellps=self.ellps)
        dem_dim_0, dem_dim_1 = self.elevation_pad.shape
        self.mask_outliers = self.mask_outliers_pad[slice_in]
        self.elevation = np.ascontiguousarray(self.elevation_pad[slice_in]) # Orthometric height (height above mean sea level)
        self.lon = self.lon_pad[slice_in[1]]
        self.lat = self.lat_pad[slice_in[0]]
        # Set self.elevation_pad, self.lon_pad and self.lat_pad to None
        if clean:
            self.elevation_pad = None
            self.lon_pad = None
            self.lat_pad = None
            self.mask_outliers_pad = None
        # Compute ENU coordinates
        trans_ecef2enu = hray.transform.TransformerEcef2enu(lon_or=self.lon[int(len(self.lon) / 2)], lat_or=self.lat[int(len(self.lat) / 2)], ellps=self.ellps)
        x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef, trans_ecef2enu)
        # Compute unit vectors (up and north) in ENU coordinates for inner domain
        vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(self.lon, self.lat))
        vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in], z_ecef[slice_in], vec_norm_ecef, ellps=self.ellps)
        del x_ecef, y_ecef, z_ecef
        vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
        vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
        del vec_norm_ecef, vec_north_ecef
        # Merge vertex coordinates and pad geometry buffer
        vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
        # Compute rotation matrix (global ENU -> local ENU)
        rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu, vec_norm_enu)
        del vec_north_enu
        # Compute slope (in global ENU coordinates!)
        slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
                       slice(slice_in[1].start - 1, slice_in[1].stop + 1))
        vec_tilt_enu = np.ascontiguousarray(hray.topo_param.slope_plane_meth(x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
                                                                             rot_mat=rot_mat_glob2loc, output_rot=False)[1:-1, 1:-1])
        # Compute surface enlargement factor
        surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
        if self.verbose: print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min() + ", %.3f" % surf_enl_fac.max())
        # Load Skyfield data -> position lies on the surface of the ellipsoid by default
        skyapi.load.directory = os.path.dirname(self.file_dem)
        planets = skyapi.load("de421.bsp")
        sun = planets["sun"]
        earth = planets["earth"]
        loc_or = earth + skyapi.wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)
        # Initialise terrain
        mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
        terrain = hray.shadow.Terrain()
        dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
        terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                        offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                        surf_enl_fac, mask=mask, elevation=self.elevation,
                        refrac_cor=True)
        # Compute sun position
        ts = skyapi.load.timescale()
        t = ts.from_datetime(date)
        astrometric = loc_or.at(t).observe(sun)
        alt, az, d = astrometric.apparent().altaz()
        x = d.m * np.cos(alt.radians) * np.sin(az.radians)
        y = d.m * np.cos(alt.radians) * np.cos(az.radians)
        z = d.m * np.sin(alt.radians)
        sun_position = np.array([x, y, z], dtype=np.float32)
        self.sun_position = {"alt": alt, "az": az, "d": d}
        # Calculate shadows from dem
        self.shadow = np.zeros(vec_tilt_enu.shape[:2], dtype=np.bool_)
        terrain.shadow(sun_position, self.shadow)
        # Remove shadow casted by DEM errors
        self.shadow[self.mask_outliers] = False
        # Apply a preprocessing step to the shadow before resizing it (in the improve_shadow function)
        if preprocess:
            # Separate weakly linked shadows
            self.shadow = binary_opening(self.shadow, footprint=disk(max(1, self.WEAKLY_LINKED_SHADOW_DISK_RADIUS // 8)))
            # Remove small shadows
            shadows = label(self.shadow, background=0, connectivity=1)
            for val in np.unique(shadows)[1:]:
                if np.sum(shadows == val) <= self.SMALL_SHADOW_LIMIT:
                    self.shadow[shadows == val] = False
            del shadows
        if self.ortho is not None and resize_to_ortho:
            self.shadow = resize(self.shadow, self.ortho.shape)
            # Second preprocessing on resized shadow
            if preprocess:
                self.shadow[self.ortho > self.SHADOW_MAX_VALUE] = False # Remove pixels with excessive value
                # Separate weakly-linked shadows before filling gaps
                self.shadow = binary_closing(binary_opening(self.shadow, footprint=disk(self.WEAKLY_LINKED_SHADOW_DISK_RADIUS)),
                                        footprint=disk(self.WEAKLY_LINKED_SHADOW_DISK_RADIUS))
            if len(pipeline) > 0:
                # Every improvement methods are made on the upsampled shadows (to ortho shape)
                self.improve_shadow(pipeline, plot_along_pipeline, **kwargs)

    def nday_shadow_map(self,
                        dates: list[dt.datetime],
                        contours: int = 0,
                        preprocess: bool = False,
                        parallelize: int | bool = False) -> np.ndarray:

        '''
        Compute a shadow map giving the number of days among {dates} when a pixel is shadowed, for each pixel of the DEM. The number of days is either
        computed in the whole shadow (if {contours} is 0), or in a {contours}-wide buffer around the side of the shadow (if {contours} > 0) as pixels
        around shadow sides are likely to be the most impacted by the shadow during a correlation.

        Parameters
        ----------
          --> dates [list[dt.datetime]]: List of the dates when shadows must be casted (use dt.datetime(..., tzinfo=dt.timezone.utc)).
          --> contours [int] (optional): If 0, for each day, add every shadowed pixels to the shadow map. If > 0, for each day, add only the pixels around
            the shadow contours ({contours}-wide buffer) to the shadow map.
          --> preprocess [bool] (optional): Preprocess the casted shadow (see cast_shadow() method)
          --> parallelize [int | bool] (optional): Whether the computation should be parallelized or not. Put the number of CPU you want to use, or False.
            If False, sequential computation is applied. Default is False.
        '''

        # Compute shadow map for the first date (gives the shape of the shadow map)
        first_date = dates.pop(0)
        self.cast_shadow(first_date, preprocess=preprocess, resize_to_ortho=False, clean=False)
        shadow_map = None
        if contours > 0:
            contours_width = int(contours / self.DEM_RESOLUTION / 2)
            shadow_map = ((self.shadow ^ binary_dilation(self.shadow, footprint=disk(contours_width))) + \
                          (self.shadow ^ binary_erosion(self.shadow, footprint=disk(contours_width)))).astype(np.uint16)
        else:
            shadow_map = np.copy(self.shadow).astype(np.uint16)

        # Update shadow map for a given date
        def cast_shadow_date(date, shadow_map):
            self.cast_shadow(date, preprocess=preprocess, resize_to_ortho=False, clean=False)
            if contours > 0:  # Add only pixels around the shadow contours to the shadow_map
                contours_width = int(contours / self.DEM_RESOLUTION / 2)
                shad_contours = (self.shadow ^ binary_dilation(self.shadow, footprint=disk(contours_width))) + \
                                (self.shadow ^ binary_erosion(self.shadow, footprint=disk(
                                    contours_width)))  # Buffer around shadow contours
                shadow_map[shad_contours] += 1
            else:  # Add every shadowed pixel to the shadow map
                shadow_map[self.shadow] += 1

        if self.verbose: dates = tqdm(dates)
        if parallelize != False:
            # Share in memory every array attributes to allow its access and modification between processes during parallelization
            self.elevation_pad = generate_memmap(self.elevation_pad, "elevation_pad")
            self.lon_pad = generate_memmap(self.lon_pad, "lon_pad")
            self.lat_pad = generate_memmap(self.lat_pad, "lat_pad")
            self.elevation = generate_memmap(self.elevation, "elevation")
            self.lon = generate_memmap(self.lon, "lon")
            self.lat = generate_memmap(self.lat, "lat")
            self.shadow = generate_memmap(self.shadow, "shadow")
            shadow_map = generate_memmap(shadow_map, "shadow_map")

            Parallel(n_jobs=parallelize, verbose=0)(delayed(cast_shadow_date)(date, shadow_map) for date in dates)

            for memmap_file in glob.glob("./joblib_memmap/*"):
                os.remove(memmap_file)
        else:
            for date in dates:
                cast_shadow_date(date, shadow_map)

        return shadow_map

    #endregion
    #%% ---------------------------------------------------------------------------
    #region                            SHADOW IMPROVEMENT
    # -----------------------------------------------------------------------------

    def improve_shadow(self, 
                       pipeline: list,
                       plot_along_pipeline: bool = False, 
                       **kwargs) -> None:
        
        '''
        Apply a shadow improvement pipeline to improve the shadow detection.

        Parameters
        ----------
          --> pipeline [list]: Shadow improvement pipeline, a list of tuple where each element corresponds to an improvement method to be applied (method to
                be applied, parameters of the method and title of the eventual plot):

                Example: for a Shadow object called shadow
                    shadow = Shadow(...)
                      ...
                    pipeline = [(shadow.adaptative_flood, {to_flood: 'ortho', extension_limits: (80, 300)}, "Flooded shadows"),
                                (shadow.refine_contours, {}, "Contours-refined shadows")]
                    shadow.improve_shadow(pipeline=pipeline, plot_along_pipeline=True)

                For now, accepted methods are (see there definition for more information) : flood, adaptative_flood, regional_flood
                and refine_contours

          --> plot_along_pipeline [bool] (optional): Plot the shadow at each step of the shadow improvement pipeline, default is False
          --> **kwargs (optional): Plot parameters if plot_along_pipeline=True (see self.plot_shadow method)
        '''

        assert self.shadow is not None, "Shadows were not casted, nothing to improve"

        # Apply shadow improvement methods and plot the shadow evolution along the way
        if plot_along_pipeline:
            fig, axes = plt.subplots(nrows=len(pipeline)//2 + 1, ncols=2 if len(pipeline) > 0 else 1, 
                                    figsize=kwargs["figsize"] if "figsize" in kwargs.keys() else (20, min((len(pipeline)//2+1)*6, 12)))
            if len(axes.shape) == 1:
                axes = np.expand_dims(axes, axis=0)
            self.plot_shadow(ax=axes[0,0], title="Original DEM-casted shadows", **kwargs)

        c = 1
        for improvement in pipeline:
            # Call improvement function
            improvement[0](**improvement[1])

            self.shadow[self.ortho > self.SHADOW_MAX_VALUE] = False # Remove pixels with excessive value
            # Separate weakly-linked shadows before filling gaps
            self.shadow = binary_closing(binary_opening(self.shadow, footprint=disk(self.WEAKLY_LINKED_SHADOW_DISK_RADIUS)), 
                                    footprint=disk(self.WEAKLY_LINKED_SHADOW_DISK_RADIUS))

            if plot_along_pipeline:
                title = improvement[2] if len(improvement) > 2 else ""
                self.plot_shadow(ax=axes[c//2, c%2], title=title, **kwargs)
                c += 1

        # Remove small shadows
        shadows = label(self.shadow, background=0, connectivity=1)
        for val in np.unique(shadows)[1:]:
            if np.sum(shadows == val) <= self.SMALL_SHADOW_LIMIT * self.DEM_ORTHO_FACTOR**2:
                self.shadow[shadows == val] = False
        del shadows

    def flood(self, 
              to_flood: str = 'ortho', 
              tolerance: int = 15, 
              **kwargs):
        
        '''

        '''

        # Label separate shadows
        shadows = label(self.shadow, background=0, connectivity=1)
        if self.verbose:
            print("Applying simple flood improvement method...") 
            print(f"{len(np.unique(shadows))-1} shadow regions to be flooded")
        self.shadow = np.zeros(self.shadow.shape, dtype=np.bool_)

        # Which image to flood (recommanded : ortho)
        if to_flood == 'ortho':
            to_flood = self.ortho
        elif to_flood == 'grad':
            if self.grad == None:
                self.grad = compute_grad(**kwargs)
            to_flood = self.grad

        # Minimum to_flood (ortho or grad) value of each shadow region
        flood_points = [np.unravel_index(np.argmin(np.where(shadows == val, self.ortho, 255)), self.ortho.shape) for val in np.unique(shadows)[1:]]

        # Flood every shadows (from its minimum value) using a constant tolerance
        iterable = range(len(flood_points))
        if self.verbose: iterable = tqdm(iterable)
        for p in iterable:
            pos = flood_points[p]
            if self.ortho[pos] <= self.SHADOW_MAX_VALUE:
                self.shadow[flood(to_flood, pos, tolerance=tolerance)] = True

    def adaptative_flood(self, 
                         to_flood='ortho', 
                         extension_limits=(80, 300),
                         tol_range=np.arange(5, 31, 1).astype(np.uint8),
                         shadow_diff_value=5,
                         **kwargs):
        
        '''
        
        '''

        # Label separate shadows
        shadows = label(self.shadow, background=0, connectivity=1)
        if self.verbose:
            print("Applying adaptative flood improvement method...") 
            print(f"{len(np.unique(shadows))-1} shadow regions to be flooded")
        self.shadow = np.zeros(self.shadow.shape, dtype=np.bool_)

        # Which image to flood (recommanded : ortho)
        if to_flood == 'ortho':
            to_flood = self.ortho
        elif to_flood == 'grad':
            if self.grad == None:
                self.grad = compute_grad(**kwargs)
            to_flood = self.grad

        # Minimum to_flood (ortho or grad) value of each shadow region
        flood_points = [np.unravel_index(np.argmin(np.where(shadows == val, self.ortho, 255)), self.ortho.shape) for val in np.unique(shadows)[1:]]

        perc_min, perc_max = extension_limits[0] / 100, extension_limits[1] / 100
        iterable = range(len(flood_points))
        if self.verbose: iterable = tqdm(iterable)
        for p in iterable:
            pos = flood_points[p]
            if self.ortho[pos] <= self.SHADOW_MAX_VALUE: # If not, the DEM-casted shadow region is probably not a real shadow                
                nb_pix_p = np.sum(shadows == p+1)
                for t in range(len(tol_range)):
                    tol = tol_range[t]
                    flooded = flood(to_flood, pos, tolerance=tol)
                    sum_flooded = np.sum(flooded)
                    if sum_flooded >= perc_max * nb_pix_p:
                        if tol > np.min(tol_range):
                            flooded = flood(to_flood, pos, tolerance=tol_range[t-1])
                        break
                    
                    if np.sum(flooded & (shadows == p+1)) >= perc_min * nb_pix_p:
                        flooded_bis = flood(to_flood, pos, tolerance=tol_range[t+1])
                        new_shadow = flooded ^ flooded_bis
                        while np.sum(flooded) < perc_max*nb_pix_p and abs(np.percentile(self.ortho[new_shadow], 75) - \
                                                                          np.percentile(self.ortho[shadows == p+1], 25)) < shadow_diff_value \
                                                                  and tol < np.max(tol_range):
                            tol = tol_range[t+1]
                            flooded = flooded_bis
                            flooded_bis = flood(to_flood, pos, tolerance=tol)
                            t += 1
                        break
                self.shadow[flooded] = True
                
    def regional_flood(self, 
                       parallelize: bool = False,
                       joblib_folder: str = './joblib_memmap',
                       to_flood: str = 'ortho',
                       tol_min: int = 5,
                       tol_max: int = 30,
                       box_size: int = 500,
                       box_buffer: int = 100,
                       box_redundancy: bool = True,
                       expansion_limit: int = 200,
                       max_shadow_prop_in_box: float = 0.95,
                       **kwargs):
        
        '''
        
        Parameters
        ----------


        '''

        if self.verbose: start_time = time.time()
        
        # Label separate shadows
        shadows = label(self.shadow, background=0, connectivity=1)
        props = regionprops(shadows)
        if self.verbose:
            print("Applying regional flood improvement method...") 
            print(f"{len(np.unique(shadows))-1} shadow regions to be flooded")

        # Which image to flood (recommanded : ortho)
        if to_flood == 'ortho':
            to_flood = self.ortho
        elif to_flood == 'grad':
            if self.grad == None:
                self.grad = compute_grad(**kwargs)
            to_flood = self.grad

        # Coordinates of the image
        ortho_size_x, ortho_size_y = self.ortho.shape[0]-1, self.ortho.shape[1]-1
        coords_x, coords_y = np.meshgrid(np.linspace(0, ortho_size_x, ortho_size_x+1),
                                         np.linspace(0, ortho_size_y, ortho_size_y+1), indexing='ij')
        coords_x = coords_x.astype(np.uint16)
        coords_y = coords_y.astype(np.uint16)

        # Regional flood on a single shadow (for parallelization purposes)
        def regional_flood_single(val, bbox, shadows, to_flood, shadow):
            shad = (shadows == val)

            # Define the regions
            lim_x = [max(0, bbox[0] - box_buffer), min(shad.shape[0], bbox[2] + box_buffer)]
            lim_y = [max(0, bbox[1] - box_buffer), min(shad.shape[1], bbox[3] + box_buffer)]
            if box_redundancy: # Add a redundancy (overlap) between the regions of half its size => double the amount of regions but more robust
                x_pos = np.linspace(lim_x[0], lim_x[1], max(1, (lim_x[1] - lim_x[0]) // (box_size // 2))+1).astype(np.uint16)
                y_pos = np.linspace(lim_y[0], lim_y[1], max(1, (lim_y[1] - lim_y[0]) // (box_size // 2))+1).astype(np.uint16)
                box_size_x = 2*(x_pos[1] - x_pos[0]) if len(x_pos) > 2 else lim_x[1] - lim_x[0]
                box_size_y = 2*(y_pos[1] - y_pos[0]) if len(y_pos) > 2 else lim_y[1] - lim_y[0]
                if lim_x[0] > 0 and len(x_pos) > 2:
                    np.insert(x_pos, 0, 2*x_pos[0] - x_pos[1])
                if lim_y[0] > 0 and len(y_pos) > 2:
                    np.insert(y_pos, 0, 2*y_pos[0] - y_pos[1])
            else: # No redundancy (overlap) between the regions
                x_pos = np.linspace(lim_x[0], lim_x[1], max(1, (lim_x[1] - lim_x[0]) // box_size)+1).astype(np.uint16)
                y_pos = np.linspace(lim_y[0], lim_y[1], max(1, (lim_y[1] - lim_y[0]) // box_size)+1).astype(np.uint16)
                box_size_x = x_pos[1] - x_pos[0] if len(x_pos) > 2 else lim_x[1] - lim_x[0]
                box_size_y = y_pos[1] - y_pos[0] if len(y_pos) > 2 else lim_y[1] - lim_y[0]

            # Compute every region one after the other
            for x in range(len(x_pos)-1):
                for y in range(len(y_pos)-1):
                    box_mask = (coords_x >= x_pos[x]) & (coords_x <= x_pos[x] + box_size_x) & (coords_y >= y_pos[y]) & (coords_y <= y_pos[y] + box_size_y)

                    # If the region is completly shadowed or non-shadowed, there is no improvement to be done
                    if not (np.all(shad[box_mask]) or np.all(~shad[box_mask])):
                        if np.percentile(to_flood[box_mask], 99) <= self.SHADOW_MAX_VALUE and \
                           np.sum(shad & box_mask) / np.sum(box_mask) >= max_shadow_prop_in_box:
                            # Probably a hole in the casted shadow
                            shadow[box_mask] = True
                        else:
                            # Compute the median of each area                        
                            mean_s = np.nanmedian(to_flood[shad & box_mask])
                            mean_ns = max(np.nanmedian(to_flood[~shadow & box_mask]), np.nanmean(to_flood[~shadow & box_mask]))

                            if mean_ns <= mean_s: # Shadow and non-shadow area are mixed up => minimum tolerance
                                print('mean_ns <= mean_s')
                                tol = tol_min
                            else:
                                # Sum shadow and non-shadow histograms to get a global relative histogram
                                hist, bins = histogram(to_flood[box_mask])
                                hist = hist.astype(np.float32)
                                hist /= len(to_flood[box_mask])

                                # Define the tolerance
                                select_bins = (bins >= mean_s) & (bins <= mean_ns)
                                pos_min = bins[select_bins][np.argmin(hist[select_bins])] # Position of the histogram min value between the two medians
                                if np.max(hist[bins >= pos_min]) - hist[np.where(bins == pos_min)] >= 0.05 * np.max(hist): # If this minimum is significant enough
                                    tol = pos_min - np.nanmin(to_flood[(to_flood > 0) & shad & box_mask]) # Tolerance is set as the minimum position
                                else:  
                                    # Parameters to approach the histogram of the shadow area by a Gaussian
                                    gaussian_s = [np.max(hist_s), np.argmax(hist_s) + np.min(to_flood[shad & box_mask]), 
                                                np.std(to_flood[shad & box_mask & (to_flood <= self.SHADOW_MAX_VALUE)])]

                                    # For a Gaussian, 95% of the surface is contained in the [mu - 2.sigma, mu + 2.sigma] range
                                    # Here we consider this limit as the tolerance for the flood by default
                                    tol = gaussian_s[1] + 2*abs(gaussian_s[2]) - np.nanmin(to_flood[(to_flood > 0) & shad & box_mask])

                            # Flood the orthoimage from its minimum value (in the shadow) with the computed tolerance
                            print(tol)
                            tol = max(tol_min, min(tol, tol_max))
                            flood_point = np.unravel_index(np.argmin(np.where((to_flood > 0) & shad & box_mask, to_flood, np.max(to_flood))), to_flood.shape)
                            new_shadow = flood(np.where(box_mask, to_flood, np.max(to_flood)), flood_point, tolerance=tol) & box_mask

                            # If the expansion of the shadow is too high, lower the tolerance
                            while (np.sum(new_shadow) - np.sum(shad & box_mask)) / np.sum(shad & box_mask) >= expansion_limit/100 and tol > tol_min:
                                tol -= 1
                                new_shadow = flood(np.where(box_mask, to_flood, np.max(to_flood)), flood_point, tolerance=tol) & box_mask

                            # Update the shadow
                            shadow[new_shadow] = True

        # Compute bbox outside of regional_flood_single() function because it is not picklable
        iterable = np.unique(shadows)[1:]
        props = regionprops(shadows)
        bboxs = []
        for val in iterable:
            bboxs.append(props[val-1].bbox) # Box around shadow

        # Apply to every individual shadow
        if self.verbose: iterable = tqdm(iterable)
        if parallelize != False: # Parallelize
            # Generate memory maps (shared memory) for faster computation
            shadows = generate_memmap(shadows, "shadows", folder=joblib_folder)
            to_flood = generate_memmap(to_flood, "to_flood", folder=joblib_folder)
            self.shadow = generate_memmap(self.shadow, "shadow", folder=joblib_folder)

            Parallel(n_jobs=parallelize, verbose=0)(
                delayed(regional_flood_single)(val, bboxs[val-1], shadows, to_flood, self.shadow)
            for val in iterable) # Parallel computation

            for memmap_file in glob.glob(os.path.join(joblib_folder, "*")): # Delete memmap files
                os.remove(memmap_file)

        else:
            for val in iterable: # Sequential computation
                regional_flood_single(val, bboxs[val-1], shadows, to_flood, self.shadow)

        if self.verbose: print(f"Regional flooding took {round(time.time() - start_time, 2)} s")

    def refine_contours(self,
                        parallelize: bool = False,
                        joblib_folder: str = './joblib_memmap',
                        contours_buffer_size: int = 10, 
                        shadow_diff_value: int = 10, 
                        win_size: int = 100):
        
        '''
        Refine the contours of the shadow by comparing pixel values to the median value of shadowed and non-shadowed areas. Thus classifying a pixel value
        to the area with the nearest median. For each pixel of shadow contours, medians are computed in a {win_size}-wide window, outside a contours buffer
        ({contours_buffer_size} around the contours). All pixels within this buffer are then classified as shadow (closer to shadow's median) or 
        non-shadow (closer to non-shadow's median).

        Parameters
        ----------
          --> parallelize [bool] (optional): Whether the computation should be parallelized or not. Put the number of CPU you want to use, or False.
            If False, sequential computation is applied. Default is False.
          --> joblib_folder [str] (optional): Folder where memory map files (for faster parallelization) must be saved.
          --> contours_buffer_size [int] (optional): Radius of the buffer around the contours where pixels must be classified. The buffered contours will
            be two times {contours_buffer_size} wide. Default is 10.
          --> shadow_diff_value [int] (optional): If the difference between the two medians (shadowed and non-shadowed areas) is too small 
            (< {shadow_diff_value}), the pixel (and its surroundings) is considered to be missing in the detected shadow (hole) and is directly 
            classified as a shadowed pixel. Default is 10.
          --> win_size [int] (optional): Size of the window around the pixel in which medians are calculated. Default is 100.
        '''

        if self.verbose: start_time = time.time()

        contours = self.shadow ^ binary_dilation(self.shadow, footprint=disk(1)) # Shallow contours
        buffer = (self.shadow ^ binary_dilation(self.shadow, footprint=disk(contours_buffer_size))) + \
                 (self.shadow ^ binary_erosion(self.shadow, footprint=disk(contours_buffer_size))) # Buffer contours in which pixels are classified
        calculated = np.ones(buffer.shape, dtype=np.bool_) # To be considered a valid shadow, it must always be classified as a shadow
        shadow_refine = np.copy(self.shadow)
        win_size = int(win_size / 2)

        # Coordinates of the image
        ortho_size_x, ortho_size_y = self.ortho.shape
        coords_x, coords_y = np.meshgrid(np.linspace(0, ortho_size_x, ortho_size_x),
                                         np.linspace(0, ortho_size_y, ortho_size_y), indexing='ij')
        coords_x = coords_x.astype(np.uint16)
        coords_y = coords_y.astype(np.uint16)

        # Contours refining on a single shadow (for parallelization purposes)
        def refine_contours_single(x, contours_coordinates, ortho, shadow, buffer, calculated, shadow_refine):
            i, j = contours_coordinates[x, :]
            # Compute medians of shadowed and non-shadowed areas within win_size wide window, outside of contous buffer
            select = (slice(max(0, i - win_size), min(i + win_size, ortho_size_x) + 1),
                      slice(max(0, j - win_size), min(j + win_size, ortho_size_y) + 1))
            m = np.nanmedian(self.ortho[select][self.shadow[select] & ~buffer[select]]) # Median of shadow area
            M = np.nanmedian(self.ortho[select][~self.shadow[select] & ~buffer[select]]) # Median of non-shadow area

            if M - m < shadow_diff_value: # The difference is too small, certainly due to a hole in the detected shadows (undetected shadows)
                shadow_refine[select] = np.where(buffer[select] & calculated[select], True, self.shadow[select])
            else: # Pixel values within the surroundings of the contours pixel are 
                condition = np.abs(self.ortho[select] - M) > np.abs(self.ortho[select] - m) # If True, shadow pixel. If False, non-shadow pixel.
                shadow_refine[select] = np.where(buffer[select] & calculated[select], condition, self.shadow[select])
                calculated[select][~condition] = False # If False once, not a shadow

        # Contours coordinates and iterator
        contours_coordinates = np.concatenate([coords_x[contours][:, np.newaxis], coords_y[contours][:, np.newaxis]], axis=1)
        iterable = range(contours_coordinates.shape[0])
        if self.verbose: iterable = tqdm(iterable)

        # Apply to every individual shadow
        if parallelize != False:
            # Generate memory maps (shared memory) for faster computation
            self.ortho = generate_memmap(self.ortho, "ortho", folder=joblib_folder)
            self.shadow = generate_memmap(self.shadow, "shadow", folder=joblib_folder)
            buffer = generate_memmap(buffer, "buffer", folder=joblib_folder)
            calculated = generate_memmap(calculated, "calculated", folder=joblib_folder)
            shadow_refine = generate_memmap(shadow_refine, "shadow_refine", folder=joblib_folder)

            Parallel(n_jobs=parallelize, verbose=0)(
                delayed(refine_contours_single)(x, contours_coordinates, self.ortho, self.shadow, buffer, calculated, shadow_refine)
            for x in iterable) # Parallel computation

            for memmap_file in glob.glob(os.path.join(joblib_folder, "*")): # Delete memmap files
                os.remove(memmap_file)

        else:
            for x in iterable: # Sequential computation
                refine_contours_single(x, contours_coordinates, self.ortho, self.shadow, buffer, calculated, shadow_refine)

        # Update the shadow map with refined contours
        self.shadow = shadow_refine

        if self.verbose: print(f"Contours refining took {round(time.time() - start_time, 2)} s")

    #endregion
    #%% ---------------------------------------------------------------------------
    #region                         INSIDE SHADOW REMOVAL
    # -----------------------------------------------------------------------------

    def remove_shadow_constant(self,
                               parallelize: bool = False,
                               joblib_folder: str = './joblib_memmap',
                               contours_width: int = 8,
                               bbox_buffer: int = 10):
        
        '''
        Remove the inside part of the shadows using a constant illumination factor for each shadow. This illumination factor is computed from
        the median value of shadowed and non-shadowed areas in the box encapsulating the shadow buffered by {bbox_buffer}, and outside the
        contours of width {contours_width}.

        Parameters
        ----------
          --> parallelize [bool | int] (optional): Whether the computation should be parallelized or not. Put the number of CPU you want to use, or False.
            If False, sequential computation is applied. Not necessarily faster here (for small areas). Default is False.
          --> joblib_folder [str] (optional): Folder where memory map files (for faster parallelization) must be saved.
          --> contours_width [int] (optional): Width of the contours, diameter of the disk used for the morphological operators. Default is 8.
          --> bbox_buffer [int] (optional) : Buffer for the box encapsulating the studied shadow, in order to have more pixel to compute the 
            medians. Default is 10.
        '''

        if self.verbose: start_time = time.time()

        # Initialization
        shadows = label(self.shadow, background=0, connectivity=1)
        self.contours = (self.shadow ^ binary_erosion(self.shadow, footprint=disk(contours_width//2))) + \
                        (self.shadow ^ binary_dilation(self.shadow, footprint=disk(contours_width//2)))
        self.illumination = np.zeros(self.shadow.shape, dtype=np.float16)
        ortho_size_x, ortho_size_y = self.ortho.shape

        if self.verbose: print(f"{len(np.unique(shadows))} shadow areas to remove")

        # Constant shadow removing for a single shadow (for parallelization purposes)
        def remove_shadow_constant_single(val, bbox, shadows, contours, ortho, illumination):
            # Compute medians
            slice_x = slice(bbox[0], bbox[2]+1)
            slice_y = slice(bbox[1], bbox[3]+1)
            val_s = np.where((shadows[slice_x, slice_y] == val) & ~contours[slice_x, slice_y], 
                              ortho[slice_x, slice_y], -1) # Ortho value in shadowed areas
            m = np.ma.median(np.ma.masked_where(val_s == -1, val_s))
            del val_s
            val_ns = np.where((shadows[slice_x, slice_y] != val) & ~contours[slice_x, slice_y], 
                               ortho[slice_x, slice_y], -1) # Ortho value in non-shadowed areas
            M = np.ma.median(np.ma.masked_where(val_ns == -1, val_ns))
            del val_ns, slice_x, slice_y

            illumination[shadows == val] = M/m # Illumination factor of the shadow

        # Compute bbox outside of remove_shadow_constant_single() function because it is not picklable
        iterable = np.unique(shadows)[1:]
        props = regionprops(shadows)
        bboxs = []
        for val in iterable:
            bbox = props[val-1].bbox # Box around shadow
            bboxs.append([max(0, bbox[0]-bbox_buffer), max(0, bbox[1]-bbox_buffer), 
                          min(bbox[2]+bbox_buffer, ortho_size_x), min(bbox[3]+bbox_buffer, ortho_size_y)])

        # Apply to every individual shadow
        if self.verbose: iterable = tqdm(iterable)
        if parallelize != False: # Parallelize : not necessarily faster here...
            # Generate memory maps (shared memory) for faster computation
            shadows = generate_memmap(shadows, "shadows", folder=joblib_folder)
            self.contours = generate_memmap(self.contours, "contours", folder=joblib_folder)
            self.ortho = generate_memmap(self.ortho, "ortho", folder=joblib_folder)
            self.illumination = generate_memmap(self.illumination, "illumination", folder=joblib_folder)

            Parallel(n_jobs=parallelize, verbose=0)(
                delayed(remove_shadow_constant_single)(val, bboxs[val-1], shadows, self.contours, self.ortho, self.illumination)
            for val in iterable) # Parallel computation

            for memmap_file in glob.glob(os.path.join(joblib_folder, "*")): # Delete memmap files
                os.remove(memmap_file)

        else:
            for val in iterable: # Sequential computation
                remove_shadow_constant_single(val, bboxs[val-1], shadows, self.contours, self.ortho, self.illumination)

        # Apply the illumination factor to the orthoimage to get a shadow-free image
        maxi = np.max(self.ortho)
        self.shadow_free = np.where((self.illumination > 0) & (self.illumination * self.ortho <= maxi) & ~self.contours, 
                                     self.illumination * self.ortho, self.ortho).astype(np.uint8)

        if self.verbose: print(f"Constant removing inside shadows took {round(time.time() - start_time, 2)} s")

    def remove_shadow_block(self,
                            parallelize: bool = False,
                            joblib_folder: str = './joblib_memmap',
                            bbox_max_size_y: int = 250, 
                            bbox_buffer: int = 50,
                            contours_method: str = "constant",
                            contours_width: int = 8,
                            min_contour_size: tuple[int] = (5, 50000),
                            max_contour_size: tuple[int] = (25, 5000000)):

        '''
        
        '''

        if self.verbose: start_time = time.time()

        # Label shadows
        shadows = label(self.shadow, background=0, connectivity=1)
        props = regionprops(shadows)
        self.illumination = np.zeros(self.shadow.shape, dtype=np.float16)
        if self.verbose: print(f"{len(np.unique(shadows))} shadow areas to remove")

        # Shadow direction
        az = np.radians(self.sun_position["az"].degrees)
        ortho_size_x, ortho_size_y = self.ortho.shape[0]-1, self.ortho.shape[1]-1
        bbox_max_size_x = abs(int(bbox_max_size_y / np.tan(az)))

        # Initialize contours according to the chosen method
        if contours_method == "constant":
            assert isinstance(contours_width, int), "If using 'constant' contours method, contours_width must be an integer"

            # Contours are considered constant for every shadow (neglict the penumbra area variations in length)
            self.contours = (self.shadow ^ binary_erosion(self.shadow, footprint=disk(contours_width//2))) + \
                            (self.shadow ^ binary_dilation(self.shadow, footprint=disk(contours_width//2)))
            
        elif contours_method == "nb_pixel":
            assert ((isinstance(min_contour_size, tuple) or isinstance(min_contour_size, list)) and len(min_contour_size) == 2) \
               and ((isinstance(max_contour_size, tuple) or isinstance(max_contour_size, list)) and len(max_contour_size) == 2), \
                   "If using 'nb_pixel' contours method, min_contour_size and max_contour_size must be tuples or lists of size 2 (contours width, nb pixel)"

            self.contours = np.zeros(self.shadow.shape, dtype=np.bool_)

            # Linear function giving the contour width of a shadow according to its amount of pixel : the bigger the shadow, the wider the contour
            coef_contours = (max_contour_size[0] - min_contour_size[0]) / (max_contour_size[1] - min_contour_size[1])
            offset_contours = min_contour_size[0] - coef_contours * min_contour_size[1]
            contour_size_func = lambda nb_pixel: max(min_contour_size[0], min(coef_contours * nb_pixel + offset_contours, max_contour_size[0]))

        elif contours_method == "physics":
            assert isinstance(contours_width, int), "If using 'physics' contours method, contours_width must be an integer"
            assert (((isinstance(min_contour_size, tuple) or isinstance(min_contour_size, list)) and len(min_contour_size) == 2) or isinstance(min_contour_size, int)) \
               and (((isinstance(max_contour_size, tuple) or isinstance(max_contour_size, list)) and len(max_contour_size) == 2) or isinstance(max_contour_size, int)), \
                   "If using 'physics' contours method, min_contour_size and max_contour_size must be tuples or lists of size 2 (contours width, nb pixel), or integers"

            if isinstance(min_contour_size, tuple) or isinstance(min_contour_size, list):
                min_contour_size = min_contour_size[0]
            if isinstance(max_contour_size, tuple) or isinstance(max_contour_size, list):
                max_contour_size = max_contour_size[0]

            self.contours = np.zeros(self.shadow.shape, dtype=np.bool_)

            # Compute the slopes
            dem = xdem.DEM.from_array(self.elevation, crs='EPSG:4326', transform=(self.ORTHO_RESOLUTION * self.DEM_ORTHO_FACTOR, 0, np.min(self.lon), 
                                                                                  0, self.ORTHO_RESOLUTION * self.DEM_ORTHO_FACTOR, np.max(self.lat)), 
                                      nodata=-1.0)
            slopes = dem.slope()

            # Interpolate missing slope data
            coords_x, coords_y = np.meshgrid(np.arange(0, slopes.data.shape[0]), np.arange(0, slopes.data.shape[1]), indexing='ij')
            points = np.concatenate([coords_x[~slopes.data.mask][:, np.newaxis], 
                                     coords_y[~slopes.data.mask][:, np.newaxis]], axis=1)
            values = slopes.data[points[:,0], points[:,1]]
            points_interp = np.concatenate([coords_x[slopes.data.mask][:, np.newaxis], 
                                            coords_y[slopes.data.mask][:, np.newaxis]], axis=1)
            slopes.data[points_interp[:,0], points_interp[:,1]] = griddata(points, values, points_interp, method='nearest')
            del points_interp

            # Gaussian filter slopes to attenuate the impact of local steep slopes (crevasses and seracs)
            slopes.data = gaussian(slopes.data, sigma=10)

            # For future interpolation to a point
            points = (coords_x[:, 0], coords_y[0, :])

            # Shifting factors in the azimutal direction
            y_fac = np.cos(az - np.pi/2)
            x_fac = np.sin(az - np.pi/2)

            alt = np.radians(self.sun_position["alt"].degrees) # Sun altitude (in degrees)
            az = np.pi - az

        else:
            raise ValueError("contours_method must be one of ['constant', 'nb_pixel', 'physics']")

        # Coordinates of the image
        coords_x, coords_y = np.meshgrid(np.linspace(0, ortho_size_x, ortho_size_x+1),
                                         np.linspace(0, ortho_size_y, ortho_size_y+1), indexing='ij')
        coords_x = coords_x.astype(np.uint16)
        coords_y = coords_y.astype(np.uint16)

        # Block shadow removing on a single shadow (for parallelization purposes)
        def remove_shadow_block_single(val, bbox, shadows, contours, shadow, ortho, illumination):
            shad = (shadows == val)

            if contours_method == "nb_pixel":
                nb_pix_val = np.sum(shad)
                contour_width = int(contour_size_func(nb_pix_val)/2)
                select_x = slice(max(0, bbox[0]-contour_width), min(bbox[2]+contour_width, ortho_size_x))
                select_y = slice(max(0, bbox[1]-contour_width), min(bbox[3]+contour_width, ortho_size_y))
                contours[select_x, select_y] += (shad[select_x, select_y] ^ binary_erosion(shad[select_x, select_y], footprint=disk(contour_width))) \
                                              + (shad[select_x, select_y] ^ binary_dilation(shad[select_x, select_y], footprint=disk(contour_width)))
            
            elif contours_method == "physics":   
                shadow_len = []
                contour_size_list = []                
                thin_contours = shad ^ binary_erosion(shad, footprint=disk(1))
                it = np.nditer(thin_contours[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1], flags=['multi_index'])
                for cont in it:
                    if cont:
                        i, j = it.multi_index
                        i += bbox[0]
                        j += bbox[1]

                        #  The pixel is at the base of the shadow
                        if shad[int(max(0, min(i + 0.5 - 4*x_fac, ortho_size_x))), 
                                int(max(0, min(j + 0.5 - 4*y_fac, ortho_size_y)))]:

                            if i != 0 and i != ortho_size_x and j != 0 and j != ortho_size_y:
                                contours[skdraw.disk((i, j), contours_width, shape=contours.shape)] = True

                            x_az, y_az = max(0, min(i + 0.5 - x_fac, ortho_size_x)), max(0, min(j + 0.5 - y_fac, ortho_size_y))
                            while shadow[int(x_az), int(y_az)] and x_az != 0 and x_az != ortho_size_x and y_az != 0 and y_az != ortho_size_y:
                                x_az = max(0, min(x_az - x_fac, ortho_size_x))
                                y_az = max(0, min(y_az - y_fac, ortho_size_y))

                            if x_az != 0 and x_az != ortho_size_x and y_az != 0 and y_az != ortho_size_y:
                                x_az, y_az = x_az + x_fac, y_az + y_fac
                                length = np.sqrt((x_az - i - 0.5) ** 2 + (y_az - j - 0.5) ** 2) * self.ORTHO_RESOLUTION

                                alpha = np.radians(interpn(points, slopes.data, (x_az / self.DEM_ORTHO_FACTOR, y_az / self.DEM_ORTHO_FACTOR), method='nearest')[0])
                                contour_size = abs((np.tan(alt + self.SUN_WIDTH/2) - np.tan(alpha)) / 
                                                   (np.tan(alt - self.SUN_WIDTH/2) - np.tan(alpha)) - 1) * length / self.ORTHO_RESOLUTION
                                # if contour_size > max_contour_size:
                                #     print(f"(x, y) : ({int(x_az)}, {int(y_az)}) ; size={contour_size} ; alpha={round(np.degrees(alpha), 1)} ; length={length}")
                                contour_size_list.append(contour_size)
                                contours[skdraw.disk((int(x_az), int(y_az)), max(min_contour_size, min(contour_size, max_contour_size)), shape=contours.shape)] = True
                                shadow_len.append(length)

            bbox = [max(0, bbox[0]-bbox_buffer), max(0, bbox[1]-bbox_buffer), 
                    min(bbox[2]+bbox_buffer, ortho_size_x), min(bbox[3]+bbox_buffer, ortho_size_y)] # Buffered box around shadow
            nb_step_x = (bbox[2] - bbox[0]) // bbox_max_size_x
            nb_step_y = (bbox[3] - bbox[1]) // bbox_max_size_y
            if nb_step_x + nb_step_y > 0:
                step = int((bbox[3] - bbox[1]) / (nb_step_y + 1))
                xmax = bbox[2] - bbox[0]    
                shift = abs(xmax * np.tan(az))
                if az >= 0:
                    y_i = np.array([bbox[1] - shift] + [bbox[1] - nb_step_x*step + n*step for n in range(nb_step_x + nb_step_y)] + [bbox[3] - step, bbox[3]])
                else:
                    y_i = np.array([bbox[1], bbox[1] + step] + [bbox[3] - nb_step_y*step + n*step for n in range(nb_step_x + nb_step_y)] + [bbox[3] + shift])
                y_diff = np.diff(y_i)
                
                for start in range(len(y_i)-1):
                    mask = coords_y - shift / xmax * (coords_x - bbox[0]) - y_i[start]
                    mask = (mask >= 0) & (mask < y_diff[start]) & (coords_x >= bbox[0]) & \
                        (coords_x <= bbox[2]) & (coords_y >= bbox[1]) & (coords_y <= bbox[3])
                    
                    if np.sum(shad & mask) != 0:
                        # Approach the histogram of the shadow area by a Gaussian
                        hist_s, _ = histogram(ortho[shad & mask])
                        hist_s = hist_s.astype(np.float32)
                        hist_s /= len(ortho[shad & mask])
                        gaussian_s = [np.max(hist_s), np.argmax(hist_s) + np.min(ortho[shad & mask]), 
                                    np.std(ortho[shad & mask & (ortho <= self.SHADOW_MAX_VALUE)])]
                        
                        # Median of shadowed and non-shadowed areas within the current block and outside contours
                        m = np.nanmedian(ortho[shad & ~contours & mask & (ortho <= self.SHADOW_MAX_VALUE)])
                        M = np.nanmedian(ortho[~shadow & ~contours & mask & (ortho > (gaussian_s[1] + 2*abs(gaussian_s[2])))])

                        illumination[shad & ~contours & mask] = M/m

            else:
                mask = (coords_x >= bbox[0]) & (coords_x <= bbox[2]+1) & (coords_y >= bbox[1]) & (coords_y <= bbox[3]+1)

                # Approach the histogram of the shadow area by a Gaussian
                hist_s, _ = histogram(ortho[shad])
                hist_s = hist_s.astype(np.float32)
                hist_s /= len(ortho[shad])
                gaussian_s = [np.max(hist_s), np.argmax(hist_s) + np.min(ortho[shad]), 
                              np.std(ortho[shad & mask & (ortho <= self.SHADOW_MAX_VALUE)])]
                
                # Median of shadowed and non-shadowed areas within the studied shadow and outside contours
                m = np.nanmedian(ortho[shad & ~contours & mask & (ortho <= self.SHADOW_MAX_VALUE)])
                M = np.nanmedian(ortho[~shadow & ~contours & mask & (ortho > (gaussian_s[1] + 2*abs(gaussian_s[2])))])

                illumination[shad & ~contours] = M/m

        # Compute bbox outside of remove_shadow_constant_single() function because it is not picklable
        iterable = np.unique(shadows)[1:]
        props = regionprops(shadows)
        bboxs = []
        for val in iterable:
            bboxs.append(props[val-1].bbox) # Box around shadow

        # Apply to every individual shadow
        if self.verbose: iterable = tqdm(iterable)
        if parallelize != False: # Parallelize : not necessarily faster here...
            # Generate memory maps (shared memory) for faster computation
            shadows = generate_memmap(shadows, "shadows", folder=joblib_folder)
            self.contours = generate_memmap(self.contours, "contours", folder=joblib_folder)
            self.shadow = generate_memmap(self.shadow, "shadow", folder=joblib_folder)
            self.ortho = generate_memmap(self.ortho, "ortho", folder=joblib_folder)
            self.illumination = generate_memmap(self.illumination, "illumination", folder=joblib_folder)

            Parallel(n_jobs=parallelize, verbose=0)(
                delayed(remove_shadow_block_single)(val, bboxs[val-1], shadows, self.contours, self.shadow, self.ortho, self.illumination)
            for val in iterable) # Parallel computation

            for memmap_file in glob.glob(os.path.join(joblib_folder, "*")): # Delete memmap files
                os.remove(memmap_file)

        else:
            for val in iterable: # Sequential computation
                remove_shadow_block_single(val, bboxs[val-1], shadows, self.contours, self.shadow, self.ortho, self.illumination)

        # Smooth the illumination factor within the shadows
        weigths = gaussian(self.shadow & ~self.contours, sigma=bbox_max_size_y//2)
        self.illumination = np.where((~self.contours) & self.shadow, gaussian(np.where(self.shadow & ~self.contours, self.illumination, 0), 
                                                                              sigma=bbox_max_size_y//2) / weigths, self.illumination)

        # Remove inside shadows
        maxi = np.max(self.ortho)
        self.shadow_free = np.where((self.illumination > 0) & (self.illumination * self.ortho <= maxi), self.illumination * self.ortho, self.ortho)
        self.shadow_free[self.contours] = 0

        # Sharpen the shadow areas (increase contrast)
        #TODO find something better
        # inter = 3 * self.shadow_free - 2 * gaussian(self.shadow_free, sigma=2)
        # inter[inter > maxi] = maxi
        # inter[inter < 0] = 0
        # self.shadow_free = np.where(self.shadow, inter.round().astype(np.uint8), self.shadow_free)

        if self.verbose: print(f"Block removing inside shadows took {round(time.time() - start_time, 2)} s")

    #endregion
    #%% ---------------------------------------------------------------------------
    #region                          CONTOURS REMOVAL
    # -----------------------------------------------------------------------------

    def gaussian_filtering_contours(self, first_ortho_sigma=5, second_ortho_sigma=2, second_ortho_buffer=2):
        if self.verbose: start_time = time.time()

        self.shadow_free = np.where(self.contours, gaussian(self.shadow_free, sigma=first_ortho_sigma), self.shadow_free)
        self.shadow_free = np.where(self.contours ^ binary_dilation(self.contours, footprint=disk(second_ortho_buffer)), 
                                    gaussian(self.shadow_free, sigma=second_ortho_sigma), self.shadow_free)
        
        if self.verbose: print(f"Gaussian filtering inside contours took {round(time.time() - start_time, 2)} s")
        
    def median_filtering_contours(self, win_size=50):
        if self.verbose: start_time = time.time()

        win_size = int(win_size / 2)
        self.shadow_free = np.where(self.contours, median_filter(self.shadow_free, size=win_size), self.shadow_free)

        if self.verbose: print(f"Median filtering inside contours took {round(time.time() - start_time, 2)} s")

    def interpolation_in_contours(self, around_contours_buffer=10, interpolation_method='linear'):
        if self.verbose: start_time = time.time()

        # Coordinates of the image
        ortho_size_x, ortho_size_y = self.ortho.shape[0]-1, self.ortho.shape[1]-1
        coords_x, coords_y = np.meshgrid(np.linspace(0, ortho_size_x, ortho_size_x+1),
                                         np.linspace(0, ortho_size_y, ortho_size_y+1), indexing='ij')
        coords_x = coords_x.astype(np.uint16)
        coords_y = coords_y.astype(np.uint16)

        around_contours = self.contours ^ binary_dilation(self.contours, footprint=disk(around_contours_buffer))
        points = np.concatenate([coords_x[around_contours][:, np.newaxis], coords_y[around_contours][:, np.newaxis]], axis=1)
        values = self.shadow_free[points[:,0], points[:,1]]
        points_interp = np.concatenate([coords_x[self.contours][:, np.newaxis], 
                                        coords_y[self.contours][:, np.newaxis]], axis=1)
        self.shadow_free[points_interp[:,0], points_interp[:,1]] = griddata(points, values, points_interp, method=interpolation_method)
        del points, values, points_interp

        if self.verbose: print(f"Interpolation inside contours took {round(time.time() - start_time, 2)} s")

    #endregion
    #%% ---------------------------------------------------------------------------
    #region                            PLOT METHODS
    # -----------------------------------------------------------------------------

    def plot_dem(self,
                 ax=None, 
                 plot_mode='colormesh',
                 cmap='terrain', 
                 equidistance=50, 
                 alt_range=[0, 4800], 
                 savefig=None,
                 title="Elevation"):
        
        '''
        
        '''
        
        if self.elevation is None:
            assert self.elevation_pad is not None, "DEM was not loaded, nothing to plot"
            dem_plot = np.ma.masked_where(self.mask_outliers, self.elevation_pad)
            lon = self.lon_pad
            lat = self.lat_pad
        else:
            dem_plot = np.ma.masked_where(self.mask_outliers, self.elevation)
            lon = self.lon
            lat = self.lat

        fig = None
        if ax == None:
            fig, ax = plt.subplots(figsize=(20, 12))

        #TODO add a colorbar
        ax.set_facecolor(plt.get_cmap(cmap)(0.15)[:3] + (0.25,))
        levels = np.arange(alt_range[0], alt_range[1]+1, equidistance)
        cmap = colors.LinearSegmentedColormap.from_list(
            cmap, plt.get_cmap(cmap)(np.linspace(0.25, 1.0, 1000)))
        try: norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
        except ValueError: norm = None #TODO manage vmin, vmax if ncolors exceeds cmap.N and norm=None

        if plot_mode == 'colormesh': 
            ax.pcolormesh(lon, lat, dem_plot, cmap=cmap, norm=norm)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylim(lat.min(), lat.max())
        elif plot_mode == 'imshow': 
            ax.imshow(dem_plot, cmap=cmap, norm=norm)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)

    def plot_ortho(self, 
                   ax=None, 
                   plot_mode='colormesh',
                   savefig=None, 
                   cmap='Greys_r', 
                   vmin=0, vmax=255, 
                   figsize=(20, 12), 
                   title="Original orthoimage"):
        
        '''

        '''

        assert self.ortho is not None, "Orthoimage was not loaded, nothing to plot"

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if plot_mode == 'colormesh':
            ax.pcolormesh(self.lon_ortho, self.lat_ortho, np.ascontiguousarray(self.ortho), cmap=cmap, vmin=vmin, vmax=vmax)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            ax.imshow(self.ortho, cmap=cmap, vmin=vmin, vmax=vmax)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)
            
    def plot_shadow_free(self, 
                         ax=None, 
                         savefig=None,
                         plot_mode='colormesh', 
                         cmap='Greys_r', 
                         vmin=0, vmax=255, 
                         figsize=(20, 12), 
                         title="Shadow-free orthoimage"):
        
        '''

        '''

        assert self.shadow_free is not None, "Shadows were not removed from orthoimage, nothing to plot"

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if plot_mode == 'colormesh':
            ax.pcolormesh(self.lon_ortho, self.lat_ortho, np.ascontiguousarray(self.shadow_free), cmap=cmap, vmin=vmin, vmax=vmax)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            ax.imshow(self.shadow_free)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)

    def plot_grad_ortho(self, 
                        background=None, 
                        ax=None, 
                        plot_mode='colormesh',
                        savefig=None, 
                        figsize=(20, 12), 
                        cmap='inferno', 
                        vmin=None, vmax=None, 
                        alpha=0.5, 
                        title="Orthoimage gradient", 
                        grad_kwargs={}, 
                        background_kwargs={}):
        
        '''
        
        '''

        if self.grad is None:
            assert self.ortho is not None, "Orthoimage was not loaded, thus orthoimage gradient cannot be computed"
            self.grad = compute_grad(self.ortho, **grad_kwargs)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if background == 'ortho':
            self.plot_ortho(ax=ax, plot_mode=plot_mode, **background_kwargs)
        elif background == 'shadow_free':
            self.plot_shadow_free(ax=ax, plot_mode=plot_mode, **background_kwargs)
        
        if plot_mode == 'colormesh':
            ax.pcolormesh(self.lon_ortho, self.lat_ortho, np.ascontiguousarray(self.grad), cmap=cmap, 
                            vmin=vmin, vmax=vmax, alpha=alpha if background is not None else 1)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            ax.imshow(np.ascontiguousarray(self.grad), cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha if background is not None else 1)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)
            
    def plot_grad_shadow_free(self,
                              background=None,
                              ax=None,
                              plot_mode='colormesh',
                              savefig=None,
                              figsize=(20, 12),
                              cmap='inferno',
                              vmin=None, vmax=None,
                              alpha=0.5,
                              title="Shadow-free image gradient",
                              grad_kwargs={},
                              background_kwargs={}):
        
        '''
        
        '''
        
        shadow_free_grad = compute_grad(self.shadow_free, **grad_kwargs)
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if background == 'ortho':
            self.plot_ortho(ax=ax, plot_mode=plot_mode, **background_kwargs)
        elif background == 'shadow_free':
            self.plot_shadow_free(ax=ax, plot_mode=plot_mode, **background_kwargs)
        
        if plot_mode == 'colormesh':
            ax.pcolormesh(self.lon_ortho, self.lat_ortho, np.ascontiguousarray(shadow_free_grad), cmap=cmap, 
                        vmin=vmin, vmax=vmax, alpha=alpha if background is not None else 1)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            ax.imshow(np.ascontiguousarray(shadow_free_grad), cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha if background is not None else 1)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)

    def plot_shadow(self, 
                    background: str = 'dem', 
                    ax=None, 
                    plot_mode='colormesh',
                    savefig=None, 
                    figsize=(20, 12), 
                    color='magenta', 
                    alpha=0.3, 
                    title="DEM-casted shadows", 
                    background_kwargs={}):
        
        '''
        Plot the detected shadows.

        Parameters
        ----------
          --> 
        
        '''

        assert self.shadow is not None, "Shadows were not casted, nothing to plot"

        fig = None
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)

        if background == 'ortho':
            self.plot_ortho(ax=ax, plot_mode=plot_mode, **background_kwargs)
            shadow = resize(self.shadow, self.ortho.shape)
            lon = self.lon_ortho
            lat = self.lat_ortho
        elif background == 'dem':
            self.plot_dem(ax=ax, plot_mode=plot_mode, **background_kwargs)
            shadow = resize(self.shadow, self.elevation.shape)
            lon = self.lon
            lat = self.lat
        else:
            shadow = self.shadow
            lon = self.lon if self.shadow.shape[0] == self.elevation.shape[0] else self.lon_ortho
            lat = self.lat if self.shadow.shape[1] == self.elevation.shape[1] else self.lat_ortho

        cmap_s = mpl.colors.ListedColormap([color, 'blue'])
        shadow_plot = np.ma.masked_where(shadow == 0, shadow)
        if plot_mode == 'colormesh':
            ax.pcolormesh(lon, lat, shadow_plot, cmap=cmap_s, 
                        alpha=alpha if background is not None else 1)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylim(lat.min(), lat.max())
        elif plot_mode == 'imshow':
            ax.imshow(shadow_plot, cmap=cmap_s, alpha=alpha if background is not None else 1)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)

    def plot_shadows(self, 
                     background: str = 'dem', 
                     ax=None, 
                     plot_mode='colormesh',
                     savefig=None, 
                     figsize=(20, 12), 
                     cmap='nipy_spectral',
                     alpha=0.3,
                     number=True,
                     title="Indidual DEM-casted shadows", 
                     background_kwargs={}):
        
        '''
        Plot individual shadows.

        Parameters
        ----------
          --> 
        
        '''

        assert self.shadow is not None, "Shadows were not casted, nothing to plot"

        fig = None
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)

        if background == 'ortho':
            self.plot_ortho(ax=ax, plot_mode=plot_mode, **background_kwargs)
            shadow = resize(self.shadow, self.ortho.shape)
            lon = self.lon_ortho
            lat = self.lat_ortho
        elif background == 'dem':
            self.plot_dem(ax=ax, plot_mode=plot_mode, **background_kwargs)
            shadow = resize(self.shadow, self.elevation.shape)
            lon = self.lon
            lat = self.lat
        else:
            shadow = self.shadow
            lon = self.lon
            lat = self.lat

        shadows_plot = label(shadow, background=0, connectivity=1)
        if plot_mode == 'colormesh':
            ax.pcolormesh(lon, lat, shadows_plot, cmap=cmap, 
                        alpha=alpha if background is not None else 1)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            ax.imshow(shadows_plot, cmap=cmap, alpha=alpha if background is not None else 1)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)
        
        if number:
            props = regionprops(shadows_plot)
            for val in np.unique(shadows_plot)[1:]:
                bbox = props[val-1].bbox
                ax.text(lon[(bbox[1] + bbox[3]) // 2], lat[(bbox[0] + bbox[2]) // 2], str(val), fontsize=10, color='red')

        if fig != None and savefig != None:
            fig.savefig(savefig)
            
    def plot_shadow_contours(self,
                             background='ortho',
                             ax=None,
                             plot_mode='colormesh',
                             savefig=None,
                             figsize=(20, 12),
                             color='red',
                             alpha=0.5,
                             title="Shadow contours",
                             background_kwargs={}):
        
        assert self.contours is not None, "Contours were not computed, nothing to plot"
        
        fig = None
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)
                
        resize_to_dem = False
        if 'shadow' in background:
            if 'ortho' in background:
                self.plot_shadow(ax=ax, plot_mode=plot_mode, background='ortho', **background_kwargs)
                
            elif 'dem' in background:
                self.plot_shadow(ax=ax, plot_mode=plot_mode, background='dem', **background_kwargs)
                resize_to_dem = True
            else:
                self.plot_shadow(ax=ax, plot_mode=plot_mode, background=None, **background_kwargs)
        elif 'ortho' in background:
            self.plot_ortho(ax=ax, plot_mode=plot_mode, **background_kwargs)
        elif 'dem' in background:
            self.plot_dem(ax=ax, plot_mode=plot_mode, **background_kwargs)
            resize_to_dem = True
        else:
            background = None
            
        if resize_to_dem:
            contours = resize(self.contours, self.elevation.shape)
            lon = self.lon
            lat = self.lat
        else:
            contours = self.contours
            lon = self.lon_ortho
            lat = self.lat_ortho
        
        cmap_c = mpl.colors.ListedColormap([color, 'blue'])
        contours_plot = np.ma.masked_where(contours == 0, contours)
        if plot_mode == 'colormesh':
            ax.pcolormesh(lon, lat, contours_plot, cmap=cmap_c, 
                            alpha=alpha if background is not None else 1)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            ax.imshow(contours_plot, cmap=cmap_c, alpha=alpha if background is not None else 1)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)
        
        if fig != None and savefig != None:
            fig.savefig(savefig)

    def plot_shadow_map(self,
                        shadow_map,
                        background:str='dem',
                        ax=None,
                        plot_mode='colormesh',
                        savefig=None,
                        figsize=(20, 12),
                        cmap='viridis',
                        alpha=0.5,
                        title="Shadow map",
                        cbar_label="Nb days in shadow",
                        background_kwargs={}):
        """

        :param shadow_map:
        :param background: [str] 'dem', 'ortho', 'shadow_free' or None
        :param ax:
        :param plot_mode:
        :param savefig:
        :param figsize:
        :param cmap:
        :param alpha:
        :param title:
        :param cbar_label:
        :param background_kwargs:
        :return:
        """
        
        fig = None
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)

        if background == 'ortho':
            shadow_map = shadow_map.astype(np.float16) / 365.0
            self.plot_ortho(ax=ax, plot_mode=plot_mode, **background_kwargs)
            shadow_map = (resize(shadow_map, self.ortho.shape) * 365).astype(np.int16)
            lon = self.lon_ortho
            lat = self.lat_ortho
        elif background == 'dem':
            shadow_map = shadow_map.astype(np.float16) / 365.0
            self.plot_dem(ax=ax, plot_mode=plot_mode, **background_kwargs)
            shadow_map = (resize(shadow_map, self.elevation.shape) * 365).astype(np.int16)
            lon = self.lon
            lat = self.lat
        else:
            lon = self.lon if shadow_map.shape[0] == self.elevation.shape[0] else self.lon_ortho
            lat = self.lat if shadow_map.shape[1] == self.elevation.shape[1] else self.lat_ortho

        shadow_map_plot = np.ma.masked_where(shadow_map == 0, shadow_map)
        if plot_mode == 'colormesh':
            ax.pcolormesh(lon, lat, shadow_map_plot, cmap=cmap, 
                        alpha=alpha if background is not None else 1)
            x_ticks = np.arange(self.domain_outer['lon_min'], self.domain_outer['lon_max'], 0.02)
            ax.set_xticks(x_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
            y_ticks = np.arange(self.domain_outer['lat_min'], self.domain_outer['lat_max'], 0.02)
            ax.set_yticks(y_ticks, ["%.1f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
            ax.set_xlim(self.lon_ortho.min(), self.lon_ortho.max())
            ax.set_ylim(self.lat_ortho.min(), self.lat_ortho.max())
        elif plot_mode == 'imshow':
            pcm = ax.imshow(shadow_map_plot, cmap=cmap, alpha=alpha if background is not None else 1)
            if fig != None:
                fig.colorbar(pcm, ax=ax, label=cbar_label)
            #TODO set_xticks, set_yticks
        else: 
            raise ValueError("plot_mode parameter must be 'colormesh' or 'imshow'")
        ax.set_title(title)

        if fig != None and savefig != None:
            fig.savefig(savefig)

    #endregion
    #%% ---------------------------------------------------------------------------
    #region                          ADDITIONAL METHODS
    # -----------------------------------------------------------------------------

