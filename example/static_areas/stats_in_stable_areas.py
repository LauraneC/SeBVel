from ticoi.cube_data_classxr import CubeDataClass
import glob
import xarray as xr

cube_folder = "/home/charriel/Documents/PostdocIGE/Seasonality/Cubes_Alps/cubes_S2_non_filter/"
cube_name_list = glob.glob(f'{cube_folder}/*.nc')
mask_file = f'/home/charriel/Documents/PostdocIGE/Form@ter/Metadata/RGI/Alpes_RGI7.shp'

list_mask = []
cube = CubeDataClass()
cube.load(cube_name_list[0])
list_mask.append(cube.mask_cube(mask_file, invert=True))
for cube_name_el in cube_name_list[1:]:
    cube = CubeDataClass()
    cube.load(cube_name_el)
    mask = cube.mask_cube(mask_file, invert=True)
    list_mask.append(mask)

t = xr.concat(list_mask,dim="mid_date")

t_mean_time = t.mean(dim=["x", "y"], skipna=True)
t_mean_time['vx'].plot.scatter()
t_mean_time.to_netcdf("t_mean_time.nc")

