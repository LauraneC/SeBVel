import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

input_path = "/home/charrierl/Documents/Collaborations/Ariane/shadow/COP30_NWArg_PS.tif"
output_path = "/home/charrierl/Documents/Collaborations/Ariane/shadow/COP30_NWArg_PS_downscaled.tif"

upscale_factor = 1/10

with rasterio.open(input_path) as src:

    # Read and upscale
    data = src.read(
        out_shape=(
            src.count,
            int(src.height * upscale_factor),
            int(src.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # Compute new transform
    scale_x = src.width / data.shape[-1]
    scale_y = src.height / data.shape[-2]
    new_transform = src.transform * Affine.scale(scale_x, scale_y)

    # Save the upscaled raster
    profile = src.profile
    profile.update({
        "height": data.shape[1],
        "width": data.shape[2],
        "transform": new_transform
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

print("Upscaled image saved to:", output_path)
