import rasterio
import numpy as np
from rasterio.transform import from_origin

data = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
transform = from_origin(0, 0, 1, 1)

with rasterio.open(
    'valid_crs.tif',
    'w',
    driver='GTiff',
    height=256,
    width=256,
    count=3,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
) as dst:
    dst.write(data)
