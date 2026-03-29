import os
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import cv2
from rasterio.windows import Window
from shapely.geometry import box
from rasterio.features import rasterize

tif_path = "dataset/images/SAMLUR.tif"

shapefiles = [
    "dataset/labels/Built_up_Area_Type.shp",
    "dataset/labels/Road.shp",
    "dataset/labels/Water_Body.shp"
]

os.makedirs("dataset/train_images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

tile_size = 512
tile_id = 0

with rasterio.open(tif_path) as src:

    width = src.width
    height = src.height
    transform = src.transform

    # Load shapefiles
    gdfs = []
    for shp in shapefiles:

        gdf = gpd.read_file(shp)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:3857")

        gdf = gdf.to_crs(src.crs)

        gdfs.append(gdf)

    gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):

            window = Window(x, y, tile_size, tile_size)
            img = src.read([1,2,3], window=window)

            tile_transform = src.window_transform(window)

            bounds = rasterio.windows.bounds(window, transform)
            tile_geom = box(*bounds)

            intersects = gdf_all[gdf_all.intersects(tile_geom)]

            if len(intersects) == 0:
                continue

            mask = rasterize(
                [(geom,1) for geom in intersects.geometry],
                out_shape=(tile_size,tile_size),
                transform=tile_transform,
                fill=0,
                dtype=np.uint8
            )

            img = np.transpose(img,(1,2,0))

            cv2.imwrite(f"dataset/train_images/tile_{tile_id}.jpg", img)
            cv2.imwrite(f"dataset/masks/tile_{tile_id}.png", mask*255)

            tile_id += 1

            if tile_id % 100 == 0:
                print("Saved tiles:", tile_id)

print("Dataset created:", tile_id)