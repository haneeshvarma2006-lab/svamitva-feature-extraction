import geopandas as gpd
import rasterio
import numpy as np
import cv2

image_path = "dataset/images/SAMLUR.tif"
shp_path = "dataset/labels/Built_up_Area_type.shp"

image = rasterio.open(image_path)
gdf = gpd.read_file(shp_path)

mask = np.zeros((image.height, image.width), dtype=np.uint8)

for geom in gdf.geometry:
    coords = np.array(geom.exterior.coords)
    coords = coords.astype(int)
    cv2.fillPoly(mask, [coords], 1)

cv2.imwrite("masks/image1.png", mask*255)

print("Mask created")