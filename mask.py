import os
import cv2
import numpy as np
import geopandas as gpd

tiles_folder = "dataset/tiles"
mask_folder = "dataset/masks"
filtered_tiles_folder = "dataset/train_images"

shp_path = "dataset/labels/Built_up_Area_type.shp"

os.makedirs(mask_folder, exist_ok=True)
os.makedirs(filtered_tiles_folder, exist_ok=True)

gdf = gpd.read_file(shp_path)

kept = 0
skipped = 0

for tile_name in os.listdir(tiles_folder):

    tile_path = os.path.join(tiles_folder, tile_name)
    image = cv2.imread(tile_path)

    if image is None:
        continue

    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for geom in gdf.geometry:

        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords).astype(int)
            cv2.fillPoly(mask, [coords], 1)

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords).astype(int)
                cv2.fillPoly(mask, [coords], 1)

    # check if mask contains useful pixels
    if np.sum(mask) == 0:
        skipped += 1
        continue

    mask_name = tile_name.replace(".jpg", ".png")

    cv2.imwrite(os.path.join(mask_folder, mask_name), mask * 255)
    cv2.imwrite(os.path.join(filtered_tiles_folder, tile_name), image)

    kept += 1

print("Useful samples:", kept)
print("Skipped empty masks:", skipped)