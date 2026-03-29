import rasterio
import numpy as np
import cv2
import os

image_path = "dataset/images/SAMLUR.tif"
output_folder = "dataset/tiles"

os.makedirs(output_folder, exist_ok=True)

tile_size = 720

useful = 0
removed = 0

with rasterio.open(image_path) as src:

    for y in range(0, src.height, tile_size):
        for x in range(0, src.width, tile_size):

            window = rasterio.windows.Window(x, y, tile_size, tile_size)

            tile = src.read(window=window)

            if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                continue

            tile = np.transpose(tile,(1,2,0))

            tile = cv2.normalize(tile,None,0,255,cv2.NORM_MINMAX)
            tile = tile.astype(np.uint8)

            gray = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)

            mean = np.mean(gray)
            std = np.std(gray)

            # filter useless tiles
            if std > 10 and mean > 20 and mean < 235:

                name = f"tile_{y}_{x}.jpg"

                cv2.imwrite(os.path.join(output_folder,name),tile)

                useful += 1

            else:
                removed += 1

print("Useful tiles saved:", useful)
print("Waste tiles removed:", removed)