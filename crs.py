import geopandas as gpd
import rasterio

# load shapefile
gdf = gpd.read_file("dataset/labels/Built_up_Area_type.shp")

# open orthophoto
with rasterio.open("dataset/images/SAMLUR.tif") as src:
    print("Image CRS:", src.crs)

print("Shapefile CRS:", gdf.crs)