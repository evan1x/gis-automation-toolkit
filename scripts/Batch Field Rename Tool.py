import geopandas as gpd
import os

folder = r"C:\GIS\shapefiles" # Replace with your path
field_map = {"FID": "id", "Shape_Area": "area"}

for file in os.listdir(folder):
    if file.endswith(".shp"):
        gdf = gpd.read_file(os.path.join(folder, file))
        gdf.rename(columns=field_map, inplace=True)
        gdf.to_file(os.path.join(folder, file))

        