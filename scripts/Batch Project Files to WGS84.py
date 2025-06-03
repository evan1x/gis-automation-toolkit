import geopandas as gpd
import os

folder = "path_to_your_folder"
for file in os.listdir(folder):
    if file.endswith(".shp"):
        shp = gpd.read_file(os.path.join(folder, file))
        shp = shp.to_crs("EPSG:4326")
        shp.to_file(os.path.join(folder, f"{file[:-4]}_wgs84.shp"))
