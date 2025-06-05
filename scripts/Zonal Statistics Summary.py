import geopandas as gpd
from rasterstats import zonal_stats
import os

# Input files
polygon_fp = "path"
raster_fp = "path"
output_fp = "output\path"

# Load polygons
gdf = gpd.read_file(polygon_fp)

# Calculate zonal stats
stats = zonal_stats(
    vectors=gdf,
    raster=raster_fp,
    stats=["mean", "min", "max", "median", "std"],
    geojson_out=False,
    nodata=-9999
)

# Attach stats to gdf
for key in stats[0].keys():
    gdf[f"ndvi_{key}"] = [s[key] for s in stats]

# Export result
gdf.to_file(output_fp)

print("Zonal statistics complete and saved to", output_fp)