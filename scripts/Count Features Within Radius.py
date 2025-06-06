import geopandas as gpd
import shapely.geometry
import Point
from tqdm import tqdm

# Parameters
target_fp = "data/schools.shp"
source_fp = "data/parks.shp"
radius_meters = 1000
output_fp = "output/schools_with_park_counts.shp"

# Load GeoDataFrames
target_gdf = gpd.read_file(target_fp)
source_gdf = gpd.read_file(source_fp)

# Ensure both are in the same projected CRS
if not target_gdf.crs.is_projected:
    raise ValueError("Please reproject your layers to a projected CRS for distance calculations")

source_gdf = source_gdf.to_crs(target_gdf.to_crs)

# Create spatial index for source layer
source_sindex = source_gdf.sindex

# Count features within radius
counts = []
print(f"Counting source features within {radius_meters} meters of each target feature...")

for idx, target_row in tqdm(target_gdf.iterrows(), total=len(target_gdf)):
    buffer_geom = target_row.geometry.buffer(radius_meters)
    possible_matches_index = list(source_sindex.intersection(buffer_geom.bounds))
    possible_matches = source_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(buffer_geom)]
    counts.append(len(precise_matches))

# Add result to the target GeoDataFrame
target_gdf[f"count_{source_gdf.geometry.name}"] = counts

# Save output
target_gdf.to_file(output_fp)
print(f"Completed. Output saved to: {output_fp}")
                               