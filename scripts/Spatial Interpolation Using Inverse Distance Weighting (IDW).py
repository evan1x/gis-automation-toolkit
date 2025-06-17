import geopandas as gpd
import duckdb
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree

# Load dataset
gdf = gpd.read_file("sensor_points.geojson")  # Assumes points with 'value' field

# Connect to DuckDB and run SQL filter
con = duckdb.connect()
con.register("gdf", gdf)
filtered_df = con.execute("""
    SELECT * FROM gdf WHERE value IS NOT NULL
""").fetchdf()

# Convert to GeoDataFrame again
points_gdf = gpd.GeoDataFrame(filtered_df, geometry="geometry", crs=gdf.crs)

# Create grid for interpolation
xmin, ymin, xmax, ymax = points_gdf.total_bounds
grid_spacing = 100  # meters

x_coords = np.arange(xmin, xmax, grid_spacing)
y_coords = np.arange(ymin, ymax, grid_spacing)
grid_points = [Point(x, y) for x in x_coords for y in y_coords]

# Build KDTree from known points
known_coords = np.array([[pt.x, pt.y] for pt in points_gdf.geometry])
known_values = points_gdf["value"].values
tree = cKDTree(known_coords)

# Perform IDW
def idw(xy, power=2):
    distances, idxs = tree.query(xy, k=6)
    if np.any(distances == 0):
        return known_values[idxs[distances == 0][0]]
    weights = 1 / distances**power
    return np.sum(weights * known_values[idxs]) / np.sum(weights)

interpolated_values = [idw((pt.x, pt.y)) for pt in grid_points]

# Output as GeoDataFrame
interpolated_gdf = gpd.GeoDataFrame({
    "value": interpolated_values,
    "geometry": grid_points
}, crs=gdf.crs)

interpolated_gdf.to_file("interpolated_idw_points.geojson", driver="GeoJSON")
