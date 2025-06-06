import geopandas as gpd
import shapely.geometry
import box
import math

# Parameters
minx, miny, maxx, maxy = 0, 0, 5000, 3000
cell_width = 1000
cell_height = 1000
crs = "EPSG:32633" # Example

# Generate grid
cols = int(math.ceil((maxx - minx) / cell_width))
rows = int(math.ceil((maxy - miny) / cell_height))

grid_cells = []

for i in range(cols):
    for j in range(rows):
        x1 = minx + i * cell_width
        y1 = miny + j * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        grid_cells.append(box(x1, y1, x2, y2))

# Convert to GeoDataFrame
grid = gpd.GeoDataFrame({"geometry": grid_cells})
grid.set_crs(crs, inplace=True)

# Save to output
grid.to_file("fishnet_grid.shp")

print(f"Created {len(grid)} grid cells and saved to fishnet_grid.shp")
