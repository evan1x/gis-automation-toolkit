import geopandas as gpd

# Convert Shapefile to GeoJSON
gdf = gpd.read_file("input.shp") # Replace with your shapefile path
gdf.to_file("output.geojson", driver="GeoJSON") 