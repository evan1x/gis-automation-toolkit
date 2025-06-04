import geopandas as gpd

gdf = gpd.read_file("file.shp")
gdf_clean = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
gdf_clean.to_file("cleaned.shp") 
