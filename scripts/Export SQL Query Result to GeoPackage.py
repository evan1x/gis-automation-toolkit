import geopandas as gpd
import duckdb

result = duckdb.sql("SELECT * FROM 'roads.parquet' WHERE speed > 60").df()
gdf_result = gpd.GeoDataFrame(result, geometry='geometry')
gdf_result.to_file("fast_roads.gpkg", driver="GPKG")

