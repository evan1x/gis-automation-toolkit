import geopandas as gpd
import duckdb

gdf = gpd.read_file("buildings.shp")
gdf.to_parquet("buildings.parquet")

con = duckdb.connect()
result = con.execute("""
    SELECT * FROM 'buildings.parquet' WHERE area > 500 AND type = 'residential'
""").fetchdf()

gdf_result = gpd.GeoDataFrame(result, geometry='geometry')
gdf_result.to_file("filtered_buildings.shp")

