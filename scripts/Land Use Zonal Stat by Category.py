from rasterstats import zonal_stats
import geopandas as gpd
import duckdb

land_use = gpd.read_file("land_use.shp")
stats = zonal_stats("ndvi.tif", land_use, stats="mean", geojson_out=True)

gdf_stats = gpd.GeoDataFrame.from_features(stats)

duckdb.register("zonal", gdf_stats)
query = "SELECT category, AVG(mean) as avg_ndvi FROM zonal GROUP BY category"
result = duckdb.sql(query).df()

print(result)

