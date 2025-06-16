import geopandas as gpd
from rasterstats import zonal_stats
import duckdb

zones = gpd.read_file("zones.shp")
stats = zonal_stats("input.tif", zones, stats=["mean", "min", "max"], geojson_out=True)
zones_stats = gpd.GeoDataFrame.from_features(stats)

duckdb.register("zstats", zones_stats)
result = duckdb.query("""
                      SELECT category, AVG(mean) as avg_elevation
                      FROM zstats
                      GROUP BY category
                      """).to_df()

result.to_csv("zonal_summary.csv")

