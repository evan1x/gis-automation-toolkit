import geopandas as gpd
import duckdb

gdf = gpd.read_file("data.shp")
gdf["centroid"] = gdf.geometry.centroid

duckdb.register("gdf", gdf)

query = """
SELECT *, ST_Distance(geometry, centroid) as dist
FROM gdf
WHERE dist > (
    SELECT AVG(ST_Distance(geometry, centroid)) + 2 * STDDEV_POP(ST_Distance(geometry, centroid))
    FROM gdf
)
"""

outliers = duckdb.sql(query).df()
print(outliers)