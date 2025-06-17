import geopandas as gpd
import duckdb

streets = gpd.read_file("streets.shp")
intersections = gpd.overlay(streets, streets, how="intersection")
intersections["x"] = intersections.geometry.centroid.x.round(6)
intersections["y"] = intersections.geometry.centroid.y.round(6)

duckdb.register("intersecctions", intersections)
dedup_query = """
SELECT DISTINCT x, y
FROM intersections
"""

dedup = duckdb.sql(dedup_query).df()
print(f"Unique intersections: {len(dedup)}")

