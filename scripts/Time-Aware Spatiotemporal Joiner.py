import geopandas as gpd
import pandas as pd
from shapely.geometry import point
import duckdb

a = gpd.read_file("layer_a.shp")
b = gpd.read_file("layer_b.shp")

a['timestamp'] = pd.to_datetime(a['timestamp'])
b['timestamp'] = pd.to_datetime(b['timestamp'])

a = a.to_crs("EPSG:4326")
b = b.to_crs("EPSG:4326")

duckdb.register("a", a)
duckdb.register("b", b)

query = """
SELECT
    a.*,
    b.id AS b_id,
    b.timestamp as b_time
FROM a
JOIN b
ON ST_Intersects(a.geometry, b.geometry)
WHERE abs(strftime('%s', a.timestamp) - strftime('%s', b.timestamp)) < 3600
"""
joined = duckdb.query(query).to_df()
