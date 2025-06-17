import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import duckdb

df = pd.read_csv("movement_logs.csv")  # columns: id, lat, lon, timestamp
df["geometry"] = df.apply(lambda row: Point(row.lon, row.lat), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

duckdb.register("tracks", gdf)

query = """
SELECT id, MIN(timestamp) as start_time, MAX(timestamp) as end_time,
       COUNT(*) as points_count
FROM tracks
GROUP BY id
"""

result = duckdb.sql(query).df()
print(result)
