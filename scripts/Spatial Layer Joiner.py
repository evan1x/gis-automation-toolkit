from sqlalchemy import create_engine
import geopandas as gpd

engine = create_engine("postgresql://user:password@localhost:5432/yourdb")

query = """
SELECT a.*, b.attribute_name
FROM layer1 a
JOIN layer2 b
ON ST_Intersects(a.geom, b.geom)
"""
joined = gpd.read_postgis(query, engine, geom_col="geom")
joined.to_file("output.shp")

