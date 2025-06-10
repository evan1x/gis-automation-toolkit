import psycopg2
import geopandas as gpd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost:5432/gisdb")
query = """
    SELECT *, ST_Area(geom) as area
    FROM parcels
    WHERE zoning = 'R2' AND ST_Area(geom) > 1000
"""
gdf = gpd.read_postgis(query, engine, geom_col='geom')

