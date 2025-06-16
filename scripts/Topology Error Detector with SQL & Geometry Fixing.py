import geopandas as gpd
import duckdb

gdf = gpd.read_file("layer.shp")
gdf['valid'] = gdf.is_valid

duckdb.register("geo", gdf)
errors = duckdb.query("""
    SELECT *
    FROM geo
    WHERE NOT valid
""").to_df()

gdf.loc[~gdf['valid'], 'geometry'] = gdf.loc[~gdf['valid'], 'geometry'].buffer(0)
gdf.to_file("fixed_layer.shp")