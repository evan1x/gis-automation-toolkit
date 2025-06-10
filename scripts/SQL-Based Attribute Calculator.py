import geopandas as gpd

gdf = gpd.read_file("parcels.shp")
gdf["land_use_code"] = gdf.query("area > 1000 and zoning == 'R1'")["area"].apply(lambda x: "LargeRes")
gdf.fillna("SmallRes", inplace=True)
gdf.to_file("parcels_updated.shp")

