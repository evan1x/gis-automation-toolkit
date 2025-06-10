import geopandas as gpd

parcels = gpd.read_file("parcels.shp")
owners = gpd.read_file("owners.shp")

joined = parcels.merge(owners, how="left", on="parcel_id")
joined.to_file("parcels_with_owners.shp")
