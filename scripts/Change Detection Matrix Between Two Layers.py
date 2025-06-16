import geopandas as gpd
import duckdb

a = gpd.read_file("old.shp")
b = gpd.read_file("new.shp")

a['src'] = 'old'
b['src'] = 'new'

combined = pd.concat([a, b])
duckdb.register("features", combined)

# Naive geometry comparison (you can expand this with buffer/intersects)
change_matrix = duckdb.query("""
SELECT 
    id,
    COUNT(DISTINCT src) as versions
FROM features
GROUP BY id
""").to_df()

# You can then flag changes:
# versions = 1 → deleted or added
# versions = 2 → modified or unchanged (depending on geometry comparison)
