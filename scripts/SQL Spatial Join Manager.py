import argparse
import geopandas as gpd
import duckdb
import os
from shapely import wkb
from shapely.geometry import shape

def run_spatial_join(input_a, input_b, join_type, output_file, buffer=0.0):
    a = gpd.read_file(input_a)
    b = gpd.read_file(input_b)

    if buffer > 0:
        a["geometry"] = a.geometry.buffer(buffer)


    a["geom"] = a.geometry.to_wkb()
    b["geom"] = b.geometry.to_wkb()

    a_path = "a_temp.parquet"
    b_path = "b_temp.parquet"
    a.drop(columns="geometry").to_parquet(a_path)
    b.drop(columns="geometry").to_parquet(b_path)

    predicate = {
        "intersects": "ST_Intersects",
        "contains": "ST_Contains",
        "within": "ST_Within",
        "dwithin": "ST_Dwithin"

    }.get(join_type.lower(), "ST_Intersects")

    query = f"""
    SELECT a.*. b.*
    FROM read_parquet('{a_path}) a
    JOIN read_parquet('{b_path}') b
    ON {predicate}(ST_GeomFromWKB(a.geom), ST_GeomFromWKB(b.geom))
    """

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extensioon("spatial")
    result = con.execute(query).fetchdf()

    result["geometry"] = result["geom"].apply(lambda g: wkb.loads(g, hex=False))

    result_gdf = gpd.GeoDataFrame(result, geometry="geometry", crs=a.crs)
    result_gdf.drop(columns=["geom"], inplace=True)

    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".shp":
        result_gdf.to_file(output_file)
    elif ext == ".gpkg":
        result_gdf.to_file(output_file, driver="GPKG")
    elif ext == ".geojson":
        result_gdf.to_file(output_file, driver="GeoJSON")
    else:
        print("ERROR: Unsupported output format.")

    os.remove(a_path)
    os.remove(b_path)

    print("Spatial join complete. Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Join using SQL")
    parser.add_argument("--input_a", required=True, help="Path to first layer")
    parser.add_argument("--input_b", required=True, help="Path to second layer")
    parser.add_argument("--join_type", choices=["intersects", "contains", "within", "dwithin"], default="intersects", help="Type of spatial join")
    parser.add_argument("--output", required=True, help="Output file path (.shp, .gpkg, .geojson)")
    parser.add_argument("--buffer", type=float, default=0.0, help="Optional buffer distance")

    args = parser.parse_args()
    run_spatial_join(args.input_a, args.input_b, args.join_type, args.output, args.buffer)
    