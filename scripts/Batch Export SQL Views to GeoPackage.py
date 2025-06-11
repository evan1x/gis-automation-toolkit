import os
import geopandas as gpd
import duckdb
import sqlalchemy
from sqlalchemy import create_engine
import argparse

def run_postgis_query(sql_text, conn_str):
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        gdf = gpd.read_postgis(sql_text, conn)
    return gdf

def run_duckdb_query(sql_text, db_path):
    con = duckdb.connect(database=db_path)
    con.install_extension("spatial")
    con.load_extension("spatial")
    df = con.execute(sql_text).fetchdf()
    if 'geometry' in df.columns:
        df['geometry'] = gpd.GeoSeries.from_wkb(df['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
    else:
        gdf = gpd.GeoDataFrame(df)
    return gdf

def export_views(sql_folder, output_gpkg, db_type, conn_str):
    sql_files = [f for f in os.listdir(sql_folder) if f.endswith('.sql')]
    if not sql_files:
        print("No SQL files found.")
        return

    print(f"🔁 Executing {len(sql_files)} SQL files...")
    for sql_file in sql_files:
        path = os.path.join(sql_folder, sql_file)
        with open(path, 'r') as f:
            sql = f.read()

        layer_name = os.path.splitext(sql_file)[0].lower().replace(" ", "_")
        try:
            if db_type == "postgis":
                gdf = run_postgis_query(sql, conn_str)
            elif db_type == "duckdb":
                gdf = run_duckdb_query(sql, conn_str)
            else:
                print(f"Unsupported DB type: {db_type}")
                continue

            if gdf.empty:
                print(f"{sql_file} returned no rows.")
                continue

            gdf.to_file(output_gpkg, driver="GPKG", layer=layer_name)
            print(f"Exported {layer_name} to {output_gpkg}")

        except Exception as e:
            print(f"Error processing {sql_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Export SQL Views to GeoPackage")
    parser.add_argument("--sql_dir", required=True, help="Directory containing .sql files")
    parser.add_argument("--output", required=True, help="Output GeoPackage file path")
    parser.add_argument("--db_type", choices=["postgis", "duckdb"], required=True, help="Database type")
    parser.add_argument("--conn", required=True, help="Connection string or path")

    args = parser.parse_args()
    export_views(args.sql_dir, args.output, args.db_type, args.conn)
