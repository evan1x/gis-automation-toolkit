import rasterio
import geopandas as gpd
from rasterio.vrt import WarpedVRT

points = gpd.read_file("points_with_time.shp")

def extract_raster_value_at_time(point, timestamp):
    raster_path = f"./rasters/{timestamp.strftime('%Y%m%d')}.tif"
    try:
        with rasterio.open(raster_path) as src:
            for val in src.sample([(point.x, point.y)]):
                return val[0]
    except:
        return None

points['value'] = points.apply(lambda row: extract_raster_value_at_time(row.geometry, row.timestamp), axis=1)
points.to_file("points_with_raster_values.shp")
