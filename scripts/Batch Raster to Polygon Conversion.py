import arcpy
import os

input_folder = r"C:\rasters" # Replace with your path
output_folder = r"C:\raster_to_polygon" # Replace with desired output path

arcpy.env.workspace = input_folder
rasters = arcpy.ListRasters()

for raster in rasters:
    out_fc = os.path.join(output_folder, f"{os.path.splitext(raster)[0]}_poly.shp")
    arcpy.RasterToPolygon_conversion(raster, out_fc, "NO_SIMPLIFY", "VALUE")