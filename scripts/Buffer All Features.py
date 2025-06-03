import arcpy

in_fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"
out_fc = r"C:\GIS\Intern_Test_Project\shapefiles\buffered_polygons.shp"

arcpy.Buffer_analysis(in_fc, out_fc, "50 Meters", dissolve_option="NONE")