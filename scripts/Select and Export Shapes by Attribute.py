import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"
out_fc = r"C:\GIS\Intern_Test_Project\shapefiles\large_polygons.shp"
arcpy.MakeFeatureLayer_management(fc, "poly_layer")

arcpy.SelectLayerByAttribute_management("poly_layer", "NEW_SELECTION", '"Area_m2" > 1000')
arcpy.CopyFeatures_management("poly_layer", out_fc)
