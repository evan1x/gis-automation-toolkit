import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"
count = int(arcpy.GetCount_management(fc)[0])
print(f"Total features: {count}")