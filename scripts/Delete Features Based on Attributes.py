import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"


with arcpy.da.UpdateCursor(fc, ["Area_m2"]) as cursor: # Change attribute
	for row in cursor:
		if row[0] < 50:
			cursor.deleteRow()