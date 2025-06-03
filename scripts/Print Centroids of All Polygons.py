import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"

with arcpy.da.SearchCursor(fc, ["OID@", "SHAPE@XY"]) as cursor:
	for row in cursor:
		print(f"Feature {row[0]} centroid: {row[1]}")