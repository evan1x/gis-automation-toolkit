import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"

deleted_any = False

with arcpy.da.UpdateCursor(fc, ["OID@", "SHAPE@"]) as cursor:
	for row in cursor:
		area = row[1].getArea("PLANAR", "SQUAREMETERS")
		print(f"Shape {row[0]} has area {area}")
		if area < 100: # Adjust area condition
			print(f"Deleting shape with area {area}")
			cursor.deleteRow()
			deleted_any = True

if not deleted_any:
	print("No shapes meet the area condition.")