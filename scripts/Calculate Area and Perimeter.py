import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"

if arcpy.ListFields(fc, "Area_m2") == []:
	arcpy.AddField_management(fc, "Area_m2", "DOUBLE")
if arcpy.ListFields(fc, "Perim_m") == []:
	arcpy.AddField_management(fc, "Perim_m", "DOUBLE")

with arcpy.da.UpdateCursor(fc, ["SHAPE@", "Area_m2", "Perim_m"]) as cursor:
	for row in cursor:
		geom = row[0]
		area = geom.getArea("PLANAR", "SQUAREMETERS")
		perimeter = geom.getLength("PLANAR", "METERS")
		print(f"Area: {area:.2f} sq.m | Perimeter: {perimeter:.2f} m")
		row[1] = area
		row[2] = perimeter
		cursor.updateRow(row)