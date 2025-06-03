import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"

field_name = "Area_m2"

existing_fields = [f.name for f in arcpy.ListFields(fc)]
if field_name not in existing_fields:
	arcpy.AddField_management(fc, field_name, "DOUBLE")
	print(f"Field '{field_name}' added.")
else:
	print(f"Field '{field_name}' already exists.")

updated_count = 0


with arcpy.da.UpdateCursor(fc, ["SHAPE@", field_name]) as cursor:
	for row in cursor:
		area = row[0].getArea("PLANAR", "SQUAREMETERS")
		row[1] = area
		cursor.updateRow(row)
		updated_count += 1
		print(f"Updated area: {area:.2f} square meters")

print(f"Finished updating {updated_count} features with area values.")