import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"

if "SizeCat" not in [f.name for f in arcpy.ListFields(fc)]:
	arcpy.AddField_management(fc, "SizeCat", "TEXT", field_length=10)

# Adjust numbers to change parameters
with arcpy.da.UpdateCursor(fc, ["SHAPE@", "SizeCat"]) as cursor:
	for row in cursor:
		area = row[0].getArea("PLANAR", "SQUAREMETERS")
		if area < 500: 
			size = "Small" 
		elif area < 2000: # Anything greater than this number will be classified as large
			size = "Medium"
		else:
			size = "Large"
		print(f"Feature with area {area:.2f} classified as {size}")
		row[1] = size
		cursor.updateRow(row)

