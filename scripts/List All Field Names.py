import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"
fields = arcpy.ListFields(fc)

for field in fields:
	print(field.name, field.type)