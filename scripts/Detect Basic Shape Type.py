import arcpy

fc = r"C:\GIS\Intern_Test_Project\shapefiles\test_polygons.shp"

field_name = "ShapeType"

fields = [f.name for f in arcpy.ListFields(fc)]
if field_name not in fields:
    arcpy.AddField_management(fc, field_name, "TEXT", field_length=20)

with arcpy.da.UpdateCursor(fc, ["SHAPE@", field_name]) as cursor:
    for row in cursor:
        shape = row[0]
        area = shape.getArea("PLANAR", "SQUAREMETERS")
        perimeter = shape.getLength("PLANAR", "METERS")

        # Compactness metric
        if perimeter == 0:
            shape_type = "Unknown"
        else:
            compactness = (4 * 3.14159 * area) / (perimeter ** 2)
            if compactness > 0.85:
                shape_type = "Circle"
            elif compactness > 0.7:
                shape_type = "Square"
            else:
                shape_type = "Rectangle"

        row[1] = shape_type
        cursor.updateRow(row)
        print(f"Updated shape with type: {shape_type}")
