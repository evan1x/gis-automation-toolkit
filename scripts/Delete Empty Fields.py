import arcpy

gdb_path = r"C:\GIS\Projects\YourProject\YourGeodatabase.gdb"
feature_class = "your_feature_class_name"
fc = f"{gdb_path}\\{feature_class}"

fields = arcpy.ListFields(fc)

for field in fields:
    if field.type not in ['OID', 'Geometry']:
        values = [row[0] for row in arcpy.da.SearchCursor(fc, field_name)]
        if all(v in (None, '', ' ') for v in values):
            print(f"Deleting empty field: {field.name}")
            arcpy.DeleteField_management(fc, field.name)