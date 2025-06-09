import arcpy

gdb_path = r"C:\Data\GDB" # Replace with your path
output_fc = r"C:\Data\MergedOutput.shp" # Replace with desired output path

arcpy.env.workspace = gdb_path
fcs = arcpy.ListFeatureClasses()

arcpy.Merge_management(fcs, output_fc)