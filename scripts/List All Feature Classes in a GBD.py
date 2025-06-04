import arcpy

arcpy.env.workspace = "path_to.gbd" # Replace with your path
fcs = arcpy.ListFeatureClasses()
print(fcs)