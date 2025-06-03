import arcpy

arcpy.FeatureClassToFeatureClass_conversion(
    "input_fc", "C:/output_folder", "output_shapefile"
)
