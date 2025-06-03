import arcpy
import os

# Workspace setup
workspace = r"C:\Users\mmeyer\Documents\AE_Facilities_and_Streets_-_Streetlights"
arcpy.env.workspace = workspace
arcpy.env.overwriteOutput = True

# Use your actual streetlight data
streetlight_fc = os.path.join(workspace, "streetlight.shp")
buffer_fc = os.path.join(workspace, "buffer_area.shp")

# Center point file
point_fc = os.path.join(workspace, "center.shp")
if arcpy.Exists(point_fc):
    arcpy.Delete_management(point_fc)
arcpy.CreateFeatureclass_management(workspace, "center.shp", "POINT", spatial_reference=4326)
with arcpy.da.InsertCursor(point_fc, ["SHAPE@XY"]) as cursor:
    cursor.insertRow([(-81.1461208, 44.4781249)])  # Replace with your town center (Lon, Lat)

# Create buffer (5 km)
buffer_distance = "5000 Meters"
arcpy.Buffer_analysis(point_fc, buffer_fc, buffer_distance, dissolve_option="ALL")

# Make feature layer from streetlights
arcpy.MakeFeatureLayer_management(streetlight_fc, "streetlights_layer")

# Select streetlights within buffer
arcpy.SelectLayerByLocation_management(
    "streetlights_layer",
    overlap_type="INTERSECT",
    select_features=buffer_fc,
    selection_type="NEW_SELECTION"
)

# Count them
count = int(arcpy.GetCount_management("streetlights_layer")[0])
print(f"Number of streetlights within {buffer_distance} of the center point: {count}")
