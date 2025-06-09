import arcpy
import csv

gdb = r"C:\GIS\Project.gdb" # Replace with your path
out_csv = r"C:\schema_report.csv" # Replace with desired output path

arcpy.env.workspace = gdb
fcs = arcpy.ListFeatureClasses()

with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Feature Class", "Field Name", "Type", "Length", "Domain"])

    for fc in fcs:
        fields = arcpy.ListFields(fc)
        for f in fields:
            writer.writerow([fc, f.name, f.type, f.length, f.domain])

            