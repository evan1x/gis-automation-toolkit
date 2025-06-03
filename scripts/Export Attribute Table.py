import shapefile  # pyshp
import csv

shapefile_path = r"C:\Users\mmeyer\Desktop\Parcels\parcels.shp"

# Read the shapefile
sf = shapefile.Reader(shapefile_path)

# Get fields (skip first deletion flag field)
fields = sf.fields[1:]
field_names = [field[0] for field in fields] 

# Get records (each row is a list of attribute values)
records = sf.records()

# Write attributes to CSV
output_csv = r"C:\Users\mmeyer\Documents\parcels_output.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(field_names)  # write header
    writer.writerows(records)     # write attribute rows

print(f"Attribute table exported successfully to '{output_csv}'")


