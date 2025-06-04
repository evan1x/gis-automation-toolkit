import geopandas
import matplotlib.pyplot as plt # For plotting

# Read a shapefile
try:
    gdf = geopandas.read_file("C:/data/world_cities.shp") # Replace with your shapefile path

    # Print the first few rows of the GeoDataFrame
    print("GeoDataFrame Head:")
    print(gdf.head())

    # Print the coordinate reference system (CRS)
    print(f"\nCoordinate Reference System (CRS): {gdf.crs}")

    # Create a simple plot
    gdf.plot()
    plt.title("World Cities")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    print("\nSuccessfully read and plotted the shapefile.")

except Exception as e:
    print(f"An error occurred: {e}") 