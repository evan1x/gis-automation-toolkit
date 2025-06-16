# GIS Automation Toolkit

This repository contains a collection of Python scripts developed to automate common GIS workflows during my internship. These tools streamline spatial data processing, analysis, and cleanup using libraries such as ArcPy and GeoPandas.

## Scripts Included

- **Buffer All Features.py** – Buffers all vector features by a defined distance.
- **Calculate Area and Add to Field.py** – Computes the area of each feature and writes the value to a new attribute field.
- **Calculate Area and Perimeter.py** – Calculates both area and perimeter for polygon features.
- **Count Features Within a Radius.py** – Counts the number of features within a user-specified radius from each point.
- **Count Features.py** – Counts the total number of features in a shapefile.
- **Delete Features Based on Attributes.py** – Deletes features that match a specific attribute condition.
- **Delete Shapes Less Than 100m Squared.py** – Removes small polygon features based on an area threshold.
- **Detect Basic Shape Type.py** – Identifies whether features are polygons, lines, or points.
- **Detect Size of Shape (Small, Med, Large).py** – Classifies features based on their spatial size into small, medium, or large.
- **Export Attribute Table.py** – Exports attribute data from a shapefile to CSV or Excel.
- **List All Field Names.py** – Lists all attribute field names from a given shapefile.
- **Print Centroids of All Polygons.py** – Calculates and prints centroid coordinates for all polygon features.
- **Select and Export Shapes by Attribute.py** – Selects features by attribute query and exports them to a new shapefile.
- **Automated Data Validator.py** – Performs automated checks on shapefile attributes and geometry to ensure data quality and consistency.
- **Batch Coordinate System Transformer.py** – Transforms the coordinate system of multiple shapefiles to a user-specified projection.
- **Batch Project Files to WGS84.py** – Projects all shapefiles in a folder to the WGS84 coordinate system (EPSG:4326).
- **Delete Empty Fields.py** – Removes attribute fields that contain only null or empty values from a shapefile.
- **Feature Class to Shapefile.py** – Converts a feature class from a geodatabase to a standalone shapefile.
- **List All Feature Classes in a GBD.py** – Lists the names of all feature classes contained within a geodatabase.
- **Read a Shapefile and Plot It.py** – Loads a shapefile and generates a simple plot of its features using matplotlib.
- **Remove Null Geometry.py** – Deletes features from a shapefile that have null or invalid geometry.
- **Shapefile to GeoJSON.py** – Converts a shapefile to the GeoJSON format for web and application use.
- **Auto Schema Reporter.py** – Extracts all fields, types, lengths, and domain from a feature class and outputs a clean CSV/Excel for documentation.
- **Batch Field Rename Tool.py** – Standardizes field names across datasets using a mapping dictionary.
- **Merge Feature Classes in a GDB.py** – Merge all feature classes in a GDB into a single output.
- **Batch Raster to Polygon Conversion.py** – Convert multiple classified rasters to vector polygons.
- **Spatial SQL Query Runner.py** – Runs SQL queries on spatial layers.
- **SQL-Based Attribute Calculator.py** – Uses SQL logic to update or calculate fields.
- **Dynamic Layer Joiner.py** – Joins two layers on attribute using SQL logic.
- **PostGIS Query Executor.py** – Directly connect to a PostGIS database and run spatial queries or batch processes.
- **Export SQL Query Result to GeoPackage.py** – Turn and SQL expression into a portable result.
- **SQL Spatial Join Manager.py** - Specifies the spatial join type and optionally buffers features.
- **Bath Export SQL Views to GeoPackage.py** - Loops through a folder of .sql files and runs them against a spatial database, then exports each result as a layer in a GeoPackage.
- **Spatial Clustering Analysis** - Performs advanced DBSCAN clustering on point features, creates cluster polygons, and exports detailed stats.
- **Network Analysis Tool.py** - Network analysis including shortest path calculations, service area analysis, and network topology stats using NetworkX.
- **Raster Terrain Analysis.py** - Terrain analysis with slope/aspect, curvature, terrain classification, watershed delineation and viewshed analysis.
- **Topology Error Detector with SQL & Geometry Fixing** - Detects and optionally fixes topology errors (self-intersections, overlaps, gaps) using SQL queries and Shapely fixes.
- **Zonal Statistics Exporter with SQL Grouping** - Calculates zonal statistics (mean, min, max) from a raster over vector zones using rasterstats, then groups and aggregates via SQL.
- **Time-Aware Spatiotemporal Joiner** - Joins two layers based on spatial relationship and timestamp similarity.

## Technologies Used

- Python 3.x
- ArcPy (requires ArcGIS Pro)
- GeoPandas

## Example Usage

Depending on the script, usage may vary. Here's an example for a command-line script:

python scripts/buffer_all_features.py --input "data/input.shp" --output "output/buffered.shp" --distance 50

Refer to inline comments and function definitions in each script for details on how to use them.

## Installation

⚠️ Note: ArcPy scripts must be run in an ArcGIS Pro environment.

1. Create a virtual environment (optional but recommended):

python -m venv venv  
source venv/bin/activate  (macOS/Linux)  
venv\Scripts\activate     (Windows)

2. Install dependencies:

pip install -r requirements.txt

(Only non-ArcPy scripts can be run outside ArcGIS Pro.)

## Folder Structure

gis-automation-toolkit/  
├── scripts/                  - Python automation scripts  
├── data/                     - (Optional) mock sample data  
├── requirements.txt          - Python dependencies  
├── README.md                 - Project overview  
└── .gitignore                - Files to ignore in Git

## Notes

- **Data**: No proprietary data is included. Use your own shapefiles or mock data for testing.
- **Environment**: Ensure your Python environment has access to ArcGIS Pro's ArcPy if using those scripts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## About

Developed by Evan Nickason during a GIS internship. 
"GIS" name under commits is me :)
GitHub: github.com/evan1x
