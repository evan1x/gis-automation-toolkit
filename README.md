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

Developed by Evan Nickason during a GIS internship as part of a broader goal to transition into other roles.
GitHub: github.com/evan1x
