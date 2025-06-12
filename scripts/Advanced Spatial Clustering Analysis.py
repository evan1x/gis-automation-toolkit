import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import argparse
import os

class SpatialClusterAnalyzer:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.gdf = None
        self.clusters = None
    
    def load_data(self):
        try:
            self.gdf = gpd.read_file(self.input_path)
            print(f"Loaded{len(self.gdf)} features from {self.input_path}")

            if self.gdf.geometry.geom_type.iloc[0] != 'Point':
                print("Converting centroids for point-based clustering")
                self.gdf.geometry = self.gdf.geometry.centroid

        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        return True
    
    def perform_clustering(self, eps_masters=500, min_samples=5):
        utm_crs = self.gdf.estimate_utm_crs()
        gdf_utm = self.gdf.to_crs(utm_crs)


        coords = np.array([[point.x, point.y] for point in gdf_utm.geometry])

        db = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords)
        labels = db.labels_

        self.gdf['cluster_id'] = labels
        self.gdf['is_noise'] = labels == -1
        

