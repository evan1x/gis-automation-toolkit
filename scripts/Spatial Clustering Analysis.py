#!/usr/bin/env python3
"""
Advanced Spatial Clustering Analysis Tool
Performs DBSCAN clustering on point features and creates cluster polygons
Uses GeoPandas and scikit-learn for advanced spatial analysis
"""

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
        """Load spatial data from various formats"""
        try:
            self.gdf = gpd.read_file(self.input_path)
            print(f"Loaded {len(self.gdf)} features from {self.input_path}")
            
            # Ensure we have point geometry
            if self.gdf.geometry.geom_type.iloc[0] != 'Point':
                print("Converting to centroids for point-based clustering")
                self.gdf.geometry = self.gdf.geometry.centroid
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        return True
    
    def perform_clustering(self, eps_meters=500, min_samples=5):
        """
        Perform DBSCAN clustering on point features
        
        Args:
            eps_meters: Maximum distance between points in same cluster (meters)
            min_samples: Minimum number of points to form a cluster
        """
        # Project to appropriate UTM zone for distance calculations
        utm_crs = self.gdf.estimate_utm_crs()
        gdf_utm = self.gdf.to_crs(utm_crs)
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in gdf_utm.geometry])
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords)
        labels = db.labels_
        
        # Add cluster labels to geodataframe
        self.gdf['cluster_id'] = labels
        self.gdf['is_noise'] = labels == -1
        
        # Statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        return n_clusters, n_noise
    
    def create_cluster_polygons(self, buffer_distance=100):
        """
        Create convex hull polygons for each cluster with optional buffer
        
        Args:
            buffer_distance: Buffer distance around cluster polygons (meters)
        """
        cluster_polygons = []
        
        # Project to UTM for accurate distance calculations
        utm_crs = self.gdf.estimate_utm_crs()
        gdf_utm = self.gdf.to_crs(utm_crs)
        
        for cluster_id in self.gdf['cluster_id'].unique():
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_points = gdf_utm[gdf_utm['cluster_id'] == cluster_id]
            
            if len(cluster_points) < 3:
                # For small clusters, create circular buffer around centroid
                centroid = cluster_points.geometry.unary_union.centroid
                polygon = centroid.buffer(buffer_distance)
            else:
                # Create convex hull for larger clusters
                coords = np.array([[point.x, point.y] for point in cluster_points.geometry])
                hull = ConvexHull(coords)
                hull_points = coords[hull.vertices]
                polygon = Polygon(hull_points).buffer(buffer_distance)
            
            cluster_polygons.append({
                'cluster_id': cluster_id,
                'point_count': len(cluster_points),
                'geometry': polygon
            })
        
        # Create GeoDataFrame of cluster polygons
        self.cluster_polygons = gpd.GeoDataFrame(cluster_polygons, crs=utm_crs)
        self.cluster_polygons = self.cluster_polygons.to_crs(self.gdf.crs)
        
        return self.cluster_polygons
    
    def calculate_cluster_statistics(self):
        """Calculate detailed statistics for each cluster"""
        stats = []
        
        for cluster_id in self.gdf['cluster_id'].unique():
            if cluster_id == -1:
                continue
                
            cluster_data = self.gdf[self.gdf['cluster_id'] == cluster_id]
            
            # Basic statistics
            point_count = len(cluster_data)
            
            # Spatial statistics (in projected coordinates)
            utm_crs = self.gdf.estimate_utm_crs()
            cluster_utm = cluster_data.to_crs(utm_crs)
            
            # Calculate cluster area (convex hull)
            if len(cluster_utm) >= 3:
                coords = np.array([[point.x, point.y] for point in cluster_utm.geometry])
                hull = ConvexHull(coords)
                area = hull.volume  # In 2D, volume = area
            else:
                area = 0
            
            # Calculate density
            density = point_count / max(area, 1) if area > 0 else 0
            
            # Centroid
            centroid = cluster_data.geometry.unary_union.centroid
            
            stats.append({
                'cluster_id': cluster_id,
                'point_count': point_count,
                'area_sqm': area,
                'density_per_sqm': density,
                'centroid_x': centroid.x,
                'centroid_y': centroid.y
            })
        
        return pd.DataFrame(stats)
    
    def export_results(self):
        """Export all results to files"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Export clustered points
        points_output = os.path.join(self.output_dir, 'clustered_points.shp')
        self.gdf.to_file(points_output)
        print(f"Clustered points exported to: {points_output}")
        
        # Export cluster polygons
        if hasattr(self, 'cluster_polygons'):
            polygons_output = os.path.join(self.output_dir, 'cluster_polygons.shp')
            self.cluster_polygons.to_file(polygons_output)
            print(f"Cluster polygons exported to: {polygons_output}")
        
        # Export statistics
        stats = self.calculate_cluster_statistics()
        stats_output = os.path.join(self.output_dir, 'cluster_statistics.csv')
        stats.to_csv(stats_output, index=False)
        print(f"Cluster statistics exported to: {stats_output}")
        
        return stats
    
    def create_visualization(self):
        """Create comprehensive visualization of clustering results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original points colored by cluster
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.gdf['cluster_id'].unique())))
        for i, cluster_id in enumerate(self.gdf['cluster_id'].unique()):
            cluster_data = self.gdf[self.gdf['cluster_id'] == cluster_id]
            if cluster_id == -1:
                ax1.scatter(cluster_data.geometry.x, cluster_data.geometry.y, 
                           c='black', marker='x', s=20, alpha=0.6, label='Noise')
            else:
                ax1.scatter(cluster_data.geometry.x, cluster_data.geometry.y,
                           c=[colors[i]], s=30, alpha=0.7, label=f'Cluster {cluster_id}')
        
        ax1.set_title('Spatial Clusters')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Plot 2: Cluster polygons
        if hasattr(self, 'cluster_polygons'):
            self.cluster_polygons.plot(ax=ax2, alpha=0.3, edgecolor='red')
            self.gdf[self.gdf['cluster_id'] != -1].plot(ax=ax2, markersize=10, alpha=0.7)
            ax2.set_title('Cluster Polygons (Convex Hulls)')
        
        # Plot 3: Cluster size distribution
        cluster_sizes = self.gdf[self.gdf['cluster_id'] != -1]['cluster_id'].value_counts()
        ax3.bar(cluster_sizes.index, cluster_sizes.values)
        ax3.set_title('Points per Cluster')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Points')
        
        # Plot 4: Distance to nearest neighbor analysis
        from sklearn.neighbors import NearestNeighbors
        utm_crs = self.gdf.estimate_utm_crs()
        gdf_utm = self.gdf.to_crs(utm_crs)
        coords = np.array([[point.x, point.y] for point in gdf_utm.geometry])
        
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        nearest_distances = distances[:, 1]  # Distance to nearest neighbor (not self)
        
        ax4.hist(nearest_distances, bins=30, alpha=0.7)
        ax4.axvline(np.mean(nearest_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(nearest_distances):.1f}m')
        ax4.set_title('Nearest Neighbor Distance Distribution')
        ax4.set_xlabel('Distance (meters)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_output = os.path.join(self.output_dir, 'clustering_analysis.png')
        plt.savefig(viz_output, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {viz_output}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Spatial Clustering Analysis')
    parser.add_argument('--input', required=True, help='Input shapefile path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--eps', type=float, default=500, 
                       help='Maximum distance between points in cluster (meters)')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='Minimum samples per cluster')
    parser.add_argument('--buffer', type=float, default=100,
                       help='Buffer distance for cluster polygons (meters)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SpatialClusterAnalyzer(args.input, args.output)
    
    # Load and process data
    if not analyzer.load_data():
        return
    
    # Perform clustering
    n_clusters, n_noise = analyzer.perform_clustering(args.eps, args.min_samples)
    
    # Create cluster polygons
    cluster_polygons = analyzer.create_cluster_polygons(args.buffer)
    
    # Export results
    stats = analyzer.export_results()
    
    # Create visualization if requested
    if args.visualize:
        analyzer.create_visualization()
    
    print(f"\nClustering completed successfully!")
    print(f"Found {n_clusters} clusters with {n_noise} noise points")
    print(f"Results exported to: {args.output}")

if __name__ == "__main__":
    main()