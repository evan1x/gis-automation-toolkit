#!/usr/bin/env python3
"""
Advanced Network Analysis Tool
Performs shortest path analysis, service area calculations, and network statistics
Uses NetworkX and GeoPandas for comprehensive network analysis
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, snap
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import argparse
import os
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class NetworkAnalyzer:
    def __init__(self, network_path, points_path=None):
        self.network_path = network_path
        self.points_path = points_path
        self.network_gdf = None
        self.points_gdf = None
        self.graph = None
        self.node_positions = {}
        
    def load_network(self):
        """Load network data and create NetworkX graph"""
        try:
            self.network_gdf = gpd.read_file(self.network_path)
            print(f"Loaded {len(self.network_gdf)} network segments")
            
            # Ensure projected CRS for accurate distance calculations
            if self.network_gdf.crs.is_geographic:
                utm_crs = self.network_gdf.estimate_utm_crs()
                self.network_gdf = self.network_gdf.to_crs(utm_crs)
                print(f"Projected to {utm_crs}")
            
            self._create_graph()
            return True
            
        except Exception as e:
            print(f"Error loading network: {e}")
            return False
    
    def load_points(self):
        """Load point data for analysis"""
        if not self.points_path:
            return True
            
        try:
            self.points_gdf = gpd.read_file(self.points_path)
            print(f"Loaded {len(self.points_gdf)} points")
            
            # Ensure same CRS as network
            if self.points_gdf.crs != self.network_gdf.crs:
                self.points_gdf = self.points_gdf.to_crs(self.network_gdf.crs)
            
            return True
            
        except Exception as e:
            print(f"Error loading points: {e}")
            return False
    
    def _create_graph(self):
        """Create NetworkX graph from network geometry"""
        self.graph = nx.Graph()
        tolerance = 1.0  # Meter tolerance for snapping nodes
        
        # Create nodes and edges from line geometry
        for idx, row in self.network_gdf.iterrows():
            line = row.geometry
            
            if line.geom_type == 'LineString':
                coords = list(line.coords)
                start_coord = coords[0]
                end_coord = coords[-1]
                
                # Snap coordinates to avoid floating point issues
                start_coord = (round(start_coord[0], 1), round(start_coord[1], 1))
                end_coord = (round(end_coord[0], 1), round(end_coord[1], 1))
                
                # Calculate edge weight (length)
                length = line.length
                
                # Add edge attributes from original data
                edge_attrs = {
                    'length': length,
                    'geometry': line,
                    'original_id': idx
                }
                
                # Add speed/travel time if available
                if 'speed' in row:
                    speed_kmh = row['speed'] if row['speed'] > 0 else 50  # Default 50 km/h
                    travel_time = (length / 1000) / speed_kmh * 60  # Minutes
                    edge_attrs['travel_time'] = travel_time
                else:
                    edge_attrs['travel_time'] = length / 1000 * 1.2  # Assume 50 km/h walking
                
                self.graph.add_edge(start_coord, end_coord, **edge_attrs)
                
                # Store node positions
                self.node_positions[start_coord] = Point(start_coord)
                self.node_positions[end_coord] = Point(end_coord)
        
        print(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def find_nearest_node(self, point):
        """Find nearest network node to a given point"""
        nodes = list(self.graph.nodes())
        node_coords = np.array(nodes)
        point_coords = np.array([point.x, point.y])
        
        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(node_coords)
        distance, index = tree.query(point_coords)
        
        return nodes[index], distance
    
    def shortest_path_analysis(self, origin_points, destination_points, weight='length'):
        """
        Calculate shortest paths between origin and destination points
        
        Args:
            origin_points: List of origin point indices or coordinates
            destination_points: List of destination point indices or coordinates
            weight: Edge attribute to use as weight ('length' or 'travel_time')
        """
        results = []
        
        for orig_idx, orig_point in enumerate(origin_points):
            if isinstance(orig_point, int):
                orig_geom = self.points_gdf.iloc[orig_point].geometry
            else:
                orig_geom = Point(orig_point)
            
            orig_node, orig_dist = self.find_nearest_node(orig_geom)
            
            for dest_idx, dest_point in enumerate(destination_points):
                if isinstance(dest_point, int):
                    dest_geom = self.points_gdf.iloc[dest_point].geometry
                else:
                    dest_geom = Point(dest_point)
                
                dest_node, dest_dist = self.find_nearest_node(dest_geom)
                
                try:
                    # Calculate shortest path
                    path = nx.shortest_path(self.graph, orig_node, dest_node, weight=weight)
                    path_length = nx.shortest_path_length(self.graph, orig_node, dest_node, weight=weight)
                    
                    # Create path geometry
                    path_coords = []
                    for i in range(len(path) - 1):
                        edge_data = self.graph[path[i]][path[i + 1]]
                        if 'geometry' in edge_data:
                            line_coords = list(edge_data['geometry'].coords)
                            if i == 0:
                                path_coords.extend(line_coords)
                            else:
                                path_coords.extend(line_coords[1:])  # Skip first coord to avoid duplicates
                    
                    path_geometry = LineString(path_coords) if len(path_coords) > 1 else None
                    
                    results.append({
                        'origin_id': orig_idx,
                        'destination_id': dest_idx,
                        'path_length': path_length,
                        'path_nodes': len(path),
                        'origin_snap_distance': orig_dist,
                        'destination_snap_distance': dest_dist,
                        'geometry': path_geometry
                    })
                    
                except nx.NetworkXNoPath:
                    results.append({
                        'origin_id': orig_idx,
                        'destination_id': dest_idx,
                        'path_length': None,
                        'path_nodes': None,
                        'origin_snap_distance': orig_dist,
                        'destination_snap_distance': dest_dist,
                        'geometry': None
                    })
        
        return pd.DataFrame(results)
    
    def service_area_analysis(self, center_points, max_distance, weight='length'):
        """
        Calculate service areas (isochrones) from center points
        
        Args:
            center_points: List of center point indices or coordinates
            max_distance: Maximum distance/time for service area
            weight: Edge attribute to use as weight
        """
        service_areas = []
        
        for center_idx, center_point in enumerate(center_points):
            if isinstance(center_point, int):
                center_geom = self.points_gdf.iloc[center_point].geometry
            else:
                center_geom = Point(center_point)
            
            center_node, snap_dist = self.find_nearest_node(center_geom)
            
            # Find all nodes within max_distance
            reachable_nodes = []
            
            try:
                # Use Dijkstra's algorithm to find shortest paths
                lengths = nx.single_source_dijkstra_path_length(
                    self.graph, center_node, cutoff=max_distance, weight=weight
                )
                
                reachable_nodes = list(lengths.keys())
                
                # Create service area polygon (convex hull of reachable nodes)
                if len(reachable_nodes) >= 3:
                    node_points = [Point(node) for node in reachable_nodes]
                    service_area_geom = gpd.GeoSeries(node_points).unary_union.convex_hull
                else:
                    # Fallback: buffer around center point
                    service_area_geom = center_geom.buffer(max_distance)
                
                service_areas.append({
                    'center_id': center_idx,
                    'reachable_nodes': len(reachable_nodes),
                    'max_distance': max_distance,
                    'snap_distance': snap_dist,
                    'geometry': service_area_geom
                })
                
            except Exception as e:
                print(f"Error calculating service area for point {center_idx}: {e}")
                service_areas.append({
                    'center_id': center_idx,
                    'reachable_nodes': 0,
                    'max_distance': max_distance,
                    'snap_distance': snap_dist,
                    'geometry': None
                })
        
        return gpd.GeoDataFrame(service_areas, crs=self.network_gdf.crs)
    
    def network_statistics(self):
        """Calculate comprehensive network statistics"""
        stats = {}
        
        # Basic network topology
        stats['total_nodes'] = self.graph.number_of_nodes()
        stats['total_edges'] = self.graph.number_of_edges()
        stats['total_length_km'] = sum([data['length'] for _, _, data in self.graph.edges(data=True)]) / 1000
        
        # Connectivity analysis
        stats['is_connected'] = nx.is_connected(self.graph)
        stats['number_of_components'] = nx.number_connected_components(self.graph)
        
        if stats['is_connected']:
            # Global network measures (only for connected graphs)
            stats['diameter'] = nx.diameter(self.graph, weight='length')
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph, weight='length')
            stats['global_efficiency'] = nx.global_efficiency(self.graph)
        
        # Node degree statistics
        degrees = dict(self.graph.degree())
        stats['avg_node_degree'] = np.mean(list(degrees.values()))
        stats['max_node_degree'] = max(degrees.values())
        stats['min_node_degree'] = min(degrees.values())
        
        # Edge length statistics
        edge_lengths = [data['length'] for _, _, data in self.graph.edges(data=True)]
        stats['avg_edge_length'] = np.mean(edge_lengths)
        stats['max_edge_length'] = max(edge_lengths)
        stats['min_edge_length'] = min(edge_lengths)
        stats['total_length_m'] = sum(edge_lengths)
        
        # Network density
        stats['edge_density'] = nx.density(self.graph)
        
        # Centrality measures (sample for large networks)
        sample_size = min(100, stats['total_nodes'])
        if sample_size > 10:
            sample_nodes = list(self.graph.nodes())[:sample_size]
            subgraph = self.graph.subgraph(sample_nodes)
            
            if nx.is_connected(subgraph):
                centrality = nx.betweenness_centrality(subgraph, weight='length')
                stats['avg_betweenness_centrality'] = np.mean(list(centrality.values()))
                stats['max_betweenness_centrality'] = max(centrality.values())
        
        return stats
    
    def find_critical_nodes(self, n_critical=10):
        """Identify critical nodes based on betweenness centrality"""
        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(self.graph, weight='length', k=min(100, self.graph.number_of_nodes()))
        
        # Sort nodes by centrality
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        critical_nodes = []
        for node, centrality_score in sorted_nodes[:n_critical]:
            node_point = Point(node)
            critical_nodes.append({
                'node_id': str(node),
                'centrality_score': centrality_score,
                'degree': self.graph.degree(node),
                'geometry': node_point
            })
        
        return gpd.GeoDataFrame(critical_nodes, crs=self.network_gdf.crs)
    
    def export_results(self, output_dir, shortest_paths_df=None, service_areas_gdf=None, 
                      critical_nodes_gdf=None, network_stats=None):
        """Export all analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export shortest paths
        if shortest_paths_df is not None and not shortest_paths_df.empty:
            paths_with_geom = shortest_paths_df[shortest_paths_df['geometry'].notna()]
            if not paths_with_geom.empty:
                paths_gdf = gpd.GeoDataFrame(paths_with_geom, crs=self.network_gdf.crs)
                paths_output = os.path.join(output_dir, 'shortest_paths.shp')
                paths_gdf.to_file(paths_output)
                print(f"Shortest paths exported to: {paths_output}")
        
        # Export service areas
        if service_areas_gdf is not None and not service_areas_gdf.empty:
            service_areas_output = os.path.join(output_dir, 'service_areas.shp')
            service_areas_gdf.to_file(service_areas_output)
            print(f"Service areas exported to: {service_areas_output}")
        
        # Export critical nodes
        if critical_nodes_gdf is not None and not critical_nodes_gdf.empty:
            critical_nodes_output = os.path.join(output_dir, 'critical_nodes.shp')
            critical_nodes_gdf.to_file(critical_nodes_output)
            print(f"Critical nodes exported to: {critical_nodes_output}")
        
        # Export network statistics
        if network_stats:
            stats_output = os.path.join(output_dir, 'network_statistics.csv')
            stats_df = pd.DataFrame([network_stats])
            stats_df.to_csv(stats_output, index=False)
            print(f"Network statistics exported to: {stats_output}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Network Analysis Tool')
    parser.add_argument('--network', required=True, help='Network shapefile path')
    parser.add_argument('--points', help='Points shapefile path (optional)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--analysis', choices=['shortest_path', 'service_area', 'statistics', 'all'],
                       default='all', help='Type of analysis to perform')
    parser.add_argument('--max_distance', type=float, default=5000,
                       help='Maximum distance for service area analysis (meters)')
    parser.add_argument('--weight', choices=['length', 'travel_time'], default='length',
                       help='Edge weight for analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NetworkAnalyzer(args.network, args.points)
    
    # Load data
    if not analyzer.load_network():
        return
    
    if args.points:
        if not analyzer.load_points():
            return
    
    results = {}
    
    # Perform analysis based on user selection
    if args.analysis in ['shortest_path', 'all'] and args.points:
        print("\nPerforming shortest path analysis...")
        # Use all points as both origins and destinations
        point_indices = list(range(len(analyzer.points_gdf)))
        if len(point_indices) > 10:
            # Sample for large datasets
            point_indices = point_indices[:10]
            print(f"Sampling first 10 points for analysis")
        
        results['shortest_paths'] = analyzer.shortest_path_analysis(
            point_indices, point_indices, weight=args.weight
        )
    
    if args.analysis in ['service_area', 'all'] and args.points:
        print("\nPerforming service area analysis...")
        point_indices = list(range(min(5, len(analyzer.points_gdf))))
        results['service_areas'] = analyzer.service_area_analysis(
            point_indices, args.max_distance, weight=args.weight
        )
    
    if args.analysis in ['statistics', 'all']:
        print("\nCalculating network statistics...")
        results['network_stats'] = analyzer.network_statistics()
        results['critical_nodes'] = analyzer.find_critical_nodes()
    
    # Export results
    analyzer.export_results(
        args.output,
        shortest_paths_df=results.get('shortest_paths'),
        service_areas_gdf=results.get('service_areas'),
        critical_nodes_gdf=results.get('critical_nodes'),
        network_stats=results.get('network_stats')
    )
    
    # Print summary
    print(f"\nAnalysis completed successfully!")
    if 'network_stats' in results:
        stats = results['network_stats']
        print(f"Network has {stats['total_nodes']} nodes and {stats['total_edges']} edges")
        print(f"Total network length: {stats['total_length_km']:.2f} km")
        print(f"Network is {'connected' if stats['is_connected'] else 'disconnected'}")
    
    print(f"All results exported to: {args.output}")

if __name__ == "__main__":
    main()