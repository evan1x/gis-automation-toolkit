#!/usr/bin/env python3
"""
Advanced Raster Terrain Analysis Tool
Performs comprehensive terrain analysis including slope, aspect, hillshade,
viewshed analysis, watershed delineation, and terrain classification
Uses rasterio, numpy, and scipy for advanced raster operations
"""

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes
from skimage import measure, filters, morphology
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

class TerrainAnalyzer:
    def __init__(self, dem_path):
        self.dem_path = dem_path
        self.dem_data = None
        self.transform = None
        self.crs = None
        self.nodata = None
        self.pixel_size = None
        
    def load_dem(self):
        """Load DEM raster data"""
        try:
            with rasterio.open(self.dem_path) as src:
                self.dem_data = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                self.nodata = src.nodata
                self.pixel_size = abs(src.transform[0])  # Assuming square pixels
                
            print(f"Loaded DEM: {self.dem_data.shape} pixels")
            print(f"Pixel size: {self.pixel_size:.2f} units")
            print(f"Elevation range: {np.nanmin(self.dem_data):.1f} to {np.nanmax(self.dem_data):.1f}")
            
            # Handle nodata values
            if self.nodata is not None:
                self.dem_data = np.where(self.dem_data == self.nodata, np.nan, self.dem_data)
            
            return True
            
        except Exception as e:
            print(f"Error loading DEM: {e}")
            return False
    
    def calculate_slope_aspect(self):
        """Calculate slope and aspect from DEM"""
        # Calculate gradients
        dy, dx = np.gradient(self.dem_data, self.pixel_size)
        
        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Calculate aspect in degrees (0-360)
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
        
        return slope_deg, aspect_deg
    
    def calculate_hillshade(self, azimuth=315, altitude=45):
        """Calculate hillshade for visualization"""
        # Calculate slope and aspect
        slope_deg, aspect_deg = self.calculate_slope_aspect()
        
        # Convert to radians
        slope_rad = np.radians(slope_deg)
        aspect_rad = np.radians(aspect_deg)
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        # Calculate hillshade
        hillshade = (np.sin(altitude_rad) * np.sin(slope_rad) +
                    np.cos(altitude_rad) * np.cos(slope_rad) *
                    np.cos(azimuth_rad - aspect_rad))
        
        # Scale to 0-255
        hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
        
        return hillshade
    
    def calculate_curvature(self):
        """Calculate plan and profile curvature"""
        # Calculate first derivatives
        fy, fx = np.gradient(self.dem_data, self.pixel_size)
        
        # Calculate second derivatives
        fxy, fxx = np.gradient(fx, self.pixel_size)
        fyy, fyx = np.gradient(fy, self.pixel_size)
        
        # Avoid division by zero
        fx2_fy2 = fx**2 + fy**2
        fx2_fy2 = np.where(fx2_fy2 == 0, 1e-8, fx2_fy2)
        
        # Plan curvature (perpendicular to slope direction)
        plan_curvature = (fxx * fy**2 - 2 * fxy * fx * fy + fyy * fx**2) / (fx2_fy2**1.5)
        
        # Profile curvature (parallel to slope direction)
        profile_curvature = (fxx * fx**2 + 2 * fxy * fx * fy + fyy * fy**2) / (fx2_fy2**1.5)
        
        return plan_curvature, profile_curvature
    
    def classify_terrain(self, slope_threshold=15, curvature_threshold=0.01):
        """Classify terrain into morphological units"""
        slope_deg, _ = self.calculate_slope_aspect()
        plan_curv, profile_curv = self.calculate_curvature()
        
        # Initialize classification array
        terrain_class = np.zeros_like(self.dem_data, dtype=np.int8)
        
        # Classification based on slope and curvature
        # 1: Ridge (convex profile, low slope or any slope with high convexity)
        ridge_mask = (profile_curv < -curvature_threshold) & (slope_deg < slope_threshold * 2)
        terrain_class[ridge_mask] = 1
        
        # 2: Valley (concave profile, low slope or any slope with high concavity)
        valley_mask = (profile_curv > curvature_threshold) & (slope_deg < slope_threshold * 2)
        terrain_class[valley_mask] = 2
        
        # 3: Steep slopes
        steep_mask = slope_deg > slope_threshold * 2
        terrain_class[steep_mask] = 3
        
        # 4: Moderate slopes
        moderate_mask = (slope_deg > slope_threshold) & (slope_deg <= slope_threshold * 2)
        terrain_class[moderate_mask & (terrain_class == 0)] = 4
        
        # 5: Flat areas (remaining areas with low slope)
        flat_mask = slope_deg <= slope_threshold
        terrain_class[flat_mask & (terrain_class == 0)] = 5
        
        # Create class names
        class_names = {
            0: 'Unclassified',
            1: 'Ridge',
            2: 'Valley',
            3: 'Steep Slope',
            4: 'Moderate Slope',
            5: 'Flat'
        }
        
        return terrain_class, class_names
    
    def watershed_analysis(self, pour_points=None):
        """Perform watershed delineation"""
        # Fill sinks in DEM
        filled_dem = self._fill_sinks(self.dem_data)
        
        # Calculate flow direction
        flow_dir = self._calculate_flow_direction(filled_dem)
        
        # Calculate flow accumulation
        flow_acc = self._calculate_flow_accumulation(flow_dir)
        
        # Extract stream network
        stream_threshold = np.percentile(flow_acc[flow_acc > 0], 95)
        streams = flow_acc > stream_threshold
        
        # If pour points provided, delineate watersheds
        watersheds = None
        if pour_points is not None:
            watersheds = self._delineate_watersheds(flow_dir, pour_points)
        
        return flow_dir, flow_acc, streams, watersheds
    
    def _fill_sinks(self, dem):
        """Fill sinks in DEM using morphological reconstruction"""
        # Create marker image (dem + small increment)
        marker = dem + 0.001
        marker[0, :] = dem[0, :]  # Keep boundary values
        marker[-1, :] = dem[-1, :]
        marker[:, 0] = dem[:, 0]
        marker[:, -1] = dem[:, -1]
        
        # Morphological reconstruction
        filled = dem.copy()
        while True:
            new_marker = np.maximum(marker, 
                                  np.minimum(ndimage.grey_dilation(marker, size=3), dem))
            if np.array_equal(marker, new_marker):
                break
            marker = new_marker
        
        return marker
    
    def _calculate_flow_direction(self, dem):
        """Calculate flow direction using D8 algorithm"""
        rows, cols = dem.shape
        flow_dir = np.zeros_like(dem, dtype=np.int8)
        
        # D8 flow directions (1-8, with 0 for no flow)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                     (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if np.isnan(dem[i, j]):
                    continue
                    
                max_slope = -1
                max_dir = 0
                
                for k, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(dem[ni, nj]):
                        slope = (dem[i, j] - dem[ni, nj]) / (self.pixel_size * np.sqrt(di**2 + dj**2))
                        if slope > max_slope:
                            max_slope = slope
                            max_dir = k + 1
                
                flow_dir[i, j] = max_dir
        
        return flow_dir
    
    def _calculate_flow_accumulation(self, flow_dir):
        """Calculate flow accumulation"""
        rows, cols = flow_dir.shape
        flow_acc = np.ones_like(flow_dir, dtype=np.float32)
        
        # D8 flow directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                     (1, 1), (1, 0), (1, -1), (0, -1)]
        
        # Process cells from highest to lowest elevation
        valid_mask = ~np.isnan(self.dem_data)
        valid_indices = np.where(valid_mask)
        elevations = self.dem_data[valid_indices]
        
        # Sort by elevation (descending)
        sorted_indices = np.argsort(elevations)[::-1]
        
        for idx in sorted_indices:
            i, j = valid_indices[0][idx], valid_indices[1][idx]
            
            if flow_dir[i, j] > 0:
                # Get downstream cell
                di, dj = directions[flow_dir[i, j] - 1]
                ni, nj = i + di, j + dj
                
                if 0 <= ni < rows and 0 <= nj < cols:
                    flow_acc[ni, nj] += flow_acc[i, j]
        
        return flow_acc
    
    def _delineate_watersheds(self, flow_dir, pour_points):
        """Delineate watersheds from pour points"""
        # This is a simplified version - full implementation would be more complex
        watersheds = np.zeros_like(flow_dir, dtype=np.int16)
        
        for i, point in enumerate(pour_points):
            # Convert point to grid coordinates
            col = int((point.x - self.transform[2]) / self.transform[0])
            row = int((point.y - self.transform[5]) / self.transform[4])
            
            if 0 <= row < flow_dir.shape[0] and 0 <= col < flow_dir.shape[1]:
                # Simple upstream tracing (simplified algorithm)
                watershed_id = i + 1
                self._trace_upstream(flow_dir, watersheds, row, col, watershed_id)
        
        return watersheds
    
    def _trace_upstream(self, flow_dir, watersheds, start_row, start_col, watershed_id):
        """Trace upstream from a point to delineate watershed"""
        # Simplified upstream tracing
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                     (1, 1), (1, 0), (1, -1), (0, -1)]
        
        stack = [(start_row, start_col)]
        visited = set()
        
        while stack:
            row, col = stack.pop()
            
            if (row, col) in visited or watersheds[row, col] != 0:
                continue
                
            visited.add((row, col))
            watersheds[row, col] = watershed_id
            
            # Find upstream cells
            for i, (di, dj) in enumerate(directions):
                ni, nj = row + di, col + dj
                
                if (0 <= ni < flow_dir.shape[0] and 0 <= nj < flow_dir.shape[1] and
                    flow_dir[ni, nj] == (i + 5) % 8 + 1):  # Points to current cell
                    stack.append((ni, nj))
    
    def viewshed_analysis(self, observer_points, observer_height=1.7, target_height=0.0,
                         max_distance=5000):
        """Calculate viewshed from observer points"""
        viewsheds = []
        
        for i, point in enumerate(observer_points):
            # Convert point to grid coordinates
            col = int((point.x - self.transform[2]) / self.transform[0])
            row = int((point.y - self.transform[5]) / self.transform[4])
            
            if not (0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]):
                continue
            
            observer_elev = self.dem_data[row, col] + observer_height
            
            # Calculate viewshed using line-of-sight analysis
            viewshed = self._calculate_viewshed_los(row, col, observer_elev, 
                                                   target_height, max_distance)
            
            viewsheds.append({
                'observer_id': i,
                'observer_x': point.x,
                'observer_y': point.y,
                'observer_elevation': observer_elev,
                'viewshed': viewshed
            })
        
        return viewsheds
    
    def _calculate_viewshed_los(self, obs_row, obs_col, obs_elev, target_height, max_distance):
        """Calculate viewshed using line-of-sight analysis"""
        rows, cols = self.dem_data.shape
        viewshed = np.zeros((rows, cols), dtype=bool)
        
        max_pixels = int(max_distance / self.pixel_size)
        
        # Check visibility in all directions
        for target_row in range(max(0, obs_row - max_pixels), 
                               min(rows, obs_row + max_pixels + 1)):
            for target_col in range(max(0, obs_col - max_pixels), 
                                   min(cols, obs_col + max_pixels + 1)):
                
                if target_row == obs_row and target_col == obs_col:
                    viewshed[target_row, target_col] = True
                    continue
                
                # Check if within max distance
                distance = np.sqrt((target_row - obs_row)**2 + (target_col - obs_col)**2) * self.pixel_size
                if distance > max_distance:
                    continue
                
                # Line-of-sight analysis
                if self._line_of_sight(obs_row, obs_col, obs_elev,
                                     target_row, target_col, target_height):
                    viewshed[target_row, target_col] = True
        
        return viewshed
    
    def _line_of_sight(self, obs_row, obs_col, obs_elev, target_row, target_col, target_height):
        """Check line of sight between observer and target"""
        target_elev = self.dem_data[target_row, target_col] + target_height
        
        if np.isnan(target_elev):
            return False
        
        # Get line coordinates
        line_coords = self._get_line_coords(obs_row, obs_col, target_row, target_col)
        
        # Check for obstructions along the line
        for i, (r, c) in enumerate(line_coords[1:-1], 1):  # Skip start and end points
            if not (0 <= r < self.dem_data.shape[0] and 0 <= c < self.dem_data.shape[1]):
                continue
                
            ground_elev = self.dem_data[r, c]
            if np.isnan(ground_elev):
                continue
            
            # Calculate required elevation for line of sight
            progress = i / (len(line_coords) - 1)
            required_elev = obs_elev + progress * (target_elev - obs_elev)
            
            if ground_elev > required_elev:
                return False
        
        return True
    
    def _get_line_coords(self, row0, col0, row1, col1):
        """Get coordinates along a line using Bresenham's algorithm"""
        coords = []
        
        dx = abs(col1 - col0)
        dy = abs(row1 - row0)
        sx = 1 if col0 < col1 else -1
        sy = 1 if row0 < row1 else -1
        err = dx - dy
        
        row, col = row0, col0
        
        while True:
            coords.append((int(row), int(col)))
            
            if row == row1 and col == col1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                col += sx
            if e2 < dx:
                err += dx
                row += sy
        
        return coords
    
    def export_raster(self, data, output_path, dtype=None):
        """Export numpy array as raster file"""
        if dtype is None:
            dtype = data.dtype
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=np.nan if dtype == np.float32 else None
        ) as dst:
            dst.write(data.astype(dtype), 1)
    
    def create_terrain_visualization(self, output_dir):
        """Create comprehensive terrain visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Original DEM
        im1 = axes[0].imshow(self.dem_data, cmap='terrain', aspect='equal')
        axes[0].set_title('Digital Elevation Model')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Elevation (m)')
        
        # 2. Slope
        slope_deg, _ = self.calculate_slope_aspect()
        im2 = axes[1].imshow(slope_deg, cmap='YlOrRd', aspect='equal')
        axes[1].set_title('Slope (degrees)')
        plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Slope (°)')
        
        # 3. Hillshade
        hillshade = self.calculate_hillshade()
        axes[2].imshow(hillshade, cmap='gray', aspect='equal')
        axes[2].set_title('Hillshade')
        
        # 4. Terrain Classification
        terrain_class, class_names = self.classify_terrain()
        im4 = axes[3].imshow(terrain_class, cmap='Set3', aspect='equal', vmin=0, vmax=5)
        axes[3].set_title('Terrain Classification')
        
        # 5. Curvature
        plan_curv, profile_curv = self.calculate_curvature()
        im5 = axes[4].imshow(plan_curv, cmap='RdBu', aspect='equal', 
                            vmin=np.nanpercentile(plan_curv, 5),
                            vmax=np.nanpercentile(plan_curv, 95))
        axes[4].set_title('Plan Curvature')
        plt.colorbar(im5, ax=axes[4], shrink=0.8, label='Curvature')
        
        # 6. Flow Accumulation
        _, flow_acc, streams, _ = self.watershed_analysis()
        log_flow_acc = np.log10(flow_acc + 1)
        im6 = axes[5].imshow(log_flow_acc, cmap='Blues', aspect='equal')
        axes[5].contour(streams, levels=[0.5], colors='red', linewidths=1)
        axes[5].set_title('Flow Accumulation + Streams')
        plt.colorbar(im6, ax=axes[5], shrink=0.8, label='Log Flow Accumulation')
        
        # Remove axis ticks for all subplots
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save visualization
        viz_output = os.path.join(output_dir, 'terrain_analysis.png')
        plt.savefig(viz_output, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Terrain visualization saved to: {viz_output}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Raster Terrain Analysis')
    parser.add_argument('--dem', required=True, help='Input DEM raster path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--analysis', 
                       choices=['slope', 'curvature', 'classification', 'watershed', 'viewshed', 'all'],
                       default='all', help='Type of analysis to perform')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TerrainAnalyzer(args.dem)
    
    # Load DEM
    if not analyzer.load_dem():
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    # Perform analyses
    if args.analysis in ['slope', 'all']:
        print("Calculating slope and aspect...")
        slope, aspect = analyzer.calculate_slope_aspect()
        analyzer.export_raster(slope, os.path.join(args.output, 'slope.tif'), np.float32)
        analyzer.export_raster(aspect, os.path.join(args.output, 'aspect.tif'), np.float32)
    
    if args.analysis in ['curvature', 'all']:
        print("Calculating curvature...")
        plan_curv, profile_curv = analyzer.calculate_curvature()
        analyzer.export_raster(plan_curv, os.path.join(args.output, 'plan_curvature.tif'), np.float32)
        analyzer.export_raster(profile_curv, os.path.join(args.output, 'profile_curvature.tif'), np.float32)
    
    if args.analysis in ['classification', 'all']:
        print("Classifying terrain...")
        terrain_class, class_names = analyzer.classify_terrain()
        analyzer.export_raster(terrain_class, os.path.join(args.output, 'terrain_classification.tif'), np.int8)
        
        # Export class names
        class_df = pd.DataFrame(list(class_names.items()), columns=['Value', 'Class'])
        class_df.to_csv(os.path.join(args.output, 'terrain_classes.csv'), index=False)
    
    if args.analysis in ['watershed', 'all']:
        print("Performing watershed analysis...")
        flow_dir, flow_acc, streams, _ = analyzer.watershed_analysis()
        analyzer.export_raster(flow_dir, os.path.join(args.output, 'flow_direction.tif'), np.int8)
        analyzer.export_raster(flow_acc, os.path.join(args.output, 'flow_accumulation.tif'), np.float32)
        analyzer.export_raster(streams.astype(np.int8), os.path.join(args.output, 'streams.tif'), np.int8)
    
    # Create visualization if requested
    if args.visualize:
        print("Creating terrain visualization...")
        analyzer.create_terrain_visualization(args.output)
    
    print(f"\nTerrain analysis completed successfully!")
    print(f"Results exported to: {args.output}")

if __name__ == "__main__":
    main()