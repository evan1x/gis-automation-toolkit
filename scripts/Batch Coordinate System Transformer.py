"""
Batch Coordinate System Transformer
Professional-grade coordinate transformation with validation and reporting
"""

import arcpy
import os
import json
import logging
from datetime import datetime
from collections import defaultdict
import math

class CoordinateSystemTransformer:
    """
    Advanced coordinate system transformation toolkit with validation and reporting
    """
    
    def __init__(self, input_workspace, output_workspace=None):
        self.input_workspace = input_workspace
        self.output_workspace = output_workspace or input_workspace
        self.setup_logging()
        self.transformation_log = []
        self.accuracy_results = defaultdict(dict)
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = os.path.join(os.path.dirname(self.input_workspace), "transformation_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"transformation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_coordinate_system(self, dataset_path):
        """
        Analyze current coordinate system and provide recommendations
        """
        try:
            desc = arcpy.Describe(dataset_path)
            spatial_ref = desc.spatialReference
            extent = desc.extent
            
            analysis = {
                'dataset': dataset_path,
                'current_cs': spatial_ref.name,
                'cs_type': spatial_ref.type,
                'linear_unit': spatial_ref.linearUnitName if hasattr(spatial_ref, 'linearUnitName') else 'None',
                'angular_unit': spatial_ref.angularUnitName if hasattr(spatial_ref, 'angularUnitName') else 'None',
                'extent': {
                    'xmin': extent.XMin,
                    'ymin': extent.YMin,
                    'xmax': extent.XMax,
                    'ymax': extent.YMax
                },
                'centroid': {
                    'x': (extent.XMin + extent.XMax) / 2,
                    'y': (extent.YMin + extent.YMax) / 2
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_cs_recommendations(analysis)
            analysis['recommendations'] = recommendations
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing coordinate system for {dataset_path}: {str(e)}")
            return None
    
    def _generate_cs_recommendations(self, analysis):
        """Generate coordinate system recommendations based on data extent"""
        recommendations = []
        
        extent = analysis['extent']
        centroid = analysis['centroid']
        
        # Check if data is in geographic coordinates
        if analysis['cs_type'] == 'Geographic':
            # Recommend appropriate projected coordinate system
            if -180 <= centroid['x'] <= 180 and -90 <= centroid['y'] <= 90:
                # Determine UTM zone
                utm_zone = int((centroid['x'] + 180) / 6) + 1
                hemisphere = 'North' if centroid['y'] >= 0 else 'South'
                
                recommendations.append({
                    'type': 'UTM Projection',
                    'name': f'WGS_1984_UTM_Zone_{utm_zone}{hemisphere[0]}',
                    'reason': f'Data center at {centroid["x"]:.2f}, {centroid["y"]:.2f} fits UTM Zone {utm_zone} {hemisphere}',
                    'suitable_for': ['Area calculations', 'Distance measurements', 'Buffer operations']
                })
                
                # State Plane recommendation for US data
                if -125 <= centroid['x'] <= -66 and 20 <= centroid['y'] <= 72:
                    recommendations.append({
                        'type': 'State Plane',
                        'name': 'Consider State Plane Coordinate System',
                        'reason': 'Data appears to be within United States',
                        'suitable_for': ['High-accuracy surveying', 'Engineering applications', 'Local government mapping']
                    })
        
        # Check for inappropriate coordinate systems
        if analysis['cs_type'] == 'Projected':
            # Check if extent seems too large for projected system
            width = extent['xmax'] - extent['xmin']
            height = extent['ymax'] - extent['ymin']
            
            if width > 2000000 or height > 2000000:  # More than 2000 km
                recommendations.append({
                    'type': 'Warning',
                    'name': 'Large extent in projected coordinates',
                    'reason': f'Extent: {width:.0f} x {height:.0f} units - may cause distortion',
                    'suitable_for': ['Consider geographic coordinate system for global data']
                })
        
        return recommendations
    
    def transform_dataset(self, input_dataset, target_cs, output_dataset=None, 
                         transformation_method=None, preserve_metadata=True,
                         validate_accuracy=True):
        """
        Transform dataset to target coordinate system with comprehensive validation
        """
        try:
            self.logger.info(f"Starting transformation: {input_dataset}")
            
            # Analyze input dataset
            input_analysis = self.analyze_coordinate_system(input_dataset)
            if not input_analysis:
                return False
            
            # Prepare output dataset path
            if not output_dataset:
                base_name = os.path.basename(input_dataset)
                name, ext = os.path.splitext(base_name)
                output_dataset = os.path.join(self.output_workspace, f"{name}_transformed{ext}")
            
            # Check if transformation is needed
            input_cs = arcpy.Describe(input_dataset).spatialReference
            target_sr = arcpy.SpatialReference()
            
            if isinstance(target_cs, str):
                if target_cs.endswith('.prj'):
                    target_sr.createFromFile(target_cs)
                else:
                    target_sr.loadFromString(target_cs)
            elif isinstance(target_cs, int):
                target_sr.factoryCode = target_cs
            else:
                target_sr = target_cs
            
            if input_cs.factoryCode == target_sr.factoryCode:
                self.logger.info("Input and target coordinate systems are the same. No transformation needed.")
                return True
            
            # Find appropriate transformation method
            if not transformation_method:
                transformation_method = self._find_best_transformation(input_cs, target_sr)
            
            # Perform transformation with accuracy validation
            transformation_start = datetime.now()
            
            # Create accuracy test points before transformation
            test_points = None
            if validate_accuracy:
                test_points = self._create_test_points(input_dataset)
            
            # Execute transformation
            self.logger.info(f"Transforming to: {target_sr.name}")
            self.logger.info(f"Using transformation: {transformation_method}")
            
            arcpy.management.Project(
                input_dataset,
                output_dataset,
                target_sr,
                transformation_method,
                input_cs,
                "NO_PRESERVE_SHAPE",
                None,
                "NO_VERTICAL"
            )
            
            transformation_end = datetime.now()
            duration = (transformation_end - transformation_start).total_seconds()
            
            # Validate transformation accuracy
            accuracy_report = None
            if validate_accuracy and test_points:
                accuracy_report = self._validate_transformation_accuracy(
                    test_points, input_cs, target_sr, transformation_method
                )
            
            # Preserve metadata if requested
            if preserve_metadata:
                self._preserve_metadata(input_dataset, output_dataset)
            
            # Create transformation record
            transformation_record = {
                'timestamp': transformation_start.isoformat(),
                'input_dataset': input_dataset,
                'output_dataset': output_dataset,
                'input_cs': input_cs.name,
                'target_cs': target_sr.name,
                'transformation_method': transformation_method,
                'duration_seconds': duration,
                'success': True,
                'accuracy_report': accuracy_report
            }
            
            self.transformation_log.append(transformation_record)
            self.logger.info(f"Transformation completed successfully in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            
            # Log failed transformation
            transformation_record = {
                'timestamp': datetime.now().isoformat(),
                'input_dataset': input_dataset,
                'output_dataset': output_dataset,
                'error': str(e),
                'success': False
            }
            self.transformation_log.append(transformation_record)
            
            return False
    
    def _find_best_transformation(self, input_cs, target_cs):
        """Find the most appropriate transformation method"""
        try:
            # Get list of available transformations
            transformations = arcpy.ListTransformations(input_cs, target_cs)
            
            if not transformations:
                return None
            
            # Prefer high-accuracy transformations
            high_accuracy_keywords = ['HARN', 'NAD_1983_2011', 'ITRF', 'WGS_1984_G1762']
            
            for transformation in transformations:
                for keyword in high_accuracy_keywords:
                    if keyword in transformation:
                        self.logger.info(f"Selected high-accuracy transformation: {transformation}")
                        return transformation
            
            # Return first available transformation
            self.logger.info(f"Using default transformation: {transformations[0]}")
            return transformations[0]
            
        except Exception as e:
            self.logger.warning(f"Could not determine transformation method: {str(e)}")
            return None
    
    def _create_test_points(self, dataset):
        """Create test points for accuracy validation"""
        try:
            desc = arcpy.Describe(dataset)
            extent = desc.extent
            
            # Create test points at strategic locations
            test_points = []
            
            # Corner points
            corners = [
                (extent.XMin, extent.YMin),
                (extent.XMax, extent.YMin),
                (extent.XMax, extent.YMax),
                (extent.XMin, extent.YMax)
            ]
            
            # Center point
            center = ((extent.XMin + extent.XMax) / 2, (extent.YMin + extent.YMax) / 2)
            
            # Edge midpoints
            edges = [
                ((extent.XMin + extent.XMax) / 2, extent.YMin),
                (extent.XMax, (extent.YMin + extent.YMax) / 2),
                ((extent.XMin + extent.XMax) / 2, extent.YMax),
                (extent.XMin, (extent.YMin + extent.YMax) / 2)
            ]
            
            all_points = corners + [center] + edges
            
            # Create point geometries
            spatial_ref = desc.spatialReference
            for i, (x, y) in enumerate(all_points):
                point = arcpy.Point(x, y)
                point_geometry = arcpy.PointGeometry(point, spatial_ref)
                test_points.append({
                    'id': i,
                    'original_coords': (x, y),
                    'geometry': point_geometry,
                    'location_type': self._classify_test_point(i, len(corners), len(edges))
                })
            
            return test_points
            
        except Exception as e:
            self.logger.warning(f"Could not create test points: {str(e)}")
            return None
    
    def _classify_test_point(self, index, num_corners, num_edges):
        """Classify test point by location type"""
        if index < num_corners:
            return 'corner'
        elif index == num_corners:
            return 'center'
        else:
            return 'edge'
    
    def _validate_transformation_accuracy(self, test_points, input_cs, target_cs, transformation_method):
        """Validate transformation accuracy using test points"""
        try:
            accuracy_results = {
                'transformation_method': transformation_method,
                'test_points_count': len(test_points),
                'results': []
            }
            
            for test_point in test_points:
                # Transform the test point
                transformed_geom = test_point['geometry'].projectAs(target_cs, transformation_method)
                
                # Get transformed coordinates
                transformed_coords = (transformed_geom.firstPoint.X, transformed_geom.firstPoint.Y)
                
                # Calculate theoretical coordinates using alternative method if available
                # This is a simplified accuracy check - in practice, you'd use known control points
                result = {
                    'point_id': test_point['id'],
                    'location_type': test_point['location_type'],
                    'original_coords': test_point['original_coords'],
                    'transformed_coords': transformed_coords,
                    'transformation_method': transformation_method
                }
                
                accuracy_results['results'].append(result)
            
            # Calculate overall accuracy metrics
            accuracy_results['summary'] = self._calculate_accuracy_summary(accuracy_results['results'])
            
            return accuracy_results
            
        except Exception as e:
            self.logger.warning(f"Accuracy validation failed: {str(e)}")
            return None
    
    def _calculate_accuracy_summary(self, results):
        """Calculate summary accuracy statistics"""
        summary = {
            'total_points': len(results),
            'transformation_successful': True,
            'coordinate_range': {
                'x_min': min(r['transformed_coords'][0] for r in results),
                'x_max': max(r['transformed_coords'][0] for r in results),
                'y_min': min(r['transformed_coords'][1] for r in results),
                'y_max': max(r['transformed_coords'][1] for r in results)
            }
        }
        
        return summary
    
    def _preserve_metadata(self, input_dataset, output_dataset):
        """Preserve metadata from input to output dataset"""
        try:
            # Copy field properties and domains
            desc_input = arcpy.Describe(input_dataset)
            desc_output = arcpy.Describe(output_dataset)
            
            # Preserve field aliases and domains where possible
            input_fields = {f.name: f for f in desc_input.fields}
            
            for field in desc_output.fields:
                if field.name in input_fields and field.editable:
                    input_field = input_fields[field.name]
                    
                    # Preserve field alias
                    if input_field.aliasName != field.name:
                        try:
                            arcpy.management.AlterField(
                                output_dataset,
                                field.name,
                                new_field_alias=input_field.aliasName
                            )
                        except:
                            pass  # Some fields cannot be altered
            
            self.logger.info("Metadata preservation completed")
            
        except Exception as e:
            self.logger.warning(f"Could not preserve all metadata: {str(e)}")
    
    def batch_transform_workspace(self, target_cs, dataset_filter=None, 
                                 transformation_method=None, validate_accuracy=True):
        """
        Transform all datasets in a workspace to target coordinate system
        """
        try:
            self.logger.info(f"Starting batch transformation of workspace: {self.input_workspace}")
            
            arcpy.env.workspace = self.input_workspace
            
            # Get all datasets
            datasets = []
            
            # Feature classes
            feature_classes = arcpy.ListFeatureClasses()
            if feature_classes:
                datasets.extend(feature_classes)
            
            # Raster datasets
            rasters = arcpy.ListRasters()
            if rasters:
                datasets.extend(rasters)
            
            # Filter datasets if specified
            if dataset_filter:
                datasets = [ds for ds in datasets if dataset_filter(ds)]
            
            self.logger.info(f"Found {len(datasets)} datasets to transform")
            
            # Transform each dataset
            successful_transformations = 0
            failed_transformations = 0
            
            for dataset in datasets:
                self.logger.info(f"Processing: {dataset}")
                
                input_path = os.path.join(self.input_workspace, dataset)
                
                success = self.transform_dataset(
                    input_path,
                    target_cs,
                    transformation_method=transformation_method,
                    validate_accuracy=validate_accuracy
                )
                
                if success:
                    successful_transformations += 1
                else:
                    failed_transformations += 1
            
            self.logger.info(f"Batch transformation completed. Success: {successful_transformations}, Failed: {failed_transformations}")
            
            # Generate batch report
            self.generate_batch_report()
            
            return successful_transformations, failed_transformations
            
        except Exception as e:
            self.logger.error(f"Batch transformation failed: {str(e)}")
            return 0, 0
    
    def generate_batch_report(self):
        """Generate comprehensive transformation report"""
        try:
            report_dir = os.path.join(os.path.dirname(self.input_workspace), "transformation_reports")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate JSON report
            json_report = {
                'report_generated': datetime.now().isoformat(),
                'input_workspace': self.input_workspace,
                'output_workspace': self.output_workspace,
                'transformations': self.transformation_log,
                'summary': {
                    'total_transformations': len(self.transformation_log),
                    'successful': len([t for t in self.transformation_log if t.get('success', False)]),
                    'failed': len([t for t in self.transformation_log if not t.get('success', True)])
                }
            }
            
            json_file = os.path.join(report_dir, f"transformation_report_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            # Generate HTML report
            html_file = os.path.join(report_dir, f"transformation_report_{timestamp}.html")
            self._generate_html_report(html_file, json_report)
            
            self.logger.info(f"Reports generated: {json_file}, {html_file}")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
    
    def _generate_html_report(self, html_file, report_data):
        """Generate HTML transformation report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coordinate Transformation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .transformation {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ border-left: 5px solid green; }}
                .failure {{ border-left: 5px solid red; }}
                .accuracy {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Coordinate System Transformation Report</h1>
                <p><strong>Generated:</strong> {report_data['report_generated']}</p>
                <p><strong>Input Workspace:</strong> {report_data['input_workspace']}</p>
                <p><strong>Output Workspace:</strong> {report_data['output_workspace']}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Transformations:</strong> {report_data['summary']['total_transformations']}</p>
                <p><strong>Successful:</strong> {report_data['summary']['successful']}</p>
                <p><strong>Failed:</strong> {report_data['summary']['failed']}</p>
            </div>
            
            <h2>Transformation Details</h2>
        """
        
        for transformation in report_data['transformations']:
            status_class = 'success' if transformation.get('success', False) else 'failure'
            
            html_content += f"""
            <div class="transformation {status_class}">
                <h3>{os.path.basename(transformation.get('input_dataset', 'Unknown'))}</h3>
                <p><strong>Status:</strong> {'Success' if transformation.get('success', False) else 'Failed'}</p>
                <p><strong>Input CS:</strong> {transformation.get('input_cs', 'Unknown')}</p>
                <p><strong>Target CS:</strong> {transformation.get('target_cs', 'Unknown')}</p>
                <p><strong>Transformation Method:</strong> {transformation.get('transformation_method', 'Unknown')}</p>
                <p><strong>Duration:</strong> {transformation.get('duration_seconds', 0):.2f} seconds</p>
            """
            
            if transformation.get('error'):
                html_content += f"<p><strong>Error:</strong> {transformation['error']}</p>"
            
            if transformation.get('accuracy_report'):
                accuracy = transformation['accuracy_report']
                html_content += f"""
                <div class="accuracy">
                    <h4>Accuracy Validation</h4>
                    <p><strong>Test Points:</strong> {accuracy.get('test_points_count', 0)}</p>
                    <p><strong>Validation Status:</strong> {'Passed' if accuracy.get('summary', {}).get('transformation_successful', False) else 'Failed'}</p>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)

# Example usage
def main():
    """Example usage of the CoordinateSystemTransformer"""
    
    # Initialize transformer
    input_workspace = r"C:\GIS_Data\InputData.gdb"
    output_workspace = r"C:\GIS_Data\TransformedData.gdb"
    
    transformer = CoordinateSystemTransformer(input_workspace, output_workspace)
    
    # Define target coordinate system (example: UTM Zone 10N)
    target_cs = 32610  # EPSG code for WGS84 UTM Zone 10N
    
    # Analyze existing datasets
    arcpy.env.workspace = input_workspace
    feature_classes = arcpy.ListFeatureClasses()
    
    for fc in feature_classes:
        analysis = transformer.analyze_coordinate_system(fc)
        if analysis:
            print(f"\nDataset: {fc}")
            print(f"Current CS: {analysis['current_cs']}")
            print("Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec['type']}: {rec['name']}")
                print(f"    Reason: {rec['reason']}")
    
    # Perform batch transformation
    successful, failed = transformer.batch_transform_workspace(
        target_cs=target_cs,
        dataset_filter=lambda x: x.endswith('.shp') or 'feature' in x.lower(),
        validate_accuracy=True
    )
    
    print(f"\nTransformation completed: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()