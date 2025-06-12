"""
Automated Spatial Data Validator
Comprehensive data quality assessment and automated fixing for spatial datasets
"""

import arcpy
import os
import json
import logging
from datetime import datetime
from collections import defaultdict
import traceback

class SpatialDataValidator:
    """
    Comprehensive spatial data validation and quality assurance toolkit
    """
    
    def __init__(self, workspace, output_dir=None):
        self.workspace = workspace
        self.output_dir = output_dir or os.path.join(os.path.dirname(workspace), "validation_reports")
        self.setup_logging()
        self.validation_results = defaultdict(list)
        self.fixes_applied = defaultdict(int)
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.output_dir, f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self, dataset_path, fix_issues=True, validation_config=None):
        """
        Main validation function for a single dataset
        """
        try:
            self.logger.info(f"Starting validation for: {dataset_path}")
            
            # Check if dataset exists and is accessible
            if not arcpy.Exists(dataset_path):
                self.validation_results[dataset_path].append({
                    'test': 'Dataset Existence',
                    'status': 'FAIL',
                    'message': 'Dataset does not exist or is not accessible',
                    'severity': 'CRITICAL'
                })
                return
            
            # Get dataset properties
            desc = arcpy.Describe(dataset_path)
            dataset_type = desc.dataType
            
            self.logger.info(f"Dataset type: {dataset_type}")
            
            # Run validation tests based on dataset type
            if dataset_type in ['FeatureClass', 'ShapeFile']:
                self._validate_feature_class(dataset_path, fix_issues, validation_config)
            elif dataset_type == 'RasterDataset':
                self._validate_raster(dataset_path, fix_issues, validation_config)
            elif dataset_type == 'Table':
                self._validate_table(dataset_path, fix_issues, validation_config)
            
            self.logger.info(f"Validation completed for: {dataset_path}")
            
        except Exception as e:
            self.logger.error(f"Error validating {dataset_path}: {str(e)}")
            self.validation_results[dataset_path].append({
                'test': 'Validation Process',
                'status': 'ERROR',
                'message': f'Validation failed: {str(e)}',
                'severity': 'CRITICAL'
            })
    
    def _validate_feature_class(self, fc_path, fix_issues=True, config=None):
        """Comprehensive feature class validation"""
        
        # 1. Geometry Validation
        self._check_geometry_validity(fc_path, fix_issues)
        
        # 2. Coordinate System Check
        self._check_coordinate_system(fc_path)
        
        # 3. Empty Features Check
        self._check_empty_features(fc_path, fix_issues)
        
        # 4. Duplicate Geometry Check
        self._check_duplicate_geometries(fc_path, fix_issues)
        
        # 5. Attribute Validation
        self._validate_attributes(fc_path, fix_issues, config)
        
        # 6. Topology Checks
        self._check_basic_topology(fc_path, fix_issues)
        
        # 7. Extent and Scale Validation
        self._check_extent_validity(fc_path)
        
        # 8. Field Schema Validation
        self._validate_field_schema(fc_path, config)
    
    def _check_geometry_validity(self, fc_path, fix_issues=True):
        """Check for invalid geometries and optionally fix them"""
        try:
            self.logger.info("Checking geometry validity...")
            
            invalid_count = 0
            fixed_count = 0
            
            with arcpy.da.UpdateCursor(fc_path, ['SHAPE@', 'OID@']) as cursor:
                for row in cursor:
                    geometry = row[0]
                    oid = row[1]
                    
                    if geometry is None:
                        invalid_count += 1
                        self.validation_results[fc_path].append({
                            'test': 'Geometry Validity',
                            'status': 'FAIL',
                            'message': f'NULL geometry found at OID {oid}',
                            'severity': 'HIGH',
                            'oid': oid
                        })
                        continue
                    
                    # Check if geometry is valid
                    if hasattr(geometry, 'isValid') and not geometry.isValid:
                        invalid_count += 1
                        
                        if fix_issues:
                            try:
                                # Attempt to fix geometry
                                fixed_geometry = geometry.buffer(0)  # Often fixes self-intersections
                                if fixed_geometry.isValid:
                                    row[0] = fixed_geometry
                                    cursor.updateRow(row)
                                    fixed_count += 1
                                    self.validation_results[fc_path].append({
                                        'test': 'Geometry Validity',
                                        'status': 'FIXED',
                                        'message': f'Invalid geometry fixed at OID {oid}',
                                        'severity': 'MEDIUM',
                                        'oid': oid
                                    })
                                else:
                                    self.validation_results[fc_path].append({
                                        'test': 'Geometry Validity',
                                        'status': 'FAIL',
                                        'message': f'Invalid geometry could not be fixed at OID {oid}',
                                        'severity': 'HIGH',
                                        'oid': oid
                                    })
                            except Exception as e:
                                self.validation_results[fc_path].append({
                                    'test': 'Geometry Validity',
                                    'status': 'FAIL',
                                    'message': f'Error fixing geometry at OID {oid}: {str(e)}',
                                    'severity': 'HIGH',
                                    'oid': oid
                                })
                        else:
                            self.validation_results[fc_path].append({
                                'test': 'Geometry Validity',
                                'status': 'FAIL',
                                'message': f'Invalid geometry found at OID {oid}',
                                'severity': 'HIGH',
                                'oid': oid
                            })
            
            if invalid_count == 0:
                self.validation_results[fc_path].append({
                    'test': 'Geometry Validity',
                    'status': 'PASS',
                    'message': 'All geometries are valid',
                    'severity': 'INFO'
                })
            
            self.fixes_applied[fc_path] += fixed_count
            self.logger.info(f"Geometry check complete. Invalid: {invalid_count}, Fixed: {fixed_count}")
            
        except Exception as e:
            self.logger.error(f"Error in geometry validation: {str(e)}")
            self.validation_results[fc_path].append({
                'test': 'Geometry Validity',
                'status': 'ERROR',
                'message': f'Geometry validation failed: {str(e)}',
                'severity': 'CRITICAL'
            })
    
    def _check_coordinate_system(self, fc_path):
        """Validate coordinate system information"""
        try:
            desc = arcpy.Describe(fc_path)
            spatial_ref = desc.spatialReference
            
            if spatial_ref.name == "Unknown":
                self.validation_results[fc_path].append({
                    'test': 'Coordinate System',
                    'status': 'FAIL',
                    'message': 'Coordinate system not defined',
                    'severity': 'HIGH'
                })
            else:
                self.validation_results[fc_path].append({
                    'test': 'Coordinate System',
                    'status': 'PASS',
                    'message': f'Coordinate system: {spatial_ref.name}',
                    'severity': 'INFO'
                })
                
                # Check for appropriate coordinate system based on extent
                extent = desc.extent
                if spatial_ref.type == "Geographic":
                    if (extent.XMax - extent.XMin > 180) or (extent.YMax - extent.YMin > 90):
                        self.validation_results[fc_path].append({
                            'test': 'Coordinate System Appropriateness',
                            'status': 'WARNING',
                            'message': 'Geographic coordinate system with suspicious extent values',
                            'severity': 'MEDIUM'
                        })
                        
        except Exception as e:
            self.validation_results[fc_path].append({
                'test': 'Coordinate System',
                'status': 'ERROR',
                'message': f'Error checking coordinate system: {str(e)}',
                'severity': 'MEDIUM'
            })
    
    def _check_empty_features(self, fc_path, fix_issues=True):
        """Check for features with empty geometries"""
        try:
            empty_count = 0
            deleted_count = 0
            
            with arcpy.da.UpdateCursor(fc_path, ['SHAPE@', 'OID@']) as cursor:
                for row in cursor:
                    geometry = row[0]
                    oid = row[1]
                    
                    if geometry is None or geometry.area == 0 or geometry.length == 0:
                        empty_count += 1
                        
                        if fix_issues:
                            cursor.deleteRow()
                            deleted_count += 1
                            self.validation_results[fc_path].append({
                                'test': 'Empty Features',
                                'status': 'FIXED',
                                'message': f'Empty feature deleted at OID {oid}',
                                'severity': 'MEDIUM',
                                'oid': oid
                            })
                        else:
                            self.validation_results[fc_path].append({
                                'test': 'Empty Features',
                                'status': 'FAIL',
                                'message': f'Empty feature found at OID {oid}',
                                'severity': 'MEDIUM',
                                'oid': oid
                            })
            
            if empty_count == 0:
                self.validation_results[fc_path].append({
                    'test': 'Empty Features',
                    'status': 'PASS',
                    'message': 'No empty features found',
                    'severity': 'INFO'
                })
            
            self.fixes_applied[fc_path] += deleted_count
            
        except Exception as e:
            self.validation_results[fc_path].append({
                'test': 'Empty Features',
                'status': 'ERROR',
                'message': f'Error checking empty features: {str(e)}',
                'severity': 'MEDIUM'
            })
    
    def _check_duplicate_geometries(self, fc_path, fix_issues=True):
        """Check for duplicate geometries"""
        try:
            self.logger.info("Checking for duplicate geometries...")
            
            # Use Find Identical tool to identify duplicates
            temp_table = "in_memory\\duplicates_temp"
            
            arcpy.management.FindIdentical(
                fc_path, temp_table, 
                ["Shape"], 
                output_record_option="ONLY_DUPLICATES"
            )
            
            duplicate_count = int(arcpy.management.GetCount(temp_table)[0])
            
            if duplicate_count > 0:
                if fix_issues:
                    # Get list of duplicate OIDs to delete (keep first occurrence)
                    duplicate_oids = set()
                    with arcpy.da.SearchCursor(temp_table, ["IN_FID", "FEAT_SEQ"]) as cursor:
                        for row in cursor:
                            if row[1] > 1:  # Keep first occurrence (FEAT_SEQ = 1)
                                duplicate_oids.add(row[0])
                    
                    # Delete duplicates
                    deleted_count = 0
                    with arcpy.da.UpdateCursor(fc_path, ["OID@"]) as cursor:
                        for row in cursor:
                            if row[0] in duplicate_oids:
                                cursor.deleteRow()
                                deleted_count += 1
                    
                    self.validation_results[fc_path].append({
                        'test': 'Duplicate Geometries',
                        'status': 'FIXED',
                        'message': f'{deleted_count} duplicate geometries removed',
                        'severity': 'MEDIUM'
                    })
                    self.fixes_applied[fc_path] += deleted_count
                else:
                    self.validation_results[fc_path].append({
                        'test': 'Duplicate Geometries',
                        'status': 'FAIL',
                        'message': f'{duplicate_count} duplicate geometries found',
                        'severity': 'MEDIUM'
                    })
            else:
                self.validation_results[fc_path].append({
                    'test': 'Duplicate Geometries',
                    'status': 'PASS',
                    'message': 'No duplicate geometries found',
                    'severity': 'INFO'
                })
            
            # Cleanup
            if arcpy.Exists(temp_table):
                arcpy.management.Delete(temp_table)
                
        except Exception as e:
            self.validation_results[fc_path].append({
                'test': 'Duplicate Geometries',
                'status': 'ERROR',
                'message': f'Error checking duplicates: {str(e)}',
                'severity': 'MEDIUM'
            })
    
    def _validate_attributes(self, fc_path, fix_issues=True, config=None):
        """Validate attribute data quality"""
        try:
            desc = arcpy.Describe(fc_path)
            fields = desc.fields
            
            for field in fields:
                if field.type in ['String', 'Integer', 'Double', 'Single']:
                    self._check_field_completeness(fc_path, field.name, fix_issues)
                    
                    if field.type == 'String':
                        self._check_string_field_quality(fc_path, field.name, fix_issues)
                    elif field.type in ['Integer', 'Double', 'Single']:
                        self._check_numeric_field_quality(fc_path, field.name, fix_issues)
                        
        except Exception as e:
            self.validation_results[fc_path].append({
                'test': 'Attribute Validation',
                'status': 'ERROR',
                'message': f'Error in attribute validation: {str(e)}',
                'severity': 'MEDIUM'
            })
    
    def _check_field_completeness(self, fc_path, field_name, fix_issues=True):
        """Check for null/empty values in fields"""
        try:
            null_count = 0
            total_count = 0
            
            with arcpy.da.SearchCursor(fc_path, [field_name, 'OID@']) as cursor:
                for row in cursor:
                    total_count += 1
                    if row[0] is None or (isinstance(row[0], str) and row[0].strip() == ''):
                        null_count += 1
            
            if null_count > 0:
                completion_rate = ((total_count - null_count) / total_count) * 100
                severity = 'HIGH' if completion_rate < 80 else 'MEDIUM' if completion_rate < 95 else 'LOW'
                
                self.validation_results[fc_path].append({
                    'test': f'Field Completeness - {field_name}',
                    'status': 'WARNING',
                    'message': f'{null_count} null/empty values ({completion_rate:.1f}% complete)',
                    'severity': severity
                })
            else:
                self.validation_results[fc_path].append({
                    'test': f'Field Completeness - {field_name}',
                    'status': 'PASS',
                    'message': 'Field is 100% complete',
                    'severity': 'INFO'
                })
                
        except Exception as e:
            self.validation_results[fc_path].append({
                'test': f'Field Completeness - {field_name}',
                'status': 'ERROR',
                'message': f'Error checking field completeness: {str(e)}',
                'severity': 'LOW'
            })
    
    def generate_report(self, output_format='html'):
        """Generate comprehensive validation report"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if output_format.lower() == 'html':
                self._generate_html_report(timestamp)
            elif output_format.lower() == 'json':
                self._generate_json_report(timestamp)
            else:
                self._generate_text_report(timestamp)
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
    
    def _generate_html_report(self, timestamp):
        """Generate HTML validation report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spatial Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .dataset {{ margin: 20px 0; border: 1px solid #ccc; border-radius: 5px; }}
                .dataset-header {{ background-color: #e8e8e8; padding: 10px; font-weight: bold; }}
                .test-result {{ padding: 10px; border-bottom: 1px solid #eee; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                .warning {{ color: orange; }}
                .error {{ color: purple; }}
                .fixed {{ color: blue; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Spatial Data Validation Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Workspace:</strong> {self.workspace}</p>
            </div>
        """
        
        # Summary statistics
        total_tests = sum(len(results) for results in self.validation_results.values())
        total_fixes = sum(self.fixes_applied.values())
        
        html_content += f"""
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Datasets Validated:</strong> {len(self.validation_results)}</p>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Total Fixes Applied:</strong> {total_fixes}</p>
            </div>
        """
        
        # Individual dataset results
        for dataset, results in self.validation_results.items():
            html_content += f"""
                <div class="dataset">
                    <div class="dataset-header">{os.path.basename(dataset)}</div>
            """
            
            for result in results:
                status_class = result['status'].lower()
                html_content += f"""
                    <div class="test-result">
                        <strong>{result['test']}:</strong> 
                        <span class="{status_class}">{result['status']}</span> - 
                        {result['message']}
                        <small> (Severity: {result['severity']})</small>
                    </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        report_file = os.path.join(self.output_dir, f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {report_file}")

# Example usage and configuration
def main():
    """Example usage of the SpatialDataValidator"""
    
    # Configuration for validation rules
    validation_config = {
        'required_fields': ['Name', 'Type', 'Status'],
        'numeric_ranges': {
            'Population': {'min': 0, 'max': 10000000},
            'Area': {'min': 0, 'max': None}
        },
        'string_patterns': {
            'Email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'Phone': r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
        }
    }
    
    # Initialize validator
    workspace = r"C:\GIS_Data\MyGeodatabase.gdb"
    validator = SpatialDataValidator(workspace)
    
    # Get all feature classes in workspace
    arcpy.env.workspace = workspace
    feature_classes = arcpy.ListFeatureClasses()
    
    # Validate each feature class
    for fc in feature_classes:
        validator.validate_dataset(fc, fix_issues=True, validation_config=validation_config)
    
    # Generate comprehensive report
    validator.generate_report('html')
    
    print("Validation complete! Check the output directory for detailed reports.")

if __name__ == "__main__":
    main()