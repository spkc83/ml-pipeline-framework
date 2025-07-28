"""
CSV Data Profiler for ML Pipeline Framework

This module provides comprehensive data profiling capabilities for CSV files,
including data quality analysis, type detection, and statistical summaries.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import chardet
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CSVDataProfiler:
    """
    Comprehensive data profiler for CSV files with quality analysis and reporting.
    """
    
    def __init__(self, file_paths: List[str], sample_size: int = 10000):
        """
        Initialize CSV data profiler.
        
        Args:
            file_paths: List of CSV file paths to profile
            sample_size: Number of rows to sample for analysis
        """
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.sample_size = sample_size
        self.profile_results = {}
        
    def detect_file_properties(self, file_path: str) -> Dict[str, Any]:
        """
        Detect encoding, delimiter, and basic file properties.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dict with detected properties
        """
        properties = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / 1024**2,
            'modified_date': datetime.fromtimestamp(os.path.getmtime(file_path))
        }
        
        # Detect encoding
        try:
            with open(file_path, 'rb') as f:
                raw_sample = f.read(50000)  # Read 50KB for detection
            
            encoding_result = chardet.detect(raw_sample)
            properties['encoding'] = encoding_result['encoding']
            properties['encoding_confidence'] = encoding_result['confidence']
            
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path}: {e}")
            properties['encoding'] = 'utf-8'
            properties['encoding_confidence'] = 0.0
        
        # Detect delimiter
        try:
            with open(file_path, 'r', encoding=properties['encoding']) as f:
                sample_lines = [f.readline() for _ in range(10)]
            
            # Count potential delimiters
            delimiters = [',', ';', '\t', '|', ' ']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                counts = [line.count(delimiter) for line in sample_lines if line.strip()]
                if counts:
                    avg_count = np.mean(counts)
                    std_count = np.std(counts)
                    # Good delimiter should have consistent count across lines
                    consistency = 1 - (std_count / max(avg_count, 1))
                    delimiter_counts[delimiter] = {
                        'avg_count': avg_count,
                        'consistency': consistency,
                        'score': avg_count * consistency
                    }
            
            # Choose best delimiter
            if delimiter_counts:
                best_delimiter = max(delimiter_counts, key=lambda x: delimiter_counts[x]['score'])
                properties['delimiter'] = best_delimiter
                properties['delimiter_confidence'] = delimiter_counts[best_delimiter]['consistency']
            else:
                properties['delimiter'] = ','
                properties['delimiter_confidence'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error detecting delimiter for {file_path}: {e}")
            properties['delimiter'] = ','
            properties['delimiter_confidence'] = 0.0
        
        # Detect number of columns and rows (from header)
        try:
            df_sample = pd.read_csv(
                file_path, 
                encoding=properties['encoding'],
                sep=properties['delimiter'],
                nrows=0  # Just header
            )
            properties['num_columns'] = len(df_sample.columns)
            properties['column_names'] = df_sample.columns.tolist()
            
            # Estimate row count for small files
            if properties['file_size_mb'] < 50:
                df_count = pd.read_csv(
                    file_path,
                    encoding=properties['encoding'],
                    sep=properties['delimiter'],
                    usecols=[0]  # Just first column for counting
                )
                properties['num_rows'] = len(df_count)
            else:
                properties['num_rows'] = 'large_file'
                
        except Exception as e:
            logger.warning(f"Error reading file structure for {file_path}: {e}")
            properties['num_columns'] = 'unknown'
            properties['column_names'] = []
            properties['num_rows'] = 'unknown'
        
        return properties
    
    def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze and recommend data types for each column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict with data type analysis for each column
        """
        type_analysis = {}
        
        for col in df.columns:
            analysis = {
                'current_dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Get non-null values for analysis
            non_null_values = df[col].dropna()
            
            if len(non_null_values) == 0:
                analysis['recommended_dtype'] = 'object'
                analysis['recommendations'] = ['Column is entirely null']
                type_analysis[col] = analysis
                continue
            
            recommendations = []
            
            # Check for boolean
            unique_vals = set(non_null_values.astype(str).str.lower().unique())
            bool_vals = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
            if unique_vals.issubset(bool_vals) and len(unique_vals) <= 2:
                analysis['recommended_dtype'] = 'bool'
                recommendations.append('Convert to boolean')
            
            # Check for numeric types
            elif pd.api.types.is_numeric_dtype(non_null_values):
                if pd.api.types.is_integer_dtype(non_null_values):
                    # Integer optimization
                    min_val = non_null_values.min()
                    max_val = non_null_values.max()
                    
                    if min_val >= 0:
                        if max_val < 255:
                            analysis['recommended_dtype'] = 'uint8'
                        elif max_val < 65535:
                            analysis['recommended_dtype'] = 'uint16'
                        elif max_val < 4294967295:
                            analysis['recommended_dtype'] = 'uint32'
                        else:
                            analysis['recommended_dtype'] = 'uint64'
                    else:
                        if min_val >= -128 and max_val < 128:
                            analysis['recommended_dtype'] = 'int8'
                        elif min_val >= -32768 and max_val < 32768:
                            analysis['recommended_dtype'] = 'int16'
                        elif min_val >= -2147483648 and max_val < 2147483648:
                            analysis['recommended_dtype'] = 'int32'
                        else:
                            analysis['recommended_dtype'] = 'int64'
                    
                    recommendations.append('Optimize integer size')
                else:
                    # Float type
                    analysis['recommended_dtype'] = 'float32'
                    recommendations.append('Use float32 instead of float64')
            
            # Check for categorical
            elif analysis['unique_percentage'] < 50 and analysis['unique_count'] < 1000:
                analysis['recommended_dtype'] = 'category'
                recommendations.append('Convert to categorical for memory efficiency')
            
            # Check for datetime
            elif non_null_values.dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(non_null_values.head(100), errors='raise')
                    analysis['recommended_dtype'] = 'datetime64[ns]'
                    recommendations.append('Parse as datetime')
                except:
                    # Check if it looks like a date string
                    sample_values = non_null_values.head(10).astype(str)
                    date_patterns = [
                        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                    ]
                    
                    for pattern in date_patterns:
                        import re
                        if any(re.search(pattern, str(val)) for val in sample_values):
                            analysis['recommended_dtype'] = 'datetime64[ns]'
                            recommendations.append('Possible date format - verify and parse')
                            break
                    else:
                        analysis['recommended_dtype'] = 'object'
                        recommendations.append('Keep as string')
            else:
                analysis['recommended_dtype'] = 'object'
                recommendations.append('Keep current type')
            
            analysis['recommendations'] = recommendations
            type_analysis[col] = analysis
        
        return type_analysis
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate comprehensive statistics for each column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict with statistics for each column
        """
        statistics = {}
        
        for col in df.columns:
            stats = {
                'count': len(df[col]),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Non-null values for analysis
            non_null = df[col].dropna()
            
            if len(non_null) == 0:
                statistics[col] = stats
                continue
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(non_null):
                stats.update({
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'q25': float(non_null.quantile(0.25)),
                    'q75': float(non_null.quantile(0.75)),
                    'skewness': float(non_null.skew()),
                    'kurtosis': float(non_null.kurtosis())
                })
                
                # Outlier detection
                q1 = non_null.quantile(0.25)
                q3 = non_null.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                
                stats['outlier_count'] = len(outliers)
                stats['outlier_percentage'] = (len(outliers) / len(non_null)) * 100
            
            # String statistics
            elif non_null.dtype == 'object':
                str_lengths = non_null.astype(str).str.len()
                stats.update({
                    'min_length': int(str_lengths.min()),
                    'max_length': int(str_lengths.max()),
                    'avg_length': float(str_lengths.mean()),
                    'median_length': float(str_lengths.median())
                })
                
                # Most common values
                value_counts = non_null.value_counts().head(10)
                stats['top_values'] = [
                    {'value': str(val), 'count': int(count), 'percentage': float(count / len(non_null) * 100)}
                    for val, count in value_counts.items()
                ]
            
            # Datetime statistics
            elif pd.api.types.is_datetime64_any_dtype(non_null):
                stats.update({
                    'min_date': str(non_null.min()),
                    'max_date': str(non_null.max()),
                    'date_range_days': (non_null.max() - non_null.min()).days
                })
            
            statistics[col] = stats
        
        return statistics
    
    def detect_data_quality_issues(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect various data quality issues.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict with categorized data quality issues
        """
        issues = {
            'missing_data': [],
            'duplicates': [],
            'outliers': [],
            'inconsistencies': [],
            'format_issues': []
        }
        
        # Missing data issues
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 0:
                severity = 'high' if null_pct > 50 else 'medium' if null_pct > 10 else 'low'
                issues['missing_data'].append({
                    'column': col,
                    'null_percentage': null_pct,
                    'null_count': df[col].isnull().sum(),
                    'severity': severity,
                    'description': f'Column has {null_pct:.1f}% missing values'
                })
        
        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            severity = 'high' if duplicate_pct > 10 else 'medium' if duplicate_pct > 1 else 'low'
            issues['duplicates'].append({
                'type': 'complete_duplicates',
                'count': duplicate_count,
                'percentage': duplicate_pct,
                'severity': severity,
                'description': f'{duplicate_count} complete duplicate rows found'
            })
        
        # Column-specific issues
        for col in df.columns:
            non_null = df[col].dropna()
            
            if len(non_null) == 0:
                continue
            
            # Outliers for numeric columns
            if pd.api.types.is_numeric_dtype(non_null):
                q1 = non_null.quantile(0.25)
                q3 = non_null.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = len(non_null[(non_null < lower_bound) | (non_null > upper_bound)])
                
                if outlier_count > 0:
                    outlier_pct = (outlier_count / len(non_null)) * 100
                    severity = 'high' if outlier_pct > 10 else 'medium' if outlier_pct > 5 else 'low'
                    issues['outliers'].append({
                        'column': col,
                        'outlier_count': outlier_count,
                        'outlier_percentage': outlier_pct,
                        'severity': severity,
                        'description': f'Column has {outlier_count} outliers ({outlier_pct:.1f}%)'
                    })
            
            # Format inconsistencies for string columns
            elif non_null.dtype == 'object':
                # Check for mixed case
                if len(non_null) > 1:
                    has_upper = any(str(val).isupper() for val in non_null.head(100))
                    has_lower = any(str(val).islower() for val in non_null.head(100))
                    has_title = any(str(val).istitle() for val in non_null.head(100))
                    
                    case_variations = sum([has_upper, has_lower, has_title])
                    if case_variations > 1:
                        issues['format_issues'].append({
                            'column': col,
                            'issue_type': 'mixed_case',
                            'severity': 'low',
                            'description': f'Column has mixed case formatting'
                        })
                
                # Check for leading/trailing whitespace
                has_whitespace = any(str(val) != str(val).strip() for val in non_null.head(100))
                if has_whitespace:
                    issues['format_issues'].append({
                        'column': col,
                        'issue_type': 'whitespace',
                        'severity': 'medium',
                        'description': f'Column has leading/trailing whitespace'
                    })
        
        return issues
    
    def generate_profile_report(self, df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive profile report for a DataFrame.
        
        Args:
            df: DataFrame to profile
            file_path: Original file path
            
        Returns:
            Dict with complete profile report
        """
        logger.info(f"Generating profile report for {file_path}")
        
        # Basic file properties
        file_properties = self.detect_file_properties(file_path)
        
        # Data type analysis
        type_analysis = self.analyze_data_types(df)
        
        # Statistical analysis
        statistics = self.calculate_statistics(df)
        
        # Data quality issues
        quality_issues = self.detect_data_quality_issues(df)
        
        # Overall summary
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'null_cells': df.isnull().sum().sum(),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Quality score calculation
        quality_score = 100
        
        # Penalize for missing data
        if summary['null_percentage'] > 0:
            quality_score -= min(summary['null_percentage'] * 2, 30)
        
        # Penalize for duplicates
        duplicate_pct = (summary['duplicate_rows'] / summary['total_rows']) * 100
        if duplicate_pct > 0:
            quality_score -= min(duplicate_pct * 3, 20)
        
        # Penalize for high number of quality issues
        total_issues = sum(len(issues) for issues in quality_issues.values())
        quality_score -= min(total_issues * 2, 30)
        
        summary['data_quality_score'] = max(quality_score, 0)
        
        # Compile complete report
        report = {
            'profile_timestamp': datetime.now().isoformat(),
            'file_properties': file_properties,
            'summary': summary,
            'statistics': statistics,
            'type_analysis': type_analysis,
            'quality_issues': quality_issues,
            'recommendations': self._generate_recommendations(summary, type_analysis, quality_issues)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict, type_analysis: Dict, quality_issues: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Memory optimization
        potential_savings = 0
        for col, analysis in type_analysis.items():
            if analysis['current_dtype'] != analysis['recommended_dtype']:
                if 'int64' in analysis['current_dtype'] and 'int' in analysis['recommended_dtype']:
                    potential_savings += 1
                elif 'float64' in analysis['current_dtype'] and 'float32' in analysis['recommended_dtype']:
                    potential_savings += 1
                elif analysis['recommended_dtype'] == 'category':
                    potential_savings += 2
        
        if potential_savings > 0:
            recommendations.append(f"Optimize data types to reduce memory usage (potential {potential_savings} columns)")
        
        # Data quality recommendations
        if summary['null_percentage'] > 10:
            recommendations.append("Address high percentage of missing values")
        
        if summary['duplicate_rows'] > 0:
            recommendations.append("Remove duplicate rows to clean dataset")
        
        # Quality issues
        if quality_issues['outliers']:
            recommendations.append("Review and handle outliers in numeric columns")
        
        if quality_issues['format_issues']:
            recommendations.append("Standardize text formatting (case, whitespace)")
        
        if quality_issues['missing_data']:
            high_missing = [issue for issue in quality_issues['missing_data'] if issue['severity'] == 'high']
            if high_missing:
                recommendations.append("Consider dropping columns with very high missing values")
        
        # Performance recommendations
        if summary['total_rows'] > 100000:
            recommendations.append("Consider using chunked processing for large dataset")
        
        if summary['categorical_columns'] > 5:
            recommendations.append("Convert categorical columns to 'category' dtype for better performance")
        
        return recommendations
    
    def profile_all_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Profile all CSV files and return comprehensive results.
        
        Returns:
            Dict with profile results for each file
        """
        logger.info(f"Profiling {len(self.file_paths)} CSV files")
        
        all_results = {}
        
        for file_path in self.file_paths:
            try:
                # Read sample data
                file_props = self.detect_file_properties(file_path)
                
                # Read with detected parameters
                df = pd.read_csv(
                    file_path,
                    encoding=file_props['encoding'],
                    sep=file_props['delimiter'],
                    nrows=self.sample_size if isinstance(file_props['num_rows'], int) 
                          and file_props['num_rows'] > self.sample_size else None
                )
                
                # Generate profile
                profile = self.generate_profile_report(df, file_path)
                all_results[file_path] = profile
                
                logger.info(f"Successfully profiled {file_path}")
                
            except Exception as e:
                logger.error(f"Error profiling {file_path}: {e}")
                all_results[file_path] = {
                    'error': str(e),
                    'profile_timestamp': datetime.now().isoformat()
                }
        
        self.profile_results = all_results
        return all_results
    
    def save_profile_report(self, output_path: str, format: str = 'json') -> None:
        """
        Save profile results to file.
        
        Args:
            output_path: Output file path
            format: Output format ('json', 'html')
        """
        if not self.profile_results:
            logger.warning("No profile results to save. Run profile_all_files() first.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.profile_results, f, indent=2, default=str)
        
        elif format.lower() == 'html':
            self._generate_html_report(output_path)
        
        logger.info(f"Profile report saved to {output_path}")
    
    def _generate_html_report(self, output_path: str) -> None:
        """Generate HTML report from profile results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CSV Data Profile Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .summary-table { width: 100%; border-collapse: collapse; }
                .summary-table th, .summary-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .quality-score { font-size: 24px; font-weight: bold; }
                .high-severity { color: #d32f2f; }
                .medium-severity { color: #f57c00; }
                .low-severity { color: #388e3c; }
                .recommendation { background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
        """
        
        html_content += f"""
        <div class="header">
            <h1>CSV Data Profile Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Files analyzed: {len(self.profile_results)}</p>
        </div>
        """
        
        for file_path, profile in self.profile_results.items():
            if 'error' in profile:
                html_content += f"""
                <div class="section">
                    <h2>{file_path}</h2>
                    <p style="color: red;">Error: {profile['error']}</p>
                </div>
                """
                continue
            
            summary = profile['summary']
            quality_score = summary['data_quality_score']
            
            html_content += f"""
            <div class="section">
                <h2>{os.path.basename(file_path)}</h2>
                <div class="quality-score">Data Quality Score: {quality_score:.1f}/100</div>
                
                <h3>Summary</h3>
                <table class="summary-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Rows</td><td>{summary['total_rows']:,}</td></tr>
                    <tr><td>Total Columns</td><td>{summary['total_columns']}</td></tr>
                    <tr><td>Memory Usage</td><td>{summary['memory_usage_mb']:.2f} MB</td></tr>
                    <tr><td>Missing Values</td><td>{summary['null_percentage']:.2f}%</td></tr>
                    <tr><td>Duplicate Rows</td><td>{summary['duplicate_rows']:,}</td></tr>
                </table>
            """
            
            # Quality issues
            quality_issues = profile['quality_issues']
            total_issues = sum(len(issues) for issues in quality_issues.values())
            
            if total_issues > 0:
                html_content += "<h3>Data Quality Issues</h3><ul>"
                for category, issues in quality_issues.items():
                    for issue in issues:
                        severity_class = f"{issue.get('severity', 'low')}-severity"
                        html_content += f'<li class="{severity_class}">{issue["description"]}</li>'
                html_content += "</ul>"
            
            # Recommendations
            recommendations = profile['recommendations']
            if recommendations:
                html_content += "<h3>Recommendations</h3>"
                for rec in recommendations:
                    html_content += f'<div class="recommendation">{rec}</div>'
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def create_visualizations(self, output_dir: str) -> None:
        """
        Create data visualization plots and save to directory.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.profile_results:
            logger.warning("No profile results available for visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path, profile in self.profile_results.items():
            if 'error' in profile:
                continue
            
            file_name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                # Read sample data for plotting
                file_props = profile['file_properties']
                df = pd.read_csv(
                    file_path,
                    encoding=file_props['encoding'],
                    sep=file_props['delimiter'],
                    nrows=min(self.sample_size, 1000)  # Limit for plotting
                )
                
                # Data quality overview
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Data Quality Overview - {file_name}', fontsize=16)
                
                # Missing values heatmap
                if df.isnull().sum().sum() > 0:
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    axes[0, 0].bar(range(len(missing_data)), missing_data.values)
                    axes[0, 0].set_xticks(range(len(missing_data)))
                    axes[0, 0].set_xticklabels(missing_data.index, rotation=45)
                    axes[0, 0].set_title('Missing Values by Column')
                    axes[0, 0].set_ylabel('Count')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
                    axes[0, 0].set_title('Missing Values by Column')
                
                # Data types distribution
                type_counts = df.dtypes.value_counts()
                axes[0, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
                axes[0, 1].set_title('Data Types Distribution')
                
                # Numeric columns distribution
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sample_col = numeric_cols[0]
                    axes[1, 0].hist(df[sample_col].dropna(), bins=30, alpha=0.7)
                    axes[1, 0].set_title(f'Distribution of {sample_col}')
                    axes[1, 0].set_xlabel(sample_col)
                    axes[1, 0].set_ylabel('Frequency')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center')
                    axes[1, 0].set_title('Numeric Distribution')
                
                # Unique values per column
                unique_counts = df.nunique().sort_values(ascending=False)[:10]
                axes[1, 1].bar(range(len(unique_counts)), unique_counts.values)
                axes[1, 1].set_xticks(range(len(unique_counts)))
                axes[1, 1].set_xticklabels(unique_counts.index, rotation=45)
                axes[1, 1].set_title('Top 10 Columns by Unique Values')
                axes[1, 1].set_ylabel('Unique Count')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{file_name}_profile.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Created visualization for {file_name}")
                
            except Exception as e:
                logger.error(f"Error creating visualization for {file_path}: {e}")
        
        logger.info(f"Visualizations saved to {output_dir}")


def profile_csv_files(file_paths: List[str], output_dir: str = './data_profile') -> CSVDataProfiler:
    """
    Convenience function to profile CSV files and generate reports.
    
    Args:
        file_paths: List of CSV file paths
        output_dir: Output directory for reports
        
    Returns:
        CSVDataProfiler instance with results
    """
    profiler = CSVDataProfiler(file_paths)
    
    # Generate profiles
    results = profiler.profile_all_files()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save reports
    profiler.save_profile_report(f"{output_dir}/profile_report.json", format='json')
    profiler.save_profile_report(f"{output_dir}/profile_report.html", format='html')
    
    # Create visualizations
    profiler.create_visualizations(f"{output_dir}/plots")
    
    logger.info(f"CSV profiling complete. Results saved to {output_dir}")
    
    return profiler