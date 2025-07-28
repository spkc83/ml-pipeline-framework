"""
CSV Data Connector for ML Pipeline Framework

This module provides a robust CSV connector with support for multiple files,
chunked reading, compression formats, and automatic data type inference.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Iterator
import logging
from pathlib import Path
import chardet
import gzip
import zipfile
import io
from datetime import datetime

from .data_connector import DataConnector

logger = logging.getLogger(__name__)


class CSVConnector(DataConnector):
    """
    CSV connector for reading single or multiple CSV files with advanced features.
    
    Features:
        - Multiple CSV file support with pattern matching
        - Chunked reading for large files
        - Automatic encoding and delimiter detection
        - Compression support (gzip, zip)
        - Data type inference and custom dtype mapping
        - Header validation and consistency checking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV connector with configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - file_paths (str or List[str]): Path(s) to CSV files or patterns
                - separator (str, optional): Column separator, auto-detected if not provided
                - encoding (str, optional): File encoding, auto-detected if not provided
                - compression (str, optional): Compression format ('gzip', 'zip', None)
                - chunk_size (int, optional): Number of rows per chunk for large files
                - date_columns (List[str], optional): Columns to parse as dates
                - dtype_mapping (Dict[str, str], optional): Custom dtype mapping
                - header_row (int, optional): Row number containing headers (default: 0)
                - validate_headers (bool, optional): Validate header consistency (default: True)
        """
        super().__init__(config)
        
        self.file_paths = self._resolve_file_paths(config.get('file_paths', []))
        self.separator = config.get('separator', None)
        self.encoding = config.get('encoding', None)
        self.compression = config.get('compression', None)
        self.chunk_size = config.get('chunk_size', None)
        self.date_columns = config.get('date_columns', [])
        self.dtype_mapping = config.get('dtype_mapping', {})
        self.header_row = config.get('header_row', 0)
        self.validate_headers = config.get('validate_headers', True)
        
        # Cache for detected parameters
        self._detected_params = {}
        
    def _resolve_file_paths(self, paths: Union[str, List[str]]) -> List[str]:
        """Resolve file paths including glob patterns."""
        if isinstance(paths, str):
            paths = [paths]
        
        resolved_paths = []
        for path in paths:
            if '*' in path or '?' in path:
                # Handle glob patterns
                matched_files = glob.glob(path)
                resolved_paths.extend(matched_files)
            else:
                # Single file
                if os.path.exists(path):
                    resolved_paths.append(path)
                else:
                    logger.warning(f"File not found: {path}")
        
        if not resolved_paths:
            raise FileNotFoundError(f"No files found for paths: {paths}")
        
        # Sort for consistent ordering
        resolved_paths.sort()
        logger.info(f"Resolved {len(resolved_paths)} CSV files")
        
        return resolved_paths
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        try:
            # Read first 10000 bytes for detection
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
            
            # Handle compressed files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    raw_data = f.read(10000)
            elif file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    file_name = zf.namelist()[0]
                    raw_data = zf.read(file_name)[:10000]
            
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Default to utf-8 if confidence is low
            if confidence < 0.7:
                encoding = 'utf-8'
                
            return encoding
            
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}. Using utf-8")
            return 'utf-8'
    
    def _detect_separator(self, file_path: str, encoding: str) -> str:
        """Detect column separator by analyzing first few lines."""
        try:
            # Read first few lines
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding=encoding) as f:
                    lines = [f.readline() for _ in range(5)]
            elif file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    file_name = zf.namelist()[0]
                    with zf.open(file_name, 'r') as f:
                        lines = [f.readline().decode(encoding) for _ in range(5)]
            else:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [f.readline() for _ in range(5)]
            
            # Count occurrences of common separators
            separators = [',', ';', '\t', '|', ' ']
            separator_counts = {sep: 0 for sep in separators}
            
            for line in lines:
                for sep in separators:
                    separator_counts[sep] += line.count(sep)
            
            # Choose separator with highest count
            detected_sep = max(separator_counts, key=separator_counts.get)
            
            logger.info(f"Detected separator: '{detected_sep}'")
            return detected_sep
            
        except Exception as e:
            logger.warning(f"Error detecting separator: {e}. Using comma")
            return ','
    
    def _infer_dtypes(self, df_sample: pd.DataFrame) -> Dict[str, str]:
        """Infer optimal data types from a sample of data."""
        dtype_mapping = {}
        
        for col in df_sample.columns:
            # Skip if already specified in config
            if col in self.dtype_mapping:
                continue
            
            # Get non-null values
            non_null = df_sample[col].dropna()
            
            if len(non_null) == 0:
                continue
            
            # Try to infer type
            try:
                # Check if boolean
                unique_vals = non_null.unique()
                if len(unique_vals) <= 2 and all(val in [True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0'] for val in unique_vals):
                    dtype_mapping[col] = 'bool'
                    continue
                
                # Check if integer
                if pd.api.types.is_integer_dtype(non_null):
                    if non_null.min() >= 0 and non_null.max() < 255:
                        dtype_mapping[col] = 'uint8'
                    elif non_null.min() >= -128 and non_null.max() < 128:
                        dtype_mapping[col] = 'int8'
                    elif non_null.min() >= -32768 and non_null.max() < 32768:
                        dtype_mapping[col] = 'int16'
                    elif non_null.min() >= -2147483648 and non_null.max() < 2147483648:
                        dtype_mapping[col] = 'int32'
                    else:
                        dtype_mapping[col] = 'int64'
                    continue
                
                # Check if float
                try:
                    pd.to_numeric(non_null)
                    dtype_mapping[col] = 'float32'  # Use float32 by default for memory efficiency
                    continue
                except:
                    pass
                
                # Check if date
                if col in self.date_columns:
                    dtype_mapping[col] = 'datetime64[ns]'
                    continue
                
                # Check if categorical (low cardinality strings)
                if non_null.dtype == 'object':
                    n_unique = non_null.nunique()
                    n_total = len(non_null)
                    if n_unique / n_total < 0.5 and n_unique < 1000:  # Less than 50% unique and < 1000 categories
                        dtype_mapping[col] = 'category'
                    else:
                        dtype_mapping[col] = 'object'
                
            except Exception as e:
                logger.warning(f"Error inferring dtype for column {col}: {e}")
                dtype_mapping[col] = 'object'
        
        return dtype_mapping
    
    def _validate_headers(self, headers_list: List[List[str]]) -> bool:
        """Validate that all files have consistent headers."""
        if not self.validate_headers or len(headers_list) <= 1:
            return True
        
        reference_headers = set(headers_list[0])
        
        for i, headers in enumerate(headers_list[1:], 1):
            current_headers = set(headers)
            
            if current_headers != reference_headers:
                missing = reference_headers - current_headers
                extra = current_headers - reference_headers
                
                logger.error(f"Header mismatch in file {self.file_paths[i]}")
                if missing:
                    logger.error(f"  Missing columns: {missing}")
                if extra:
                    logger.error(f"  Extra columns: {extra}")
                
                if not self.config.get('ignore_header_mismatch', False):
                    raise ValueError(f"Inconsistent headers across CSV files")
        
        return True
    
    def _read_single_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read a single CSV file with proper parameter detection."""
        # Detect parameters if not provided
        if self.encoding is None:
            encoding = self._detect_encoding(file_path)
        else:
            encoding = self.encoding
        
        if self.separator is None:
            separator = self._detect_separator(file_path, encoding)
        else:
            separator = self.separator
        
        # Prepare read_csv parameters
        read_params = {
            'sep': separator,
            'encoding': encoding,
            'header': self.header_row,
            'parse_dates': self.date_columns if self.date_columns else False,
        }
        
        # Add compression if needed
        if file_path.endswith('.gz'):
            read_params['compression'] = 'gzip'
        elif file_path.endswith('.zip'):
            read_params['compression'] = 'zip'
        elif self.compression:
            read_params['compression'] = self.compression
        
        # Add any additional kwargs
        read_params.update(kwargs)
        
        # Read with dtype mapping if available
        if self.dtype_mapping:
            read_params['dtype'] = self.dtype_mapping
        
        try:
            df = pd.read_csv(file_path, **read_params)
            logger.info(f"Successfully read {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def read_data(self, **kwargs) -> pd.DataFrame:
        """
        Read data from CSV file(s).
        
        Returns:
            pd.DataFrame: Combined dataframe from all CSV files
        """
        if len(self.file_paths) == 1:
            # Single file
            df = self._read_single_file(self.file_paths[0], **kwargs)
        else:
            # Multiple files
            dfs = []
            headers_list = []
            
            for file_path in self.file_paths:
                df = self._read_single_file(file_path, **kwargs)
                dfs.append(df)
                headers_list.append(df.columns.tolist())
            
            # Validate headers
            self._validate_headers(headers_list)
            
            # Combine dataframes
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(dfs)} files into dataframe with {len(df)} total rows")
        
        # Infer and optimize dtypes if not specified
        if not self.dtype_mapping and self.config.get('optimize_dtypes', True):
            # Sample for dtype inference
            sample_size = min(10000, len(df))
            df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
            
            inferred_dtypes = self._infer_dtypes(df_sample)
            
            # Apply inferred dtypes
            for col, dtype in inferred_dtypes.items():
                try:
                    if dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to {dtype}: {e}")
        
        # Log memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"DataFrame memory usage: {memory_usage:.2f} MB")
        
        return df
    
    def read_chunked(self, chunk_size: Optional[int] = None, **kwargs) -> Iterator[pd.DataFrame]:
        """
        Read CSV file(s) in chunks for memory-efficient processing.
        
        Args:
            chunk_size: Number of rows per chunk (overrides config)
            **kwargs: Additional parameters for pd.read_csv
            
        Yields:
            pd.DataFrame: Chunks of the CSV data
        """
        chunk_size = chunk_size or self.chunk_size or 10000
        
        for file_path in self.file_paths:
            # Detect parameters
            if self.encoding is None:
                encoding = self._detect_encoding(file_path)
            else:
                encoding = self.encoding
            
            if self.separator is None:
                separator = self._detect_separator(file_path, encoding)
            else:
                separator = self.separator
            
            # Prepare read parameters
            read_params = {
                'sep': separator,
                'encoding': encoding,
                'header': self.header_row,
                'parse_dates': self.date_columns if self.date_columns else False,
                'chunksize': chunk_size
            }
            
            # Add compression
            if file_path.endswith('.gz'):
                read_params['compression'] = 'gzip'
            elif file_path.endswith('.zip'):
                read_params['compression'] = 'zip'
            elif self.compression:
                read_params['compression'] = self.compression
            
            # Add dtype mapping
            if self.dtype_mapping:
                read_params['dtype'] = self.dtype_mapping
            
            read_params.update(kwargs)
            
            # Read in chunks
            try:
                chunk_reader = pd.read_csv(file_path, **read_params)
                
                for i, chunk in enumerate(chunk_reader):
                    logger.debug(f"Processing chunk {i+1} from {file_path} ({len(chunk)} rows)")
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Error reading chunks from {file_path}: {e}")
                raise
    
    def write_data(self, data: pd.DataFrame, output_path: str, **kwargs) -> None:
        """
        Write dataframe to CSV file.
        
        Args:
            data: DataFrame to write
            output_path: Output file path
            **kwargs: Additional parameters for pd.to_csv
        """
        write_params = {
            'index': False,
            'sep': self.separator or ',',
            'encoding': self.encoding or 'utf-8'
        }
        
        # Add compression
        if output_path.endswith('.gz'):
            write_params['compression'] = 'gzip'
        elif output_path.endswith('.zip'):
            write_params['compression'] = 'zip'
        elif self.compression:
            write_params['compression'] = self.compression
        
        write_params.update(kwargs)
        
        try:
            data.to_csv(output_path, **write_params)
            logger.info(f"Successfully wrote {len(data)} rows to {output_path}")
            
        except Exception as e:
            logger.error(f"Error writing to {output_path}: {e}")
            raise
    
    def query(self, query: str) -> pd.DataFrame:
        """
        Execute a query on the CSV data using pandas query syntax.
        
        Args:
            query: Pandas query string
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        df = self.read_data()
        try:
            result = df.query(query)
            logger.info(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_schema(self) -> Dict[str, str]:
        """
        Get schema information for the CSV file(s).
        
        Returns:
            Dict[str, str]: Column names and their data types
        """
        # Read a sample to get schema
        sample_df = self.read_data(nrows=1000)
        
        schema = {}
        for col in sample_df.columns:
            dtype = str(sample_df[col].dtype)
            schema[col] = dtype
        
        return schema
    
    def validate_connection(self) -> bool:
        """
        Validate that CSV files exist and are readable.
        
        Returns:
            bool: True if all files are accessible
        """
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            if not os.access(file_path, os.R_OK):
                logger.error(f"File not readable: {file_path}")
                return False
        
        # Try reading first few rows
        try:
            df = self.read_data(nrows=5)
            logger.info(f"Successfully validated {len(self.file_paths)} CSV file(s)")
            return True
        except Exception as e:
            logger.error(f"Error validating CSV files: {e}")
            return False
    
    def get_row_count(self) -> int:
        """Get total row count across all CSV files."""
        total_rows = 0
        
        for file_path in self.file_paths:
            # Use chunked reading for memory efficiency
            for chunk in self.read_chunked():
                total_rows += len(chunk)
        
        return total_rows
    
    def get_file_info(self) -> List[Dict[str, Any]]:
        """Get information about each CSV file."""
        file_info = []
        
        for file_path in self.file_paths:
            info = {
                'path': file_path,
                'size_mb': os.path.getsize(file_path) / 1024**2,
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                'encoding': self._detect_encoding(file_path),
                'separator': self._detect_separator(file_path, self.encoding or 'utf-8')
            }
            
            # Get row count for small files
            if info['size_mb'] < 100:
                try:
                    df = self._read_single_file(file_path, nrows=None)
                    info['rows'] = len(df)
                    info['columns'] = len(df.columns)
                except:
                    info['rows'] = 'unknown'
                    info['columns'] = 'unknown'
            else:
                info['rows'] = 'large file'
                info['columns'] = 'large file'
            
            file_info.append(info)
        
        return file_info