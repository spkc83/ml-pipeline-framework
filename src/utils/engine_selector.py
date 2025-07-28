"""
Data processing engine selector module.

This module provides intelligent selection between pandas, Polars, and DuckDB
based on data size, operation type, and system resources.
"""

import logging
import psutil
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataEngine(Enum):
    """Supported data processing engines."""
    PANDAS = "pandas"
    POLARS = "polars"
    DUCKDB = "duckdb"


class OperationType(Enum):
    """Types of data operations."""
    READ = "read"
    WRITE = "write"
    FILTER = "filter"
    GROUPBY = "groupby"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    TRANSFORM = "transform"
    WINDOW = "window"
    ANALYTICS = "analytics"


@dataclass
class SystemResources:
    """System resource information."""
    available_memory_gb: float
    cpu_cores: int
    has_gpu: bool = False
    disk_io_speed_mbps: Optional[float] = None


@dataclass
class DataProfile:
    """Data profiling information."""
    row_count: int
    column_count: int
    memory_usage_mb: float
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    null_percentage: float
    data_types: Dict[str, str]


@dataclass
class EngineRecommendation:
    """Engine recommendation with reasoning."""
    engine: DataEngine
    confidence: float
    reasoning: List[str]
    performance_estimate: Optional[Dict[str, float]] = None
    fallback_engines: List[DataEngine] = None


class BaseEngineAdapter(ABC):
    """Base class for engine adapters."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available."""
        pass
    
    @abstractmethod
    def estimate_memory_usage(self, data_profile: DataProfile) -> float:
        """Estimate memory usage for the given data profile."""
        pass
    
    @abstractmethod
    def estimate_performance(self, operation: OperationType, data_profile: DataProfile) -> float:
        """Estimate relative performance score (0-100) for the operation."""
        pass
    
    @abstractmethod
    def get_supported_operations(self) -> List[OperationType]:
        """Get list of supported operations."""
        pass


class PandasAdapter(BaseEngineAdapter):
    """Pandas engine adapter."""
    
    def is_available(self) -> bool:
        """Check if pandas is available."""
        try:
            import pandas
            return True
        except ImportError:
            return False
    
    def estimate_memory_usage(self, data_profile: DataProfile) -> float:
        """Estimate pandas memory usage in MB."""
        # Pandas typically uses 8-10x more memory than raw data
        base_memory = data_profile.memory_usage_mb
        
        # Add overhead for object dtypes (strings)
        string_overhead = data_profile.categorical_columns * 0.5
        
        # Add index overhead
        index_overhead = data_profile.row_count * 8 / (1024 * 1024)  # 8 bytes per row
        
        return base_memory * 8.5 + string_overhead + index_overhead
    
    def estimate_performance(self, operation: OperationType, data_profile: DataProfile) -> float:
        """Estimate pandas performance score."""
        base_score = 70  # Pandas is generally good but not the fastest
        
        # Adjust based on operation type
        if operation == OperationType.READ:
            if data_profile.row_count < 1_000_000:
                base_score = 85
            elif data_profile.row_count < 10_000_000:
                base_score = 65
            else:
                base_score = 45
        
        elif operation == OperationType.GROUPBY:
            if data_profile.row_count < 500_000:
                base_score = 80
            else:
                base_score = 50
        
        elif operation == OperationType.JOIN:
            if data_profile.row_count < 1_000_000:
                base_score = 75
            else:
                base_score = 40
        
        elif operation in [OperationType.FILTER, OperationType.SORT]:
            base_score = 70
        
        elif operation == OperationType.ANALYTICS:
            base_score = 90  # Pandas excels at analytics with scipy/numpy
        
        return min(100, max(0, base_score))
    
    def get_supported_operations(self) -> List[OperationType]:
        """Get pandas supported operations."""
        return list(OperationType)  # Pandas supports all operations


class PolarsAdapter(BaseEngineAdapter):
    """Polars engine adapter."""
    
    def is_available(self) -> bool:
        """Check if polars is available."""
        try:
            import polars
            return True
        except ImportError:
            return False
    
    def estimate_memory_usage(self, data_profile: DataProfile) -> float:
        """Estimate polars memory usage in MB."""
        # Polars is more memory efficient than pandas
        base_memory = data_profile.memory_usage_mb
        
        # Polars uses columnar format, more efficient
        columnar_efficiency = 0.6  # 40% reduction
        
        # Add some overhead for lazy evaluation cache
        cache_overhead = base_memory * 0.2
        
        return base_memory * columnar_efficiency + cache_overhead
    
    def estimate_performance(self, operation: OperationType, data_profile: DataProfile) -> float:
        """Estimate polars performance score."""
        base_score = 85  # Polars is generally faster than pandas
        
        # Adjust based on operation type
        if operation == OperationType.READ:
            base_score = 90  # Excellent reading performance
        
        elif operation == OperationType.GROUPBY:
            base_score = 95  # Outstanding groupby performance
        
        elif operation == OperationType.JOIN:
            if data_profile.row_count > 1_000_000:
                base_score = 90  # Excellent for large joins
            else:
                base_score = 85
        
        elif operation in [OperationType.FILTER, OperationType.SORT]:
            base_score = 90  # Very good at these operations
        
        elif operation == OperationType.ANALYTICS:
            base_score = 70  # Good but not as mature as pandas ecosystem
        
        elif operation == OperationType.WINDOW:
            base_score = 95  # Excellent window functions
        
        # Boost score for larger datasets
        if data_profile.row_count > 1_000_000:
            base_score = min(100, base_score + 10)
        
        return min(100, max(0, base_score))
    
    def get_supported_operations(self) -> List[OperationType]:
        """Get polars supported operations."""
        return list(OperationType)  # Polars supports all operations


class DuckDBAdapter(BaseEngineAdapter):
    """DuckDB engine adapter."""
    
    def is_available(self) -> bool:
        """Check if duckdb is available."""
        try:
            import duckdb
            return True
        except ImportError:
            return False
    
    def estimate_memory_usage(self, data_profile: DataProfile) -> float:
        """Estimate duckdb memory usage in MB."""
        # DuckDB is very memory efficient with columnar storage
        base_memory = data_profile.memory_usage_mb
        
        # Columnar compression
        compression_ratio = 0.4  # 60% reduction
        
        # Add overhead for query processing
        query_overhead = base_memory * 0.3
        
        return base_memory * compression_ratio + query_overhead
    
    def estimate_performance(self, operation: OperationType, data_profile: DataProfile) -> float:
        """Estimate duckdb performance score."""
        base_score = 80
        
        # DuckDB excels at analytical queries
        if operation == OperationType.ANALYTICS:
            base_score = 95
        
        elif operation == OperationType.GROUPBY:
            base_score = 90
        
        elif operation == OperationType.JOIN:
            if data_profile.row_count > 5_000_000:
                base_score = 95  # Excellent for very large joins
            else:
                base_score = 85
        
        elif operation == OperationType.AGGREGATE:
            base_score = 95
        
        elif operation == OperationType.WINDOW:
            base_score = 90
        
        elif operation in [OperationType.READ, OperationType.WRITE]:
            base_score = 85
        
        elif operation in [OperationType.FILTER, OperationType.SORT]:
            base_score = 85
        
        # Significant boost for large datasets
        if data_profile.row_count > 10_000_000:
            base_score = min(100, base_score + 15)
        elif data_profile.row_count > 1_000_000:
            base_score = min(100, base_score + 5)
        
        return min(100, max(0, base_score))
    
    def get_supported_operations(self) -> List[OperationType]:
        """Get duckdb supported operations."""
        # DuckDB is SQL-focused, some operations need translation
        return [
            OperationType.READ, OperationType.WRITE, OperationType.FILTER,
            OperationType.GROUPBY, OperationType.JOIN, OperationType.AGGREGATE,
            OperationType.SORT, OperationType.WINDOW, OperationType.ANALYTICS
        ]


class DataEngineSelector:
    """
    Intelligent data processing engine selector.
    
    Chooses between pandas, Polars, and DuckDB based on data characteristics,
    operation type, and system resources.
    """
    
    def __init__(self, 
                 memory_threshold_gb: float = 2.0,
                 large_data_threshold: int = 1_000_000,
                 very_large_data_threshold: int = 10_000_000,
                 enable_benchmarking: bool = False,
                 cache_recommendations: bool = True):
        """
        Initialize DataEngineSelector.
        
        Args:
            memory_threshold_gb: Memory threshold for switching engines
            large_data_threshold: Row count threshold for large data
            very_large_data_threshold: Row count threshold for very large data
            enable_benchmarking: Enable runtime benchmarking
            cache_recommendations: Cache recommendations for similar data profiles
        """
        self.memory_threshold_gb = memory_threshold_gb
        self.large_data_threshold = large_data_threshold
        self.very_large_data_threshold = very_large_data_threshold
        self.enable_benchmarking = enable_benchmarking
        self.cache_recommendations = cache_recommendations
        
        # Initialize adapters
        self.adapters = {
            DataEngine.PANDAS: PandasAdapter(),
            DataEngine.POLARS: PolarsAdapter(),
            DataEngine.DUCKDB: DuckDBAdapter()
        }
        
        # Check available engines
        self.available_engines = {
            engine: adapter.is_available() 
            for engine, adapter in self.adapters.items()
        }
        
        # Cache for recommendations
        self._recommendation_cache = {} if cache_recommendations else None
        
        # Benchmark results
        self._benchmark_results = {}
        
        logger.info(f"DataEngineSelector initialized")
        logger.info(f"Available engines: {[e.value for e, available in self.available_engines.items() if available]}")
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resource information."""
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        cpu_cores = psutil.cpu_count(logical=False)
        
        # Check for GPU (basic check)
        has_gpu = False
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            has_gpu = len(gpus) > 0
        except ImportError:
            pass
        
        return SystemResources(
            available_memory_gb=available_memory_gb,
            cpu_cores=cpu_cores,
            has_gpu=has_gpu
        )
    
    def profile_data(self, data: Union[pd.DataFrame, str, Path]) -> DataProfile:
        """
        Profile data to understand its characteristics.
        
        Args:
            data: DataFrame or path to data file
            
        Returns:
            DataProfile with data characteristics
        """
        if isinstance(data, (str, Path)):
            # For file paths, estimate from file info
            file_path = Path(data)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Rough estimates based on file type and size
            if file_path.suffix.lower() == '.csv':
                estimated_rows = int(file_size_mb * 15000)  # ~15k rows per MB for typical CSV
                estimated_cols = 20  # Default estimate
            elif file_path.suffix.lower() in ['.parquet', '.pq']:
                estimated_rows = int(file_size_mb * 50000)  # More compressed
                estimated_cols = 25
            else:
                estimated_rows = int(file_size_mb * 10000)  # Conservative estimate
                estimated_cols = 15
            
            return DataProfile(
                row_count=estimated_rows,
                column_count=estimated_cols,
                memory_usage_mb=file_size_mb * 5,  # Estimate in-memory size
                numeric_columns=estimated_cols // 2,
                categorical_columns=estimated_cols // 3,
                datetime_columns=1,
                null_percentage=5.0,  # Default estimate
                data_types={}
            )
        
        elif isinstance(data, pd.DataFrame):
            # Profile actual DataFrame
            memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
            datetime_cols = len(data.select_dtypes(include=['datetime64']).columns)
            
            null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            
            data_types = {str(col): str(dtype) for col, dtype in data.dtypes.items()}
            
            return DataProfile(
                row_count=len(data),
                column_count=len(data.columns),
                memory_usage_mb=memory_usage,
                numeric_columns=numeric_cols,
                categorical_columns=categorical_cols,
                datetime_columns=datetime_cols,
                null_percentage=null_percentage,
                data_types=data_types
            )
        
        else:
            raise TypeError("Data must be a pandas DataFrame or file path")
    
    def recommend_engine(self, 
                        data_profile: DataProfile,
                        operation: OperationType,
                        system_resources: Optional[SystemResources] = None) -> EngineRecommendation:
        """
        Recommend the best engine for the given data and operation.
        
        Args:
            data_profile: Data characteristics
            operation: Type of operation to perform
            system_resources: System resource information
            
        Returns:
            EngineRecommendation with engine choice and reasoning
        """
        if system_resources is None:
            system_resources = self.get_system_resources()
        
        # Check cache
        cache_key = None
        if self._recommendation_cache is not None:
            cache_key = (
                data_profile.row_count // 100000,  # Bucket by 100k rows
                data_profile.column_count,
                operation.value,
                int(system_resources.available_memory_gb)
            )
            
            if cache_key in self._recommendation_cache:
                cached_rec = self._recommendation_cache[cache_key]
                logger.debug(f"Using cached recommendation: {cached_rec.engine.value}")
                return cached_rec
        
        # Calculate scores for each available engine
        engine_scores = {}
        reasoning_details = {}
        
        for engine, adapter in self.adapters.items():
            if not self.available_engines[engine]:
                continue
            
            scores = []
            reasons = []
            
            # Performance score
            perf_score = adapter.estimate_performance(operation, data_profile)
            scores.append(perf_score * 0.4)  # 40% weight
            reasons.append(f"Performance score: {perf_score:.1f}")
            
            # Memory efficiency score
            estimated_memory = adapter.estimate_memory_usage(data_profile)
            memory_ratio = estimated_memory / (system_resources.available_memory_gb * 1024)
            
            if memory_ratio > 0.8:
                memory_score = 0  # Too much memory
                reasons.append(f"Memory usage too high: {estimated_memory:.1f}MB")
            elif memory_ratio > 0.5:
                memory_score = 30
                reasons.append(f"High memory usage: {estimated_memory:.1f}MB")
            elif memory_ratio > 0.2:
                memory_score = 70
                reasons.append(f"Moderate memory usage: {estimated_memory:.1f}MB")
            else:
                memory_score = 100
                reasons.append(f"Low memory usage: {estimated_memory:.1f}MB")
            
            scores.append(memory_score * 0.3)  # 30% weight
            
            # Data size appropriateness
            size_score = self._calculate_size_appropriateness(engine, data_profile)
            scores.append(size_score * 0.2)  # 20% weight
            reasons.append(f"Size appropriateness: {size_score:.1f}")
            
            # Operation support score
            if operation in adapter.get_supported_operations():
                support_score = 100
                reasons.append("Operation fully supported")
            else:
                support_score = 0
                reasons.append("Operation not supported")
            
            scores.append(support_score * 0.1)  # 10% weight
            
            total_score = sum(scores)
            engine_scores[engine] = total_score
            reasoning_details[engine] = reasons
        
        # Select best engine
        if not engine_scores:
            raise RuntimeError("No suitable engines available")
        
        best_engine = max(engine_scores.items(), key=lambda x: x[1])[0]
        confidence = engine_scores[best_engine] / 100.0
        
        # Create fallback list
        sorted_engines = sorted(engine_scores.items(), key=lambda x: x[1], reverse=True)
        fallback_engines = [engine for engine, _ in sorted_engines[1:]]
        
        # Build final reasoning
        final_reasoning = [
            f"Selected {best_engine.value} with score {engine_scores[best_engine]:.1f}",
            f"Data size: {data_profile.row_count:,} rows, {data_profile.column_count} columns",
            f"Operation: {operation.value}",
            f"Available memory: {system_resources.available_memory_gb:.1f}GB"
        ]
        final_reasoning.extend(reasoning_details[best_engine])
        
        recommendation = EngineRecommendation(
            engine=best_engine,
            confidence=confidence,
            reasoning=final_reasoning,
            performance_estimate=engine_scores,
            fallback_engines=fallback_engines
        )
        
        # Cache recommendation
        if self._recommendation_cache is not None and cache_key is not None:
            self._recommendation_cache[cache_key] = recommendation
        
        logger.info(f"Recommended engine: {best_engine.value} (confidence: {confidence:.2f})")
        
        return recommendation
    
    def select_engine(self, 
                     data: Union[pd.DataFrame, str, Path],
                     operation: OperationType,
                     force_engine: Optional[DataEngine] = None) -> EngineRecommendation:
        """
        Select the best engine for the given data and operation.
        
        Args:
            data: Data to process (DataFrame or file path)
            operation: Type of operation to perform
            force_engine: Force a specific engine (overrides recommendation)
            
        Returns:
            EngineRecommendation
        """
        if force_engine is not None:
            if not self.available_engines.get(force_engine, False):
                raise RuntimeError(f"Forced engine {force_engine.value} is not available")
            
            return EngineRecommendation(
                engine=force_engine,
                confidence=1.0,
                reasoning=[f"Engine forced by user: {force_engine.value}"]
            )
        
        # Profile the data
        data_profile = self.profile_data(data)
        
        # Get system resources
        system_resources = self.get_system_resources()
        
        # Get recommendation
        recommendation = self.recommend_engine(data_profile, operation, system_resources)
        
        # Optionally run benchmark
        if self.enable_benchmarking and isinstance(data, pd.DataFrame):
            benchmark_results = self._benchmark_engines(data, operation)
            recommendation.performance_estimate = benchmark_results
        
        return recommendation
    
    def get_engine_adapter(self, engine: DataEngine):
        """Get the adapter for a specific engine."""
        if engine not in self.available_engines or not self.available_engines[engine]:
            raise RuntimeError(f"Engine {engine.value} is not available")
        
        return self.adapters[engine]
    
    def _calculate_size_appropriateness(self, engine: DataEngine, data_profile: DataProfile) -> float:
        """Calculate how appropriate an engine is for the data size."""
        row_count = data_profile.row_count
        
        if engine == DataEngine.PANDAS:
            if row_count < 100_000:
                return 100  # Perfect for small data
            elif row_count < 1_000_000:
                return 85   # Good for medium data
            elif row_count < 10_000_000:
                return 60   # Okay for large data
            else:
                return 30   # Not ideal for very large data
        
        elif engine == DataEngine.POLARS:
            if row_count < 10_000:
                return 70   # Overhead for very small data
            elif row_count < 1_000_000:
                return 90   # Good for medium data
            elif row_count < 50_000_000:
                return 100  # Excellent for large data
            else:
                return 85   # Still very good for very large data
        
        elif engine == DataEngine.DUCKDB:
            if row_count < 100_000:
                return 60   # Overhead for small data
            elif row_count < 1_000_000:
                return 80   # Good for medium data
            elif row_count < 100_000_000:
                return 100  # Excellent for large data
            else:
                return 95   # Outstanding for very large data
        
        return 50  # Default score
    
    def _benchmark_engines(self, data: pd.DataFrame, operation: OperationType) -> Dict[str, float]:
        """
        Run simple benchmarks on available engines.
        
        Args:
            data: Sample data for benchmarking
            operation: Operation to benchmark
            
        Returns:
            Dictionary of engine performance times
        """
        results = {}
        
        # Limit benchmark data size to avoid long waits
        benchmark_data = data.head(min(10000, len(data)))
        
        for engine in self.available_engines:
            if not self.available_engines[engine]:
                continue
            
            try:
                start_time = time.time()
                
                if engine == DataEngine.PANDAS:
                    self._benchmark_pandas(benchmark_data, operation)
                elif engine == DataEngine.POLARS:
                    self._benchmark_polars(benchmark_data, operation)
                elif engine == DataEngine.DUCKDB:
                    self._benchmark_duckdb(benchmark_data, operation)
                
                execution_time = time.time() - start_time
                results[engine.value] = execution_time
                
            except Exception as e:
                logger.warning(f"Benchmark failed for {engine.value}: {e}")
                results[engine.value] = float('inf')
        
        return results
    
    def _benchmark_pandas(self, data: pd.DataFrame, operation: OperationType):
        """Benchmark pandas operation."""
        if operation == OperationType.GROUPBY:
            if len(data.select_dtypes(include=['object']).columns) > 0:
                col = data.select_dtypes(include=['object']).columns[0]
                data.groupby(col).size()
        elif operation == OperationType.FILTER:
            if len(data.select_dtypes(include=[np.number]).columns) > 0:
                col = data.select_dtypes(include=[np.number]).columns[0]
                data[data[col] > data[col].median()]
        elif operation == OperationType.SORT:
            if len(data.columns) > 0:
                data.sort_values(data.columns[0])
    
    def _benchmark_polars(self, data: pd.DataFrame, operation: OperationType):
        """Benchmark polars operation."""
        try:
            import polars as pl
            
            # Convert to polars
            pl_data = pl.from_pandas(data)
            
            if operation == OperationType.GROUPBY:
                if len(data.select_dtypes(include=['object']).columns) > 0:
                    col = data.select_dtypes(include=['object']).columns[0]
                    pl_data.groupby(col).count().collect()
            elif operation == OperationType.FILTER:
                if len(data.select_dtypes(include=[np.number]).columns) > 0:
                    col = data.select_dtypes(include=[np.number]).columns[0]
                    median_val = data[col].median()
                    pl_data.filter(pl.col(col) > median_val).collect()
            elif operation == OperationType.SORT:
                if len(data.columns) > 0:
                    pl_data.sort(data.columns[0]).collect()
        
        except ImportError:
            raise RuntimeError("Polars not available for benchmarking")
    
    def _benchmark_duckdb(self, data: pd.DataFrame, operation: OperationType):
        """Benchmark duckdb operation."""
        try:
            import duckdb
            
            # Create temporary table
            conn = duckdb.connect(':memory:')
            conn.register('temp_table', data)
            
            if operation == OperationType.GROUPBY:
                if len(data.select_dtypes(include=['object']).columns) > 0:
                    col = data.select_dtypes(include=['object']).columns[0]
                    conn.execute(f'SELECT {col}, COUNT(*) FROM temp_table GROUP BY {col}').fetchall()
            elif operation == OperationType.FILTER:
                if len(data.select_dtypes(include=[np.number]).columns) > 0:
                    col = data.select_dtypes(include=[np.number]).columns[0]
                    median_val = data[col].median()
                    conn.execute(f'SELECT * FROM temp_table WHERE {col} > {median_val}').fetchall()
            elif operation == OperationType.SORT:
                if len(data.columns) > 0:
                    col = data.columns[0]
                    conn.execute(f'SELECT * FROM temp_table ORDER BY {col}').fetchall()
            
            conn.close()
        
        except ImportError:
            raise RuntimeError("DuckDB not available for benchmarking")
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get summary of engine capabilities and recommendations."""
        summary = {
            'available_engines': [
                engine.value for engine, available in self.available_engines.items() 
                if available
            ],
            'system_resources': self.get_system_resources().__dict__,
            'configuration': {
                'memory_threshold_gb': self.memory_threshold_gb,
                'large_data_threshold': self.large_data_threshold,
                'very_large_data_threshold': self.very_large_data_threshold,
                'enable_benchmarking': self.enable_benchmarking,
                'cache_recommendations': self.cache_recommendations
            },
            'engine_capabilities': {}
        }
        
        for engine, adapter in self.adapters.items():
            if self.available_engines[engine]:
                summary['engine_capabilities'][engine.value] = {
                    'supported_operations': [op.value for op in adapter.get_supported_operations()],
                    'strengths': self._get_engine_strengths(engine),
                    'ideal_use_cases': self._get_engine_use_cases(engine)
                }
        
        return summary
    
    def _get_engine_strengths(self, engine: DataEngine) -> List[str]:
        """Get engine strengths description."""
        if engine == DataEngine.PANDAS:
            return [
                "Mature ecosystem with extensive libraries",
                "Excellent for data analysis and visualization",
                "Great for small to medium datasets",
                "Rich statistical and ML integration"
            ]
        elif engine == DataEngine.POLARS:
            return [
                "Very fast query execution",
                "Memory efficient",
                "Excellent for data transformations",
                "Lazy evaluation for complex queries"
            ]
        elif engine == DataEngine.DUCKDB:
            return [
                "Outstanding analytical performance",
                "SQL interface",
                "Excellent compression",
                "Great for very large datasets"
            ]
        
        return []
    
    def _get_engine_use_cases(self, engine: DataEngine) -> List[str]:
        """Get ideal use cases for engine."""
        if engine == DataEngine.PANDAS:
            return [
                "Exploratory data analysis",
                "Machine learning preprocessing",
                "Statistical analysis",
                "Small to medium datasets"
            ]
        elif engine == DataEngine.POLARS:
            return [
                "ETL pipelines",
                "Large dataset transformations",
                "Complex aggregations",
                "Performance-critical applications"
            ]
        elif engine == DataEngine.DUCKDB:
            return [
                "Data warehousing",
                "OLAP queries",
                "Very large dataset analysis",
                "SQL-based workflows"
            ]
        
        return []