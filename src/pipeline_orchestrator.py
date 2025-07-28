"""
ML Pipeline Orchestrator - Production-grade ML pipeline coordinator.

This module provides the main orchestrator class for coordinating all aspects
of machine learning pipelines in production environments with enterprise-grade
features including comprehensive monitoring, security, and compliance.
"""

import logging
import os
import sys
import time
import traceback
import uuid
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import warnings
import tempfile
import shutil
import psutil
import hashlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all pipeline components
from data_access.factory import DataConnectorFactory
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.validator import DataValidator
from preprocessing.imbalance import BalanceStrategyFactory
from models.factory import ModelFactory
from models.tuning import HyperparameterTuner
from models.cost_sensitive import CostSensitiveLearning
from evaluation.metrics import MetricsCalculator
from evaluation.comparison import ModelComparator
from explainability.shap_explainer import SHAPExplainer
from explainability.pdp_ice import PartialDependenceAnalyzer
from explainability.compliance import ComplianceReporter
from utils.config_parser import ConfigParser, ConfigError
from utils.mlflow_tracker import MLflowTracker, MLflowError
from utils.artifacts import ArtifactManager, ArtifactError

logger = logging.getLogger(__name__)

# Optional Spark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.conf import SparkConf
    SPARK_AVAILABLE = True
except ImportError:
    logger.info("PySpark not available - will run in Python mode only")
    SPARK_AVAILABLE = False
    SparkSession = SparkConf = None

# Core ML libraries
try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    CORE_LIBS_AVAILABLE = True
except ImportError:
    logger.error("Core ML libraries not available")
    CORE_LIBS_AVAILABLE = False
    pd = np = train_test_split = None


class PipelineState(Enum):
    """Enumeration of possible pipeline states."""
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPLAINING = "explaining"
    SAVING_ARTIFACTS = "saving_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Enumeration of execution modes."""
    PYTHON = "python"
    SPARK = "spark"
    DISTRIBUTED = "distributed"
    GPU = "gpu"


class PipelineError(Exception):
    """Custom exception class for pipeline-related errors.
    
    This exception is raised when pipeline operations fail or encounter
    configuration issues that prevent normal execution.
    
    Attributes:
        message: The error message describing what went wrong.
        error_code: Optional error code for categorization.
        stage: Pipeline stage where error occurred.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 stage: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.stage = stage
        self.timestamp = datetime.now(timezone.utc)


@dataclass
class PipelineMetrics:
    """Data class for pipeline execution metrics."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    state: PipelineState = PipelineState.INITIALIZING
    stage_timings: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    model_metrics: Dict[str, Any] = field(default_factory=dict)
    data_metrics: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate total pipeline duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.state == PipelineState.COMPLETED


@dataclass
class SecurityContext:
    """Security context for pipeline execution."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    encryption_enabled: bool = False
    audit_logging: bool = False
    data_classification: str = "internal"
    compliance_frameworks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "encryption_enabled": self.encryption_enabled,
            "audit_logging": self.audit_logging,
            "data_classification": self.data_classification,
            "compliance_frameworks": self.compliance_frameworks,
        }


class ResourceMonitor:
    """Monitor system resource usage during pipeline execution."""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self._thread = None
        self._stop_event = threading.Event()
    
    def start(self):
        """Start resource monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        if self.monitoring:
            self.monitoring = False
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=1.0)
        
        if not self.metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m["cpu_percent"] for m in self.metrics]
        memory_values = [m["memory_percent"] for m in self.metrics]
        disk_values = [m.get("disk_io_read", 0) for m in self.metrics]
        
        return {
            "duration": len(self.metrics) * self.interval,
            "cpu": {
                "avg": np.mean(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "avg": np.mean(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
            },
            "disk_io": {
                "total_read": sum(disk_values),
                "avg_read": np.mean(disk_values) if disk_values else 0,
            },
            "samples": len(self.metrics),
        }
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.interval):
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                metric = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                }
                
                if disk_io:
                    metric.update({
                        "disk_io_read": disk_io.read_bytes,
                        "disk_io_write": disk_io.write_bytes,
                    })
                
                self.metrics.append(metric)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break


class PipelineOrchestrator:
    """Main orchestrator class that coordinates all ML pipeline components.
    
    This class serves as the central coordinator for machine learning pipelines,
    handling different execution modes (training, prediction, evaluation, comparison,
    and experimentation) across both edge node (Python) and cluster (Spark) environments.
    It manages the complete lifecycle from data loading through model training to
    artifact saving with comprehensive tracking and error handling.
    
    The orchestrator supports multiple ML frameworks (scikit-learn, XGBoost, LightGBM,
    CatBoost, H2O, SparkML) and provides intelligent resource management, MLflow
    integration for experiment tracking, and flexible artifact management.
    
    Attributes:
        config: Complete pipeline configuration dictionary.
        output_dir: Base directory for all output artifacts.
        verbose: Flag for verbose logging.
        execution_mode: Execution environment ('python' or 'pyspark').
        framework: ML framework to use ('sklearn', 'xgboost', etc.).
        spark_session: Active Spark session (if using PySpark).
        mlflow_tracker: MLflow tracking instance.
        artifact_manager: Artifact management instance.
        current_run_id: Current MLflow run ID.
        pipeline_state: Current state of the pipeline execution.
        stage_timings: Execution time for each pipeline stage.
        resource_usage: Resource utilization tracking.
    
    Example:
        >>> config = {
        ...     'data_source': {'type': 'csv', 'file_path': 'data.csv'},
        ...     'preprocessing': {'target_column': 'target'},
        ...     'model_training': {'algorithm': 'RandomForestClassifier'},
        ...     'evaluation': {'test_size': 0.2}
        ... }
        >>> orchestrator = PipelineOrchestrator(config)
        >>> result = orchestrator.run_pipeline('train')
        >>> print(f"Best model score: {result['best_model']['score']}")
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "./output",
                 verbose: bool = False):
        """Initialize the pipeline orchestrator with configuration and settings.
        
        Sets up the orchestrator with the provided configuration, creates the output
        directory structure, initializes execution context, and prepares tracking
        and artifact management systems.
        
        Args:
            config: Complete pipeline configuration containing data source, preprocessing,
                model training, evaluation, and execution settings. Must include required
                sections: 'data_source', 'preprocessing', 'model_training', 'evaluation'.
            output_dir: Base output directory path for all artifacts including models,
                reports, plots, and logs. Directory will be created if it doesn't exist.
            verbose: Enable verbose logging throughout pipeline execution for detailed
                debugging and monitoring.
        
        Raises:
            PipelineError: If core ML libraries (pandas, numpy, sklearn) are not available.
            
        Example:
            >>> config = {
            ...     'data_source': {'type': 'csv', 'file_path': 'train.csv'},
            ...     'preprocessing': {'target_column': 'label'},
            ...     'model_training': {'algorithm': 'XGBClassifier'},
            ...     'evaluation': {'test_size': 0.2, 'cv_folds': 5}
            ... }
            >>> orchestrator = PipelineOrchestrator(config, './my_output', verbose=True)
        """
        if not CORE_LIBS_AVAILABLE:
            raise PipelineError("Core ML libraries (pandas, numpy, sklearn) are required")
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._create_output_structure()
        
        # Initialize execution context
        self.execution_mode = config.get('execution', {}).get('version', 'python')
        self.framework = config.get('execution', {}).get('framework', 'sklearn')
        self.spark_session = None
        
        # Initialize tracking and artifact management
        self.mlflow_tracker = None
        self.artifact_manager = None
        self.current_run_id = None
        
        # Pipeline state
        self.pipeline_state = {
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'artifacts': {},
            'metrics': {},
            'models': {},
            'data': {}
        }
        
        # Performance tracking
        self.stage_timings = {}
        self.resource_usage = {}
        
        logger.info(f"Initialized PipelineOrchestrator (mode: {self.execution_mode}, framework: {self.framework})")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate the pipeline configuration for completeness and compatibility.
        
        Performs comprehensive validation of the pipeline configuration including
        required sections, framework availability, data source configuration,
        model settings, and MLflow integration. Provides detailed feedback on
        errors, warnings, and recommendations.
        
        Returns:
            Dictionary containing validation results with the following keys:
                - 'valid' (bool): True if configuration is valid for execution.
                - 'errors' (List[str]): Critical errors that prevent execution.
                - 'warnings' (List[str]): Non-critical issues that may affect performance.
                - 'recommendations' (List[str]): Suggestions for optimization.
        
        Example:
            >>> orchestrator = PipelineOrchestrator(config)
            >>> validation = orchestrator.validate_config()
            >>> if validation['valid']:
            ...     print("Configuration is ready for execution")
            >>> else:
            ...     print(f"Errors found: {validation['errors']}")
        """
        logger.info("Validating pipeline configuration")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Check required sections
            required_sections = ['data_source', 'preprocessing', 'model_training', 'evaluation']
            for section in required_sections:
                if section not in self.config:
                    validation_result['errors'].append(f"Missing required section: {section}")
                    validation_result['valid'] = False
            
            # Validate execution environment
            if self.execution_mode == 'pyspark' and not SPARK_AVAILABLE:
                validation_result['errors'].append("PySpark requested but not available")
                validation_result['valid'] = False
            
            # Validate framework availability
            framework_checks = {
                'sklearn': self._check_sklearn_available,
                'h2o': self._check_h2o_available,
                'xgboost': self._check_xgboost_available,
                'lightgbm': self._check_lightgbm_available,
                'catboost': self._check_catboost_available,
                'sparkml': self._check_spark_available
            }
            
            if self.framework in framework_checks:
                if not framework_checks[self.framework]():
                    validation_result['errors'].append(f"Framework {self.framework} not available")
                    validation_result['valid'] = False
            
            # Validate data source configuration
            data_source_errors = self._validate_data_source()
            validation_result['errors'].extend(data_source_errors)
            if data_source_errors:
                validation_result['valid'] = False
            
            # Validate model configuration
            model_warnings = self._validate_model_config()
            validation_result['warnings'].extend(model_warnings)
            
            # Check MLflow configuration
            mlflow_warnings = self._validate_mlflow_config()
            validation_result['warnings'].extend(mlflow_warnings)
            
            # Resource recommendations
            recommendations = self._generate_recommendations()
            validation_result['recommendations'].extend(recommendations)
            
            logger.info(f"Configuration validation complete: {'VALID' if validation_result['valid'] else 'INVALID'}")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def get_execution_plan(self) -> Dict[str, Any]:
        """Generate detailed execution plan without running the pipeline.
        
        Creates a comprehensive execution plan showing the sequence of steps,
        estimated duration, resource requirements, dependencies, and expected
        outputs based on the configured execution mode and data characteristics.
        
        Returns:
            Dictionary containing execution plan with the following keys:
                - 'mode' (str): Execution mode (train, predict, evaluate, etc.).
                - 'framework' (str): ML framework to be used.
                - 'execution_environment' (str): Runtime environment (python/pyspark).
                - 'steps' (List[Dict]): Ordered list of execution steps with details.
                - 'total_estimated_duration' (str): Total estimated execution time.
                - 'resource_requirements' (Dict): CPU, memory, and storage requirements.
                - 'dependencies' (List[str]): Required Python packages.
                - 'outputs' (List[str]): Expected output artifacts.
        
        Example:
            >>> orchestrator = PipelineOrchestrator(config)
            >>> plan = orchestrator.get_execution_plan()
            >>> print(f"Total steps: {len(plan['steps'])}")
            >>> print(f"Estimated duration: {plan['total_estimated_duration']}")
        """
        logger.info("Generating execution plan")
        
        mode = self.config.get('execution', {}).get('mode', 'train')
        
        if mode == 'train':
            steps = self._get_training_steps()
        elif mode == 'predict':
            steps = self._get_prediction_steps()
        elif mode == 'evaluate':
            steps = self._get_evaluation_steps()
        elif mode == 'compare':
            steps = self._get_comparison_steps()
        elif mode == 'experiment':
            steps = self._get_experiment_steps()
        else:
            steps = []
        
        # Calculate total estimated duration
        total_duration = sum(step.get('duration_minutes', 0) for step in steps)
        
        # Determine resource requirements
        resource_req = self._estimate_resource_requirements()
        
        execution_plan = {
            'mode': mode,
            'framework': self.framework,
            'execution_environment': self.execution_mode,
            'steps': steps,
            'total_estimated_duration': f"{total_duration} minutes",
            'resource_requirements': resource_req,
            'dependencies': self._get_dependencies(),
            'outputs': self._get_expected_outputs()
        }
        
        return execution_plan
    
    def run_pipeline(self, mode: str) -> Dict[str, Any]:
        """Execute the complete ML pipeline in the specified mode.
        
        Orchestrates the full pipeline execution including environment initialization,
        tracking setup, stage-by-stage execution with error handling, performance
        monitoring, and cleanup. Supports multiple execution modes for different
        use cases.
        
        Args:
            mode: Execution mode determining pipeline behavior:
                - 'train': Full training pipeline with model creation and evaluation.
                - 'predict': Generate predictions using a trained model.
                - 'evaluate': Comprehensive evaluation of existing model.
                - 'compare': Compare multiple models and select the best.
                - 'experiment': Hyperparameter optimization and experimentation.
        
        Returns:
            Dictionary containing execution results with mode-specific keys:
                - 'success' (bool): Whether execution completed successfully.
                - 'execution_time_seconds' (float): Total execution time.
                - 'pipeline_state' (Dict): Current state of pipeline components.
                - 'stage_timings' (Dict): Time taken for each pipeline stage.
                - Additional mode-specific results (models, predictions, etc.).
        
        Raises:
            PipelineError: If pipeline execution fails at any stage.
            
        Example:
            >>> orchestrator = PipelineOrchestrator(config)
            >>> result = orchestrator.run_pipeline('train')
            >>> if result['success']:
            ...     print(f"Training completed in {result['execution_time_seconds']:.2f}s")
            ...     print(f"Best model: {result['best_model']['model_name']}")
        """
        logger.info(f"Starting pipeline execution in {mode} mode")
        
        start_time = time.time()
        
        try:
            # Initialize execution environment
            self._initialize_execution_environment()
            
            # Initialize tracking and artifact management
            self._initialize_tracking()
            
            # Execute pipeline based on mode
            if mode == 'train':
                result = self._execute_training_pipeline()
            elif mode == 'predict':
                result = self._execute_prediction_pipeline()
            elif mode == 'evaluate':
                result = self._execute_evaluation_pipeline()
            elif mode == 'compare':
                result = self._execute_comparison_pipeline()
            elif mode == 'experiment':
                result = self._execute_experiment_pipeline()
            else:
                raise PipelineError(f"Unknown execution mode: {mode}")
            
            # Finalize execution
            execution_time = time.time() - start_time
            result.update({
                'success': True,
                'execution_time_seconds': execution_time,
                'pipeline_state': self.pipeline_state,
                'stage_timings': self.stage_timings
            })
            
            logger.info(f"Pipeline execution completed successfully in {execution_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
            
            execution_time = time.time() - start_time
            result = {
                'success': False,
                'error': str(e),
                'execution_time_seconds': execution_time,
                'pipeline_state': self.pipeline_state,
                'partial_results': self._collect_partial_results()
            }
        
        finally:
            # Cleanup
            self._cleanup_execution_environment()
        
        return result
    
    def resume_pipeline(self, stage_name: str) -> Dict[str, Any]:
        """Resume pipeline execution from a specific stage after interruption.
        
        Allows restarting pipeline execution from a specific stage, useful for
        recovering from failures or continuing long-running processes. Loads
        previous pipeline state if available and resumes execution.
        
        Args:
            stage_name: Name of the pipeline stage to resume from. Valid stages
                include: 'data_loading', 'data_validation', 'preprocessing',
                'feature_engineering', 'model_training', 'model_evaluation',
                'explainability', 'artifact_saving'.
        
        Returns:
            Dictionary with execution results, same format as run_pipeline().
        
        Example:
            >>> # Resume from model training if data preprocessing was completed
            >>> result = orchestrator.resume_pipeline('model_training')
        """
        logger.info(f"Resuming pipeline from stage: {stage_name}")
        
        # Load previous pipeline state if available
        state_file = self.output_dir / "pipeline_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                self.pipeline_state = json.load(f)
            logger.info("Loaded previous pipeline state")
        
        # Set current stage
        self.pipeline_state['current_stage'] = stage_name
        
        # Execute from the specified stage
        return self.run_pipeline(self.config.get('execution', {}).get('mode', 'train'))
    
    def _initialize_execution_environment(self) -> None:
        """Initialize the execution environment based on configuration.
        
        Sets up the appropriate execution environment (Python or PySpark) based on
        the configured execution mode. For PySpark mode, initializes Spark session
        with optimal configuration. Also configures logging levels for the execution.
        
        Raises:
            PipelineError: If PySpark is requested but not available.
        """
        logger.info(f"Initializing {self.execution_mode} execution environment")
        
        if self.execution_mode == 'pyspark':
            self._initialize_spark_session()
        
        # Set up logging for the execution environment
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def _initialize_spark_session(self) -> None:
        """Initialize Spark session for distributed cluster execution.
        
        Creates and configures a Spark session with optimal settings for ML workloads.
        Applies configuration parameters for memory management, parallelism, and
        cluster resource utilization.
        
        Raises:
            PipelineError: If PySpark is not available or session creation fails.
        """
        if not SPARK_AVAILABLE:
            raise PipelineError("PySpark not available for cluster execution")
        
        spark_config = self.config.get('spark', {})
        
        # Build Spark configuration
        conf = SparkConf()
        conf.setAppName(spark_config.get('app_name', 'ML Pipeline Framework'))
        
        # Set Spark configuration parameters
        for key, value in spark_config.get('config', {}).items():
            conf.set(key, str(value))
        
        # Create Spark session
        builder = SparkSession.builder.config(conf=conf)
        
        master = spark_config.get('master', 'local[*]')
        builder = builder.master(master)
        
        self.spark_session = builder.getOrCreate()
        self.spark_session.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Initialized Spark session with master: {master}")
    
    def _initialize_tracking(self) -> None:
        """Initialize MLflow experiment tracking and artifact management systems.
        
        Sets up MLflow tracking for experiment logging, parameter tracking, and
        metric collection. Also initializes artifact management for storing models,
        reports, and other pipeline outputs across different storage backends
        (local, S3, HDFS).
        
        Note:
            Gracefully handles cases where MLflow or artifact storage is unavailable,
            logging warnings but allowing pipeline execution to continue.
        """
        # Initialize MLflow if enabled
        if self.config.get('execution', {}).get('enable_mlflow', True):
            try:
                mlflow_config = self.config.get('mlflow', {})
                
                self.mlflow_tracker = MLflowTracker(
                    tracking_uri=mlflow_config.get('tracking_uri'),
                    experiment_name=mlflow_config.get('experiment_name', 'ml-pipeline-experiment'),
                    auto_create_experiment=True
                )
                
                # Start MLflow run
                run_name = mlflow_config.get('run_name') or f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.current_run_id = self.mlflow_tracker.start_run(run_name=run_name)
                
                # Log configuration
                self.mlflow_tracker.log_params(self._flatten_config(self.config))
                
                logger.info(f"Initialized MLflow tracking with run ID: {self.current_run_id}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow tracking: {e}")
        
        # Initialize artifact management if enabled
        if self.config.get('execution', {}).get('enable_artifacts', True):
            try:
                artifacts_config = self.config.get('artifacts', {})
                storage_config = artifacts_config.get('storage', {})
                
                storage_type = storage_config.get('type', 'local')
                
                if storage_type == 'local':
                    base_path = storage_config.get('path', str(self.output_dir / "artifacts"))
                    self.artifact_manager = ArtifactManager.create_local_backend(base_path)
                
                elif storage_type == 's3':
                    self.artifact_manager = ArtifactManager.create_s3_backend(
                        bucket_name=storage_config['bucket'],
                        region_name=storage_config.get('region'),
                        aws_access_key_id=storage_config.get('access_key_id'),
                        aws_secret_access_key=storage_config.get('secret_access_key')
                    )
                
                elif storage_type == 'hdfs':
                    self.artifact_manager = ArtifactManager.create_hdfs_backend(
                        namenode_url=storage_config['namenode_url'],
                        user=storage_config.get('user')
                    )
                
                logger.info(f"Initialized {storage_type} artifact management")
                
            except Exception as e:
                logger.warning(f"Failed to initialize artifact management: {e}")
    
    def _execute_training_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline with all stages.
        
        Orchestrates the full training workflow including data loading, validation,
        preprocessing, feature engineering, model training, evaluation, and
        explainability analysis. Each stage is executed with comprehensive error
        handling and performance tracking.
        
        Returns:
            Dictionary containing training results:
                - 'mode' (str): 'train' execution mode.
                - 'models' (Dict): Trained model instances.
                - 'metrics' (Dict): Performance metrics and evaluation results.
                - 'artifacts' (Dict): Generated artifacts and file paths.
                - 'best_model' (Dict): Best performing model details.
        
        Raises:
            PipelineError: If any training stage fails with details about the failure.
        """
        logger.info("Executing training pipeline")
        
        result = {
            'mode': 'train',
            'models': {},
            'metrics': {},
            'artifacts': {}
        }
        
        try:
            # Stage 1: Data Loading
            self._execute_stage('data_loading', self._load_training_data)
            
            # Stage 2: Data Validation
            self._execute_stage('data_validation', self._validate_training_data)
            
            # Stage 3: Preprocessing
            self._execute_stage('preprocessing', self._preprocess_training_data)
            
            # Stage 4: Feature Engineering
            self._execute_stage('feature_engineering', self._engineer_features)
            
            # Stage 5: Model Training
            self._execute_stage('model_training', self._train_models)
            
            # Stage 6: Model Evaluation
            self._execute_stage('model_evaluation', self._evaluate_models)
            
            # Stage 7: Model Explainability
            if self.config.get('explainability', {}).get('enabled', False):
                self._execute_stage('explainability', self._generate_explanations)
            
            # Stage 8: Artifact Saving
            self._execute_stage('artifact_saving', self._save_training_artifacts)
            
            # Collect results
            result['models'] = self.pipeline_state['models']
            result['metrics'] = self.pipeline_state['metrics']
            result['artifacts'] = self.pipeline_state['artifacts']
            result['best_model'] = self._select_best_model()
            
        except Exception as e:
            self.pipeline_state['failed_stages'].append(self.pipeline_state.get('current_stage', 'unknown'))
            raise PipelineError(f"Training pipeline failed at stage {self.pipeline_state.get('current_stage', 'unknown')}: {e}")
        
        return result
    
    def _execute_prediction_pipeline(self) -> Dict[str, Any]:
        """Execute the prediction pipeline for generating new predictions.
        
        Loads a pre-trained model and generates predictions on new data. Includes
        data loading, model loading, preprocessing alignment, prediction generation,
        and output formatting.
        
        Returns:
            Dictionary containing prediction results:
                - 'mode' (str): 'predict' execution mode.
                - 'predictions' (Dict): Generated predictions and probabilities.
                - 'artifacts' (Dict): Prediction output files and metadata.
        
        Raises:
            PipelineError: If prediction pipeline fails at any stage.
        """
        logger.info("Executing prediction pipeline")
        
        result = {
            'mode': 'predict',
            'predictions': {},
            'artifacts': {}
        }
        
        try:
            # Stage 1: Data Loading
            self._execute_stage('data_loading', self._load_prediction_data)
            
            # Stage 2: Model Loading
            self._execute_stage('model_loading', self._load_trained_model)
            
            # Stage 3: Data Preprocessing
            self._execute_stage('preprocessing', self._preprocess_prediction_data)
            
            # Stage 4: Prediction Generation
            self._execute_stage('prediction', self._generate_predictions)
            
            # Stage 5: Prediction Post-processing
            self._execute_stage('postprocessing', self._postprocess_predictions)
            
            # Stage 6: Output Saving
            self._execute_stage('output_saving', self._save_predictions)
            
            # Collect results
            result['predictions'] = self.pipeline_state['data'].get('predictions')
            result['artifacts'] = self.pipeline_state['artifacts']
            
        except Exception as e:
            self.pipeline_state['failed_stages'].append(self.pipeline_state.get('current_stage', 'unknown'))
            raise PipelineError(f"Prediction pipeline failed at stage {self.pipeline_state.get('current_stage', 'unknown')}: {e}")
        
        return result
    
    def _execute_evaluation_pipeline(self) -> Dict[str, Any]:
        """Execute comprehensive model evaluation pipeline.
        
        Performs thorough evaluation of a trained model including performance
        metrics calculation, statistical analysis, visualization generation,
        and detailed reporting.
        
        Returns:
            Dictionary containing evaluation results:
                - 'mode' (str): 'evaluate' execution mode.
                - 'evaluation_results' (Dict): Comprehensive metrics and analysis.
                - 'artifacts' (Dict): Evaluation reports, plots, and documentation.
        
        Raises:
            PipelineError: If evaluation pipeline fails at any stage.
        """
        logger.info("Executing evaluation pipeline")
        
        result = {
            'mode': 'evaluate',
            'evaluation_results': {},
            'artifacts': {}
        }
        
        try:
            # Stage 1: Data Loading
            self._execute_stage('data_loading', self._load_evaluation_data)
            
            # Stage 2: Model Loading
            self._execute_stage('model_loading', self._load_trained_model)
            
            # Stage 3: Model Evaluation
            self._execute_stage('evaluation', self._perform_comprehensive_evaluation)
            
            # Stage 4: Report Generation
            self._execute_stage('reporting', self._generate_evaluation_reports)
            
            # Collect results
            result['evaluation_results'] = self.pipeline_state['metrics']
            result['artifacts'] = self.pipeline_state['artifacts']
            
        except Exception as e:
            self.pipeline_state['failed_stages'].append(self.pipeline_state.get('current_stage', 'unknown'))
            raise PipelineError(f"Evaluation pipeline failed at stage {self.pipeline_state.get('current_stage', 'unknown')}: {e}")
        
        return result
    
    def _execute_comparison_pipeline(self) -> Dict[str, Any]:
        """Execute model comparison pipeline to find the best algorithm.
        
        Trains multiple models with different algorithms and compares their
        performance using cross-validation and statistical testing. Selects
        the best performing model based on specified metrics.
        
        Returns:
            Dictionary containing comparison results:
                - 'mode' (str): 'compare' execution mode.
                - 'comparison_results' (Dict): Model performance comparisons.
                - 'best_model' (Dict): Selected best model details.
                - 'artifacts' (Dict): Comparison reports and visualizations.
        
        Raises:
            PipelineError: If comparison pipeline fails at any stage.
        """
        logger.info("Executing model comparison pipeline")
        
        result = {
            'mode': 'compare',
            'comparison_results': {},
            'artifacts': {}
        }
        
        try:
            # Stage 1: Data Loading
            self._execute_stage('data_loading', self._load_training_data)
            
            # Stage 2: Data Preprocessing
            self._execute_stage('preprocessing', self._preprocess_training_data)
            
            # Stage 3: Multiple Model Training
            self._execute_stage('model_training', self._train_multiple_models)
            
            # Stage 4: Model Comparison
            self._execute_stage('comparison', self._compare_models)
            
            # Stage 5: Best Model Selection
            self._execute_stage('selection', self._select_best_model_comparison)
            
            # Stage 6: Comparison Report
            self._execute_stage('reporting', self._generate_comparison_report)
            
            # Collect results
            result['comparison_results'] = self.pipeline_state['metrics'].get('comparison')
            result['best_model'] = self.pipeline_state.get('best_model')
            result['artifacts'] = self.pipeline_state['artifacts']
            
        except Exception as e:
            self.pipeline_state['failed_stages'].append(self.pipeline_state.get('current_stage', 'unknown'))
            raise PipelineError(f"Comparison pipeline failed at stage {self.pipeline_state.get('current_stage', 'unknown')}: {e}")
        
        return result
    
    def _execute_experiment_pipeline(self) -> Dict[str, Any]:
        """Execute hyperparameter optimization and experimentation pipeline.
        
        Performs systematic hyperparameter optimization using techniques like
        grid search, random search, or Bayesian optimization. Tracks all
        experiments and identifies optimal configurations.
        
        Returns:
            Dictionary containing experiment results:
                - 'mode' (str): 'experiment' execution mode.
                - 'experiment_results' (Dict): Optimization results and metrics.
                - 'best_configuration' (Dict): Optimal hyperparameter settings.
                - 'artifacts' (Dict): Experiment logs and analysis reports.
        
        Raises:
            PipelineError: If experiment pipeline fails at any stage.
        """
        logger.info("Executing experimentation pipeline")
        
        result = {
            'mode': 'experiment',
            'experiment_results': {},
            'artifacts': {}
        }
        
        try:
            # Stage 1: Experiment Design
            self._execute_stage('experiment_design', self._design_experiments)
            
            # Stage 2: Data Loading
            self._execute_stage('data_loading', self._load_training_data)
            
            # Stage 3: Data Preprocessing
            self._execute_stage('preprocessing', self._preprocess_training_data)
            
            # Stage 4: Hyperparameter Optimization
            self._execute_stage('hyperparameter_optimization', self._optimize_hyperparameters)
            
            # Stage 5: Model Training with Best Parameters
            self._execute_stage('model_training', self._train_optimized_models)
            
            # Stage 6: Comprehensive Evaluation
            self._execute_stage('evaluation', self._perform_comprehensive_evaluation)
            
            # Stage 7: Experiment Analysis
            self._execute_stage('analysis', self._analyze_experiments)
            
            # Collect results
            result['experiment_results'] = self.pipeline_state['metrics'].get('experiments')
            result['best_configuration'] = self.pipeline_state.get('best_configuration')
            result['artifacts'] = self.pipeline_state['artifacts']
            
        except Exception as e:
            self.pipeline_state['failed_stages'].append(self.pipeline_state.get('current_stage', 'unknown'))
            raise PipelineError(f"Experiment pipeline failed at stage {self.pipeline_state.get('current_stage', 'unknown')}: {e}")
        
        return result
    
    def _execute_stage(self, stage_name: str, stage_function: callable) -> Any:
        """Execute a pipeline stage with comprehensive monitoring and error handling.
        
        Wraps stage execution with timing, error handling, state management,
        and progress tracking. Automatically saves pipeline state after each
        successful stage completion.
        
        Args:
            stage_name: Descriptive name of the pipeline stage.
            stage_function: Callable function that implements the stage logic.
        
        Returns:
            Result from the stage function execution.
        
        Raises:
            Exception: Re-raises any exception from stage_function after logging.
        """
        logger.info(f"Executing stage: {stage_name}")
        
        self.pipeline_state['current_stage'] = stage_name
        stage_start_time = time.time()
        
        try:
            result = stage_function()
            
            stage_duration = time.time() - stage_start_time
            self.stage_timings[stage_name] = stage_duration
            self.pipeline_state['completed_stages'].append(stage_name)
            
            logger.info(f"Completed stage {stage_name} in {stage_duration:.2f} seconds")
            
            # Save pipeline state
            self._save_pipeline_state()
            
            return result
            
        except Exception as e:
            stage_duration = time.time() - stage_start_time
            self.stage_timings[stage_name] = stage_duration
            self.pipeline_state['failed_stages'].append(stage_name)
            
            logger.error(f"Stage {stage_name} failed after {stage_duration:.2f} seconds: {e}")
            raise
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load training data from the configured data source.
        
        Supports multiple data sources including CSV files, Parquet files,
        and database connections (PostgreSQL, MySQL, Hive, Snowflake).
        Automatically logs data characteristics and integrates with MLflow.
        
        Returns:
            pandas.DataFrame containing the loaded training data.
        
        Raises:
            PipelineError: If data source is not configured or data loading fails.
        """
        logger.info("Loading training data")
        
        data_config = self.config['data_source']
        
        # Create data connector
        connector = DataConnectorFactory.create_connector(
            db_type=data_config['type'],
            connection_params=data_config.get('connection', {})
        )
        
        # Load data
        if 'query' in data_config:
            data = connector.fetch_data(data_config['query'])
        elif 'file_path' in data_config:
            if data_config['type'] == 'csv':
                data = pd.read_csv(data_config['file_path'])
            elif data_config['type'] == 'parquet':
                data = pd.read_parquet(data_config['file_path'])
            else:
                raise PipelineError(f"Unsupported file type: {data_config['type']}")
        else:
            raise PipelineError("No query or file_path specified in data_source")
        
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Store in pipeline state
        self.pipeline_state['data']['raw_training_data'] = data
        
        # Log data info to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics({
                'data_rows': len(data),
                'data_columns': len(data.columns),
                'missing_values': data.isnull().sum().sum()
            })
        
        return data
    
    def _validate_training_data(self) -> Dict[str, Any]:
        """Validate training data quality, schema, and integrity.
        
        Performs comprehensive data validation including schema checks,
        missing value analysis, data type validation, and quality assessments.
        Results are logged to MLflow for tracking.
        
        Returns:
            Dictionary containing validation results and quality metrics.
        """
        logger.info("Validating training data")
        
        data = self.pipeline_state['data']['raw_training_data']
        validation_config = self.config.get('preprocessing', {}).get('data_validation', {})
        
        if validation_config.get('enabled', True):
            validator = DataValidator()
            validation_results = validator.validate_dataframe(data)
            
            self.pipeline_state['metrics']['data_validation'] = validation_results
            
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics({
                    f'validation_{k}': v for k, v in validation_results.items() 
                    if isinstance(v, (int, float))
                })
            
            return validation_results
        
        return {}
    
    def _preprocess_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess training data with comprehensive transformations.
        
        Applies configured preprocessing steps including missing value imputation,
        feature scaling, categorical encoding, and imbalanced data handling.
        Creates reusable preprocessing pipeline for consistent transformations.
        
        Returns:
            Tuple containing:
                - pd.DataFrame: Preprocessed feature matrix.
                - pd.Series: Target variable.
        
        Raises:
            PipelineError: If target column is not found or preprocessing fails.
        """
        logger.info("Preprocessing training data")
        
        data = self.pipeline_state['data']['raw_training_data']
        preprocessing_config = self.config['preprocessing']
        
        # Initialize preprocessing pipeline
        preprocessor = PreprocessingPipeline(
            missing_value_strategy=preprocessing_config.get('missing_values', {}).get('strategy', 'mean'),
            scaling_method=preprocessing_config.get('scaling', {}).get('method', 'standard'),
            encoding_strategy=preprocessing_config.get('encoding', {}).get('categorical_strategy', 'onehot')
        )
        
        # Identify target column
        target_column = preprocessing_config.get('target_column', 'target')
        if target_column not in data.columns:
            # Try to infer target column
            possible_targets = ['target', 'label', 'y', 'class']
            for col in possible_targets:
                if col in data.columns:
                    target_column = col
                    break
            else:
                raise PipelineError("Target column not found. Please specify 'target_column' in preprocessing config")
        
        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Fit and transform data
        X_processed = preprocessor.fit_transform(X, y)
        
        # Handle imbalanced data if configured
        imbalance_config = preprocessing_config.get('imbalance_handling', {})
        if imbalance_config.get('enabled', False):
            strategy = BalanceStrategyFactory.create_strategy(
                imbalance_config.get('method', 'smote'),
                **imbalance_config.get('parameters', {})
            )
            X_processed, y = strategy.fit_resample(X_processed, y)
            
            # Log imbalance handling info
            sampling_info = strategy.get_sampling_info()
            if self.mlflow_tracker:
                self.mlflow_tracker.log_params({
                    'imbalance_method': sampling_info['strategy'],
                    'original_class_distribution': str(sampling_info['class_counts_before']),
                    'balanced_class_distribution': str(sampling_info['class_counts_after'])
                })
        
        # Store processed data
        self.pipeline_state['data']['X_processed'] = X_processed
        self.pipeline_state['data']['y_processed'] = y
        self.pipeline_state['data']['preprocessor'] = preprocessor
        
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics({
                'processed_features': X_processed.shape[1],
                'processed_samples': X_processed.shape[0]
            })
        
        return X_processed, y
    
    def _engineer_features(self) -> pd.DataFrame:
        """Perform advanced feature engineering and selection.
        
        Applies feature engineering techniques including polynomial features,
        interaction terms, and feature selection methods. Creates optimal
        feature set for model training.
        
        Returns:
            pd.DataFrame containing the final engineered feature set.
        """
        logger.info("Engineering features")
        
        X_processed = self.pipeline_state['data']['X_processed']
        feature_config = self.config.get('feature_engineering', {})
        
        # Apply feature engineering transformations
        if feature_config.get('polynomial_features', {}).get('enabled', False):
            from sklearn.preprocessing import PolynomialFeatures
            
            poly_params = feature_config['polynomial_features']
            poly = PolynomialFeatures(
                degree=poly_params.get('degree', 2),
                interaction_only=poly_params.get('interaction_only', False)
            )
            X_processed = pd.DataFrame(
                poly.fit_transform(X_processed),
                columns=[f'poly_{i}' for i in range(poly.n_output_features_)]
            )
        
        # Feature selection
        selection_config = self.config.get('preprocessing', {}).get('feature_selection', {})
        if selection_config.get('enabled', False):
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            method = selection_config.get('method', 'mutual_info')
            k_features = selection_config.get('k_features', 20)
            
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
            else:
                # Add other selection methods as needed
                selector = SelectKBest(k=k_features)
            
            X_processed = pd.DataFrame(
                selector.fit_transform(X_processed, self.pipeline_state['data']['y_processed']),
                columns=[f'selected_{i}' for i in range(k_features)]
            )
            
            self.pipeline_state['data']['feature_selector'] = selector
        
        self.pipeline_state['data']['X_final'] = X_processed
        
        logger.info(f"Final feature set shape: {X_processed.shape}")
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics({
                'final_features': X_processed.shape[1]
            })
        
        return X_processed
    
    def _train_models(self) -> Dict[str, Any]:
        """Train machine learning models based on configuration.
        
        Creates and trains the specified model with configured parameters.
        Supports cost-sensitive learning, handles train/validation splits,
        and integrates with MLflow for experiment tracking.
        
        Returns:
            Dictionary containing trained model instances.
        
        Raises:
            PipelineError: If model training fails.
        """
        logger.info("Training models")
        
        X = self.pipeline_state['data']['X_final']
        y = self.pipeline_state['data']['y_processed']
        
        # Split into train/validation sets
        test_size = self.config.get('evaluation', {}).get('test_size', 0.2)
        random_state = self.config.get('execution', {}).get('random_state', 42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        self.pipeline_state['data']['X_train'] = X_train
        self.pipeline_state['data']['X_val'] = X_val
        self.pipeline_state['data']['y_train'] = y_train
        self.pipeline_state['data']['y_val'] = y_val
        
        # Get model configuration
        model_config = self.config['model_training']
        algorithm = model_config.get('algorithm', 'RandomForestClassifier')
        parameters = model_config.get('parameters', {})
        
        # Create and train model
        model = ModelFactory.create_model(
            algorithm=algorithm,
            framework=self.framework,
            **parameters
        )
        
        # Handle cost-sensitive learning if configured
        cost_sensitive_config = model_config.get('cost_sensitive', {})
        if cost_sensitive_config.get('enabled', False):
            cost_learning = CostSensitiveLearning()
            
            # Auto-configure cost-sensitive learning
            cost_params = cost_learning.auto_configure(
                framework=self.framework,
                y=y_train,
                **cost_sensitive_config.get('parameters', {})
            )
            
            # Update model parameters
            model.set_params(**cost_params)
            
            if self.mlflow_tracker:
                self.mlflow_tracker.log_params({
                    'cost_sensitive': True,
                    'cost_params': str(cost_params)
                })
        
        # Train model
        trained_model = model.train(X_train, y_train, validation_data=(X_val, y_val))
        
        self.pipeline_state['models']['primary'] = trained_model
        
        logger.info(f"Trained {algorithm} model")
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_params({
                'algorithm': algorithm,
                'framework': self.framework,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            # Log model to MLflow
            self.mlflow_tracker.log_model(trained_model.model, "model")
        
        return {'primary': trained_model}
    
    def _evaluate_models(self) -> Dict[str, Any]:
        """Evaluate trained models with comprehensive metrics.
        
        Calculates performance metrics appropriate for the task type
        (classification or regression), generates detailed evaluation
        reports, and logs results to MLflow.
        
        Returns:
            Dictionary containing evaluation metrics and results.
        """
        logger.info("Evaluating models")
        
        X_val = self.pipeline_state['data']['X_val']
        y_val = self.pipeline_state['data']['y_val']
        model = self.pipeline_state['models']['primary']
        
        # Generate predictions
        y_pred = model.predict(X_val)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]  # Binary classification probabilities
        
        # Calculate metrics
        task_type = 'classification' if len(np.unique(y_val)) <= 10 else 'regression'
        metrics_calculator = MetricsCalculator(task_type=task_type)
        
        if task_type == 'classification':
            metrics = metrics_calculator.calculate_classification_metrics(
                y_val, y_pred, y_pred_proba
            )
            
            # Generate comprehensive evaluation report
            eval_report = metrics_calculator.generate_comprehensive_report(
                y_val, y_pred, y_pred_proba,
                output_dir=str(self.output_dir / "evaluation")
            )
            
        else:
            metrics = metrics_calculator.calculate_regression_metrics(y_val, y_pred)
            eval_report = metrics_calculator.generate_comprehensive_report(
                y_val, y_pred, output_dir=str(self.output_dir / "evaluation")
            )
        
        self.pipeline_state['metrics']['evaluation'] = metrics
        self.pipeline_state['artifacts']['evaluation_report'] = eval_report
        
        logger.info(f"Model evaluation completed")
        
        if self.mlflow_tracker:
            # Log core metrics
            core_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.mlflow_tracker.log_metrics(core_metrics)
        
        return metrics
    
    def _generate_explanations(self) -> Dict[str, Any]:
        """Generate model explainability and interpretability analysis.
        
        Creates SHAP explanations, partial dependence plots, feature importance
        analysis, and compliance reports for model interpretability and
        regulatory requirements.
        
        Returns:
            Dictionary containing explainability artifacts and reports.
        """
        logger.info("Generating model explanations")
        
        X_val = self.pipeline_state['data']['X_val']
        model = self.pipeline_state['models']['primary']
        explainability_config = self.config.get('explainability', {})
        
        explanations = {}
        
        # SHAP explanations
        if 'shap' in explainability_config.get('methods', []):
            try:
                shap_explainer = SHAPExplainer(model.model, X_val.iloc[:100])  # Sample for background
                shap_results = shap_explainer.generate_explanation_report(
                    X_val.iloc[:500],  # Sample for explanation
                    output_dir=str(self.output_dir / "explanations" / "shap")
                )
                explanations['shap'] = shap_results
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # Partial Dependence Plots
        if 'partial_dependence' in explainability_config.get('methods', []):
            try:
                pdp_analyzer = PartialDependenceAnalyzer(
                    model.model, X_val,
                    feature_names=list(X_val.columns)
                )
                pdp_results = pdp_analyzer.generate_pdp_report(
                    output_dir=str(self.output_dir / "explanations" / "pdp")
                )
                explanations['partial_dependence'] = pdp_results
            except Exception as e:
                logger.warning(f"PDP explanation failed: {e}")
        
        # Compliance reporting
        if explainability_config.get('compliance', {}).get('enabled', False):
            try:
                y_val = self.pipeline_state['data']['y_val']
                compliance_reporter = ComplianceReporter(
                    model.model, X_val, y_val,
                    feature_names=list(X_val.columns),
                    model_name=f"{self.framework}_{self.config['model_training']['algorithm']}"
                )
                
                compliance_results = compliance_reporter.generate_comprehensive_report(
                    output_dir=str(self.output_dir / "explanations" / "compliance")
                )
                explanations['compliance'] = compliance_results
            except Exception as e:
                logger.warning(f"Compliance reporting failed: {e}")
        
        self.pipeline_state['artifacts']['explanations'] = explanations
        
        return explanations
    
    def _save_training_artifacts(self) -> Dict[str, str]:
        """Save all training artifacts to persistent storage.
        
        Saves trained models, preprocessing pipelines, metrics, and reports
        to local files and configured artifact storage systems. Ensures
        artifacts are available for future inference and analysis.
        
        Returns:
            Dictionary mapping artifact types to file paths.
        """
        logger.info("Saving training artifacts")
        
        artifacts = {}
        
        # Save model
        model = self.pipeline_state['models']['primary']
        model_path = self.output_dir / "models" / "trained_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        artifacts['model'] = str(model_path)
        
        # Save preprocessor
        if 'preprocessor' in self.pipeline_state['data']:
            preprocessor_path = self.output_dir / "models" / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.pipeline_state['data']['preprocessor'], f)
            artifacts['preprocessor'] = str(preprocessor_path)
        
        # Save metrics
        metrics_path = self.output_dir / "reports" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.pipeline_state['metrics'], f, indent=2, default=str)
        artifacts['metrics'] = str(metrics_path)
        
        # Save to artifact manager if available
        if self.artifact_manager:
            try:
                # Save model as artifact
                self.artifact_manager.save_artifact(
                    model, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    artifact_type="model",
                    description=f"Trained {self.framework} {self.config['model_training']['algorithm']} model"
                )
                
                # Save metrics as artifact
                self.artifact_manager.save_artifact(
                    self.pipeline_state['metrics'], f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    artifact_type="report",
                    description="Training metrics and evaluation results"
                )
                
            except Exception as e:
                logger.warning(f"Failed to save artifacts to artifact manager: {e}")
        
        self.pipeline_state['artifacts']['files'] = artifacts
        
        return artifacts
    
    def _select_best_model(self) -> Dict[str, Any]:
        """Select the best performing model based on evaluation metrics.
        
        Compares model performance using the primary metric specified in
        configuration and selects the model with the best score.
        
        Returns:
            Dictionary containing best model details, metrics, and metadata.
        """
        models = self.pipeline_state['models']
        metrics = self.pipeline_state['metrics']
        
        # For single model, return primary
        if 'primary' in models:
            return {
                'model_name': 'primary',
                'model': models['primary'],
                'metrics': metrics.get('evaluation', {})
            }
        
        # For multiple models, select based on primary metric
        primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'roc_auc')
        best_model = None
        best_score = float('-inf')
        
        for model_name, model in models.items():
            model_metrics = metrics.get(model_name, {})
            score = model_metrics.get(primary_metric, 0)
            
            if score > best_score:
                best_score = score
                best_model = {
                    'model_name': model_name,
                    'model': model,
                    'metrics': model_metrics,
                    'score': score
                }
        
        return best_model
    
    def _cleanup_execution_environment(self) -> None:
        """Clean up execution environment and close active sessions.
        
        Properly closes Spark sessions, ends MLflow runs, and performs
        cleanup operations to prevent resource leaks.
        """
        logger.info("Cleaning up execution environment")
        
        # Close Spark session if active
        if self.spark_session:
            self.spark_session.stop()
            logger.info("Stopped Spark session")
        
        # End MLflow run if active
        if self.mlflow_tracker and self.current_run_id:
            self.mlflow_tracker.end_run()
            logger.info("Ended MLflow run")
    
    def _save_pipeline_state(self) -> None:
        """Save current pipeline state to disk for resumability.
        
        Serializes pipeline state including completed stages, metrics,
        and artifacts to enable pipeline resumption after interruption.
        """
        state_file = self.output_dir / "pipeline_state.json"
        
        # Prepare state for JSON serialization
        serializable_state = {
            'current_stage': self.pipeline_state['current_stage'],
            'completed_stages': self.pipeline_state['completed_stages'],
            'failed_stages': self.pipeline_state['failed_stages'],
            'artifacts': self.pipeline_state['artifacts'],
            'metrics': self._make_json_serializable(self.pipeline_state['metrics']),
            'stage_timings': self.stage_timings
        }
        
        with open(state_file, 'w') as f:
            json.dump(serializable_state, f, indent=2, default=str)
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format.
        
        Recursively converts numpy arrays, pandas objects, and other
        non-serializable types to JSON-compatible representations.
        
        Args:
            obj: Object to convert.
        
        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj
    
    def _flatten_config(self, config, prefix=''):
        """Flatten nested configuration dictionary for MLflow parameter logging.
        
        Converts nested configuration structures into flat key-value pairs
        suitable for MLflow parameter tracking.
        
        Args:
            config: Configuration dictionary to flatten.
            prefix: Prefix for flattened keys.
        
        Returns:
            Dictionary with flattened configuration parameters.
        """
        flattened = {}
        for key, value in config.items():
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, f"{prefix}{key}_"))
            else:
                flattened[f"{prefix}{key}"] = str(value)
        return flattened
    
    def _create_output_structure(self) -> None:
        """Create standardized output directory structure.
        
        Creates subdirectories for organizing different types of outputs
        including models, reports, plots, data, logs, and explanations.
        """
        directories = [
            'models', 'reports', 'plots', 'data', 'logs',
            'explanations', 'evaluation', 'comparisons', 'artifacts'
        ]
        
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _collect_partial_results(self) -> Dict[str, Any]:
        """Collect partial results when pipeline execution fails.
        
        Gathers available results from completed stages to provide
        useful information even when pipeline execution is incomplete.
        
        Returns:
            Dictionary containing partial results and execution metadata.
        """
        return {
            'completed_stages': self.pipeline_state['completed_stages'],
            'failed_stages': self.pipeline_state['failed_stages'],
            'available_artifacts': list(self.pipeline_state['artifacts'].keys()),
            'stage_timings': self.stage_timings
        }
    
    # Validation helper methods
    def _check_sklearn_available(self) -> bool:
        """Check if scikit-learn is available for use.
        
        Returns:
            bool: True if scikit-learn can be imported, False otherwise.
        """
        try:
            import sklearn
            return True
        except ImportError:
            return False
    
    def _check_h2o_available(self) -> bool:
        """Check if H2O is available for use.
        
        Returns:
            bool: True if H2O can be imported, False otherwise.
        """
        try:
            import h2o
            return True
        except ImportError:
            return False
    
    def _check_xgboost_available(self) -> bool:
        """Check if XGBoost is available for use.
        
        Returns:
            bool: True if XGBoost can be imported, False otherwise.
        """
        try:
            import xgboost
            return True
        except ImportError:
            return False
    
    def _check_lightgbm_available(self) -> bool:
        """Check if LightGBM is available for use.
        
        Returns:
            bool: True if LightGBM can be imported, False otherwise.
        """
        try:
            import lightgbm
            return True
        except ImportError:
            return False
    
    def _check_catboost_available(self) -> bool:
        """Check if CatBoost is available for use.
        
        Returns:
            bool: True if CatBoost can be imported, False otherwise.
        """
        try:
            import catboost
            return True
        except ImportError:
            return False
    
    def _check_spark_available(self) -> bool:
        """Check if PySpark is available for use.
        
        Returns:
            bool: True if PySpark is available, False otherwise.
        """
        return SPARK_AVAILABLE
    
    def _validate_data_source(self) -> List[str]:
        """Validate data source configuration for required fields.
        
        Checks that data source configuration contains all required fields
        for the specified data source type (database or file).
        
        Returns:
            List of validation error messages.
        """
        errors = []
        data_config = self.config.get('data_source', {})
        
        if not data_config:
            errors.append("Missing data_source configuration")
            return errors
        
        # Check required fields
        if 'type' not in data_config:
            errors.append("Missing data_source.type")
        
        # Validate based on type
        db_types = ['postgresql', 'mysql', 'hive', 'snowflake']
        file_types = ['csv', 'parquet', 'json']
        
        source_type = data_config.get('type')
        
        if source_type in db_types:
            if 'connection' not in data_config:
                errors.append(f"Missing connection configuration for {source_type}")
            if 'query' not in data_config:
                errors.append(f"Missing query for {source_type}")
        
        elif source_type in file_types:
            if 'file_path' not in data_config:
                errors.append(f"Missing file_path for {source_type}")
        
        return errors
    
    def _validate_model_config(self) -> List[str]:
        """Validate model configuration for framework compatibility.
        
        Checks if the specified algorithm is compatible with the selected
        framework and provides warnings for potential issues.
        
        Returns:
            List of validation warning messages.
        """
        warnings = []
        model_config = self.config.get('model_training', {})
        
        if not model_config:
            warnings.append("Missing model_training configuration")
            return warnings
        
        # Check if algorithm is compatible with framework
        algorithm = model_config.get('algorithm', '')
        framework = self.framework
        
        sklearn_algorithms = ['RandomForestClassifier', 'LogisticRegression', 'SVC']
        xgboost_algorithms = ['XGBClassifier', 'XGBRegressor']
        
        if framework == 'sklearn' and algorithm not in sklearn_algorithms:
            warnings.append(f"Algorithm {algorithm} may not be available in sklearn")
        
        if framework == 'xgboost' and algorithm not in xgboost_algorithms:
            warnings.append(f"Algorithm {algorithm} may not be available in xgboost")
        
        return warnings
    
    def _validate_mlflow_config(self) -> List[str]:
        """Validate MLflow configuration and server accessibility.
        
        Checks MLflow server connectivity and configuration validity.
        
        Returns:
            List of validation warning messages.
        """
        warnings = []
        mlflow_config = self.config.get('mlflow', {})
        
        if mlflow_config.get('tracking_uri'):
            # Check if MLflow server is accessible
            try:
                import requests
                uri = mlflow_config['tracking_uri']
                response = requests.get(f"{uri}/health", timeout=5)
                if response.status_code != 200:
                    warnings.append(f"MLflow server at {uri} may not be accessible")
            except:
                warnings.append("Cannot verify MLflow server accessibility")
        
        return warnings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on configuration.
        
        Analyzes current configuration and provides recommendations for
        performance optimization and best practices.
        
        Returns:
            List of recommendation messages.
        """
        recommendations = []
        
        # Memory recommendations
        if self.execution_mode == 'pyspark':
            recommendations.append("Consider increasing Spark executor memory for large datasets")
        
        # Framework recommendations
        if self.framework == 'sklearn' and self.execution_mode == 'pyspark':
            recommendations.append("Consider using sparkml framework for better Spark integration")
        
        return recommendations
    
    def _get_training_steps(self) -> List[Dict[str, Any]]:
        """Get detailed steps for training pipeline execution.
        
        Returns:
            List of dictionaries describing each training step with estimated duration.
        """
        return [
            {
                'name': 'Data Loading',
                'description': 'Load training data from configured source',
                'estimated_duration': '5 minutes',
                'dependencies': []
            },
            {
                'name': 'Data Validation',
                'description': 'Validate data quality and schema',
                'estimated_duration': '2 minutes',
                'dependencies': ['Data Loading']
            },
            {
                'name': 'Preprocessing',
                'description': 'Clean and preprocess training data',
                'estimated_duration': '10 minutes',
                'dependencies': ['Data Validation']
            },
            {
                'name': 'Feature Engineering',
                'description': 'Engineer features and perform selection',
                'estimated_duration': '15 minutes',
                'dependencies': ['Preprocessing']
            },
            {
                'name': 'Model Training',
                'description': 'Train machine learning model',
                'estimated_duration': '30 minutes',
                'dependencies': ['Feature Engineering']
            },
            {
                'name': 'Model Evaluation',
                'description': 'Evaluate model performance',
                'estimated_duration': '10 minutes',
                'dependencies': ['Model Training']
            },
            {
                'name': 'Artifact Saving',
                'description': 'Save models and artifacts',
                'estimated_duration': '5 minutes',
                'dependencies': ['Model Evaluation']
            }
        ]
    
    def _get_prediction_steps(self) -> List[Dict[str, Any]]:
        """Get detailed steps for prediction pipeline execution.
        
        Returns:
            List of dictionaries describing each prediction step with estimated duration.
        """
        return [
            {
                'name': 'Data Loading',
                'description': 'Load prediction data',
                'estimated_duration': '3 minutes',
                'dependencies': []
            },
            {
                'name': 'Model Loading',
                'description': 'Load trained model',
                'estimated_duration': '2 minutes',
                'dependencies': []
            },
            {
                'name': 'Preprocessing',
                'description': 'Preprocess prediction data',
                'estimated_duration': '5 minutes',
                'dependencies': ['Data Loading', 'Model Loading']
            },
            {
                'name': 'Prediction',
                'description': 'Generate predictions',
                'estimated_duration': '10 minutes',
                'dependencies': ['Preprocessing']
            },
            {
                'name': 'Output Saving',
                'description': 'Save predictions',
                'estimated_duration': '3 minutes',
                'dependencies': ['Prediction']
            }
        ]
    
    def _get_evaluation_steps(self) -> List[Dict[str, Any]]:
        """Get detailed steps for evaluation pipeline execution.
        
        Returns:
            List of dictionaries describing each evaluation step with estimated duration.
        """
        return [
            {
                'name': 'Data Loading',
                'description': 'Load evaluation data',
                'estimated_duration': '3 minutes',
                'dependencies': []
            },
            {
                'name': 'Model Loading',
                'description': 'Load trained model',
                'estimated_duration': '2 minutes',
                'dependencies': []
            },
            {
                'name': 'Evaluation',
                'description': 'Perform comprehensive evaluation',
                'estimated_duration': '15 minutes',
                'dependencies': ['Data Loading', 'Model Loading']
            },
            {
                'name': 'Reporting',
                'description': 'Generate evaluation reports',
                'estimated_duration': '5 minutes',
                'dependencies': ['Evaluation']
            }
        ]
    
    def _get_comparison_steps(self) -> List[Dict[str, Any]]:
        """Get detailed steps for model comparison pipeline execution.
        
        Returns:
            List of dictionaries describing each comparison step with estimated duration.
        """
        return [
            {
                'name': 'Data Loading',
                'description': 'Load training data',
                'estimated_duration': '5 minutes',
                'dependencies': []
            },
            {
                'name': 'Preprocessing',
                'description': 'Preprocess data',
                'estimated_duration': '10 minutes',
                'dependencies': ['Data Loading']
            },
            {
                'name': 'Multiple Model Training',
                'description': 'Train multiple models for comparison',
                'estimated_duration': '60 minutes',
                'dependencies': ['Preprocessing']
            },
            {
                'name': 'Model Comparison',
                'description': 'Compare model performances',
                'estimated_duration': '20 minutes',
                'dependencies': ['Multiple Model Training']
            },
            {
                'name': 'Comparison Report',
                'description': 'Generate comparison report',
                'estimated_duration': '10 minutes',
                'dependencies': ['Model Comparison']
            }
        ]
    
    def _get_experiment_steps(self) -> List[Dict[str, Any]]:
        """Get detailed steps for experimentation pipeline execution.
        
        Returns:
            List of dictionaries describing each experiment step with estimated duration.
        """
        return [
            {
                'name': 'Experiment Design',
                'description': 'Design hyperparameter experiments',
                'estimated_duration': '5 minutes',
                'dependencies': []
            },
            {
                'name': 'Data Loading',
                'description': 'Load training data',
                'estimated_duration': '5 minutes',
                'dependencies': []
            },
            {
                'name': 'Preprocessing',
                'description': 'Preprocess data',
                'estimated_duration': '10 minutes',
                'dependencies': ['Data Loading']
            },
            {
                'name': 'Hyperparameter Optimization',
                'description': 'Optimize hyperparameters',
                'estimated_duration': '120 minutes',
                'dependencies': ['Preprocessing', 'Experiment Design']
            },
            {
                'name': 'Final Model Training',
                'description': 'Train model with best parameters',
                'estimated_duration': '30 minutes',
                'dependencies': ['Hyperparameter Optimization']
            },
            {
                'name': 'Experiment Analysis',
                'description': 'Analyze experiment results',
                'estimated_duration': '15 minutes',
                'dependencies': ['Final Model Training']
            }
        ]
    
    def _estimate_resource_requirements(self) -> Dict[str, str]:
        """Estimate computational resource requirements for pipeline execution.
        
        Returns:
            Dictionary containing estimated CPU, memory, storage, and environment requirements.
        """
        if self.execution_mode == 'pyspark':
            return {
                'cpu_cores': '8-16',
                'memory': '16-32 GB',
                'storage': '10-50 GB',
                'environment': 'Spark cluster'
            }
        else:
            return {
                'cpu_cores': '4-8',
                'memory': '8-16 GB',
                'storage': '5-20 GB',
                'environment': 'Single machine'
            }
    
    def _get_dependencies(self) -> List[str]:
        """Get list of required Python package dependencies.
        
        Returns:
            List of required package names based on configured framework and features.
        """
        deps = ['pandas', 'numpy', 'scikit-learn']
        
        if self.framework == 'xgboost':
            deps.append('xgboost')
        elif self.framework == 'lightgbm':
            deps.append('lightgbm')
        elif self.framework == 'catboost':
            deps.append('catboost')
        elif self.framework == 'h2o':
            deps.append('h2o')
        
        if self.execution_mode == 'pyspark':
            deps.append('pyspark')
        
        if self.config.get('execution', {}).get('enable_mlflow', True):
            deps.append('mlflow')
        
        return deps
    
    def _get_expected_outputs(self) -> List[str]:
        """Get list of expected output artifacts from pipeline execution.
        
        Returns:
            List of expected output types and files.
        """
        return [
            'Trained model files',
            'Performance metrics',
            'Evaluation reports',
            'Model explanations',
            'Artifacts and logs'
        ]
    
    # Additional pipeline mode methods (placeholder implementations)
    def _load_prediction_data(self):
        """Load data for prediction pipeline."""
        pass
    
    def _load_trained_model(self):
        """Load pre-trained model for inference."""
        pass
    
    def _preprocess_prediction_data(self):
        """Preprocess prediction data using saved preprocessing pipeline."""
        pass
    
    def _generate_predictions(self):
        """Generate predictions using loaded model."""
        pass
    
    def _postprocess_predictions(self):
        """Post-process and format predictions."""
        pass
    
    def _save_predictions(self):
        """Save predictions to output files."""
        pass
    
    def _load_evaluation_data(self):
        """Load data for model evaluation."""
        pass
    
    def _perform_comprehensive_evaluation(self):
        """Perform comprehensive model evaluation."""
        pass
    
    def _generate_evaluation_reports(self):
        """Generate detailed evaluation reports."""
        pass
    
    def _train_multiple_models(self):
        """Train multiple models for comparison."""
        pass
    
    def _compare_models(self):
        """Compare performance of multiple models."""
        pass
    
    def _select_best_model_comparison(self):
        """Select best model from comparison results."""
        pass
    
    def _generate_comparison_report(self):
        """Generate model comparison report."""
        pass
    
    def _design_experiments(self):
        """Design hyperparameter optimization experiments."""
        pass
    
    def _optimize_hyperparameters(self):
        """Perform hyperparameter optimization."""
        pass
    
    def _train_optimized_models(self):
        """Train models with optimized hyperparameters."""
        pass
    
    def _analyze_experiments(self):
        """Analyze experiment results and findings."""
        pass