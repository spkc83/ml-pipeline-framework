"""
Configuration validation utilities for ML Pipeline Framework.

This module provides comprehensive validation for pipeline configuration files
including schema validation, dependency checks, and security validation.
"""

import os
import yaml
import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []

class ConfigValidator:
    """Comprehensive configuration validator for ML Pipeline Framework."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: Enable strict validation with additional checks
        """
        self.strict_mode = strict_mode
        self.schema = self._load_schema()
        
    def validate_config(self, config_path: str) -> ValidationResult:
        """Validate a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        try:
            # Load configuration
            config = self._load_config(config_path)
            
            # Schema validation
            self._validate_schema(config, result)
            
            # Business logic validation
            self._validate_business_logic(config, result)
            
            # Security validation
            self._validate_security(config, result)
            
            # Dependency validation
            self._validate_dependencies(config, result)
            
            # Performance validation
            self._validate_performance(config, result)
            
            # Environment validation
            self._validate_environment(config, result)
            
            # Set overall validation status
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation failed: {str(e)}")
            
        return result
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for configuration validation."""
        return {
            "type": "object",
            "properties": {
                "pipeline": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                        "description": {"type": "string"},
                        "environment": {"type": "string", "enum": ["dev", "staging", "prod"]},
                        "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                    },
                    "required": ["name", "version"]
                },
                "data_source": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["csv", "postgresql", "mysql", "snowflake", "redshift", "hive"]},
                        "csv": {"type": "object"},
                        "database": {"type": "object"}
                    },
                    "required": ["type"]
                },
                "preprocessing": {
                    "type": "object",
                    "properties": {
                        "data_quality": {"type": "object"},
                        "cleaning": {"type": "object"},
                        "transformation": {"type": "object"}
                    }
                },
                "model_training": {
                    "type": "object",
                    "properties": {
                        "automl_enabled": {"type": "boolean"},
                        "data_split": {"type": "object"},
                        "target": {"type": "object"},
                        "models": {"type": "array"}
                    }
                },
                "evaluation": {
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "object"},
                        "reports": {"type": "object"}
                    }
                },
                "explainability": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "interpretability_methods": {"type": "array"}
                    }
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "model_artifacts": {"type": "object"},
                        "predictions": {"type": "object"},
                        "reports": {"type": "object"}
                    }
                },
                "monitoring": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "data_drift": {"type": "object"},
                        "performance_monitoring": {"type": "object"}
                    }
                }
            },
            "required": ["pipeline", "data_source"]
        }
    
    def _validate_schema(self, config: Dict[str, Any], result: ValidationResult):
        """Validate configuration against JSON schema."""
        try:
            jsonschema.validate(config, self.schema)
        except jsonschema.ValidationError as e:
            result.errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            result.errors.append(f"Schema error: {e.message}")
    
    def _validate_business_logic(self, config: Dict[str, Any], result: ValidationResult):
        """Validate business logic and configuration consistency."""
        
        # Validate data source configuration
        self._validate_data_source(config, result)
        
        # Validate model training configuration
        self._validate_model_training(config, result)
        
        # Validate AutoML configuration
        self._validate_automl_config(config, result)
        
        # Validate explainability configuration
        self._validate_explainability_config(config, result)
        
        # Validate monitoring configuration
        self._validate_monitoring_config(config, result)
    
    def _validate_data_source(self, config: Dict[str, Any], result: ValidationResult):
        """Validate data source configuration."""
        if 'data_source' not in config:
            result.errors.append("Missing required 'data_source' configuration")
            return
        
        data_source = config['data_source']
        source_type = data_source.get('type')
        
        if source_type == 'csv':
            csv_config = data_source.get('csv', {})
            file_paths = csv_config.get('file_paths', [])
            
            if not file_paths:
                result.errors.append("CSV data source requires at least one file path")
            else:
                # Check if files exist (in strict mode)
                if self.strict_mode:
                    for file_path in file_paths:
                        # Handle environment variables
                        expanded_path = os.path.expandvars(file_path)
                        if not Path(expanded_path).exists() and not '*' in expanded_path:
                            result.warnings.append(f"CSV file may not exist: {file_path}")
        
        elif source_type in ['postgresql', 'mysql', 'snowflake', 'redshift']:
            db_config = data_source.get('database', {})
            connection = db_config.get('connection', {})
            
            required_fields = ['host', 'database', 'username']
            for field in required_fields:
                if not connection.get(field):
                    result.errors.append(f"Database connection missing required field: {field}")
            
            # Check for password or other auth methods
            if not connection.get('password') and not connection.get('auth_method'):
                result.warnings.append("Database connection may be missing authentication")
    
    def _validate_model_training(self, config: Dict[str, Any], result: ValidationResult):
        """Validate model training configuration."""
        if 'model_training' not in config:
            result.warnings.append("Missing 'model_training' configuration")
            return
        
        training_config = config['model_training']
        
        # Validate data split
        data_split = training_config.get('data_split', {})
        train_ratio = data_split.get('train_ratio', 0.7)
        val_ratio = data_split.get('validation_ratio', 0.15)
        test_ratio = data_split.get('test_ratio', 0.15)
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            result.errors.append(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        if train_ratio < 0.5:
            result.warnings.append("Training ratio is less than 0.5, may impact model quality")
        
        # Validate target configuration
        target_config = training_config.get('target', {})
        if not target_config.get('column'):
            result.errors.append("Target column not specified")
        
        target_type = target_config.get('type')
        if target_type not in ['classification', 'regression']:
            result.errors.append(f"Invalid target type: {target_type}")
    
    def _validate_automl_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate AutoML configuration."""
        training_config = config.get('model_training', {})
        automl_enabled = training_config.get('automl_enabled', False)
        
        if automl_enabled:
            automl_config_path = training_config.get('automl_config_path')
            
            if not automl_config_path:
                result.errors.append("AutoML enabled but no config path specified")
            elif self.strict_mode:
                # Check if AutoML config file exists
                expanded_path = os.path.expandvars(automl_config_path)
                if not Path(expanded_path).exists():
                    result.warnings.append(f"AutoML config file may not exist: {automl_config_path}")
        
        # Check for conflicting configurations
        models = training_config.get('models', [])
        if automl_enabled and models:
            result.warnings.append("AutoML enabled with manual model configuration - AutoML will take precedence")
    
    def _validate_explainability_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate explainability configuration."""
        explainability = config.get('explainability', {})
        
        if explainability.get('enabled', False):
            methods = explainability.get('interpretability_methods', [])
            valid_methods = ['shap', 'lime', 'ale', 'anchors', 'counterfactuals']
            
            for method in methods:
                if method not in valid_methods:
                    result.warnings.append(f"Unknown interpretability method: {method}")
            
            if not methods:
                result.warnings.append("Explainability enabled but no methods specified")
    
    def _validate_monitoring_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate monitoring configuration."""
        monitoring = config.get('monitoring', {})
        
        if monitoring.get('enabled', False):
            # Check data drift configuration
            data_drift = monitoring.get('data_drift', {})
            if data_drift.get('enabled', False):
                threshold = data_drift.get('drift_threshold', 0.05)
                if not 0 < threshold < 1:
                    result.errors.append(f"Drift threshold must be between 0 and 1, got {threshold}")
            
            # Check performance monitoring
            perf_monitoring = monitoring.get('performance_monitoring', {})
            if perf_monitoring.get('enabled', False):
                alert_threshold = perf_monitoring.get('alert_threshold', 0.1)
                if not 0 < alert_threshold < 1:
                    result.errors.append(f"Performance alert threshold must be between 0 and 1, got {alert_threshold}")
    
    def _validate_security(self, config: Dict[str, Any], result: ValidationResult):
        """Validate security configuration."""
        
        # Check for hardcoded credentials
        config_str = str(config).lower()
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential']
        
        for pattern in sensitive_patterns:
            if pattern in config_str and not ('${' in config_str or 'env' in config_str):
                result.warnings.append(f"Potential hardcoded {pattern} in configuration")
        
        # Check for secure connections
        data_source = config.get('data_source', {})
        if data_source.get('type') in ['postgresql', 'mysql']:
            connection = data_source.get('database', {}).get('connection', {})
            ssl_mode = connection.get('sslmode', 'disable')
            if ssl_mode == 'disable':
                result.warnings.append("Database connection not using SSL - consider enabling for security")
        
        # Check MLflow configuration
        mlflow_config = config.get('output', {}).get('mlflow', {})
        if mlflow_config.get('enabled', False):
            tracking_uri = mlflow_config.get('tracking_uri', '')
            if tracking_uri.startswith('http://'):
                result.warnings.append("MLflow tracking URI using HTTP - consider HTTPS for production")
    
    def _validate_dependencies(self, config: Dict[str, Any], result: ValidationResult):
        """Validate configuration dependencies."""
        
        # Check if required packages are available for configured features
        data_source_type = config.get('data_source', {}).get('type')
        
        if data_source_type == 'postgresql':
            try:
                import psycopg2
            except ImportError:
                result.warnings.append("PostgreSQL configured but psycopg2 not installed")
        
        elif data_source_type == 'mysql':
            try:
                import pymysql
            except ImportError:
                result.warnings.append("MySQL configured but pymysql not installed")
        
        elif data_source_type == 'snowflake':
            try:
                import snowflake.connector
            except ImportError:
                result.warnings.append("Snowflake configured but snowflake-connector-python not installed")
        
        # Check explainability dependencies
        explainability = config.get('explainability', {})
        if explainability.get('enabled', False):
            methods = explainability.get('interpretability_methods', [])
            
            if 'shap' in methods:
                try:
                    import shap
                except ImportError:
                    result.warnings.append("SHAP method configured but shap not installed")
            
            if 'lime' in methods:
                try:
                    import lime
                except ImportError:
                    result.warnings.append("LIME method configured but lime not installed")
    
    def _validate_performance(self, config: Dict[str, Any], result: ValidationResult):
        """Validate performance-related configuration."""
        
        # Check resource configuration
        resources = config.get('resources', {})
        if resources:
            compute = resources.get('compute', {})
            n_jobs = compute.get('n_jobs', 1)
            
            if isinstance(n_jobs, str) and n_jobs != '-1':
                result.warnings.append("n_jobs should be integer or -1 for all cores")
            elif isinstance(n_jobs, int) and n_jobs > 16:
                result.warnings.append("n_jobs > 16 may cause resource contention")
        
        # Check batch sizes
        output_config = config.get('output', {})
        predictions_config = output_config.get('predictions', {})
        batch_size = predictions_config.get('batch_size', 10000)
        
        if batch_size > 100000:
            result.warnings.append("Large batch size may cause memory issues")
        elif batch_size < 100:
            result.warnings.append("Very small batch size may impact performance")
    
    def _validate_environment(self, config: Dict[str, Any], result: ValidationResult):
        """Validate environment-specific configuration."""
        
        pipeline_config = config.get('pipeline', {})
        environment = pipeline_config.get('environment', 'dev')
        
        # Production-specific validations
        if environment == 'prod':
            # Check if monitoring is enabled
            monitoring = config.get('monitoring', {})
            if not monitoring.get('enabled', False):
                result.warnings.append("Monitoring should be enabled in production environment")
            
            # Check if logging level is appropriate
            log_level = pipeline_config.get('log_level', 'INFO')
            if log_level == 'DEBUG':
                result.warnings.append("DEBUG logging not recommended for production")
            
            # Check if versioning is enabled
            output_config = config.get('output', {})
            model_artifacts = output_config.get('model_artifacts', {})
            if not model_artifacts.get('versioning', False):
                result.warnings.append("Model versioning should be enabled in production")
        
        # Development-specific suggestions
        elif environment == 'dev':
            result.suggestions.append("Consider enabling verbose logging for development")
            result.suggestions.append("Use smaller datasets for faster iteration in development")

def validate_config_file(config_path: str, strict: bool = False) -> ValidationResult:
    """Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        strict: Enable strict validation mode
        
    Returns:
        ValidationResult
    """
    validator = ConfigValidator(strict_mode=strict)
    return validator.validate_config(config_path)