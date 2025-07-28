import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

logger = logging.getLogger(__name__)


class EnvironmentType(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    HIVE = "hive"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"


class PipelineConfig(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    environment: EnvironmentType = EnvironmentType.DEV
    log_level: LogLevel = LogLevel.INFO


class ConnectionConfig(BaseModel):
    host: str
    port: int
    database: str
    username: str
    password: str
    db_schema: Optional[str] = "public"
    sslmode: Optional[str] = "prefer"


class ExtractionConfig(BaseModel):
    query: Optional[str] = None
    table: Optional[str] = None
    filters: Optional[List[str]] = None
    chunk_size: int = Field(default=10000, gt=0)
    cache_data: bool = True
    cache_location: str = "./artifacts/cache"
    
    @model_validator(mode='after')
    def validate_query_or_table(self):
        if not self.query and not self.table:
            raise ValueError("Either 'query' or 'table' must be specified")
        return self


class DataSourceConfig(BaseModel):
    type: DatabaseType
    connection: ConnectionConfig
    extraction: ExtractionConfig


class DataQualityCheck(BaseModel):
    type: str
    threshold: Optional[float] = None
    action: str = Field(default="warn", pattern="^(warn|error|drop|remove|cap)$")


class DataQualityConfig(BaseModel):
    enabled: bool = True
    checks: List[DataQualityCheck] = []


class CleaningConfig(BaseModel):
    handle_missing: Optional[Dict[str, Any]] = None
    handle_outliers: Optional[Dict[str, Any]] = None
    date_parsing: Optional[Dict[str, Any]] = None


class TransformationConfig(BaseModel):
    scaling: Optional[Dict[str, Any]] = None
    encoding: Optional[Dict[str, Any]] = None
    feature_selection: Optional[Dict[str, Any]] = None


class PreprocessingConfig(BaseModel):
    data_quality: Optional[DataQualityConfig] = None
    cleaning: Optional[CleaningConfig] = None
    transformation: Optional[TransformationConfig] = None


class DerivedFeature(BaseModel):
    name: str
    expression: str
    type: str = Field(default="numeric", pattern="^(numeric|categorical|boolean|datetime)$")


class TimeFeatureConfig(BaseModel):
    enabled: bool = True
    datetime_column: str
    features: List[str] = []


class AggregationFeature(BaseModel):
    groupby: List[str]
    features: List[Dict[str, str]]


class FeatureEngineeringConfig(BaseModel):
    enabled: bool = True
    derived_features: Optional[List[DerivedFeature]] = None
    time_features: Optional[TimeFeatureConfig] = None
    aggregations: Optional[List[AggregationFeature]] = None


class DataSplitConfig(BaseModel):
    method: str = Field(default="random", pattern="^(random|stratified|time_based)$")
    train_ratio: float = Field(default=0.7, gt=0, lt=1)
    validation_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)
    stratify_column: Optional[str] = None
    time_column: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_ratios_sum_to_one(self):
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")
        return self


class TargetConfig(BaseModel):
    column: str
    type: str = Field(pattern="^(classification|regression)$")
    classes: Optional[List[Union[int, str]]] = None
    transform: Optional[str] = None


class HyperparameterTuningConfig(BaseModel):
    enabled: bool = False
    method: str = Field(default="random_search", pattern="^(grid_search|random_search|bayesian)$")
    cv_folds: int = Field(default=5, gt=1)
    n_iter: int = Field(default=50, gt=0)
    param_grid: Dict[str, List[Any]] = {}


class ModelConfig(BaseModel):
    name: str
    type: str
    hyperparameters: Dict[str, Any] = {}
    hyperparameter_tuning: Optional[HyperparameterTuningConfig] = None


class CrossValidationConfig(BaseModel):
    enabled: bool = True
    method: str = Field(default="stratified_kfold", pattern="^(kfold|stratified_kfold|time_series_split)$")
    n_folds: int = Field(default=5, gt=1)
    shuffle: bool = True
    random_state: Optional[int] = None


class ModelSelectionConfig(BaseModel):
    metric: str
    higher_is_better: bool = True


class ModelTrainingConfig(BaseModel):
    data_split: DataSplitConfig
    target: TargetConfig
    models: List[ModelConfig]
    cross_validation: Optional[CrossValidationConfig] = None
    model_selection: ModelSelectionConfig


class EvaluationMetricsConfig(BaseModel):
    classification: Optional[List[str]] = None
    regression: Optional[List[str]] = None


class EvaluationReportsConfig(BaseModel):
    enabled: bool = True
    output_format: List[str] = ["json", "html"]
    include_plots: bool = True
    plots: Optional[List[str]] = None


class EvaluationComparisonConfig(BaseModel):
    enabled: bool = True
    baseline_model: str = "dummy_classifier"
    significance_test: str = Field(default="mcnemar", pattern="^(mcnemar|wilcoxon)$")


class EvaluationConfig(BaseModel):
    metrics: EvaluationMetricsConfig
    reports: Optional[EvaluationReportsConfig] = None
    comparison: Optional[EvaluationComparisonConfig] = None


class ShapConfig(BaseModel):
    enabled: bool = True
    explainer_type: str = Field(default="tree", pattern="^(tree|linear|kernel|deep)$")
    sample_size: int = Field(default=1000, gt=0)
    plots: Optional[List[str]] = None


class FeatureImportanceConfig(BaseModel):
    enabled: bool = True
    methods: List[str] = ["permutation", "model_specific"]
    plot_top_n: int = Field(default=20, gt=0)


class LimeConfig(BaseModel):
    enabled: bool = False
    mode: str = Field(default="tabular", pattern="^(tabular|text|image)$")
    sample_size: int = Field(default=100, gt=0)


class ExplainabilityConfig(BaseModel):
    enabled: bool = True
    shap: Optional[ShapConfig] = None
    feature_importance: Optional[FeatureImportanceConfig] = None
    lime: Optional[LimeConfig] = None


class ModelArtifactsConfig(BaseModel):
    save_location: str = "./artifacts/models"
    format: str = Field(default="joblib", pattern="^(joblib|pickle|mlflow)$")
    versioning: bool = True
    compression: bool = True


class PredictionsConfig(BaseModel):
    save_location: str = "./artifacts/predictions"
    format: str = Field(default="parquet", pattern="^(csv|parquet|json)$")
    include_probabilities: bool = True
    batch_size: int = Field(default=10000, gt=0)


class ReportsConfig(BaseModel):
    save_location: str = "./artifacts/reports"
    format: List[str] = ["html", "json"]
    include_data_profiling: bool = True


class MLflowConfig(BaseModel):
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "ml-pipeline-framework"
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_model: bool = True
    tags: Optional[Dict[str, str]] = None


class OutputConfig(BaseModel):
    model_artifacts: ModelArtifactsConfig
    predictions: PredictionsConfig
    reports: ReportsConfig
    mlflow: Optional[MLflowConfig] = None


class DataDriftConfig(BaseModel):
    enabled: bool = True
    reference_dataset: str = Field(default="training", pattern="^(training|validation|custom)$")
    drift_threshold: float = Field(default=0.05, gt=0, lt=1)
    statistical_tests: List[str] = ["ks_test", "chi2_test"]


class PerformanceMonitoringConfig(BaseModel):
    enabled: bool = True
    baseline_metrics: str = "training_metrics"
    alert_threshold: float = Field(default=0.1, gt=0)


class AlertChannel(BaseModel):
    type: str = Field(pattern="^(email|slack|webhook)$")
    recipients: Optional[List[str]] = None
    webhook_url: Optional[str] = None


class AlertsConfig(BaseModel):
    enabled: bool = False
    channels: Optional[List[AlertChannel]] = None


class MonitoringConfig(BaseModel):
    enabled: bool = True
    data_drift: Optional[DataDriftConfig] = None
    performance_monitoring: Optional[PerformanceMonitoringConfig] = None
    alerts: Optional[AlertsConfig] = None


class ComputeConfig(BaseModel):
    n_jobs: int = -1
    memory_limit: str = "8GB"


class SparkConfig(BaseModel):
    enabled: bool = False
    app_name: str = "ml-pipeline-framework"
    master: str = "local[*]"
    config: Optional[Dict[str, str]] = None


class ResourcesConfig(BaseModel):
    compute: ComputeConfig
    spark: Optional[SparkConfig] = None


class MLPipelineConfig(BaseModel):
    pipeline: PipelineConfig
    data_source: DataSourceConfig
    preprocessing: Optional[PreprocessingConfig] = None
    feature_engineering: Optional[FeatureEngineeringConfig] = None
    model_training: ModelTrainingConfig
    evaluation: EvaluationConfig
    explainability: Optional[ExplainabilityConfig] = None
    output: OutputConfig
    monitoring: Optional[MonitoringConfig] = None
    resources: Optional[ResourcesConfig] = None


class ConfigParser:
    """
    Configuration parser that handles YAML loading, environment variable substitution,
    and Pydantic validation.
    """
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self):
        self.config = None
    
    def load_config(self, config_path: Union[str, Path]) -> MLPipelineConfig:
        """
        Load and parse configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated MLPipelineConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If config validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        
        # Load YAML content
        with open(config_path, 'r') as file:
            raw_config = yaml.safe_load(file)
        
        # Substitute environment variables
        config_dict = self._substitute_env_vars(raw_config)
        
        # Validate with Pydantic
        self.config = MLPipelineConfig(**config_dict)
        
        logger.info("Configuration loaded and validated successfully")
        return self.config
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports syntax: ${VAR_NAME} or ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_string_env_vars(obj)
        else:
            return obj
    
    def _substitute_string_env_vars(self, text: str) -> str:
        """
        Substitute environment variables in a string.
        """
        def replace_env_var(match):
            var_spec = match.group(1)
            
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
                value = os.getenv(var_name, default_value)
            else:
                var_name = var_spec
                value = os.getenv(var_name)
                
                if value is None:
                    raise ValueError(f"Environment variable '{var_name}' is not set and no default value provided")
            
            # Try to convert to appropriate type
            return self._convert_value(value)
        
        return self.ENV_VAR_PATTERN.sub(replace_env_var, text)
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert string value to appropriate Python type.
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        # Additional custom validation logic can be added here
        return True
    
    def get_config(self) -> MLPipelineConfig:
        """
        Get the loaded configuration.
        
        Returns:
            MLPipelineConfig object
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        return self.config
    
    def save_config(self, output_path: Union[str, Path], exclude_none: bool = True) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration
            exclude_none: Whether to exclude None values from output
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and save as YAML
        config_dict = self.config.dict(exclude_none=exclude_none)
        
        with open(output_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")


def load_config(config_path: Union[str, Path]) -> MLPipelineConfig:
    """
    Convenience function to load and parse configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated MLPipelineConfig object
    """
    parser = ConfigParser()
    return parser.load_config(config_path)