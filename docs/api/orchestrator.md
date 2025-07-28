# Pipeline Orchestrator API

The `PipelineOrchestrator` is the main coordination class that manages the entire ML pipeline execution flow.

## Classes

### PipelineOrchestrator

```python
class PipelineOrchestrator
```

Main orchestrator class for coordinating ML pipeline execution across multiple components.

The orchestrator manages data loading, preprocessing, model training, evaluation, and artifact storage. It provides a unified interface for running complete ML workflows with support for different execution modes and environments.

**Attributes:**
- `config` (Dict[str, Any]): Pipeline configuration dictionary
- `data_connector` (BaseConnector): Data source connector instance
- `model` (BaseModel): Machine learning model instance
- `preprocessor` (DataPreprocessor): Data preprocessing pipeline
- `evaluator` (ModelEvaluator): Model evaluation component
- `explainer` (SHAPExplainer): Model explainability component
- `tracker` (MLflowTracker): Experiment tracking component
- `artifact_manager` (ArtifactManager): Artifact storage manager

**Example:**
```python
from ml_pipeline_framework import PipelineOrchestrator
from ml_pipeline_framework.utils import ConfigParser

# Load configuration
config = ConfigParser.from_yaml('configs/pipeline_config.yaml')

# Initialize orchestrator
orchestrator = PipelineOrchestrator(config)

# Run training pipeline
results = orchestrator.run_training()
print(f"Training completed with accuracy: {results['accuracy']:.3f}")

# Generate predictions
predictions = orchestrator.run_prediction(test_data)
```

#### Methods

##### `__init__(config, experiment_name=None, run_name=None)`

Initialize the pipeline orchestrator.

**Args:**
- `config` (Dict[str, Any]): Pipeline configuration dictionary containing all component settings
- `experiment_name` (str, optional): MLflow experiment name for tracking. Defaults to None
- `run_name` (str, optional): MLflow run name for this execution. Defaults to None

**Raises:**
- `ValueError`: If required configuration sections are missing
- `ImportError`: If required dependencies for configured components are not available

**Example:**
```python
config = {
    'data_source': {'type': 'postgres', 'connection_string': 'postgresql://...'},
    'model': {'type': 'sklearn', 'algorithm': 'random_forest'},
    'preprocessing': {'feature_elimination': {'enabled': True}}
}

orchestrator = PipelineOrchestrator(
    config=config,
    experiment_name="customer_churn_prediction",
    run_name="v1.0_baseline"
)
```

##### `from_config_file(config_path, experiment_name=None, run_name=None)`

Create orchestrator instance from configuration file.

**Args:**
- `config_path` (Union[str, Path]): Path to YAML/JSON configuration file
- `experiment_name` (str, optional): MLflow experiment name. Defaults to None
- `run_name` (str, optional): MLflow run name. Defaults to None

**Returns:**
- `PipelineOrchestrator`: Configured orchestrator instance

**Raises:**
- `FileNotFoundError`: If configuration file doesn't exist
- `ValueError`: If configuration file format is invalid

**Example:**
```python
orchestrator = PipelineOrchestrator.from_config_file(
    'configs/production_config.yaml',
    experiment_name="production_model",
    run_name="weekly_retrain"
)
```

##### `run_training(data_query=None, target_column=None, test_size=0.2, validation_size=0.2, random_state=42)`

Execute complete training pipeline.

**Args:**
- `data_query` (str, optional): SQL query or data identifier. Uses config if None
- `target_column` (str, optional): Target variable column name. Uses config if None
- `test_size` (float): Proportion of data for testing. Defaults to 0.2
- `validation_size` (float): Proportion of training data for validation. Defaults to 0.2
- `random_state` (int): Random seed for reproducibility. Defaults to 42

**Returns:**
- `Dict[str, Any]`: Training results containing:
  - `model_metrics` (Dict): Performance metrics on test set
  - `training_time` (float): Total training time in seconds
  - `feature_count` (int): Number of features used
  - `model_path` (str): Path to saved model artifact
  - `experiment_info` (Dict): MLflow experiment tracking information

**Raises:**
- `RuntimeError`: If training pipeline fails at any stage
- `ValueError`: If data validation fails

**Example:**
```python
results = orchestrator.run_training(
    data_query="SELECT * FROM customer_data WHERE date >= '2023-01-01'",
    target_column="churn_flag",
    test_size=0.15,
    random_state=123
)

print(f"Model ROC-AUC: {results['model_metrics']['roc_auc']:.3f}")
print(f"Training completed in {results['training_time']:.1f} seconds")
```

##### `run_prediction(data, return_probabilities=False, batch_size=None)`

Generate predictions using trained model.

**Args:**
- `data` (Union[pd.DataFrame, str, Path]): Input data for prediction or path to data
- `return_probabilities` (bool): Return prediction probabilities. Defaults to False
- `batch_size` (int, optional): Batch size for large dataset prediction. Defaults to None

**Returns:**
- `Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]`: Predictions array, or tuple of (predictions, probabilities) if return_probabilities=True

**Raises:**
- `RuntimeError`: If model hasn't been trained or loaded
- `ValueError`: If input data format is invalid

**Example:**
```python
# Predict on new data
predictions = orchestrator.run_prediction(new_customer_data)

# Get probabilities for classification
pred_classes, pred_probs = orchestrator.run_prediction(
    data=new_customer_data,
    return_probabilities=True
)

print(f"Predicted {len(predictions)} samples")
print(f"Average churn probability: {pred_probs[:, 1].mean():.3f}")
```

##### `run_evaluation(test_data=None, metrics=None, generate_report=True)`

Evaluate model performance with comprehensive metrics.

**Args:**
- `test_data` (pd.DataFrame, optional): Test dataset. Uses internal test set if None
- `metrics` (List[str], optional): Specific metrics to compute. Uses default set if None
- `generate_report` (bool): Generate detailed evaluation report. Defaults to True

**Returns:**
- `Dict[str, Any]`: Evaluation results containing:
  - `metrics` (Dict): Performance metric values
  - `confusion_matrix` (np.ndarray): Confusion matrix for classification
  - `feature_importance` (Dict): Feature importance scores
  - `report_path` (str): Path to detailed evaluation report

**Raises:**
- `RuntimeError`: If model hasn't been trained
- `ValueError`: If test data format is incompatible

**Example:**
```python
eval_results = orchestrator.run_evaluation(
    test_data=holdout_dataset,
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    generate_report=True
)

print(f"Model Performance:")
for metric, value in eval_results['metrics'].items():
    print(f"  {metric}: {value:.3f}")
```

##### `run_explainability(data_sample=None, sample_size=100, explanation_types=None)`

Generate model explanations and interpretability analysis.

**Args:**
- `data_sample` (pd.DataFrame, optional): Data sample for explanation. Uses random sample if None
- `sample_size` (int): Number of samples for explanation analysis. Defaults to 100
- `explanation_types` (List[str], optional): Types of explanations to generate. Uses all if None

**Returns:**
- `Dict[str, Any]`: Explainability results containing:
  - `shap_values` (np.ndarray): SHAP values for feature importance
  - `global_importance` (Dict): Global feature importance ranking
  - `local_explanations` (List): Individual prediction explanations
  - `plots_path` (str): Directory containing explanation visualizations

**Raises:**
- `RuntimeError`: If model hasn't been trained
- `ImportError`: If SHAP dependencies are not available

**Example:**
```python
explanations = orchestrator.run_explainability(
    data_sample=representative_sample,
    sample_size=200,
    explanation_types=['shap', 'permutation', 'pdp']
)

print(f"Top 5 most important features:")
for feature, importance in list(explanations['global_importance'].items())[:5]:
    print(f"  {feature}: {importance:.3f}")
```

##### `save_pipeline(path, include_data=False)`

Save complete pipeline state to disk.

**Args:**
- `path` (Union[str, Path]): Directory path to save pipeline artifacts
- `include_data` (bool): Include processed data in saved pipeline. Defaults to False

**Returns:**
- `Dict[str, str]`: Paths to saved components

**Raises:**
- `RuntimeError`: If pipeline hasn't been trained
- `IOError`: If save location is not accessible

**Example:**
```python
saved_paths = orchestrator.save_pipeline(
    path='models/production_v1.0',
    include_data=False
)

print(f"Pipeline saved to: {saved_paths['base_path']}")
print(f"Model artifact: {saved_paths['model']}")
```

##### `load_pipeline(path)`

Load previously saved pipeline from disk.

**Args:**
- `path` (Union[str, Path]): Directory path containing saved pipeline

**Raises:**
- `FileNotFoundError`: If pipeline files are not found
- `ValueError`: If saved pipeline format is incompatible

**Example:**
```python
orchestrator.load_pipeline('models/production_v1.0')
print("Pipeline loaded successfully")

# Now ready for prediction
predictions = orchestrator.run_prediction(new_data)
```

##### `get_pipeline_status()`

Get current pipeline status and component information.

**Returns:**
- `Dict[str, Any]`: Status information containing:
  - `is_trained` (bool): Whether model has been trained
  - `components_initialized` (Dict): Status of each component
  - `last_training_time` (str): Timestamp of last training
  - `model_metrics` (Dict): Current model performance metrics
  - `configuration` (Dict): Current pipeline configuration

**Example:**
```python
status = orchestrator.get_pipeline_status()

print(f"Pipeline trained: {status['is_trained']}")
print(f"Components ready: {status['components_initialized']}")

if status['model_metrics']:
    print(f"Current accuracy: {status['model_metrics'].get('accuracy', 'N/A')}")
```

##### `validate_configuration()`

Validate pipeline configuration for completeness and compatibility.

**Returns:**
- `Dict[str, Any]`: Validation results containing:
  - `is_valid` (bool): Whether configuration is valid
  - `errors` (List[str]): Configuration errors found
  - `warnings` (List[str]): Configuration warnings
  - `missing_components` (List[str]): Required components not configured

**Raises:**
- `ValueError`: If critical configuration errors are found

**Example:**
```python
validation = orchestrator.validate_configuration()

if not validation['is_valid']:
    print("Configuration errors found:")
    for error in validation['errors']:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

##### `get_component_info(component_name=None)`

Get detailed information about pipeline components.

**Args:**
- `component_name` (str, optional): Specific component name. Returns all if None

**Returns:**
- `Dict[str, Any]`: Component information including configuration, status, and capabilities

**Example:**
```python
# Get all component info
all_info = orchestrator.get_component_info()

# Get specific component info
model_info = orchestrator.get_component_info('model')
print(f"Model type: {model_info['type']}")
print(f"Model parameters: {model_info['parameters']}")
```

## Configuration Schema

The orchestrator expects a configuration dictionary with the following structure:

```yaml
# Data source configuration
data_source:
  type: "postgres"  # postgres, snowflake, redshift, hive, mysql
  connection_string: "postgresql://user:pass@host:5432/db"
  query_timeout: 300
  
# Model configuration
model:
  type: "sklearn"  # sklearn, sparkml, h2o, statsmodels
  algorithm: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
# Preprocessing configuration
preprocessing:
  feature_elimination:
    enabled: true
    method: "backward"
    cv_folds: 5
    min_features: 10
  imbalance_handling:
    method: "smote"
    sampling_strategy: "auto"
  
# Evaluation configuration
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  cross_validation:
    enabled: true
    cv_folds: 5
  
# Tracking configuration
tracking:
  mlflow:
    tracking_uri: "http://localhost:5000"
    experiment_name: "default"
  
# Artifact storage
artifacts:
  storage_type: "local"  # local, s3, azure, gcp
  base_path: "./artifacts"
```

## Error Handling

The orchestrator implements comprehensive error handling with specific exception types:

- **Configuration Errors**: Raised for invalid or missing configuration
- **Component Errors**: Raised when individual components fail
- **Data Errors**: Raised for data validation or processing failures
- **Model Errors**: Raised for model training or prediction failures

All errors include detailed messages and suggestions for resolution.

## Performance Considerations

- **Memory Management**: Automatic memory optimization for large datasets
- **Parallel Processing**: Multi-core support for CPU-intensive operations
- **Lazy Loading**: Components initialized only when needed
- **Caching**: Intelligent caching of intermediate results
- **Batch Processing**: Efficient handling of large prediction requests

## Integration Points

The orchestrator integrates with:

- **MLflow**: Experiment tracking and model registry
- **Cloud Storage**: S3, Azure Blob Storage, Google Cloud Storage
- **Databases**: PostgreSQL, Snowflake, Redshift, Hive, MySQL
- **ML Frameworks**: scikit-learn, XGBoost, H2O, Spark ML, StatsModels
- **Monitoring**: Structured logging and metrics collection