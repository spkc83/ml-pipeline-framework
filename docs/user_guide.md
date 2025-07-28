# ML Pipeline Framework User Guide

A comprehensive guide to installing, configuring, and using the ML Pipeline Framework for production machine learning workflows.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running Pipelines](#running-pipelines)
4. [Interpreting Outputs](#interpreting-outputs)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended for large datasets
- Compatible operating systems: Linux, macOS, Windows

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/ml-pipeline-framework.git
cd ml-pipeline-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Install via pip (when available)

```bash
pip install ml-pipeline-framework
```

### Verify Installation

```bash
# Check installation
python -c "import ml_pipeline_framework; print('Installation successful!')"

# Run help command
python run_pipeline.py --help

# Run configuration validation
make validate-config
```

### Optional Dependencies

Install additional dependencies based on your use case:

```bash
# For Spark ML support
pip install pyspark

# For H2O.ai support
pip install h2o

# For Snowflake connectivity
pip install snowflake-connector-python

# For advanced visualizations
pip install plotly seaborn

# For GPU acceleration (if available)
pip install cudf cuml  # RAPIDS
```

## Configuration

### Configuration File Structure

The framework uses YAML configuration files to define all pipeline parameters. Here's the complete structure:

```yaml
# configs/pipeline_config.yaml

# Data source configuration
data_source:
  type: "postgres"  # postgres, snowflake, redshift, hive, mysql
  connection_string: "postgresql://user:pass@host:5432/database"
  query_timeout: 300
  pool_size: 10
  ssl_mode: "require"

# Data query and target definition
data:
  query: |
    SELECT 
      customer_id,
      age, income, purchase_history,
      credit_score, account_balance,
      churn_flag as target
    FROM customer_analytics 
    WHERE date >= '2023-01-01'
  target_column: "target"
  id_column: "customer_id"
  
# Model configuration
model:
  type: "sklearn"  # sklearn, sparkml, h2o, statsmodels
  algorithm: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    random_state: 42
  
# Preprocessing pipeline
preprocessing:
  # Feature elimination
  feature_elimination:
    enabled: true
    method: "backward"
    cv_folds: 5
    min_features: 10
    tolerance: 0.001
    scoring: "roc_auc"
  
  # Handle imbalanced data
  imbalance_handling:
    enabled: true
    method: "smote"  # smote, adasyn, random_oversample, random_undersample
    sampling_strategy: "auto"
    
  # Data validation
  validation:
    enabled: true
    expectation_suite: "customer_data_expectations"
    fail_on_error: false
    
  # Feature scaling
  scaling:
    method: "standard"  # standard, minmax, robust, quantile
    feature_range: [0, 1]

# Model evaluation
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  cross_validation:
    enabled: true
    cv_folds: 5
    stratified: true
  test_size: 0.2
  validation_size: 0.2
  
# Model explainability
explainability:
  enabled: true
  methods: ["shap", "permutation", "pdp"]
  sample_size: 1000
  top_features: 20
  
# Hyperparameter tuning
tuning:
  enabled: false
  method: "optuna"  # optuna, gridsearch, randomsearch
  n_trials: 100
  optimization_direction: "maximize"
  scoring: "roc_auc"
  
# Experiment tracking
tracking:
  mlflow:
    tracking_uri: "http://localhost:5000"
    experiment_name: "customer_churn_prediction"
    artifact_location: "s3://my-bucket/mlflow-artifacts"
    
# Artifact storage
artifacts:
  storage_type: "s3"  # local, s3, azure, gcp
  base_path: "s3://my-ml-artifacts/customer-churn/"
  versioning: true
  
# Logging configuration
logging:
  level: "INFO"
  enable_file: true
  enable_json: true
  enable_mlflow: true
  log_dir: "./logs"
  
# Engine selection
engine_selection:
  data_processing:
    enabled: true
    memory_threshold_gb: 4.0
    large_data_threshold: 1000000
  enable_benchmarking: false
```

### Environment Variables

Use environment variables for sensitive information:

```bash
# Database credentials
export DB_PASSWORD="your_secure_password"
export SNOWFLAKE_PASSWORD="your_snowflake_password"

# Cloud credentials
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"

# MLflow tracking
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
```

Reference environment variables in config files:

```yaml
data_source:
  connection_string: "postgresql://user:${DB_PASSWORD}@host:5432/db"
```

### Configuration Validation

Validate your configuration before running:

```bash
# Validate configuration file
python run_pipeline.py validate --config configs/pipeline_config.yaml

# Check component compatibility
python run_pipeline.py check-deps --config configs/pipeline_config.yaml

# Generate configuration template
python run_pipeline.py init --config-type basic --output configs/
```

## Running Pipelines

### Command Line Interface

The framework provides a comprehensive CLI for pipeline operations:

#### Basic Training Pipeline

```bash
# Run complete training pipeline
python run_pipeline.py run \
  --config configs/pipeline_config.yaml \
  --mode train \
  --experiment-name "customer_churn_v1" \
  --run-name "baseline_model"
```

#### Prediction Pipeline

```bash
# Generate predictions on new data
python run_pipeline.py run \
  --config configs/pipeline_config.yaml \
  --mode predict \
  --input-data "data/new_customers.csv" \
  --output-path "predictions/churn_predictions.csv" \
  --model-path "models/customer_churn_v1"
```

#### Evaluation Pipeline

```bash
# Evaluate existing model
python run_pipeline.py run \
  --config configs/pipeline_config.yaml \
  --mode evaluate \
  --model-path "models/customer_churn_v1" \
  --test-data "data/holdout_test.csv"
```

#### Available CLI Options

```bash
# Main command options
python run_pipeline.py run [OPTIONS]

Options:
  --config PATH                   Configuration file path [required]
  --mode [train|predict|evaluate] Pipeline execution mode [required]
  --experiment-name TEXT          MLflow experiment name
  --run-name TEXT                 MLflow run name
  --input-data PATH              Input data path (for prediction mode)
  --output-path PATH             Output path for results
  --model-path PATH              Path to saved model
  --test-data PATH               Test data path (for evaluation mode)
  --dry-run                      Validate configuration without execution
  --verbose                      Enable verbose output
  --parallel                     Enable parallel processing
  --help                         Show help message
```

### Python API

Use the framework programmatically in Python:

#### Basic Usage

```python
from ml_pipeline_framework import PipelineOrchestrator
from ml_pipeline_framework.utils import ConfigParser

# Load configuration
config = ConfigParser.from_yaml('configs/pipeline_config.yaml')

# Initialize orchestrator
orchestrator = PipelineOrchestrator(
    config=config,
    experiment_name="customer_churn_prediction",
    run_name="experiment_1"
)

# Run training pipeline
results = orchestrator.run_training()

print(f"Training completed with ROC-AUC: {results['model_metrics']['roc_auc']:.3f}")
print(f"Model saved to: {results['model_path']}")
```

#### Advanced Usage

```python
# Custom data loading
data = pd.read_csv('custom_data.csv')

# Run with custom parameters
results = orchestrator.run_training(
    data_query=None,  # Use data parameter instead
    data=data,
    target_column='churn_flag',
    test_size=0.15,
    validation_size=0.15,
    random_state=123
)

# Generate explanations
explanations = orchestrator.run_explainability(
    data_sample=data.sample(200),
    explanation_types=['shap', 'pdp']
)

# Make predictions
new_data = pd.read_csv('new_customers.csv')
predictions = orchestrator.run_prediction(
    data=new_data,
    return_probabilities=True
)
```

### Makefile Commands

The framework includes a comprehensive Makefile for common operations:

```bash
# Setup and installation
make install           # Install the package
make install-dev       # Install with development dependencies

# Testing
make test             # Run all tests
make test-unit        # Run unit tests only
make test-integration # Run integration tests
make test-coverage    # Run tests with coverage

# Code quality
make lint             # Run code linting
make format           # Format code
make type-check       # Run type checking
make quality          # Run all quality checks
make quality-fix      # Auto-fix quality issues

# Pipeline operations
make run-example      # Run example pipeline
make validate-config  # Validate configuration
```

## Interpreting Outputs

### Training Results

After training, the pipeline generates comprehensive results:

#### Console Output

```
🚀 Starting ML Pipeline Training
📊 Data loaded: 50,000 rows, 25 features
🔍 Feature elimination: 25 → 15 features (10 eliminated)
⚖️  Handling imbalanced data: SMOTE applied (2:1 → 1:1 ratio)
🧠 Training Random Forest model...
📈 Cross-validation ROC-AUC: 0.847 ± 0.023
🎯 Test set performance:
   • Accuracy: 0.824
   • Precision: 0.789
   • Recall: 0.856
   • F1-Score: 0.821
   • ROC-AUC: 0.901
💾 Model saved to: artifacts/models/customer_churn_v1
📋 Training completed in 127.3 seconds
```

#### MLflow Tracking

All experiments are automatically tracked in MLflow:

- **Metrics**: Performance metrics for each run
- **Parameters**: Model hyperparameters and configuration
- **Artifacts**: Saved models, plots, and data
- **Tags**: Environment and experiment metadata

Access MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

#### Generated Artifacts

Training creates several output files:

```
artifacts/
├── models/
│   ├── customer_churn_v1/
│   │   ├── model.pkl           # Trained model
│   │   ├── preprocessor.pkl    # Data preprocessor
│   │   ├── config.yaml         # Training configuration
│   │   └── metadata.json       # Model metadata
├── reports/
│   ├── evaluation_report.html  # Comprehensive evaluation
│   ├── feature_importance.png  # Feature importance plot
│   ├── confusion_matrix.png    # Confusion matrix
│   └── roc_curve.png          # ROC curve
├── explanations/
│   ├── shap_summary.png       # SHAP summary plot
│   ├── shap_waterfall.png     # Individual explanations
│   └── partial_dependence/    # PDP plots
└── logs/
    ├── pipeline.log           # Detailed logs
    └── pipeline.json          # Structured logs
```

### Feature Elimination Results

When feature elimination is enabled, detailed results are saved:

#### Excel Report

The `feature_elimination_results.xlsx` file contains:

- **Summary**: Elimination steps and performance
- **Best_Features**: Final selected features
- **Feature_Rankings**: All features ranked by importance
- **Iteration_X_Importance**: Feature importance at each step
- **CV_Details**: Cross-validation results
- **Configuration**: Elimination parameters

#### Visualization

Three plots are generated:

1. **Elimination Curve**: Performance vs. number of features
2. **Importance Evolution**: How feature importance changes
3. **Elimination Heatmap**: Timeline of feature elimination

### Prediction Results

Prediction outputs include:

#### CSV Format

```csv
customer_id,prediction,churn_probability,risk_level
C001,0,0.123,low
C002,1,0.847,high
C003,0,0.234,medium
```

#### JSON Format

```json
{
  "predictions": [
    {
      "customer_id": "C001",
      "prediction": 0,
      "probability": 0.123,
      "risk_level": "low",
      "explanation": {
        "top_features": [
          {"feature": "account_balance", "contribution": -0.15},
          {"feature": "tenure", "contribution": -0.08}
        ]
      }
    }
  ],
  "metadata": {
    "model_version": "v1.0",
    "prediction_date": "2024-01-15T10:30:00Z",
    "total_predictions": 1000
  }
}
```

### Evaluation Reports

Comprehensive HTML reports include:

- **Model Performance**: All metrics with confidence intervals
- **Feature Analysis**: Importance and distribution plots
- **Prediction Distribution**: Histogram of predictions
- **Calibration Plots**: Prediction calibration analysis
- **Error Analysis**: Misclassification patterns
- **Stability Analysis**: Performance across data segments

## Troubleshooting

### Common Issues

#### 1. Memory Errors

**Problem**: `MemoryError` during training with large datasets

**Solutions:**

```bash
# Enable data engine selection
# In config.yaml:
engine_selection:
  data_processing:
    enabled: true
    memory_threshold_gb: 2.0

# Use batch processing
# In config.yaml:
model:
  batch_size: 1000

# Reduce feature elimination sample size
preprocessing:
  feature_elimination:
    sample_size: 10000
```

#### 2. Connection Errors

**Problem**: Database connection failures

**Solutions:**

```bash
# Test connection
python -c "
from ml_pipeline_framework.data_access import ConnectorFactory
config = {'type': 'postgres', 'connection_string': 'postgresql://...'}
connector = ConnectorFactory.create_connector(config)
connector.validate_connection()
"

# Check firewall and network connectivity
telnet your-db-host 5432

# Verify credentials
psql -h your-db-host -U your-user -d your-database -c "SELECT 1"
```

#### 3. Missing Dependencies

**Problem**: `ImportError` for optional dependencies

**Solutions:**

```bash
# Install missing dependencies
pip install pyspark  # For Spark ML
pip install h2o      # For H2O models
pip install shap     # For SHAP explanations

# Check installed packages
pip list | grep -E "(pyspark|h2o|shap)"

# Install all optional dependencies
pip install -r requirements-dev.txt
```

#### 4. Configuration Errors

**Problem**: Invalid configuration parameters

**Solutions:**

```bash
# Validate configuration
python run_pipeline.py validate --config configs/pipeline_config.yaml

# Check configuration schema
python -c "
from ml_pipeline_framework.utils import ConfigParser
ConfigParser.validate_schema('configs/pipeline_config.yaml')
"

# Generate template configuration
python run_pipeline.py init --config-type full --output configs/
```

#### 5. MLflow Tracking Issues

**Problem**: MLflow tracking not working

**Solutions:**

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Check MLflow connection
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
print(mlflow.get_tracking_uri())
"

# Check MLflow logs
tail -f logs/mlflow.log
```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Run with debug logging
python run_pipeline.py run \
  --config configs/pipeline_config.yaml \
  --mode train \
  --verbose \
  --log-level DEBUG

# Check detailed logs
tail -f logs/pipeline.log
```

### Performance Issues

#### Slow Training

**Possible causes and solutions:**

1. **Large dataset**: Enable parallel processing or use sampling
2. **Complex model**: Reduce hyperparameters or use simpler model
3. **Feature elimination**: Reduce CV folds or sample size
4. **Data loading**: Optimize database queries or use faster connector

```yaml
# Performance optimizations
model:
  n_jobs: -1  # Use all CPU cores

preprocessing:
  feature_elimination:
    cv_folds: 3  # Reduce from 5
    sample_size: 5000  # Limit sample size

engine_selection:
  enable_benchmarking: false  # Disable benchmarking
```

#### High Memory Usage

**Solutions:**

```yaml
# Memory optimizations
engine_selection:
  data_processing:
    enabled: true
    memory_threshold_gb: 1.0

model:
  batch_size: 500  # Process in smaller batches

preprocessing:
  chunked_processing: true
  chunk_size: 10000
```

### Getting Help

#### Log Analysis

The framework generates comprehensive logs:

```bash
# View recent errors
grep -i error logs/pipeline.log | tail -20

# View performance metrics
grep -i "execution_time" logs/pipeline.json

# View configuration issues
grep -i "config" logs/pipeline.log
```

#### Health Checks

Run built-in health checks:

```bash
# System health check
python run_pipeline.py health-check

# Component availability check
python run_pipeline.py check-deps

# Configuration validation
python run_pipeline.py validate --config configs/pipeline_config.yaml
```

#### Support Resources

- **Documentation**: [API Reference](api/index.md)
- **Examples**: Check `notebooks/` directory
- **GitHub Issues**: Report bugs and feature requests
- **Community**: Join our discussion forum

## Advanced Usage

### Custom Components

#### Custom Data Connector

```python
from ml_pipeline_framework.data_access import BaseConnector

class CustomConnector(BaseConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
    
    def connect(self):
        # Custom connection logic
        pass
    
    def query(self, sql, parameters=None):
        # Custom query logic
        return pd.DataFrame()

# Register custom connector
from ml_pipeline_framework.data_access import ConnectorFactory
ConnectorFactory.register_connector('custom', CustomConnector)
```

#### Custom Model

```python
from ml_pipeline_framework.models import BaseModel

class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X, y):
        # Custom training logic
        pass
    
    def predict(self, X):
        # Custom prediction logic
        return predictions

# Register custom model
from ml_pipeline_framework.models import ModelFactory
ModelFactory.register_model('custom', CustomModel)
```

### Batch Processing

For large-scale processing:

```python
from ml_pipeline_framework import PipelineOrchestrator
import pandas as pd

# Process data in batches
def process_large_dataset(orchestrator, data_query, batch_size=10000):
    total_rows = orchestrator.data_connector.query(
        f"SELECT COUNT(*) as count FROM ({data_query}) t"
    ).iloc[0]['count']
    
    predictions = []
    for offset in range(0, total_rows, batch_size):
        batch_query = f"""
        {data_query} 
        LIMIT {batch_size} OFFSET {offset}
        """
        
        batch_data = orchestrator.data_connector.query(batch_query)
        batch_predictions = orchestrator.run_prediction(batch_data)
        predictions.extend(batch_predictions)
    
    return predictions
```

### Model Monitoring

Set up continuous monitoring:

```python
from ml_pipeline_framework.evaluation import StabilityAnalyzer

# Monitor model performance over time
stability_analyzer = StabilityAnalyzer(
    reference_data=training_data,
    monitoring_window='7d'
)

# Check for data drift
drift_report = stability_analyzer.detect_drift(new_data)

if drift_report['drift_detected']:
    print("Data drift detected - consider retraining")
    # Trigger retraining pipeline
```

### A/B Testing

Compare multiple models:

```python
from ml_pipeline_framework.evaluation import ModelComparison

# Compare multiple models
models = {
    'baseline': 'models/baseline_v1',
    'experiment': 'models/experiment_v1'
}

comparison = ModelComparison(models)
results = comparison.compare_models(test_data)

# Statistical significance testing
significance = comparison.statistical_test(
    metric='roc_auc',
    alpha=0.05
)
```

## Best Practices

### Configuration Management

1. **Version Control**: Store configurations in version control
2. **Environment-Specific**: Use separate configs for dev/staging/prod
3. **Secrets Management**: Use environment variables for sensitive data
4. **Validation**: Always validate configurations before deployment

### Data Management

1. **Data Quality**: Implement comprehensive data validation
2. **Feature Engineering**: Document feature transformations
3. **Data Lineage**: Track data sources and transformations
4. **Backup**: Regularly backup training data and models

### Model Development

1. **Experimentation**: Use MLflow for experiment tracking
2. **Reproducibility**: Set random seeds and document environment
3. **Testing**: Implement unit tests for custom components
4. **Documentation**: Document model assumptions and limitations

### Production Deployment

1. **Monitoring**: Implement comprehensive monitoring
2. **Rollback Strategy**: Plan for model rollbacks
3. **Performance**: Monitor prediction latency and throughput
4. **Security**: Implement proper authentication and authorization

### Code Quality

1. **Testing**: Maintain high test coverage
2. **Linting**: Use automated code quality checks
3. **Documentation**: Keep documentation up to date
4. **Code Review**: Implement peer review process

## Examples

### Customer Churn Prediction

Complete example for predicting customer churn:

```yaml
# configs/churn_config.yaml
data_source:
  type: "postgres"
  connection_string: "postgresql://user:pass@host:5432/crm"

data:
  query: |
    SELECT 
      customer_id,
      tenure_months,
      monthly_charges,
      total_charges,
      contract_type,
      payment_method,
      internet_service,
      online_security,
      tech_support,
      churn_flag
    FROM customer_data 
    WHERE data_date >= '2023-01-01'
  target_column: "churn_flag"

model:
  type: "sklearn"
  algorithm: "gradient_boosting"
  parameters:
    n_estimators: 200
    learning_rate: 0.1
    max_depth: 6

preprocessing:
  feature_elimination:
    enabled: true
    scoring: "roc_auc"
  imbalance_handling:
    enabled: true
    method: "smote"

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
```

Run the pipeline:

```bash
python run_pipeline.py run \
  --config configs/churn_config.yaml \
  --mode train \
  --experiment-name "customer_churn" \
  --run-name "gradient_boosting_v1"
```

### Credit Risk Assessment

Example for credit risk modeling:

```yaml
# configs/credit_risk_config.yaml
data_source:
  type: "snowflake"
  account: "mycompany.snowflakecomputing.com"
  warehouse: "RISK_WH"
  database: "CREDIT_DATA"

model:
  type: "sklearn"
  algorithm: "logistic_regression"
  parameters:
    C: 1.0
    penalty: "l2"
    class_weight: "balanced"

preprocessing:
  feature_elimination:
    enabled: true
    min_features: 20
    cv_folds: 10  # More rigorous for financial models
    
explainability:
  enabled: true
  methods: ["shap", "permutation"]
  generate_compliance_report: true
```

### Time Series Forecasting

Example for demand forecasting:

```yaml
# configs/demand_forecast_config.yaml
data:
  query: |
    SELECT 
      date,
      product_id,
      sales_quantity,
      price,
      promotion_flag,
      day_of_week,
      month,
      seasonal_index
    FROM sales_data
  target_column: "sales_quantity"

model:
  type: "sklearn"
  algorithm: "random_forest"
  parameters:
    n_estimators: 500
    max_features: "sqrt"

preprocessing:
  time_series_features:
    enabled: true
    lag_features: [1, 7, 30]
    rolling_features: [7, 30]
```

This user guide provides comprehensive coverage of the ML Pipeline Framework, from basic installation to advanced usage scenarios. Users can follow this guide to successfully implement production ML workflows with the framework.