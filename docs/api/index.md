# ML Pipeline Framework API Documentation

This documentation provides comprehensive reference for all classes, methods, and modules in the ML Pipeline Framework.

## Overview

The ML Pipeline Framework is a production-ready machine learning pipeline system designed for scalability, maintainability, and enterprise use. It provides a modular architecture with support for multiple ML frameworks, data sources, and deployment environments.

## Core Modules

### Pipeline Orchestration
- **[PipelineOrchestrator](orchestrator.md)** - Main coordination class for ML pipeline execution

### Data Access
- **[BaseConnector](data_access.md#baseconnector)** - Abstract base class for data connectors
- **[ConnectorFactory](data_access.md#connectorfactory)** - Factory for creating data connectors
- **[PostgresConnector](data_access.md#postgresconnector)** - PostgreSQL database connector
- **[SnowflakeConnector](data_access.md#snowflakeconnector)** - Snowflake data warehouse connector
- **[RedshiftConnector](data_access.md#redshiftconnector)** - Amazon Redshift connector
- **[HiveConnector](data_access.md#hiveconnector)** - Apache Hive connector
- **[MySQLConnector](data_access.md#mysqlconnector)** - MySQL database connector

### Models
- **[BaseModel](models.md#basemodel)** - Abstract base class for ML models
- **[ModelFactory](models.md#modelfactory)** - Factory for creating model instances
- **[SklearnModel](models.md#sklearnmodel)** - Scikit-learn model wrapper
- **[SparkMLModel](models.md#sparkmlmodel)** - Spark ML model wrapper
- **[H2OModel](models.md#h2omodel)** - H2O.ai model wrapper
- **[StatsModelsModel](models.md#statsmodelsmodel)** - StatsModels wrapper
- **[CostSensitiveModel](models.md#costsensitivemodel)** - Cost-sensitive learning wrapper
- **[HyperparameterTuner](models.md#hyperparametertuner)** - Automated hyperparameter optimization

### Preprocessing
- **[DataPreprocessor](preprocessing.md#datapreprocessor)** - Main data preprocessing pipeline
- **[FeatureEliminator](preprocessing.md#featureeliminator)** - Backward feature elimination
- **[ImbalanceHandler](preprocessing.md#imbalancehandler)** - Imbalanced dataset handling
- **[DataValidator](preprocessing.md#datavalidator)** - Data quality validation
- **[CustomTransformer](preprocessing.md#customtransformer)** - Custom feature transformations

### Evaluation
- **[ModelEvaluator](evaluation.md#modelevaluator)** - Model performance evaluation
- **[ModelComparison](evaluation.md#modelcomparison)** - Multi-model comparison
- **[StabilityAnalyzer](evaluation.md#stabilityanalyzer)** - Model stability analysis

### Explainability
- **[SHAPExplainer](explainability.md#shapexplainer)** - SHAP-based model explanations
- **[PDPAnalyzer](explainability.md#pdpanalyzer)** - Partial Dependence Plots
- **[ComplianceReporter](explainability.md#compliancereporter)** - Model compliance reporting

### Utilities
- **[ConfigParser](utils.md#configparser)** - Configuration management
- **[MLflowTracker](utils.md#mlflowtracker)** - MLflow experiment tracking
- **[ArtifactManager](utils.md#artifactmanager)** - Multi-cloud artifact storage
- **[DataEngineSelector](utils.md#dataengineselector)** - Intelligent data engine selection
- **[LoggingConfig](utils.md#loggingconfig)** - Structured logging configuration

## Quick Start

```python
from ml_pipeline_framework import PipelineOrchestrator
from ml_pipeline_framework.utils import ConfigParser

# Load configuration
config = ConfigParser.from_yaml('configs/pipeline_config.yaml')

# Initialize pipeline
orchestrator = PipelineOrchestrator(config)

# Run training pipeline
results = orchestrator.run_training()

# Generate predictions
predictions = orchestrator.run_prediction(test_data)
```

## Installation

```bash
# Install from source
git clone https://github.com/your-org/ml-pipeline-framework.git
cd ml-pipeline-framework
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Configuration

The framework uses YAML configuration files for all settings:

```yaml
# pipeline_config.yaml
data_source:
  type: "postgres"
  connection_string: "postgresql://user:pass@host:5432/db"

model:
  type: "sklearn"
  algorithm: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10

preprocessing:
  feature_elimination:
    enabled: true
    method: "backward"
    cv_folds: 5
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Access   │───▶│ Preprocessing   │───▶│     Models      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validation    │    │   Explainability│    │   Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │ Orchestrator    │
                    └─────────────────┘
```

## Features

### Enterprise Ready
- **Production Scalability**: Designed for large-scale data processing
- **Multi-Framework Support**: Works with scikit-learn, XGBoost, H2O, Spark ML
- **Cloud Integration**: Native support for AWS, GCP, Azure
- **Monitoring**: Built-in logging, metrics, and MLflow tracking

### Data Processing
- **Multiple Data Sources**: PostgreSQL, Snowflake, Redshift, Hive, MySQL
- **Intelligent Engine Selection**: Automatic choice between pandas, Polars, DuckDB
- **Data Validation**: Comprehensive quality checks with Great Expectations
- **Feature Engineering**: Advanced preprocessing and feature elimination

### Model Management
- **AutoML Capabilities**: Automated hyperparameter tuning
- **Cost-Sensitive Learning**: Handle imbalanced datasets effectively
- **Model Comparison**: Side-by-side performance analysis
- **Explainability**: SHAP explanations and compliance reporting

### Deployment
- **Configuration-Driven**: Fully configurable through YAML files
- **CLI Interface**: Command-line tools for operations
- **Testing**: Comprehensive unit and integration tests
- **Quality Assurance**: Automated code quality checks

## Examples

### Basic Training Pipeline

```python
from ml_pipeline_framework import PipelineOrchestrator

# Initialize with configuration
orchestrator = PipelineOrchestrator.from_config_file('config.yaml')

# Run complete training pipeline
results = orchestrator.run_training(
    data_query="SELECT * FROM training_data",
    target_column="target",
    test_size=0.2
)

print(f"Model accuracy: {results['accuracy']:.3f}")
```

### Feature Elimination

```python
from ml_pipeline_framework.preprocessing import FeatureEliminator
from sklearn.ensemble import RandomForestClassifier

# Initialize feature eliminator
eliminator = FeatureEliminator(
    estimator=RandomForestClassifier(),
    scoring='accuracy',
    cv=5,
    min_features=10
)

# Fit and transform data
X_selected = eliminator.fit_transform(X_train, y_train)

# Export detailed results
eliminator.export_to_excel('feature_elimination_results.xlsx')
eliminator.plot_elimination_curve(save_path='elimination_curve.png')
```

### Engine Selection

```python
from ml_pipeline_framework.utils import DataEngineSelector
from ml_pipeline_framework.utils.engine_selector import OperationType

# Initialize selector
selector = DataEngineSelector()

# Get recommendation for large dataset aggregation
recommendation = selector.select_engine(
    data='large_dataset.parquet',
    operation=OperationType.GROUPBY
)

print(f"Recommended engine: {recommendation.engine.value}")
print(f"Confidence: {recommendation.confidence:.2f}")
```

### Structured Logging

```python
from ml_pipeline_framework.utils import LoggingConfig, ComponentType

# Initialize logging
logging_config = LoggingConfig(
    log_level='INFO',
    enable_mlflow=True,
    enable_json=True
)

# Get component logger
logger = logging_config.get_logger(ComponentType.MODEL)

# Log with structured data
logger.log_experiment(
    logging.INFO,
    "Model training completed",
    experiment_id="exp_123",
    accuracy=0.95,
    features_used=50
)
```

## Support

- **Documentation**: [User Guide](../user_guide.md)
- **Examples**: [notebooks/](../../notebooks/)
- **Issues**: [GitHub Issues](https://github.com/your-org/ml-pipeline-framework/issues)
- **Contributing**: [CONTRIBUTING.md](../../CONTRIBUTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.