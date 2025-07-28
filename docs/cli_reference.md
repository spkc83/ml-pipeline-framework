# CLI Reference Guide

Complete reference for the ML Pipeline Framework command-line interface.

## 📋 Table of Contents

- [Global Options](#global-options)
- [Commands Overview](#commands-overview)
- [Command Details](#command-details)
  - [train](#train)
  - [predict](#predict)
  - [explain](#explain)
  - [deploy](#deploy)
  - [monitor](#monitor)
  - [init](#init)
  - [validate](#validate)
  - [version](#version)
- [Configuration](#configuration)
- [Examples](#examples)
- [Environment Variables](#environment-variables)

## 🌐 Global Options

Options available for all commands:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to configuration file | `configs/pipeline_config.yaml` |
| `--verbose` | `-v` | Enable verbose output | False |
| `--quiet` | `-q` | Suppress non-error output | False |
| `--log-level` | | Set logging level | `INFO` |
| `--help` | `-h` | Show help message | |

### Log Levels
- `DEBUG`: Detailed diagnostic information
- `INFO`: General information about program execution
- `WARNING`: Warning messages about potential issues
- `ERROR`: Error messages only

## 📚 Commands Overview

| Command | Purpose | Use Case |
|---------|---------|----------|
| [`train`](#train) | Train ML models | Model development and AutoML |
| [`predict`](#predict) | Generate predictions | Batch inference |
| [`explain`](#explain) | Model interpretability | Understanding model decisions |
| [`deploy`](#deploy) | Deploy models | Production deployment |
| [`monitor`](#monitor) | Monitor models | Production monitoring |
| [`init`](#init) | Initialize configs | Project setup |
| [`validate`](#validate) | Validate configs | Configuration validation |
| [`version`](#version) | Show version info | System information |

## 🔧 Command Details

### train

Train machine learning models with various modes and configurations.

```bash
ml-pipeline train [OPTIONS]
```

#### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--mode` | Choice | Training mode: `train`, `validate`, `tune`, `automl` | `train` |
| `--data` | Path | Path to training data file | From config |
| `--output` | Path | Output directory for model artifacts | `./artifacts/models` |
| `--experiment-name` | String | MLflow experiment name | From config |
| `--tags` | JSON | Experiment tags as JSON string | `{}` |
| `--dry-run` | Flag | Validate configuration without training | False |

#### Training Modes

##### `train` - Standard Training
Trains models specified in configuration:
```bash
ml-pipeline train --config configs/pipeline_config.yaml --mode train
```

##### `automl` - Automated ML
Automatically selects and tunes models:
```bash
ml-pipeline train --mode automl --experiment-name "fraud-automl"
```

##### `tune` - Hyperparameter Tuning
Optimizes hyperparameters for specified models:
```bash
ml-pipeline train --mode tune --output ./tuned_models/
```

##### `validate` - Cross Validation
Performs cross-validation without saving models:
```bash
ml-pipeline train --mode validate --data validation_data.csv
```

#### Examples

```bash
# Basic training
ml-pipeline train

# AutoML with custom data
ml-pipeline train --mode automl --data fraud_data.csv --verbose

# Hyperparameter tuning with experiment tracking
ml-pipeline train --mode tune --experiment-name "xgb-tuning" --tags '{"team": "fraud", "version": "v2"}'

# Dry run to validate configuration
ml-pipeline train --dry-run --config configs/test_config.yaml
```

---

### predict

Generate predictions from trained models.

```bash
ml-pipeline predict [OPTIONS]
```

#### Required Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--model` | `-m` | Path | Path to trained model file |
| `--data` | `-d` | Path | Path to input data for prediction |

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output` | Path | Output file for predictions | Auto-generated |
| `--batch-size` | Integer | Batch size for processing | 1000 |
| `--include-probabilities` | Flag | Include prediction probabilities | False |
| `--format` | Choice | Output format: `csv`, `parquet`, `json` | `csv` |

#### Examples

```bash
# Basic prediction
ml-pipeline predict --model model.pkl --data test_data.csv

# Batch prediction with probabilities
ml-pipeline predict -m model.pkl -d large_dataset.csv --batch-size 5000 --include-probabilities

# JSON output format
ml-pipeline predict --model model.pkl --data data.csv --format json --output results.json
```

#### Output Format

The prediction output includes:
- All original columns from input data
- `prediction`: Model predictions
- `probability`: Prediction probabilities (if requested)
- `probability_class_N`: Individual class probabilities for multi-class

---

### explain

Generate model explanations and interpretability analysis.

```bash
ml-pipeline explain [OPTIONS]
```

#### Required Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--model` | `-m` | Path | Path to trained model file |

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--data` | Path | Path to data for explanation | Model training data sample |
| `--method` | Choice | Explanation method | `shap` |
| `--output` | Path | Output directory for explanations | Auto-generated |
| `--sample-size` | Integer | Number of samples to explain | 100 |
| `--generate-plots` | Flag | Generate explanation plots | False |

#### Explanation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `shap` | SHAP (SHapley Additive exPlanations) | Feature importance and interactions |
| `lime` | Local Interpretable Model-agnostic Explanations | Local explanations |
| `ale` | Accumulated Local Effects plots | Feature effects without correlation |
| `anchors` | High-precision explanations | Rule-based explanations |
| `counterfactuals` | Counterfactual explanations | "What-if" scenarios |
| `all` | All available methods | Comprehensive analysis |

#### Examples

```bash
# SHAP explanations with plots
ml-pipeline explain --model model.pkl --method shap --generate-plots

# Multiple explanation methods
ml-pipeline explain -m model.pkl -d data.csv --method all --output explanations/

# Local explanations for specific samples
ml-pipeline explain --model model.pkl --method lime --sample-size 50
```

#### Output Files

- `explanation_summary.json`: Summary of explanation results
- `shap_summary.html`: Interactive SHAP dashboard
- `feature_importance.png`: Feature importance plots
- `individual_explanations/`: Individual sample explanations

---

### deploy

Deploy trained models to production environments.

```bash
ml-pipeline deploy [OPTIONS]
```

#### Required Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--model` | `-m` | Path | Path to trained model file |

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--environment` | Choice | Target environment: `development`, `staging`, `production` | `development` |
| `--platform` | Choice | Deployment platform | `kubernetes` |
| `--replicas` | Integer | Number of replicas | 3 |
| `--cpu-request` | String | CPU request per replica | `500m` |
| `--memory-request` | String | Memory request per replica | `1Gi` |
| `--dry-run` | Flag | Generate configs without deploying | False |

#### Deployment Platforms

| Platform | Description | Requirements |
|----------|-------------|--------------|
| `kubernetes` | Kubernetes cluster deployment | `kubectl` configured |
| `docker` | Docker container deployment | Docker installed |
| `aws` | AWS SageMaker deployment | AWS CLI configured |
| `gcp` | Google Cloud AI Platform | `gcloud` configured |
| `azure` | Azure Machine Learning | Azure CLI configured |

#### Examples

```bash
# Kubernetes production deployment
ml-pipeline deploy --model model.pkl --environment production --platform kubernetes

# Docker development deployment
ml-pipeline deploy -m model.pkl -e development --platform docker --replicas 1

# Dry run to generate configs
ml-pipeline deploy --model model.pkl --dry-run --platform kubernetes
```

#### Generated Files

- `deployment_config.json`: Deployment configuration
- `deployment.yaml`: Kubernetes manifest (for K8s)
- `Dockerfile`: Docker configuration (for Docker)
- `requirements.txt`: Python dependencies

---

### monitor

Monitor deployed models and pipeline performance.

```bash
ml-pipeline monitor [OPTIONS]
```

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--deployment-name` | String | Name of deployment to monitor | Auto-detect |
| `--environment` | Choice | Environment to monitor | All |
| `--metrics` | Multiple | Metrics to monitor | `performance`, `drift`, `fairness` |
| `--dashboard` | Flag | Launch monitoring dashboard | False |
| `--alerts` | Flag | Enable alert monitoring | False |

#### Monitoring Metrics

| Metric | Description | Alerts |
|--------|-------------|---------|
| `performance` | Model accuracy, latency, throughput | Performance degradation |
| `drift` | Data and concept drift detection | Distribution changes |
| `fairness` | Bias and fairness monitoring | Fairness violations |
| `business` | Custom business metrics | Business KPI changes |
| `system` | System resource utilization | Resource limits |

#### Examples

```bash
# Launch monitoring dashboard
ml-pipeline monitor --dashboard --metrics performance drift

# Monitor specific deployment
ml-pipeline monitor --deployment-name fraud-model --environment production --alerts

# Monitor all metrics
ml-pipeline monitor --metrics performance drift fairness business system
```

#### Dashboard Features

- Real-time performance metrics
- Data drift visualization
- Fairness metrics tracking
- Alert configuration
- Historical trend analysis

---

### init

Initialize new pipeline configuration files.

```bash
ml-pipeline init [OPTIONS]
```

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--config-type` | Choice | Configuration template type | `basic` |
| `--output` | Path | Output directory | `configs/` |
| `--format` | Choice | File format: `yaml`, `json` | `yaml` |

#### Configuration Types

| Type | Description | Use Case |
|------|-------------|----------|
| `basic` | Simple ML pipeline | Getting started, prototyping |
| `fraud-detection` | Fraud detection pipeline | Financial services, security |
| `automl` | AutoML configuration | Automated model selection |
| `enterprise` | Enterprise-grade setup | Production, compliance |

#### Examples

```bash
# Initialize basic configuration
ml-pipeline init

# Create fraud detection setup
ml-pipeline init --config-type fraud-detection --output fraud-configs/

# Generate enterprise configuration in JSON
ml-pipeline init --config-type enterprise --format json
```

#### Generated Files

Based on configuration type:
- `pipeline_config.yaml`: Main pipeline configuration
- `automl_config.yaml`: AutoML settings (if applicable)
- `deployment_config.yaml`: Deployment settings
- `monitoring_config.yaml`: Monitoring configuration

---

### validate

Validate pipeline configuration files.

```bash
ml-pipeline validate [CONFIG_FILE] [OPTIONS]
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `CONFIG_FILE` | Path | Path to configuration file to validate |

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--strict` | Flag | Enable strict validation mode | False |

#### Validation Checks

##### Schema Validation
- YAML/JSON syntax validation
- Required fields verification
- Data type validation
- Value range checks

##### Business Logic Validation
- Configuration consistency
- Dependency compatibility
- Resource requirements
- Security best practices

##### Strict Mode Checks
- File existence verification
- Database connectivity
- Package availability
- Performance optimization

#### Examples

```bash
# Basic validation
ml-pipeline validate configs/pipeline_config.yaml

# Strict validation with all checks
ml-pipeline validate configs/pipeline_config.yaml --strict

# Validate multiple files
ml-pipeline validate configs/*.yaml
```

#### Validation Output

```
✅ Configuration validation passed!

⚠️  Warnings:
  • Database connection not using SSL
  • Debug logging enabled in production

💡 Suggestions:
  • Consider enabling model versioning
  • Add monitoring configuration
```

---

### version

Display version and build information.

```bash
ml-pipeline version [OPTIONS]
```

#### Optional Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--format` | Choice | Output format: `table`, `json`, `yaml` | `table` |

#### Examples

```bash
# Default table format
ml-pipeline version

# JSON format for programmatic use
ml-pipeline version --format json

# YAML format
ml-pipeline version --format yaml
```

#### Version Information

- Framework version
- Build information
- API version
- Feature availability
- Python version
- System information

## 🔧 Configuration

### Configuration File Priority

1. Command-line `--config` option
2. `ML_PIPELINE_CONFIG` environment variable
3. `./configs/pipeline_config.yaml`
4. `~/.ml-pipeline/config.yaml`

### Configuration Validation

All commands automatically validate configuration before execution:

```bash
# Explicit validation
ml-pipeline validate configs/pipeline_config.yaml

# Validation during training
ml-pipeline train --config configs/pipeline_config.yaml --dry-run
```

## 🌍 Environment Variables

### Core Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ML_PIPELINE_CONFIG` | Default configuration file path | `configs/pipeline_config.yaml` |
| `ML_PIPELINE_LOG_LEVEL` | Default log level | `INFO` |
| `ML_PIPELINE_ARTIFACTS_DIR` | Default artifacts directory | `./artifacts` |

### Data Source Variables

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Data files directory |
| `DB_HOST` | Database host |
| `DB_PORT` | Database port |
| `DB_NAME` | Database name |
| `DB_USERNAME` | Database username |
| `DB_PASSWORD` | Database password |

### MLflow Variables

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `MLFLOW_EXPERIMENT_NAME` | Default experiment name |

### Cloud Variables

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account key |
| `AZURE_CLIENT_ID` | Azure client ID |

## 📋 Examples

### Complete Workflow

```bash
# 1. Initialize project
ml-pipeline init --config-type fraud-detection

# 2. Validate configuration
ml-pipeline validate configs/fraud_detection_config.yaml --strict

# 3. Train models with AutoML
ml-pipeline train --mode automl --experiment-name "fraud-v1" --verbose

# 4. Generate predictions
ml-pipeline predict --model ./artifacts/models/best_model.pkl --data test_data.csv

# 5. Explain model decisions
ml-pipeline explain --model ./artifacts/models/best_model.pkl --method all --generate-plots

# 6. Deploy to staging
ml-pipeline deploy --model ./artifacts/models/best_model.pkl --environment staging --dry-run

# 7. Monitor performance
ml-pipeline monitor --deployment-name fraud-model --dashboard
```

### Production Pipeline

```bash
# Production training with monitoring
ml-pipeline train \
  --config configs/production_config.yaml \
  --mode automl \
  --experiment-name "production-fraud-$(date +%Y%m%d)" \
  --tags '{"environment": "production", "version": "v2.1"}' \
  --verbose

# Batch prediction pipeline
ml-pipeline predict \
  --model s3://models/fraud-model-v2.1.pkl \
  --data s3://data/daily-transactions.parquet \
  --output s3://predictions/$(date +%Y%m%d)/ \
  --batch-size 10000 \
  --include-probabilities \
  --format parquet

# Automated deployment
ml-pipeline deploy \
  --model s3://models/fraud-model-v2.1.pkl \
  --environment production \
  --platform kubernetes \
  --replicas 5 \
  --cpu-request "1000m" \
  --memory-request "2Gi"
```

### Development Workflow

```bash
# Quick prototyping
ml-pipeline init --config-type basic
ml-pipeline train --mode validate --data sample_data.csv
ml-pipeline explain --model ./artifacts/models/best_model.pkl --method shap

# Experimentation
ml-pipeline train --mode tune --experiment-name "experiment-$(whoami)"
ml-pipeline monitor --metrics performance --dashboard
```

## 🆘 Troubleshooting

### Common Issues

#### Configuration Errors
```bash
# Validate before running
ml-pipeline validate configs/pipeline_config.yaml --strict

# Check specific configuration sections
ml-pipeline train --dry-run --verbose
```

#### Memory Issues
```bash
# Reduce batch size
ml-pipeline predict --model model.pkl --data large_data.csv --batch-size 1000

# Use streaming for large datasets
# Enable in configuration: data_processing.streaming: true
```

#### Permission Issues
```bash
# Check file permissions
ls -la configs/pipeline_config.yaml

# Verify database access
ml-pipeline validate configs/pipeline_config.yaml --strict
```

### Getting Help

```bash
# Command-specific help
ml-pipeline train --help
ml-pipeline predict --help

# Version and feature information
ml-pipeline version

# Verbose output for debugging
ml-pipeline train --verbose --log-level DEBUG
```

---

**Need more help?** Check our [troubleshooting guide](operations/troubleshooting.md) or [FAQ](FAQ.md).