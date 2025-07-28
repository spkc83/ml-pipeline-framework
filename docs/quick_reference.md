# Quick Reference Card

Essential commands, configurations, and code snippets for the ML Pipeline Framework v2.0.

## 🚀 Quick Start Commands

```bash
# Install framework
pip install -e .

# Run AutoML training
ml-pipeline train --config configs/pipeline_config.yaml --mode automl

# Generate explanations
ml-pipeline explain --model artifacts/best_model.pkl --methods shap,lime,anchors

# Deploy to production
ml-pipeline deploy --model artifacts/best_model.pkl --environment production

# Monitor model performance
ml-pipeline monitor --model artifacts/best_model.pkl --drift-detection
```

## ⚙️ Essential Configuration

### Minimal CSV Configuration
```yaml
pipeline:
  name: "fraud-detection"
  version: "2.0.0"

data_source:
  type: csv
  csv_options:
    file_paths: ["data/transactions.csv"]

model_training:
  automl_enabled: true
  automl:
    algorithms: ["xgboost", "lightgbm"]
    time_budget: 1800  # 30 minutes
    optimization_metric: "precision_at_1_percent"

explainability:
  enabled: true
  methods:
    global: ["shap"]
    local: ["lime"]
```

### Production Configuration Template
```yaml
pipeline:
  name: "fraud-detection-prod"
  version: "2.0.0"
  environment: "production"

data_source:
  type: csv
  csv_options:
    file_paths: ["data/fraud_transactions.csv"]
    chunk_size: 50000
    optimize_dtypes: true

data_processing:
  engine: "auto"  # Auto-selects best engine
  memory_limit: "16GB"
  parallel_processing: true

model_training:
  automl_enabled: true
  automl:
    algorithms: ["xgboost", "lightgbm", "catboost", "h2o"]
    time_budget: 7200  # 2 hours
    optimization_metric: "precision_at_1_percent"
    ensemble_methods: ["voting", "stacking"]

imbalance_handling:
  strategy: "preserve_natural"
  fraud_aware_sampling: true
  cost_sensitive_learning: true

explainability:
  enabled: true
  methods:
    global: ["shap", "ale_plots", "permutation_importance"]
    local: ["lime", "anchors", "counterfactuals"]
    advanced: ["trust_scores", "prototypes"]
  fraud_specific:
    reason_codes: true
    narrative_explanations: true

monitoring:
  enabled: true
  drift_detection: true
  ab_testing_enabled: true
  fairness_monitoring: true

security:
  encryption: true
  audit_logging: true
  rbac_enabled: true
```

## 🐍 Essential Python Code

### Basic Pipeline Usage
```python
from src.pipeline_orchestrator import PipelineOrchestrator
from src.utils.config_parser import ConfigParser

# Load config and run pipeline
config = ConfigParser.load_config('configs/pipeline_config.yaml')
pipeline = PipelineOrchestrator(config)
results = pipeline.run(mode='automl')

# Access results
best_model = results.best_model
print(f"Best algorithm: {results.best_model_name}")
print(f"Score: {results.best_score:.4f}")
```

### Fraud Detection Pipeline
```python
# CSV-based fraud detection
config = {
    'data_source': {
        'type': 'csv',
        'csv_options': {
            'file_paths': ['data/fraud_data.csv'],
            'chunk_size': 50000
        }
    },
    'model_training': {
        'automl_enabled': True,
        'automl': {
            'algorithms': ['xgboost', 'lightgbm', 'catboost'],
            'time_budget': 3600,
            'optimization_metric': 'precision_at_1_percent'
        }
    },
    'imbalance_handling': {
        'strategy': 'preserve_natural',
        'fraud_aware_sampling': True
    }
}

pipeline = PipelineOrchestrator(config)
results = pipeline.run(mode='automl')

# Business impact analysis
business_metrics = results.business_metrics
print(f"Annual savings: ${business_metrics['annual_savings']:,.2f}")
print(f"ROI: {business_metrics['roi']:.1%}")
```

### Model Explanations
```python
from explainability.interpretability_pipeline import InterpretabilityPipeline

# Generate comprehensive explanations
pipeline = InterpretabilityPipeline(model)
explanations = pipeline.explain_all(
    X_test, 
    methods=['shap', 'lime', 'anchors', 'counterfactuals'],
    generate_report=True
)

# Access specific explanations
shap_values = explanations['shap']['feature_importance']
lime_explanations = explanations['lime']['local_explanations']
decision_rules = explanations['anchors']['rules']
```

## 📊 Key Metrics & Formulas

### Fraud Detection Metrics
```python
# Precision at K% (e.g., Precision at 1%)
def precision_at_k(y_true, y_proba, k=0.01):
    threshold = np.percentile(y_proba, (1-k)*100)
    y_pred = (y_proba >= threshold).astype(int)
    return precision_score(y_true, y_pred)

# Expected Value
def expected_value(y_true, y_pred, fraud_value=1000, fp_cost=100):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp * fraud_value - fp * fp_cost

# Business ROI
def calculate_roi(tp, fp, fn, fraud_value=1000, fp_cost=100, fn_cost=500, operating_cost=10000):
    benefits = tp * fraud_value
    costs = fp * fp_cost + fn * fn_cost + operating_cost
    return (benefits - costs) / costs
```

### Data Drift Detection
```python
# Population Stability Index (PSI)
def calculate_psi(reference, current, bins=10):
    ref_hist, bin_edges = np.histogram(reference, bins=bins, density=True)
    curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
    
    ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
    curr_hist = np.where(curr_hist == 0, 0.0001, curr_hist)
    
    psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
    return psi

# Drift interpretation
# PSI < 0.1: No significant drift
# 0.1 <= PSI < 0.2: Moderate drift
# PSI >= 0.2: Significant drift
```

## 🔧 Configuration Shortcuts

### Algorithm Selection by Use Case
```yaml
# Quick prototyping (< 10 minutes)
algorithms: ["logistic_regression", "random_forest"]
time_budget: 600

# Development testing (30 minutes)
algorithms: ["xgboost", "lightgbm"]
time_budget: 1800

# Production training (2+ hours)
algorithms: ["xgboost", "lightgbm", "catboost", "h2o"]
time_budget: 7200
ensemble_methods: ["voting", "stacking"]
```

### Data Processing Engine Selection
```yaml
# Small datasets (< 1GB)
data_processing:
  engine: "pandas"

# Medium datasets (1-10GB)
data_processing:
  engine: "polars"

# Large datasets (> 10GB)
data_processing:
  engine: "duckdb"

# Auto-selection based on data size
data_processing:
  engine: "auto"
```

### Interpretability Methods by Need
```yaml
# Regulatory compliance
explainability:
  methods:
    global: ["shap"]
    local: ["lime", "anchors"]
  fraud_specific:
    reason_codes: true
    narrative_explanations: true

# Research & analysis
explainability:
  methods:
    global: ["shap", "ale_plots", "functional_anova"]
    local: ["lime", "counterfactuals"]
    advanced: ["trust_scores", "prototypes"]

# Quick insights
explainability:
  methods:
    global: ["shap", "permutation_importance"]
    local: ["lime"]
```

## 🐳 Docker Quick Commands

```bash
# Build image
docker build -t ml-pipeline:2.0.0 .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f ml-pipeline-app

# Execute commands in container
docker-compose exec ml-pipeline-app python src/cli.py --help

# Scale services
docker-compose up -d --scale ml-pipeline-app=3

# Clean up
docker-compose down
docker system prune -a
```

## ☸️ Kubernetes Quick Commands

```bash
# Deploy to production
kubectl apply -f deploy/kubernetes/production/

# Check deployment status
kubectl get pods -n ml-pipeline-production
kubectl get services -n ml-pipeline-production

# View logs
kubectl logs -f deployment/ml-pipeline-app -n ml-pipeline-production

# Scale deployment
kubectl scale deployment ml-pipeline-app --replicas=5 -n ml-pipeline-production

# Port forward for local access
kubectl port-forward svc/ml-pipeline-service 8000:8000 -n ml-pipeline-production

# Check resource usage
kubectl top pods -n ml-pipeline-production
kubectl top nodes
```

## 📈 Monitoring Quick Setup

### Prometheus Metrics Endpoints
```bash
# Application metrics
curl http://localhost:8080/metrics

# Business metrics
curl http://localhost:8080/business-metrics

# Data drift metrics
curl http://localhost:8080/drift-metrics

# Health check
curl http://localhost:8090/health
curl http://localhost:8090/ready
```

### Key Grafana Queries
```promql
# Model accuracy over time
ml_model_accuracy

# Prediction rate
rate(ml_predictions_total[5m])

# Data drift score
ml_data_drift_psi_score

# Business ROI
ml_business_roi

# System resource usage
system_cpu_usage_percent
system_memory_usage_percent
```

## 🚨 Troubleshooting Checklist

### Model Performance Issues
- [ ] Check data quality and drift metrics
- [ ] Verify feature engineering pipeline
- [ ] Review model configuration and hyperparameters
- [ ] Check for data leakage or target contamination
- [ ] Validate training/test split

### Data Processing Issues
- [ ] Verify data source connectivity
- [ ] Check file paths and permissions
- [ ] Validate data schema and types
- [ ] Monitor memory usage and chunk sizes
- [ ] Check data processing engine compatibility

### Deployment Issues
- [ ] Verify container image builds successfully
- [ ] Check Kubernetes resource limits
- [ ] Validate environment variables and secrets
- [ ] Monitor pod startup and health checks
- [ ] Check service connectivity and networking

### Monitoring Issues
- [ ] Verify metrics endpoints are accessible
- [ ] Check Prometheus scraping configuration
- [ ] Validate Grafana data source connections
- [ ] Review alert rule syntax and thresholds
- [ ] Test notification channels

## 📞 Common File Locations

```
ml-pipeline-framework/
├── configs/
│   ├── pipeline_config.yaml      # Main configuration
│   ├── automl_config.yaml        # AutoML settings
│   └── explainability_config.yaml # Explanation settings
├── data/                          # Data files
├── models/                        # Trained models
├── artifacts/                     # Training artifacts
├── logs/                          # Log files
├── docs/                          # Documentation
├── deploy/
│   ├── docker/                    # Docker configs
│   └── kubernetes/                # K8s configs
└── monitoring/                    # Monitoring configs
```

## 🔗 Useful Links

- [Full Documentation](docs/README.md)
- [AutoML Guide](docs/features/automl.md)
- [Interpretability Guide](docs/features/interpretability.md)
- [Migration Guide](docs/migration_guide.md)
- [Deployment Guide](docs/deployment.md)
- [Monitoring Guide](docs/monitoring.md)

---

**Quick Reference Complete!** 🎯 Keep this handy for rapid development and troubleshooting with ML Pipeline Framework v2.0.