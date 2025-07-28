# Migration Guide: v1.x to v2.0

This guide helps you migrate from ML Pipeline Framework v1.x to the new v2.0 with enhanced enterprise features, improved AutoML capabilities, and comprehensive interpretability.

## 📋 Table of Contents

- [Overview of Changes](#overview-of-changes)
- [Breaking Changes](#breaking-changes)
- [Migration Steps](#migration-steps)
- [Configuration Updates](#configuration-updates)
- [Code Changes](#code-changes)
- [New Features to Adopt](#new-features-to-adopt)
- [Migration Scripts](#migration-scripts)
- [Testing Your Migration](#testing-your-migration)
- [Rollback Plan](#rollback-plan)

## 🔄 Overview of Changes

### Major Improvements in v2.0

1. **Data Processing**: Multi-engine support (Pandas, Polars, DuckDB)
2. **Data Sources**: CSV as default, enhanced database support
3. **AutoML**: Expanded algorithm support with business metric optimization
4. **Interpretability**: 15+ explanation methods with regulatory compliance
5. **Fraud Detection**: Specialized features preserving natural imbalance
6. **Production**: Enhanced monitoring, A/B testing, and deployment options

### Architecture Changes

```
v1.x Architecture                    v2.0 Architecture
┌─────────────────┐                 ┌─────────────────┐
│ Database Only   │       →         │ Multi-Source    │
│ Single Engine   │                 │ Multi-Engine    │
│ Basic ML        │                 │ Advanced AutoML │
│ Limited Explain │                 │ Full Explain    │
└─────────────────┘                 └─────────────────┘
```

## ⚠️ Breaking Changes

### 1. Default Data Source Changed

**v1.x (Old)**:
```yaml
data_source:
  type: hive
  connection:
    jdbc_url: "jdbc:hive://..."
```

**v2.0 (New)**:
```yaml
data_source:
  type: csv  # CSV is now default
  csv_options:
    file_paths: ["data/transactions.csv"]
    separator: ","
    encoding: "utf-8"
```

### 2. Imbalance Handling Strategy

**v1.x (Old)**:
```yaml
preprocessing:
  balance_strategy: "smote"
  target_balance_ratio: 0.5
```

**v2.0 (New)**:
```yaml
imbalance_handling:
  strategy: "preserve_natural"  # Preserves fraud imbalance
  fraud_aware_sampling: true
  cost_sensitive_learning: true
```

### 3. Model Configuration Structure

**v1.x (Old)**:
```yaml
models:
  - algorithm: "XGBClassifier"
    hyperparameters: {...}
```

**v2.0 (New)**:
```yaml
model_training:
  automl_enabled: true
  automl_config_path: "./configs/automl_config.yaml"
  models: [...]  # Optional manual specification
```

### 4. Explainability API Changes

**v1.x (Old)**:
```python
from explainability import SHAPExplainer
explainer = SHAPExplainer(model)
explanations = explainer.explain(data)
```

**v2.0 (New)**:
```python
from explainability.interpretability_pipeline import InterpretabilityPipeline
pipeline = InterpretabilityPipeline(model)
explanations = pipeline.explain_all(data, methods=['shap', 'lime', 'anchors'])
```

## 🚀 Migration Steps

### Step 1: Backup Current Setup

```bash
# Create backup directory
mkdir -p migration_backup/$(date +%Y%m%d)

# Backup configurations
cp -r configs/ migration_backup/$(date +%Y%m%d)/
cp -r artifacts/ migration_backup/$(date +%Y%m%d)/
cp -r models/ migration_backup/$(date +%Y%m%d)/

# Backup custom code
cp -r src/ migration_backup/$(date +%Y%m%d)/
```

### Step 2: Install v2.0

```bash
# Update environment
conda env update -f environment.yml

# Or reinstall from scratch
conda env create -n ml-pipeline-v2 -f environment.yml
conda activate ml-pipeline-v2
```

### Step 3: Update Configuration Files

Run the migration script to automatically update configurations:

```bash
python scripts/migrate_config.py \
  --old-config configs/pipeline_config.yaml \
  --new-config configs/pipeline_config_v2.yaml \
  --version v2.0
```

### Step 4: Migrate Data Sources

#### From Hive to CSV

```bash
# Export Hive data to CSV
python scripts/export_hive_to_csv.py \
  --connection-string "jdbc:hive://..." \
  --query "SELECT * FROM transactions" \
  --output data/transactions.csv
```

#### Database to CSV Migration

```python
# migration_scripts/db_to_csv.py
import pandas as pd
from sqlalchemy import create_engine

def migrate_database_to_csv(connection_string, query, output_path):
    """Migrate database data to CSV format."""
    engine = create_engine(connection_string)
    
    # Read in chunks for large datasets
    chunk_size = 50000
    chunks = []
    
    for chunk in pd.read_sql_query(query, engine, chunksize=chunk_size):
        chunks.append(chunk)
    
    # Combine and save
    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Migrated {len(df)} rows to {output_path}")

# Usage
migrate_database_to_csv(
    "postgresql://user:pass@host:5432/db",
    "SELECT * FROM fraud_transactions WHERE date >= '2023-01-01'",
    "data/fraud_transactions.csv"
)
```

### Step 5: Update Custom Code

#### Model Training Code

**v1.x (Old)**:
```python
from models import XGBoostModel
from preprocessing import DataPreprocessor

# Manual model training
preprocessor = DataPreprocessor(balance_strategy='smote')
model = XGBoostModel(n_estimators=100)
model.fit(X_train, y_train)
```

**v2.0 (New)**:
```python
from pipeline_orchestrator import PipelineOrchestrator

# AutoML-based training
config = load_config('configs/pipeline_config.yaml')
orchestrator = PipelineOrchestrator(config)
results = orchestrator.run(mode='automl')
best_model = results.best_model
```

#### Explainability Code

**v1.x (Old)**:
```python
from explainability import SHAPExplainer, LIMEExplainer

# Manual explanation
shap_explainer = SHAPExplainer(model)
lime_explainer = LIMEExplainer(model)

shap_values = shap_explainer.explain(X_test)
lime_explanations = lime_explainer.explain(X_test[0])
```

**v2.0 (New)**:
```python
from explainability.interpretability_pipeline import InterpretabilityPipeline

# Unified explanation pipeline
pipeline = InterpretabilityPipeline(model)
explanations = pipeline.explain_all(
    X_test, 
    methods=['shap', 'lime', 'anchors', 'counterfactuals'],
    generate_report=True
)
```

## ⚙️ Configuration Updates

### Complete Configuration Migration

Use our automated migration tool:

```bash
python scripts/config_migrator.py --input configs/ --output configs_v2/
```

Or manually update each section:

#### Data Source Configuration

**Before (v1.x)**:
```yaml
data_source:
  type: hive
  connection:
    jdbc_url: "jdbc:hive://hive-server:10000/default"
    username: "ml_user"
    password: "password"
  query: "SELECT * FROM fraud_transactions"
```

**After (v2.0)**:
```yaml
data_source:
  type: csv
  csv_options:
    file_paths: ["data/fraud_transactions.csv"]
    separator: ","
    encoding: "utf-8"
    chunk_size: 50000
    optimize_dtypes: true
    
# Alternative: Keep database source
data_source:
  type: hive
  database:
    connection:
      jdbc_url: "jdbc:hive://hive-server:10000/default"
      username: "ml_user"
      password: "${HIVE_PASSWORD}"  # Use env vars
    extraction:
      query: "SELECT * FROM fraud_transactions"
      chunk_size: 10000
```

#### Data Processing Engine

**New in v2.0**:
```yaml
data_processing:
  engine: "auto"  # pandas, polars, duckdb, auto
  memory_limit: "8GB"
  parallel_processing: true
  max_workers: 4
```

#### AutoML Configuration

**Before (v1.x)**:
```yaml
models:
  - algorithm: "XGBClassifier"
    hyperparameters:
      n_estimators: 100
      max_depth: 6
```

**After (v2.0)**:
```yaml
model_training:
  automl_enabled: true
  automl_config_path: "./configs/automl_config.yaml"
  automl:
    algorithms:
      - "logistic_regression"
      - "random_forest"
      - "xgboost"
      - "lightgbm"
      - "catboost"
    time_budget: 3600
    optimization_metric: "precision_at_1_percent"
```

#### Explainability Configuration

**New in v2.0**:
```yaml
explainability:
  enabled: true
  methods:
    global: ["shap", "ale_plots", "permutation_importance"]
    local: ["lime", "anchors", "counterfactuals"]
    advanced: ["trust_scores", "prototypes"]
  generate_reports: true
  fraud_specific:
    reason_codes: true
    narrative_explanations: true
```

## 🆕 New Features to Adopt

### 1. Enhanced AutoML

```yaml
# Enable comprehensive AutoML
automl:
  enabled: true
  algorithms:
    - "xgboost"
    - "lightgbm"
    - "catboost"
    - "h2o_automl"
  time_budget: 3600
  ensemble_methods: ["voting", "stacking"]
  interpretability_constraint: 0.8
```

### 2. Business Metrics Optimization

```yaml
# Optimize for fraud detection business metrics
business_metrics:
  primary_objective: "maximize_revenue"
  cost_matrix:
    true_negative_value: 0
    false_positive_cost: 100
    false_negative_cost: 500
    true_positive_value: 1000
```

### 3. Comprehensive Monitoring

```yaml
# Enable production monitoring
monitoring:
  enabled: true
  drift_detection: true
  performance_tracking: true
  fairness_monitoring: true
  ab_testing_enabled: true
```

### 4. Regulatory Compliance

```yaml
# Enable compliance features
output:
  admissible_ml_reports: true
  model_cards: true
  fairness_analysis: true
  regulatory_compliance: ["sr11-7", "gdpr", "fair_lending"]
```

## 🔧 Migration Scripts

### Automated Configuration Migration

```python
# scripts/migrate_config.py
import yaml
import argparse
from pathlib import Path

def migrate_config(old_config_path, new_config_path):
    """Migrate v1.x configuration to v2.0 format."""
    
    with open(old_config_path, 'r') as f:
        old_config = yaml.safe_load(f)
    
    # Create new configuration structure
    new_config = {
        'pipeline': {
            'name': old_config.get('name', 'ml-pipeline'),
            'version': '2.0.0',
            'description': 'Migrated from v1.x',
            'environment': old_config.get('environment', 'dev'),
        }
    }
    
    # Migrate data source
    if old_config.get('data_source', {}).get('type') == 'hive':
        new_config['data_source'] = {
            'type': 'csv',
            'csv_options': {
                'file_paths': ['data/migrated_data.csv'],
                'separator': ',',
                'encoding': 'utf-8',
                'chunk_size': 50000,
            }
        }
        print("⚠️  Data source changed to CSV. Export your Hive data manually.")
    
    # Migrate model configuration to AutoML
    if 'models' in old_config:
        algorithms = []
        for model in old_config['models']:
            if 'XGB' in model.get('algorithm', ''):
                algorithms.append('xgboost')
            elif 'RandomForest' in model.get('algorithm', ''):
                algorithms.append('random_forest')
        
        new_config['model_training'] = {
            'automl_enabled': True,
            'automl': {
                'algorithms': algorithms or ['xgboost', 'lightgbm'],
                'time_budget': 3600,
                'optimization_metric': 'precision_at_1_percent',
            }
        }
    
    # Add new v2.0 features
    new_config.update({
        'data_processing': {
            'engine': 'auto',
            'memory_limit': '8GB',
            'parallel_processing': True,
        },
        'imbalance_handling': {
            'strategy': 'preserve_natural',
            'fraud_aware_sampling': True,
            'cost_sensitive_learning': True,
        },
        'explainability': {
            'enabled': True,
            'methods': {
                'global': ['shap', 'ale_plots'],
                'local': ['lime', 'anchors'],
            },
            'generate_reports': True,
        },
        'monitoring': {
            'enabled': True,
            'drift_detection': True,
            'performance_tracking': True,
        }
    })
    
    # Save new configuration
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration migrated to {new_config_path}")
    return new_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Migrate ML Pipeline config')
    parser.add_argument('--old-config', required=True, help='Path to v1.x config')
    parser.add_argument('--new-config', required=True, help='Path for v2.0 config')
    
    args = parser.parse_args()
    migrate_config(args.old_config, args.new_config)
```

### Data Migration Script

```python
# scripts/migrate_data.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import argparse

def migrate_database_to_csv(connection_string, table_name, output_path, chunk_size=50000):
    """Migrate database table to CSV with chunked processing."""
    
    engine = create_engine(connection_string)
    
    # Get total row count
    count_query = f"SELECT COUNT(*) as count FROM {table_name}"
    total_rows = pd.read_sql_query(count_query, engine)['count'].iloc[0]
    
    print(f"Migrating {total_rows} rows from {table_name}...")
    
    # Process in chunks
    chunks_processed = 0
    for chunk in pd.read_sql_query(
        f"SELECT * FROM {table_name}", 
        engine, 
        chunksize=chunk_size
    ):
        # Write first chunk with header, subsequent chunks without
        mode = 'w' if chunks_processed == 0 else 'a'
        header = chunks_processed == 0
        
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
        chunks_processed += 1
        
        print(f"Processed chunk {chunks_processed} ({chunks_processed * chunk_size} / {total_rows} rows)")
    
    print(f"✅ Data migration complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Migrate database to CSV')
    parser.add_argument('--connection', required=True, help='Database connection string')
    parser.add_argument('--table', required=True, help='Table name to migrate')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Chunk size')
    
    args = parser.parse_args()
    migrate_database_to_csv(args.connection, args.table, args.output, args.chunk_size)
```

## 🧪 Testing Your Migration

### 1. Configuration Validation

```bash
# Validate new configuration
python -m src.utils.config_validator configs/pipeline_config_v2.yaml --strict
```

### 2. Data Processing Test

```python
# test_migration.py
from src.data_access.factory import DataConnectorFactory
from src.pipeline_orchestrator import PipelineOrchestrator

def test_data_loading():
    """Test that data loads correctly with new configuration."""
    config = {
        'data_source': {
            'type': 'csv',
            'csv_options': {
                'file_paths': ['data/migrated_data.csv'],
                'chunk_size': 10000,
            }
        }
    }
    
    connector = DataConnectorFactory.create_connector(config['data_source'])
    data = connector.load_data()
    
    print(f"✅ Data loaded: {data.shape}")
    return data

def test_pipeline_execution():
    """Test full pipeline execution."""
    from src.utils.config_parser import ConfigParser
    
    config = ConfigParser.load_config('configs/pipeline_config_v2.yaml')
    orchestrator = PipelineOrchestrator(config)
    
    # Test dry run first
    results = orchestrator.validate_configuration()
    print("✅ Configuration validation passed")
    
    return results

if __name__ == "__main__":
    test_data_loading()
    test_pipeline_execution()
```

### 3. Model Comparison

```python
# Compare v1.x vs v2.0 model performance
def compare_model_performance():
    """Compare old vs new model performance."""
    
    # Load old model results
    old_results = load_old_results('artifacts_v1/model_results.json')
    
    # Train new model
    new_results = train_new_model('configs/pipeline_config_v2.yaml')
    
    # Compare metrics
    comparison = {
        'accuracy': {
            'v1': old_results['accuracy'],
            'v2': new_results['accuracy'],
            'improvement': new_results['accuracy'] - old_results['accuracy']
        },
        'precision_at_1_percent': {
            'v1': old_results.get('precision_at_1_percent', 'N/A'),
            'v2': new_results['precision_at_1_percent'],
        }
    }
    
    print("📊 Model Performance Comparison:")
    for metric, values in comparison.items():
        print(f"  {metric}: {values}")
```

## 🔄 Rollback Plan

If you encounter issues with v2.0, here's how to rollback:

### 1. Quick Rollback

```bash
# Restore from backup
cp -r migration_backup/$(date +%Y%m%d)/* ./
conda activate ml-pipeline-v1  # Switch back to old environment
```

### 2. Selective Rollback

```bash
# Keep new features but use old data source
cp migration_backup/configs/data_source_config.yaml configs/
```

### 3. Gradual Migration

```yaml
# Use hybrid configuration during transition
data_source:
  type: csv  # New
  csv_options: {...}

model_training:
  automl_enabled: false  # Keep old manual config
  models: [...]  # Old model configuration

explainability:
  enabled: false  # Disable new features initially
```

## 📞 Migration Support

If you encounter issues during migration:

1. **Check Migration Log**: Review `migration.log` for detailed error messages
2. **Validate Configuration**: Use `config_validator.py --strict`
3. **Test Data Loading**: Verify your data migrated correctly
4. **Community Support**: Check [GitHub Discussions](https://github.com/ml-pipeline-framework/discussions)
5. **Enterprise Support**: Contact support@ml-pipeline-framework.io

## 📈 Post-Migration Optimization

After successful migration, optimize your setup:

### 1. Performance Tuning

```yaml
# Optimize for your data size
data_processing:
  engine: "polars"  # For 1-10GB datasets
  memory_limit: "16GB"
  max_workers: 8
```

### 2. Enable Advanced Features

```yaml
# Gradually enable new features
explainability:
  enabled: true
  methods:
    global: ["shap"]  # Start with one method
    
monitoring:
  enabled: true
  drift_detection: true  # Enable monitoring
```

### 3. Production Readiness

```yaml
# Configure for production
monitoring:
  comprehensive_monitoring: true
  alert_thresholds:
    performance_degradation: 0.05
    
security:
  encryption: true
  audit_logging: true
```

---

**Migration Complete!** 🎉 You're now ready to leverage all the powerful new features in ML Pipeline Framework v2.0.