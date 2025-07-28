# Configuration Guide

Comprehensive guide to configuring the ML Pipeline Framework for your specific needs.

## 📋 Table of Contents

- [Configuration Overview](#configuration-overview)
- [Configuration Files](#configuration-files)
- [Configuration Structure](#configuration-structure)
- [Data Source Configuration](#data-source-configuration)
- [Model Training Configuration](#model-training-configuration)
- [AutoML Configuration](#automl-configuration)
- [Explainability Configuration](#explainability-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Security Configuration](#security-configuration)
- [Environment-Specific Configuration](#environment-specific-configuration)
- [Best Practices](#best-practices)

## 🔧 Configuration Overview

The ML Pipeline Framework uses YAML configuration files to define all aspects of your ML pipeline:

- **Data Sources**: Where and how to load data
- **Preprocessing**: Data cleaning and transformation
- **Feature Engineering**: Automated feature creation
- **Model Training**: Algorithm selection and hyperparameters
- **Evaluation**: Metrics and validation strategies
- **Explainability**: Interpretability methods and compliance
- **Deployment**: Production deployment settings
- **Monitoring**: Performance and drift monitoring

### Configuration Hierarchy

```
configs/
├── pipeline_config.yaml      # Main pipeline configuration
├── automl_config.yaml        # AutoML-specific settings
├── environment/
│   ├── development.yaml      # Development overrides
│   ├── staging.yaml          # Staging overrides
│   └── production.yaml       # Production overrides
└── templates/
    ├── basic.yaml            # Basic template
    ├── fraud-detection.yaml  # Fraud detection template
    └── enterprise.yaml       # Enterprise template
```

## 📄 Configuration Files

### Main Configuration Files

#### pipeline_config.yaml
The primary configuration file containing all pipeline settings:

```yaml
pipeline:
  name: "ml-pipeline-framework"
  version: "2.0.0"
  description: "Production-grade ML pipeline"
  environment: "${ENVIRONMENT:dev}"
  log_level: "${LOG_LEVEL:INFO}"

data_source:
  type: "csv"
  # ... data source configuration

model_training:
  automl_enabled: true
  # ... training configuration

# ... other sections
```

#### automl_config.yaml
Detailed AutoML configuration for automated model selection:

```yaml
automl_settings:
  name: "fraud-detection-automl"
  version: "2.0.0"
  search_strategy: "bayesian"

algorithm_selection:
  enable_linear: true
  enable_tree_based: true
  # ... algorithm settings

# ... other AutoML sections
```

### Environment-Specific Configurations

Override settings for different environments:

```yaml
# environment/production.yaml
pipeline:
  environment: "production"
  log_level: "INFO"

monitoring:
  enabled: true
  drift_detection: true
  performance_monitoring: true

security:
  encryption: true
  audit_logging: true
```

## 🏗️ Configuration Structure

### Top-Level Sections

| Section | Purpose | Required |
|---------|---------|----------|
| `pipeline` | Pipeline metadata and settings | Yes |
| `data_source` | Data input configuration | Yes |
| `preprocessing` | Data cleaning and validation | No |
| `feature_engineering` | Feature creation and selection | No |
| `model_training` | Model training configuration | Yes |
| `evaluation` | Model evaluation settings | No |
| `explainability` | Interpretability configuration | No |
| `output` | Output and artifact settings | No |
| `monitoring` | Production monitoring | No |
| `security` | Security and compliance | No |
| `resources` | Compute and memory settings | No |

### Environment Variable Substitution

Use environment variables in configuration:

```yaml
data_source:
  database:
    host: "${DB_HOST:localhost}"        # Default to localhost
    port: "${DB_PORT:5432}"             # Default to 5432
    username: "${DB_USERNAME}"          # Required, no default
    password: "${DB_PASSWORD}"          # Required, no default
```

### Configuration Validation

All configurations are automatically validated:

```bash
# Validate configuration
ml-pipeline validate configs/pipeline_config.yaml

# Strict validation (checks file existence, connectivity)
ml-pipeline validate configs/pipeline_config.yaml --strict
```

## 💾 Data Source Configuration

### CSV Data Sources

```yaml
data_source:
  type: "csv"
  csv_options:
    file_paths:
      - "${DATA_DIR:./data}/transactions.csv"
      - "${DATA_DIR:./data}/historical/*.csv"  # Glob patterns
    separator: ","
    encoding: "utf-8"
    compression: null  # gzip, zip, bz2, xz
    chunk_size: 50000
    header_row: 0
    validate_headers: true
    optimize_dtypes: true
    memory_map: true
    low_memory: false
    
    # Date parsing
    date_columns:
      - "transaction_date"
      - "created_at"
    date_format: "%Y-%m-%d %H:%M:%S"
    
    # Data type mapping
    dtype_mapping:
      customer_id: "str"
      transaction_id: "str"
      amount: "float32"
      is_fraud: "bool"
      merchant_category: "category"
    
    # Performance settings
    parallel_reading: true
    max_workers: 4
    cache_sample: true
```

### Database Data Sources

#### PostgreSQL
```yaml
data_source:
  type: "postgresql"
  database:
    connection:
      host: "${DB_HOST:localhost}"
      port: "${DB_PORT:5432}"
      database: "${DB_NAME:ml_data}"
      username: "${DB_USERNAME}"
      password: "${DB_PASSWORD}"
      schema: "${DB_SCHEMA:public}"
      sslmode: "require"
      pool_size: 10
      max_overflow: 20
    
    extraction:
      query: |
        SELECT 
          customer_id,
          transaction_amount,
          merchant_category,
          is_fraud,
          transaction_date
        FROM transactions 
        WHERE transaction_date >= '${START_DATE:2023-01-01}'
          AND transaction_date <= '${END_DATE:2023-12-31}'
      
      chunk_size: 10000
      cache_data: true
      cache_location: "${CACHE_DIR:./artifacts/cache}"
      encryption: true
```

#### Snowflake
```yaml
data_source:
  type: "snowflake"
  database:
    connection:
      account: "${SNOWFLAKE_ACCOUNT}"
      user: "${SNOWFLAKE_USER}"
      password: "${SNOWFLAKE_PASSWORD}"
      warehouse: "${SNOWFLAKE_WAREHOUSE}"
      database: "${SNOWFLAKE_DATABASE}"
      schema: "${SNOWFLAKE_SCHEMA}"
      role: "${SNOWFLAKE_ROLE}"
    
    extraction:
      query: "SELECT * FROM ML_TRAINING_DATA"
      chunk_size: 100000
```

### Cloud Storage Data Sources

#### AWS S3
```yaml
data_source:
  type: "s3"
  s3_options:
    bucket: "${S3_BUCKET}"
    prefix: "ml-data/"
    access_key_id: "${AWS_ACCESS_KEY_ID}"
    secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    region: "${AWS_DEFAULT_REGION:us-west-2}"
    file_format: "parquet"
```

## 🤖 Model Training Configuration

### Basic Training Setup

```yaml
model_training:
  # Data splitting
  data_split:
    method: "time_based"  # random, stratified, time_based
    train_ratio: 0.7
    validation_ratio: 0.15
    test_ratio: 0.15
    stratify_column: "target_variable"
    time_column: "created_at"
    random_state: 42
  
  # Target variable
  target:
    column: "is_fraud"
    type: "classification"  # classification, regression
    classes: [0, 1]
    # For regression: transform: "log"
  
  # Cross-validation
  cross_validation:
    enabled: true
    method: "stratified_kfold"  # kfold, stratified_kfold, time_series_split
    n_folds: 5
    shuffle: true
    random_state: 42
  
  # Model selection
  model_selection:
    metric: "roc_auc"  # accuracy, precision, recall, f1, roc_auc
    higher_is_better: true
```

### Individual Model Configuration

```yaml
model_training:
  models:
    - name: "xgboost_classifier"
      type: "xgboost.XGBClassifier"
      hyperparameters:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        subsample: 0.8
        colsample_bytree: 0.8
        random_state: 42
      
      # Hyperparameter tuning
      hyperparameter_tuning:
        enabled: true
        method: "bayesian"  # grid_search, random_search, bayesian
        cv_folds: 5
        n_iter: 50
        param_grid:
          n_estimators: [50, 100, 200, 500]
          max_depth: [3, 6, 9, 12]
          learning_rate: [0.01, 0.1, 0.2, 0.3]
          subsample: [0.6, 0.8, 1.0]
    
    - name: "random_forest"
      type: "sklearn.ensemble.RandomForestClassifier"
      hyperparameters:
        n_estimators: 100
        max_depth: 10
        min_samples_split: 2
        min_samples_leaf: 1
        random_state: 42
      
      hyperparameter_tuning:
        enabled: true
        method: "random_search"
        n_iter: 100
        param_grid:
          n_estimators: [50, 100, 200]
          max_depth: [5, 10, 20, null]
          min_samples_split: [2, 5, 10]
          min_samples_leaf: [1, 2, 4]
```

### AutoML Integration

```yaml
model_training:
  # Enable AutoML
  automl_enabled: true
  automl_config_path: "./configs/automl_config.yaml"
  
  # Quick AutoML settings
  automl:
    enabled: true
    algorithms:
      - "logistic_regression"
      - "random_forest"
      - "xgboost"
      - "lightgbm"
      - "catboost"
    time_budget: 3600  # seconds
    optimization_metric: "precision_at_1_percent"
    ensemble_methods: ["voting", "stacking"]
    interpretability_constraint: 0.8
    early_stopping_patience: 10
```

## 🤖 AutoML Configuration

Detailed AutoML configuration for comprehensive automated machine learning:

### Algorithm Selection

```yaml
algorithm_selection:
  # Enable/disable algorithm families
  enable_linear: true
  enable_tree_based: true
  enable_ensemble: true
  enable_neural_networks: true
  enable_naive_bayes: true
  enable_svm: true
  enable_neighbors: true
  
  # Classification algorithms
  classification:
    logistic_regression:
      enabled: true
      max_time: 300
      include_preprocessors: ["no_preprocessing", "standard_scaler", "robust_scaler"]
    
    random_forest:
      enabled: true
      max_time: 600
    
    xgboost:
      enabled: true
      max_time: 900
      use_gpu: false
    
    lightgbm:
      enabled: true
      max_time: 900
      use_gpu: false
    
    neural_networks:
      enabled: true
      max_time: 1200
      architectures: ["mlp", "deep"]
```

### Hyperparameter Search Spaces

```yaml
hyperparameter_spaces:
  search_strategy: "bayesian"  # random, grid, bayesian, hyperband
  n_iterations: 100
  n_initial_points: 10
  acquisition_function: "ei"  # ei, pi, lcb
  
  # Algorithm-specific spaces
  random_forest:
    n_estimators:
      type: "int_uniform"
      low: 10
      high: 500
    max_depth:
      type: "int_uniform"
      low: 1
      high: 20
    min_samples_split:
      type: "int_uniform"
      low: 2
      high: 20
  
  xgboost:
    n_estimators: [100, 300, 500]
    max_depth: [3, 6, 10]
    learning_rate: [0.01, 0.1, 0.3]
    subsample: [0.6, 0.8, 1.0]
    colsample_bytree: [0.6, 0.8, 1.0]
    scale_pos_weight: "auto"  # for imbalanced data
```

### Business Metrics and Optimization

```yaml
business_metrics:
  # Primary business objectives
  primary_objective: "maximize_revenue"  # maximize_revenue, minimize_cost, maximize_f1
  
  # Cost matrix for classification
  cost_matrix:
    binary_classification:
      true_negative_value: 0
      false_positive_cost: 100
      false_negative_cost: 500
      true_positive_value: 1000
  
  # Metric weights
  metric_weights:
    accuracy: 0.2
    precision: 0.3
    recall: 0.3
    f1_score: 0.2
    roc_auc: 0.4
    business_value: 0.6
  
  # ROI calculation
  roi_calculation:
    implementation_cost: 50000
    maintenance_cost_annual: 60000
    false_positive_cost_per_case: 100
    false_negative_cost_per_case: 500
    true_positive_value_per_case: 1000
    expected_cases_per_year: 10000
```

## 🔍 Explainability Configuration

### Global Interpretability

```yaml
explainability:
  enabled: true
  compliance_mode: true
  generate_reports: true
  interactive_dashboard: true
  
  # Global interpretability methods
  global_interpretability:
    # SHAP analysis
    shap:
      enabled: true
      explainer_type: "auto"  # auto, tree, linear, kernel, deep
      sample_size: 1000
      interaction_analysis: true
      plots:
        - "summary_plot"
        - "waterfall_plot"
        - "force_plot"
        - "dependence_plot"
        - "interaction_plot"
    
    # Functional ANOVA
    functional_anova:
      enabled: true
      max_order: 2
      n_permutations: 100
      interaction_strength_threshold: 0.1
    
    # ALE (Accumulated Local Effects) plots
    ale_plots:
      enabled: true
      sample_size: 1000
      features: "auto"  # auto, top_10, all
      n_bins: 20
      center: true
      plot_pdp_comparison: true
    
    # Permutation importance
    permutation_importance:
      enabled: true
      n_repeats: 10
      scoring_metric: "precision_at_k"
      plot_top_n: 20
```

### Local Interpretability

```yaml
explainability:
  # Local interpretability methods
  local_interpretability:
    # LIME analysis
    lime:
      enabled: true
      mode: "tabular"
      sample_size: 100
      n_samples: 5000
      n_features: 10
      discretize_continuous: true
      feature_selection: "auto"
    
    # Anchors explanations
    anchors:
      enabled: true
      threshold: 0.95
      max_anchor_size: 5
      sample_size: 1000
      beam_size: 2
      coverage_samples: 10000
    
    # Counterfactual explanations
    counterfactuals:
      enabled: true
      method: "dice"  # dice, wachter, prototype
      num_cfs: 5
      max_features_changed: 3
      proximity_weight: 0.5
      diversity_weight: 1.0
      actionability_constraints: true
```

### Advanced Interpretability

```yaml
explainability:
  # Advanced interpretability methods
  advanced_interpretability:
    # Trust scores and uncertainty
    trust_scores:
      enabled: true
      k_neighbors: 10
      confidence_threshold: 0.8
      uncertainty_quantification: true
    
    # Prototypes and criticisms
    prototypes:
      enabled: true
      n_prototypes_per_class: 10
      selection_method: "mmcriticism"  # kmeans, random, mmcriticism
      include_criticisms: true
    
    # Concept activation vectors
    concept_activation:
      enabled: true
      n_concepts: 20
      significance_level: 0.05
      concept_sensitivity: true
  
  # Fraud-specific explanations
  fraud_specific:
    reason_codes: true
    narrative_explanations: true
    risk_factors: true
    pattern_detection: true
    regulatory_explanations: true
```

## 📊 Monitoring Configuration

### Performance Monitoring

```yaml
monitoring:
  enabled: true
  monitoring_setup: true
  comprehensive_monitoring: true
  
  # Performance monitoring
  performance_monitoring:
    enabled: true
    baseline_metrics: "training_metrics"
    alert_threshold: 0.05  # 5% degradation
    business_impact_tracking: true
    sla_monitoring: true
    latency_monitoring: true
    throughput_monitoring: true
```

### Data Drift Detection

```yaml
monitoring:
  # Data drift detection
  data_drift:
    enabled: true
    reference_dataset: "training"  # training, validation, custom
    drift_threshold: 0.05
    statistical_tests:
      - "ks_test"
      - "chi2_test"
      - "psi_test"  # Population Stability Index
      - "wasserstein_distance"
      - "jensen_shannon_divergence"
    feature_importance_drift: true
    concept_drift_detection: true
```

### Fairness Monitoring

```yaml
monitoring:
  # Fairness monitoring
  fairness_monitoring:
    enabled: true
    protected_attributes: ["age", "gender", "race", "ethnicity"]
    fairness_metrics: ["demographic_parity", "equalized_odds", "calibration"]
    alert_threshold: 0.8  # 80% rule for disparate impact
    continuous_monitoring: true
    bias_detection: true
```

### Alerting Configuration

```yaml
monitoring:
  # Alert configuration
  alerts:
    enabled: true
    channels:
      - type: "email"
        recipients: ["${ALERT_EMAIL}"]
        severity_levels: ["critical", "warning"]
      - type: "slack"
        webhook_url: "${SLACK_WEBHOOK_URL}"
        severity_levels: ["critical", "warning", "info"]
      - type: "pagerduty"
        service_key: "${PAGERDUTY_SERVICE_KEY}"
        severity_levels: ["critical"]
    
    # Alert thresholds
    alert_thresholds:
      drift_threshold: 0.05
      performance_degradation: 0.05
      fairness_violation: 0.8
      latency_threshold: 100  # ms
      error_rate_threshold: 0.01
    
    # Alert aggregation
    aggregation:
      enabled: true
      time_window: 300  # 5 minutes
      max_alerts_per_window: 5
```

## 🔒 Security Configuration

### Data Security

```yaml
security:
  # Data security
  data_encryption:
    enabled: true
    algorithm: "AES-256"
    key_management: "vault"  # vault, aws_kms, azure_keyvault
  
  # Model security
  model_encryption:
    enabled: true
    digital_signature: true
    version_control: true
  
  # Communication security
  secure_communication:
    tls_enabled: true
    certificate_validation: true
    mutual_tls: false
```

### Access Control

```yaml
security:
  # Access control
  access_control:
    rbac_enabled: true
    authentication: "jwt"  # jwt, oauth2, ldap
    authorization: "attribute_based"  # role_based, attribute_based
    session_management: true
    
    # User roles
    roles:
      data_scientist:
        permissions: ["read_data", "train_models", "view_reports"]
      ml_engineer:
        permissions: ["deploy_models", "manage_infrastructure"]
      auditor:
        permissions: ["view_audit_logs", "read_reports"]
  
  # Audit logging
  audit_logging:
    enabled: true
    level: "detailed"  # basic, detailed, comprehensive
    retention_days: 365
    encryption: true
    immutable_records: true
```

### Compliance

```yaml
security:
  # Compliance frameworks
  compliance:
    frameworks: ["gdpr", "sox", "hipaa", "pci_dss"]
    data_residency: "EU"  # For GDPR compliance
    retention_policy: "7_years"
    
    # Data privacy
    data_privacy:
      anonymization: true
      pii_detection: true
      consent_management: true
      right_to_be_forgotten: true
    
    # Regulatory reporting
    regulatory_reporting:
      enabled: true
      automated_reports: true
      compliance_dashboard: true
```

## 🌍 Environment-Specific Configuration

### Development Environment

```yaml
# environment/development.yaml
pipeline:
  environment: "development"
  log_level: "DEBUG"

data_source:
  csv_options:
    chunk_size: 1000  # Smaller for faster iteration

model_training:
  automl:
    time_budget: 300  # 5 minutes for quick testing

monitoring:
  enabled: false  # Disable monitoring in dev

security:
  data_encryption: false
  audit_logging: false
```

### Production Environment

```yaml
# environment/production.yaml
pipeline:
  environment: "production"
  log_level: "INFO"

data_source:
  csv_options:
    chunk_size: 100000  # Larger for performance

model_training:
  automl:
    time_budget: 7200  # 2 hours for thorough search

monitoring:
  enabled: true
  comprehensive_monitoring: true
  alert_thresholds:
    performance_degradation: 0.02  # Stricter in production

security:
  data_encryption: true
  audit_logging: true
  compliance:
    frameworks: ["gdpr", "sox", "pci_dss"]

resources:
  compute:
    n_jobs: -1  # Use all cores
    memory_limit: "32GB"
```

## 🎯 Best Practices

### Configuration Management

1. **Use Environment Variables**:
   ```yaml
   database:
     host: "${DB_HOST:localhost}"
     password: "${DB_PASSWORD}"  # Never hardcode passwords
   ```

2. **Environment-Specific Overrides**:
   ```bash
   # Load base config + environment overrides
   ml-pipeline train --config configs/pipeline_config.yaml --env production
   ```

3. **Configuration Validation**:
   ```bash
   # Always validate before deployment
   ml-pipeline validate configs/pipeline_config.yaml --strict
   ```

### Security Best Practices

1. **Never Commit Secrets**:
   ```yaml
   # ❌ Bad
   database:
     password: "mysecretpassword"
   
   # ✅ Good
   database:
     password: "${DB_PASSWORD}"
   ```

2. **Use Encrypted Configuration**:
   ```bash
   # Encrypt sensitive configurations
   ansible-vault encrypt configs/production_secrets.yaml
   ```

3. **Implement RBAC**:
   ```yaml
   security:
     access_control:
       rbac_enabled: true
       roles:
         junior_ds: ["read_data", "train_models"]
         senior_ds: ["read_data", "train_models", "deploy_staging"]
   ```

### Performance Optimization

1. **Resource Configuration**:
   ```yaml
   resources:
     compute:
       n_jobs: -1  # Use all cores
       memory_limit: "16GB"
     
     # Enable caching
     caching:
       enabled: true
       cache_location: "/fast/ssd/cache"
   ```

2. **Data Processing Optimization**:
   ```yaml
   data_source:
     csv_options:
       chunk_size: 50000  # Balance memory vs. speed
       parallel_reading: true
       memory_map: true
       optimize_dtypes: true
   ```

3. **Model Training Optimization**:
   ```yaml
   model_training:
     automl:
       early_stopping: true
       ensemble: true  # Often better than individual models
       time_budget: 3600  # Balance thoroughness vs. time
   ```

### Monitoring Best Practices

1. **Comprehensive Monitoring**:
   ```yaml
   monitoring:
     enabled: true
     metrics: ["performance", "drift", "fairness", "business"]
     alert_channels: ["email", "slack", "pagerduty"]
   ```

2. **Appropriate Alert Thresholds**:
   ```yaml
   monitoring:
     alert_thresholds:
       performance_degradation: 0.05  # 5% degradation
       drift_threshold: 0.02  # 2% distribution change
       fairness_violation: 0.8  # 80% rule
   ```

3. **Business Impact Tracking**:
   ```yaml
   monitoring:
     business_metrics:
       enabled: true
       metrics: ["precision_at_1_percent", "cost_savings", "roi"]
       thresholds:
         precision_at_1_percent: 0.80
         cost_savings_monthly: 50000
   ```

### Configuration Templates

Use templates for different use cases:

```bash
# Basic ML pipeline
ml-pipeline init --config-type basic

# Fraud detection
ml-pipeline init --config-type fraud-detection

# Enterprise setup
ml-pipeline init --config-type enterprise
```

### Version Control

1. **Track Configuration Changes**:
   ```bash
   git add configs/
   git commit -m "Update production config for v2.1 deployment"
   ```

2. **Configuration Branching**:
   ```bash
   # Feature branch for configuration changes
   git checkout -b config/automl-optimization
   # Make changes
   git commit -m "Optimize AutoML hyperparameter spaces"
   ```

3. **Environment Promotion**:
   ```bash
   # Promote from staging to production
   cp configs/environment/staging.yaml configs/environment/production.yaml
   # Review and adjust for production
   ```

---

**Need help with configuration?** Check our [troubleshooting guide](../operations/troubleshooting.md) or [examples](../examples/README.md).