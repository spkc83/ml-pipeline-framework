# AutoML Guide

Comprehensive guide to using the Automated Machine Learning (AutoML) capabilities in the ML Pipeline Framework for fraud detection and other use cases.

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Algorithms](#supported-algorithms)
- [Configuration](#configuration)
- [Business Metrics Optimization](#business-metrics-optimization)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Ensemble Methods](#ensemble-methods)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## 🤖 Overview

The AutoML engine automatically selects, trains, and optimizes machine learning models for your specific use case. It's particularly optimized for fraud detection scenarios with imbalanced datasets and business metric requirements.

### Key Features

- **Algorithm Selection**: Automatically tests 6+ algorithms
- **Hyperparameter Optimization**: Bayesian optimization for efficient search
- **Business Metrics**: Optimizes for fraud-specific metrics like precision@k
- **Ensemble Methods**: Combines models for better performance
- **Interpretability**: Maintains explainability constraints
- **Time Management**: Intelligent time budget allocation

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  AutoML Engine  │───▶│ Model Selection │
│                 │    │                 │    │                 │
│ • CSV/Database  │    │ • Algorithm     │    │ • Performance   │
│ • Preprocessing │    │   Selection     │    │ • Interpretabil │
│ • Validation    │    │ • Hyperparam    │    │ • Business Value│
└─────────────────┘    │   Tuning        │    └─────────────────┘
                       │ • Ensembling    │
                       │ • Evaluation    │
                       └─────────────────┘
```

## 🧠 Supported Algorithms

### Linear Models

#### Logistic Regression
- **Best for**: Baseline models, high interpretability
- **Strengths**: Fast training, explainable, stable
- **Fraud Use Case**: Good for simple fraud patterns
- **Configuration**:
  ```yaml
  logistic_regression:
    enabled: true
    max_time: 300
    include_preprocessors: ["standard_scaler", "robust_scaler"]
  ```

#### Ridge Classifier
- **Best for**: Regularized linear models
- **Strengths**: Handles multicollinearity well
- **Configuration**:
  ```yaml
  ridge_classifier:
    enabled: true
    max_time: 180
  ```

### Tree-Based Models

#### Random Forest
- **Best for**: Robust baseline, feature importance
- **Strengths**: Handles mixed data types, built-in feature selection
- **Fraud Use Case**: Excellent for fraud detection patterns
- **Configuration**:
  ```yaml
  random_forest:
    enabled: true
    max_time: 600
    hyperparameters:
      class_weight: ["balanced", "balanced_subsample"]
  ```

#### Extra Trees
- **Best for**: Faster alternative to Random Forest
- **Strengths**: Reduced overfitting, faster training
- **Configuration**:
  ```yaml
  extra_trees:
    enabled: true
    max_time: 600
  ```

### Gradient Boosting Models

#### XGBoost
- **Best for**: High performance, fraud detection
- **Strengths**: Excellent with imbalanced data, fast inference
- **Fraud Use Case**: Top performer for fraud detection
- **Configuration**:
  ```yaml
  xgboost:
    enabled: true
    max_time: 900
    use_gpu: false
    hyperparameters:
      scale_pos_weight: "auto"  # Handles imbalance
  ```

#### LightGBM
- **Best for**: Large datasets, categorical features
- **Strengths**: Fast training, memory efficient, handles categories
- **Configuration**:
  ```yaml
  lightgbm:
    enabled: true
    max_time: 900
    hyperparameters:
      categorical_features: "auto"
      class_weight: ["balanced"]
  ```

#### CatBoost
- **Best for**: Categorical-heavy datasets
- **Strengths**: Automatic categorical handling, robust to overfitting
- **Configuration**:
  ```yaml
  catboost:
    enabled: true
    max_time: 900
    hyperparameters:
      auto_class_weights: "Balanced"
      cat_features: "auto"
  ```

### Neural Networks

#### MLP Classifier
- **Best for**: Complex non-linear patterns
- **Strengths**: Universal approximator, handles complex interactions
- **Configuration**:
  ```yaml
  mlp_classifier:
    enabled: true
    max_time: 1200
    architectures: ["basic", "deep"]
  ```

### Other Algorithms

#### Support Vector Machines
- **Best for**: High-dimensional data, non-linear patterns
- **Configuration**:
  ```yaml
  svm_linear:
    enabled: true
    max_time: 600
  ```

#### Naive Bayes
- **Best for**: Text features, quick baseline
- **Configuration**:
  ```yaml
  gaussian_nb:
    enabled: true
    max_time: 60
  ```

## ⚙️ Configuration

### Basic AutoML Configuration

```yaml
# configs/pipeline_config.yaml
model_training:
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
    time_budget: 3600  # 1 hour
    optimization_metric: "precision_at_1_percent"
    ensemble_methods: ["voting", "stacking"]
    interpretability_constraint: 0.8
    early_stopping_patience: 10
```

### Detailed AutoML Configuration

```yaml
# configs/automl_config.yaml
automl_settings:
  name: "fraud-detection-automl"
  version: "2.0.0"
  description: "Fraud detection AutoML with business metric optimization"
  random_state: 42
  
  # Search strategy
  search_strategy: "bayesian"  # grid, random, bayesian, adaptive
  
  # General settings
  general:
    max_total_time: 3600  # Total time budget in seconds
    max_eval_time: 300    # Maximum time per model evaluation
    ensemble: true        # Enable ensemble methods
    stack_models: true    # Enable model stacking
    early_stopping: true  # Early stopping for optimization
    n_jobs: -1           # Use all available cores
    
    # Cross-validation
    cv_folds: 5
    cv_method: "stratified_kfold"
    test_size: 0.2
    validation_size: 0.15
```

### Algorithm Selection Configuration

```yaml
algorithm_selection:
  # Enable/disable algorithm families
  enable_linear: true
  enable_tree_based: true
  enable_ensemble: true
  enable_neural_networks: true
  
  # Classification algorithms with time budgets
  classification:
    logistic_regression:
      enabled: true
      max_time: 300
      include_preprocessors: ["no_preprocessing", "standard_scaler"]
    
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
```

## 💰 Business Metrics Optimization

### Fraud Detection Metrics

AutoML can optimize for fraud-specific business metrics:

```yaml
business_metrics:
  # Primary business objective
  primary_objective: "maximize_revenue"
  
  # Cost matrix for fraud detection
  cost_matrix:
    binary_classification:
      true_negative_value: 0     # No cost for correct rejection
      false_positive_cost: 100   # Cost of false alarm ($100)
      false_negative_cost: 500   # Cost of missed fraud ($500)
      true_positive_value: 1000  # Revenue from catching fraud ($1000)
  
  # Metric weights for multi-objective optimization
  metric_weights:
    accuracy: 0.2
    precision: 0.3
    recall: 0.3
    f1_score: 0.2
    roc_auc: 0.4
    business_value: 0.6  # Highest weight on business value
```

### Custom Business Metrics

Define custom metrics for your use case:

```yaml
business_metrics:
  # ROI calculation
  roi_calculation:
    implementation_cost: 50000        # One-time setup cost
    maintenance_cost_annual: 60000    # Annual maintenance
    false_positive_cost_per_case: 100 # Cost per false alarm
    false_negative_cost_per_case: 500 # Cost per missed fraud
    true_positive_value_per_case: 1000 # Value per caught fraud
    expected_cases_per_year: 10000    # Annual transaction volume
  
  # Threshold optimization
  threshold_optimization:
    enabled: true
    optimize_for: "business_value"
    search_range: [0.1, 0.9]
    search_resolution: 0.01
```

### Fraud-Specific Optimization

```yaml
fraud_specific:
  # Fraud detection optimizations
  imbalance_aware: true
  cost_sensitive: true
  threshold_optimization: true
  
  # Business constraints
  regulatory_compliance: ["sr11-7", "gdpr", "fair_lending"]
  explainability_requirements: "high"
  latency_constraints: 100  # milliseconds
  
  # Fraud-specific metrics
  primary_metrics:
    - "precision_at_1_percent"
    - "precision_at_5_percent"
    - "recall_at_fixed_fpr"
    - "expected_value"
  
  # Alert tuning
  false_positive_tolerance: 0.05
  minimum_recall: 0.50
  business_impact_weight: 0.7
```

## 🔍 Hyperparameter Tuning

### Search Strategies

#### Bayesian Optimization (Recommended)
```yaml
hyperparameter_spaces:
  search_strategy: "bayesian"
  n_iterations: 100
  n_initial_points: 10
  acquisition_function: "ei"  # Expected improvement
```

#### Random Search
```yaml
hyperparameter_spaces:
  search_strategy: "random"
  n_iterations: 200
  random_state: 42
```

#### Grid Search
```yaml
hyperparameter_spaces:
  search_strategy: "grid"
  exhaustive: false  # Use selective grid
```

### Algorithm-Specific Hyperparameters

#### XGBoost Hyperparameters
```yaml
xgboost:
  n_estimators: [100, 300, 500]
  max_depth: [3, 6, 10]
  learning_rate: [0.01, 0.1, 0.3]
  subsample: [0.6, 0.8, 1.0]
  colsample_bytree: [0.6, 0.8, 1.0]
  scale_pos_weight: "auto"  # Calculated from data imbalance
  reg_alpha: [0, 0.1, 1.0]
  reg_lambda: [1, 1.5, 2.0]
  gamma: [0, 0.1, 0.5]
```

#### Random Forest Hyperparameters
```yaml
random_forest:
  n_estimators: [100, 200, 500]
  max_depth: [5, 10, 20, null]
  min_samples_split: [2, 10, 20]
  min_samples_leaf: [1, 5, 10]
  max_features: ["sqrt", "log2", 0.5]
  class_weight: ["balanced", "balanced_subsample", null]
```

#### Advanced Hyperparameter Spaces
```yaml
lightgbm:
  n_estimators:
    type: "int_uniform"
    low: 100
    high: 1000
  num_leaves:
    type: "int_uniform"
    low: 10
    high: 300
  learning_rate:
    type: "log_uniform"
    low: 0.01
    high: 0.3
  feature_fraction:
    type: "uniform"
    low: 0.4
    high: 1.0
```

## 🏗️ Ensemble Methods

### Voting Classifiers

Combine multiple models by voting:

```yaml
ensemble_methods:
  voting_classifier:
    enabled: true
    voting: "soft"  # Use predicted probabilities
    weights: null   # Equal weights (auto-calculated)
    max_models: 5   # Limit number of models in ensemble
```

### Stacking

Use meta-learner on top of base models:

```yaml
ensemble_methods:
  stacking_classifier:
    enabled: true
    cv_folds: 5
    meta_learner: "logistic_regression"  # Simple meta-learner
    use_probas: true
    stack_method: "cv"  # Cross-validation predictions
```

### Custom Ensemble Configuration

```yaml
ensemble_methods:
  # Advanced ensemble settings
  ensemble_selection:
    method: "greedy"  # greedy, hillclimb, bestfirst
    max_models: 10
    with_replacement: false
    bagged_ensemble_size: 20
    
  # Ensemble diversity
  diversity_metrics:
    - "disagreement"
    - "correlation"
    - "q_statistic"
  min_diversity: 0.1
```

## 🚀 Usage Examples

### Basic AutoML Training

```python
from src.pipeline_orchestrator import PipelineOrchestrator
from src.utils.config_parser import ConfigParser

# Load configuration
config = ConfigParser.load_config('configs/pipeline_config.yaml')

# Initialize orchestrator
orchestrator = PipelineOrchestrator(config)

# Run AutoML
results = orchestrator.run(mode='automl')

# Get best model
best_model = results.best_model
print(f"Best model: {results.best_model_name}")
print(f"Best score: {results.best_score:.4f}")
```

### Custom AutoML Configuration

```python
# Custom AutoML configuration
automl_config = {
    'automl_settings': {
        'search_strategy': 'bayesian',
        'general': {
            'max_total_time': 1800,  # 30 minutes
            'ensemble': True,
            'n_jobs': -1
        }
    },
    'algorithm_selection': {
        'classification': {
            'xgboost': {'enabled': True, 'max_time': 600},
            'lightgbm': {'enabled': True, 'max_time': 600},
            'random_forest': {'enabled': True, 'max_time': 400}
        }
    },
    'business_metrics': {
        'primary_objective': 'precision_at_1_percent',
        'cost_matrix': {
            'false_positive_cost': 50,
            'false_negative_cost': 1000
        }
    }
}

# Run with custom config
orchestrator = PipelineOrchestrator(config, automl_config=automl_config)
results = orchestrator.run(mode='automl')
```

### Interpreting AutoML Results

```python
# Access detailed results
leaderboard = results.leaderboard
print("Model Leaderboard:")
for i, model_info in enumerate(leaderboard[:5]):
    print(f"{i+1}. {model_info['name']}: {model_info['score']:.4f}")

# Get model details
best_model_info = results.get_model_info(results.best_model_name)
print(f"Best model hyperparameters: {best_model_info['hyperparameters']}")
print(f"Training time: {best_model_info['training_time']:.2f}s")
print(f"Cross-validation score: {best_model_info['cv_score']:.4f}")

# Access ensemble information
if results.ensemble_model:
    print(f"Ensemble composition: {results.ensemble_composition}")
    print(f"Ensemble score: {results.ensemble_score:.4f}")
```

### Fraud Detection Example

```python
# Fraud detection specific AutoML
fraud_config = {
    'data_source': {
        'type': 'csv',
        'csv_options': {
            'file_paths': ['data/credit_card_fraud.csv'],
            'dtype_mapping': {
                'transaction_id': 'str',
                'amount': 'float32',
                'is_fraud': 'int8'
            }
        }
    },
    'model_training': {
        'target': {'column': 'is_fraud', 'type': 'classification'},
        'automl_enabled': True,
        'automl': {
            'algorithms': ['xgboost', 'lightgbm', 'catboost'],
            'time_budget': 3600,
            'optimization_metric': 'precision_at_1_percent'
        }
    },
    'business_metrics': {
        'primary_objective': 'maximize_revenue',
        'cost_matrix': {
            'false_positive_cost': 100,
            'false_negative_cost': 500,
            'true_positive_value': 1000
        }
    }
}

orchestrator = PipelineOrchestrator(fraud_config)
results = orchestrator.run(mode='automl')

# Analyze fraud detection performance
fraud_metrics = results.business_metrics
print(f"Expected daily savings: ${fraud_metrics['expected_daily_savings']:,.2f}")
print(f"False positive rate: {fraud_metrics['false_positive_rate']:.2%}")
print(f"Fraud detection rate: {fraud_metrics['true_positive_rate']:.2%}")
```

## 📊 Best Practices

### 1. Time Budget Planning

```yaml
# Recommended time budgets by use case
quick_prototype:
  time_budget: 600  # 10 minutes
  algorithms: ["xgboost", "lightgbm"]

development:
  time_budget: 1800  # 30 minutes
  algorithms: ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

production:
  time_budget: 7200  # 2 hours
  algorithms: "all"
  ensemble: true
```

### 2. Algorithm Selection Strategy

```python
# Algorithm selection by data characteristics
def select_algorithms_by_data(data_shape, has_categorical, use_case):
    """Select optimal algorithms based on data characteristics."""
    
    algorithms = []
    
    # Always include these robust algorithms
    algorithms.extend(["logistic_regression", "random_forest"])
    
    # Add based on data size
    if data_shape[0] > 100000:  # Large dataset
        algorithms.extend(["lightgbm", "xgboost"])
    
    if data_shape[0] < 10000:   # Small dataset
        algorithms.append("svm_linear")
    
    # Add based on categorical features
    if has_categorical:
        algorithms.append("catboost")
    
    # Add based on use case
    if use_case == "fraud_detection":
        algorithms.extend(["xgboost", "lightgbm", "catboost"])
    
    return algorithms
```

### 3. Fraud Detection Optimization

```yaml
# Fraud-specific best practices
fraud_detection:
  # Preserve natural imbalance
  imbalance_handling:
    strategy: "preserve_natural"
    cost_sensitive_learning: true
  
  # Focus on business metrics
  optimization_metric: "precision_at_1_percent"
  
  # Enable fraud-specific algorithms
  algorithms:
    - "xgboost"      # Excellent with imbalanced data
    - "lightgbm"     # Fast and effective
    - "catboost"     # Good with mixed features
    - "random_forest" # Robust baseline
  
  # Business constraints
  constraints:
    min_recall: 0.5          # Catch at least 50% of fraud
    max_false_positive_rate: 0.05  # Keep false alarms low
```

### 4. Interpretability Balance

```yaml
# Balance performance with interpretability
interpretability_constraint: 0.8  # Minimum interpretability score

# Interpretable algorithm preference
algorithm_weights:
  logistic_regression: 1.2    # Boost interpretable models
  random_forest: 1.1
  xgboost: 1.0
  neural_networks: 0.8        # Penalize black box models
```

### 5. Cross-Validation Strategy

```yaml
# CV strategy by use case
time_series_data:
  cv_method: "time_series_split"
  n_splits: 5

fraud_detection:
  cv_method: "stratified_kfold"  # Preserve class distribution
  n_folds: 5

small_dataset:
  cv_method: "leave_one_out"
```

## 🔬 Advanced Features

### Meta-Learning

Enable meta-learning for faster optimization:

```yaml
advanced:
  meta_learning:
    enabled: true
    use_openml: true
    similarity_threshold: 0.8
    min_datasets: 5
    transfer_learning: true
```

### Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```yaml
advanced:
  multi_objective:
    enabled: true
    objectives: ["business_value", "interpretability", "training_time", "fairness"]
    optimization_method: "nsga2"
    pareto_front_analysis: true
```

### Automated Feature Engineering

Generate features automatically:

```yaml
advanced:
  automated_feature_engineering:
    enabled: true
    max_new_features: 50
    include_polynomial: true
    include_interactions: true
    include_domain_specific: true
    feature_selection_methods: ["mutual_info", "rfe", "lasso"]
```

### Distributed Computing

Scale AutoML across multiple machines:

```yaml
advanced:
  distributed:
    enabled: true
    framework: "dask"  # dask, ray, spark
    n_workers: 4
    memory_per_worker: "8GB"
    cluster_scaling: "auto"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Issues
```yaml
# Reduce memory usage
general:
  max_eval_time: 180  # Reduce per-model time
  chunk_size: 10000   # Process data in smaller chunks

# Use memory-efficient algorithms
algorithms:
  - "logistic_regression"
  - "lightgbm"  # More memory efficient than XGBoost
```

#### 2. Time Budget Exceeded
```python
# Monitor progress
import logging
logging.basicConfig(level=logging.INFO)

# Enable early stopping
automl_config['general']['early_stopping'] = True
automl_config['general']['patience'] = 5
```

#### 3. Poor Performance
```yaml
# Debugging poor performance
debugging:
  verbose: true
  log_level: "DEBUG"
  save_intermediate_results: true
  
# Try different metrics
optimization_metric: "f1_score"  # Instead of accuracy
```

#### 4. Imbalanced Data Issues
```yaml
# Handle severe imbalance
imbalance_handling:
  strategy: "cost_sensitive"
  focal_loss_gamma: 2.0
  
# Adjust algorithms for imbalance
xgboost:
  scale_pos_weight: "auto"
lightgbm:
  class_weight: "balanced"
```

### Performance Tuning

```python
# Performance monitoring
def monitor_automl_performance(orchestrator):
    """Monitor AutoML performance metrics."""
    
    # Track resource usage
    import psutil
    process = psutil.Process()
    
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
    print(f"CPU usage: {process.cpu_percent()}%")
    
    # Track model performance
    results = orchestrator.get_intermediate_results()
    if results:
        print(f"Best score so far: {results.current_best_score:.4f}")
        print(f"Models evaluated: {len(results.evaluated_models)}")
```

### Debugging Configuration

```yaml
# Debug configuration
debugging:
  verbose: true
  log_level: "DEBUG"
  save_all_models: true
  save_intermediate_results: true
  profiling: true
  
# Validation settings
validation:
  cross_validation:
    n_folds: 3  # Reduce for debugging
  holdout_size: 0.3
```

---

**Ready to Get Started?** 🚀

1. **Quick Start**: Use the basic configuration with `time_budget: 600`
2. **Customize**: Adjust algorithms and metrics for your use case
3. **Scale Up**: Increase time budget and enable ensemble methods
4. **Optimize**: Fine-tune hyperparameters and business metrics
5. **Deploy**: Use the best model for production inference

For more examples and advanced usage, check out our [example notebooks](../examples/automl_examples.ipynb) and [API reference](../api/automl.md).