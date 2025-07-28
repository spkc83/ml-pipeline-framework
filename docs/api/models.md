# Models API

The models module provides a unified interface for various machine learning frameworks with automatic hyperparameter tuning and cost-sensitive learning capabilities.

## Classes

### BaseModel

```python
class BaseModel(ABC)
```

Abstract base class for all machine learning models providing a unified interface.

This class defines the contract that all model implementations must follow, ensuring consistency across different ML frameworks. It provides common functionality for training, prediction, model persistence, and performance tracking.

**Attributes:**
- `model_type` (str): Type identifier for the model
- `algorithm` (str): Specific algorithm being used
- `parameters` (Dict[str, Any]): Model hyperparameters
- `is_fitted` (bool): Whether the model has been trained
- `feature_names` (List[str]): Names of features used for training
- `training_time` (float): Time taken for model training
- `memory_usage` (float): Peak memory usage during training

**Example:**
```python
from ml_pipeline_framework.models import SklearnModel

model = SklearnModel(
    algorithm='random_forest',
    parameters={'n_estimators': 100, 'max_depth': 10}
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

#### Methods

##### `fit(X, y, validation_data=None, **kwargs)`

Train the model on the provided data.

**Args:**
- `X` (Union[pd.DataFrame, np.ndarray]): Feature matrix
- `y` (Union[pd.Series, np.ndarray]): Target variable
- `validation_data` (Tuple[np.ndarray, np.ndarray], optional): Validation data for early stopping
- `**kwargs`: Additional training parameters

**Returns:**
- `BaseModel`: Self for method chaining

**Raises:**
- `ValueError`: If input data format is invalid
- `RuntimeError`: If training fails

##### `predict(X)`

Generate predictions for the input data.

**Args:**
- `X` (Union[pd.DataFrame, np.ndarray]): Feature matrix for prediction

**Returns:**
- `np.ndarray`: Predictions array

**Raises:**
- `RuntimeError`: If model hasn't been fitted
- `ValueError`: If input features don't match training features

##### `predict_proba(X)`

Generate prediction probabilities (for classification models).

**Args:**
- `X` (Union[pd.DataFrame, np.ndarray]): Feature matrix for prediction

**Returns:**
- `np.ndarray`: Probability matrix with shape (n_samples, n_classes)

**Raises:**
- `RuntimeError`: If model hasn't been fitted or doesn't support probabilities

##### `get_feature_importance()`

Get feature importance scores.

**Returns:**
- `Dict[str, float]`: Feature importance scores mapped to feature names

##### `save_model(path)`

Save the trained model to disk.

**Args:**
- `path` (Union[str, Path]): Path to save the model

##### `load_model(path)`

Load a previously saved model from disk.

**Args:**
- `path` (Union[str, Path]): Path to the saved model

---

### ModelFactory

```python
class ModelFactory
```

Factory class for creating model instances based on configuration.

The factory provides centralized model creation with automatic parameter validation, dependency checking, and support for custom model registration.

**Example:**
```python
from ml_pipeline_framework.models import ModelFactory

config = {
    'type': 'sklearn',
    'algorithm': 'random_forest',
    'parameters': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}

model = ModelFactory.create_model(config)
```

#### Methods

##### `create_model(config)`

Create model instance from configuration.

**Args:**
- `config` (Dict[str, Any]): Model configuration dictionary

**Returns:**
- `BaseModel`: Configured model instance

**Raises:**
- `ValueError`: If model type is not supported
- `ConfigurationError`: If configuration is invalid

##### `register_model(model_type, model_class)`

Register a custom model type.

**Args:**
- `model_type` (str): Type identifier for the model
- `model_class` (Type[BaseModel]): Model class to register

##### `get_supported_models()`

Get list of supported model types and algorithms.

**Returns:**
- `Dict[str, List[str]]`: Mapping of model types to supported algorithms

---

### SklearnModel

```python
class SklearnModel(BaseModel)
```

Scikit-learn model wrapper with support for all sklearn estimators.

Provides optimized interface for scikit-learn models with automatic pipeline creation, cross-validation, and hyperparameter optimization integration.

**Attributes:**
- `estimator` (sklearn.base.BaseEstimator): The underlying sklearn estimator
- `pipeline` (sklearn.pipeline.Pipeline): Optional preprocessing pipeline
- `cv_results` (Dict): Cross-validation results if performed

**Example:**
```python
from ml_pipeline_framework.models import SklearnModel

# Random Forest Classifier
rf_model = SklearnModel(
    algorithm='random_forest',
    parameters={
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'random_state': 42
    }
)

# Gradient Boosting with early stopping
gb_model = SklearnModel(
    algorithm='gradient_boosting',
    parameters={
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'validation_fraction': 0.2,
        'n_iter_no_change': 10
    }
)
```

#### Methods

##### `__init__(algorithm, parameters=None, pipeline_steps=None, **kwargs)`

Initialize sklearn model wrapper.

**Args:**
- `algorithm` (str): Sklearn algorithm name
- `parameters` (Dict[str, Any], optional): Model hyperparameters
- `pipeline_steps` (List[Tuple], optional): Preprocessing pipeline steps
- `**kwargs`: Additional model configuration

##### `cross_validate(X, y, cv=5, scoring=None, return_train_score=False)`

Perform cross-validation on the model.

**Args:**
- `X` (Union[pd.DataFrame, np.ndarray]): Feature matrix
- `y` (Union[pd.Series, np.ndarray]): Target variable
- `cv` (int or cross-validation generator): Cross-validation strategy
- `scoring` (str or list): Scoring metrics
- `return_train_score` (bool): Whether to return training scores

**Returns:**
- `Dict[str, np.ndarray]`: Cross-validation results

##### `get_params(deep=True)`

Get model parameters.

**Args:**
- `deep` (bool): Return parameters of sub-estimators

**Returns:**
- `Dict[str, Any]`: Model parameters

##### `set_params(**params)`

Set model parameters.

**Args:**
- `**params`: Parameters to set

**Returns:**
- `SklearnModel`: Self for method chaining

---

### SparkMLModel

```python
class SparkMLModel(BaseModel)
```

Apache Spark ML model wrapper for distributed machine learning.

Provides integration with Spark ML for large-scale machine learning with automatic data distribution, parallel processing, and optimized performance for big data scenarios.

**Attributes:**
- `spark_session` (pyspark.sql.SparkSession): Active Spark session
- `ml_pipeline` (pyspark.ml.Pipeline): Spark ML pipeline
- `feature_assembler` (pyspark.ml.feature.VectorAssembler): Feature vector assembler

**Example:**
```python
from ml_pipeline_framework.models import SparkMLModel

spark_model = SparkMLModel(
    algorithm='random_forest',
    parameters={
        'numTrees': 100,
        'maxDepth': 10,
        'subsamplingRate': 0.8
    },
    spark_config={
        'spark.executor.memory': '4g',
        'spark.executor.cores': '2'
    }
)
```

#### Methods

##### `__init__(algorithm, parameters=None, spark_config=None, **kwargs)`

Initialize Spark ML model wrapper.

**Args:**
- `algorithm` (str): Spark ML algorithm name
- `parameters` (Dict[str, Any], optional): Model hyperparameters
- `spark_config` (Dict[str, str], optional): Spark configuration
- `**kwargs`: Additional configuration

##### `fit_distributed(spark_df, target_column, feature_columns=None)`

Train model on Spark DataFrame.

**Args:**
- `spark_df` (pyspark.sql.DataFrame): Training data as Spark DataFrame
- `target_column` (str): Name of target column
- `feature_columns` (List[str], optional): Feature column names

**Returns:**
- `SparkMLModel`: Self for method chaining

##### `transform_distributed(spark_df)`

Apply trained model to Spark DataFrame.

**Args:**
- `spark_df` (pyspark.sql.DataFrame): Data to transform

**Returns:**
- `pyspark.sql.DataFrame`: DataFrame with predictions

---

### H2OModel

```python
class H2OModel(BaseModel)
```

H2O.ai model wrapper with AutoML capabilities.

Provides integration with H2O.ai platform for automated machine learning, including automatic feature engineering, hyperparameter tuning, and model selection.

**Attributes:**
- `h2o_model` (h2o.estimators.BaseEstimator): H2O model instance
- `automl` (h2o.automl.H2OAutoML): AutoML instance if used
- `leaderboard` (h2o.frame.H2OFrame): AutoML leaderboard

**Example:**
```python
from ml_pipeline_framework.models import H2OModel

# Manual model configuration
h2o_model = H2OModel(
    algorithm='gbm',
    parameters={
        'ntrees': 100,
        'max_depth': 6,
        'learn_rate': 0.1
    }
)

# AutoML mode
automl_model = H2OModel(
    algorithm='automl',
    parameters={
        'max_runtime_secs': 3600,
        'max_models': 20,
        'include_algos': ['GBM', 'RF', 'XGBoost']
    }
)
```

#### Methods

##### `fit_automl(X, y, test_data=None, max_runtime_secs=3600, max_models=None)`

Train model using H2O AutoML.

**Args:**
- `X` (Union[pd.DataFrame, h2o.H2OFrame]): Training features
- `y` (Union[pd.Series, str]): Target variable
- `test_data` (Tuple, optional): Test data for validation
- `max_runtime_secs` (int): Maximum runtime for AutoML
- `max_models` (int, optional): Maximum number of models to try

**Returns:**
- `H2OModel`: Self for method chaining

##### `get_leaderboard()`

Get AutoML leaderboard with model performance.

**Returns:**
- `pd.DataFrame`: Leaderboard as pandas DataFrame

##### `explain_model(test_data=None)`

Generate model explanations using H2O's explainability.

**Args:**
- `test_data` (h2o.H2OFrame, optional): Test data for explanations

**Returns:**
- `Dict[str, Any]`: Explanation results

---

### StatsModelsModel

```python
class StatsModelsModel(BaseModel)
```

StatsModels wrapper for statistical modeling and hypothesis testing.

Provides interface for statistical models with detailed statistical output, hypothesis testing, and diagnostic capabilities for econometric and statistical analysis.

**Example:**
```python
from ml_pipeline_framework.models import StatsModelsModel

# Linear regression with statistical output
stats_model = StatsModelsModel(
    algorithm='ols',
    parameters={
        'fit_intercept': True,
        'missing': 'drop'
    }
)

# Logistic regression
logit_model = StatsModelsModel(
    algorithm='logit',
    parameters={
        'method': 'bfgs',
        'maxiter': 100
    }
)
```

#### Methods

##### `get_summary()`

Get detailed statistical summary of the fitted model.

**Returns:**
- `str`: Formatted statistical summary

##### `get_diagnostics()`

Get model diagnostic statistics.

**Returns:**
- `Dict[str, float]`: Diagnostic statistics including R-squared, AIC, BIC

##### `hypothesis_test(hypothesis, alpha=0.05)`

Perform hypothesis testing on model parameters.

**Args:**
- `hypothesis` (str): Hypothesis to test
- `alpha` (float): Significance level

**Returns:**
- `Dict[str, Any]`: Test results including p-value and decision

---

### CostSensitiveModel

```python
class CostSensitiveModel(BaseModel)
```

Wrapper for cost-sensitive learning with custom cost matrices.

Implements various cost-sensitive learning strategies including cost-sensitive training, threshold optimization, and ensemble methods for imbalanced datasets.

**Attributes:**
- `cost_matrix` (np.ndarray): Cost matrix for misclassification
- `base_model` (BaseModel): Underlying base model
- `threshold` (float): Optimal decision threshold
- `cost_sensitive_method` (str): Method used for cost-sensitive learning

**Example:**
```python
from ml_pipeline_framework.models import CostSensitiveModel
import numpy as np

# Define cost matrix (rows: true, cols: predicted)
cost_matrix = np.array([
    [0, 1],    # True negative: 0 cost, False positive: 1 cost
    [10, 0]    # False negative: 10 cost, True positive: 0 cost
])

cost_model = CostSensitiveModel(
    base_model_config={
        'type': 'sklearn',
        'algorithm': 'random_forest',
        'parameters': {'n_estimators': 100}
    },
    cost_matrix=cost_matrix,
    method='threshold_optimization'  # or 'cost_sensitive_training'
)
```

#### Methods

##### `__init__(base_model_config, cost_matrix, method='threshold_optimization', **kwargs)`

Initialize cost-sensitive model.

**Args:**
- `base_model_config` (Dict): Configuration for base model
- `cost_matrix` (np.ndarray): Cost matrix for misclassification
- `method` (str): Cost-sensitive learning method
- `**kwargs`: Additional parameters

##### `optimize_threshold(X_val, y_val)`

Optimize decision threshold based on cost matrix.

**Args:**
- `X_val` (np.ndarray): Validation features
- `y_val` (np.ndarray): Validation targets

**Returns:**
- `float`: Optimal threshold

##### `calculate_cost(y_true, y_pred)`

Calculate total cost based on predictions and cost matrix.

**Args:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels

**Returns:**
- `float`: Total cost

##### `predict_with_threshold(X, threshold=None)`

Make predictions using specified threshold.

**Args:**
- `X` (np.ndarray): Features for prediction
- `threshold` (float, optional): Decision threshold

**Returns:**
- `np.ndarray`: Binary predictions

---

### HyperparameterTuner

```python
class HyperparameterTuner
```

Automated hyperparameter optimization using various search strategies.

Provides unified interface for hyperparameter tuning with support for grid search, random search, Bayesian optimization, and multi-objective optimization.

**Attributes:**
- `search_method` (str): Search strategy being used
- `search_space` (Dict): Hyperparameter search space
- `best_params` (Dict): Best parameters found
- `best_score` (float): Best score achieved
- `optimization_history` (List): History of optimization trials

**Example:**
```python
from ml_pipeline_framework.models import HyperparameterTuner

# Define search space
search_space = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tuner = HyperparameterTuner(
    model_config={
        'type': 'sklearn',
        'algorithm': 'random_forest'
    },
    search_space=search_space,
    search_method='optuna',  # or 'grid', 'random', 'bayes'
    scoring='roc_auc',
    cv_folds=5
)

best_model = tuner.optimize(X_train, y_train, n_trials=100)
```

#### Methods

##### `__init__(model_config, search_space, search_method='optuna', scoring='accuracy', cv_folds=5, **kwargs)`

Initialize hyperparameter tuner.

**Args:**
- `model_config` (Dict): Base model configuration
- `search_space` (Dict): Hyperparameter search space
- `search_method` (str): Search strategy
- `scoring` (str): Optimization metric
- `cv_folds` (int): Cross-validation folds
- `**kwargs`: Additional tuning parameters

##### `optimize(X, y, n_trials=100, timeout=None, callbacks=None)`

Perform hyperparameter optimization.

**Args:**
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training targets
- `n_trials` (int): Maximum number of trials
- `timeout` (int, optional): Maximum optimization time in seconds
- `callbacks` (List, optional): Optimization callbacks

**Returns:**
- `BaseModel`: Best model found

##### `get_optimization_history()`

Get detailed optimization history.

**Returns:**
- `pd.DataFrame`: Optimization trials with parameters and scores

##### `plot_optimization_history(save_path=None)`

Plot optimization history and parameter importance.

**Args:**
- `save_path` (str, optional): Path to save plots

**Returns:**
- `matplotlib.figure.Figure`: Optimization plots

##### `get_best_params()`

Get best hyperparameters found.

**Returns:**
- `Dict[str, Any]`: Best hyperparameters

##### `cross_validate_best(X, y, cv=None)`

Cross-validate the best model found.

**Args:**
- `X` (np.ndarray): Features for validation
- `y` (np.ndarray): Targets for validation
- `cv` (int or CV generator, optional): Cross-validation strategy

**Returns:**
- `Dict[str, np.ndarray]`: Cross-validation results

## Configuration Examples

### Scikit-learn Models

```yaml
model:
  type: "sklearn"
  algorithm: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: "sqrt"
    random_state: 42
    n_jobs: -1
```

### Spark ML Models

```yaml
model:
  type: "sparkml"
  algorithm: "random_forest"
  parameters:
    numTrees: 100
    maxDepth: 10
    subsamplingRate: 0.8
    featureSubsetStrategy: "sqrt"
  spark_config:
    spark.executor.memory: "4g"
    spark.executor.cores: "2"
    spark.sql.adaptive.enabled: "true"
```

### H2O AutoML

```yaml
model:
  type: "h2o"
  algorithm: "automl"
  parameters:
    max_runtime_secs: 3600
    max_models: 20
    include_algos: ["GBM", "RF", "XGBoost", "DeepLearning"]
    exclude_algos: ["GLM"]
    balance_classes: true
    max_after_balance_size: 5.0
```

### Cost-Sensitive Learning

```yaml
model:
  type: "cost_sensitive"
  base_model:
    type: "sklearn"
    algorithm: "random_forest"
    parameters:
      n_estimators: 100
  cost_matrix: [[0, 1], [10, 0]]  # FN cost = 10, FP cost = 1
  method: "threshold_optimization"
```

### Hyperparameter Tuning

```yaml
tuning:
  enabled: true
  method: "optuna"
  n_trials: 100
  optimization_direction: "maximize"
  scoring: "roc_auc"
  search_space:
    n_estimators: [50, 100, 200, 500]
    max_depth: [3, 5, 7, 10, null]
    min_samples_split: [2, 5, 10]
    learning_rate: [0.01, 0.1, 0.2]  # For gradient boosting
```

## Supported Algorithms

### Scikit-learn Algorithms

**Classification:**
- `logistic_regression`
- `random_forest`
- `gradient_boosting`
- `svm`
- `naive_bayes`
- `knn`
- `decision_tree`
- `extra_trees`
- `ada_boost`

**Regression:**
- `linear_regression`
- `ridge`
- `lasso`
- `elastic_net`
- `random_forest_regressor`
- `gradient_boosting_regressor`
- `svr`
- `decision_tree_regressor`

### Spark ML Algorithms

**Classification:**
- `logistic_regression`
- `random_forest`
- `gradient_boosted_trees`
- `decision_tree`
- `naive_bayes`
- `multilayer_perceptron`

**Regression:**
- `linear_regression`
- `random_forest_regressor`
- `gradient_boosted_trees_regressor`
- `decision_tree_regressor`

### H2O Algorithms

- `automl` - Automated machine learning
- `gbm` - Gradient Boosting Machine
- `rf` - Random Forest
- `xgboost` - XGBoost
- `glm` - Generalized Linear Models
- `deeplearning` - Deep Neural Networks
- `naive_bayes` - Naive Bayes

### StatsModels Algorithms

- `ols` - Ordinary Least Squares
- `logit` - Logistic Regression
- `probit` - Probit Regression
- `arima` - ARIMA Time Series
- `var` - Vector Autoregression

## Error Handling

The models module implements comprehensive error handling:

- **Configuration Errors**: Invalid model parameters or missing dependencies
- **Training Errors**: Data incompatibility or convergence issues
- **Prediction Errors**: Feature mismatch or model not fitted
- **Resource Errors**: Memory or computational limitations

## Performance Considerations

- **Memory Management**: Automatic memory optimization for large datasets
- **Parallel Processing**: Multi-core support for supported algorithms
- **Model Persistence**: Efficient model serialization and loading
- **Batch Processing**: Optimized prediction for large datasets
- **GPU Support**: Acceleration where available (H2O, some sklearn algorithms)