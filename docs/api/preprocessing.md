# Preprocessing API

The preprocessing module provides comprehensive data preprocessing capabilities including feature elimination, imbalance handling, data validation, and custom transformations.

## Classes

### DataPreprocessor

```python
class DataPreprocessor
```

Main data preprocessing pipeline that orchestrates all preprocessing operations.

The preprocessor manages the complete data transformation workflow including cleaning, validation, feature engineering, scaling, and imbalance handling. It provides a unified interface for all preprocessing operations with automatic pipeline optimization.

**Attributes:**
- `pipeline` (sklearn.pipeline.Pipeline): Main preprocessing pipeline
- `feature_selector` (BaseFeatureSelector): Feature selection component
- `scaler` (BaseScaler): Feature scaling component
- `validator` (DataValidator): Data validation component
- `imbalance_handler` (ImbalanceHandler): Imbalance handling component
- `is_fitted` (bool): Whether the preprocessor has been fitted

**Example:**
```python
from ml_pipeline_framework.preprocessing import DataPreprocessor

config = {
    'feature_elimination': {'enabled': True, 'method': 'backward'},
    'scaling': {'method': 'standard'},
    'imbalance_handling': {'method': 'smote'},
    'validation': {'enabled': True}
}

preprocessor = DataPreprocessor(config)
X_processed, y_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)
```

#### Methods

##### `__init__(config, random_state=None)`

Initialize data preprocessor with configuration.

**Args:**
- `config` (Dict[str, Any]): Preprocessing configuration dictionary
- `random_state` (int, optional): Random seed for reproducibility

**Raises:**
- `ValueError`: If configuration is invalid
- `ImportError`: If required dependencies are missing

##### `fit(X, y=None)`

Fit the preprocessing pipeline on training data.

**Args:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series, optional): Target variable

**Returns:**
- `DataPreprocessor`: Self for method chaining

**Raises:**
- `ValueError`: If input data format is invalid
- `RuntimeError`: If fitting fails

##### `transform(X)`

Apply fitted preprocessing transformations to data.

**Args:**
- `X` (pd.DataFrame): Feature matrix to transform

**Returns:**
- `pd.DataFrame`: Transformed feature matrix

**Raises:**
- `RuntimeError`: If preprocessor hasn't been fitted
- `ValueError`: If input features don't match training features

##### `fit_transform(X, y=None)`

Fit the preprocessor and transform the data in one step.

**Args:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series, optional): Target variable

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Transformed features and target (if provided)

##### `inverse_transform(X)`

Apply inverse transformations where possible.

**Args:**
- `X` (pd.DataFrame): Transformed feature matrix

**Returns:**
- `pd.DataFrame`: Original feature matrix (approximate)

##### `get_feature_names()`

Get names of features after preprocessing.

**Returns:**
- `List[str]`: Feature names after all transformations

##### `get_preprocessing_report()`

Generate comprehensive preprocessing report.

**Returns:**
- `Dict[str, Any]`: Detailed report of all preprocessing steps and their effects

---

### FeatureEliminator

```python
class FeatureEliminator
```

Iterative backward feature elimination with comprehensive tracking and reporting.

Performs backward elimination by iteratively removing the least important feature and tracking performance changes. Results are logged to Excel files and visualized through various plots.

**Attributes:**
- `estimator` (Any): Scikit-learn compatible estimator for evaluation
- `scoring` (str): Scoring metric for feature evaluation
- `cv` (int): Number of cross-validation folds
- `elimination_steps_` (List[EliminationStep]): History of elimination steps
- `best_features_` (List[str]): Final selected features
- `best_score_` (float): Best cross-validation score achieved
- `feature_rankings_` (Dict[str, int]): Feature importance rankings

**Example:**
```python
from ml_pipeline_framework.preprocessing import FeatureEliminator
from sklearn.ensemble import RandomForestClassifier

eliminator = FeatureEliminator(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    scoring='roc_auc',
    cv=5,
    min_features=10,
    tolerance=0.001,
    early_stopping_rounds=5
)

# Fit and get selected features
X_selected = eliminator.fit_transform(X_train, y_train)

# Generate comprehensive reports
eliminator.export_to_excel('feature_elimination_results.xlsx')
eliminator.plot_elimination_curve(save_path='elimination_curve.png')
eliminator.generate_comprehensive_report('reports/')
```

#### Methods

##### `__init__(estimator, scoring='accuracy', cv=5, min_features=1, tolerance=0.001, max_iterations=None, early_stopping_rounds=None, feature_importance_method='auto', random_state=None, verbose=True)`

Initialize feature eliminator.

**Args:**
- `estimator` (Any): Scikit-learn compatible estimator
- `scoring` (Union[str, Callable]): Scoring metric ('accuracy', 'roc_auc', etc.)
- `cv` (int): Number of cross-validation folds. Defaults to 5
- `min_features` (int): Minimum number of features to retain. Defaults to 1
- `tolerance` (float): Minimum improvement required to continue. Defaults to 0.001
- `max_iterations` (int, optional): Maximum elimination iterations
- `early_stopping_rounds` (int, optional): Stop if no improvement for N rounds
- `feature_importance_method` (str): Method for feature importance ('auto', 'permutation', 'coefficients')
- `random_state` (int, optional): Random seed for reproducibility
- `verbose` (bool): Enable verbose logging. Defaults to True

##### `fit(X, y)`

Perform backward feature elimination.

**Args:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable

**Returns:**
- `FeatureEliminator`: Self for method chaining

**Raises:**
- `ValueError`: If input data is invalid
- `RuntimeError`: If elimination process fails

##### `transform(X)`

Transform data to include only selected features.

**Args:**
- `X` (pd.DataFrame): Feature matrix to transform

**Returns:**
- `pd.DataFrame`: Transformed feature matrix with selected features only

##### `export_to_excel(output_path, include_feature_importance=True, include_cv_details=True)`

Export elimination results to Excel file with multiple sheets.

**Args:**
- `output_path` (Union[str, Path]): Path to save Excel file
- `include_feature_importance` (bool): Include detailed feature importance. Defaults to True
- `include_cv_details` (bool): Include cross-validation details. Defaults to True

**Raises:**
- `RuntimeError`: If eliminator hasn't been fitted
- `IOError`: If file cannot be written

##### `plot_elimination_curve(figsize=(12, 8), save_path=None, show_std=True)`

Plot performance metric vs number of features.

**Args:**
- `figsize` (Tuple[int, int]): Figure size. Defaults to (12, 8)
- `save_path` (Union[str, Path], optional): Path to save plot
- `show_std` (bool): Show standard deviation bands. Defaults to True

**Returns:**
- `matplotlib.figure.Figure`: Elimination curve plot

##### `plot_feature_importance_evolution(top_n=15, figsize=(14, 10), save_path=None)`

Plot evolution of feature importance over iterations.

**Args:**
- `top_n` (int): Number of top features to show. Defaults to 15
- `figsize` (Tuple[int, int]): Figure size. Defaults to (14, 10)
- `save_path` (Union[str, Path], optional): Path to save plot

**Returns:**
- `matplotlib.figure.Figure`: Feature importance evolution plot

##### `generate_comprehensive_report(output_dir, report_name=None)`

Generate comprehensive report with Excel export and all visualizations.

**Args:**
- `output_dir` (Union[str, Path]): Directory to save report files
- `report_name` (str, optional): Base name for report files

**Returns:**
- `Dict[str, str]`: Dictionary mapping report type to file path

---

### ImbalanceHandler

```python
class ImbalanceHandler
```

Handle imbalanced datasets using various sampling and ensemble techniques.

Provides multiple strategies for handling class imbalance including oversampling, undersampling, and ensemble methods. Supports both binary and multiclass scenarios with automatic strategy selection.

**Attributes:**
- `sampling_method` (str): Sampling strategy being used
- `sampler` (Any): Underlying sampler instance
- `original_distribution` (Dict): Original class distribution
- `resampled_distribution` (Dict): Distribution after resampling
- `sampling_strategy` (Union[str, Dict]): Sampling strategy configuration

**Example:**
```python
from ml_pipeline_framework.preprocessing import ImbalanceHandler

# SMOTE oversampling
smote_handler = ImbalanceHandler(
    method='smote',
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

# Random undersampling
undersample_handler = ImbalanceHandler(
    method='random_undersample',
    sampling_strategy=0.5  # Majority:minority ratio
)

# Apply sampling
X_resampled, y_resampled = smote_handler.fit_resample(X_train, y_train)
```

#### Methods

##### `__init__(method='smote', sampling_strategy='auto', random_state=None, **kwargs)`

Initialize imbalance handler.

**Args:**
- `method` (str): Sampling method ('smote', 'adasyn', 'random_oversample', 'random_undersample', 'edited_nearest_neighbors', 'tomek_links')
- `sampling_strategy` (Union[str, float, Dict]): Sampling strategy configuration
- `random_state` (int, optional): Random seed for reproducibility
- `**kwargs`: Additional parameters for specific sampling methods

**Raises:**
- `ValueError`: If method is not supported
- `ImportError`: If required dependencies are missing

##### `fit_resample(X, y)`

Fit the sampler and resample the data.

**Args:**
- `X` (Union[pd.DataFrame, np.ndarray]): Feature matrix
- `y` (Union[pd.Series, np.ndarray]): Target variable

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Resampled features and target

**Raises:**
- `ValueError`: If input data format is invalid
- `RuntimeError`: If resampling fails

##### `get_sampling_report()`

Generate detailed report of sampling effects.

**Returns:**
- `Dict[str, Any]`: Report containing:
  - Original and resampled class distributions
  - Sampling statistics
  - Quality metrics for generated samples

##### `plot_class_distribution(save_path=None)`

Plot class distribution before and after sampling.

**Args:**
- `save_path` (Union[str, Path], optional): Path to save plot

**Returns:**
- `matplotlib.figure.Figure`: Class distribution comparison plot

##### `validate_sampling_quality(X_original, X_resampled, y_original, y_resampled)`

Validate quality of resampled data.

**Args:**
- `X_original` (pd.DataFrame): Original feature matrix
- `X_resampled` (pd.DataFrame): Resampled feature matrix
- `y_original` (pd.Series): Original target variable
- `y_resampled` (pd.Series): Resampled target variable

**Returns:**
- `Dict[str, float]`: Quality metrics including distribution similarity and feature correlation preservation

---

### DataValidator

```python
class DataValidator
```

Comprehensive data quality validation using Great Expectations framework.

Performs extensive data quality checks including missing values, outliers, data types, value ranges, and custom business rules. Integrates with Great Expectations for enterprise-grade data validation.

**Attributes:**
- `expectation_suite` (str): Name of Great Expectations suite
- `context` (great_expectations.DataContext): Great Expectations context
- `validation_results` (Dict): Results of latest validation
- `data_docs_enabled` (bool): Whether data documentation is enabled

**Example:**
```python
from ml_pipeline_framework.preprocessing import DataValidator

validator = DataValidator(
    expectation_suite='customer_data_expectations',
    fail_on_error=False,
    generate_data_docs=True
)

# Create expectations
validator.add_expectation('expect_column_to_exist', column='customer_id')
validator.add_expectation('expect_column_values_to_not_be_null', column='customer_id')
validator.add_expectation('expect_column_values_to_be_between', 
                         column='age', min_value=18, max_value=120)

# Validate data
validation_result = validator.validate(data)
if not validation_result['success']:
    print("Data validation failed!")
    validator.generate_validation_report()
```

#### Methods

##### `__init__(expectation_suite=None, context_root_dir=None, fail_on_error=True, generate_data_docs=True)`

Initialize data validator.

**Args:**
- `expectation_suite` (str, optional): Name of expectation suite to use
- `context_root_dir` (str, optional): Great Expectations context directory
- `fail_on_error` (bool): Raise exception on validation failure. Defaults to True
- `generate_data_docs` (bool): Generate data documentation. Defaults to True

##### `add_expectation(expectation_type, **kwargs)`

Add data expectation to the suite.

**Args:**
- `expectation_type` (str): Type of expectation (e.g., 'expect_column_to_exist')
- `**kwargs`: Expectation-specific parameters

**Example:**
```python
validator.add_expectation('expect_column_values_to_be_unique', column='id')
validator.add_expectation('expect_column_values_to_be_in_set', 
                         column='status', value_set=['active', 'inactive'])
```

##### `validate(data, checkpoint_name=None)`

Validate data against expectations.

**Args:**
- `data` (pd.DataFrame): Data to validate
- `checkpoint_name` (str, optional): Name of checkpoint to use

**Returns:**
- `Dict[str, Any]`: Validation results with success status and detailed results

##### `generate_validation_report(output_path=None)`

Generate HTML validation report.

**Args:**
- `output_path` (str, optional): Path to save report

**Returns:**
- `str`: Path to generated report

##### `get_data_profile(data)`

Generate comprehensive data profile.

**Args:**
- `data` (pd.DataFrame): Data to profile

**Returns:**
- `Dict[str, Any]`: Detailed data profile including statistics, missing values, and distributions

---

### CustomTransformer

```python
class CustomTransformer(BaseEstimator, TransformerMixin)
```

Base class for creating custom feature transformations with sklearn compatibility.

Provides framework for implementing domain-specific transformations that integrate seamlessly with sklearn pipelines and the preprocessing framework.

**Attributes:**
- `transformation_config` (Dict): Configuration for transformations
- `fitted_transformers` (Dict): Fitted transformation objects
- `feature_names_in_` (List[str]): Input feature names
- `feature_names_out_` (List[str]): Output feature names

**Example:**
```python
from ml_pipeline_framework.preprocessing import CustomTransformer

class LogTransformer(CustomTransformer):
    def __init__(self, columns=None, base='e'):
        self.columns = columns
        self.base = base
        super().__init__()
    
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if self.base == 'e':
                X_transformed[col] = np.log1p(X[col])
            elif self.base == 10:
                X_transformed[col] = np.log10(X[col] + 1)
        return X_transformed

# Use in pipeline
log_transformer = LogTransformer(columns=['income', 'amount'])
X_transformed = log_transformer.fit_transform(X)
```

#### Methods

##### `__init__(transformation_config=None)`

Initialize custom transformer.

**Args:**
- `transformation_config` (Dict, optional): Configuration for transformations

##### `fit(X, y=None)`

Fit the transformer on training data.

**Args:**
- `X` (pd.DataFrame): Training data
- `y` (pd.Series, optional): Target variable

**Returns:**
- `CustomTransformer`: Self for method chaining

##### `transform(X)`

Apply transformations to data.

**Args:**
- `X` (pd.DataFrame): Data to transform

**Returns:**
- `pd.DataFrame`: Transformed data

##### `get_feature_names_out(input_features=None)`

Get output feature names.

**Args:**
- `input_features` (List[str], optional): Input feature names

**Returns:**
- `List[str]`: Output feature names

## Configuration Examples

### Complete Preprocessing Configuration

```yaml
preprocessing:
  # Feature elimination
  feature_elimination:
    enabled: true
    method: "backward"
    cv_folds: 5
    min_features: 10
    tolerance: 0.001
    scoring: "roc_auc"
    early_stopping_rounds: 5
    feature_importance_method: "auto"
    
  # Imbalance handling
  imbalance_handling:
    enabled: true
    method: "smote"
    sampling_strategy: "auto"
    k_neighbors: 5
    random_state: 42
    
  # Data validation
  validation:
    enabled: true
    expectation_suite: "ml_data_expectations"
    fail_on_error: false
    generate_data_docs: true
    
  # Feature scaling
  scaling:
    method: "standard"  # standard, minmax, robust, quantile
    feature_range: [0, 1]  # For minmax scaling
    quantile_range: [25.0, 75.0]  # For quantile scaling
    
  # Missing value handling
  missing_values:
    strategy: "auto"  # auto, drop, impute
    numeric_strategy: "median"  # mean, median, mode, constant
    categorical_strategy: "mode"  # mode, constant, drop
    fill_value: null  # For constant strategy
    
  # Outlier detection
  outlier_detection:
    enabled: true
    method: "isolation_forest"  # isolation_forest, lof, elliptic_envelope
    contamination: 0.1
    action: "flag"  # flag, remove, clip
    
  # Feature encoding
  encoding:
    categorical_encoding: "onehot"  # onehot, label, target, ordinal
    handle_unknown: "ignore"
    drop_first: true
    
  # Custom transformations
  custom_transformations:
    log_transform:
      enabled: true
      columns: ["income", "amount"]
      base: "e"
    polynomial_features:
      enabled: false
      degree: 2
      include_bias: false
```

### Feature Elimination Configuration

```yaml
preprocessing:
  feature_elimination:
    enabled: true
    method: "backward"  # backward, forward, rfe, sfm
    cv_folds: 10
    min_features: 5
    max_features: 50
    tolerance: 0.001
    scoring: "f1_weighted"
    early_stopping_rounds: 3
    feature_importance_method: "permutation"
    random_state: 42
    verbose: true
    
    # Advanced options
    sample_size: 10000  # For large datasets
    parallel_jobs: -1
    export_results: true
    export_path: "feature_elimination_results.xlsx"
    generate_plots: true
    plot_directory: "feature_plots/"
```

### Imbalance Handling Configuration

```yaml
preprocessing:
  imbalance_handling:
    enabled: true
    method: "smote"
    
    # SMOTE parameters
    k_neighbors: 5
    sampling_strategy: "auto"  # or specific ratios like {0: 1000, 1: 800}
    
    # Alternative methods
    # method: "adasyn"
    # n_neighbors: 5
    
    # method: "random_oversample"
    # shrinkage: null
    
    # method: "borderline_smote"
    # kind: "borderline-1"
    
    # method: "random_undersample"
    # replacement: false
    
    # Quality validation
    validate_quality: true
    quality_threshold: 0.95
    
    # Reporting
    generate_report: true
    plot_distributions: true
```

## Supported Preprocessing Methods

### Feature Selection Methods

- **Backward Elimination**: Iteratively remove least important features
- **Forward Selection**: Iteratively add most important features  
- **Recursive Feature Elimination (RFE)**: Sklearn's RFE implementation
- **SelectFromModel (SFM)**: Feature selection based on model importance
- **Univariate Selection**: Statistical tests for feature selection
- **Variance Threshold**: Remove low-variance features

### Imbalance Handling Methods

**Oversampling:**
- `smote` - Synthetic Minority Oversampling Technique
- `adasyn` - Adaptive Synthetic Sampling
- `borderline_smote` - Borderline SMOTE variants
- `random_oversample` - Random oversampling with replacement
- `kmeans_smote` - K-means SMOTE

**Undersampling:**
- `random_undersample` - Random undersampling
- `edited_nearest_neighbors` - Edited Nearest Neighbors rule
- `tomek_links` - Tomek links removal
- `cluster_centroids` - Cluster centroids undersampling

**Combination:**
- `smoteenn` - SMOTE + Edited Nearest Neighbors
- `smotetomek` - SMOTE + Tomek links

### Scaling Methods

- `standard` - StandardScaler (mean=0, std=1)
- `minmax` - MinMaxScaler (range [0,1] or custom)
- `robust` - RobustScaler (median and IQR)
- `quantile` - QuantileTransformer (uniform or normal distribution)
- `power` - PowerTransformer (Box-Cox or Yeo-Johnson)
- `normalizer` - Normalizer (unit norm)

### Encoding Methods

- `onehot` - One-hot encoding for categorical variables
- `label` - Label encoding (ordinal integers)
- `target` - Target encoding (mean target per category)
- `ordinal` - Ordinal encoding with custom ordering
- `binary` - Binary encoding
- `hashing` - Feature hashing

### Missing Value Strategies

**Numeric Data:**
- `mean` - Replace with column mean
- `median` - Replace with column median  
- `mode` - Replace with most frequent value
- `constant` - Replace with specified constant
- `knn` - K-nearest neighbors imputation
- `iterative` - Iterative imputation

**Categorical Data:**
- `mode` - Replace with most frequent category
- `constant` - Replace with specified constant
- `new_category` - Create new "missing" category

## Error Handling

The preprocessing module implements robust error handling:

- **Data Format Errors**: Automatic type conversion and validation
- **Missing Data Errors**: Configurable strategies for handling nulls
- **Memory Errors**: Chunked processing for large datasets
- **Configuration Errors**: Comprehensive validation with helpful messages
- **Pipeline Errors**: Graceful handling of transformation failures

## Performance Optimization

- **Parallel Processing**: Multi-core support for CPU-intensive operations
- **Memory Management**: Efficient memory usage for large datasets
- **Caching**: Intelligent caching of expensive computations
- **Chunked Processing**: Process large datasets in manageable chunks
- **Pipeline Optimization**: Automatic optimization of transformation order