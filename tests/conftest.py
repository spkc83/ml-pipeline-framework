"""
Pytest configuration and shared fixtures for ML Pipeline Framework tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any
import json
import os

# Test data fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.uniform(0, 10, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.randint(1, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # Introduce some missing values
    data['feature_1'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['feature_2'][np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_regression_dataframe():
    """Create a sample DataFrame for regression testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.uniform(0, 10, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.randint(1, 100, n_samples),
        'target': np.random.normal(50, 15, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_imbalanced_dataframe():
    """Create an imbalanced DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.uniform(0, 10, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # Highly imbalanced
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def basic_config():
    """Basic pipeline configuration for testing."""
    return {
        'data_source': {
            'type': 'csv',
            'file_path': '/path/to/data.csv'
        },
        'preprocessing': {
            'target_column': 'target',
            'missing_values': {
                'strategy': 'mean'
            },
            'scaling': {
                'method': 'standard'
            },
            'encoding': {
                'categorical_strategy': 'onehot'
            },
            'data_validation': {
                'enabled': True
            },
            'feature_selection': {
                'enabled': False
            },
            'imbalance_handling': {
                'enabled': False
            }
        },
        'model_training': {
            'algorithm': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': 100,
                'random_state': 42
            },
            'cost_sensitive': {
                'enabled': False
            }
        },
        'evaluation': {
            'test_size': 0.2,
            'primary_metric': 'roc_auc',
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        },
        'execution': {
            'version': 'python',
            'framework': 'sklearn',
            'mode': 'train',
            'random_state': 42,
            'enable_mlflow': False,
            'enable_artifacts': False
        },
        'explainability': {
            'enabled': False,
            'methods': ['shap'],
            'compliance': {
                'enabled': False
            }
        }
    }

@pytest.fixture
def spark_config():
    """Spark-specific configuration for testing."""
    return {
        'data_source': {
            'type': 'parquet',
            'file_path': '/path/to/data.parquet'
        },
        'preprocessing': {
            'target_column': 'target',
            'missing_values': {
                'strategy': 'mean'
            },
            'scaling': {
                'method': 'standard'
            }
        },
        'model_training': {
            'algorithm': 'RandomForestClassifier',
            'parameters': {
                'numTrees': 100,
                'seed': 42
            }
        },
        'execution': {
            'version': 'pyspark',
            'framework': 'sparkml',
            'mode': 'train'
        },
        'spark': {
            'app_name': 'ML Pipeline Test',
            'master': 'local[2]',
            'config': {
                'spark.sql.adaptive.enabled': 'true',
                'spark.sql.adaptive.coalescePartitions.enabled': 'true'
            }
        }
    }

@pytest.fixture
def mlflow_config():
    """MLflow configuration for testing."""
    return {
        'mlflow': {
            'tracking_uri': 'file:///tmp/mlruns',
            'experiment_name': 'test_experiment',
            'run_name': 'test_run'
        },
        'execution': {
            'enable_mlflow': True
        }
    }

@pytest.fixture
def artifacts_config():
    """Artifacts configuration for testing."""
    return {
        'artifacts': {
            'storage': {
                'type': 'local',
                'path': '/tmp/test_artifacts'
            }
        },
        'execution': {
            'enable_artifacts': True
        }
    }

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        (1, 'test_value_1', 0.5, 0),
        (2, 'test_value_2', 1.5, 1),
        (3, 'test_value_3', 2.5, 0)
    ]
    mock_cursor.description = [
        ('id',), ('feature_1',), ('feature_2',), ('target',)
    ]
    return mock_conn

@pytest.fixture
def mock_sklearn_model():
    """Mock scikit-learn model for testing."""
    mock_model = MagicMock()
    mock_model.fit.return_value = mock_model
    mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
    mock_model.predict_proba.return_value = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]
    ])
    mock_model.score.return_value = 0.85
    mock_model.get_params.return_value = {'n_estimators': 100, 'random_state': 42}
    return mock_model

@pytest.fixture
def mock_h2o_model():
    """Mock H2O model for testing."""
    mock_model = MagicMock()
    mock_model.train.return_value = None
    mock_model.predict.return_value = MagicMock()
    mock_model.model_performance.return_value = MagicMock()
    return mock_model

@pytest.fixture
def sample_metrics():
    """Sample evaluation metrics for testing."""
    return {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.78,
        'f1_score': 0.80,
        'roc_auc': 0.88,
        'pr_auc': 0.75,
        'brier_score': 0.15,
        'ks_statistic': 0.55,
        'confusion_matrix': [[450, 50], [75, 425]]
    }

@pytest.fixture
def sample_shap_values():
    """Sample SHAP values for testing."""
    np.random.seed(42)
    n_samples, n_features = 100, 5
    return {
        'shap_values': np.random.normal(0, 0.5, (n_samples, n_features)),
        'expected_value': 0.3,
        'feature_names': [f'feature_{i}' for i in range(n_features)],
        'base_values': np.full(n_samples, 0.3)
    }

@pytest.fixture
def sample_cost_matrix():
    """Sample cost matrix for testing."""
    return np.array([
        [0, 1],    # Cost of classifying class 0 as 0 or 1
        [5, 0]     # Cost of classifying class 1 as 0 or 1 (FN is expensive)
    ])

# Configuration file fixtures
@pytest.fixture
def config_file(temp_directory, basic_config):
    """Create a temporary configuration file."""
    config_path = Path(temp_directory) / "test_config.yaml"
    
    # Convert to YAML format
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(basic_config, f, default_flow_style=False)
    
    return str(config_path)

@pytest.fixture
def invalid_config_file(temp_directory):
    """Create an invalid configuration file."""
    config_path = Path(temp_directory) / "invalid_config.yaml"
    
    with open(config_path, 'w') as f:
        f.write("invalid: yaml: content: [")
    
    return str(config_path)

# Mock external services
@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('src.utils.mlflow_tracker.mlflow') as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "test_experiment_id"
        yield mock_mlflow

@pytest.fixture
def mock_spark_session():
    """Mock Spark session for testing."""
    mock_spark = MagicMock()
    mock_spark.sparkContext.setLogLevel.return_value = None
    mock_spark.stop.return_value = None
    
    # Mock DataFrame operations
    mock_df = MagicMock()
    mock_df.count.return_value = 1000
    mock_df.columns = ['feature_1', 'feature_2', 'target']
    mock_spark.createDataFrame.return_value = mock_df
    
    return mock_spark

# Environment setup fixtures
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Clean up
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'LOG_LEVEL' in os.environ:
        del os.environ['LOG_LEVEL']

# Parameterized test data
@pytest.fixture(params=['sklearn', 'xgboost', 'lightgbm', 'catboost'])
def framework_name(request):
    """Parameterized fixture for different ML frameworks."""
    return request.param

@pytest.fixture(params=['classification', 'regression'])
def task_type(request):
    """Parameterized fixture for different task types."""
    return request.param

@pytest.fixture(params=['smote', 'adasyn', 'random_over', 'random_under'])
def sampling_strategy(request):
    """Parameterized fixture for different sampling strategies."""
    return request.param

# Performance test fixtures
@pytest.fixture
def large_dataframe():
    """Create a large DataFrame for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        f'feature_{i}': np.random.normal(0, 1, n_samples) 
        for i in range(50)
    }
    data['target'] = np.random.choice([0, 1], n_samples)
    
    return pd.DataFrame(data)

@pytest.fixture
def memory_profiler():
    """Memory profiler for performance tests."""
    try:
        from memory_profiler import profile
        return profile
    except ImportError:
        pytest.skip("memory_profiler not available")

# Database test fixtures
@pytest.fixture
def postgres_config():
    """PostgreSQL configuration for testing."""
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_db',
        'username': 'test_user',
        'password': 'test_password'
    }

@pytest.fixture
def mysql_config():
    """MySQL configuration for testing."""
    return {
        'host': 'localhost',
        'port': 3306,
        'database': 'test_db',
        'username': 'test_user',
        'password': 'test_password'
    }

@pytest.fixture
def snowflake_config():
    """Snowflake configuration for testing."""
    return {
        'account': 'test_account',
        'warehouse': 'test_warehouse',
        'database': 'test_db',
        'schema': 'test_schema',
        'username': 'test_user',
        'password': 'test_password'
    }

# Utility functions for tests
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype=True):
    """Assert that two DataFrames are equal with better error messages."""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    except AssertionError as e:
        print(f"DataFrames are not equal:")
        print(f"First DataFrame shape: {df1.shape}")
        print(f"Second DataFrame shape: {df2.shape}")
        print(f"First DataFrame columns: {df1.columns.tolist()}")
        print(f"Second DataFrame columns: {df2.columns.tolist()}")
        raise e

def assert_metrics_valid(metrics: Dict[str, Any]):
    """Assert that metrics dictionary contains valid values."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert not np.isnan(value), f"Metric {key} is NaN"
            assert not np.isinf(value), f"Metric {key} is infinite"
            if key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
                assert 0 <= value <= 1, f"Metric {key} should be between 0 and 1, got {value}"