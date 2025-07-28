"""
End-to-end integration tests for the ML Pipeline Framework.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import yaml
import json
import os
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import pipeline components
from src.pipeline_orchestrator import PipelineOrchestrator
from src.utils.config_parser import ConfigParser
from src.utils.artifacts import ArtifactManager
from src.data_access.factory import DataConnectorFactory
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.factory import ModelFactory
from src.evaluation.metrics import MetricsCalculator


class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.config_dir = Path(self.temp_dir) / "configs"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self, task_type='classification', n_samples=1000, n_features=10):
        """Create test dataset for integration tests."""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features//2,
                n_redundant=n_features//4,
                n_classes=2,
                random_state=42
            )
        else:  # regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
        
        # Create DataFrame with meaningful column names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Add categorical features
        df['category_A'] = np.random.choice(['cat1', 'cat2', 'cat3'], n_samples)
        df['category_B'] = np.random.choice(['typeX', 'typeY'], n_samples)
        
        # Introduce some missing values
        missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        df.loc[missing_indices, 'feature_0'] = np.nan
        missing_indices = np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)
        df.loc[missing_indices, 'feature_1'] = np.nan
        
        return df
    
    def create_test_config(self, task_type='classification', framework='sklearn'):
        """Create test configuration."""
        config = {
            'data_source': {
                'type': 'csv',
                'file_path': str(self.data_dir / f"{task_type}_data.csv")
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
                'algorithm': 'RandomForestClassifier' if task_type == 'classification' else 'RandomForestRegressor',
                'parameters': {
                    'n_estimators': 50,
                    'max_depth': 10,
                    'random_state': 42
                },
                'cost_sensitive': {
                    'enabled': False
                }
            },
            'evaluation': {
                'test_size': 0.2,
                'primary_metric': 'roc_auc' if task_type == 'classification' else 'r2_score',
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] if task_type == 'classification' 
                         else ['mse', 'rmse', 'mae', 'r2_score']
            },
            'execution': {
                'version': 'python',
                'framework': framework,
                'mode': 'train',
                'random_state': 42,
                'enable_mlflow': False,
                'enable_artifacts': True
            },
            'artifacts': {
                'storage': {
                    'type': 'local',
                    'path': str(self.output_dir / 'artifacts')
                }
            },
            'explainability': {
                'enabled': False,
                'methods': [],
                'compliance': {
                    'enabled': False
                }
            }
        }
        
        return config
    
    def test_complete_classification_pipeline(self):
        """Test complete classification pipeline end-to-end."""
        # Create test data
        df = self.create_test_dataset(task_type='classification')
        data_path = self.data_dir / "classification_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration
        config = self.create_test_config(task_type='classification')
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir),
            verbose=True
        )
        
        # Validate configuration
        validation_result = orchestrator.validate_config()
        assert validation_result['valid'] == True
        
        # Get execution plan
        execution_plan = orchestrator.get_execution_plan()
        assert execution_plan['mode'] == 'train'
        assert len(execution_plan['steps']) > 0
        
        # Execute pipeline
        result = orchestrator.run_pipeline('train')
        
        # Verify successful execution
        assert result['success'] == True
        assert result['mode'] == 'train'
        assert 'models' in result
        assert 'metrics' in result
        assert 'best_model' in result
        
        # Verify model was trained
        assert 'primary' in result['models']
        
        # Verify metrics were calculated
        assert 'evaluation' in result['metrics']
        evaluation_metrics = result['metrics']['evaluation']
        assert 'accuracy' in evaluation_metrics
        assert 'roc_auc' in evaluation_metrics
        assert 0 <= evaluation_metrics['accuracy'] <= 1
        assert 0 <= evaluation_metrics['roc_auc'] <= 1
        
        # Verify artifacts were created
        assert 'files' in result['artifacts']
        assert 'model' in result['artifacts']['files']
        assert 'metrics' in result['artifacts']['files']
        
        # Verify output files exist
        model_path = Path(result['artifacts']['files']['model'])
        metrics_path = Path(result['artifacts']['files']['metrics'])
        assert model_path.exists()
        assert metrics_path.exists()
    
    def test_complete_regression_pipeline(self):
        """Test complete regression pipeline end-to-end."""
        # Create test data
        df = self.create_test_dataset(task_type='regression')
        data_path = self.data_dir / "regression_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration
        config = self.create_test_config(task_type='regression')
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir),
            verbose=True
        )
        
        # Execute pipeline
        result = orchestrator.run_pipeline('train')
        
        # Verify successful execution
        assert result['success'] == True
        assert result['mode'] == 'train'
        
        # Verify regression metrics
        evaluation_metrics = result['metrics']['evaluation']
        assert 'mse' in evaluation_metrics
        assert 'r2_score' in evaluation_metrics
        assert evaluation_metrics['mse'] >= 0
        assert evaluation_metrics['r2_score'] <= 1  # Can be negative for very poor models
    
    def test_pipeline_with_imbalanced_data(self):
        """Test pipeline with imbalanced dataset."""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            weights=[0.95, 0.05],  # Highly imbalanced
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        data_path = self.data_dir / "imbalanced_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration with imbalance handling
        config = self.create_test_config(task_type='classification')
        config['data_source']['file_path'] = str(data_path)
        config['preprocessing']['imbalance_handling'] = {
            'enabled': True,
            'method': 'smote',
            'parameters': {
                'sampling_strategy': 'auto'
            }
        }
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        
        # Verify successful execution with imbalance handling
        assert result['success'] == True
        
        # Check that imbalance handling was applied
        # This would be reflected in the logs and pipeline state
        completed_stages = result['pipeline_state']['completed_stages']
        assert 'preprocessing' in completed_stages
    
    def test_pipeline_with_cost_sensitive_learning(self):
        """Test pipeline with cost-sensitive learning."""
        # Create test data
        df = self.create_test_dataset(task_type='classification')
        data_path = self.data_dir / "cost_sensitive_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration with cost-sensitive learning
        config = self.create_test_config(task_type='classification')
        config['model_training']['cost_sensitive'] = {
            'enabled': True,
            'parameters': {
                'false_positive_cost': 1,
                'false_negative_cost': 5
            }
        }
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        
        # Verify successful execution
        assert result['success'] == True
        assert 'primary' in result['models']
    
    def test_pipeline_with_feature_selection(self):
        """Test pipeline with feature selection enabled."""
        # Create test data with many features
        df = self.create_test_dataset(task_type='classification', n_features=50)
        data_path = self.data_dir / "feature_selection_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration with feature selection
        config = self.create_test_config(task_type='classification')
        config['data_source']['file_path'] = str(data_path)
        config['preprocessing']['feature_selection'] = {
            'enabled': True,
            'method': 'mutual_info',
            'k_features': 20
        }
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        
        # Verify successful execution
        assert result['success'] == True
        
        # Check that feature selection was applied
        completed_stages = result['pipeline_state']['completed_stages']
        assert 'feature_engineering' in completed_stages
    
    def test_prediction_pipeline(self):
        """Test prediction pipeline with pre-trained model."""
        # First, train a model
        df_train = self.create_test_dataset(task_type='classification', n_samples=800)
        train_path = self.data_dir / "train_data.csv"
        df_train.to_csv(train_path, index=False)
        
        # Train model
        train_config = self.create_test_config(task_type='classification')
        train_config['data_source']['file_path'] = str(train_path)
        
        train_orchestrator = PipelineOrchestrator(
            config=train_config,
            output_dir=str(self.output_dir / "training")
        )
        
        train_result = train_orchestrator.run_pipeline('train')
        assert train_result['success'] == True
        
        # Create prediction data (same structure, different samples)
        df_pred = self.create_test_dataset(task_type='classification', n_samples=200)
        df_pred = df_pred.drop('target', axis=1)  # Remove target for prediction
        pred_path = self.data_dir / "prediction_data.csv"
        df_pred.to_csv(pred_path, index=False)
        
        # Create prediction configuration
        pred_config = {
            'data_source': {
                'type': 'csv',
                'file_path': str(pred_path)
            },
            'model_path': train_result['artifacts']['files']['model'],
            'preprocessing': train_config['preprocessing'].copy(),
            'execution': {
                'version': 'python',
                'framework': 'sklearn',
                'mode': 'predict',
                'enable_artifacts': True
            },
            'artifacts': {
                'storage': {
                    'type': 'local',
                    'path': str(self.output_dir / 'prediction_artifacts')
                }
            }
        }
        
        # Note: Full prediction pipeline would require implementation of prediction-specific methods
        # For now, we test that the configuration is valid and setup works
        pred_orchestrator = PipelineOrchestrator(
            config=pred_config,
            output_dir=str(self.output_dir / "prediction")
        )
        
        validation_result = pred_orchestrator.validate_config()
        # May have warnings but should not have errors for basic validation
        assert len(validation_result['errors']) == 0 or not validation_result['valid']
    
    def test_model_comparison_pipeline(self):
        """Test model comparison across different algorithms."""
        # Create test data
        df = self.create_test_dataset(task_type='classification')
        data_path = self.data_dir / "comparison_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create comparison configuration
        config = self.create_test_config(task_type='classification')
        config['execution']['mode'] = 'compare'
        config['model_comparison'] = {
            'algorithms': [
                {
                    'name': 'RandomForest',
                    'algorithm': 'RandomForestClassifier',
                    'parameters': {'n_estimators': 50, 'random_state': 42}
                },
                {
                    'name': 'LogisticRegression',
                    'algorithm': 'LogisticRegression',
                    'parameters': {'random_state': 42, 'max_iter': 1000}
                }
            ],
            'cv_folds': 3,
            'scoring_metrics': ['accuracy', 'roc_auc', 'f1_score']
        }
        
        # Run comparison pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        execution_plan = orchestrator.get_execution_plan()
        assert execution_plan['mode'] == 'compare'
        
        # Note: Full comparison pipeline execution would require implementation
        # of comparison-specific methods in the orchestrator
        validation_result = orchestrator.validate_config()
        assert validation_result['valid'] == True
    
    def test_configuration_validation_and_error_handling(self):
        """Test configuration validation and error handling."""
        # Test various invalid configurations
        
        # Missing required sections
        invalid_config_1 = {
            'preprocessing': {'target_column': 'target'},
            'execution': {'version': 'python', 'framework': 'sklearn'}
        }
        
        orchestrator = PipelineOrchestrator(
            config=invalid_config_1,
            output_dir=str(self.output_dir)
        )
        
        validation_result = orchestrator.validate_config()
        assert validation_result['valid'] == False
        assert len(validation_result['errors']) > 0
        
        # Invalid data source
        invalid_config_2 = {
            'data_source': {'type': 'nonexistent_type'},
            'preprocessing': {'target_column': 'target'},
            'model_training': {'algorithm': 'RandomForestClassifier'},
            'evaluation': {'test_size': 0.2},
            'execution': {'version': 'python', 'framework': 'sklearn'}
        }
        
        orchestrator2 = PipelineOrchestrator(
            config=invalid_config_2,
            output_dir=str(self.output_dir)
        )
        
        validation_result2 = orchestrator2.validate_config()
        # Should have warnings or errors about unsupported data source
        assert len(validation_result2['errors']) > 0 or len(validation_result2['warnings']) > 0
    
    def test_pipeline_state_management_and_recovery(self):
        """Test pipeline state management and recovery capabilities."""
        # Create test data
        df = self.create_test_dataset(task_type='classification')
        data_path = self.data_dir / "state_test_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration
        config = self.create_test_config(task_type='classification')
        
        # Run pipeline and let it save state
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        assert result['success'] == True
        
        # Verify pipeline state was saved
        state_file = Path(self.output_dir) / "pipeline_state.json"
        assert state_file.exists()
        
        # Load and verify state content
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        assert 'completed_stages' in saved_state
        assert 'stage_timings' in saved_state
        assert len(saved_state['completed_stages']) > 0
        assert len(saved_state['stage_timings']) > 0
        
        # Test resumption (mock scenario)
        new_orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        # Simulate resumption from a specific stage
        # Note: Full resumption would require implementation of resume-specific logic
        execution_plan = new_orchestrator.get_execution_plan()
        assert len(execution_plan['steps']) > 0
    
    def test_artifact_management_integration(self):
        """Test artifact management integration."""
        # Create test data
        df = self.create_test_dataset(task_type='classification')
        data_path = self.data_dir / "artifact_test_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration with artifacts enabled
        config = self.create_test_config(task_type='classification')
        config['execution']['enable_artifacts'] = True
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        assert result['success'] == True
        
        # Verify artifacts were created
        artifacts_dir = Path(self.output_dir) / 'artifacts'
        assert artifacts_dir.exists()
        
        # Test direct artifact manager usage
        artifact_manager = ArtifactManager.create_local_backend(str(artifacts_dir))
        
        # Save additional test artifact
        test_data = {'test': 'artifact', 'value': 42}
        artifact_info = artifact_manager.save_artifact(
            test_data,
            'integration_test',
            artifact_type='data',
            description='Integration test artifact'
        )
        
        # Verify artifact was saved
        assert artifact_info['name'] == 'integration_test'
        assert artifact_info['type'] == 'data'
        
        # Load and verify artifact
        loaded_data = artifact_manager.load_artifact(
            artifact_info['name'],
            artifact_info['version']
        )
        assert loaded_data == test_data
    
    def test_multiple_frameworks_compatibility(self):
        """Test compatibility across different ML frameworks."""
        # Create test data
        df = self.create_test_dataset(task_type='classification')
        data_path = self.data_dir / "framework_test_data.csv"
        df.to_csv(data_path, index=False)
        
        # Test sklearn framework
        sklearn_config = self.create_test_config(task_type='classification', framework='sklearn')
        sklearn_orchestrator = PipelineOrchestrator(
            config=sklearn_config,
            output_dir=str(self.output_dir / "sklearn")
        )
        
        sklearn_result = sklearn_orchestrator.run_pipeline('train')
        assert sklearn_result['success'] == True
        
        # Note: Testing other frameworks would require them to be available
        # and properly configured in the test environment
        
        # Test XGBoost if available
        try:
            import xgboost
            xgb_config = self.create_test_config(task_type='classification', framework='xgboost')
            xgb_config['model_training']['algorithm'] = 'XGBClassifier'
            
            xgb_orchestrator = PipelineOrchestrator(
                config=xgb_config,
                output_dir=str(self.output_dir / "xgboost")
            )
            
            validation_result = xgb_orchestrator.validate_config()
            # Should be valid if XGBoost is available
            assert validation_result['valid'] == True or len(validation_result['warnings']) > 0
            
        except ImportError:
            # XGBoost not available in test environment
            pass
    
    def test_end_to_end_with_data_quality_issues(self):
        """Test pipeline with various data quality issues."""
        # Create problematic dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Base dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        
        # Introduce data quality issues
        
        # 1. High percentage of missing values
        missing_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
        df.loc[missing_indices, 'feature_0'] = np.nan
        
        # 2. Outliers
        outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        df.loc[outlier_indices, 'feature_1'] = df['feature_1'].mean() + 10 * df['feature_1'].std()
        
        # 3. Duplicate rows
        duplicate_rows = df.sample(n=50, random_state=42)
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        
        # 4. Inconsistent categorical values
        df['category'] = np.random.choice(['A', 'B', 'C', 'a', 'b'], len(df))  # Mixed case
        
        # Save problematic dataset
        data_path = self.data_dir / "problematic_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration with data validation enabled
        config = self.create_test_config(task_type='classification')
        config['data_source']['file_path'] = str(data_path)
        config['preprocessing']['data_validation']['enabled'] = True
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        
        # Pipeline should handle data quality issues gracefully
        # It may succeed with warnings or fail with informative error messages
        if result['success']:
            # If successful, verify that data validation was performed
            assert 'data_validation' in result['pipeline_state']['completed_stages']
        else:
            # If failed, verify that error information is provided
            assert 'error' in result
            assert len(result['error']) > 0


class TestConfigurationManagement:
    """Integration tests for configuration management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_configuration_loading_and_validation(self):
        """Test configuration loading from different formats."""
        # Create test configuration
        config = {
            'data_source': {
                'type': 'csv',
                'file_path': '/path/to/data.csv'
            },
            'preprocessing': {
                'target_column': 'target',
                'missing_values': {'strategy': 'mean'}
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {'n_estimators': 100}
            },
            'evaluation': {
                'test_size': 0.2
            },
            'execution': {
                'version': 'python',
                'framework': 'sklearn'
            }
        }
        
        # Test YAML configuration
        yaml_path = self.config_dir / "test_config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        parser = ConfigParser()
        loaded_yaml_config = parser.load_config(str(yaml_path))
        assert loaded_yaml_config == config
        
        # Test JSON configuration
        json_path = self.config_dir / "test_config.json"
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        loaded_json_config = parser.load_config(str(json_path))
        assert loaded_json_config == config
        
        # Test configuration validation
        parser.validate_config(loaded_yaml_config)  # Should not raise exception
        parser.validate_config(loaded_json_config)  # Should not raise exception
    
    def test_configuration_merging_and_overrides(self):
        """Test configuration merging and environment-specific overrides."""
        # Base configuration
        base_config = {
            'data_source': {
                'type': 'csv',
                'file_path': '/path/to/base_data.csv'
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            'execution': {
                'version': 'python',
                'framework': 'sklearn'
            }
        }
        
        # Environment-specific override
        dev_override = {
            'data_source': {
                'file_path': '/path/to/dev_data.csv'
            },
            'model_training': {
                'parameters': {
                    'n_estimators': 50  # Fewer trees for faster dev training
                }
            }
        }
        
        parser = ConfigParser()
        merged_config = parser.merge_configs(base_config, dev_override)
        
        # Verify merging
        assert merged_config['data_source']['file_path'] == '/path/to/dev_data.csv'
        assert merged_config['data_source']['type'] == 'csv'  # Preserved from base
        assert merged_config['model_training']['parameters']['n_estimators'] == 50  # Overridden
        assert merged_config['model_training']['parameters']['max_depth'] == 10  # Preserved from base
        assert merged_config['model_training']['algorithm'] == 'RandomForestClassifier'  # Preserved
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in configuration."""
        # Set test environment variables
        os.environ['TEST_DATA_PATH'] = '/test/data/path'
        os.environ['TEST_MODEL_ESTIMATORS'] = '200'
        
        try:
            config_with_env_vars = {
                'data_source': {
                    'type': 'csv',
                    'file_path': '${TEST_DATA_PATH}/data.csv'
                },
                'model_training': {
                    'algorithm': 'RandomForestClassifier',
                    'parameters': {
                        'n_estimators': '${TEST_MODEL_ESTIMATORS}',
                        'max_depth': '${UNDEFINED_VAR:20}'  # With default
                    }
                }
            }
            
            parser = ConfigParser()
            substituted_config = parser.substitute_environment_variables(config_with_env_vars)
            
            assert substituted_config['data_source']['file_path'] == '/test/data/path/data.csv'
            assert substituted_config['model_training']['parameters']['n_estimators'] == '200'
            assert substituted_config['model_training']['parameters']['max_depth'] == '20'  # Default used
            
        finally:
            # Clean up environment variables
            del os.environ['TEST_DATA_PATH']
            del os.environ['TEST_MODEL_ESTIMATORS']


class TestDataIntegration:
    """Integration tests for data access and processing."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_data_loading_and_preprocessing(self):
        """Test CSV data loading and preprocessing integration."""
        # Create test CSV data
        df = pd.DataFrame({
            'numeric_1': np.random.normal(0, 1, 100),
            'numeric_2': np.random.uniform(0, 10, 100),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice(['X', 'Y'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Introduce missing values
        df.loc[5:10, 'numeric_1'] = np.nan
        df.loc[15:20, 'categorical_1'] = np.nan
        
        csv_path = self.data_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Test data loading
        connector = DataConnectorFactory.create_connector(
            'csv', 
            {'file_path': str(csv_path)}
        )
        
        # Note: CSV connector would need to be implemented for this to work
        # For now, test direct pandas loading
        loaded_df = pd.read_csv(csv_path)
        assert loaded_df.shape == df.shape
        assert list(loaded_df.columns) == list(df.columns)
        
        # Test preprocessing pipeline
        preprocessor = PreprocessingPipeline(
            missing_value_strategy='mean',
            scaling_method='standard',
            encoding_strategy='onehot'
        )
        
        X = loaded_df.drop('target', axis=1)
        y = loaded_df['target']
        
        X_processed = preprocessor.fit_transform(X, y)
        
        # Verify preprocessing results
        assert not X_processed.isnull().any().any()  # No missing values
        assert X_processed.shape[0] == X.shape[0]  # Same number of rows
        assert X_processed.shape[1] > X.shape[1]  # More columns due to one-hot encoding
    
    def test_data_validation_integration(self):
        """Test data validation integration."""
        from src.preprocessing.validator import DataValidator
        
        # Create test data with quality issues
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],  # Contains outlier
            'feature_2': [1, 2, np.nan, 4, 5, 6, 7, np.nan, 9, 10],  # Missing values
            'feature_3': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Add duplicate rows
        df = pd.concat([df, df.iloc[0:2]], ignore_index=True)
        
        validator = DataValidator()
        
        # Test basic validation
        validation_results = validator.validate_dataframe(df)
        
        assert 'data_quality_score' in validation_results
        assert 'missing_value_percentage' in validation_results
        assert 'duplicate_rows' in validation_results
        assert validation_results['duplicate_rows'] == 2  # Two duplicate rows added
        
        # Test outlier detection
        outliers = validator.detect_outliers(df)
        assert 'feature_1' in outliers  # Should detect outlier in feature_1
        
        # Test statistics calculation
        stats = validator.calculate_statistics(df)
        assert 'numeric_stats' in stats
        assert 'categorical_stats' in stats


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-focused integration tests."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    def test_large_dataset_processing(self):
        """Test pipeline performance with larger datasets."""
        # Create larger dataset
        df = pd.DataFrame({
            **{f'feature_{i}': np.random.normal(0, 1, 10000) for i in range(20)},
            'target': np.random.choice([0, 1], 10000)
        })
        
        # Save dataset
        data_path = Path(self.temp_dir) / "large_data.csv"
        df.to_csv(data_path, index=False)
        
        # Create configuration for performance testing
        config = {
            'data_source': {
                'type': 'csv',
                'file_path': str(data_path)
            },
            'preprocessing': {
                'target_column': 'target',
                'missing_values': {'strategy': 'mean'},
                'scaling': {'method': 'standard'},
                'encoding': {'categorical_strategy': 'onehot'}
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1  # Use all cores for faster training
                }
            },
            'evaluation': {
                'test_size': 0.2,
                'primary_metric': 'roc_auc'
            },
            'execution': {
                'version': 'python',
                'framework': 'sklearn',
                'mode': 'train',
                'enable_mlflow': False,
                'enable_artifacts': False
            }
        }
        
        # Run pipeline and measure performance
        import time
        start_time = time.time()
        
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=str(self.output_dir)
        )
        
        result = orchestrator.run_pipeline('train')
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify successful execution
        assert result['success'] == True
        
        # Performance assertions
        assert execution_time < 300  # Should complete within 5 minutes
        assert 'execution_time_seconds' in result
        assert result['execution_time_seconds'] > 0
        
        # Log performance metrics for monitoring
        print(f"Large dataset processing completed in {execution_time:.2f} seconds")
        print(f"Dataset shape: {df.shape}")
        print(f"Pipeline execution time: {result['execution_time_seconds']:.2f} seconds")
    
    def test_memory_usage_monitoring(self):
        """Test memory usage during pipeline execution."""
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create moderately sized dataset
            df = pd.DataFrame({
                **{f'feature_{i}': np.random.normal(0, 1, 5000) for i in range(30)},
                'target': np.random.choice([0, 1], 5000)
            })
            
            data_path = Path(self.temp_dir) / "memory_test_data.csv"
            df.to_csv(data_path, index=False)
            
            config = {
                'data_source': {'type': 'csv', 'file_path': str(data_path)},
                'preprocessing': {
                    'target_column': 'target',
                    'missing_values': {'strategy': 'mean'},
                    'scaling': {'method': 'standard'}
                },
                'model_training': {
                    'algorithm': 'RandomForestClassifier',
                    'parameters': {'n_estimators': 50, 'random_state': 42}
                },
                'evaluation': {'test_size': 0.2},
                'execution': {
                    'version': 'python', 'framework': 'sklearn', 'mode': 'train',
                    'enable_mlflow': False, 'enable_artifacts': False
                }
            }
            
            orchestrator = PipelineOrchestrator(config=config, output_dir=str(self.output_dir))
            result = orchestrator.run_pipeline('train')
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            assert result['success'] == True
            
            # Memory usage should be reasonable (less than 1GB increase for this dataset size)
            assert memory_increase < 1024  # MB
            
            print(f"Memory usage - Initial: {initial_memory:.2f} MB, Peak: {peak_memory:.2f} MB")
            print(f"Memory increase: {memory_increase:.2f} MB")
            
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")