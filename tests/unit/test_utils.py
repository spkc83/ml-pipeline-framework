"""
Unit tests for utility modules.
"""

import pytest
import pandas as pd
import numpy as np
import json
import yaml
import tempfile
import pickle
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

from src.utils.config_parser import ConfigParser, ConfigError
from src.utils.mlflow_tracker import MLflowTracker, MLflowError
from src.utils.artifacts import ArtifactManager, ArtifactError


class TestConfigParser:
    """Test cases for ConfigParser."""
    
    def test_load_yaml_config(self, config_file):
        """Test loading YAML configuration."""
        parser = ConfigParser()
        config = parser.load_config(config_file)
        
        assert isinstance(config, dict)
        assert 'data_source' in config
        assert 'preprocessing' in config
        assert 'model_training' in config
        assert 'evaluation' in config
    
    def test_load_json_config(self, temp_directory, basic_config):
        """Test loading JSON configuration."""
        # Create JSON config file
        json_config_path = Path(temp_directory) / "test_config.json"
        with open(json_config_path, 'w') as f:
            json.dump(basic_config, f, indent=2)
        
        parser = ConfigParser()
        config = parser.load_config(str(json_config_path))
        
        assert isinstance(config, dict)
        assert config == basic_config
    
    def test_load_nonexistent_file(self):
        """Test error for nonexistent config file."""
        parser = ConfigParser()
        
        with pytest.raises(ConfigError, match="Config file not found"):
            parser.load_config("/nonexistent/config.yaml")
    
    def test_load_invalid_yaml(self, invalid_config_file):
        """Test error for invalid YAML."""
        parser = ConfigParser()
        
        with pytest.raises(ConfigError, match="Error parsing YAML config"):
            parser.load_config(invalid_config_file)
    
    def test_validate_config_valid(self, basic_config):
        """Test validation of valid configuration."""
        parser = ConfigParser()
        
        # Should not raise any exception
        parser.validate_config(basic_config)
    
    def test_validate_config_missing_section(self, basic_config):
        """Test validation with missing required section."""
        parser = ConfigParser()
        invalid_config = basic_config.copy()
        del invalid_config['data_source']
        
        with pytest.raises(ConfigError, match="Missing required section"):
            parser.validate_config(invalid_config)
    
    def test_validate_config_invalid_structure(self):
        """Test validation with invalid structure."""
        parser = ConfigParser()
        invalid_config = "not a dictionary"
        
        with pytest.raises(ConfigError, match="Config must be a dictionary"):
            parser.validate_config(invalid_config)
    
    def test_merge_configs(self, basic_config):
        """Test merging configurations."""
        parser = ConfigParser()
        
        override_config = {
            'model_training': {
                'algorithm': 'XGBClassifier',
                'parameters': {
                    'n_estimators': 200
                }
            },
            'new_section': {
                'new_param': 'new_value'
            }
        }
        
        merged_config = parser.merge_configs(basic_config, override_config)
        
        # Check that override took effect
        assert merged_config['model_training']['algorithm'] == 'XGBClassifier'
        assert merged_config['model_training']['parameters']['n_estimators'] == 200
        
        # Check that new section was added
        assert 'new_section' in merged_config
        assert merged_config['new_section']['new_param'] == 'new_value'
        
        # Check that other values were preserved
        assert merged_config['preprocessing']['target_column'] == 'target'
    
    def test_substitute_environment_variables(self):
        """Test environment variable substitution."""
        parser = ConfigParser()
        
        # Set test environment variable
        os.environ['TEST_VAR'] = 'test_value'
        os.environ['TEST_PORT'] = '5432'
        
        try:
            config_with_env = {
                'database': {
                    'host': '${TEST_VAR}',
                    'port': '${TEST_PORT}',
                    'name': '${UNDEFINED_VAR:default_db}'
                }
            }
            
            substituted = parser.substitute_environment_variables(config_with_env)
            
            assert substituted['database']['host'] == 'test_value'
            assert substituted['database']['port'] == '5432'
            assert substituted['database']['name'] == 'default_db'
            
        finally:
            # Clean up environment variables
            del os.environ['TEST_VAR']
            del os.environ['TEST_PORT']
    
    def test_get_nested_value(self, basic_config):
        """Test getting nested configuration values."""
        parser = ConfigParser()
        
        # Test existing nested value
        value = parser.get_nested_value(basic_config, 'model_training.algorithm')
        assert value == 'RandomForestClassifier'
        
        # Test deeply nested value
        value = parser.get_nested_value(basic_config, 'model_training.parameters.n_estimators')
        assert value == 100
        
        # Test non-existent path with default
        value = parser.get_nested_value(basic_config, 'nonexistent.path', default='default_value')
        assert value == 'default_value'
        
        # Test non-existent path without default
        with pytest.raises(KeyError):
            parser.get_nested_value(basic_config, 'nonexistent.path')
    
    def test_set_nested_value(self, basic_config):
        """Test setting nested configuration values."""
        parser = ConfigParser()
        config_copy = basic_config.copy()
        
        # Set existing nested value
        parser.set_nested_value(config_copy, 'model_training.algorithm', 'XGBClassifier')
        assert config_copy['model_training']['algorithm'] == 'XGBClassifier'
        
        # Set new nested value
        parser.set_nested_value(config_copy, 'new_section.new_param', 'new_value')
        assert config_copy['new_section']['new_param'] == 'new_value'
        
        # Set deeply nested value
        parser.set_nested_value(config_copy, 'model_training.parameters.max_depth', 10)
        assert config_copy['model_training']['parameters']['max_depth'] == 10
    
    def test_validate_data_source_config(self, basic_config):
        """Test data source configuration validation."""
        parser = ConfigParser()
        
        # Valid config should pass
        parser._validate_data_source_config(basic_config['data_source'])
        
        # Missing type should fail
        invalid_config = {'file_path': '/path/to/file.csv'}
        with pytest.raises(ConfigError, match="Data source type is required"):
            parser._validate_data_source_config(invalid_config)
        
        # Database type without connection should fail
        invalid_config = {'type': 'postgresql'}
        with pytest.raises(ConfigError, match="Database connection parameters are required"):
            parser._validate_data_source_config(invalid_config)
    
    def test_validate_model_config(self, basic_config):
        """Test model configuration validation."""
        parser = ConfigParser()
        
        # Valid config should pass
        parser._validate_model_config(basic_config['model_training'])
        
        # Missing algorithm should fail
        invalid_config = {'parameters': {'n_estimators': 100}}
        with pytest.raises(ConfigError, match="Model algorithm is required"):
            parser._validate_model_config(invalid_config)
    
    def test_save_config(self, temp_directory, basic_config):
        """Test saving configuration to file."""
        parser = ConfigParser()
        
        # Save as YAML
        yaml_path = Path(temp_directory) / "saved_config.yaml"
        parser.save_config(basic_config, str(yaml_path))
        
        # Load and verify
        loaded_config = parser.load_config(str(yaml_path))
        assert loaded_config == basic_config
        
        # Save as JSON
        json_path = Path(temp_directory) / "saved_config.json"
        parser.save_config(basic_config, str(json_path))
        
        # Load and verify
        loaded_config = parser.load_config(str(json_path))
        assert loaded_config == basic_config
    
    def test_get_config_schema(self):
        """Test getting configuration schema."""
        parser = ConfigParser()
        schema = parser.get_config_schema()
        
        assert isinstance(schema, dict)
        assert 'data_source' in schema
        assert 'preprocessing' in schema
        assert 'model_training' in schema
        assert 'evaluation' in schema


class TestMLflowTracker:
    """Test cases for MLflowTracker."""
    
    @patch('src.utils.mlflow_tracker.mlflow')
    @patch('src.utils.mlflow_tracker.MlflowClient')
    def test_initialization(self, mock_client, mock_mlflow):
        """Test MLflow tracker initialization."""
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "test_experiment_id"
        mock_mlflow.set_experiment.return_value = None
        
        tracker = MLflowTracker(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment"
        )
        
        assert tracker.tracking_uri == "http://localhost:5000"
        assert tracker.experiment_name == "test_experiment"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_create_experiment(self, mock_mlflow):
        """Test creating MLflow experiment."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_experiment_id"
        
        tracker = MLflowTracker()
        experiment_id = tracker.create_experiment(
            "new_experiment",
            artifact_location="s3://bucket/path",
            tags={"team": "ml"}
        )
        
        assert experiment_id == "new_experiment_id"
        mock_mlflow.create_experiment.assert_called_once_with(
            name="new_experiment",
            artifact_location="s3://bucket/path",
            tags={"team": "ml"}
        )
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_start_run(self, mock_mlflow):
        """Test starting MLflow run."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run
        
        tracker = MLflowTracker(experiment_name="test_experiment")
        tracker.experiment_id = "test_experiment_id"
        
        run_id = tracker.start_run(
            run_name="test_run",
            tags={"version": "1.0"}
        )
        
        assert run_id == "test_run_id"
        assert tracker.current_run == mock_run
        mock_mlflow.start_run.assert_called_once()
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_end_run(self, mock_mlflow):
        """Test ending MLflow run."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        tracker.end_run(status="FINISHED")
        
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
        assert tracker.current_run is None
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_params(self, mock_mlflow):
        """Test logging parameters."""
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'nested_param': {'sub_param': 'value'}
        }
        
        tracker.log_params(params)
        
        # Verify that mlflow.log_param was called for each parameter
        assert mock_mlflow.log_param.call_count >= 3
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.78
        }
        
        tracker.log_metrics(metrics, step=1)
        
        # Verify that mlflow.log_metric was called for each metric
        assert mock_mlflow.log_metric.call_count == 3
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_artifact(self, mock_mlflow, temp_file):
        """Test logging artifact."""
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        # Create temporary file
        with open(temp_file, 'w') as f:
            f.write("test content")
        
        tracker.log_artifact(temp_file, "artifacts/test.txt")
        
        mock_mlflow.log_artifact.assert_called_once_with(temp_file, "artifacts/test.txt")
    
    @patch('src.utils.mlflow_tracker.mlflow.sklearn.log_model')
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_sklearn_model(self, mock_mlflow, mock_log_sklearn_model):
        """Test logging sklearn model."""
        from sklearn.ensemble import RandomForestClassifier
        
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        model = RandomForestClassifier(n_estimators=10)
        tracker.log_model(model, "model")
        
        mock_log_sklearn_model.assert_called_once()
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_dataset_dataframe(self, mock_mlflow, sample_dataframe):
        """Test logging pandas DataFrame."""
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        tracker.log_dataset(sample_dataframe, "training_data", "csv")
        
        mock_mlflow.log_artifact.assert_called_once()
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_dataset_numpy(self, mock_mlflow):
        """Test logging numpy array."""
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        data = np.random.random((100, 5))
        tracker.log_dataset(data, "test_array", "csv")
        
        mock_mlflow.log_artifact.assert_called_once()
    
    @patch('src.utils.mlflow_tracker.mlflow')
    @patch('matplotlib.pyplot.savefig')
    def test_log_figure(self, mock_savefig, mock_mlflow):
        """Test logging matplotlib figure."""
        import matplotlib.pyplot as plt
        
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        tracker.log_figure(fig, "test_plot.png", "plots")
        
        mock_savefig.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()
        plt.close(fig)
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_log_text(self, mock_mlflow):
        """Test logging text content."""
        mock_run = MagicMock()
        tracker = MLflowTracker()
        tracker.current_run = mock_run
        
        text_content = "This is a test report."
        tracker.log_text(text_content, "report.txt", "reports")
        
        mock_mlflow.log_artifact.assert_called_once()
    
    @patch('src.utils.mlflow_tracker.mlflow')
    def test_search_runs(self, mock_mlflow):
        """Test searching runs."""
        mock_runs_df = pd.DataFrame({
            'run_id': ['run1', 'run2'],
            'metrics.accuracy': [0.85, 0.90],
            'params.n_estimators': [100, 200]
        })
        mock_mlflow.search_runs.return_value = mock_runs_df
        
        tracker = MLflowTracker()
        tracker.experiment_id = "test_experiment_id"
        
        runs = tracker.search_runs(
            filter_string="metrics.accuracy > 0.8",
            max_results=10
        )
        
        assert isinstance(runs, pd.DataFrame)
        assert len(runs) == 2
        mock_mlflow.search_runs.assert_called_once()
    
    @patch('src.utils.mlflow_tracker.MlflowClient')
    def test_get_best_run(self, mock_client_class):
        """Test getting best run by metric."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_run = MagicMock()
        mock_run.info.run_id = "best_run_id"
        mock_run.data.metrics = {'accuracy': 0.95}
        mock_run.data.params = {'n_estimators': 200}
        mock_run.data.tags = {'version': '1.0'}
        mock_run.info.start_time = 1640995200000
        mock_run.info.end_time = 1640995800000
        mock_run.info.status = 'FINISHED'
        
        mock_client.search_runs.return_value = [mock_run]
        
        tracker = MLflowTracker()
        tracker.experiment_id = "test_experiment_id"
        
        best_run = tracker.get_best_run("accuracy", mode="max")
        
        assert best_run['run_id'] == "best_run_id"
        assert best_run['metrics']['accuracy'] == 0.95
    
    def test_log_without_active_run(self):
        """Test logging without active run raises error."""
        tracker = MLflowTracker()
        
        with pytest.raises(MLflowError, match="No active run"):
            tracker.log_params({'param': 'value'})
        
        with pytest.raises(MLflowError, match="No active run"):
            tracker.log_metrics({'metric': 0.5})
    
    def test_detect_model_type(self):
        """Test model type detection."""
        tracker = MLflowTracker()
        
        # Test sklearn model
        sklearn_model = MagicMock()
        sklearn_model.__class__.__module__ = 'sklearn.ensemble._forest'
        model_type = tracker._detect_model_type(sklearn_model)
        assert model_type == 'sklearn'
        
        # Test XGBoost model
        xgb_model = MagicMock()
        xgb_model.__class__.__name__ = 'XGBClassifier'
        model_type = tracker._detect_model_type(xgb_model)
        assert model_type == 'xgboost'
        
        # Test unknown model
        unknown_model = MagicMock()
        unknown_model.__class__.__module__ = 'custom.module'
        model_type = tracker._detect_model_type(unknown_model)
        assert model_type == 'unknown'


class TestArtifactManager:
    """Test cases for ArtifactManager."""
    
    def test_create_local_backend(self, temp_directory):
        """Test creating local artifact backend."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        assert manager.storage_type == 'local'
        assert manager.config['base_path'] == temp_directory
    
    @patch('src.utils.artifacts.boto3')
    def test_create_s3_backend(self, mock_boto3):
        """Test creating S3 artifact backend."""
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        manager = ArtifactManager.create_s3_backend(
            bucket_name='test-bucket',
            region_name='us-east-1'
        )
        
        assert manager.storage_type == 's3'
        assert manager.config['bucket_name'] == 'test-bucket'
        mock_boto3.client.assert_called_once_with(
            's3',
            region_name='us-east-1',
            aws_access_key_id=None,
            aws_secret_access_key=None
        )
    
    @patch('src.utils.artifacts.hdfs')
    def test_create_hdfs_backend(self, mock_hdfs):
        """Test creating HDFS artifact backend."""
        mock_hdfs_client = MagicMock()
        mock_hdfs.InsecureClient.return_value = mock_hdfs_client
        
        manager = ArtifactManager.create_hdfs_backend(
            namenode_url='http://namenode:9870',
            user='hdfs'
        )
        
        assert manager.storage_type == 'hdfs'
        assert manager.config['namenode_url'] == 'http://namenode:9870'
        mock_hdfs.InsecureClient.assert_called_once_with(
            'http://namenode:9870',
            user='hdfs'
        )
    
    def test_save_artifact_local(self, temp_directory):
        """Test saving artifact to local storage."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Test saving dictionary artifact
        artifact_data = {'key': 'value', 'number': 42}
        artifact_info = manager.save_artifact(
            artifact_data,
            'test_dict',
            artifact_type='data',
            description='Test dictionary'
        )
        
        assert artifact_info['name'] == 'test_dict'
        assert artifact_info['type'] == 'data'
        assert artifact_info['version'] == '1.0.0'
        assert 'path' in artifact_info
        assert 'size_bytes' in artifact_info
        
        # Verify file was created
        artifact_path = Path(artifact_info['path'])
        assert artifact_path.exists()
    
    def test_save_dataframe_artifact_local(self, temp_directory, sample_dataframe):
        """Test saving DataFrame artifact to local storage."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        artifact_info = manager.save_artifact(
            sample_dataframe,
            'test_dataframe',
            artifact_type='data',
            format='parquet'
        )
        
        assert artifact_info['name'] == 'test_dataframe'
        assert artifact_info['format'] == 'parquet'
        
        # Verify file was created
        artifact_path = Path(artifact_info['path'])
        assert artifact_path.exists()
        assert artifact_path.suffix == '.parquet'
    
    def test_save_model_artifact_local(self, temp_directory):
        """Test saving model artifact to local storage."""
        from sklearn.ensemble import RandomForestClassifier
        
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Create and train a simple model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((50, 4))
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        artifact_info = manager.save_artifact(
            model,
            'test_model',
            artifact_type='model',
            description='Test sklearn model'
        )
        
        assert artifact_info['name'] == 'test_model'
        assert artifact_info['type'] == 'model'
        
        # Verify file was created
        artifact_path = Path(artifact_info['path'])
        assert artifact_path.exists()
    
    def test_load_artifact_local(self, temp_directory):
        """Test loading artifact from local storage."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save artifact first
        original_data = {'test': 'data', 'number': 123}
        artifact_info = manager.save_artifact(
            original_data,
            'test_load',
            artifact_type='data'
        )
        
        # Load artifact
        loaded_data = manager.load_artifact(artifact_info['name'], artifact_info['version'])
        
        assert loaded_data == original_data
    
    def test_list_artifacts_local(self, temp_directory):
        """Test listing artifacts in local storage."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save multiple artifacts
        for i in range(3):
            manager.save_artifact(
                {'data': f'test_{i}'},
                f'artifact_{i}',
                artifact_type='data'
            )
        
        # List artifacts
        artifacts = manager.list_artifacts()
        
        assert len(artifacts) == 3
        assert all('name' in artifact for artifact in artifacts)
        assert all('version' in artifact for artifact in artifacts)
        assert all('type' in artifact for artifact in artifacts)
    
    def test_delete_artifact_local(self, temp_directory):
        """Test deleting artifact from local storage."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save artifact
        artifact_info = manager.save_artifact(
            {'test': 'data'},
            'test_delete',
            artifact_type='data'
        )
        
        # Verify artifact exists
        artifacts_before = manager.list_artifacts()
        assert len(artifacts_before) == 1
        
        # Delete artifact
        manager.delete_artifact(artifact_info['name'], artifact_info['version'])
        
        # Verify artifact is deleted
        artifacts_after = manager.list_artifacts()
        assert len(artifacts_after) == 0
    
    def test_versioning(self, temp_directory):
        """Test artifact versioning."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save same artifact multiple times
        artifact_name = 'versioned_artifact'
        
        # Version 1.0.0
        v1_info = manager.save_artifact(
            {'version': 1},
            artifact_name,
            artifact_type='data'
        )
        assert v1_info['version'] == '1.0.0'
        
        # Version 1.0.1 (automatic increment)
        v2_info = manager.save_artifact(
            {'version': 2},
            artifact_name,
            artifact_type='data'
        )
        assert v2_info['version'] == '1.0.1'
        
        # Version 1.0.2 (automatic increment)
        v3_info = manager.save_artifact(
            {'version': 3},
            artifact_name,
            artifact_type='data'
        )
        assert v3_info['version'] == '1.0.2'
        
        # List all versions
        artifacts = manager.list_artifacts(name_filter=artifact_name)
        assert len(artifacts) == 3
        
        # Load specific version
        v1_data = manager.load_artifact(artifact_name, '1.0.0')
        assert v1_data['version'] == 1
        
        v2_data = manager.load_artifact(artifact_name, '1.0.1')
        assert v2_data['version'] == 2
    
    def test_metadata_tracking(self, temp_directory):
        """Test artifact metadata tracking."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save artifact with metadata
        artifact_info = manager.save_artifact(
            {'test': 'data'},
            'metadata_test',
            artifact_type='data',
            description='Test artifact with metadata',
            tags=['test', 'metadata'],
            custom_metadata={'author': 'test_user', 'project': 'ml_pipeline'}
        )
        
        # Verify metadata is stored
        assert artifact_info['description'] == 'Test artifact with metadata'
        assert artifact_info['tags'] == ['test', 'metadata']
        assert artifact_info['custom_metadata']['author'] == 'test_user'
        assert artifact_info['custom_metadata']['project'] == 'ml_pipeline'
        assert 'created_at' in artifact_info
        assert 'size_bytes' in artifact_info
    
    def test_compression(self, temp_directory):
        """Test artifact compression."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Create large data for compression testing
        large_data = {'data': list(range(10000))}
        
        # Save with compression
        artifact_info = manager.save_artifact(
            large_data,
            'compressed_test',
            artifact_type='data',
            compress=True
        )
        
        assert artifact_info['compressed'] == True
        
        # Load and verify data integrity
        loaded_data = manager.load_artifact(
            artifact_info['name'], 
            artifact_info['version']
        )
        assert loaded_data == large_data
    
    def test_create_artifact_bundle(self, temp_directory):
        """Test creating artifact bundle."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save multiple artifacts
        artifacts_to_bundle = []
        for i in range(3):
            info = manager.save_artifact(
                {'data': f'test_{i}'},
                f'bundle_artifact_{i}',
                artifact_type='data'
            )
            artifacts_to_bundle.append((info['name'], info['version']))
        
        # Create bundle
        bundle_info = manager.create_artifact_bundle(
            'test_bundle',
            artifacts_to_bundle,
            description='Test artifact bundle'
        )
        
        assert bundle_info['name'] == 'test_bundle'
        assert bundle_info['type'] == 'bundle'
        assert bundle_info['description'] == 'Test artifact bundle'
        assert len(bundle_info['bundled_artifacts']) == 3
    
    def test_get_artifact_info(self, temp_directory):
        """Test getting artifact information."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        # Save artifact
        original_info = manager.save_artifact(
            {'test': 'data'},
            'info_test',
            artifact_type='data',
            description='Test artifact info'
        )
        
        # Get artifact info
        retrieved_info = manager.get_artifact_info(
            original_info['name'], 
            original_info['version']
        )
        
        assert retrieved_info['name'] == original_info['name']
        assert retrieved_info['version'] == original_info['version']
        assert retrieved_info['type'] == original_info['type']
        assert retrieved_info['description'] == original_info['description']
    
    def test_unsupported_storage_type(self):
        """Test error for unsupported storage type."""
        with pytest.raises(ArtifactError, match="Unsupported storage type"):
            ArtifactManager(storage_type='unsupported', config={})
    
    def test_invalid_artifact_name(self, temp_directory):
        """Test error for invalid artifact name."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        with pytest.raises(ArtifactError, match="Invalid artifact name"):
            manager.save_artifact(
                {'test': 'data'},
                'invalid/name*with$special',  # Invalid characters
                artifact_type='data'
            )
    
    def test_load_nonexistent_artifact(self, temp_directory):
        """Test error when loading nonexistent artifact."""
        manager = ArtifactManager.create_local_backend(temp_directory)
        
        with pytest.raises(ArtifactError, match="Artifact not found"):
            manager.load_artifact('nonexistent', '1.0.0')


class TestUtilsIntegration:
    """Integration tests for utility modules."""
    
    def test_config_with_mlflow_integration(self, temp_directory):
        """Test configuration with MLflow integration."""
        # Create config with MLflow settings
        config = {
            'execution': {
                'enable_mlflow': True
            },
            'mlflow': {
                'tracking_uri': f'file://{temp_directory}/mlruns',
                'experiment_name': 'integration_test'
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {
                    'n_estimators': 50
                }
            }
        }
        
        # Save config
        parser = ConfigParser()
        config_path = Path(temp_directory) / 'config.yaml'
        parser.save_config(config, str(config_path))
        
        # Load config
        loaded_config = parser.load_config(str(config_path))
        
        # Use config to initialize MLflow tracker
        with patch('src.utils.mlflow_tracker.mlflow') as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "test_exp_id"
            
            tracker = MLflowTracker(
                tracking_uri=loaded_config['mlflow']['tracking_uri'],
                experiment_name=loaded_config['mlflow']['experiment_name']
            )
            
            assert tracker.experiment_name == 'integration_test'
    
    def test_artifacts_with_config_integration(self, temp_directory):
        """Test artifact manager with configuration."""
        # Create config with artifact settings
        config = {
            'artifacts': {
                'storage': {
                    'type': 'local',
                    'path': temp_directory
                }
            }
        }
        
        # Initialize artifact manager from config
        artifact_config = config['artifacts']['storage']
        if artifact_config['type'] == 'local':
            manager = ArtifactManager.create_local_backend(artifact_config['path'])
        
        # Test saving and loading
        test_data = {'config_test': True, 'value': 42}
        artifact_info = manager.save_artifact(
            test_data,
            'config_integration_test',
            artifact_type='data'
        )
        
        loaded_data = manager.load_artifact(
            artifact_info['name'],
            artifact_info['version']
        )
        
        assert loaded_data == test_data
    
    def test_complete_utils_workflow(self, temp_directory, basic_config):
        """Test complete workflow using all utility modules."""
        # 1. Load and validate configuration
        parser = ConfigParser()
        config_path = Path(temp_directory) / 'workflow_config.yaml'
        parser.save_config(basic_config, str(config_path))
        
        loaded_config = parser.load_config(str(config_path))
        parser.validate_config(loaded_config)
        
        # 2. Initialize artifact manager
        artifact_manager = ArtifactManager.create_local_backend(
            str(Path(temp_directory) / 'artifacts')
        )
        
        # 3. Initialize MLflow tracker (mocked)
        with patch('src.utils.mlflow_tracker.mlflow') as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "workflow_exp_id"
            mock_run = MagicMock()
            mock_run.info.run_id = "workflow_run_id"
            mock_mlflow.start_run.return_value = mock_run
            
            tracker = MLflowTracker(experiment_name='workflow_test')
            run_id = tracker.start_run('workflow_run')
            
            # 4. Log configuration as parameters
            tracker.log_params(loaded_config['model_training']['parameters'])
            
            # 5. Save configuration as artifact
            config_artifact = artifact_manager.save_artifact(
                loaded_config,
                'workflow_config',
                artifact_type='config',
                description='Configuration used in workflow test'
            )
            
            # 6. Log artifact to MLflow
            tracker.log_artifact(config_artifact['path'], 'config')
            
            # 7. End MLflow run
            tracker.end_run()
        
        # Verify everything worked
        assert run_id == "workflow_run_id"
        assert config_artifact['name'] == 'workflow_config'
        
        # Verify artifact can be loaded
        loaded_config_from_artifact = artifact_manager.load_artifact(
            config_artifact['name'],
            config_artifact['version']
        )
        assert loaded_config_from_artifact == loaded_config