"""
Unit tests for pipeline orchestrator.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

from src.pipeline_orchestrator import PipelineOrchestrator, PipelineError


class TestPipelineOrchestrator:
    """Test cases for PipelineOrchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.basic_config = {
            'data_source': {
                'type': 'csv',
                'file_path': '/path/to/data.csv'
            },
            'preprocessing': {
                'target_column': 'target',
                'missing_values': {'strategy': 'mean'},
                'scaling': {'method': 'standard'},
                'encoding': {'categorical_strategy': 'onehot'},
                'data_validation': {'enabled': True},
                'imbalance_handling': {'enabled': False}
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {'n_estimators': 10, 'random_state': 42}
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
            },
            'explainability': {
                'enabled': False
            }
        }
    
    def test_initialization(self, temp_directory):
        """Test orchestrator initialization."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory,
            verbose=True
        )
        
        assert orchestrator.config == self.basic_config
        assert orchestrator.output_dir == Path(temp_directory)
        assert orchestrator.verbose == True
        assert orchestrator.execution_mode == 'python'
        assert orchestrator.framework == 'sklearn'
        
        # Check output directory structure was created
        assert (Path(temp_directory) / 'models').exists()
        assert (Path(temp_directory) / 'reports').exists()
        assert (Path(temp_directory) / 'plots').exists()
    
    def test_initialization_missing_core_libs(self):
        """Test error when core libraries are missing."""
        with patch('src.pipeline_orchestrator.CORE_LIBS_AVAILABLE', False):
            with pytest.raises(PipelineError, match="Core ML libraries"):
                PipelineOrchestrator(config=self.basic_config)
    
    def test_validate_config_valid(self, temp_directory):
        """Test configuration validation with valid config."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        validation_result = orchestrator.validate_config()
        
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        assert 'recommendations' in validation_result
        assert validation_result['valid'] == True
    
    def test_validate_config_missing_section(self, temp_directory):
        """Test configuration validation with missing section."""
        invalid_config = self.basic_config.copy()
        del invalid_config['data_source']
        
        orchestrator = PipelineOrchestrator(
            config=invalid_config,
            output_dir=temp_directory
        )
        
        validation_result = orchestrator.validate_config()
        
        assert validation_result['valid'] == False
        assert any('Missing required section: data_source' in error 
                  for error in validation_result['errors'])
    
    def test_validate_config_pyspark_unavailable(self, temp_directory):
        """Test validation when PySpark is requested but unavailable."""
        spark_config = self.basic_config.copy()
        spark_config['execution']['version'] = 'pyspark'
        
        with patch('src.pipeline_orchestrator.SPARK_AVAILABLE', False):
            orchestrator = PipelineOrchestrator(
                config=spark_config,
                output_dir=temp_directory
            )
            
            validation_result = orchestrator.validate_config()
            
            assert validation_result['valid'] == False
            assert any('PySpark requested but not available' in error 
                      for error in validation_result['errors'])
    
    def test_get_execution_plan_train_mode(self, temp_directory):
        """Test execution plan generation for training mode."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        execution_plan = orchestrator.get_execution_plan()
        
        assert isinstance(execution_plan, dict)
        assert execution_plan['mode'] == 'train'
        assert execution_plan['framework'] == 'sklearn'
        assert execution_plan['execution_environment'] == 'python'
        assert 'steps' in execution_plan
        assert 'total_estimated_duration' in execution_plan
        assert 'resource_requirements' in execution_plan
        assert 'dependencies' in execution_plan
        assert 'outputs' in execution_plan
        
        # Check that training steps are included
        step_names = [step['name'] for step in execution_plan['steps']]
        assert 'Data Loading' in step_names
        assert 'Model Training' in step_names
        assert 'Model Evaluation' in step_names
    
    def test_get_execution_plan_predict_mode(self, temp_directory):
        """Test execution plan generation for prediction mode."""
        predict_config = self.basic_config.copy()
        predict_config['execution']['mode'] = 'predict'
        
        orchestrator = PipelineOrchestrator(
            config=predict_config,
            output_dir=temp_directory
        )
        
        execution_plan = orchestrator.get_execution_plan()
        
        assert execution_plan['mode'] == 'predict'
        step_names = [step['name'] for step in execution_plan['steps']]
        assert 'Data Loading' in step_names
        assert 'Model Loading' in step_names
        assert 'Prediction' in step_names
    
    @patch('src.pipeline_orchestrator.DataConnectorFactory')
    @patch('src.pipeline_orchestrator.PreprocessingPipeline')
    @patch('src.pipeline_orchestrator.ModelFactory')
    def test_run_training_pipeline_success(self, mock_model_factory, mock_preprocessing, 
                                         mock_data_factory, temp_directory, sample_dataframe):
        """Test successful training pipeline execution."""
        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.fetch_data.return_value = sample_dataframe
        mock_data_factory.create_connector.return_value = mock_connector
        
        mock_preprocessor = MagicMock()
        mock_preprocessor.fit_transform.return_value = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        mock_preprocessing.return_value = mock_preprocessor
        
        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
        mock_model.evaluate.return_value = {'accuracy': 0.85, 'roc_auc': 0.88}
        mock_model_factory.create_model.return_value = mock_model
        
        # Create orchestrator and run training
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        result = orchestrator.run_pipeline('train')
        
        assert result['success'] == True
        assert result['mode'] == 'train'
        assert 'models' in result
        assert 'metrics' in result
        assert 'artifacts' in result
        assert 'execution_time_seconds' in result
        assert result['execution_time_seconds'] > 0
    
    def test_run_pipeline_invalid_mode(self, temp_directory):
        """Test error for invalid pipeline mode."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        result = orchestrator.run_pipeline('invalid_mode')
        
        assert result['success'] == False
        assert 'error' in result
        assert 'Unknown execution mode' in result['error']
    
    def test_run_pipeline_with_failure(self, temp_directory):
        """Test pipeline execution with failure."""
        # Create config that will cause failure
        failing_config = self.basic_config.copy()
        failing_config['data_source']['file_path'] = '/nonexistent/file.csv'
        
        orchestrator = PipelineOrchestrator(
            config=failing_config,
            output_dir=temp_directory
        )
        
        with patch('src.pipeline_orchestrator.DataConnectorFactory') as mock_factory:
            mock_factory.create_connector.side_effect = Exception("Connection failed")
            
            result = orchestrator.run_pipeline('train')
            
            assert result['success'] == False
            assert 'error' in result
            assert 'partial_results' in result
    
    @patch('src.pipeline_orchestrator.SparkSession')
    def test_initialize_spark_session(self, mock_spark_session, temp_directory):
        """Test Spark session initialization."""
        spark_config = self.basic_config.copy()
        spark_config['execution']['version'] = 'pyspark'
        spark_config['spark'] = {
            'app_name': 'Test App',
            'master': 'local[2]',
            'config': {
                'spark.sql.adaptive.enabled': 'true'
            }
        }
        
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.getOrCreate.return_value = mock_spark
        
        with patch('src.pipeline_orchestrator.SPARK_AVAILABLE', True):
            orchestrator = PipelineOrchestrator(
                config=spark_config,
                output_dir=temp_directory
            )
            
            orchestrator._initialize_spark_session()
            
            assert orchestrator.spark_session == mock_spark
            mock_spark.sparkContext.setLogLevel.assert_called_once_with("WARN")
    
    def test_initialize_spark_session_unavailable(self, temp_directory):
        """Test error when Spark is unavailable."""
        spark_config = self.basic_config.copy()
        spark_config['execution']['version'] = 'pyspark'
        
        with patch('src.pipeline_orchestrator.SPARK_AVAILABLE', False):
            orchestrator = PipelineOrchestrator(
                config=spark_config,
                output_dir=temp_directory
            )
            
            with pytest.raises(PipelineError, match="PySpark not available"):
                orchestrator._initialize_spark_session()
    
    @patch('src.pipeline_orchestrator.MLflowTracker')
    def test_initialize_tracking_mlflow(self, mock_mlflow_tracker, temp_directory):
        """Test MLflow tracking initialization."""
        mlflow_config = self.basic_config.copy()
        mlflow_config['execution']['enable_mlflow'] = True
        mlflow_config['mlflow'] = {
            'tracking_uri': 'http://localhost:5000',
            'experiment_name': 'test_experiment'
        }
        
        mock_tracker = MagicMock()
        mock_tracker.start_run.return_value = 'test_run_id'
        mock_mlflow_tracker.return_value = mock_tracker
        
        orchestrator = PipelineOrchestrator(
            config=mlflow_config,
            output_dir=temp_directory
        )
        
        orchestrator._initialize_tracking()
        
        assert orchestrator.mlflow_tracker == mock_tracker
        assert orchestrator.current_run_id == 'test_run_id'
        mock_tracker.log_params.assert_called_once()
    
    @patch('src.pipeline_orchestrator.ArtifactManager')
    def test_initialize_tracking_artifacts_local(self, mock_artifact_manager, temp_directory):
        """Test artifact management initialization for local storage."""
        artifacts_config = self.basic_config.copy()
        artifacts_config['execution']['enable_artifacts'] = True
        artifacts_config['artifacts'] = {
            'storage': {
                'type': 'local',
                'path': temp_directory
            }
        }
        
        mock_manager = MagicMock()
        mock_artifact_manager.create_local_backend.return_value = mock_manager
        
        orchestrator = PipelineOrchestrator(
            config=artifacts_config,
            output_dir=temp_directory
        )
        
        orchestrator._initialize_tracking()
        
        assert orchestrator.artifact_manager == mock_manager
        mock_artifact_manager.create_local_backend.assert_called_once()
    
    @patch('src.pipeline_orchestrator.ArtifactManager')
    def test_initialize_tracking_artifacts_s3(self, mock_artifact_manager, temp_directory):
        """Test artifact management initialization for S3 storage."""
        s3_config = self.basic_config.copy()
        s3_config['execution']['enable_artifacts'] = True
        s3_config['artifacts'] = {
            'storage': {
                'type': 's3',
                'bucket': 'test-bucket',
                'region': 'us-east-1'
            }
        }
        
        mock_manager = MagicMock()
        mock_artifact_manager.create_s3_backend.return_value = mock_manager
        
        orchestrator = PipelineOrchestrator(
            config=s3_config,
            output_dir=temp_directory
        )
        
        orchestrator._initialize_tracking()
        
        assert orchestrator.artifact_manager == mock_manager
        mock_artifact_manager.create_s3_backend.assert_called_once()
    
    def test_execute_stage_success(self, temp_directory):
        """Test successful stage execution."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Define a simple stage function
        def test_stage():
            return "stage_result"
        
        result = orchestrator._execute_stage('test_stage', test_stage)
        
        assert result == "stage_result"
        assert 'test_stage' in orchestrator.pipeline_state['completed_stages']
        assert 'test_stage' in orchestrator.stage_timings
        assert orchestrator.stage_timings['test_stage'] > 0
    
    def test_execute_stage_failure(self, temp_directory):
        """Test stage execution with failure."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Define a failing stage function
        def failing_stage():
            raise ValueError("Stage failed")
        
        with pytest.raises(ValueError, match="Stage failed"):
            orchestrator._execute_stage('failing_stage', failing_stage)
        
        assert 'failing_stage' in orchestrator.pipeline_state['failed_stages']
        assert 'failing_stage' in orchestrator.stage_timings
    
    def test_save_pipeline_state(self, temp_directory):
        """Test pipeline state saving."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Set some pipeline state
        orchestrator.pipeline_state['current_stage'] = 'test_stage'
        orchestrator.pipeline_state['completed_stages'] = ['stage1', 'stage2']
        orchestrator.pipeline_state['metrics'] = {'accuracy': 0.85}
        orchestrator.stage_timings = {'stage1': 10.5, 'stage2': 15.2}
        
        orchestrator._save_pipeline_state()
        
        # Check that state file was created
        state_file = orchestrator.output_dir / "pipeline_state.json"
        assert state_file.exists()
        
        # Verify content
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        assert saved_state['current_stage'] == 'test_stage'
        assert saved_state['completed_stages'] == ['stage1', 'stage2']
        assert saved_state['metrics']['accuracy'] == 0.85
        assert saved_state['stage_timings']['stage1'] == 10.5
    
    def test_resume_pipeline(self, temp_directory):
        """Test pipeline resumption from saved state."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Create a saved state
        saved_state = {
            'current_stage': 'model_training',
            'completed_stages': ['data_loading', 'preprocessing'],
            'failed_stages': [],
            'artifacts': {},
            'metrics': {'data_quality': 0.9},
            'stage_timings': {'data_loading': 5.0, 'preprocessing': 10.0}
        }
        
        state_file = orchestrator.output_dir / "pipeline_state.json"
        with open(state_file, 'w') as f:
            json.dump(saved_state, f)
        
        # Mock the run_pipeline method to avoid actual execution
        with patch.object(orchestrator, 'run_pipeline') as mock_run:
            mock_run.return_value = {'success': True}
            
            result = orchestrator.resume_pipeline('model_training')
            
            # Verify state was loaded
            assert orchestrator.pipeline_state['completed_stages'] == ['data_loading', 'preprocessing']
            assert orchestrator.pipeline_state['current_stage'] == 'model_training'
            assert orchestrator.pipeline_state['metrics']['data_quality'] == 0.9
            
            # Verify run_pipeline was called
            mock_run.assert_called_once_with('train')
    
    def test_cleanup_execution_environment(self, temp_directory):
        """Test execution environment cleanup."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Mock Spark session and MLflow tracker
        mock_spark = MagicMock()
        mock_tracker = MagicMock()
        
        orchestrator.spark_session = mock_spark
        orchestrator.mlflow_tracker = mock_tracker
        orchestrator.current_run_id = 'test_run_id'
        
        orchestrator._cleanup_execution_environment()
        
        # Verify cleanup was called
        mock_spark.stop.assert_called_once()
        mock_tracker.end_run.assert_called_once()
    
    def test_flatten_config(self, temp_directory):
        """Test configuration flattening for MLflow logging."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        flattened = orchestrator._flatten_config(self.basic_config)
        
        assert isinstance(flattened, dict)
        assert 'data_source_type' in flattened
        assert 'model_training_algorithm' in flattened
        assert 'model_training_parameters_n_estimators' in flattened
        assert flattened['data_source_type'] == 'csv'
        assert flattened['model_training_algorithm'] == 'RandomForestClassifier'
        assert flattened['model_training_parameters_n_estimators'] == '10'
    
    def test_make_json_serializable(self, temp_directory):
        """Test making objects JSON serializable."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Test with various data types
        test_data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3]),
            'regular_dict': {'key': 'value'},
            'regular_list': [1, 2, 3]
        }
        
        serializable = orchestrator._make_json_serializable(test_data)
        
        # Should be able to serialize to JSON
        json_str = json.dumps(serializable)
        assert json_str is not None
        
        # Verify specific conversions
        assert isinstance(serializable['numpy_int'], float)
        assert isinstance(serializable['numpy_float'], float)
        assert isinstance(serializable['numpy_array'], list)
        assert serializable['numpy_array'] == [1, 2, 3]
    
    def test_collect_partial_results(self, temp_directory):
        """Test collecting partial results on failure."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Set some pipeline state to simulate partial execution
        orchestrator.pipeline_state['completed_stages'] = ['data_loading', 'preprocessing']
        orchestrator.pipeline_state['failed_stages'] = ['model_training']
        orchestrator.pipeline_state['artifacts'] = {'preprocessor': '/path/to/preprocessor.pkl'}
        orchestrator.stage_timings = {'data_loading': 5.0, 'preprocessing': 10.0}
        
        partial_results = orchestrator._collect_partial_results()
        
        assert 'completed_stages' in partial_results
        assert 'failed_stages' in partial_results
        assert 'available_artifacts' in partial_results
        assert 'stage_timings' in partial_results
        
        assert partial_results['completed_stages'] == ['data_loading', 'preprocessing']
        assert partial_results['failed_stages'] == ['model_training']
        assert 'preprocessor' in partial_results['available_artifacts']
    
    def test_framework_availability_checks(self, temp_directory):
        """Test framework availability checking methods."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Test sklearn check (should be available in test environment)
        assert orchestrator._check_sklearn_available() == True
        
        # Test other frameworks (may or may not be available)
        # Just verify methods don't raise exceptions
        try:
            orchestrator._check_h2o_available()
            orchestrator._check_xgboost_available()
            orchestrator._check_lightgbm_available()
            orchestrator._check_catboost_available()
            orchestrator._check_spark_available()
        except Exception as e:
            pytest.fail(f"Framework availability check raised exception: {e}")
    
    def test_data_source_validation(self, temp_directory):
        """Test data source configuration validation."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Test valid data source
        valid_config = {'type': 'csv', 'file_path': '/path/to/file.csv'}
        errors = orchestrator._validate_data_source()
        assert len(errors) == 0  # Should have no errors for basic config
        
        # Test invalid data source (no type)
        invalid_orchestrator = PipelineOrchestrator(
            config={'data_source': {'file_path': '/path/to/file.csv'}},
            output_dir=temp_directory
        )
        errors = invalid_orchestrator._validate_data_source()
        assert len(errors) > 0
        assert any('type' in error for error in errors)
    
    def test_model_config_validation(self, temp_directory):
        """Test model configuration validation."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        warnings = orchestrator._validate_model_config()
        # Should have no warnings for basic sklearn config
        assert len(warnings) == 0
        
        # Test with potentially incompatible algorithm
        incompatible_config = self.basic_config.copy()
        incompatible_config['execution']['framework'] = 'sklearn'
        incompatible_config['model_training']['algorithm'] = 'XGBClassifier'
        
        incompatible_orchestrator = PipelineOrchestrator(
            config=incompatible_config,
            output_dir=temp_directory
        )
        warnings = incompatible_orchestrator._validate_model_config()
        # May have warnings about algorithm availability
    
    def test_resource_estimation(self, temp_directory):
        """Test resource requirements estimation."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Test Python mode
        python_resources = orchestrator._estimate_resource_requirements()
        assert 'cpu_cores' in python_resources
        assert 'memory' in python_resources
        assert 'storage' in python_resources
        assert 'environment' in python_resources
        assert python_resources['environment'] == 'Single machine'
        
        # Test Spark mode
        spark_config = self.basic_config.copy()
        spark_config['execution']['version'] = 'pyspark'
        
        spark_orchestrator = PipelineOrchestrator(
            config=spark_config,
            output_dir=temp_directory
        )
        spark_resources = spark_orchestrator._estimate_resource_requirements()
        assert spark_resources['environment'] == 'Spark cluster'
    
    def test_dependencies_list(self, temp_directory):
        """Test getting pipeline dependencies."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        dependencies = orchestrator._get_dependencies()
        
        # Should include core dependencies
        assert 'pandas' in dependencies
        assert 'numpy' in dependencies
        assert 'scikit-learn' in dependencies
        
        # Test with different frameworks
        xgb_config = self.basic_config.copy()
        xgb_config['execution']['framework'] = 'xgboost'
        
        xgb_orchestrator = PipelineOrchestrator(
            config=xgb_config,
            output_dir=temp_directory
        )
        xgb_dependencies = xgb_orchestrator._get_dependencies()
        assert 'xgboost' in xgb_dependencies
    
    def test_expected_outputs(self, temp_directory):
        """Test getting expected pipeline outputs."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        outputs = orchestrator._get_expected_outputs()
        
        assert isinstance(outputs, list)
        assert 'Trained model files' in outputs
        assert 'Performance metrics' in outputs
        assert 'Evaluation reports' in outputs
    
    def test_execution_steps_for_different_modes(self, temp_directory):
        """Test execution steps for different pipeline modes."""
        orchestrator = PipelineOrchestrator(
            config=self.basic_config,
            output_dir=temp_directory
        )
        
        # Test training steps
        train_steps = orchestrator._get_training_steps()
        assert len(train_steps) > 0
        assert any('Data Loading' in step['name'] for step in train_steps)
        assert any('Model Training' in step['name'] for step in train_steps)
        
        # Test prediction steps
        predict_steps = orchestrator._get_prediction_steps()
        assert len(predict_steps) > 0
        assert any('Model Loading' in step['name'] for step in predict_steps)
        assert any('Prediction' in step['name'] for step in predict_steps)
        
        # Test evaluation steps
        eval_steps = orchestrator._get_evaluation_steps()
        assert len(eval_steps) > 0
        assert any('Evaluation' in step['name'] for step in eval_steps)
        
        # Test comparison steps
        comp_steps = orchestrator._get_comparison_steps()
        assert len(comp_steps) > 0
        assert any('Model Comparison' in step['name'] for step in comp_steps)
        
        # Test experiment steps
        exp_steps = orchestrator._get_experiment_steps()
        assert len(exp_steps) > 0
        assert any('Hyperparameter Optimization' in step['name'] for step in exp_steps)


class TestPipelineOrchestratorIntegration:
    """Integration tests for PipelineOrchestrator."""
    
    @patch('src.pipeline_orchestrator.DataConnectorFactory')
    @patch('src.pipeline_orchestrator.PreprocessingPipeline')
    @patch('src.pipeline_orchestrator.ModelFactory')
    @patch('src.pipeline_orchestrator.MetricsCalculator')
    def test_complete_training_workflow(self, mock_metrics, mock_model_factory, 
                                      mock_preprocessing, mock_data_factory, 
                                      temp_directory, sample_dataframe):
        """Test complete training workflow integration."""
        # Setup comprehensive mocks
        mock_connector = MagicMock()
        mock_connector.fetch_data.return_value = sample_dataframe
        mock_data_factory.create_connector.return_value = mock_connector
        
        mock_preprocessor = MagicMock()
        X_processed = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        mock_preprocessor.fit_transform.return_value = X_processed
        mock_preprocessing.return_value = mock_preprocessor
        
        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_model.predict.return_value = np.random.choice([0, 1], len(sample_dataframe))
        mock_model.predict_proba.return_value = np.random.uniform(0, 1, (len(sample_dataframe), 2))
        mock_model.evaluate.return_value = {'accuracy': 0.85, 'roc_auc': 0.88}
        mock_model_factory.create_model.return_value = mock_model
        
        mock_metrics_calc = MagicMock()
        mock_metrics_calc.calculate_classification_metrics.return_value = {
            'accuracy': 0.85, 'precision': 0.82, 'recall': 0.78, 'f1_score': 0.80, 'roc_auc': 0.88
        }
        mock_metrics_calc.generate_comprehensive_report.return_value = {
            'metrics': {'accuracy': 0.85}, 'plots': ['roc_curve.png'], 'business_metrics': {}
        }
        mock_metrics.return_value = mock_metrics_calc
        
        # Create config with explainability enabled
        config = {
            'data_source': {
                'type': 'csv',
                'file_path': '/path/to/data.csv'
            },
            'preprocessing': {
                'target_column': 'target',
                'missing_values': {'strategy': 'mean'},
                'scaling': {'method': 'standard'},
                'encoding': {'categorical_strategy': 'onehot'},
                'data_validation': {'enabled': True},
                'imbalance_handling': {'enabled': False}
            },
            'model_training': {
                'algorithm': 'RandomForestClassifier',
                'parameters': {'n_estimators': 10, 'random_state': 42},
                'cost_sensitive': {'enabled': False}
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
            },
            'explainability': {
                'enabled': True,
                'methods': ['shap'],
                'compliance': {'enabled': False}
            }
        }
        
        # Run complete workflow
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=temp_directory,
            verbose=True
        )
        
        result = orchestrator.run_pipeline('train')
        
        # Verify successful execution
        assert result['success'] == True
        assert result['mode'] == 'train'
        assert 'models' in result
        assert 'metrics' in result
        assert 'best_model' in result
        assert 'execution_time_seconds' in result
        
        # Verify all major stages were completed
        completed_stages = result['pipeline_state']['completed_stages']
        assert 'data_loading' in completed_stages
        assert 'preprocessing' in completed_stages
        assert 'model_training' in completed_stages
        assert 'model_evaluation' in completed_stages
        
        # Verify timing information
        assert 'stage_timings' in result
        assert len(result['stage_timings']) > 0
    
    def test_configuration_validation_workflow(self, temp_directory):
        """Test complete configuration validation workflow."""
        # Test with various configuration scenarios
        test_configs = [
            # Valid basic config
            {
                'data_source': {'type': 'csv', 'file_path': '/path/to/data.csv'},
                'preprocessing': {'target_column': 'target'},
                'model_training': {'algorithm': 'RandomForestClassifier'},
                'evaluation': {'test_size': 0.2},
                'execution': {'version': 'python', 'framework': 'sklearn'}
            },
            # Invalid config - missing data_source
            {
                'preprocessing': {'target_column': 'target'},
                'model_training': {'algorithm': 'RandomForestClassifier'},
                'evaluation': {'test_size': 0.2},
                'execution': {'version': 'python', 'framework': 'sklearn'}
            },
            # Invalid config - unsupported framework combination
            {
                'data_source': {'type': 'csv', 'file_path': '/path/to/data.csv'},
                'preprocessing': {'target_column': 'target'},
                'model_training': {'algorithm': 'XGBClassifier'},  # XGBoost algorithm
                'evaluation': {'test_size': 0.2},
                'execution': {'version': 'python', 'framework': 'sklearn'}  # But sklearn framework
            }
        ]
        
        expected_results = [True, False, True]  # Valid, Invalid, Valid (with warnings)
        
        for i, config in enumerate(test_configs):
            orchestrator = PipelineOrchestrator(
                config=config,
                output_dir=temp_directory
            )
            
            validation_result = orchestrator.validate_config()
            
            assert validation_result['valid'] == expected_results[i]
            assert isinstance(validation_result['errors'], list)
            assert isinstance(validation_result['warnings'], list)
            assert isinstance(validation_result['recommendations'], list)
    
    def test_execution_plan_generation_workflow(self, temp_directory):
        """Test execution plan generation for different modes."""
        base_config = {
            'data_source': {'type': 'csv', 'file_path': '/path/to/data.csv'},
            'preprocessing': {'target_column': 'target'},
            'model_training': {'algorithm': 'RandomForestClassifier'},
            'evaluation': {'test_size': 0.2},
            'execution': {'version': 'python', 'framework': 'sklearn'}
        }
        
        modes = ['train', 'predict', 'evaluate', 'compare', 'experiment']
        
        for mode in modes:
            config = base_config.copy()
            config['execution']['mode'] = mode
            
            orchestrator = PipelineOrchestrator(
                config=config,
                output_dir=temp_directory
            )
            
            execution_plan = orchestrator.get_execution_plan()
            
            assert execution_plan['mode'] == mode
            assert 'steps' in execution_plan
            assert len(execution_plan['steps']) > 0
            assert 'total_estimated_duration' in execution_plan
            assert 'resource_requirements' in execution_plan
            assert 'dependencies' in execution_plan
            assert 'outputs' in execution_plan
            
            # Verify mode-specific steps
            step_names = [step['name'] for step in execution_plan['steps']]
            
            if mode == 'train':
                assert any('Model Training' in name for name in step_names)
            elif mode == 'predict':
                assert any('Prediction' in name for name in step_names)
            elif mode == 'evaluate':
                assert any('Evaluation' in name for name in step_names)
            elif mode == 'compare':
                assert any('Comparison' in name for name in step_names)
            elif mode == 'experiment':
                assert any('Hyperparameter Optimization' in name for name in step_names)
    
    def test_error_handling_and_recovery_workflow(self, temp_directory):
        """Test error handling and recovery workflow."""
        config = {
            'data_source': {'type': 'csv', 'file_path': '/nonexistent/file.csv'},
            'preprocessing': {'target_column': 'target'},
            'model_training': {'algorithm': 'RandomForestClassifier'},
            'evaluation': {'test_size': 0.2},
            'execution': {'version': 'python', 'framework': 'sklearn'}
        }
        
        orchestrator = PipelineOrchestrator(
            config=config,
            output_dir=temp_directory
        )
        
        # Mock data connector to fail
        with patch('src.pipeline_orchestrator.DataConnectorFactory') as mock_factory:
            mock_factory.create_connector.side_effect = Exception("Data loading failed")
            
            result = orchestrator.run_pipeline('train')
            
            # Verify failure handling
            assert result['success'] == False
            assert 'error' in result
            assert 'partial_results' in result
            assert 'execution_time_seconds' in result
            
            # Verify partial results collection
            partial_results = result['partial_results']
            assert 'completed_stages' in partial_results
            assert 'failed_stages' in partial_results
            assert 'stage_timings' in partial_results
        
        # Test recovery by fixing the error and resuming
        fixed_config = config.copy()
        fixed_config['data_source']['file_path'] = '/path/to/valid/file.csv'
        
        # Create a new orchestrator with fixed config
        recovery_orchestrator = PipelineOrchestrator(
            config=fixed_config,
            output_dir=temp_directory
        )
        
        # Simulate resuming from a failed stage
        with patch.object(recovery_orchestrator, 'run_pipeline') as mock_run:
            mock_run.return_value = {'success': True, 'resumed': True}
            
            recovery_result = recovery_orchestrator.resume_pipeline('data_loading')
            
            assert recovery_result['success'] == True
            mock_run.assert_called_once()