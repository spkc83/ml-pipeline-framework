"""
Custom Airflow operators for ML Pipeline Framework

This module provides specialized operators for ML workflows including
data extraction, model training, evaluation, and deployment operations.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable, Connection
from airflow.hooks.base import BaseHook

# ML Pipeline Framework imports
try:
    from ml_pipeline_framework import PipelineOrchestrator
    from ml_pipeline_framework.utils import ConfigParser
    from ml_pipeline_framework.data_access import ConnectorFactory
    from ml_pipeline_framework.models import ModelFactory
    from ml_pipeline_framework.preprocessing import DataValidator
    from ml_pipeline_framework.evaluation import ModelEvaluator
    ML_FRAMEWORK_AVAILABLE = True
except ImportError:
    ML_FRAMEWORK_AVAILABLE = False
    logging.warning("ML Pipeline Framework not available. Some operators may not work.")

logger = logging.getLogger(__name__)


class MLPipelineBaseOperator(BaseOperator):
    """Base operator for ML Pipeline Framework operations."""
    
    @apply_defaults
    def __init__(self, 
                 config_path: Optional[str] = None,
                 mlflow_tracking_uri: Optional[str] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = config_path or Variable.get("ml_pipeline_config_path", "/opt/ml-pipeline/configs/pipeline_config.yaml")
        self.mlflow_tracking_uri = mlflow_tracking_uri or Variable.get("mlflow_tracking_uri", "http://mlflow-service:5000")
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Load ML pipeline configuration."""
        if not ML_FRAMEWORK_AVAILABLE:
            raise ImportError("ML Pipeline Framework is not available")
        
        return ConfigParser.from_yaml(self.config_path)


class ExtractDataOperator(MLPipelineBaseOperator):
    """Operator to extract data from various sources."""
    
    @apply_defaults
    def __init__(self,
                 connection_id: str,
                 query: Optional[str] = None,
                 table_name: Optional[str] = None,
                 output_path: Optional[str] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connection_id = connection_id
        self.query = query
        self.table_name = table_name
        self.output_path = output_path
    
    def execute(self, context):
        """Extract data from source."""
        logger.info(f"Extracting data using connection: {self.connection_id}")
        
        # Get connection details
        connection = BaseHook.get_connection(self.connection_id)
        
        if connection.conn_type == 'postgres':
            hook = PostgresHook(postgres_conn_id=self.connection_id)
            
            if self.query:
                df = hook.get_pandas_df(self.query)
            elif self.table_name:
                df = hook.get_pandas_df(f"SELECT * FROM {self.table_name}")
            else:
                raise ValueError("Either query or table_name must be provided")
        
        else:
            # Use ML Pipeline Framework connector
            if not ML_FRAMEWORK_AVAILABLE:
                raise ImportError("ML Pipeline Framework required for non-postgres connections")
            
            config = {
                'type': connection.conn_type,
                'connection_string': connection.get_uri()
            }
            
            connector = ConnectorFactory.create_connector(config)
            
            if self.query:
                df = connector.query(self.query)
            elif self.table_name:
                df = connector.query(f"SELECT * FROM {self.table_name}")
            else:
                raise ValueError("Either query or table_name must be provided")
        
        logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
        
        # Save data if output path specified
        if self.output_path:
            df.to_parquet(self.output_path, index=False)
            logger.info(f"Data saved to {self.output_path}")
        
        # Return data info for downstream tasks
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'output_path': self.output_path,
            'columns': list(df.columns)
        }


class DataQualityValidationOperator(MLPipelineBaseOperator):
    """Operator to validate data quality using Great Expectations."""
    
    @apply_defaults
    def __init__(self,
                 data_path: str,
                 expectation_suite: str,
                 fail_on_error: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.expectation_suite = expectation_suite
        self.fail_on_error = fail_on_error
    
    def execute(self, context):
        """Validate data quality."""
        if not ML_FRAMEWORK_AVAILABLE:
            raise ImportError("ML Pipeline Framework is required for data validation")
        
        logger.info(f"Validating data quality for: {self.data_path}")
        
        # Load data
        df = pd.read_parquet(self.data_path)
        
        # Initialize validator
        validator = DataValidator(
            expectation_suite=self.expectation_suite,
            fail_on_error=self.fail_on_error
        )
        
        # Run validation
        results = validator.validate(df)
        
        if not results['success'] and self.fail_on_error:
            raise ValueError(f"Data validation failed: {results}")
        
        logger.info(f"Data validation completed. Success: {results['success']}")
        
        return {
            'validation_success': results['success'],
            'validation_results': results,
            'data_path': self.data_path
        }


class TrainModelOperator(MLPipelineBaseOperator):
    """Operator to train ML models."""
    
    @apply_defaults
    def __init__(self,
                 data_path: str,
                 target_column: str,
                 experiment_name: str,
                 model_name: str,
                 test_size: float = 0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.target_column = target_column
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.test_size = test_size
    
    def execute(self, context):
        """Train ML model."""
        if not ML_FRAMEWORK_AVAILABLE:
            raise ImportError("ML Pipeline Framework is required for model training")
        
        logger.info(f"Training model: {self.model_name}")
        
        # Load configuration
        config = self.get_ml_config()
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            experiment_name=self.experiment_name,
            run_name=f"airflow_run_{context['ds_nodash']}"
        )
        
        # Load data
        df = pd.read_parquet(self.data_path)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Run training
        results = orchestrator.run_training(
            data=(X, y),
            test_size=self.test_size
        )
        
        logger.info(f"Model training completed. Results: {results}")
        
        return {
            'model_metrics': results.get('model_metrics', {}),
            'model_path': results.get('model_path', ''),
            'training_time': results.get('training_time', 0),
            'experiment_name': self.experiment_name,
            'model_name': self.model_name
        }


class EvaluateModelOperator(MLPipelineBaseOperator):
    """Operator to evaluate trained models."""
    
    @apply_defaults
    def __init__(self,
                 model_path: str,
                 test_data_path: str,
                 target_column: str,
                 metrics: List[str] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.target_column = target_column
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    def execute(self, context):
        """Evaluate model performance."""
        if not ML_FRAMEWORK_AVAILABLE:
            raise ImportError("ML Pipeline Framework is required for model evaluation")
        
        logger.info(f"Evaluating model: {self.model_path}")
        
        # Load test data
        df = pd.read_parquet(self.test_data_path)
        X_test = df.drop(columns=[self.target_column])
        y_test = df[self.target_column]
        
        # Load model and orchestrator
        orchestrator = PipelineOrchestrator.from_saved_pipeline(self.model_path)
        
        # Run evaluation
        evaluation_results = orchestrator.run_evaluation(
            test_data=(X_test, y_test),
            metrics=self.metrics
        )
        
        logger.info(f"Model evaluation completed: {evaluation_results}")
        
        return {
            'evaluation_metrics': evaluation_results.get('metrics', {}),
            'model_path': self.model_path,
            'test_data_path': self.test_data_path
        }


class RegisterModelMLflowOperator(MLPipelineBaseOperator):
    """Operator to register models in MLflow Model Registry."""
    
    @apply_defaults
    def __init__(self,
                 model_path: str,
                 model_name: str,
                 stage: str = "Staging",
                 description: str = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.model_name = model_name
        self.stage = stage
        self.description = description
    
    def execute(self, context):
        """Register model in MLflow."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            raise ImportError("MLflow is required for model registration")
        
        logger.info(f"Registering model: {self.model_name}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        client = MlflowClient()
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=f"file://{self.model_path}",
            name=self.model_name
        )
        
        # Transition to specified stage
        client.transition_model_version_stage(
            name=self.model_name,
            version=model_version.version,
            stage=self.stage
        )
        
        # Update description if provided
        if self.description:
            client.update_model_version(
                name=self.model_name,
                version=model_version.version,
                description=self.description
            )
        
        logger.info(f"Model registered: {self.model_name} v{model_version.version}")
        
        return {
            'model_name': self.model_name,
            'version': model_version.version,
            'stage': self.stage,
            'model_uri': f"models:/{self.model_name}/{model_version.version}"
        }


class RunPredictionsOperator(MLPipelineBaseOperator):
    """Operator to run batch predictions."""
    
    @apply_defaults
    def __init__(self,
                 model_path: str,
                 input_data_path: str,
                 output_path: str,
                 batch_size: int = 1000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.input_data_path = input_data_path
        self.output_path = output_path
        self.batch_size = batch_size
    
    def execute(self, context):
        """Run batch predictions."""
        if not ML_FRAMEWORK_AVAILABLE:
            raise ImportError("ML Pipeline Framework is required for predictions")
        
        logger.info(f"Running predictions with model: {self.model_path}")
        
        # Load input data
        df = pd.read_parquet(self.input_data_path)
        
        # Load model
        orchestrator = PipelineOrchestrator.from_saved_pipeline(self.model_path)
        
        # Run predictions in batches
        predictions = []
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i+self.batch_size]
            batch_predictions = orchestrator.run_prediction(batch, return_probabilities=True)
            predictions.extend(batch_predictions)
        
        # Save predictions
        if isinstance(predictions[0], tuple):
            # Handle case where probabilities are returned
            pred_classes = [p[0] for p in predictions]
            pred_probs = [p[1] for p in predictions]
            
            results_df = pd.DataFrame({
                'prediction': pred_classes,
                'probability': [prob[1] if len(prob) > 1 else prob[0] for prob in pred_probs]
            })
        else:
            results_df = pd.DataFrame({'prediction': predictions})
        
        # Add original data identifiers if available
        if 'id' in df.columns:
            results_df['id'] = df['id'].values
        
        results_df.to_parquet(self.output_path, index=False)
        
        logger.info(f"Predictions completed. Results saved to: {self.output_path}")
        
        return {
            'prediction_count': len(results_df),
            'output_path': self.output_path,
            'input_data_path': self.input_data_path
        }


class ModelMonitoringOperator(MLPipelineBaseOperator):
    """Operator to setup model monitoring."""
    
    @apply_defaults
    def __init__(self,
                 model_name: str,
                 reference_data_path: str,
                 monitoring_frequency: str = "daily",
                 drift_threshold: float = 0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.reference_data_path = reference_data_path
        self.monitoring_frequency = monitoring_frequency
        self.drift_threshold = drift_threshold
    
    def execute(self, context):
        """Setup model monitoring."""
        logger.info(f"Setting up monitoring for model: {self.model_name}")
        
        # Load reference data
        reference_data = pd.read_parquet(self.reference_data_path)
        
        # Setup monitoring configuration
        monitoring_config = {
            'model_name': self.model_name,
            'reference_data_path': self.reference_data_path,
            'monitoring_frequency': self.monitoring_frequency,
            'drift_threshold': self.drift_threshold,
            'setup_date': datetime.now().isoformat(),
            'reference_data_stats': {
                'row_count': len(reference_data),
                'column_count': len(reference_data.columns),
                'columns': list(reference_data.columns)
            }
        }
        
        # Save monitoring configuration
        config_path = f"/opt/ml-pipeline/monitoring/{self.model_name}_monitoring_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        import json
        with open(config_path, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        logger.info(f"Monitoring setup completed for {self.model_name}")
        
        return monitoring_config


class DeployModelOperator(MLPipelineBaseOperator):
    """Operator to deploy models to different environments."""
    
    @apply_defaults
    def __init__(self,
                 model_name: str,
                 model_version: str,
                 environment: str,
                 deployment_config: Dict[str, Any] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.model_version = model_version
        self.environment = environment
        self.deployment_config = deployment_config or {}
    
    def execute(self, context):
        """Deploy model to specified environment."""
        logger.info(f"Deploying model {self.model_name} v{self.model_version} to {self.environment}")
        
        # Create deployment manifest
        deployment_info = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'environment': self.environment,
            'deployment_date': datetime.now().isoformat(),
            'deployment_config': self.deployment_config,
            'status': 'deployed'
        }
        
        # Save deployment info
        deployment_path = f"/opt/ml-pipeline/deployments/{self.environment}/{self.model_name}_v{self.model_version}.json"
        os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
        
        import json
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Model deployment completed: {deployment_path}")
        
        return deployment_info


# Convenience functions for use in PythonOperator
def extract_data(**context):
    """Extract data function for PythonOperator."""
    return ExtractDataOperator(
        task_id='extract_data',
        connection_id=context['params'].get('connection_id', 'postgres_default'),
        query=context['params'].get('query'),
        table_name=context['params'].get('table_name')
    ).execute(context)


def validate_data_quality(**context):
    """Validate data quality function for PythonOperator."""
    return DataQualityValidationOperator(
        task_id='validate_data_quality',
        data_path=context['params'].get('data_path'),
        expectation_suite=context['params'].get('expectation_suite'),
        fail_on_error=context['params'].get('fail_on_error', True)
    ).execute(context)


def evaluate_model(**context):
    """Evaluate model function for PythonOperator."""
    return EvaluateModelOperator(
        task_id='evaluate_model',
        model_path=context['params'].get('model_path'),
        test_data_path=context['params'].get('test_data_path'),
        target_column=context['params'].get('target_column'),
        metrics=context['params'].get('metrics')
    ).execute(context)


def register_model_mlflow(**context):
    """Register model in MLflow function for PythonOperator."""
    return RegisterModelMLflowOperator(
        task_id='register_model',
        model_path=context['params'].get('model_path'),
        model_name=context['params'].get('model_name'),
        stage=context['params'].get('stage', 'Staging'),
        description=context['params'].get('description')
    ).execute(context)


def setup_model_monitoring(**context):
    """Setup model monitoring function for PythonOperator."""
    return ModelMonitoringOperator(
        task_id='setup_monitoring',
        model_name=context['params'].get('model_name'),
        reference_data_path=context['params'].get('reference_data_path'),
        monitoring_frequency=context['params'].get('monitoring_frequency', 'daily'),
        drift_threshold=context['params'].get('drift_threshold', 0.1)
    ).execute(context)