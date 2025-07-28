import logging
import os
import pickle
import json
import tempfile
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import shutil

logger = logging.getLogger(__name__)

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    import mlflow.catboost
    import mlflow.pytorch
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow not available. Install with: pip install mlflow")
    MLFLOW_AVAILABLE = False
    mlflow = MlflowClient = ViewType = MlflowException = None

# Model-specific imports for auto-detection
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None


class MLflowError(Exception):
    pass


class MLflowTracker:
    """
    Comprehensive MLflow tracking utility for logging parameters, metrics, artifacts,
    and models with automated experiment management and run lifecycle handling.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 auto_create_experiment: bool = True,
                 nested_runs: bool = False):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (None for local)
            experiment_name: Name of MLflow experiment
            auto_create_experiment: Whether to auto-create experiment if it doesn't exist
            nested_runs: Whether to support nested runs
        """
        if not MLFLOW_AVAILABLE:
            raise MLflowError("MLflow is required. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.auto_create_experiment = auto_create_experiment
        self.nested_runs = nested_runs
        
        # Initialize MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self.current_run = None
        self.experiment_id = None
        
        # Setup experiment
        if experiment_name:
            self._setup_experiment(experiment_name)
        
        # Run context stack for nested runs
        self.run_stack = []
        
        logger.info(f"Initialized MLflowTracker with experiment: {experiment_name}")
    
    def _setup_experiment(self, experiment_name: str) -> str:
        """Setup or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if self.auto_create_experiment:
                    self.experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new experiment: {experiment_name}")
                else:
                    raise MLflowError(f"Experiment '{experiment_name}' not found and auto_create_experiment=False")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
            return self.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise MLflowError(f"Experiment setup failed: {e}")
    
    def create_experiment(self, experiment_name: str, 
                         artifact_location: Optional[str] = None,
                         tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new MLflow experiment.
        
        Args:
            experiment_name: Name of the experiment
            artifact_location: Location to store artifacts (S3, HDFS, etc.)
            tags: Tags for the experiment
            
        Returns:
            Experiment ID
        """
        logger.info(f"Creating experiment: {experiment_name}")
        
        try:
            # Check if experiment already exists
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment is not None:
                logger.warning(f"Experiment '{experiment_name}' already exists")
                return existing_experiment.experiment_id
            
            # Create experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
                tags=tags
            )
            
            self.experiment_name = experiment_name
            self.experiment_id = experiment_id
            
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise MLflowError(f"Experiment creation failed: {e}")
    
    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  nested: bool = False) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags for the run
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        try:
            # Handle nested runs
            if nested and self.current_run is not None:
                self.run_stack.append(self.current_run)
            elif not nested and self.current_run is not None:
                logger.warning("Starting new run while another is active. Ending current run.")
                self.end_run()
            
            # Start new run
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested
            )
            
            self.current_run = run
            
            # Set default tags
            default_tags = {
                'mlflow.user': os.environ.get('USER', 'unknown'),
                'framework': 'ml-pipeline-framework',
                'start_time': datetime.now().isoformat()
            }
            
            if tags:
                default_tags.update(tags)
            
            # Log tags
            for key, value in default_tags.items():
                mlflow.set_tag(key, value)
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            raise MLflowError(f"Run start failed: {e}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")
        """
        try:
            if self.current_run is not None:
                # Log end time
                mlflow.set_tag('end_time', datetime.now().isoformat())
                
                # End run
                mlflow.end_run(status=status)
                
                logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
                
                # Handle nested runs
                if self.run_stack:
                    self.current_run = self.run_stack.pop()
                else:
                    self.current_run = None
            else:
                logger.warning("No active run to end")
                
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
            raise MLflowError(f"Run end failed: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            # Convert parameters to strings and handle nested dictionaries
            processed_params = self._process_params(params)
            
            for key, value in processed_params.items():
                mlflow.log_param(key, value)
            
            logger.debug(f"Logged {len(processed_params)} parameters")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise MLflowError(f"Parameter logging failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], 
                   step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for time series metrics
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    mlflow.log_metric(key, float(value), step=step)
                else:
                    logger.warning(f"Skipping metric '{key}' with invalid value: {value}")
            
            logger.debug(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise MLflowError(f"Metric logging failed: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact to MLflow.
        
        Args:
            local_path: Local path to artifact file
            artifact_path: Relative path within run's artifact directory
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            if not os.path.exists(local_path):
                raise MLflowError(f"Artifact file not found: {local_path}")
            
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise MLflowError(f"Artifact logging failed: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log directory of artifacts to MLflow.
        
        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Relative path within run's artifact directory
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            if not os.path.exists(local_dir):
                raise MLflowError(f"Artifact directory not found: {local_dir}")
            
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from directory: {local_dir}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            raise MLflowError(f"Artifacts logging failed: {e}")
    
    def log_model(self, model: Any, artifact_path: str = "model",
                  signature: Optional[Any] = None,
                  input_example: Optional[Any] = None,
                  conda_env: Optional[str] = None,
                  pip_requirements: Optional[List[str]] = None) -> None:
        """
        Log model to MLflow with auto-detection of model type.
        
        Args:
            model: Trained model object
            artifact_path: Relative path to store model
            signature: Model signature (input/output schema)
            input_example: Example input for the model
            conda_env: Conda environment file
            pip_requirements: List of pip requirements
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            # Auto-detect model type and log appropriately
            model_type = self._detect_model_type(model)
            
            logger.info(f"Logging {model_type} model to MLflow")
            
            if model_type == "sklearn" and SKLEARN_AVAILABLE:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            
            elif model_type == "xgboost" and XGBOOST_AVAILABLE:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            
            elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            
            elif model_type == "catboost" and CATBOOST_AVAILABLE:
                mlflow.catboost.log_model(
                    cb_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            
            else:
                # Fallback to pickle
                logger.warning(f"Using pickle for {model_type} model")
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(model, f)
                    temp_path = f.name
                
                try:
                    self.log_artifact(temp_path, f"{artifact_path}/model.pkl")
                finally:
                    os.unlink(temp_path)
            
            # Log model metadata
            model_info = {
                'model_type': model_type,
                'model_class': model.__class__.__name__,
                'model_module': model.__class__.__module__,
                'artifact_path': artifact_path
            }
            
            # Try to get model parameters
            if hasattr(model, 'get_params'):
                try:
                    model_params = model.get_params()
                    self.log_params({f"model_{k}": v for k, v in model_params.items()})
                except:
                    pass
            
            mlflow.set_tag('model_info', json.dumps(model_info))
            logger.info(f"Successfully logged {model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise MLflowError(f"Model logging failed: {e}")
    
    def log_dataset(self, dataset: Union[pd.DataFrame, np.ndarray],
                   name: str = "dataset", format: str = "csv") -> None:
        """
        Log dataset as artifact.
        
        Args:
            dataset: Dataset to log
            name: Name for the dataset file
            format: File format ('csv', 'parquet', 'json')
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as f:
                temp_path = f.name
            
            try:
                if isinstance(dataset, pd.DataFrame):
                    if format == 'csv':
                        dataset.to_csv(temp_path, index=False)
                    elif format == 'parquet':
                        dataset.to_parquet(temp_path, index=False)
                    elif format == 'json':
                        dataset.to_json(temp_path, orient='records')
                    else:
                        raise ValueError(f"Unsupported format for DataFrame: {format}")
                
                elif isinstance(dataset, np.ndarray):
                    if format == 'csv':
                        np.savetxt(temp_path, dataset, delimiter=',')
                    else:
                        # Save as numpy binary format
                        np.save(temp_path.replace(f'.{format}', '.npy'), dataset)
                        temp_path = temp_path.replace(f'.{format}', '.npy')
                else:
                    raise ValueError(f"Unsupported dataset type: {type(dataset)}")
                
                # Log the dataset file
                self.log_artifact(temp_path, f"datasets/{name}.{format}")
                
                # Log dataset metadata
                dataset_info = {
                    'name': name,
                    'format': format,
                    'shape': list(dataset.shape) if hasattr(dataset, 'shape') else None,
                    'type': str(type(dataset).__name__),
                    'size_bytes': os.path.getsize(temp_path)
                }
                
                mlflow.set_tag(f'dataset_{name}_info', json.dumps(dataset_info))
                logger.info(f"Logged dataset '{name}' with shape {dataset.shape}")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Failed to log dataset: {e}")
            raise MLflowError(f"Dataset logging failed: {e}")
    
    def log_figure(self, figure, filename: str, artifact_path: str = "plots") -> None:
        """
        Log matplotlib figure as artifact.
        
        Args:
            figure: Matplotlib figure object
            filename: Filename for the plot
            artifact_path: Artifact path within run
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            try:
                figure.savefig(temp_path, dpi=300, bbox_inches='tight')
                self.log_artifact(temp_path, f"{artifact_path}/{filename}")
                logger.debug(f"Logged figure: {filename}")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")
            raise MLflowError(f"Figure logging failed: {e}")
    
    def log_text(self, text: str, filename: str, artifact_path: str = "reports") -> None:
        """
        Log text content as artifact.
        
        Args:
            text: Text content to log
            filename: Filename for the text file
            artifact_path: Artifact path within run
        """
        try:
            if self.current_run is None:
                raise MLflowError("No active run. Call start_run() first.")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_path = f.name
            
            try:
                self.log_artifact(temp_path, f"{artifact_path}/{filename}")
                logger.debug(f"Logged text file: {filename}")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Failed to log text: {e}")
            raise MLflowError(f"Text logging failed: {e}")
    
    def search_runs(self, filter_string: Optional[str] = None,
                   run_view_type: str = "ACTIVE_ONLY",
                   max_results: int = 1000) -> pd.DataFrame:
        """
        Search runs in the current experiment.
        
        Args:
            filter_string: Filter string for runs
            run_view_type: Type of runs to return ("ACTIVE_ONLY", "DELETED_ONLY", "ALL")
            max_results: Maximum number of results
            
        Returns:
            DataFrame with run information
        """
        try:
            if self.experiment_id is None:
                raise MLflowError("No experiment set. Set experiment first.")
            
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                run_view_type=getattr(ViewType, run_view_type),
                max_results=max_results
            )
            
            logger.info(f"Found {len(runs)} runs")
            return runs
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            raise MLflowError(f"Run search failed: {e}")
    
    def get_best_run(self, metric: str, mode: str = "max") -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to optimize
            mode: "max" or "min"
            
        Returns:
            Best run information
        """
        try:
            if mode == "max":
                order_by = [f"metrics.{metric} DESC"]
            else:
                order_by = [f"metrics.{metric} ASC"]
            
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=order_by,
                max_results=1
            )
            
            if runs:
                best_run = runs[0]
                logger.info(f"Best run by {metric}: {best_run.info.run_id}")
                
                return {
                    'run_id': best_run.info.run_id,
                    'metrics': best_run.data.metrics,
                    'params': best_run.data.params,
                    'tags': best_run.data.tags,
                    'start_time': best_run.info.start_time,
                    'end_time': best_run.info.end_time,
                    'status': best_run.info.status
                }
            else:
                logger.warning("No runs found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            raise MLflowError(f"Best run search failed: {e}")
    
    def load_model(self, run_id: str, artifact_path: str = "model") -> Any:
        """
        Load model from a specific run.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact
            
        Returns:
            Loaded model object
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Try to load with specific MLflow loaders first
            run = self.client.get_run(run_id)
            model_info_tag = run.data.tags.get('model_info')
            
            if model_info_tag:
                model_info = json.loads(model_info_tag)
                model_type = model_info.get('model_type')
                
                if model_type == "sklearn" and SKLEARN_AVAILABLE:
                    return mlflow.sklearn.load_model(model_uri)
                elif model_type == "xgboost" and XGBOOST_AVAILABLE:
                    return mlflow.xgboost.load_model(model_uri)
                elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                    return mlflow.lightgbm.load_model(model_uri)
                elif model_type == "catboost" and CATBOOST_AVAILABLE:
                    return mlflow.catboost.load_model(model_uri)
            
            # Fallback to generic MLflow loader
            try:
                return mlflow.pyfunc.load_model(model_uri)
            except:
                # Last resort: try to load pickle file
                artifacts = self.client.list_artifacts(run_id, artifact_path)
                for artifact in artifacts:
                    if artifact.path.endswith('.pkl'):
                        artifact_path = self.client.download_artifacts(run_id, artifact.path)
                        with open(artifact_path, 'rb') as f:
                            return pickle.load(f)
                
                raise MLflowError("Could not load model with any method")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise MLflowError(f"Model loading failed: {e}")
    
    def delete_run(self, run_id: str) -> None:
        """
        Delete a specific run.
        
        Args:
            run_id: MLflow run ID to delete
        """
        try:
            self.client.delete_run(run_id)
            logger.info(f"Deleted run: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete run: {e}")
            raise MLflowError(f"Run deletion failed: {e}")
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs across specified metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        try:
            comparison_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                
                row = {
                    'run_id': run_id,
                    'start_time': run.info.start_time,
                    'status': run.info.status
                }
                
                # Add requested metrics
                for metric in metrics:
                    row[f'metrics.{metric}'] = run.data.metrics.get(metric)
                
                # Add some common parameters
                for param_key, param_value in run.data.params.items():
                    row[f'params.{param_key}'] = param_value
                
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            logger.info(f"Compared {len(run_ids)} runs")
            return df
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            raise MLflowError(f"Run comparison failed: {e}")
    
    def _detect_model_type(self, model: Any) -> str:
        """Detect the type of model for appropriate logging."""
        model_class = model.__class__.__name__
        model_module = model.__class__.__module__
        
        if SKLEARN_AVAILABLE and 'sklearn' in model_module:
            return "sklearn"
        elif XGBOOST_AVAILABLE and ('xgb' in model_class.lower() or 'xgboost' in model_module):
            return "xgboost"
        elif LIGHTGBM_AVAILABLE and ('lgb' in model_class.lower() or 'lightgbm' in model_module):
            return "lightgbm"
        elif CATBOOST_AVAILABLE and ('catboost' in model_class.lower() or 'catboost' in model_module):
            return "catboost"
        elif 'torch' in model_module:
            return "pytorch"
        elif 'tensorflow' in model_module or 'keras' in model_module:
            return "tensorflow"
        else:
            return "unknown"
    
    def _process_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Process parameters for MLflow logging (flatten nested dicts)."""
        processed = {}
        
        for key, value in params.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_params = self._process_params(value, f"{full_key}_")
                processed.update(nested_params)
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to string representation
                processed[full_key] = str(value)
            elif value is None:
                processed[full_key] = "None"
            elif isinstance(value, (int, float, str, bool)):
                processed[full_key] = str(value)
            else:
                # Convert other types to string
                processed[full_key] = str(value)
        
        return processed
    
    def create_model_version(self, model_name: str, run_id: str, 
                           artifact_path: str = "model",
                           description: Optional[str] = None) -> Dict[str, Any]:
        """
        Register model version in MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            run_id: Run ID containing the model
            artifact_path: Path to model artifact within run
            description: Description for this model version
            
        Returns:
            Model version information
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={'run_id': run_id, 'artifact_path': artifact_path}
            )
            
            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model '{model_name}' version {model_version.version}")
            
            return {
                'name': model_version.name,
                'version': model_version.version,
                'creation_timestamp': model_version.creation_timestamp,
                'source': model_version.source,
                'run_id': model_version.run_id
            }
            
        except Exception as e:
            logger.error(f"Failed to register model version: {e}")
            raise MLflowError(f"Model registration failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - end any active runs."""
        if self.current_run is not None:
            if exc_type is not None:
                self.end_run(status="FAILED")
            else:
                self.end_run(status="FINISHED")
    
    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a run.
        
        Args:
            run_id: Run ID (uses current run if None)
            
        Returns:
            Run information dictionary
        """
        try:
            if run_id is None:
                if self.current_run is None:
                    raise MLflowError("No active run and no run_id provided")
                run_id = self.current_run.info.run_id
            
            run = self.client.get_run(run_id)
            
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags
            }
            
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            raise MLflowError(f"Run info retrieval failed: {e}")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                           labels: Optional[List[str]] = None,
                           artifact_path: str = "plots") -> None:
        """
        Log confusion matrix as both metric and plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            artifact_path: Path for plot artifact
        """
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Log confusion matrix values as metrics
            if labels is None:
                labels = [f"class_{i}" for i in range(len(cm))]
            
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    self.log_metrics({f"cm_{true_label}_{pred_label}": cm[i, j]})
            
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Log plot
            self.log_figure(plt.gcf(), "confusion_matrix.png", artifact_path)
            plt.close()
            
            logger.info("Logged confusion matrix")
            
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
            raise MLflowError(f"Confusion matrix logging failed: {e}")
    
    def auto_log(self, framework: Optional[str] = None) -> None:
        """
        Enable automatic logging for supported frameworks.
        
        Args:
            framework: Framework to enable autologging for (None for all)
        """
        try:
            if framework is None or framework == "sklearn":
                if SKLEARN_AVAILABLE:
                    mlflow.sklearn.autolog()
                    logger.info("Enabled sklearn autologging")
            
            if framework is None or framework == "xgboost":
                if XGBOOST_AVAILABLE:
                    mlflow.xgboost.autolog()
                    logger.info("Enabled XGBoost autologging")
            
            if framework is None or framework == "lightgbm":
                if LIGHTGBM_AVAILABLE:
                    mlflow.lightgbm.autolog()
                    logger.info("Enabled LightGBM autologging")
            
            if framework is None or framework == "catboost":
                if CATBOOST_AVAILABLE:
                    mlflow.catboost.autolog()
                    logger.info("Enabled CatBoost autologging")
                    
        except Exception as e:
            logger.warning(f"Failed to enable autologging: {e}")
    
    def disable_auto_log(self) -> None:
        """Disable automatic logging."""
        try:
            mlflow.autolog(disable=True)
            logger.info("Disabled autologging")
        except Exception as e:
            logger.warning(f"Failed to disable autologging: {e}")