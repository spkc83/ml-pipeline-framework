#!/usr/bin/env python3
"""
Hadoop Batch Job Submission Script for ML Pipeline Framework

This script provides functionality to submit ML pipeline jobs to Hadoop cluster
using YARN and Spark. Supports different job types including training, prediction,
and data processing workflows.
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hadoop_job_submission.log')
    ]
)
logger = logging.getLogger(__name__)


class HadoopJobSubmitter:
    """Submit ML Pipeline Framework jobs to Hadoop cluster."""
    
    def __init__(self, config_path: str = None, hadoop_conf_dir: str = None):
        """
        Initialize Hadoop job submitter.
        
        Args:
            config_path: Path to job configuration file
            hadoop_conf_dir: Path to Hadoop configuration directory
        """
        self.config_path = config_path
        self.hadoop_conf_dir = hadoop_conf_dir or os.environ.get('HADOOP_CONF_DIR', '/etc/hadoop/conf')
        self.spark_home = os.environ.get('SPARK_HOME', '/opt/spark')
        self.job_config = {}
        
        # Validate environment
        self._validate_environment()
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def _validate_environment(self):
        """Validate Hadoop and Spark environment."""
        logger.info("Validating Hadoop/Spark environment...")
        
        # Check Hadoop configuration
        if not os.path.exists(self.hadoop_conf_dir):
            raise EnvironmentError(f"Hadoop configuration directory not found: {self.hadoop_conf_dir}")
        
        # Check Spark installation
        if not os.path.exists(self.spark_home):
            raise EnvironmentError(f"Spark home directory not found: {self.spark_home}")
        
        spark_submit = os.path.join(self.spark_home, 'bin', 'spark-submit')
        if not os.path.exists(spark_submit):
            raise EnvironmentError(f"spark-submit not found: {spark_submit}")
        
        # Check YARN connectivity
        try:
            result = subprocess.run(['yarn', 'application', '-list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning("Cannot connect to YARN ResourceManager")
        except Exception as e:
            logger.warning(f"YARN connectivity check failed: {e}")
        
        logger.info("Environment validation completed")
    
    def load_config(self, config_path: str):
        """
        Load job configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            self.job_config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['name', 'job_type', 'spark_config']
        for field in required_fields:
            if field not in self.job_config:
                raise ValueError(f"Required field '{field}' not found in configuration")
        
        logger.info(f"Configuration loaded for job: {self.job_config['name']}")
    
    def submit_training_job(self, **kwargs) -> str:
        """
        Submit ML model training job to Hadoop cluster.
        
        Returns:
            Application ID of submitted job
        """
        logger.info("Submitting training job to Hadoop cluster...")
        
        # Build Spark configuration for training
        spark_config = self._build_spark_config('training', **kwargs)
        
        # Set training-specific parameters
        spark_config.update({
            '--class': 'org.apache.spark.deploy.SparkSubmit',
            '--files': self._get_config_files(),
            '--py-files': self._get_python_files(),
            '--conf': [
                'spark.dynamicAllocation.enabled=true',
                'spark.dynamicAllocation.minExecutors=2',
                'spark.dynamicAllocation.maxExecutors=20',
                'spark.sql.adaptive.enabled=true',
                'spark.sql.adaptive.coalescePartitions.enabled=true'
            ]
        })
        
        # Build command
        cmd = self._build_spark_submit_command(
            app_file=kwargs.get('app_file', 'src/scripts/hadoop_train.py'),
            spark_config=spark_config,
            app_args=kwargs.get('app_args', [])
        )
        
        return self._execute_spark_submit(cmd, job_type='training')
    
    def submit_prediction_job(self, **kwargs) -> str:
        """
        Submit batch prediction job to Hadoop cluster.
        
        Returns:
            Application ID of submitted job
        """
        logger.info("Submitting prediction job to Hadoop cluster...")
        
        # Build Spark configuration for prediction
        spark_config = self._build_spark_config('prediction', **kwargs)
        
        # Set prediction-specific parameters
        spark_config.update({
            '--class': 'org.apache.spark.deploy.SparkSubmit',
            '--files': self._get_config_files(),
            '--py-files': self._get_python_files(),
            '--conf': [
                'spark.dynamicAllocation.enabled=true',
                'spark.dynamicAllocation.minExecutors=1',
                'spark.dynamicAllocation.maxExecutors=10',
                'spark.sql.adaptive.enabled=true'
            ]
        })
        
        # Build command
        cmd = self._build_spark_submit_command(
            app_file=kwargs.get('app_file', 'src/scripts/hadoop_predict.py'),
            spark_config=spark_config,
            app_args=kwargs.get('app_args', [])
        )
        
        return self._execute_spark_submit(cmd, job_type='prediction')
    
    def submit_data_processing_job(self, **kwargs) -> str:
        """
        Submit data processing job to Hadoop cluster.
        
        Returns:
            Application ID of submitted job
        """
        logger.info("Submitting data processing job to Hadoop cluster...")
        
        # Build Spark configuration for data processing
        spark_config = self._build_spark_config('data_processing', **kwargs)
        
        # Set data processing-specific parameters
        spark_config.update({
            '--class': 'org.apache.spark.deploy.SparkSubmit',
            '--files': self._get_config_files(),
            '--py-files': self._get_python_files(),
            '--conf': [
                'spark.dynamicAllocation.enabled=true',
                'spark.dynamicAllocation.minExecutors=5',
                'spark.dynamicAllocation.maxExecutors=50',
                'spark.sql.adaptive.enabled=true',
                'spark.sql.adaptive.coalescePartitions.enabled=true',
                'spark.sql.adaptive.skewJoin.enabled=true'
            ]
        })
        
        # Build command
        cmd = self._build_spark_submit_command(
            app_file=kwargs.get('app_file', 'src/scripts/hadoop_data_processing.py'),
            spark_config=spark_config,
            app_args=kwargs.get('app_args', [])
        )
        
        return self._execute_spark_submit(cmd, job_type='data_processing')
    
    def submit_model_evaluation_job(self, **kwargs) -> str:
        """
        Submit model evaluation job to Hadoop cluster.
        
        Returns:
            Application ID of submitted job
        """
        logger.info("Submitting model evaluation job to Hadoop cluster...")
        
        # Build Spark configuration for evaluation
        spark_config = self._build_spark_config('evaluation', **kwargs)
        
        # Build command
        cmd = self._build_spark_submit_command(
            app_file=kwargs.get('app_file', 'src/scripts/hadoop_evaluate.py'),
            spark_config=spark_config,
            app_args=kwargs.get('app_args', [])
        )
        
        return self._execute_spark_submit(cmd, job_type='evaluation')
    
    def _build_spark_config(self, job_type: str, **kwargs) -> Dict[str, Any]:
        """Build Spark configuration based on job type and user inputs."""
        # Default configurations
        base_config = {
            '--master': 'yarn',
            '--deploy-mode': 'cluster',
            '--name': f"ml-pipeline-{job_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            '--queue': kwargs.get('queue', 'default'),
            '--num-executors': kwargs.get('num_executors', 10),
            '--executor-cores': kwargs.get('executor_cores', 4),
            '--executor-memory': kwargs.get('executor_memory', '8g'),
            '--driver-memory': kwargs.get('driver_memory', '4g'),
            '--driver-cores': kwargs.get('driver_cores', 2),
            '--conf': []
        }
        
        # Merge with job configuration if available
        if self.job_config and 'spark_config' in self.job_config:
            spark_conf = self.job_config['spark_config']
            for key, value in spark_conf.items():
                if key.startswith('spark.'):
                    base_config['--conf'].append(f"{key}={value}")
                else:
                    base_config[f"--{key}"] = value
        
        # Add environment-specific configurations
        env_conf = [
            'spark.hadoop.fs.defaultFS=hdfs://namenode:9000',
            'spark.hadoop.yarn.resourcemanager.hostname=resourcemanager',
            'spark.eventLog.enabled=true',
            'spark.eventLog.dir=hdfs://namenode:9000/spark-logs',
            'spark.history.fs.logDirectory=hdfs://namenode:9000/spark-logs',
            'spark.sql.warehouse.dir=hdfs://namenode:9000/spark-warehouse'
        ]
        
        base_config['--conf'].extend(env_conf)
        
        # Add additional configurations from kwargs
        if 'additional_conf' in kwargs:
            base_config['--conf'].extend(kwargs['additional_conf'])
        
        return base_config
    
    def _build_spark_submit_command(self, app_file: str, spark_config: Dict[str, Any], 
                                  app_args: List[str] = None) -> List[str]:
        """Build spark-submit command."""
        cmd = [os.path.join(self.spark_home, 'bin', 'spark-submit')]
        
        # Add configuration parameters
        for key, value in spark_config.items():
            if key == '--conf':
                for conf in value:
                    cmd.extend(['--conf', conf])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    cmd.extend([key, str(item)])
            else:
                cmd.extend([key, str(value)])
        
        # Add application file
        cmd.append(app_file)
        
        # Add application arguments
        if app_args:
            cmd.extend(app_args)
        
        return cmd
    
    def _execute_spark_submit(self, cmd: List[str], job_type: str) -> str:
        """Execute spark-submit command and return application ID."""
        logger.info(f"Executing {job_type} job with command: {' '.join(cmd)}")
        
        try:
            # Execute spark-submit
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Job submission failed: {result.stderr}")
                raise RuntimeError(f"Spark job submission failed: {result.stderr}")
            
            # Extract application ID from output
            app_id = self._extract_application_id(result.stdout)
            
            if app_id:
                logger.info(f"Job submitted successfully with Application ID: {app_id}")
                
                # Log job details
                self._log_job_details(app_id, job_type, cmd)
                
                return app_id
            else:
                raise RuntimeError("Could not extract application ID from Spark output")
                
        except subprocess.TimeoutExpired:
            logger.error("Spark job submission timed out")
            raise RuntimeError("Job submission timed out")
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise
    
    def _extract_application_id(self, output: str) -> Optional[str]:
        """Extract application ID from Spark submit output."""
        import re
        
        # Look for application ID pattern
        pattern = r'application_\d+_\d+'
        match = re.search(pattern, output)
        
        if match:
            return match.group(0)
        
        return None
    
    def _log_job_details(self, app_id: str, job_type: str, cmd: List[str]):
        """Log job submission details."""
        job_details = {
            'application_id': app_id,
            'job_type': job_type,
            'submission_time': datetime.now().isoformat(),
            'command': ' '.join(cmd),
            'config': self.job_config
        }
        
        # Save to log file
        log_file = f"hadoop_jobs_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing logs or create new
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(job_details)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _get_config_files(self) -> str:
        """Get comma-separated list of configuration files to distribute."""
        config_files = []
        
        # Add pipeline configuration
        if self.config_path and os.path.exists(self.config_path):
            config_files.append(self.config_path)
        
        # Add default configuration files
        default_configs = [
            'configs/pipeline_config.yaml',
            'configs/spark_config.yaml',
            'configs/hadoop_config.yaml'
        ]
        
        for config in default_configs:
            if os.path.exists(config):
                config_files.append(config)
        
        return ','.join(config_files) if config_files else ''
    
    def _get_python_files(self) -> str:
        """Get comma-separated list of Python files to distribute."""
        python_files = []
        
        # Add source directories
        src_dirs = ['src', 'ml_pipeline_framework']
        
        for src_dir in src_dirs:
            if os.path.exists(src_dir):
                # Create zip file of source code
                import zipfile
                zip_name = f"{src_dir}.zip"
                
                with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(src_dir):
                        for file in files:
                            if file.endswith('.py'):
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, '.')
                                zipf.write(file_path, arcname)
                
                python_files.append(zip_name)
        
        return ','.join(python_files) if python_files else ''
    
    def check_job_status(self, app_id: str) -> Dict[str, Any]:
        """
        Check status of submitted job.
        
        Args:
            app_id: Application ID to check
            
        Returns:
            Dictionary with job status information
        """
        logger.info(f"Checking status for application: {app_id}")
        
        try:
            # Get application info from YARN
            result = subprocess.run(['yarn', 'application', '-status', app_id],
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Failed to get application status: {result.stderr}")
                return {'status': 'unknown', 'error': result.stderr}
            
            # Parse output
            status_info = self._parse_yarn_status(result.stdout)
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _parse_yarn_status(self, output: str) -> Dict[str, Any]:
        """Parse YARN application status output."""
        status_info = {}
        
        for line in output.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                status_info[key] = value
        
        return status_info
    
    def kill_job(self, app_id: str) -> bool:
        """
        Kill running job.
        
        Args:
            app_id: Application ID to kill
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Killing application: {app_id}")
        
        try:
            result = subprocess.run(['yarn', 'application', '-kill', app_id],
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Application {app_id} killed successfully")
                return True
            else:
                logger.error(f"Failed to kill application: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error killing job: {e}")
            return False
    
    def list_running_jobs(self) -> List[Dict[str, Any]]:
        """
        List all running ML pipeline jobs.
        
        Returns:
            List of job information dictionaries
        """
        logger.info("Listing running ML pipeline jobs...")
        
        try:
            result = subprocess.run(['yarn', 'application', '-list', '-appStates', 'RUNNING'],
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Failed to list applications: {result.stderr}")
                return []
            
            # Parse output and filter ML pipeline jobs
            jobs = self._parse_yarn_list(result.stdout)
            ml_jobs = [job for job in jobs if 'ml-pipeline' in job.get('name', '').lower()]
            
            return ml_jobs
            
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []
    
    def _parse_yarn_list(self, output: str) -> List[Dict[str, Any]]:
        """Parse YARN application list output."""
        jobs = []
        lines = output.split('\n')
        
        # Find header line
        header_idx = -1
        for i, line in enumerate(lines):
            if 'Application-Id' in line:
                header_idx = i
                break
        
        if header_idx == -1:
            return jobs
        
        # Parse applications
        for line in lines[header_idx + 1:]:
            parts = line.split()
            if len(parts) >= 6:
                job = {
                    'application_id': parts[0],
                    'name': parts[1],
                    'type': parts[2],
                    'user': parts[3],
                    'queue': parts[4],
                    'state': parts[5]
                }
                if len(parts) > 6:
                    job['progress'] = parts[6]
                
                jobs.append(job)
        
        return jobs


def create_job_scripts():
    """Create Hadoop job execution scripts."""
    scripts_dir = Path("deploy/hadoop/scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Training script
    training_script = '''#!/usr/bin/env python3
"""
Hadoop Training Job Script

Executes ML model training on Hadoop cluster using Spark.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline_framework import PipelineOrchestrator
from ml_pipeline_framework.utils import ConfigParser
from ml_pipeline_framework.data_access import HDFSConnector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    try:
        # Load configuration
        config_path = os.environ.get('CONFIG_PATH', 'pipeline_config.yaml')
        config = ConfigParser.from_yaml(config_path)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            experiment_name=os.environ.get('EXPERIMENT_NAME', 'hadoop_training'),
            run_name=os.environ.get('RUN_NAME', f'hadoop_run_{int(time.time())}')
        )
        
        # Setup Spark context for Hadoop
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \\
            .appName("ML-Pipeline-Training") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .getOrCreate()
        
        # Load training data from HDFS
        data_path = os.environ.get('DATA_PATH', '/ml-pipeline/data/training.parquet')
        df = spark.read.parquet(data_path)
        
        # Convert to pandas for ML pipeline (for smaller datasets)
        # For large datasets, implement Spark ML pipeline
        if df.count() < 1000000:  # Threshold for data size
            pandas_df = df.toPandas()
            
            # Separate features and target
            target_column = config.get('target_column', 'target')
            X = pandas_df.drop(columns=[target_column])
            y = pandas_df[target_column]
            
            # Run training
            results = orchestrator.run_training(data=(X, y))
            
            logger.info(f"Training completed: {results}")
            
            # Save model to HDFS
            model_path = f"/ml-pipeline/models/{orchestrator.run_name}"
            # Implementation for saving to HDFS would go here
            
        else:
            logger.info("Large dataset detected, using Spark ML pipeline")
            # Implement Spark ML pipeline for large datasets
            # This would use Spark's MLlib instead of scikit-learn
            
        spark.stop()
        
    except Exception as e:
        logger.error(f"Training job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(scripts_dir / "hadoop_train.py", 'w') as f:
        f.write(training_script)
    
    # Prediction script
    prediction_script = '''#!/usr/bin/env python3
"""
Hadoop Prediction Job Script

Executes batch predictions on Hadoop cluster using Spark.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline_framework import PipelineOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main prediction function."""
    try:
        # Setup Spark context
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \\
            .appName("ML-Pipeline-Prediction") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .getOrCreate()
        
        # Load model
        model_path = os.environ.get('MODEL_PATH', '/ml-pipeline/models/latest')
        orchestrator = PipelineOrchestrator.from_saved_pipeline(model_path)
        
        # Load input data
        input_path = os.environ.get('INPUT_PATH', '/ml-pipeline/data/prediction_input.parquet')
        df = spark.read.parquet(input_path)
        
        # Process in batches for predictions
        batch_size = int(os.environ.get('BATCH_SIZE', 10000))
        output_path = os.environ.get('OUTPUT_PATH', '/ml-pipeline/data/predictions.parquet')
        
        # Convert to pandas for prediction (adjust based on data size)
        if df.count() < 100000:
            pandas_df = df.toPandas()
            predictions = orchestrator.run_prediction(pandas_df)
            
            # Convert back to Spark DataFrame and save
            from pyspark.sql.types import StructType, StructField, StringType, DoubleType
            
            if isinstance(predictions[0], tuple):
                # Handle predictions with probabilities
                pred_data = [(str(p[0]), float(p[1][1]) if len(p[1]) > 1 else float(p[1][0])) 
                            for p in predictions]
                schema = StructType([
                    StructField("prediction", StringType(), True),
                    StructField("probability", DoubleType(), True)
                ])
            else:
                # Handle simple predictions
                pred_data = [(str(p),) for p in predictions]
                schema = StructType([StructField("prediction", StringType(), True)])
            
            pred_df = spark.createDataFrame(pred_data, schema)
            pred_df.write.mode("overwrite").parquet(output_path)
            
        else:
            # For large datasets, process in chunks
            logger.info("Large dataset detected, processing in chunks")
            # Implementation for chunk-based processing
            
        logger.info(f"Predictions saved to {output_path}")
        spark.stop()
        
    except Exception as e:
        logger.error(f"Prediction job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(scripts_dir / "hadoop_predict.py", 'w') as f:
        f.write(prediction_script)
    
    # Data processing script
    data_processing_script = '''#!/usr/bin/env python3
"""
Hadoop Data Processing Job Script

Executes data processing and feature engineering on Hadoop cluster using Spark.
"""

import sys
import os
from pathlib import Path

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline_framework.preprocessing import DataProcessor
from ml_pipeline_framework.utils import ConfigParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main data processing function."""
    try:
        # Setup Spark context
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \\
            .appName("ML-Pipeline-DataProcessing") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \\
            .getOrCreate()
        
        # Load configuration
        config_path = os.environ.get('CONFIG_PATH', 'pipeline_config.yaml')
        config = ConfigParser.from_yaml(config_path)
        
        # Load raw data
        input_path = os.environ.get('INPUT_PATH', '/ml-pipeline/data/raw')
        output_path = os.environ.get('OUTPUT_PATH', '/ml-pipeline/data/processed')
        
        df = spark.read.option("multiline", "true").option("inferSchema", "true").csv(input_path, header=True)
        
        logger.info(f"Loaded {df.count()} rows from {input_path}")
        
        # Data cleaning and preprocessing using Spark SQL
        df.createOrReplaceTempView("raw_data")
        
        # Remove duplicates
        df_clean = spark.sql("""
            SELECT DISTINCT *
            FROM raw_data
            WHERE customer_id IS NOT NULL
        """)
        
        # Feature engineering
        df_features = spark.sql("""
            SELECT *,
                   CASE WHEN age < 25 THEN 'young'
                        WHEN age < 50 THEN 'middle'
                        ELSE 'senior' END as age_group,
                   CASE WHEN credit_score < 600 THEN 'poor'
                        WHEN credit_score < 700 THEN 'fair'
                        WHEN credit_score < 800 THEN 'good'
                        ELSE 'excellent' END as credit_category,
                   months_since_last_transaction / 30.0 as months_since_last_transaction_norm
            FROM ({}) t
        """.format(df_clean.sql_ctx.sql("SELECT * FROM raw_data").queryExecution.toString()))
        
        # Data validation
        logger.info("Running data quality checks...")
        
        # Check for nulls in critical columns
        critical_columns = ['customer_id', 'age', 'credit_score']
        for col in critical_columns:
            null_count = df_features.filter(df_features[col].isNull()).count()
            if null_count > 0:
                logger.warning(f"Found {null_count} null values in {col}")
        
        # Check data ranges
        age_stats = df_features.select("age").describe().collect()
        logger.info(f"Age statistics: {age_stats}")
        
        # Save processed data
        df_features.write.mode("overwrite").parquet(output_path)
        
        logger.info(f"Processed data saved to {output_path}")
        logger.info(f"Final dataset contains {df_features.count()} rows")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"Data processing job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(scripts_dir / "hadoop_data_processing.py", 'w') as f:
        f.write(data_processing_script)
    
    # Make scripts executable
    for script in scripts_dir.glob("*.py"):
        os.chmod(script, 0o755)
    
    logger.info(f"Created Hadoop job scripts in {scripts_dir}")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Submit ML Pipeline jobs to Hadoop cluster')
    parser.add_argument('job_type', nargs='?', choices=['training', 'prediction', 'data_processing', 'evaluation'],
                       help='Type of job to submit')
    parser.add_argument('--config', help='Path to job configuration file')
    parser.add_argument('--app-file', help='Path to application file')
    parser.add_argument('--queue', default='default', help='YARN queue name')
    parser.add_argument('--num-executors', type=int, default=10, help='Number of executors')
    parser.add_argument('--executor-cores', type=int, default=4, help='Cores per executor')
    parser.add_argument('--executor-memory', default='8g', help='Memory per executor')
    parser.add_argument('--driver-memory', default='4g', help='Driver memory')
    parser.add_argument('--check-status', help='Check status of application ID')
    parser.add_argument('--kill-job', help='Kill application with given ID')
    parser.add_argument('--list-jobs', action='store_true', help='List running ML pipeline jobs')
    parser.add_argument('--create-scripts', action='store_true', help='Create Hadoop job scripts')
    
    args = parser.parse_args()
    
    # Create scripts if requested
    if args.create_scripts:
        create_job_scripts()
        return
    
    # Check if required arguments are provided for non-script creation commands
    if not args.job_type and not (args.check_status or args.kill_job or args.list_jobs):
        parser.error("job_type is required unless using --create-scripts, --check-status, --kill-job, or --list-jobs")
    
    if not args.config and not (args.create_scripts or args.check_status or args.kill_job or args.list_jobs):
        parser.error("--config is required unless using --create-scripts, --check-status, --kill-job, or --list-jobs")
    
    # Initialize submitter
    submitter = HadoopJobSubmitter(config_path=args.config) if args.config else HadoopJobSubmitter()
    
    # Handle status check
    if args.check_status:
        status = submitter.check_job_status(args.check_status)
        print(json.dumps(status, indent=2))
        return
    
    # Handle job killing
    if args.kill_job:
        success = submitter.kill_job(args.kill_job)
        print(f"Job kill {'successful' if success else 'failed'}")
        return
    
    # Handle job listing
    if args.list_jobs:
        jobs = submitter.list_running_jobs()
        print(json.dumps(jobs, indent=2))
        return
    
    # Submit job
    kwargs = {
        'app_file': args.app_file,
        'queue': args.queue,
        'num_executors': args.num_executors,
        'executor_cores': args.executor_cores,
        'executor_memory': args.executor_memory,
        'driver_memory': args.driver_memory
    }
    
    try:
        if args.job_type == 'training':
            app_id = submitter.submit_training_job(**kwargs)
        elif args.job_type == 'prediction':
            app_id = submitter.submit_prediction_job(**kwargs)
        elif args.job_type == 'data_processing':
            app_id = submitter.submit_data_processing_job(**kwargs)
        elif args.job_type == 'evaluation':
            app_id = submitter.submit_model_evaluation_job(**kwargs)
        
        print(f"Job submitted successfully with Application ID: {app_id}")
        
        # Monitor job status
        if app_id:
            print("Monitoring job status...")
            for i in range(5):  # Check status 5 times
                time.sleep(30)  # Wait 30 seconds between checks
                status = submitter.check_job_status(app_id)
                print(f"Status check {i+1}: {status.get('state', 'unknown')}")
                
                if status.get('state') in ['FINISHED', 'FAILED', 'KILLED']:
                    break
        
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()