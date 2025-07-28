#!/usr/bin/env python3
"""
Airflow DAG Generator for ML Pipeline Framework

This script generates Airflow DAGs dynamically based on configuration files,
supporting various ML workflow patterns including training, prediction, 
monitoring, and data processing pipelines.
"""

import os
import sys
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from jinja2 import Template, Environment, FileSystemLoader
import argparse


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline DAG generation."""
    name: str
    description: str
    schedule_interval: str
    start_date: str
    catchup: bool = False
    max_active_runs: int = 1
    default_args: Dict[str, Any] = None
    tasks: List[Dict[str, Any]] = None
    dependencies: List[Dict[str, str]] = None
    variables: Dict[str, Any] = None
    connections: List[Dict[str, Any]] = None


class DAGGenerator:
    """Generate Airflow DAGs for ML Pipeline Framework."""
    
    def __init__(self, template_dir: str = None, output_dir: str = None):
        """
        Initialize DAG generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory to save generated DAGs
        """
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "dags"
        
        # Ensure directories exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.jinja_env.filters['to_json'] = json.dumps
        self.jinja_env.filters['to_python_dict'] = self._to_python_dict
    
    def _to_python_dict(self, obj: Any) -> str:
        """Convert object to Python dictionary string representation."""
        return repr(obj)
    
    def load_config(self, config_path: str) -> PipelineConfig:
        """
        Load pipeline configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            PipelineConfig object
        """
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return PipelineConfig(**config_data)
    
    def generate_training_dag(self, config: PipelineConfig) -> str:
        """
        Generate DAG for ML model training pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Generated DAG code as string
        """
        template = self.jinja_env.get_template('training_dag.py.j2')
        
        # Add training-specific context
        context = {
            'config': config,
            'dag_id': f"{config.name}_training",
            'description': f"Training pipeline for {config.description}",
            'schedule_interval': config.schedule_interval,
            'start_date': config.start_date,
            'catchup': config.catchup,
            'max_active_runs': config.max_active_runs,
            'default_args': config.default_args or self._get_default_args(),
            'tasks': config.tasks or self._get_default_training_tasks(),
            'dependencies': config.dependencies or self._get_default_training_dependencies(),
            'variables': config.variables or {},
            'timestamp': datetime.now().isoformat()
        }
        
        return template.render(**context)
    
    def generate_prediction_dag(self, config: PipelineConfig) -> str:
        """
        Generate DAG for ML prediction pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Generated DAG code as string
        """
        template = self.jinja_env.get_template('prediction_dag.py.j2')
        
        context = {
            'config': config,
            'dag_id': f"{config.name}_prediction",
            'description': f"Prediction pipeline for {config.description}",
            'schedule_interval': config.schedule_interval,
            'start_date': config.start_date,
            'catchup': config.catchup,
            'max_active_runs': config.max_active_runs,
            'default_args': config.default_args or self._get_default_args(),
            'tasks': config.tasks or self._get_default_prediction_tasks(),
            'dependencies': config.dependencies or self._get_default_prediction_dependencies(),
            'variables': config.variables or {},
            'timestamp': datetime.now().isoformat()
        }
        
        return template.render(**context)
    
    def generate_monitoring_dag(self, config: PipelineConfig) -> str:
        """
        Generate DAG for ML model monitoring pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Generated DAG code as string
        """
        template = self.jinja_env.get_template('monitoring_dag.py.j2')
        
        context = {
            'config': config,
            'dag_id': f"{config.name}_monitoring",
            'description': f"Monitoring pipeline for {config.description}",
            'schedule_interval': config.schedule_interval,
            'start_date': config.start_date,
            'catchup': config.catchup,
            'max_active_runs': config.max_active_runs,
            'default_args': config.default_args or self._get_default_args(),
            'tasks': config.tasks or self._get_default_monitoring_tasks(),
            'dependencies': config.dependencies or self._get_default_monitoring_dependencies(),
            'variables': config.variables or {},
            'timestamp': datetime.now().isoformat()
        }
        
        return template.render(**context)
    
    def generate_data_processing_dag(self, config: PipelineConfig) -> str:
        """
        Generate DAG for data processing pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Generated DAG code as string
        """
        template = self.jinja_env.get_template('data_processing_dag.py.j2')
        
        context = {
            'config': config,
            'dag_id': f"{config.name}_data_processing",
            'description': f"Data processing pipeline for {config.description}",
            'schedule_interval': config.schedule_interval,
            'start_date': config.start_date,
            'catchup': config.catchup,
            'max_active_runs': config.max_active_runs,
            'default_args': config.default_args or self._get_default_args(),
            'tasks': config.tasks or self._get_default_data_processing_tasks(),
            'dependencies': config.dependencies or self._get_default_data_processing_dependencies(),
            'variables': config.variables or {},
            'timestamp': datetime.now().isoformat()
        }
        
        return template.render(**context)
    
    def generate_complete_pipeline_dag(self, config: PipelineConfig) -> str:
        """
        Generate comprehensive DAG with all pipeline stages.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Generated DAG code as string
        """
        template = self.jinja_env.get_template('complete_pipeline_dag.py.j2')
        
        context = {
            'config': config,
            'dag_id': f"{config.name}_complete_pipeline",
            'description': f"Complete ML pipeline for {config.description}",
            'schedule_interval': config.schedule_interval,
            'start_date': config.start_date,
            'catchup': config.catchup,
            'max_active_runs': config.max_active_runs,
            'default_args': config.default_args or self._get_default_args(),
            'tasks': config.tasks or self._get_default_complete_pipeline_tasks(),
            'dependencies': config.dependencies or self._get_default_complete_pipeline_dependencies(),
            'variables': config.variables or {},
            'timestamp': datetime.now().isoformat()
        }
        
        return template.render(**context)
    
    def save_dag(self, dag_code: str, dag_name: str) -> str:
        """
        Save generated DAG code to file.
        
        Args:
            dag_code: Generated DAG code
            dag_name: Name for the DAG file
            
        Returns:
            Path to saved DAG file
        """
        dag_file = self.output_dir / f"{dag_name}.py"
        
        with open(dag_file, 'w') as f:
            f.write(dag_code)
        
        return str(dag_file)
    
    def generate_all_dags(self, config: PipelineConfig) -> Dict[str, str]:
        """
        Generate all types of DAGs for a pipeline configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Dictionary mapping DAG type to file path
        """
        generated_dags = {}
        
        # Generate training DAG
        training_dag = self.generate_training_dag(config)
        training_file = self.save_dag(training_dag, f"{config.name}_training")
        generated_dags['training'] = training_file
        
        # Generate prediction DAG
        prediction_dag = self.generate_prediction_dag(config)
        prediction_file = self.save_dag(prediction_dag, f"{config.name}_prediction")
        generated_dags['prediction'] = prediction_file
        
        # Generate monitoring DAG
        monitoring_dag = self.generate_monitoring_dag(config)
        monitoring_file = self.save_dag(monitoring_dag, f"{config.name}_monitoring")
        generated_dags['monitoring'] = monitoring_file
        
        # Generate data processing DAG
        data_processing_dag = self.generate_data_processing_dag(config)
        data_processing_file = self.save_dag(data_processing_dag, f"{config.name}_data_processing")
        generated_dags['data_processing'] = data_processing_file
        
        # Generate complete pipeline DAG
        complete_dag = self.generate_complete_pipeline_dag(config)
        complete_file = self.save_dag(complete_dag, f"{config.name}_complete_pipeline")
        generated_dags['complete_pipeline'] = complete_file
        
        return generated_dags
    
    def _get_default_args(self) -> Dict[str, Any]:
        """Get default arguments for DAGs."""
        return {
            'owner': 'ml-pipeline-framework',
            'depends_on_past': False,
            'start_date': '{{ ds }}',
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': 2,
            'retry_delay': timedelta(minutes=5),
            'execution_timeout': timedelta(hours=2)
        }
    
    def _get_default_training_tasks(self) -> List[Dict[str, Any]]:
        """Get default tasks for training pipeline."""
        return [
            {
                'task_id': 'validate_config',
                'task_type': 'bash',
                'command': 'python /opt/ml-pipeline/run_pipeline.py validate --config {{ var.value.config_path }}',
                'description': 'Validate pipeline configuration'
            },
            {
                'task_id': 'extract_data',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.extract_data',
                'description': 'Extract training data from source'
            },
            {
                'task_id': 'validate_data',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.validate_data',
                'description': 'Validate data quality'
            },
            {
                'task_id': 'preprocess_data',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.preprocess_data',
                'description': 'Preprocess and engineer features'
            },
            {
                'task_id': 'train_model',
                'task_type': 'kubernetes',
                'image': 'ml-pipeline-framework:latest',
                'command': ['python', '/app/scripts/train.py'],
                'description': 'Train ML model'
            },
            {
                'task_id': 'evaluate_model',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.evaluate_model',
                'description': 'Evaluate model performance'
            },
            {
                'task_id': 'register_model',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.register_model',
                'description': 'Register model in MLflow'
            },
            {
                'task_id': 'notify_completion',
                'task_type': 'email',
                'to': ['ml-team@company.com'],
                'subject': 'Training Pipeline Completed - {{ dag.dag_id }}',
                'description': 'Send completion notification'
            }
        ]
    
    def _get_default_training_dependencies(self) -> List[Dict[str, str]]:
        """Get default dependencies for training pipeline."""
        return [
            {'upstream': 'validate_config', 'downstream': 'extract_data'},
            {'upstream': 'extract_data', 'downstream': 'validate_data'},
            {'upstream': 'validate_data', 'downstream': 'preprocess_data'},
            {'upstream': 'preprocess_data', 'downstream': 'train_model'},
            {'upstream': 'train_model', 'downstream': 'evaluate_model'},
            {'upstream': 'evaluate_model', 'downstream': 'register_model'},
            {'upstream': 'register_model', 'downstream': 'notify_completion'}
        ]
    
    def _get_default_prediction_tasks(self) -> List[Dict[str, Any]]:
        """Get default tasks for prediction pipeline."""
        return [
            {
                'task_id': 'check_model_availability',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.check_model_availability',
                'description': 'Check if model is available for prediction'
            },
            {
                'task_id': 'extract_prediction_data',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.extract_prediction_data',
                'description': 'Extract data for prediction'
            },
            {
                'task_id': 'run_predictions',
                'task_type': 'kubernetes',
                'image': 'ml-pipeline-framework:latest',
                'command': ['python', '/app/scripts/predict.py'],
                'description': 'Generate predictions'
            },
            {
                'task_id': 'store_predictions',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.store_predictions',
                'description': 'Store predictions in database'
            },
            {
                'task_id': 'generate_prediction_report',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.generate_prediction_report',
                'description': 'Generate prediction summary report'
            }
        ]
    
    def _get_default_prediction_dependencies(self) -> List[Dict[str, str]]:
        """Get default dependencies for prediction pipeline."""
        return [
            {'upstream': 'check_model_availability', 'downstream': 'extract_prediction_data'},
            {'upstream': 'extract_prediction_data', 'downstream': 'run_predictions'},
            {'upstream': 'run_predictions', 'downstream': 'store_predictions'},
            {'upstream': 'store_predictions', 'downstream': 'generate_prediction_report'}
        ]
    
    def _get_default_monitoring_tasks(self) -> List[Dict[str, Any]]:
        """Get default tasks for monitoring pipeline."""
        return [
            {
                'task_id': 'collect_model_metrics',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.collect_model_metrics',
                'description': 'Collect model performance metrics'
            },
            {
                'task_id': 'detect_data_drift',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.detect_data_drift',
                'description': 'Detect data drift'
            },
            {
                'task_id': 'analyze_model_performance',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.analyze_model_performance',
                'description': 'Analyze model performance degradation'
            },
            {
                'task_id': 'generate_monitoring_report',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.generate_monitoring_report',
                'description': 'Generate monitoring report'
            },
            {
                'task_id': 'alert_on_issues',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.alert_on_issues',
                'description': 'Send alerts if issues detected'
            }
        ]
    
    def _get_default_monitoring_dependencies(self) -> List[Dict[str, str]]:
        """Get default dependencies for monitoring pipeline."""
        return [
            {'upstream': 'collect_model_metrics', 'downstream': 'analyze_model_performance'},
            {'upstream': 'detect_data_drift', 'downstream': 'analyze_model_performance'},
            {'upstream': 'analyze_model_performance', 'downstream': 'generate_monitoring_report'},
            {'upstream': 'generate_monitoring_report', 'downstream': 'alert_on_issues'}
        ]
    
    def _get_default_data_processing_tasks(self) -> List[Dict[str, Any]]:
        """Get default tasks for data processing pipeline."""
        return [
            {
                'task_id': 'extract_raw_data',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.extract_raw_data',
                'description': 'Extract raw data from sources'
            },
            {
                'task_id': 'clean_data',
                'task_type': 'spark',
                'application_file': '/opt/ml-pipeline/spark_jobs/data_cleaning.py',
                'description': 'Clean and prepare data'
            },
            {
                'task_id': 'feature_engineering',
                'task_type': 'spark',
                'application_file': '/opt/ml-pipeline/spark_jobs/feature_engineering.py',
                'description': 'Engineer features'
            },
            {
                'task_id': 'data_quality_checks',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.run_data_quality_checks',
                'description': 'Run data quality checks'
            },
            {
                'task_id': 'store_processed_data',
                'task_type': 'python',
                'python_callable': 'ml_pipeline_framework.airflow.operators.store_processed_data',
                'description': 'Store processed data'
            }
        ]
    
    def _get_default_data_processing_dependencies(self) -> List[Dict[str, str]]:
        """Get default dependencies for data processing pipeline."""
        return [
            {'upstream': 'extract_raw_data', 'downstream': 'clean_data'},
            {'upstream': 'clean_data', 'downstream': 'feature_engineering'},
            {'upstream': 'feature_engineering', 'downstream': 'data_quality_checks'},
            {'upstream': 'data_quality_checks', 'downstream': 'store_processed_data'}
        ]
    
    def _get_default_complete_pipeline_tasks(self) -> List[Dict[str, Any]]:
        """Get default tasks for complete pipeline."""
        return (
            self._get_default_data_processing_tasks() +
            self._get_default_training_tasks() +
            [
                {
                    'task_id': 'deploy_model',
                    'task_type': 'kubernetes',
                    'image': 'ml-pipeline-framework:latest',
                    'command': ['python', '/app/scripts/deploy_model.py'],
                    'description': 'Deploy trained model'
                },
                {
                    'task_id': 'run_acceptance_tests',
                    'task_type': 'python',
                    'python_callable': 'ml_pipeline_framework.airflow.operators.run_acceptance_tests',
                    'description': 'Run model acceptance tests'
                }
            ]
        )
    
    def _get_default_complete_pipeline_dependencies(self) -> List[Dict[str, str]]:
        """Get default dependencies for complete pipeline."""
        return (
            self._get_default_data_processing_dependencies() +
            [
                {'upstream': 'store_processed_data', 'downstream': 'validate_config'},
            ] +
            self._get_default_training_dependencies() +
            [
                {'upstream': 'register_model', 'downstream': 'deploy_model'},
                {'upstream': 'deploy_model', 'downstream': 'run_acceptance_tests'},
                {'upstream': 'run_acceptance_tests', 'downstream': 'notify_completion'}
            ]
        )
    
    def create_templates(self):
        """Create default Jinja2 templates for DAG generation."""
        templates = {
            'training_dag.py.j2': self._get_training_dag_template(),
            'prediction_dag.py.j2': self._get_prediction_dag_template(),
            'monitoring_dag.py.j2': self._get_monitoring_dag_template(),
            'data_processing_dag.py.j2': self._get_data_processing_dag_template(),
            'complete_pipeline_dag.py.j2': self._get_complete_pipeline_dag_template(),
            'operators.py.j2': self._get_operators_template()
        }
        
        for template_name, template_content in templates.items():
            template_file = self.template_dir / template_name
            with open(template_file, 'w') as f:
                f.write(template_content)
        
        print(f"Created templates in {self.template_dir}")
    
    def _get_training_dag_template(self) -> str:
        """Get training DAG template."""
        return '''"""
{{ description }}

Generated on: {{ timestamp }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from ml_pipeline_framework.airflow import operators

# Default arguments
default_args = {{ default_args | to_python_dict }}

# DAG definition
dag = DAG(
    '{{ dag_id }}',
    default_args=default_args,
    description='{{ description }}',
    schedule_interval='{{ schedule_interval }}',
    start_date=datetime.strptime('{{ start_date }}', '%Y-%m-%d'),
    catchup={{ catchup }},
    max_active_runs={{ max_active_runs }},
    tags=['ml-pipeline', 'training', '{{ config.name }}']
)

# Tasks
{% for task in tasks %}
{% if task.task_type == 'bash' %}
{{ task.task_id }} = BashOperator(
    task_id='{{ task.task_id }}',
    bash_command='{{ task.command }}',
    dag=dag
)
{% elif task.task_type == 'python' %}
{{ task.task_id }} = PythonOperator(
    task_id='{{ task.task_id }}',
    python_callable={{ task.python_callable }},
    dag=dag
)
{% elif task.task_type == 'kubernetes' %}
{{ task.task_id }} = KubernetesPodOperator(
    task_id='{{ task.task_id }}',
    name='{{ task.task_id }}-pod',
    namespace='ml-pipeline',
    image='{{ task.image }}',
    cmds={{ task.command | to_python_dict }},
    dag=dag
)
{% elif task.task_type == 'email' %}
{{ task.task_id }} = EmailOperator(
    task_id='{{ task.task_id }}',
    to={{ task.to | to_python_dict }},
    subject='{{ task.subject }}',
    html_content='Training pipeline completed successfully.',
    dag=dag
)
{% endif %}

{% endfor %}

# Dependencies
{% for dep in dependencies %}
{{ dep.upstream }} >> {{ dep.downstream }}
{% endfor %}
'''
    
    def _get_prediction_dag_template(self) -> str:
        """Get prediction DAG template."""
        return '''"""
{{ description }}

Generated on: {{ timestamp }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from ml_pipeline_framework.airflow import operators

# Default arguments
default_args = {{ default_args | to_python_dict }}

# DAG definition
dag = DAG(
    '{{ dag_id }}',
    default_args=default_args,
    description='{{ description }}',
    schedule_interval='{{ schedule_interval }}',
    start_date=datetime.strptime('{{ start_date }}', '%Y-%m-%d'),
    catchup={{ catchup }},
    max_active_runs={{ max_active_runs }},
    tags=['ml-pipeline', 'prediction', '{{ config.name }}']
)

# Tasks
{% for task in tasks %}
{% if task.task_type == 'python' %}
{{ task.task_id }} = PythonOperator(
    task_id='{{ task.task_id }}',
    python_callable={{ task.python_callable }},
    dag=dag
)
{% elif task.task_type == 'kubernetes' %}
{{ task.task_id }} = KubernetesPodOperator(
    task_id='{{ task.task_id }}',
    name='{{ task.task_id }}-pod',
    namespace='ml-pipeline',
    image='{{ task.image }}',
    cmds={{ task.command | to_python_dict }},
    dag=dag
)
{% endif %}

{% endfor %}

# Dependencies
{% for dep in dependencies %}
{{ dep.upstream }} >> {{ dep.downstream }}
{% endfor %}
'''
    
    def _get_monitoring_dag_template(self) -> str:
        """Get monitoring DAG template."""
        return '''"""
{{ description }}

Generated on: {{ timestamp }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from ml_pipeline_framework.airflow import operators

# Default arguments
default_args = {{ default_args | to_python_dict }}

# DAG definition
dag = DAG(
    '{{ dag_id }}',
    default_args=default_args,
    description='{{ description }}',
    schedule_interval='{{ schedule_interval }}',
    start_date=datetime.strptime('{{ start_date }}', '%Y-%m-%d'),
    catchup={{ catchup }},
    max_active_runs={{ max_active_runs }},
    tags=['ml-pipeline', 'monitoring', '{{ config.name }}']
)

# Tasks
{% for task in tasks %}
{{ task.task_id }} = PythonOperator(
    task_id='{{ task.task_id }}',
    python_callable={{ task.python_callable }},
    dag=dag
)

{% endfor %}

# Dependencies
{% for dep in dependencies %}
{{ dep.upstream }} >> {{ dep.downstream }}
{% endfor %}
'''
    
    def _get_data_processing_dag_template(self) -> str:
        """Get data processing DAG template."""
        return '''"""
{{ description }}

Generated on: {{ timestamp }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from ml_pipeline_framework.airflow import operators

# Default arguments
default_args = {{ default_args | to_python_dict }}

# DAG definition
dag = DAG(
    '{{ dag_id }}',
    default_args=default_args,
    description='{{ description }}',
    schedule_interval='{{ schedule_interval }}',
    start_date=datetime.strptime('{{ start_date }}', '%Y-%m-%d'),
    catchup={{ catchup }},
    max_active_runs={{ max_active_runs }},
    tags=['ml-pipeline', 'data-processing', '{{ config.name }}']
)

# Tasks
{% for task in tasks %}
{% if task.task_type == 'python' %}
{{ task.task_id }} = PythonOperator(
    task_id='{{ task.task_id }}',
    python_callable={{ task.python_callable }},
    dag=dag
)
{% elif task.task_type == 'spark' %}
{{ task.task_id }} = SparkSubmitOperator(
    task_id='{{ task.task_id }}',
    application='{{ task.application_file }}',
    conn_id='spark_default',
    dag=dag
)
{% endif %}

{% endfor %}

# Dependencies
{% for dep in dependencies %}
{{ dep.upstream }} >> {{ dep.downstream }}
{% endfor %}
'''
    
    def _get_complete_pipeline_dag_template(self) -> str:
        """Get complete pipeline DAG template."""
        return '''"""
{{ description }}

Generated on: {{ timestamp }}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.email import EmailOperator
from ml_pipeline_framework.airflow import operators

# Default arguments
default_args = {{ default_args | to_python_dict }}

# DAG definition
dag = DAG(
    '{{ dag_id }}',
    default_args=default_args,
    description='{{ description }}',
    schedule_interval='{{ schedule_interval }}',
    start_date=datetime.strptime('{{ start_date }}', '%Y-%m-%d'),
    catchup={{ catchup }},
    max_active_runs={{ max_active_runs }},
    tags=['ml-pipeline', 'complete-pipeline', '{{ config.name }}']
)

# Tasks
{% for task in tasks %}
{% if task.task_type == 'bash' %}
{{ task.task_id }} = BashOperator(
    task_id='{{ task.task_id }}',
    bash_command='{{ task.command }}',
    dag=dag
)
{% elif task.task_type == 'python' %}
{{ task.task_id }} = PythonOperator(
    task_id='{{ task.task_id }}',
    python_callable={{ task.python_callable }},
    dag=dag
)
{% elif task.task_type == 'kubernetes' %}
{{ task.task_id }} = KubernetesPodOperator(
    task_id='{{ task.task_id }}',
    name='{{ task.task_id }}-pod',
    namespace='ml-pipeline',
    image='{{ task.image }}',
    cmds={{ task.command | to_python_dict }},
    dag=dag
)
{% elif task.task_type == 'spark' %}
{{ task.task_id }} = SparkSubmitOperator(
    task_id='{{ task.task_id }}',
    application='{{ task.application_file }}',
    conn_id='spark_default',
    dag=dag
)
{% elif task.task_type == 'email' %}
{{ task.task_id }} = EmailOperator(
    task_id='{{ task.task_id }}',
    to={{ task.to | to_python_dict }},
    subject='{{ task.subject }}',
    html_content='Complete pipeline executed successfully.',
    dag=dag
)
{% endif %}

{% endfor %}

# Dependencies
{% for dep in dependencies %}
{{ dep.upstream }} >> {{ dep.downstream }}
{% endfor %}
'''
    
    def _get_operators_template(self) -> str:
        """Get custom operators template."""
        return '''"""
Custom Airflow operators for ML Pipeline Framework

Generated on: {{ timestamp }}
"""

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from ml_pipeline_framework import PipelineOrchestrator
from ml_pipeline_framework.utils import ConfigParser
import logging

logger = logging.getLogger(__name__)

def extract_data(**context):
    """Extract data for ML pipeline."""
    logger.info("Extracting data...")
    # Implementation here
    return "data_extracted"

def validate_data(**context):
    """Validate data quality."""
    logger.info("Validating data...")
    # Implementation here
    return "data_validated"

def preprocess_data(**context):
    """Preprocess data for ML pipeline."""
    logger.info("Preprocessing data...")
    # Implementation here
    return "data_preprocessed"

def evaluate_model(**context):
    """Evaluate trained model."""
    logger.info("Evaluating model...")
    # Implementation here
    return "model_evaluated"

def register_model(**context):
    """Register model in MLflow."""
    logger.info("Registering model...")
    # Implementation here
    return "model_registered"

# Additional operator functions...
'''


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Generate Airflow DAGs for ML Pipeline Framework')
    parser.add_argument('config_file', help='Path to pipeline configuration file')
    parser.add_argument('--output-dir', default='dags', help='Output directory for generated DAGs')
    parser.add_argument('--template-dir', default='templates', help='Directory containing Jinja2 templates')
    parser.add_argument('--dag-type', choices=['training', 'prediction', 'monitoring', 'data_processing', 'complete', 'all'],
                       default='all', help='Type of DAG to generate')
    parser.add_argument('--create-templates', action='store_true', help='Create default templates')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DAGGenerator(template_dir=args.template_dir, output_dir=args.output_dir)
    
    # Create templates if requested
    if args.create_templates:
        generator.create_templates()
        print(f"Templates created in {args.template_dir}")
        return
    
    # Load configuration
    config = generator.load_config(args.config_file)
    
    # Generate DAGs
    if args.dag_type == 'all':
        generated_dags = generator.generate_all_dags(config)
        print("Generated DAGs:")
        for dag_type, file_path in generated_dags.items():
            print(f"  {dag_type}: {file_path}")
    else:
        if args.dag_type == 'training':
            dag_code = generator.generate_training_dag(config)
        elif args.dag_type == 'prediction':
            dag_code = generator.generate_prediction_dag(config)
        elif args.dag_type == 'monitoring':
            dag_code = generator.generate_monitoring_dag(config)
        elif args.dag_type == 'data_processing':
            dag_code = generator.generate_data_processing_dag(config)
        elif args.dag_type == 'complete':
            dag_code = generator.generate_complete_pipeline_dag(config)
        
        file_path = generator.save_dag(dag_code, f"{config.name}_{args.dag_type}")
        print(f"Generated {args.dag_type} DAG: {file_path}")


if __name__ == "__main__":
    main()