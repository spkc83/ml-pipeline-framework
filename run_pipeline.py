#!/usr/bin/env python3
"""
ML Pipeline Framework CLI
Main entry point for running ML pipelines with configurable components.
"""

import click
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline_orchestrator import PipelineOrchestrator
from src.utils.config_parser import ConfigParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

# Framework version
__version__ = "1.0.0"

# Valid choices for CLI arguments
VALID_VERSIONS = ['python', 'pyspark']
VALID_FRAMEWORKS = ['sklearn', 'h2o', 'statsmodels', 'sparkml', 'xgboost', 'lightgbm', 'catboost']
VALID_MODES = ['train', 'predict', 'evaluate', 'experiment', 'compare']


@click.group()
@click.version_option(version=__version__, prog_name="ML Pipeline Framework")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    ML Pipeline Framework - Production-ready ML pipeline orchestration.
    
    This framework provides end-to-end ML pipeline capabilities including:
    - Data ingestion from multiple sources
    - Preprocessing and feature engineering
    - Model training across multiple frameworks
    - Model evaluation and comparison
    - Deployment and monitoring
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)
    
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj['verbose'] = True
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
        ctx.obj['quiet'] = True
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    ctx.obj['start_time'] = datetime.now()


@cli.command()
@click.option('--version', type=click.Choice(VALID_VERSIONS), default='python',
              help='Execution environment (python for edge node, pyspark for cluster)')
@click.option('--framework', type=click.Choice(VALID_FRAMEWORKS), default='sklearn',
              help='ML framework to use for model training')
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--mode', type=click.Choice(VALID_MODES), default='train',
              help='Pipeline execution mode')
@click.option('--output-dir', '-o', type=click.Path(), default='./output',
              help='Output directory for artifacts and results')
@click.option('--experiment-name', type=str, 
              help='MLflow experiment name (overrides config)')
@click.option('--run-name', type=str,
              help='MLflow run name (auto-generated if not provided)')
@click.option('--dry-run', is_flag=True,
              help='Validate configuration and show execution plan without running')
@click.option('--force', is_flag=True,
              help='Force execution even if validation warnings exist')
@click.option('--resume-from', type=str,
              help='Resume pipeline from a specific stage')
@click.option('--parallel-jobs', '-j', type=int, default=1,
              help='Number of parallel jobs for model training')
@click.option('--enable-mlflow', is_flag=True, default=True,
              help='Enable MLflow tracking')
@click.option('--enable-artifacts', is_flag=True, default=True,
              help='Enable artifact management')
@click.pass_context
def run(ctx, version, framework, config, mode, output_dir, experiment_name, 
        run_name, dry_run, force, resume_from, parallel_jobs, enable_mlflow, 
        enable_artifacts):
    """
    Run the ML pipeline with specified configuration.
    
    Examples:
    
    \b
    # Train a scikit-learn model on edge node
    python run_pipeline.py run --config configs/pipeline_config.yaml --mode train
    
    \b
    # Train a Spark ML model on cluster
    python run_pipeline.py run --version pyspark --framework sparkml --config configs/spark_config.yaml
    
    \b
    # Run model evaluation
    python run_pipeline.py run --mode evaluate --config configs/eval_config.yaml
    
    \b
    # Compare multiple models
    python run_pipeline.py run --mode compare --config configs/compare_config.yaml
    """
    try:
        logger.info(f"Starting ML Pipeline Framework v{__version__}")
        logger.info(f"Mode: {mode}, Framework: {framework}, Version: {version}")
        
        # Load and validate configuration
        config_parser = ConfigParser()
        pipeline_config = config_parser.load_config(config)
        
        logger.info(f"Loaded configuration from: {config}")
        
        # Override config with CLI arguments
        cli_overrides = {
            'execution': {
                'version': version,
                'framework': framework,
                'mode': mode,
                'output_dir': output_dir,
                'parallel_jobs': parallel_jobs,
                'enable_mlflow': enable_mlflow,
                'enable_artifacts': enable_artifacts
            }
        }
        
        if experiment_name:
            cli_overrides['mlflow'] = cli_overrides.get('mlflow', {})
            cli_overrides['mlflow']['experiment_name'] = experiment_name
        
        if run_name:
            cli_overrides['mlflow'] = cli_overrides.get('mlflow', {})
            cli_overrides['mlflow']['run_name'] = run_name
        
        # Merge CLI overrides with config
        pipeline_config = config_parser.merge_configs(pipeline_config, cli_overrides)
        
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator(
            config=pipeline_config,
            output_dir=output_dir,
            verbose=ctx.obj.get('verbose', False)
        )
        
        # Validate configuration
        validation_result = orchestrator.validate_config()
        
        if not validation_result['valid']:
            logger.error("Configuration validation failed:")
            for error in validation_result['errors']:
                logger.error(f"  - {error}")
            
            if not force:
                raise click.ClickException("Configuration validation failed. Use --force to override.")
            else:
                logger.warning("Proceeding with invalid configuration due to --force flag")
        
        if validation_result['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation_result['warnings']:
                logger.warning(f"  - {warning}")
        
        # Show execution plan for dry run
        if dry_run:
            execution_plan = orchestrator.get_execution_plan()
            
            click.echo("\n" + "="*60)
            click.echo("EXECUTION PLAN (DRY RUN)")
            click.echo("="*60)
            
            click.echo(f"\nPipeline Configuration:")
            click.echo(f"  Mode: {mode}")
            click.echo(f"  Framework: {framework}")
            click.echo(f"  Execution Version: {version}")
            click.echo(f"  Output Directory: {output_dir}")
            
            click.echo(f"\nExecution Steps:")
            for i, step in enumerate(execution_plan['steps'], 1):
                click.echo(f"  {i}. {step['name']}")
                click.echo(f"     Description: {step['description']}")
                click.echo(f"     Estimated Duration: {step['estimated_duration']}")
                if step.get('dependencies'):
                    click.echo(f"     Dependencies: {', '.join(step['dependencies'])}")
                click.echo()
            
            click.echo(f"Total Estimated Duration: {execution_plan['total_estimated_duration']}")
            click.echo(f"Resource Requirements: {execution_plan['resource_requirements']}")
            
            click.echo("\n" + "="*60)
            click.echo("DRY RUN COMPLETE - No actual execution performed")
            click.echo("="*60)
            
            return
        
        # Execute pipeline
        logger.info("Starting pipeline execution...")
        
        if resume_from:
            logger.info(f"Resuming pipeline from stage: {resume_from}")
            result = orchestrator.resume_pipeline(resume_from)
        else:
            result = orchestrator.run_pipeline(mode)
        
        # Report results
        execution_time = datetime.now() - ctx.obj['start_time']
        
        if result['success']:
            click.echo(f"\n{'='*60}")
            click.echo("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            click.echo(f"{'='*60}")
            click.echo(f"Execution Time: {execution_time}")
            click.echo(f"Mode: {mode}")
            click.echo(f"Framework: {framework}")
            
            if 'model_metrics' in result:
                click.echo(f"\nModel Performance:")
                for metric, value in result['model_metrics'].items():
                    click.echo(f"  {metric}: {value}")
            
            if 'artifacts' in result:
                click.echo(f"\nGenerated Artifacts:")
                for artifact_type, paths in result['artifacts'].items():
                    click.echo(f"  {artifact_type}: {len(paths)} files")
            
            if 'mlflow_run_id' in result:
                click.echo(f"\nMLflow Run ID: {result['mlflow_run_id']}")
            
            click.echo(f"\nOutput Directory: {output_dir}")
            
        else:
            click.echo(f"\n{'='*60}")
            click.echo("PIPELINE EXECUTION FAILED")
            click.echo(f"{'='*60}")
            click.echo(f"Error: {result.get('error', 'Unknown error')}")
            click.echo(f"Execution Time: {execution_time}")
            
            if 'partial_results' in result:
                click.echo(f"\nPartial Results Available:")
                for key, value in result['partial_results'].items():
                    click.echo(f"  {key}: {value}")
            
            sys.exit(1)
            
    except Exception as config_error:
        if "config" in str(config_error).lower():
            logger.error(f"Configuration error: {config_error}")
            raise click.ClickException(f"Configuration error: {config_error}")
        else:
            logger.error(f"Pipeline error: {config_error}")
            raise click.ClickException(f"Pipeline error: {config_error}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        raise click.ClickException(f"Unexpected error: {e}")


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--output-format', type=click.Choice(['json', 'yaml', 'table']), default='table',
              help='Output format for validation results')
@click.pass_context
def validate(ctx, config, output_format):
    """
    Validate pipeline configuration without executing.
    
    Examples:
    
    \b
    # Validate configuration file
    python run_pipeline.py validate --config configs/pipeline_config.yaml
    
    \b
    # Get validation results in JSON format
    python run_pipeline.py validate --config configs/pipeline_config.yaml --output-format json
    """
    try:
        logger.info("Validating pipeline configuration...")
        
        # Load configuration
        config_parser = ConfigParser()
        pipeline_config = config_parser.load_config(config)
        
        # Initialize orchestrator for validation
        orchestrator = PipelineOrchestrator(
            config=pipeline_config,
            output_dir="./temp",
            verbose=ctx.obj.get('verbose', False)
        )
        
        # Validate configuration
        validation_result = orchestrator.validate_config()
        
        # Output results in requested format
        if output_format == 'json':
            import json
            click.echo(json.dumps(validation_result, indent=2))
        
        elif output_format == 'yaml':
            import yaml
            click.echo(yaml.dump(validation_result, default_flow_style=False))
        
        else:  # table format
            click.echo(f"\n{'='*60}")
            click.echo("CONFIGURATION VALIDATION RESULTS")
            click.echo(f"{'='*60}")
            
            if validation_result['valid']:
                click.echo("✅ Configuration is VALID")
            else:
                click.echo("❌ Configuration is INVALID")
            
            if validation_result['errors']:
                click.echo(f"\nErrors ({len(validation_result['errors'])}):")
                for i, error in enumerate(validation_result['errors'], 1):
                    click.echo(f"  {i}. {error}")
            
            if validation_result['warnings']:
                click.echo(f"\nWarnings ({len(validation_result['warnings'])}):")
                for i, warning in enumerate(validation_result['warnings'], 1):
                    click.echo(f"  {i}. {warning}")
            
            if validation_result.get('recommendations'):
                click.echo(f"\nRecommendations:")
                for i, rec in enumerate(validation_result['recommendations'], 1):
                    click.echo(f"  {i}. {rec}")
        
        # Exit with error code if validation failed
        if not validation_result['valid']:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        raise click.ClickException(f"Validation error: {e}")


@cli.command()
@click.option('--experiment-name', type=str, help='MLflow experiment name to query')
@click.option('--metric', type=str, help='Metric to optimize for best run')
@click.option('--order', type=click.Choice(['asc', 'desc']), default='desc',
              help='Sort order for metric')
@click.option('--limit', type=int, default=10, help='Maximum number of runs to show')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table',
              help='Output format')
@click.pass_context
def experiments(ctx, experiment_name, metric, order, limit, output_format):
    """
    Query and manage MLflow experiments.
    
    Examples:
    
    \b
    # List recent runs
    python run_pipeline.py experiments --experiment-name my-experiment
    
    \b
    # Find best model by AUC
    python run_pipeline.py experiments --experiment-name my-experiment --metric roc_auc
    """
    try:
        from src.utils.mlflow_tracker import MLflowTracker
        
        # Initialize MLflow tracker
        tracker = MLflowTracker(experiment_name=experiment_name)
        
        if metric:
            # Get best run for metric
            best_run = tracker.get_best_run(metric, mode='max' if order == 'desc' else 'min')
            
            if output_format == 'json':
                import json
                click.echo(json.dumps(best_run, indent=2, default=str))
            else:
                if best_run:
                    click.echo(f"\nBest Run by {metric}:")
                    click.echo(f"Run ID: {best_run['run_id']}")
                    click.echo(f"Score: {best_run['metrics'].get(metric, 'N/A')}")
                    click.echo(f"Status: {best_run['status']}")
                    click.echo(f"Start Time: {best_run['start_time']}")
                else:
                    click.echo("No runs found")
        else:
            # List recent runs
            runs_df = tracker.search_runs(max_results=limit)
            
            if output_format == 'json':
                click.echo(runs_df.to_json(orient='records', indent=2))
            else:
                if not runs_df.empty:
                    # Display key columns
                    display_cols = ['run_id', 'status', 'start_time']
                    
                    # Add metric columns if they exist
                    metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
                    display_cols.extend(metric_cols[:5])  # Show first 5 metrics
                    
                    available_cols = [col for col in display_cols if col in runs_df.columns]
                    
                    click.echo(f"\nRecent Runs (showing {len(runs_df)} runs):")
                    click.echo(runs_df[available_cols].to_string(index=False))
                else:
                    click.echo("No runs found")
                    
    except Exception as e:
        logger.error(f"Experiment query error: {e}")
        if ctx.obj.get('verbose'):
            logger.error(traceback.format_exc())
        raise click.ClickException(f"Experiment query error: {e}")


@cli.command()
@click.option('--template', type=click.Choice(['basic', 'advanced', 'spark', 'comparison']),
              default='basic', help='Configuration template to generate')
@click.option('--output', '-o', type=click.Path(), default='pipeline_config.yaml',
              help='Output file path')
@click.option('--overwrite', is_flag=True, help='Overwrite existing file')
@click.pass_context
def init(ctx, template, output, overwrite):
    """
    Generate configuration template files.
    
    Examples:
    
    \b
    # Generate basic configuration
    python run_pipeline.py init --template basic
    
    \b
    # Generate Spark configuration
    python run_pipeline.py init --template spark --output spark_config.yaml
    """
    try:
        output_path = Path(output)
        
        # Check if file exists
        if output_path.exists() and not overwrite:
            raise click.ClickException(f"File {output} already exists. Use --overwrite to replace.")
        
        # Generate template content
        if template == 'basic':
            template_content = _generate_basic_template()
        elif template == 'advanced':
            template_content = _generate_advanced_template()
        elif template == 'spark':
            template_content = _generate_spark_template()
        elif template == 'comparison':
            template_content = _generate_comparison_template()
        else:
            raise click.ClickException(f"Unknown template: {template}")
        
        # Write template to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(template_content)
        
        click.echo(f"✅ Generated {template} configuration template: {output}")
        click.echo(f"\nNext steps:")
        click.echo(f"1. Edit {output} to match your requirements")
        click.echo(f"2. Validate: python run_pipeline.py validate --config {output}")
        click.echo(f"3. Run: python run_pipeline.py run --config {output}")
        
    except Exception as e:
        logger.error(f"Template generation error: {e}")
        raise click.ClickException(f"Template generation error: {e}")


def _generate_basic_template() -> str:
    """Generate basic configuration template."""
    return '''# ML Pipeline Framework - Basic Configuration Template

# Data source configuration
data_source:
  type: "postgresql"  # postgresql, mysql, hive, snowflake, csv, parquet
  connection:
    host: "${DB_HOST:localhost}"
    port: ${DB_PORT:5432}
    database: "${DB_NAME:mydb}"
    username: "${DB_USER:user}"
    password: "${DB_PASSWORD:password}"
  
  # Query or file path
  query: "SELECT * FROM training_data WHERE created_date >= '2023-01-01'"
  # file_path: "data/training_data.csv"

# Preprocessing configuration
preprocessing:
  missing_values:
    strategy: "mean"  # mean, median, mode, drop, fill
    fill_value: null
  
  scaling:
    method: "standard"  # standard, minmax, robust, none
  
  encoding:
    categorical_strategy: "onehot"  # onehot, target, ordinal
    handle_unknown: "ignore"
  
  feature_selection:
    enabled: true
    method: "mutual_info"  # mutual_info, chi2, f_score, rfe
    k_features: 20

# Model training configuration
model_training:
  algorithm: "RandomForestClassifier"
  framework: "sklearn"
  
  parameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  cross_validation:
    enabled: true
    folds: 5
    scoring: "roc_auc"

# Model evaluation configuration
evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "roc_auc"
  
  test_size: 0.2
  
  threshold_optimization:
    enabled: true
    metric: "f1_score"

# Output configuration
output:
  model_path: "models/"
  reports_path: "reports/"
  plots_path: "plots/"
  
  formats:
    model: ["pickle", "joblib"]
    reports: ["html", "json"]
    plots: ["png", "pdf"]

# MLflow configuration
mlflow:
  tracking_uri: null  # null for local, or "http://mlflow-server:5000"
  experiment_name: "ml-pipeline-experiment"
  auto_log: true

# Execution configuration
execution:
  parallel_jobs: 1
  random_state: 42
  verbose: true
'''

def _generate_advanced_template() -> str:
    """Generate advanced configuration template."""
    return '''# ML Pipeline Framework - Advanced Configuration Template

# Data source with multiple connections
data_source:
  primary:
    type: "postgresql"
    connection:
      host: "${DB_HOST:localhost}"
      port: ${DB_PORT:5432}
      database: "${DB_NAME:mydb}"
      username: "${DB_USER:user}"
      password: "${DB_PASSWORD:password}"
    query: "SELECT * FROM training_data"
  
  features:
    type: "hive"
    connection:
      host: "${HIVE_HOST:hive-server}"
      port: 10000
    query: "SELECT * FROM feature_store.customer_features"

# Advanced preprocessing
preprocessing:
  data_validation:
    enabled: true
    schema_file: "schemas/data_schema.json"
    
  missing_values:
    strategy: "advanced"
    numerical_strategy: "knn"
    categorical_strategy: "mode"
    
  scaling:
    method: "robust"
    feature_range: [0, 1]
    
  feature_engineering:
    polynomial_features:
      enabled: true
      degree: 2
      interaction_only: false
    
    binning:
      enabled: true
      features: ["age", "income"]
      n_bins: 5
      strategy: "quantile"
  
  imbalance_handling:
    enabled: true
    method: "smote"
    sampling_strategy: "auto"

# Model training with hyperparameter tuning
model_training:
  framework: "sklearn"
  
  models:
    - algorithm: "RandomForestClassifier"
      parameters:
        n_estimators: [100, 200, 300]
        max_depth: [5, 10, 15, null]
        min_samples_split: [2, 5, 10]
        random_state: 42
    
    - algorithm: "XGBClassifier"
      parameters:
        n_estimators: [100, 200]
        max_depth: [3, 6, 9]
        learning_rate: [0.01, 0.1, 0.2]
        random_state: 42
  
  hyperparameter_tuning:
    method: "bayesian"  # grid, random, bayesian
    n_trials: 50
    scoring: "roc_auc"
    cv_folds: 5
  
  ensemble:
    enabled: true
    method: "voting"  # voting, stacking, blending
    weights: null

# Comprehensive evaluation
evaluation:
  metrics:
    classification:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1_score"
      - "roc_auc"
      - "pr_auc"
      - "brier_score"
      - "ks_statistic"
  
  business_metrics:
    enabled: true
    cost_matrix:
      tp: 100  # True positive value
      tn: 1    # True negative value
      fp: -10  # False positive cost
      fn: -50  # False negative cost
  
  model_explainability:
    enabled: true
    methods: ["shap", "permutation", "partial_dependence"]
    
  model_comparison:
    enabled: true
    baseline_model: "LogisticRegression"

# Artifact management
artifacts:
  storage:
    type: "s3"  # local, s3, hdfs, azure
    bucket: "ml-artifacts"
    prefix: "experiments/"
  
  versioning:
    enabled: true
    strategy: "semantic"
  
  compression:
    enabled: true
    method: "gzip"

# Advanced MLflow configuration
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:http://mlflow:5000}"
  experiment_name: "advanced-ml-pipeline"
  run_name: "${RUN_NAME:auto}"
  
  model_registry:
    enabled: true
    model_name: "production-model"
    stage: "Staging"
  
  auto_log: true
  log_artifacts: true

# Monitoring and alerting
monitoring:
  enabled: true
  
  data_drift:
    enabled: true
    reference_dataset: "reference_data.parquet"
    threshold: 0.05
  
  model_performance:
    enabled: true
    thresholds:
      accuracy: 0.85
      precision: 0.80
      recall: 0.75
  
  alerts:
    email:
      enabled: false
      recipients: ["admin@company.com"]
    
    slack:
      enabled: false
      webhook_url: "${SLACK_WEBHOOK_URL}"

# Execution configuration
execution:
  environment: "production"
  parallel_jobs: -1  # Use all available cores
  memory_limit: "8GB"
  timeout: 3600  # 1 hour
  
  retry:
    enabled: true
    max_attempts: 3
    backoff_factor: 2
'''

def _generate_spark_template() -> str:
    """Generate Spark configuration template."""
    return '''# ML Pipeline Framework - Spark Configuration Template

# Spark configuration
spark:
  app_name: "ML Pipeline Framework"
  master: "yarn"  # local, yarn, spark://host:port
  
  config:
    spark.executor.memory: "4g"
    spark.executor.cores: 4
    spark.executor.instances: 10
    spark.driver.memory: "2g"
    spark.sql.adaptive.enabled: true
    spark.sql.adaptive.coalescePartitions.enabled: true

# Data source for Spark
data_source:
  type: "hive"
  connection:
    host: "${HIVE_HOST:hive-metastore}"
    port: 9083
  
  # Spark SQL query
  query: |
    SELECT *
    FROM training_database.customer_data
    WHERE partition_date >= '2023-01-01'
      AND target IS NOT NULL

# Spark-optimized preprocessing
preprocessing:
  # Use Spark DataFrame operations
  spark_optimized: true
  
  partitioning:
    enabled: true
    columns: ["partition_date"]
    num_partitions: 100
  
  caching:
    enabled: true
    storage_level: "MEMORY_AND_DISK_SER"
  
  missing_values:
    strategy: "mean"
    broadcast_small_tables: true
  
  feature_engineering:
    vectorization: "spark_ml"  # Use Spark ML VectorAssembler
    
    transformations:
      - type: "StringIndexer"
        input_cols: ["category", "region"]
      
      - type: "OneHotEncoder"
        input_cols: ["category_indexed", "region_indexed"]
      
      - type: "VectorAssembler"
        input_cols: ["feature1", "feature2", "category_encoded"]
        output_col: "features"
      
      - type: "StandardScaler"
        input_col: "features"
        output_col: "scaled_features"

# Spark ML model training
model_training:
  framework: "sparkml"
  
  models:
    - algorithm: "RandomForestClassifier"
      parameters:
        numTrees: 100
        maxDepth: 10
        seed: 42
        featuresCol: "scaled_features"
        labelCol: "target"
    
    - algorithm: "GBTClassifier"
      parameters:
        maxIter: 100
        maxDepth: 6
        seed: 42
        featuresCol: "scaled_features"
        labelCol: "target"
  
  hyperparameter_tuning:
    method: "cross_validation"
    param_grid:
      numTrees: [50, 100, 200]
      maxDepth: [5, 10, 15]
    
    evaluator: "BinaryClassificationEvaluator"
    metric: "areaUnderROC"
    num_folds: 3

# Distributed evaluation
evaluation:
  spark_distributed: true
  
  metrics:
    - "areaUnderROC"
    - "areaUnderPR"
    - "accuracy"
    - "f1"
  
  # Custom Spark SQL metrics
  custom_metrics:
    - name: "lift_top_decile"
      sql: |
        SELECT 
          (COUNT(*) FILTER (WHERE target = 1 AND decile = 1)) / 
          (COUNT(*) FILTER (WHERE target = 1)) * 10 as lift
        FROM (
          SELECT *, NTILE(10) OVER (ORDER BY prediction DESC) as decile
          FROM predictions
        )

# Output configuration for Spark
output:
  format: "parquet"  # parquet, delta, hive
  
  model_path: "hdfs://namenode:9000/models/"
  predictions_path: "hdfs://namenode:9000/predictions/"
  
  partitioning:
    enabled: true
    columns: ["model_version", "prediction_date"]
  
  # Delta Lake configuration
  delta:
    enabled: false
    optimize: true
    z_order: ["model_version"]

# Distributed MLflow with Spark
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:http://mlflow:5000}"
  experiment_name: "spark-ml-pipeline"
  
  # Log Spark ML models
  spark_ml_logging: true
  
  # Distributed logging
  log_model_signature: true
  log_input_example: true

# Resource management
resources:
  dynamic_allocation:
    enabled: true
    min_executors: 2
    max_executors: 20
    initial_executors: 5
  
  memory_management:
    fraction: 0.8
    storage_fraction: 0.5
  
  checkpointing:
    enabled: true
    directory: "hdfs://namenode:9000/checkpoints/"
    interval: 10

# Monitoring for Spark jobs
monitoring:
  spark_ui:
    enabled: true
    port: 4040
  
  metrics:
    enabled: true
    sink: "prometheus"  # console, csv, jmx, prometheus
    
  history_server:
    enabled: true
    log_directory: "hdfs://namenode:9000/spark-logs/"

# Execution environment
execution:
  submit_args:
    driver-memory: "2g"
    executor-memory: "4g"
    executor-cores: 4
    num-executors: 10
    
  dependencies:
    jars:
      - "hdfs://namenode:9000/jars/mysql-connector.jar"
      - "hdfs://namenode:9000/jars/hadoop-aws.jar"
    
    py_files:
      - "src/utils/spark_utils.py"
      - "src/preprocessing/spark_transformers.py"
'''

def _generate_comparison_template() -> str:
    """Generate model comparison configuration template."""
    return '''# ML Pipeline Framework - Model Comparison Configuration

# Data configuration
data_source:
  type: "csv"
  file_path: "data/comparison_dataset.csv"
  
  train_test_split:
    test_size: 0.2
    stratify: true
    random_state: 42

# Models to compare
models_to_compare:
  - name: "logistic_regression"
    framework: "sklearn"
    algorithm: "LogisticRegression"
    parameters:
      C: [0.1, 1.0, 10.0]
      solver: ["liblinear", "lbfgs"]
      random_state: 42
  
  - name: "random_forest"
    framework: "sklearn"
    algorithm: "RandomForestClassifier"
    parameters:
      n_estimators: [100, 200]
      max_depth: [5, 10, null]
      random_state: 42
  
  - name: "xgboost"
    framework: "xgboost"
    algorithm: "XGBClassifier"
    parameters:
      n_estimators: [100, 200]
      max_depth: [3, 6]
      learning_rate: [0.01, 0.1]
      random_state: 42
  
  - name: "lightgbm"
    framework: "lightgbm"
    algorithm: "LGBMClassifier"
    parameters:
      n_estimators: [100, 200]
      max_depth: [3, 6]
      learning_rate: [0.01, 0.1]
      random_state: 42

# Hyperparameter optimization
hyperparameter_tuning:
  method: "random_search"  # grid_search, random_search, bayesian
  n_iter: 20
  cv_folds: 5
  scoring: "roc_auc"
  n_jobs: -1

# Comparison metrics
comparison_metrics:
  primary_metric: "roc_auc"
  
  metrics_to_evaluate:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "roc_auc"
    - "pr_auc"
    - "ks_statistic"
    - "brier_score"
  
  business_metrics:
    enabled: true
    cost_matrix:
      tp: 100
      tn: 1
      fp: -10
      fn: -50

# Statistical testing
statistical_testing:
  enabled: true
  significance_level: 0.05
  
  tests:
    - "mcnemar"        # For classification accuracy
    - "paired_ttest"   # For continuous metrics
    - "wilcoxon"       # Non-parametric alternative

# Model comparison criteria
comparison_criteria:
  performance_threshold: 0.02  # Minimum improvement required
  
  guardrails:
    max_auc_degradation: 0.01
    max_ks_degradation: 0.05
  
  selection_strategy: "multi_objective"  # single_metric, multi_objective, pareto

# Cross-validation strategy
cross_validation:
  strategy: "stratified_kfold"
  n_splits: 5
  shuffle: true
  random_state: 42
  
  # Time series specific
  time_series:
    enabled: false
    time_column: "date"
    n_splits: 5

# Ensemble methods
ensemble:
  enabled: true
  
  methods:
    - type: "voting"
      voting: "soft"
      weights: null
    
    - type: "stacking"
      meta_learner: "LogisticRegression"
      cv_folds: 3

# Model explainability comparison
explainability:
  enabled: true
  
  methods:
    - "shap"
    - "permutation_importance"
    - "partial_dependence"
  
  feature_importance_comparison: true
  
  consistency_metrics:
    - "rank_correlation"
    - "feature_stability"

# Reporting configuration
reporting:
  generate_comparison_report: true
  
  report_sections:
    - "executive_summary"
    - "performance_comparison"
    - "statistical_tests"
    - "feature_importance"
    - "business_impact"
    - "recommendations"
  
  output_formats:
    - "html"
    - "pdf"
    - "json"
  
  visualizations:
    - "performance_radar"
    - "metric_comparison_bars"
    - "roc_curves"
    - "precision_recall_curves"
    - "feature_importance_comparison"

# MLflow experiment tracking
mlflow:
  tracking_uri: null
  experiment_name: "model-comparison-experiment"
  
  # Parent run for comparison
  run_name: "model_comparison_${TIMESTAMP}"
  
  # Log each model as child run
  nested_runs: true
  
  comparison_artifacts:
    - "comparison_report.html"
    - "performance_metrics.json"
    - "statistical_tests.json"

# Execution configuration
execution:
  parallel_model_training: true
  max_parallel_jobs: 4
  
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
  
  resource_limits:
    memory_per_model: "2GB"
    time_limit_per_model: 1800  # 30 minutes

# Output organization
output:
  base_directory: "./model_comparison_results"
  
  structure:
    models: "models/"
    reports: "reports/"
    plots: "plots/"
    data: "processed_data/"
    comparisons: "comparisons/"
  
  artifact_naming:
    include_timestamp: true
    include_metrics: true
    format: "{model_name}_{timestamp}_{primary_metric:.3f}"
'''


if __name__ == '__main__':
    cli()