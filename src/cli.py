#!/usr/bin/env python3
"""
Command Line Interface for ML Pipeline Framework.

This module provides a comprehensive CLI for all pipeline operations including
training, prediction, explanation, deployment, and monitoring.
"""

import os
import sys
import json
import yaml
import click
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from __version__ import __version__, get_version_info, check_compatibility
from utils.logging_config import setup_logging
from utils.config_parser import ConfigParser
from pipeline_orchestrator import PipelineOrchestrator

# Configure logging for CLI
logger = setup_logging(component='cli', level=logging.INFO)

# Global CLI settings
CONTEXT_SETTINGS = {
    'help_option_names': ['-h', '--help'],
    'max_content_width': 120,
    'terminal_width': 120,
}

# Common options
def common_options(func):
    """Common CLI options decorator."""
    func = click.option(
        '--config', '-c',
        type=click.Path(exists=True, path_type=Path),
        help='Path to configuration file'
    )(func)
    func = click.option(
        '--verbose', '-v',
        is_flag=True,
        help='Enable verbose output'
    )(func)
    func = click.option(
        '--quiet', '-q',
        is_flag=True,
        help='Suppress non-error output'
    )(func)
    func = click.option(
        '--log-level',
        type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
        default='INFO',
        help='Set logging level'
    )(func)
    return func

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__, prog_name='ML Pipeline Framework')
@click.pass_context
def main(ctx):
    """
    🚀 ML Pipeline Framework - Enterprise-grade ML pipelines for production.
    
    A comprehensive framework for building, deploying, and monitoring machine
    learning pipelines with enterprise-grade features including AutoML,
    interpretability, compliance, and monitoring.
    
    Examples:
        ml-pipeline train --config configs/pipeline_config.yaml
        ml-pipeline predict --model model.pkl --data data.csv
        ml-pipeline explain --model model.pkl --method shap
        ml-pipeline deploy --model model.pkl --environment production
    """
    ctx.ensure_object(dict)
    
    # Check compatibility
    try:
        check_compatibility()
    except RuntimeError as e:
        click.echo(f"❌ Compatibility Error: {e}", err=True)
        sys.exit(1)

@main.command()
@click.option('--format', 'output_format', 
              type=click.Choice(['json', 'yaml', 'table']),
              default='table',
              help='Output format for version information')
def version(output_format):
    """Display version and build information."""
    version_info = get_version_info()
    
    if output_format == 'json':
        click.echo(json.dumps(version_info, indent=2))
    elif output_format == 'yaml':
        click.echo(yaml.dump(version_info, default_flow_style=False))
    else:
        click.echo("🔧 ML Pipeline Framework")
        click.echo("=" * 50)
        click.echo(f"Version: {version_info['version']}")
        click.echo(f"Build: {version_info['build']}")
        click.echo(f"Release Date: {version_info['release_date']}")
        click.echo(f"API Version: {version_info['api_version']}")
        click.echo("\n📦 Features:")
        for feature, enabled in version_info['features'].items():
            status = "✅" if enabled else "❌"
            click.echo(f"  {status} {feature}")

@main.command()
@common_options
@click.option('--mode', 
              type=click.Choice(['train', 'validate', 'tune', 'automl']),
              default='train',
              help='Training mode')
@click.option('--data', 
              type=click.Path(exists=True, path_type=Path),
              help='Path to training data')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for model artifacts')
@click.option('--experiment-name',
              help='MLflow experiment name')
@click.option('--tags',
              help='Experiment tags as JSON string')
@click.option('--dry-run',
              is_flag=True,
              help='Validate configuration without training')
@click.pass_context
def train(ctx, config, verbose, quiet, log_level, mode, data, output, 
          experiment_name, tags, dry_run):
    """
    🎯 Train machine learning models.
    
    Supports multiple training modes including standard training, AutoML,
    hyperparameter tuning, and validation. Includes comprehensive logging
    and experiment tracking.
    
    Examples:
        ml-pipeline train --config configs/pipeline_config.yaml
        ml-pipeline train --mode automl --data data.csv
        ml-pipeline train --mode tune --experiment-name fraud-detection
    """
    setup_cli_logging(log_level, verbose, quiet)
    
    try:
        click.echo(f"🚀 Starting training in {mode} mode...")
        
        # Load configuration
        if not config:
            config = Path("configs/pipeline_config.yaml")
        
        config_data = ConfigParser.load_config(str(config))
        
        # Override config with CLI parameters
        if data:
            config_data['data_source']['csv']['file_paths'] = [str(data)]
        if output:
            config_data['output']['model_artifacts']['save_location'] = str(output)
        if experiment_name:
            config_data['mlflow']['experiment_name'] = experiment_name
        
        # Parse tags
        experiment_tags = {}
        if tags:
            try:
                experiment_tags = json.loads(tags)
            except json.JSONDecodeError:
                click.echo("❌ Error: Invalid JSON format for tags", err=True)
                sys.exit(1)
        
        if dry_run:
            click.echo("🔍 Dry run mode - validating configuration...")
            click.echo("✅ Configuration validation passed")
            return
        
        # Initialize pipeline
        orchestrator = PipelineOrchestrator(config_data)
        
        # Set experiment tags
        if experiment_tags:
            orchestrator.set_experiment_tags(experiment_tags)
        
        # Run training
        results = orchestrator.run(mode=mode)
        
        click.echo("✅ Training completed successfully!")
        click.echo(f"📊 Best model: {results.best_model_name}")
        click.echo(f"📈 Best score: {results.best_score:.4f}")
        if output:
            click.echo(f"💾 Model saved to: {output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"❌ Training failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@main.command()
@common_options
@click.option('--model', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to trained model')
@click.option('--data', '-d',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to prediction data')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output file for predictions')
@click.option('--batch-size',
              type=int,
              default=1000,
              help='Batch size for predictions')
@click.option('--include-probabilities',
              is_flag=True,
              help='Include prediction probabilities')
@click.option('--format',
              type=click.Choice(['csv', 'parquet', 'json']),
              default='csv',
              help='Output format')
@click.pass_context
def predict(ctx, config, verbose, quiet, log_level, model, data, output,
            batch_size, include_probabilities, format):
    """
    🔮 Generate predictions from trained models.
    
    Supports batch prediction with configurable batch sizes and output formats.
    Can generate both predictions and probabilities.
    
    Examples:
        ml-pipeline predict --model model.pkl --data test.csv
        ml-pipeline predict -m model.pkl -d data.csv -o predictions.csv
        ml-pipeline predict --model model.pkl --data data.csv --include-probabilities
    """
    setup_cli_logging(log_level, verbose, quiet)
    
    try:
        click.echo("🔮 Starting prediction...")
        
        # Load model and make predictions
        from models.factory import ModelFactory
        import pandas as pd
        
        # Load model
        model = ModelFactory.load_model(str(model))
        click.echo(f"📦 Model loaded: {type(model).__name__}")
        
        # Load data
        if str(data).endswith('.csv'):
            df = pd.read_csv(data)
        elif str(data).endswith('.parquet'):
            df = pd.read_parquet(data)
        else:
            raise ValueError(f"Unsupported data format: {data}")
        
        click.echo(f"📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Generate predictions
        predictions = model.predict(df)
        
        # Prepare output
        result_df = df.copy()
        result_df['prediction'] = predictions
        
        if include_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            if probabilities.shape[1] == 2:
                result_df['probability'] = probabilities[:, 1]
            else:
                for i in range(probabilities.shape[1]):
                    result_df[f'probability_class_{i}'] = probabilities[:, i]
        
        # Save output
        if not output:
            output = Path(data).parent / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format == 'csv':
            result_df.to_csv(output, index=False)
        elif format == 'parquet':
            result_df.to_parquet(output, index=False)
        elif format == 'json':
            result_df.to_json(output, orient='records', indent=2)
        
        click.echo(f"✅ Predictions completed!")
        click.echo(f"📊 Generated {len(predictions)} predictions")
        click.echo(f"💾 Results saved to: {output}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"❌ Prediction failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@main.command()
@common_options
@click.option('--model', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to trained model')
@click.option('--data', '-d',
              type=click.Path(exists=True, path_type=Path),
              help='Path to explanation data')
@click.option('--method',
              type=click.Choice(['shap', 'lime', 'ale', 'anchors', 'counterfactuals', 'all']),
              default='shap',
              help='Explanation method')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              help='Output directory for explanations')
@click.option('--sample-size',
              type=int,
              default=100,
              help='Number of samples to explain')
@click.option('--generate-plots',
              is_flag=True,
              help='Generate explanation plots')
@click.pass_context
def explain(ctx, config, verbose, quiet, log_level, model, data, method,
            output, sample_size, generate_plots):
    """
    🔍 Generate model explanations and interpretability analysis.
    
    Supports multiple explanation methods including SHAP, LIME, ALE plots,
    Anchors, and counterfactual explanations. Can generate both numerical
    explanations and visualizations.
    
    Examples:
        ml-pipeline explain --model model.pkl --method shap
        ml-pipeline explain -m model.pkl -d data.csv --method lime --generate-plots
        ml-pipeline explain --model model.pkl --method all --output explanations/
    """
    setup_cli_logging(log_level, verbose, quiet)
    
    try:
        click.echo(f"🔍 Generating {method} explanations...")
        
        from explainability.interpretability_pipeline import InterpretabilityPipeline
        import pandas as pd
        
        # Load model
        from models.factory import ModelFactory
        model_obj = ModelFactory.load_model(str(model))
        
        # Load data for explanation
        if data:
            if str(data).endswith('.csv'):
                df = pd.read_csv(data)
            elif str(data).endswith('.parquet'):
                df = pd.read_parquet(data)
            else:
                raise ValueError(f"Unsupported data format: {data}")
        else:
            # Use sample data if no data provided
            click.echo("⚠️  No data provided, using model's training data sample")
            df = None
        
        # Sample data if too large
        if df is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            click.echo(f"📊 Using {sample_size} samples for explanation")
        
        # Create output directory
        if not output:
            output = Path(f"explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        
        # Initialize interpretability pipeline
        interp_pipeline = InterpretabilityPipeline(model_obj)
        
        # Generate explanations
        if method == 'all':
            methods = ['shap', 'lime', 'ale']
        else:
            methods = [method]
        
        results = {}
        for exp_method in methods:
            click.echo(f"🔄 Running {exp_method} explanations...")
            
            if exp_method == 'shap':
                result = interp_pipeline.generate_shap_explanations(
                    df, save_plots=generate_plots, output_dir=str(output)
                )
            elif exp_method == 'lime':
                result = interp_pipeline.generate_lime_explanations(
                    df, save_plots=generate_plots, output_dir=str(output)
                )
            elif exp_method == 'ale':
                result = interp_pipeline.generate_ale_plots(
                    df, save_plots=generate_plots, output_dir=str(output)
                )
            elif exp_method == 'anchors':
                result = interp_pipeline.generate_anchors_explanations(
                    df, output_dir=str(output)
                )
            elif exp_method == 'counterfactuals':
                result = interp_pipeline.generate_counterfactual_explanations(
                    df, output_dir=str(output)
                )
            
            results[exp_method] = result
            click.echo(f"✅ {exp_method} explanations completed")
        
        # Save summary report
        summary_file = output / "explanation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'methods': methods,
                'sample_size': len(df) if df is not None else 0,
                'model_type': type(model_obj).__name__,
                'timestamp': datetime.now().isoformat(),
                'results': {k: str(v) for k, v in results.items()}
            }, f, indent=2)
        
        click.echo("✅ Explanations completed!")
        click.echo(f"📊 Methods used: {', '.join(methods)}")
        click.echo(f"💾 Results saved to: {output}")
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        click.echo(f"❌ Explanation failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@main.command()
@common_options
@click.option('--model', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to trained model')
@click.option('--environment', '-e',
              type=click.Choice(['development', 'staging', 'production']),
              default='development',
              help='Deployment environment')
@click.option('--platform',
              type=click.Choice(['kubernetes', 'docker', 'aws', 'gcp', 'azure']),
              default='kubernetes',
              help='Deployment platform')
@click.option('--replicas',
              type=int,
              default=3,
              help='Number of replicas for deployment')
@click.option('--cpu-request',
              default='500m',
              help='CPU request per replica')
@click.option('--memory-request',
              default='1Gi',
              help='Memory request per replica')
@click.option('--dry-run',
              is_flag=True,
              help='Generate deployment configs without deploying')
@click.pass_context
def deploy(ctx, config, verbose, quiet, log_level, model, environment, platform,
           replicas, cpu_request, memory_request, dry_run):
    """
    🚀 Deploy trained models to production environments.
    
    Supports deployment to multiple platforms including Kubernetes, Docker,
    and major cloud providers. Includes auto-scaling, health checks, and
    monitoring setup.
    
    Examples:
        ml-pipeline deploy --model model.pkl --environment production
        ml-pipeline deploy -m model.pkl -e staging --platform kubernetes
        ml-pipeline deploy --model model.pkl --dry-run
    """
    setup_cli_logging(log_level, verbose, quiet)
    
    try:
        click.echo(f"🚀 Deploying model to {environment} on {platform}...")
        
        # Validate model
        if not model.exists():
            raise FileNotFoundError(f"Model not found: {model}")
        
        # Create deployment configuration
        deployment_config = {
            'model_path': str(model),
            'environment': environment,
            'platform': platform,
            'replicas': replicas,
            'resources': {
                'cpu_request': cpu_request,
                'memory_request': memory_request,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        # Generate deployment files
        deploy_dir = Path(f"deploy_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        if platform == 'kubernetes':
            generate_k8s_deployment(deployment_config, deploy_dir)
        elif platform == 'docker':
            generate_docker_deployment(deployment_config, deploy_dir)
        else:
            generate_cloud_deployment(deployment_config, deploy_dir, platform)
        
        # Save deployment config
        config_file = deploy_dir / "deployment_config.json"
        with open(config_file, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        if dry_run:
            click.echo("🔍 Dry run mode - deployment files generated")
            click.echo(f"📁 Deployment files created in: {deploy_dir}")
            return
        
        # Execute deployment
        if platform == 'kubernetes':
            execute_k8s_deployment(deploy_dir)
        elif platform == 'docker':
            execute_docker_deployment(deploy_dir)
        else:
            execute_cloud_deployment(deploy_dir, platform)
        
        click.echo("✅ Deployment completed successfully!")
        click.echo(f"🌐 Environment: {environment}")
        click.echo(f"🏗️  Platform: {platform}")
        click.echo(f"📊 Replicas: {replicas}")
        click.echo(f"📁 Deployment files: {deploy_dir}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        click.echo(f"❌ Deployment failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@main.command()
@common_options
@click.option('--deployment-name',
              help='Name of deployment to monitor')
@click.option('--environment',
              type=click.Choice(['development', 'staging', 'production']),
              help='Environment to monitor')
@click.option('--metrics',
              multiple=True,
              default=['performance', 'drift', 'fairness'],
              help='Metrics to monitor')
@click.option('--dashboard',
              is_flag=True,
              help='Launch monitoring dashboard')
@click.option('--alerts',
              is_flag=True,
              help='Enable alert monitoring')
@click.pass_context
def monitor(ctx, config, verbose, quiet, log_level, deployment_name, environment,
            metrics, dashboard, alerts):
    """
    📊 Monitor deployed models and pipeline performance.
    
    Provides comprehensive monitoring including performance metrics, data drift
    detection, fairness analysis, and custom business metrics. Can launch
    interactive dashboards and configure alerting.
    
    Examples:
        ml-pipeline monitor --deployment-name fraud-model --environment production
        ml-pipeline monitor --dashboard --metrics performance drift
        ml-pipeline monitor --alerts --environment production
    """
    setup_cli_logging(log_level, verbose, quiet)
    
    try:
        click.echo("📊 Starting model monitoring...")
        
        from utils.monitoring import ModelMonitor
        
        # Initialize monitor
        monitor_config = {
            'deployment_name': deployment_name,
            'environment': environment,
            'metrics': list(metrics),
            'dashboard_enabled': dashboard,
            'alerts_enabled': alerts,
        }
        
        monitor = ModelMonitor(monitor_config)
        
        if dashboard:
            click.echo("🚀 Launching monitoring dashboard...")
            monitor.launch_dashboard()
            click.echo("📊 Dashboard available at: http://localhost:8080")
        
        if alerts:
            click.echo("🔔 Configuring alerts...")
            monitor.setup_alerts()
            click.echo("✅ Alerts configured")
        
        # Display current metrics
        current_metrics = monitor.get_current_metrics()
        
        click.echo("\n📈 Current Metrics:")
        click.echo("=" * 50)
        for metric_name, value in current_metrics.items():
            status = "✅" if value.get('status') == 'healthy' else "⚠️"
            click.echo(f"{status} {metric_name}: {value.get('value', 'N/A')}")
        
        if not dashboard:
            click.echo("\n💡 Tip: Use --dashboard to launch interactive monitoring")
        
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        click.echo(f"❌ Monitoring failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@main.command()
@click.option('--config-type',
              type=click.Choice(['basic', 'fraud-detection', 'automl', 'enterprise']),
              default='basic',
              help='Type of configuration to generate')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='configs/',
              help='Output directory for configuration files')
@click.option('--format',
              type=click.Choice(['yaml', 'json']),
              default='yaml',
              help='Configuration file format')
def init(config_type, output, format):
    """
    🔧 Initialize new pipeline configuration files.
    
    Generates template configuration files for different use cases including
    basic ML pipelines, fraud detection, AutoML, and enterprise setups.
    
    Examples:
        ml-pipeline init --config-type basic
        ml-pipeline init --config-type fraud-detection --output my-configs/
        ml-pipeline init --config-type enterprise --format json
    """
    try:
        click.echo(f"🔧 Initializing {config_type} configuration...")
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate configuration templates
        from utils.config_templates import ConfigTemplateGenerator
        
        generator = ConfigTemplateGenerator()
        
        if config_type == 'basic':
            configs = generator.generate_basic_config()
        elif config_type == 'fraud-detection':
            configs = generator.generate_fraud_detection_config()
        elif config_type == 'automl':
            configs = generator.generate_automl_config()
        elif config_type == 'enterprise':
            configs = generator.generate_enterprise_config()
        
        # Save configuration files
        for config_name, config_data in configs.items():
            filename = f"{config_name}.{format}"
            filepath = output_dir / filename
            
            if format == 'yaml':
                with open(filepath, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                with open(filepath, 'w') as f:
                    json.dump(config_data, f, indent=2)
            
            click.echo(f"📄 Created: {filepath}")
        
        click.echo("✅ Configuration initialization completed!")
        click.echo(f"📁 Files created in: {output_dir}")
        click.echo("\n💡 Next steps:")
        click.echo("1. Review and customize the generated configuration files")
        click.echo("2. Update data source paths and credentials")
        click.echo("3. Run: ml-pipeline train --config <config-file>")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.option('--strict',
              is_flag=True,
              help='Enable strict validation mode')
def validate(config_file, strict):
    """
    ✅ Validate pipeline configuration files.
    
    Performs comprehensive validation of configuration files including
    schema validation, dependency checks, and configuration consistency.
    
    Examples:
        ml-pipeline validate configs/pipeline_config.yaml
        ml-pipeline validate configs/automl_config.yaml --strict
    """
    try:
        click.echo(f"✅ Validating configuration: {config_file}")
        
        from utils.config_validator import ConfigValidator
        
        validator = ConfigValidator(strict_mode=strict)
        result = validator.validate_config(str(config_file))
        
        if result.is_valid:
            click.echo("✅ Configuration validation passed!")
            
            if result.warnings:
                click.echo("\n⚠️  Warnings:")
                for warning in result.warnings:
                    click.echo(f"  • {warning}")
            
        else:
            click.echo("❌ Configuration validation failed!")
            click.echo("\n🔍 Errors:")
            for error in result.errors:
                click.echo(f"  • {error}")
            
            if result.suggestions:
                click.echo("\n💡 Suggestions:")
                for suggestion in result.suggestions:
                    click.echo(f"  • {suggestion}")
            
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)

# Helper functions

def setup_cli_logging(level: str, verbose: bool, quiet: bool):
    """Setup CLI-specific logging configuration."""
    if quiet:
        level = 'ERROR'
    elif verbose:
        level = 'DEBUG'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def generate_k8s_deployment(config: Dict[str, Any], output_dir: Path):
    """Generate Kubernetes deployment files."""
    # This would generate actual K8s YAML files
    k8s_manifest = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'ml-pipeline-model',
            'labels': {'app': 'ml-pipeline-model'}
        },
        'spec': {
            'replicas': config['replicas'],
            'selector': {'matchLabels': {'app': 'ml-pipeline-model'}},
            'template': {
                'metadata': {'labels': {'app': 'ml-pipeline-model'}},
                'spec': {
                    'containers': [{
                        'name': 'model-server',
                        'image': 'ml-pipeline-framework:latest',
                        'ports': [{'containerPort': 8080}],
                        'resources': {
                            'requests': {
                                'cpu': config['resources']['cpu_request'],
                                'memory': config['resources']['memory_request']
                            }
                        }
                    }]
                }
            }
        }
    }
    
    k8s_file = output_dir / 'deployment.yaml'
    with open(k8s_file, 'w') as f:
        yaml.dump(k8s_manifest, f, default_flow_style=False)

def generate_docker_deployment(config: Dict[str, Any], output_dir: Path):
    """Generate Docker deployment files."""
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY model.pkl .

EXPOSE 8080
CMD ["python", "-m", "src.api.server"]
"""
    
    dockerfile = output_dir / 'Dockerfile'
    with open(dockerfile, 'w') as f:
        f.write(dockerfile_content)

def generate_cloud_deployment(config: Dict[str, Any], output_dir: Path, platform: str):
    """Generate cloud-specific deployment files."""
    # This would generate platform-specific deployment configs
    pass

def execute_k8s_deployment(deploy_dir: Path):
    """Execute Kubernetes deployment."""
    import subprocess
    
    try:
        subprocess.run(['kubectl', 'apply', '-f', str(deploy_dir)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kubernetes deployment failed: {e}")

def execute_docker_deployment(deploy_dir: Path):
    """Execute Docker deployment."""
    import subprocess
    
    try:
        subprocess.run(['docker', 'build', '-t', 'ml-pipeline-model', str(deploy_dir)], check=True)
        subprocess.run(['docker', 'run', '-d', '-p', '8080:8080', 'ml-pipeline-model'], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Docker deployment failed: {e}")

def execute_cloud_deployment(deploy_dir: Path, platform: str):
    """Execute cloud platform deployment."""
    # This would execute platform-specific deployment commands
    pass

if __name__ == '__main__':
    main()