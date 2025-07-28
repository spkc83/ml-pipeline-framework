#!/usr/bin/env python3
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
        spark = SparkSession.builder \
            .appName("ML-Pipeline-Training") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
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
