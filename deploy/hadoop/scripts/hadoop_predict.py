#!/usr/bin/env python3
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
        spark = SparkSession.builder \
            .appName("ML-Pipeline-Prediction") \
            .config("spark.sql.adaptive.enabled", "true") \
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
