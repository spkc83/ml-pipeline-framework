#!/usr/bin/env python3
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
        spark = SparkSession.builder \
            .appName("ML-Pipeline-DataProcessing") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
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
