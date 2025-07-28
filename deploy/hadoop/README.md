# Hadoop Deployment for ML Pipeline Framework

This directory contains scripts and configurations for deploying and running ML Pipeline Framework jobs on Hadoop clusters using YARN and Spark.

## Overview

The Hadoop deployment provides:
- **Batch Job Submission**: Submit ML training, prediction, and data processing jobs to Hadoop cluster
- **YARN Integration**: Leverage YARN for resource management and job scheduling  
- **Spark Runtime**: Use Spark for distributed processing of large datasets
- **HDFS Storage**: Store data, models, and artifacts on HDFS
- **Job Monitoring**: Monitor job progress and retrieve logs
- **Configuration Management**: Flexible configuration for different job types and environments

## Components

### Core Files

- **`submit_job.py`**: Python script for submitting various types of ML jobs to Hadoop cluster
- **`batch_submit.sh`**: Bash wrapper script providing convenient CLI for job submission and management
- **`job_config.yaml`**: Comprehensive configuration file for Hadoop job parameters
- **`scripts/`**: Directory containing Hadoop-specific job execution scripts

### Job Types Supported

1. **Training Jobs**: Train ML models on large datasets using Spark
2. **Prediction Jobs**: Run batch predictions on new data
3. **Data Processing Jobs**: ETL and feature engineering workflows
4. **Model Evaluation Jobs**: Evaluate model performance and generate reports

## Prerequisites

### Environment Setup

1. **Hadoop Cluster**: Running Hadoop cluster with YARN ResourceManager
2. **Spark**: Spark installation compatible with your Hadoop version
3. **Python**: Python 3.9+ with required ML libraries
4. **HDFS Access**: Read/write permissions to HDFS directories

### Environment Variables

```bash
export HADOOP_HOME=/opt/hadoop
export SPARK_HOME=/opt/spark
export HADOOP_CONF_DIR=/etc/hadoop/conf
export YARN_CONF_DIR=/etc/hadoop/conf
export PYTHON_PATH=/opt/conda/bin/python
```

### Required Permissions

```bash
# HDFS permissions
hdfs dfs -mkdir -p /ml-pipeline
hdfs dfs -chown ml-pipeline:ml-team /ml-pipeline
hdfs dfs -chmod 755 /ml-pipeline

# YARN queue access
# Ensure user has access to appropriate YARN queues
```

## Quick Start

### 1. Configuration

Edit `job_config.yaml` to match your environment:

```yaml
# Update paths and cluster details
environment_config:
  namenode_url: "hdfs://your-namenode:9000"
  resourcemanager_url: "your-resourcemanager:8032"
  
# Adjust resource allocation
spark_config:
  num_executors: 10
  executor_cores: 4
  executor_memory: "8g"
```

### 2. Submit a Training Job

```bash
# Basic training job submission
./batch_submit.sh submit -t training -n "churn-model-v1"

# Training job with custom parameters
./batch_submit.sh submit -t training \
  -n "churn-model-v1" \
  -q "ml-queue" \
  --num-executors 20 \
  --executor-memory "16g"

# Dry run to see command without executing
./batch_submit.sh submit -t training --dry-run
```

### 3. Submit a Prediction Job

```bash
# Batch prediction job
./batch_submit.sh submit -t prediction -n "batch-predictions"

# Custom prediction with specific input/output paths
python3 submit_job.py prediction \
  --config job_config.yaml \
  --app-args "--input-path /ml-pipeline/data/new_customers.parquet --output-path /ml-pipeline/predictions/$(date +%Y%m%d)"
```

### 4. Monitor Jobs

```bash
# List all running ML pipeline jobs
./batch_submit.sh list

# Check status of specific job
./batch_submit.sh status -a application_1234567890_0001

# Monitor job progress in real-time
./batch_submit.sh monitor -a application_1234567890_0001

# Get job logs
./batch_submit.sh logs -a application_1234567890_0001
```

## Advanced Usage

### Custom Job Scripts

Create custom Hadoop job scripts in the `scripts/` directory:

```python
#!/usr/bin/env python3
"""Custom Hadoop ML Job"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline_framework import PipelineOrchestrator
from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder.appName("CustomMLJob").getOrCreate()
    
    # Your custom ML logic here
    # Load data from HDFS
    # Process with Spark
    # Save results back to HDFS
    
    spark.stop()

if __name__ == "__main__":
    main()
```

### Environment-Specific Configurations

Create environment-specific config files:

```bash
# Development environment
cp job_config.yaml job_config_dev.yaml
# Edit for dev cluster settings

# Production environment  
cp job_config.yaml job_config_prod.yaml
# Edit for prod cluster settings

# Submit with specific config
./batch_submit.sh submit -t training -c job_config_prod.yaml
```

### Resource Management

Optimize resource allocation based on job requirements:

```yaml
# For large training jobs
spark_config:
  num_executors: 50
  executor_cores: 8
  executor_memory: "32g"
  driver_memory: "16g"

# For batch predictions
spark_config:
  num_executors: 20
  executor_cores: 4
  executor_memory: "16g"
  driver_memory: "8g"
```

## Data Management

### HDFS Directory Structure

```
/ml-pipeline/
├── data/
│   ├── training/          # Training datasets
│   ├── validation/        # Validation datasets
│   ├── input/            # Input data for predictions
│   └── output/           # Output predictions
├── models/               # Trained models
├── artifacts/            # Model artifacts and metadata
├── configs/              # Configuration files
├── logs/                 # Job logs
├── checkpoints/          # Spark checkpoints
└── metrics/              # Performance metrics
```

### Data Upload

```bash
# Upload training data
hdfs dfs -put local_data.csv /ml-pipeline/data/training/

# Upload model files
hdfs dfs -put -r model_directory/ /ml-pipeline/models/

# Verify uploads
hdfs dfs -ls -R /ml-pipeline/
```

## Job Configuration

### Training Job Configuration

```yaml
job_specific_config:
  training:
    additional_jars: []
    additional_py_files: []
    custom_spark_conf:
      spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes: "256MB"
      spark.sql.adaptive.advisoryPartitionSizeInBytes: "128MB"
    
model_config:
  algorithm: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
  cv_folds: 5
  test_size: 0.2
```

### Prediction Job Configuration

```yaml
job_specific_config:
  prediction:
    batch_size: 10000
    prediction_timeout: "1h"
    output_compression: "snappy"
    
data_config:
  prediction_input_path: "hdfs://namenode:9000/ml-pipeline/data/input/batch_data.parquet"
  predictions_output_path: "hdfs://namenode:9000/ml-pipeline/predictions"
```

## Monitoring and Logging

### Job Monitoring

```bash
# Real-time monitoring
./batch_submit.sh monitor -a application_1234567890_0001

# Check job history
yarn application -list -appStates FINISHED,FAILED

# Get detailed application info
yarn application -status application_1234567890_0001
```

### Log Analysis

```bash
# Get all logs
./batch_submit.sh logs -a application_1234567890_0001

# Get driver logs only
yarn logs -applicationId application_1234567890_0001 -containerId container_*_01_000001

# Get executor logs
yarn logs -applicationId application_1234567890_0001 | grep -A 20 "Container.*000002"
```

### Performance Monitoring

Access Spark UI and YARN UI for detailed performance metrics:

- **Spark History Server**: `http://historyserver:18080`
- **YARN ResourceManager**: `http://resourcemanager:8088`
- **HDFS NameNode**: `http://namenode:9870`

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Increase executor memory
   --executor-memory "16g" --driver-memory "8g"
   ```

2. **HDFS Permission Denied**
   ```bash
   # Fix HDFS permissions
   hdfs dfs -chmod -R 755 /ml-pipeline
   hdfs dfs -chown -R ml-pipeline:ml-team /ml-pipeline
   ```

3. **YARN Queue Access**
   ```bash
   # Check queue capacity
   yarn queue -status ml-queue
   
   # List available queues
   yarn queue -list
   ```

4. **Spark Serialization Issues**
   ```yaml
   # Add Kryo serializer configuration
   spark_config:
     spark.serializer: "org.apache.spark.serializer.KryoSerializer"
     spark.kryo.unsafe: true
   ```

### Log Locations

- **Application Logs**: `yarn logs -applicationId <app_id>`
- **Spark History**: `$SPARK_HOME/logs/`
- **HDFS Logs**: `/ml-pipeline/logs/`
- **Local Logs**: `hadoop_job_submission.log`

### Debug Mode

```bash
# Enable debug logging
export SPARK_SUBMIT_OPTS="-Dlog4j.configuration=file:log4j-debug.properties"

# Submit with verbose output
./batch_submit.sh submit -t training --verbose
```

## Security Considerations

### Kerberos Authentication

For secure clusters, configure Kerberos:

```yaml
security_config:
  enable_kerberos: true
  principal: "ml-pipeline@REALM.COM"
  keytab: "/etc/security/keytabs/ml-pipeline.keytab"
```

### SSL/TLS Configuration

```yaml
security_config:
  enable_ssl: true
  keystore_path: "/etc/ssl/keystore.jks"
  keystore_password: "password"
  truststore_path: "/etc/ssl/truststore.jks"
  truststore_password: "password"
```

## Performance Optimization

### Spark Optimization

```yaml
spark_config:
  # Enable adaptive query execution
  spark.sql.adaptive.enabled: true
  spark.sql.adaptive.coalescePartitions.enabled: true
  spark.sql.adaptive.skewJoin.enabled: true
  
  # Optimize serialization
  spark.serializer: "org.apache.spark.serializer.KryoSerializer"
  spark.kryo.unsafe: true
  
  # Dynamic allocation
  spark.dynamicAllocation.enabled: true
  spark.dynamicAllocation.minExecutors: 2
  spark.dynamicAllocation.maxExecutors: 50
```

### Resource Tuning

```yaml
resource_limits:
  max_driver_memory: "16g"
  max_executor_memory: "32g"
  max_executor_cores: 8
  max_total_cores: 400
```

## Integration Examples

### With MLflow

```python
# Training job with MLflow tracking
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("hadoop_experiments")

with mlflow.start_run():
    # Train model
    # Log metrics, parameters, artifacts
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_model(model, "model")
```

### With Airflow

```python
# Airflow DAG for Hadoop jobs
from airflow.operators.bash import BashOperator

hadoop_training_task = BashOperator(
    task_id='submit_hadoop_training',
    bash_command='cd /path/to/deploy/hadoop && ./batch_submit.sh submit -t training -n "{{ ds }}-training"',
    dag=dag
)
```

## Support and Troubleshooting

For additional support:

1. Check application logs: `yarn logs -applicationId <app_id>`
2. Review Spark UI for performance insights
3. Verify HDFS permissions and connectivity
4. Ensure proper resource allocation for job requirements
5. Check cluster capacity and queue availability

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Hadoop YARN Documentation](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html)
- [HDFS Commands Guide](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HDFSCommands.html)