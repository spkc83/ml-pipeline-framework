#!/bin/bash
set -e

# ML Pipeline Framework Docker Entrypoint Script
# This script handles initialization and command execution for the container

echo "🚀 Starting ML Pipeline Framework Container"
echo "Python version: $(python --version)"
echo "Spark version: $SPARK_VERSION"
echo "Build type: ${BUILD_TYPE:-cpu}"
echo "Working directory: $(pwd)"

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ Timeout waiting for $service_name"
    return 1
}

# Function to initialize MLflow
init_mlflow() {
    if [ "${MLFLOW_TRACKING_URI:-}" ]; then
        echo "🔧 Initializing MLflow connection..."
        
        # Extract host and port from tracking URI
        if [[ $MLFLOW_TRACKING_URI =~ http://([^:]+):([0-9]+) ]]; then
            mlflow_host="${BASH_REMATCH[1]}"
            mlflow_port="${BASH_REMATCH[2]}"
            
            if [ "$mlflow_host" != "localhost" ] && [ "$mlflow_host" != "127.0.0.1" ]; then
                wait_for_service "$mlflow_host" "$mlflow_port" "MLflow"
            fi
        fi
        
        # Test MLflow connection
        python -c "
import mlflow
import sys
try:
    mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
    experiments = mlflow.search_experiments()
    print('✅ MLflow connection successful')
except Exception as e:
    print(f'⚠️  MLflow connection failed: {e}')
    sys.exit(0)  # Don't fail container startup
"
    fi
}

# Function to initialize Spark
init_spark() {
    echo "🔧 Initializing Spark..."
    
    # Set Spark configuration based on available resources
    total_memory=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    cores=$(nproc)
    
    # Calculate optimal Spark memory settings (leave 1GB for system)
    if [ "$total_memory" -gt 4 ]; then
        driver_memory=$((total_memory - 2))
        executor_memory=$((total_memory - 2))
    else
        driver_memory=1
        executor_memory=1
    fi
    
    export SPARK_DRIVER_MEMORY="${driver_memory}g"
    export SPARK_EXECUTOR_MEMORY="${executor_memory}g"
    export SPARK_DRIVER_CORES=$cores
    export SPARK_EXECUTOR_CORES=$cores
    
    echo "   Driver memory: ${SPARK_DRIVER_MEMORY}"
    echo "   Executor memory: ${SPARK_EXECUTOR_MEMORY}"
    echo "   CPU cores: $cores"
    
    # Test Spark initialization
    python -c "
import pyspark
from pyspark.sql import SparkSession
try:
    spark = SparkSession.builder.appName('HealthCheck').getOrCreate()
    print('✅ Spark initialization successful')
    spark.stop()
except Exception as e:
    print(f'⚠️  Spark initialization failed: {e}')
"
}

# Function to check GPU availability
check_gpu() {
    if [ "${BUILD_TYPE:-cpu}" = "gpu" ]; then
        echo "🔧 Checking GPU availability..."
        
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
            
            # Test GPU with Python
            python -c "
import torch
try:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f'✅ GPU available: {device_count} device(s)')
        for i in range(device_count):
            print(f'   Device {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('⚠️  CUDA not available in PyTorch')
except Exception as e:
    print(f'⚠️  GPU check failed: {e}')
"
        else
            echo "⚠️  nvidia-smi not found"
        fi
    fi
}

# Function to initialize database connections
init_database() {
    if [ "${DB_HOST:-}" ] && [ "${DB_PORT:-}" ]; then
        echo "🔧 Checking database connectivity..."
        wait_for_service "$DB_HOST" "$DB_PORT" "Database"
        
        # Test database connection if credentials are available
        if [ "${DB_USER:-}" ] && [ "${DB_PASSWORD:-}" ] && [ "${DB_NAME:-}" ]; then
            python -c "
import psycopg2
import sys
try:
    conn = psycopg2.connect(
        host='$DB_HOST',
        port='$DB_PORT',
        database='$DB_NAME',
        user='$DB_USER',
        password='$DB_PASSWORD',
        connect_timeout=10
    )
    conn.close()
    print('✅ Database connection successful')
except Exception as e:
    print(f'⚠️  Database connection failed: {e}')
    sys.exit(0)  # Don't fail container startup
"
        fi
    fi
}

# Function to setup directories and permissions
setup_directories() {
    echo "🔧 Setting up directories..."
    
    # Ensure all required directories exist
    mkdir -p /app/logs /app/artifacts /app/models /app/data /app/configs /app/reports
    
    # Set proper permissions (if running as root, change ownership)
    if [ "$(id -u)" = "0" ]; then
        chown -R mluser:mluser /app/logs /app/artifacts /app/models /app/data /app/reports
    fi
    
    # Create MLflow artifacts directory if MLflow is configured
    if [ "${MLFLOW_ARTIFACT_ROOT:-}" ]; then
        mkdir -p "${MLFLOW_ARTIFACT_ROOT}"
        if [ "$(id -u)" = "0" ]; then
            chown -R mluser:mluser "${MLFLOW_ARTIFACT_ROOT}"
        fi
    fi
}

# Function to validate configuration
validate_config() {
    if [ "${CONFIG_FILE:-}" ] && [ -f "$CONFIG_FILE" ]; then
        echo "🔧 Validating configuration file: $CONFIG_FILE"
        
        python -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print('✅ Configuration file is valid')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"
    fi
}

# Function to run health checks
run_health_checks() {
    echo "🏥 Running health checks..."
    
    # Check Python imports
    python -c "
import sys
packages = [
    'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm',
    'mlflow', 'pyspark', 'great_expectations', 'shap'
]

failed_imports = []
for package in packages:
    try:
        __import__(package)
    except ImportError as e:
        failed_imports.append(f'{package}: {e}')

if failed_imports:
    print('⚠️  Some packages failed to import:')
    for failure in failed_imports:
        print(f'   {failure}')
else:
    print('✅ All core packages imported successfully')
"
    
    # Check available disk space
    df_output=$(df -h /app | tail -1)
    available_space=$(echo $df_output | awk '{print $4}')
    echo "💾 Available disk space: $available_space"
    
    # Check memory
    free -h | grep -E "Mem|Swap"
}

# Function to handle graceful shutdown
cleanup() {
    echo "🛑 Received shutdown signal, cleaning up..."
    
    # Stop any running Spark contexts
    python -c "
from pyspark.sql import SparkSession
try:
    spark = SparkSession.getActiveSession()
    if spark:
        spark.stop()
        print('✅ Spark context stopped')
except:
    pass
"
    
    # Kill any remaining processes
    pkill -f "pyspark" || true
    pkill -f "java.*spark" || true
    
    echo "🏁 Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main initialization
main() {
    echo "=================================================="
    echo "ML Pipeline Framework Container Initialization"
    echo "=================================================="
    
    # Setup directories first
    setup_directories
    
    # Run initialization functions
    init_spark
    check_gpu
    init_mlflow
    init_database
    validate_config
    run_health_checks
    
    echo "=================================================="
    echo "✅ Initialization completed successfully!"
    echo "=================================================="
    
    # Handle different execution modes
    case "${EXECUTION_MODE:-default}" in
        "jupyter")
            echo "🚀 Starting Jupyter Lab..."
            exec jupyter lab \
                --ip=0.0.0.0 \
                --port=8888 \
                --no-browser \
                --allow-root \
                --NotebookApp.token='' \
                --NotebookApp.password=''
            ;;
        "mlflow")
            echo "🚀 Starting MLflow server..."
            exec mlflow server \
                --host=0.0.0.0 \
                --port=5000 \
                --backend-store-uri="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}" \
                --default-artifact-root="${MLFLOW_ARTIFACT_ROOT:-./mlruns}"
            ;;
        "spark-master")
            echo "🚀 Starting Spark master..."
            exec $SPARK_HOME/sbin/start-master.sh -h 0.0.0.0 -p 7077 --webui-port 8080
            ;;
        "spark-worker")
            echo "🚀 Starting Spark worker..."
            if [ -z "${SPARK_MASTER_URL:-}" ]; then
                echo "❌ SPARK_MASTER_URL must be set for worker mode"
                exit 1
            fi
            exec $SPARK_HOME/sbin/start-worker.sh "$SPARK_MASTER_URL"
            ;;
        "daemon")
            echo "🚀 Running in daemon mode..."
            # Keep container running
            tail -f /dev/null
            ;;
        *)
            echo "🚀 Executing command: $*"
            exec "$@"
            ;;
    esac
}

# Execute main function with all arguments
main "$@"