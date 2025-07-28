# Multi-stage Dockerfile for ML Pipeline Framework
# Supports both CPU and GPU versions with Spark runtime

ARG PYTHON_VERSION=3.9
ARG SPARK_VERSION=3.4.1
ARG HADOOP_VERSION=3
ARG CUDA_VERSION=11.8
ARG BUILD_TYPE=cpu

# Base stage - common dependencies
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    openjdk-11-jdk \
    procps \
    libgomp1 \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Install Spark
ARG SPARK_VERSION
ARG HADOOP_VERSION
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python

RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && tar xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /opt/ \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "$SPARK_HOME" \
    && rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && chown -R root:root $SPARK_HOME

# Create spark configuration
RUN mkdir -p $SPARK_HOME/conf
COPY docker/spark-defaults.conf $SPARK_HOME/conf/
COPY docker/log4j.properties $SPARK_HOME/conf/

# GPU stage - NVIDIA CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 as gpu-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    openjdk-11-jdk \
    procps \
    libgomp1 \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Install Spark (same as base stage)
ARG SPARK_VERSION
ARG HADOOP_VERSION
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python

RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && tar xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /opt/ \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "$SPARK_HOME" \
    && rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && chown -R root:root $SPARK_HOME

# Create spark configuration
RUN mkdir -p $SPARK_HOME/conf
COPY docker/spark-defaults.conf $SPARK_HOME/conf/
COPY docker/log4j.properties $SPARK_HOME/conf/

# Select base image based on build type
FROM ${BUILD_TYPE}-base as selected-base

# Application stage
FROM selected-base as app

# Create application user
RUN groupadd -r mluser && useradd -r -g mluser -d /home/mluser -s /bin/bash mluser
RUN mkdir -p /home/mluser && chown -R mluser:mluser /home/mluser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install CPU-specific packages
RUN if [ "$BUILD_TYPE" = "cpu" ]; then \
    pip install -r requirements.txt && \
    pip install \
        torch==2.0.1+cpu \
        torchvision==0.15.2+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html; \
fi

# Install GPU-specific packages
RUN if [ "$BUILD_TYPE" = "gpu" ]; then \
    pip install -r requirements.txt && \
    pip install \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install \
        cudf-cu11 \
        cuml-cu11 \
        cugraph-cu11 \
        --extra-index-url=https://pypi.nvidia.com; \
fi

# Install additional ML frameworks
RUN pip install \
    xgboost \
    lightgbm \
    catboost \
    optuna \
    hyperopt \
    scikit-optimize \
    mlflow \
    great-expectations \
    evidently \
    shap \
    lime \
    alibi \
    dask[complete] \
    polars \
    duckdb \
    h2o \
    pyspark==$SPARK_VERSION \
    delta-spark \
    koalas

# Install development and testing dependencies
RUN pip install -r requirements-dev.txt

# Copy application code
COPY . .

# Install the ML framework package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/artifacts /app/models /app/data /app/configs /app/notebooks

# Set proper permissions
RUN chown -R mluser:mluser /app /home/mluser
RUN chmod +x /app/run_pipeline.py

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy health check script
COPY docker/healthcheck.py /healthcheck.py
RUN chmod +x /healthcheck.py

# Set environment variables for the application
ENV PYTHONPATH=/app:$PYTHONPATH
ENV ML_PIPELINE_HOME=/app
ENV SPARK_CONF_DIR=$SPARK_HOME/conf
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop

# Configure Spark for ML workloads
ENV SPARK_DRIVER_MEMORY=2g
ENV SPARK_EXECUTOR_MEMORY=2g
ENV SPARK_DRIVER_MAX_RESULT_SIZE=1g
ENV SPARK_SQL_ADAPTIVE_ENABLED=true
ENV SPARK_SQL_ADAPTIVE_COALESCE_PARTITIONS_ENABLED=true

# GPU-specific Spark configuration
RUN if [ "$BUILD_TYPE" = "gpu" ]; then \
    echo "spark.sql.execution.arrow.pyspark.enabled true" >> $SPARK_HOME/conf/spark-defaults.conf && \
    echo "spark.sql.execution.arrow.maxRecordsPerBatch 10000" >> $SPARK_HOME/conf/spark-defaults.conf; \
fi

# Expose ports
EXPOSE 8080 8081 4040 8888 5000

# Add labels
LABEL maintainer="ML Pipeline Team" \
      version="1.0.0" \
      description="ML Pipeline Framework with Spark and optional GPU support" \
      build_type="$BUILD_TYPE" \
      spark_version="$SPARK_VERSION" \
      python_version="$PYTHON_VERSION"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python /healthcheck.py

# Switch to non-root user
USER mluser

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "run_pipeline.py", "--help"]