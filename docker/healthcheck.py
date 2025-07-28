#!/usr/bin/env python3
"""
Health check script for ML Pipeline Framework container.

This script performs comprehensive health checks to ensure the container
is running properly and all dependencies are available.
"""

import sys
import os
import time
import subprocess
import socket
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_imports():
    """Check if critical Python packages can be imported."""
    logger.info("Checking Python package imports...")
    
    critical_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'mlflow',
        'yaml',
        'requests'
    ]
    
    optional_packages = [
        'pyspark',
        'xgboost',
        'lightgbm',
        'catboost',
        'h2o',
        'great_expectations',
        'shap',
        'torch',
        'dask',
        'polars',
        'duckdb'
    ]
    
    failed_critical = []
    failed_optional = []
    
    # Check critical packages
    for package in critical_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Critical package {package} failed: {e}")
            failed_critical.append(package)
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️  Optional package {package} not available: {e}")
            failed_optional.append(package)
    
    if failed_critical:
        logger.error(f"Critical packages failed: {failed_critical}")
        return False
    
    logger.info("All critical packages imported successfully")
    return True


def check_spark():
    """Check Spark availability and configuration."""
    logger.info("Checking Spark availability...")
    
    try:
        from pyspark.sql import SparkSession
        
        # Try to create a minimal Spark session
        spark = SparkSession.builder \
            .appName("HealthCheck") \
            .config("spark.driver.memory", "512m") \
            .config("spark.executor.memory", "512m") \
            .getOrCreate()
        
        # Test basic functionality
        df = spark.range(10)
        count = df.count()
        
        spark.stop()
        
        if count == 10:
            logger.info("✅ Spark health check passed")
            return True
        else:
            logger.error(f"❌ Spark test failed: expected 10, got {count}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Spark health check failed: {e}")
        return False


def check_gpu():
    """Check GPU availability if running GPU build."""
    build_type = os.environ.get('BUILD_TYPE', 'cpu')
    
    if build_type != 'gpu':
        logger.info("CPU build - skipping GPU check")
        return True
    
    logger.info("Checking GPU availability...")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode != 0:
            logger.error("❌ nvidia-smi failed")
            return False
        
        # Check PyTorch CUDA
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"✅ GPU available: {device_count} device(s)")
            
            # Test basic GPU operation
            x = torch.tensor([1.0, 2.0]).cuda()
            y = x + 1
            result = y.cpu().numpy()
            
            if result[0] == 2.0 and result[1] == 3.0:
                logger.info("✅ GPU computation test passed")
                return True
            else:
                logger.error("❌ GPU computation test failed")
                return False
        else:
            logger.error("❌ CUDA not available in PyTorch")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")
        return False


def check_directories():
    """Check if required directories exist and are writable."""
    logger.info("Checking directories...")
    
    required_dirs = [
        '/app/logs',
        '/app/artifacts', 
        '/app/models',
        '/app/data',
        '/app/configs'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        
        # Check if directory exists
        if not path.exists():
            logger.error(f"❌ Directory does not exist: {dir_path}")
            return False
        
        # Check if directory is writable
        if not os.access(dir_path, os.W_OK):
            logger.error(f"❌ Directory not writable: {dir_path}")
            return False
        
        logger.info(f"✅ Directory OK: {dir_path}")
    
    return True


def check_disk_space():
    """Check available disk space."""
    logger.info("Checking disk space...")
    
    try:
        import shutil
        
        # Check space in /app directory
        total, used, free = shutil.disk_usage('/app')
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_percent = (used / total) * 100
        
        logger.info(f"Disk usage: {used_percent:.1f}% used, {free_gb:.1f}GB free of {total_gb:.1f}GB total")
        
        # Warn if less than 1GB free
        if free_gb < 1.0:
            logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free")
            return False
        
        logger.info("✅ Sufficient disk space available")
        return True
        
    except Exception as e:
        logger.error(f"❌ Disk space check failed: {e}")
        return False


def check_memory():
    """Check available memory."""
    logger.info("Checking memory...")
    
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        # Parse memory information
        lines = meminfo.split('\n')
        mem_total = None
        mem_available = None
        
        for line in lines:
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) * 1024  # Convert KB to bytes
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
        
        if mem_total and mem_available:
            total_gb = mem_total / (1024**3)
            available_gb = mem_available / (1024**3)
            used_percent = ((mem_total - mem_available) / mem_total) * 100
            
            logger.info(f"Memory usage: {used_percent:.1f}% used, {available_gb:.1f}GB available of {total_gb:.1f}GB total")
            
            # Warn if less than 500MB available
            if available_gb < 0.5:
                logger.warning(f"⚠️  Low memory: {available_gb:.1f}GB available")
                return False
            
            logger.info("✅ Sufficient memory available")
            return True
        else:
            logger.error("❌ Could not parse memory information")
            return False
            
    except Exception as e:
        logger.error(f"❌ Memory check failed: {e}")
        return False


def check_network_connectivity():
    """Check basic network connectivity."""
    logger.info("Checking network connectivity...")
    
    # Test connections to common services
    test_hosts = [
        ('8.8.8.8', 53),  # Google DNS
        ('pypi.org', 443),  # PyPI for package downloads
    ]
    
    # Add MLflow server if configured
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if mlflow_uri and mlflow_uri.startswith('http://'):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(mlflow_uri)
            if parsed.hostname and parsed.port:
                test_hosts.append((parsed.hostname, parsed.port))
        except Exception:
            pass
    
    connectivity_ok = True
    
    for host, port in test_hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ Network connection OK: {host}:{port}")
            else:
                logger.warning(f"⚠️  Network connection failed: {host}:{port}")
                connectivity_ok = False
                
        except Exception as e:
            logger.warning(f"⚠️  Network test failed for {host}:{port}: {e}")
            connectivity_ok = False
    
    return connectivity_ok


def check_ml_pipeline_framework():
    """Check if the ML Pipeline Framework is properly installed."""
    logger.info("Checking ML Pipeline Framework...")
    
    try:
        # Test basic framework imports
        sys.path.append('/app/src')
        
        # Test configuration parsing
        from utils.config_parser import ConfigParser
        logger.info("✅ ConfigParser available")
        
        # Test basic components that exist
        from data_access.factory import DataConnectorFactory
        logger.info("✅ DataConnectorFactory available")
        
        from models.factory import ModelFactory
        logger.info("✅ ModelFactory available")
        
        # Test CLI availability
        from cli import main as cli_main
        logger.info("✅ CLI module available")
        
        logger.info("✅ Core components available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML Pipeline Framework check failed: {e}")
        return False


def main():
    """Run all health checks."""
    logger.info("🏥 Starting ML Pipeline Framework health check...")
    
    checks = [
        ("Python imports", check_python_imports),
        ("Directories", check_directories),
        ("Disk space", check_disk_space),
        ("Memory", check_memory),
        ("Network connectivity", check_network_connectivity),
        ("ML Pipeline Framework", check_ml_pipeline_framework),
        ("Spark", check_spark),
        ("GPU", check_gpu),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            logger.info(f"Running {check_name} check...")
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            logger.error(f"❌ {check_name} check failed with exception: {e}")
            failed_checks.append(check_name)
    
    # Summary
    if failed_checks:
        logger.error(f"❌ Health check failed. Failed checks: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        logger.info("✅ All health checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()