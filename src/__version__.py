"""
Version information for ML Pipeline Framework.

This module provides comprehensive version information including feature flags,
dependency requirements, and compatibility checks for production deployment.
"""

import os
import sys
import platform
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Core version information
__version__ = "2.0.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Build and release information
__build__ = os.environ.get("BUILD_NUMBER", "dev")
__release_date__ = "2024-12-01"
__git_revision__ = os.environ.get("GIT_COMMIT", "main")
__build_timestamp__ = datetime.now().isoformat()

# Feature flags for version compatibility
FEATURES = {
    "automl": True,
    "interpretability": True,
    "fraud_detection": True,
    "enterprise_security": True,
    "cloud_deployment": True,
    "kubernetes_support": True,
    "spark_integration": True,
    "gpu_acceleration": True,
    "model_monitoring": True,
    "compliance_reporting": True,
    "multi_engine_processing": True,
    "advanced_explainability": True,
    "a_b_testing": True,
    "fairness_monitoring": True,
    "cost_sensitive_learning": True,
    "automated_feature_engineering": True,
    "distributed_computing": False,  # Optional feature
    "real_time_inference": True,
    "batch_prediction": True,
    "model_versioning": True,
    "audit_logging": True,
}

# API version for compatibility
API_VERSION = "v2"
API_COMPATIBILITY = ["v1", "v2"]  # Supported API versions

# Minimum required versions for key dependencies
MIN_VERSIONS = {
    "python": "3.8.0",
    "pandas": "1.5.0",
    "numpy": "1.21.0",
    "scikit-learn": "1.3.0",
    "xgboost": "1.7.0",
    "lightgbm": "3.3.0",
    "catboost": "1.1.0",
    "shap": "0.41.0",
    "lime": "0.2.0",
    "mlflow": "2.6.0",
    "pyyaml": "6.0.0",
    "click": "8.0.0",
    "pydantic": "1.10.0",
    "joblib": "1.2.0",
}

# Optional dependencies with their minimum versions
OPTIONAL_VERSIONS = {
    "polars": "0.18.0",
    "duckdb": "0.8.0",
    "psycopg2-binary": "2.9.0",
    "pymysql": "1.0.0",
    "snowflake-connector-python": "3.0.0",
    "boto3": "1.26.0",
    "azure-storage-blob": "12.0.0",
    "google-cloud-storage": "2.0.0",
    "kubernetes": "26.0.0",
    "prometheus-client": "0.16.0",
    "grafana-api": "1.0.0",
}

# Development dependencies
DEV_VERSIONS = {
    "pytest": "7.0.0",
    "pytest-cov": "4.0.0",
    "black": "23.0.0",
    "flake8": "6.0.0",
    "mypy": "1.0.0",
    "pre-commit": "3.0.0",
    "sphinx": "6.0.0",
    "mkdocs": "1.4.0",
}

def get_version_info() -> Dict[str, Any]:
    """
    Get comprehensive version information.
    
    Returns:
        Dictionary containing version, build, and system information
    """
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build": __build__,
        "release_date": __release_date__,
        "git_revision": __git_revision__,
        "build_timestamp": __build_timestamp__,
        "api_version": API_VERSION,
        "api_compatibility": API_COMPATIBILITY,
        "features": FEATURES,
        "min_versions": MIN_VERSIONS,
        "optional_versions": OPTIONAL_VERSIONS,
        "system_info": get_system_info(),
    }

def get_system_info() -> Dict[str, str]:
    """
    Get system information for debugging and compatibility.
    
    Returns:
        Dictionary containing system details
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "machine": platform.machine(),
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
    }

def check_compatibility(verbose: bool = False) -> bool:
    """
    Check if current environment meets minimum requirements.
    
    Args:
        verbose: If True, print detailed compatibility information
        
    Returns:
        True if compatible, raises RuntimeError if not
        
    Raises:
        RuntimeError: If compatibility requirements are not met
    """
    # Check Python version
    python_version = ".".join(map(str, sys.version_info[:3]))
    min_python = tuple(map(int, MIN_VERSIONS["python"].split(".")))
    
    if sys.version_info < min_python:
        raise RuntimeError(
            f"Python {MIN_VERSIONS['python']} or higher is required. "
            f"Current version: {python_version}"
        )
    
    if verbose:
        print(f"✅ Python version: {python_version} (>= {MIN_VERSIONS['python']})")
    
    # Check core dependencies
    missing_deps = []
    incompatible_deps = []
    
    for package, min_version in MIN_VERSIONS.items():
        if package == "python":
            continue
            
        try:
            # Try to import and check version
            if package == "scikit-learn":
                import sklearn
                current_version = sklearn.__version__
                package_name = "sklearn"
            elif package == "pyyaml":
                import yaml
                current_version = getattr(yaml, "__version__", "unknown")
                package_name = "yaml"
            else:
                module = __import__(package.replace("-", "_"))
                current_version = getattr(module, "__version__", "unknown")
                package_name = package
            
            if current_version != "unknown":
                current_tuple = tuple(map(int, current_version.split(".")[:3]))
                min_tuple = tuple(map(int, min_version.split(".")[:3]))
                
                if current_tuple < min_tuple:
                    incompatible_deps.append((package, current_version, min_version))
                elif verbose:
                    print(f"✅ {package_name}: {current_version} (>= {min_version})")
            elif verbose:
                print(f"⚠️  {package_name}: version unknown")
                
        except ImportError:
            missing_deps.append(package)
    
    # Report issues
    if missing_deps:
        raise RuntimeError(
            f"Missing required dependencies: {', '.join(missing_deps)}. "
            f"Install with: pip install {' '.join(missing_deps)}"
        )
    
    if incompatible_deps:
        error_msg = "Incompatible dependency versions:\n"
        for pkg, current, minimum in incompatible_deps:
            error_msg += f"  {pkg}: {current} (requires >= {minimum})\n"
        error_msg += f"Upgrade with: pip install --upgrade {' '.join([pkg for pkg, _, _ in incompatible_deps])}"
        raise RuntimeError(error_msg)
    
    if verbose:
        print("✅ All core dependencies are compatible")
    
    return True

def check_optional_dependencies(verbose: bool = False) -> Dict[str, bool]:
    """
    Check availability of optional dependencies.
    
    Args:
        verbose: If True, print detailed information about optional dependencies
        
    Returns:
        Dictionary mapping dependency names to availability status
    """
    availability = {}
    
    for package, min_version in OPTIONAL_VERSIONS.items():
        try:
            # Special handling for some packages
            if package == "psycopg2-binary":
                import psycopg2
                current_version = psycopg2.__version__
                package_name = "psycopg2"
            elif package == "snowflake-connector-python":
                import snowflake.connector
                current_version = snowflake.connector.__version__
                package_name = "snowflake-connector"
            elif package == "azure-storage-blob":
                import azure.storage.blob
                current_version = getattr(azure.storage.blob, "__version__", "unknown")
                package_name = "azure-storage-blob"
            elif package == "google-cloud-storage":
                import google.cloud.storage
                current_version = getattr(google.cloud.storage, "__version__", "unknown")
                package_name = "google-cloud-storage"
            else:
                module = __import__(package.replace("-", "_"))
                current_version = getattr(module, "__version__", "unknown")
                package_name = package
            
            availability[package] = True
            
            if verbose:
                if current_version != "unknown":
                    print(f"✅ {package_name}: {current_version} (optional)")
                else:
                    print(f"✅ {package_name}: available (version unknown)")
                    
        except ImportError:
            availability[package] = False
            if verbose:
                print(f"❌ {package_name}: not available (optional)")
    
    return availability

def get_feature_status() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed status of all features including dependency requirements.
    
    Returns:
        Dictionary mapping feature names to their status and requirements
    """
    optional_deps = check_optional_dependencies()
    
    feature_status = {}
    
    for feature, enabled in FEATURES.items():
        status = {
            "enabled": enabled,
            "available": enabled,  # Start with enabled status
            "requirements": [],
            "missing_dependencies": [],
        }
        
        # Define feature-specific requirements
        feature_requirements = {
            "cloud_deployment": ["boto3", "azure-storage-blob", "google-cloud-storage"],
            "kubernetes_support": ["kubernetes"],
            "distributed_computing": ["dask", "ray"],
            "gpu_acceleration": ["cudf", "cuml", "cupy"],
            "model_monitoring": ["prometheus-client", "grafana-api"],
            "multi_engine_processing": ["polars", "duckdb"],
        }
        
        if feature in feature_requirements:
            requirements = feature_requirements[feature]
            status["requirements"] = requirements
            
            missing = []
            for req in requirements:
                if req in optional_deps and not optional_deps[req]:
                    missing.append(req)
            
            status["missing_dependencies"] = missing
            
            # Feature is available only if enabled and all deps are available
            if enabled and missing:
                status["available"] = False
        
        feature_status[feature] = status
    
    return feature_status

def print_compatibility_report():
    """Print a comprehensive compatibility and feature report."""
    print("🔧 ML Pipeline Framework Compatibility Report")
    print("=" * 60)
    
    # Version information
    version_info = get_version_info()
    print(f"Version: {version_info['version']}")
    print(f"Build: {version_info['build']}")
    print(f"API Version: {version_info['api_version']}")
    print(f"Release Date: {version_info['release_date']}")
    print()
    
    # System information
    print("💻 System Information:")
    sys_info = version_info['system_info']
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Python: {sys_info['python_version']} ({sys_info['python_implementation']})")
    print(f"  Architecture: {sys_info['architecture']}")
    print()
    
    # Core compatibility
    print("🔍 Core Compatibility Check:")
    try:
        check_compatibility(verbose=True)
        print()
    except RuntimeError as e:
        print(f"❌ Compatibility Error: {e}")
        return
    
    # Optional dependencies
    print("📦 Optional Dependencies:")
    check_optional_dependencies(verbose=True)
    print()
    
    # Feature status
    print("🎯 Feature Status:")
    feature_status = get_feature_status()
    
    for feature, status in feature_status.items():
        if status["enabled"]:
            if status["available"]:
                icon = "✅"
                availability = "Available"
            else:
                icon = "⚠️ "
                missing = ", ".join(status["missing_dependencies"])
                availability = f"Unavailable (missing: {missing})"
        else:
            icon = "❌"
            availability = "Disabled"
        
        print(f"  {icon} {feature}: {availability}")
    
    print()
    print("✅ Compatibility report complete!")

if __name__ == "__main__":
    # Run compatibility report when module is executed directly
    print_compatibility_report()