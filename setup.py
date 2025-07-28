#!/usr/bin/env python
"""Setup script for ML Pipeline Framework."""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're using Python 3.8+
if sys.version_info < (3, 8):
    sys.exit('Python 3.8 or higher is required.')

# Get the long description from the README file
HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')

# Load version from version file
def get_version():
    version_file = HERE / "src" / "__version__.py"
    if version_file.exists():
        with open(version_file) as f:
            exec(f.read())
            return locals()['__version__']
    return "0.1.0"

VERSION = get_version()

# Core dependencies
INSTALL_REQUIRES = [
    # Core data processing
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "polars>=0.18.0",
    "duckdb>=0.8.0",
    
    # Machine learning core
    "scikit-learn>=1.2.0",
    "xgboost>=1.7.0",
    "lightgbm>=3.3.0",
    "catboost>=1.1.0",
    
    # Deep learning (optional)
    "tensorflow>=2.10.0;python_version<'3.12'",
    "torch>=1.13.0",
    
    # Model interpretability
    "shap>=0.41.0",
    "lime>=0.2.0",
    "eli5>=0.13.0",
    
    # Data connectors
    "sqlalchemy>=1.4.0",
    "psycopg2-binary>=2.9.0",
    "pymysql>=1.0.0",
    "snowflake-connector-python>=3.0.0",
    "redis>=4.3.0",
    
    # Big data processing
    "pyspark>=3.3.0",
    "dask[complete]>=2022.0.0",
    
    # Configuration and utilities
    "pyyaml>=6.0",
    "hydra-core>=1.2.0",
    "omegaconf>=2.2.0",
    "python-dotenv>=0.20.0",
    
    # Monitoring and logging
    "mlflow>=2.0.0",
    "wandb>=0.13.0",
    "structlog>=22.0.0",
    
    # Web framework (for APIs)
    "fastapi>=0.85.0",
    "uvicorn>=0.18.0",
    "pydantic>=1.10.0",
    
    # Visualization
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.10.0",
    "ipywidgets>=8.0.0",
    
    # Imbalanced learning
    "imbalanced-learn>=0.9.0",
    
    # Hyperparameter optimization
    "optuna>=3.0.0",
    "hyperopt>=0.2.7",
    "scikit-optimize>=0.9.0",
    
    # Cloud and deployment
    "kubernetes>=24.0.0",
    "docker>=6.0.0",
    "boto3>=1.24.0",
    "google-cloud-storage>=2.5.0",
    "azure-storage-blob>=12.12.0",
    
    # Security and compliance
    "cryptography>=37.0.0",
    "pyjwt>=2.4.0",
    
    # Data validation
    "cerberus>=1.3.0",
    "marshmallow>=3.17.0",
    "great-expectations>=0.15.0",
    
    # Performance
    "joblib>=1.2.0",
    "numba>=0.56.0",
    "pyarrow>=9.0.0",
    
    # Utilities
    "click>=8.1.0",
    "tqdm>=4.64.0",
    "requests>=2.28.0",
    "urllib3>=1.26.0",
    "packaging>=21.0.0",
    "importlib-metadata>=4.0.0",
]

# Development dependencies
DEV_REQUIRES = [
    # Testing
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "pytest-mock>=3.0.0",
    "pytest-timeout>=2.0.0",
    "pytest-benchmark>=4.0.0",
    
    # Code quality
    "black>=22.0.0",
    "isort>=5.0.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "bandit>=1.7.0",
    "safety>=2.0.0",
    
    # Documentation
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0",
    
    # Development tools
    "pre-commit>=2.0.0",
    "pip-tools>=6.0.0",
    "build>=0.8.0",
    "twine>=4.0.0",
]

# Cloud-specific dependencies
CLOUD_REQUIRES = [
    # AWS
    "boto3>=1.24.0",
    "botocore>=1.27.0",
    "s3fs>=2022.8.0",
    
    # GCP
    "google-cloud-storage>=2.5.0",
    "google-cloud-bigquery>=3.3.0",
    "gcsfs>=2022.8.0",
    
    # Azure
    "azure-storage-blob>=12.12.0",
    "azure-identity>=1.10.0",
    "adlfs>=2022.8.0",
]

# Spark dependencies
SPARK_REQUIRES = [
    "pyspark>=3.3.0",
    "delta-spark>=2.1.0",
    "koalas>=1.8.0",
]

# GPU dependencies
GPU_REQUIRES = [
    "cupy-cuda11x>=11.0.0",
    "cudf-cu11>=22.08.0",
    "cuml-cu11>=22.08.0",
    "xgboost[gpu]>=1.7.0",
    "lightgbm[gpu]>=3.3.0",
]

# Documentation dependencies
DOCS_REQUIRES = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.0.0",
    "myst-parser>=0.18.0",
    "nbsphinx>=0.8.0",
    "sphinx-copybutton>=0.5.0",
]

# All optional dependencies
ALL_REQUIRES = (
    DEV_REQUIRES + 
    CLOUD_REQUIRES + 
    SPARK_REQUIRES + 
    DOCS_REQUIRES
)

setup(
    name="ml-pipeline-framework",
    version=VERSION,
    description="Enterprise-grade ML Pipeline Framework for Production Environments",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Organization",
    author_email="ml-team@your-org.com",
    url="https://github.com/your-org/ml-pipeline-framework",
    project_urls={
        "Documentation": "https://ml-pipeline-framework.readthedocs.io/",
        "Source": "https://github.com/your-org/ml-pipeline-framework",
        "Tracker": "https://github.com/your-org/ml-pipeline-framework/issues",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Typing :: Typed",
    ],
    keywords=[
        "machine-learning", "mlops", "automl", "pipeline", "enterprise",
        "production", "fraud-detection", "interpretability", "fairness",
        "compliance", "monitoring", "kubernetes", "docker", "cloud"
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "ml_pipeline_framework": [
            "configs/*.yaml",
            "configs/*.yml",
            "configs/*.json",
            "templates/*.yaml",
            "templates/*.yml",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "cloud": CLOUD_REQUIRES,
        "spark": SPARK_REQUIRES,
        "gpu": GPU_REQUIRES,
        "docs": DOCS_REQUIRES,
        "all": ALL_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            "ml-pipeline=src.cli:main",
        ],
    },
    zip_safe=False,
    test_suite="tests",
    tests_require=DEV_REQUIRES,
    cmdclass={},
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python",
        },
    },
)

# Post-installation checks
def post_install_check():
    """Perform post-installation verification."""
    try:
        import src
        print("✅ ML Pipeline Framework installed successfully!")
        print(f"Version: {VERSION}")
        print("Run 'ml-pipeline --help' to get started.")
    except ImportError as e:
        print(f"❌ Installation verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    post_install_check()