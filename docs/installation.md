# Installation Guide

This guide will help you install and set up the ML Pipeline Framework for your environment.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space
- **CPU**: 2 cores minimum, 4 cores recommended

### Recommended Requirements
- **Python**: 3.9+ 
- **RAM**: 16 GB or more
- **Storage**: 10 GB free space (for datasets and models)
- **CPU**: 8 cores or more
- **GPU**: NVIDIA GPU with CUDA 11.0+ (optional, for acceleration)

### Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, Amazon Linux 2
- **macOS**: 10.15+ (Catalina or newer)
- **Windows**: Windows 10 or Windows Server 2019+

## 🐍 Python Environment Setup

### Option 1: Using pip (Recommended)

```bash
# Create virtual environment
python -m venv ml-pipeline-env
source ml-pipeline-env/bin/activate  # On Windows: ml-pipeline-env\Scripts\activate

# Install the package
pip install ml-pipeline-framework

# Or install from source
git clone https://github.com/your-org/ml-pipeline-framework.git
cd ml-pipeline-framework
pip install -e .
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n ml-pipeline python=3.9
conda activate ml-pipeline

# Install from conda-forge
conda install -c conda-forge ml-pipeline-framework

# Or install with pip in conda environment
pip install ml-pipeline-framework
```

### Option 3: Using Poetry

```bash
# Clone repository
git clone https://github.com/your-org/ml-pipeline-framework.git
cd ml-pipeline-framework

# Install with Poetry
poetry install
poetry shell
```

## 📦 Installation Options

### Core Installation (Minimal)
For basic functionality with CSV data and core ML algorithms:

```bash
pip install ml-pipeline-framework[core]
```

Includes:
- Core data processing (Pandas)
- Basic ML algorithms (scikit-learn)
- Configuration management
- CLI interface

### Standard Installation (Recommended)
For most use cases with additional data engines and interpretability:

```bash
pip install ml-pipeline-framework[standard]
```

Includes everything in Core plus:
- Additional data engines (Polars, DuckDB)
- Advanced ML algorithms (XGBoost, LightGBM, CatBoost)
- Interpretability tools (SHAP, LIME)
- Visualization libraries

### Full Installation
For complete functionality including cloud connectors and enterprise features:

```bash
pip install ml-pipeline-framework[full]
```

Includes everything in Standard plus:
- Cloud database connectors (Snowflake, Redshift)
- Distributed computing (Dask, Ray)
- Enterprise security features
- Advanced monitoring tools

### Development Installation
For contributors and developers:

```bash
pip install ml-pipeline-framework[dev]
```

Includes everything in Full plus:
- Testing frameworks (pytest, hypothesis)
- Code quality tools (black, flake8, mypy)
- Documentation tools (sphinx, mkdocs)
- Pre-commit hooks

## 🗃️ Database Dependencies

### PostgreSQL
```bash
# Ubuntu/Debian
sudo apt-get install libpq-dev

# CentOS/RHEL
sudo yum install postgresql-devel

# macOS
brew install postgresql

# Python package
pip install psycopg2-binary
```

### MySQL
```bash
# Ubuntu/Debian
sudo apt-get install default-libmysqlclient-dev

# CentOS/RHEL
sudo yum install mysql-devel

# macOS
brew install mysql

# Python package
pip install mysqlclient
```

### SQL Server
```bash
# Install Microsoft ODBC driver
# Ubuntu/Debian
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Python package
pip install pyodbc
```

## ☁️ Cloud Provider Setup

### AWS
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Install additional dependencies
pip install boto3 s3fs
```

### Google Cloud Platform
```bash
# Install Google Cloud SDK
# Follow: https://cloud.google.com/sdk/docs/install

# Install additional dependencies
pip install google-cloud-storage google-cloud-bigquery
```

### Azure
```bash
# Install Azure CLI
# Follow: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Install additional dependencies
pip install azure-storage-blob azure-identity
```

## 🐳 Docker Installation

### Using Docker Hub
```bash
# Pull the latest image
docker pull ml-pipeline-framework:latest

# Run with sample configuration
docker run -v $(pwd)/configs:/configs -v $(pwd)/data:/data ml-pipeline-framework:latest train --config /configs/pipeline_config.yaml
```

### Building from Source
```bash
# Clone repository
git clone https://github.com/your-org/ml-pipeline-framework.git
cd ml-pipeline-framework

# Build Docker image
docker build -t ml-pipeline-framework:local .

# Run container
docker run -v $(pwd)/configs:/configs ml-pipeline-framework:local train --config /configs/pipeline_config.yaml
```

### Docker Compose
```bash
# Start full stack (framework + MLflow + database)
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs ml-pipeline
```

## ☸️ Kubernetes Installation

### Using Helm
```bash
# Add Helm repository
helm repo add ml-pipeline https://charts.ml-pipeline-framework.io
helm repo update

# Install with default values
helm install ml-pipeline ml-pipeline/ml-pipeline-framework

# Install with custom values
helm install ml-pipeline ml-pipeline/ml-pipeline-framework -f values.yaml
```

### Using kubectl
```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in your project directory:

```bash
# Data configuration
DATA_DIR=/path/to/your/data
MODEL_ARTIFACTS_DIR=/path/to/models
CACHE_DIR=/path/to/cache

# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ml_data
DB_USERNAME=ml_user
DB_PASSWORD=your_password

# MLflow configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=ml-pipeline-framework

# Cloud configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-west-2

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
```

### Initialize Configuration
```bash
# Generate sample configuration files
ml-pipeline init --config-type basic --output configs/

# Generate fraud detection configuration
ml-pipeline init --config-type fraud-detection --output configs/

# Generate enterprise configuration
ml-pipeline init --config-type enterprise --output configs/
```

## ✅ Verification

### Test Installation
```bash
# Check version
ml-pipeline version

# Validate sample configuration
ml-pipeline validate configs/pipeline_config.yaml

# Run a quick test
ml-pipeline train --config configs/pipeline_config.yaml --dry-run
```

### Run Health Checks
```bash
# Check system dependencies
python -c "
import sys
import platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')

# Check core dependencies
try:
    import pandas as pd
    print(f'Pandas: {pd.__version__}')
    
    import sklearn
    print(f'Scikit-learn: {sklearn.__version__}')
    
    import numpy as np
    print(f'NumPy: {np.__version__}')
    
    print('✅ Core dependencies OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"
```

### Performance Test
```bash
# Create sample dataset
python -c "
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 10000
n_features = 20

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1) > 0

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = y.astype(int)
df.to_csv('sample_data.csv', index=False)
print('Sample dataset created: sample_data.csv')
"

# Run training test
ml-pipeline train --data sample_data.csv --config configs/pipeline_config.yaml
```

## 🐛 Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'ml_pipeline'
```bash
# Ensure you're in the correct environment
which python
pip list | grep ml-pipeline-framework

# Reinstall if necessary
pip install --force-reinstall ml-pipeline-framework
```

#### 2. Database Connection Errors
```bash
# Check database connectivity
python -c "
import psycopg2  # or pymysql for MySQL
try:
    conn = psycopg2.connect(
        host='localhost',
        database='ml_data',
        user='ml_user',
        password='your_password'
    )
    print('✅ Database connection OK')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

#### 3. Memory Issues
```bash
# Check available memory
python -c "
import psutil
memory = psutil.virtual_memory()
print(f'Total memory: {memory.total / (1024**3):.1f} GB')
print(f'Available memory: {memory.available / (1024**3):.1f} GB')
print(f'Memory usage: {memory.percent}%')
"

# Reduce batch size in configuration
# Set smaller chunk_size in CSV reading
# Enable memory mapping for large files
```

#### 4. GPU Issues
```bash
# Check CUDA availability
python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')

try:
    import cudf
    print('✅ RAPIDS cuDF available')
except ImportError:
    print('❌ RAPIDS cuDF not available')
"
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the console output
2. **Check system resources**: Ensure adequate memory and disk space
3. **Verify dependencies**: Make sure all required packages are installed
4. **Check configuration**: Validate your configuration files
5. **Consult documentation**: Check the [troubleshooting guide](operations/troubleshooting.md)
6. **Community support**: Ask questions on [GitHub Discussions](https://github.com/ml-pipeline-framework/discussions)

## 🔄 Upgrading

### From pip
```bash
pip install --upgrade ml-pipeline-framework
```

### From conda
```bash
conda update ml-pipeline-framework
```

### Check for breaking changes
```bash
# Check current version
ml-pipeline version

# Check changelog
# See: https://github.com/ml-pipeline-framework/CHANGELOG.md
```

## 🎯 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quick_start.md)** - Build your first pipeline
2. **Explore [Examples](examples/README.md)** - Learn from real-world use cases
3. **Review [Configuration Options](configuration/README.md)** - Customize for your needs
4. **Try [Jupyter Notebooks](../notebooks/README.md)** - Interactive examples

---

**Need help?** Check our [FAQ](FAQ.md) or join the [community discussion](https://github.com/ml-pipeline-framework/discussions).