# Deployment Guide

Comprehensive guide for deploying the ML Pipeline Framework v2.0 in various environments, from local development to enterprise production deployments.

## 📋 Table of Contents

- [Overview](#overview)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Production](#kubernetes-production)
- [Cloud Platforms](#cloud-platforms)
- [Monitoring & Observability](#monitoring--observability)
- [Security Configuration](#security-configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The ML Pipeline Framework v2.0 supports multiple deployment strategies:

- **Local Development**: Quick setup for development and testing
- **Docker**: Containerized deployment for consistency and portability
- **Kubernetes**: Production-ready with auto-scaling and high availability
- **Cloud Platforms**: Native integration with AWS, GCP, and Azure
- **Hybrid**: Mix-and-match components based on requirements

### Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Development     │───▶│ Staging/Testing │───▶│ Production      │
│                 │    │                 │    │                 │
│ • Local Python  │    │ • Docker        │    │ • Kubernetes    │
│ • SQLite        │    │ • PostgreSQL    │    │ • PostgreSQL HA │
│ • File storage  │    │ • Redis         │    │ • Redis Cluster │
└─────────────────┘    │ • Basic Monitor │    │ • Full Monitor  │
                       └─────────────────┘    │ • Auto-scaling  │
                                              └─────────────────┘
```

## 💻 Local Development

### Quick Setup

```bash
# Clone repository
git clone https://github.com/your-org/ml-pipeline-framework.git
cd ml-pipeline-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt

# Initialize configuration
cp configs/pipeline_config.example.yaml configs/pipeline_config.yaml

# Run tests
pytest tests/

# Start development server
python src/cli.py --help
```

### Development Configuration

```yaml
# configs/dev_config.yaml
pipeline:
  name: "ml-pipeline-dev"
  version: "2.0.0"
  environment: "development"

data_source:
  type: csv
  csv_options:
    file_paths: ["data/sample_fraud_data.csv"]
    chunk_size: 10000

data_processing:
  engine: "pandas"  # Use pandas for small datasets
  memory_limit: "4GB"

model_training:
  automl_enabled: true
  automl:
    algorithms: ["logistic_regression", "random_forest"]  # Quick algorithms
    time_budget: 300  # 5 minutes for development

explainability:
  enabled: true
  methods:
    global: ["shap"]
    local: ["lime"]

monitoring:
  enabled: false  # Disable monitoring for development

output:
  model_artifacts_path: "./artifacts/dev/"
  logs_path: "./logs/dev/"
```

### Development Commands

```bash
# Run training pipeline
python src/cli.py train --config configs/dev_config.yaml

# Generate sample data
python scripts/generate_sample_data.py --output data/sample_fraud_data.csv

# Run with debug logging
python src/cli.py train --config configs/dev_config.yaml --log-level DEBUG

# Validate configuration
python src/utils/config_validator.py --config configs/dev_config.yaml
```

## 🐳 Docker Deployment

### Build Images

```bash
# Build main application image
docker build -t ml-pipeline-framework:2.0.0 .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.9 -t ml-pipeline-framework:2.0.0-py39 .

# Build GPU-enabled version
docker build -f Dockerfile.gpu -t ml-pipeline-framework:2.0.0-gpu .
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-pipeline-app:
    image: ml-pipeline-framework:2.0.0
    container_name: ml-pipeline-app
    environment:
      - ENVIRONMENT=staging
      - LOG_LEVEL=INFO
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./configs:/app/configs
    ports:
      - "8000:8000"
      - "8080:8080"  # Metrics
    depends_on:
      - postgres
      - redis
    networks:
      - ml-pipeline-network

  postgres:
    image: postgres:13
    container_name: ml-pipeline-postgres
    environment:
      - POSTGRES_DB=ml_pipeline
      - POSTGRES_USER=ml_user
      - POSTGRES_PASSWORD=ml_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - ml-pipeline-network

  redis:
    image: redis:6-alpine
    container_name: ml-pipeline-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ml-pipeline-network

  prometheus:
    image: prom/prometheus:latest
    container_name: ml-pipeline-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - ml-pipeline-network

  grafana:
    image: grafana/grafana:latest
    container_name: ml-pipeline-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - ml-pipeline-network

volumes:
  postgres_data:
  redis_data:
  grafana_data:

networks:
  ml-pipeline-network:
    driver: bridge
```

### Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ml-pipeline-app

# Scale application
docker-compose up -d --scale ml-pipeline-app=3

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Execute commands in container
docker-compose exec ml-pipeline-app python src/cli.py --help
```

## ☸️ Kubernetes Production

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace ml-pipeline-production

# Apply secrets (edit with actual values first)
kubectl apply -f deploy/kubernetes/secrets/

# Deploy PostgreSQL
helm install postgresql bitnami/postgresql \
  --namespace ml-pipeline-production \
  --set auth.database=ml_pipeline \
  --set auth.username=ml_user \
  --set auth.password=secure_password \
  --set primary.persistence.size=100Gi

# Deploy Redis
helm install redis bitnami/redis \
  --namespace ml-pipeline-production \
  --set auth.password=redis_password \
  --set replica.replicaCount=2

# Deploy application
kubectl apply -f deploy/kubernetes/production/

# Verify deployment
kubectl get pods -n ml-pipeline-production
kubectl get services -n ml-pipeline-production
```

### Production Configuration

The Kubernetes deployment includes:

- **High Availability**: 3+ replicas with anti-affinity rules
- **Auto-scaling**: HPA based on CPU, memory, and custom metrics
- **Load Balancing**: Service mesh integration
- **Persistent Storage**: PVCs for models, data, and artifacts
- **Security**: RBAC, network policies, security contexts
- **Monitoring**: Prometheus, Grafana, and alerting

### Kubernetes Monitoring

```bash
# Install monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Deploy custom ServiceMonitor
kubectl apply -f deploy/kubernetes/monitoring/

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

## ☁️ Cloud Platforms

### AWS Deployment

#### EKS Setup

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name ml-pipeline-prod \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name ml-nodes \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Deploy to EKS
kubectl apply -f deploy/aws/eks/
```

#### AWS Services Integration

```yaml
# deploy/aws/aws-config.yaml
cloud_integration:
  aws:
    region: "us-west-2"
    s3:
      bucket: "ml-pipeline-artifacts"
      model_storage: "models/"
      data_storage: "data/"
    rds:
      instance_class: "db.r5.large"
      multi_az: true
      backup_retention: 7
    elasticache:
      node_type: "cache.r5.large"
      num_cache_nodes: 2
    sagemaker:
      endpoint_config: "ml-pipeline-endpoint"
      instance_type: "ml.m5.large"
```

### GCP Deployment

#### GKE Setup

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Create GKE cluster
gcloud container clusters create ml-pipeline-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10

# Deploy to GKE
kubectl apply -f deploy/gcp/gke/
```

### Azure Deployment

#### AKS Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Create resource group
az group create --name ml-pipeline-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group ml-pipeline-rg \
  --name ml-pipeline-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group ml-pipeline-rg --name ml-pipeline-aks

# Deploy to AKS
kubectl apply -f deploy/azure/aks/
```

## 📊 Monitoring & Observability

### Metrics Collection

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-pipeline'
    static_configs:
      - targets: ['ml-pipeline-service:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'ml-pipeline-business'
    static_configs:
      - targets: ['ml-pipeline-service:8080']
    metrics_path: '/business-metrics'
    scrape_interval: 60s

  - job_name: 'ml-pipeline-drift'
    static_configs:
      - targets: ['ml-pipeline-service:8080']
    metrics_path: '/drift-metrics'
    scrape_interval: 300s
```

### Grafana Dashboards

Key dashboards included:

1. **Application Overview**: Request rate, response time, error rate
2. **ML Model Performance**: Accuracy, precision, recall, F1 score
3. **Business Metrics**: ROI, cost savings, fraud detection rate
4. **Infrastructure**: CPU, memory, disk usage, network
5. **Data Quality**: Missing values, schema changes, data drift

### Alerting Rules

```yaml
# monitoring/alert-rules.yml
groups:
- name: ml-pipeline-alerts
  rules:
  - alert: ModelPerformanceDegradation
    expr: ml_model_accuracy < 0.85
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy below threshold"
      description: "Model accuracy is {{ $value }}, below 85% threshold"

  - alert: DataDriftDetected
    expr: ml_data_drift_score > 0.5
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Data drift detected"
      description: "Data drift score is {{ $value }}, indicating significant distribution changes"

  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90%"
```

## 🔒 Security Configuration

### Production Security Checklist

- [ ] **Secrets Management**: Use Kubernetes secrets or cloud secret managers
- [ ] **Network Security**: Implement network policies and service mesh
- [ ] **RBAC**: Configure role-based access control
- [ ] **Image Security**: Scan images for vulnerabilities
- [ ] **Data Encryption**: Enable encryption at rest and in transit
- [ ] **Audit Logging**: Enable comprehensive audit logs
- [ ] **Security Updates**: Regular security patches and updates

### Secrets Management

```bash
# Create secrets
kubectl create secret generic ml-pipeline-secrets \
  --from-literal=db-password=secure_db_password \
  --from-literal=redis-password=secure_redis_password \
  --from-literal=jwt-secret=jwt_secret_key \
  --namespace ml-pipeline-production

# Use external secret manager (AWS Secrets Manager example)
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: ml-pipeline-production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        secretRef:
          accessKeyID:
            name: aws-credentials
            key: access-key-id
          secretAccessKey:
            name: aws-credentials
            key: secret-access-key
EOF
```

### Network Security

```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-pipeline-network-policy
  namespace: ml-pipeline-production
spec:
  podSelector:
    matchLabels:
      app: ml-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: ml-pipeline-production
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

## 🎯 Best Practices

### Performance Optimization

1. **Resource Limits**: Set appropriate CPU and memory limits
2. **Auto-scaling**: Configure HPA and VPA for optimal resource usage
3. **Caching**: Use Redis for caching frequent requests
4. **Connection Pooling**: Configure database connection pooling
5. **Load Balancing**: Distribute traffic across multiple replicas

### High Availability

1. **Multi-AZ Deployment**: Deploy across multiple availability zones
2. **Backup Strategy**: Regular backups of data and models
3. **Health Checks**: Implement comprehensive health checks
4. **Circuit Breakers**: Use circuit breakers for external dependencies
5. **Graceful Degradation**: Handle failures gracefully

### Cost Optimization

1. **Right-sizing**: Use appropriate instance sizes
2. **Spot Instances**: Use spot instances for non-critical workloads
3. **Auto-shutdown**: Shutdown development environments when not in use
4. **Resource Monitoring**: Monitor and optimize resource usage
5. **Reserved Instances**: Use reserved instances for predictable workloads

## 🐛 Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check pod status
kubectl get pods -n ml-pipeline-production

# Check pod logs
kubectl logs -f deployment/ml-pipeline-app -n ml-pipeline-production

# Describe pod for events
kubectl describe pod <pod-name> -n ml-pipeline-production

# Check configuration
kubectl get configmap ml-pipeline-config -o yaml -n ml-pipeline-production
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:13 --restart=Never -- psql -h postgres-service -U ml_user -d ml_pipeline

# Check database service
kubectl get svc postgres-service -n ml-pipeline-production

# Check database logs
kubectl logs -f deployment/postgresql -n ml-pipeline-production
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n ml-pipeline-production

# Check HPA status
kubectl get hpa -n ml-pipeline-production

# Check node resources
kubectl describe nodes

# Check metrics
curl http://ml-pipeline-service:8080/metrics
```

### Debugging Commands

```bash
# Execute shell in pod
kubectl exec -it deployment/ml-pipeline-app -n ml-pipeline-production -- /bin/bash

# Port forward for local access
kubectl port-forward svc/ml-pipeline-service 8000:8000 -n ml-pipeline-production

# View recent events
kubectl get events --sort-by=.metadata.creationTimestamp -n ml-pipeline-production

# Check resource quotas
kubectl describe resourcequota -n ml-pipeline-production
```

### Log Analysis

```bash
# Stream logs from all pods
kubectl logs -f -l app=ml-pipeline -n ml-pipeline-production --all-containers=true

# Get logs from specific time range
kubectl logs --since=1h deployment/ml-pipeline-app -n ml-pipeline-production

# Export logs for analysis
kubectl logs deployment/ml-pipeline-app -n ml-pipeline-production > ml-pipeline.log
```

## 📞 Support

For deployment issues:

1. **Check Documentation**: Review deployment-specific documentation
2. **GitHub Issues**: Report deployment-specific issues
3. **Community Forums**: Get help from the community
4. **Enterprise Support**: Contact professional support for production issues

---

**Deployment Success!** 🚀 Your ML Pipeline Framework v2.0 is now ready for production workloads with enterprise-grade reliability, security, and monitoring.