#!/bin/bash
set -e

# ML Pipeline Framework Kubernetes Deployment Script
# This script deploys the complete ML Pipeline Framework to Kubernetes

# Configuration
NAMESPACE="ml-pipeline"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
ML Pipeline Framework Kubernetes Deployment Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -n, --namespace         Kubernetes namespace (default: ml-pipeline)
    -t, --tag               Docker image tag (default: latest)
    -r, --registry          Docker registry URL
    -d, --dry-run           Perform dry run without actual deployment
    -s, --skip-build        Skip Docker image build
    -e, --environment       Environment (development|staging|production)
    --gpu                   Deploy GPU-enabled version
    --monitoring            Deploy monitoring stack
    --cleanup               Cleanup existing deployment before deploying

Examples:
    $0                                          # Deploy with defaults
    $0 -t v1.2.0 -r my-registry.com           # Deploy specific version
    $0 --dry-run                               # Dry run deployment
    $0 --cleanup                               # Cleanup and redeploy
    $0 --gpu --monitoring                      # Deploy with GPU and monitoring
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -s|--skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --gpu)
                ENABLE_GPU="true"
                shift
                ;;
            --monitoring)
                ENABLE_MONITORING="true"
                shift
                ;;
            --cleanup)
                CLEANUP="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if Docker is installed (if not skipping build)
    if [[ "$SKIP_BUILD" != "true" ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed or not in PATH"
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping Docker image build"
        return
    fi
    
    log_info "Building Docker images..."
    
    # Build CPU version
    BUILD_TYPE="cpu"
    if [[ "$ENABLE_GPU" == "true" ]]; then
        BUILD_TYPE="gpu"
    fi
    
    IMAGE_NAME="ml-pipeline-framework"
    if [[ -n "$REGISTRY" ]]; then
        IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
    fi
    
    log_info "Building ${BUILD_TYPE} image: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    docker build \
        --build-arg BUILD_TYPE=${BUILD_TYPE} \
        --build-arg PYTHON_VERSION=3.9 \
        --build-arg SPARK_VERSION=3.4.1 \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f Dockerfile .
    
    if [[ -n "$REGISTRY" ]]; then
        log_info "Pushing image to registry..."
        docker push "${IMAGE_NAME}:${IMAGE_TAG}"
    fi
    
    log_info "Docker image build completed"
}

# Apply Kubernetes manifests
apply_manifests() {
    local manifest_file="$1"
    local description="$2"
    
    log_info "Applying $description..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply -f "$manifest_file" --dry-run=client -o yaml
    else
        kubectl apply -f "$manifest_file"
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployment_name="$1"
    local timeout="${2:-300}"
    
    log_info "Waiting for deployment $deployment_name to be ready..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl wait --for=condition=available \
            --timeout="${timeout}s" \
            deployment/"$deployment_name" \
            -n "$NAMESPACE"
    fi
}

# Update image references in manifests
update_image_references() {
    log_info "Updating image references in manifests..."
    
    IMAGE_NAME="ml-pipeline-framework"
    if [[ -n "$REGISTRY" ]]; then
        IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
    fi
    
    # Update image references in YAML files
    find deploy/kubernetes -name "*.yaml" -exec sed -i.bak \
        "s|image: ml-pipeline-framework:latest|image: ${IMAGE_NAME}:${IMAGE_TAG}|g" {} \;
    
    # Clean up backup files
    find deploy/kubernetes -name "*.yaml.bak" -delete
}

# Cleanup existing deployment
cleanup_deployment() {
    if [[ "$CLEANUP" != "true" ]]; then
        return
    fi
    
    log_warn "Cleaning up existing deployment..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Delete in reverse order to avoid dependency issues
        kubectl delete -f deploy/kubernetes/jobs.yaml -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete -f deploy/kubernetes/ml-pipeline-app.yaml -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete -f deploy/kubernetes/mlflow.yaml -n "$NAMESPACE" --ignore-not-found=true
        kubectl delete -f deploy/kubernetes/postgres.yaml -n "$NAMESPACE" --ignore-not-found=true
        
        # Wait for pods to terminate
        kubectl wait --for=delete pod -l app=ml-pipeline-framework -n "$NAMESPACE" --timeout=120s || true
        kubectl wait --for=delete pod -l app=mlflow -n "$NAMESPACE" --timeout=60s || true
        kubectl wait --for=delete pod -l app=postgres -n "$NAMESPACE" --timeout=60s || true
    fi
    
    log_info "Cleanup completed"
}

# Deploy monitoring stack
deploy_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        return
    fi
    
    log_info "Deploying monitoring stack..."
    
    # Check if monitoring namespace exists
    if ! kubectl get namespace monitoring &> /dev/null; then
        log_info "Creating monitoring namespace..."
        kubectl create namespace monitoring
    fi
    
    # Deploy Prometheus and Grafana (simplified)
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: LoadBalancer
EOF
    
    log_info "Monitoring stack deployed"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Skipping validation in dry-run mode"
        return
    fi
    
    # Check if all pods are running
    local failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running -o name | wc -l)
    
    if [[ $failed_pods -gt 0 ]]; then
        log_warn "Some pods are not running:"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
    else
        log_info "All pods are running successfully"
    fi
    
    # Check services
    log_info "Services in namespace $NAMESPACE:"
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        log_info "Ingress resources:"
        kubectl get ingress -n "$NAMESPACE"
    fi
}

# Show deployment status
show_status() {
    log_info "Deployment Status Summary"
    echo "=========================="
    
    echo -e "\n${BLUE}Namespace:${NC} $NAMESPACE"
    echo -e "${BLUE}Image Tag:${NC} $IMAGE_TAG"
    echo -e "${BLUE}Environment:${NC} $ENVIRONMENT"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        echo -e "\n${BLUE}Pods:${NC}"
        kubectl get pods -n "$NAMESPACE" -o wide
        
        echo -e "\n${BLUE}Services:${NC}"
        kubectl get services -n "$NAMESPACE"
        
        echo -e "\n${BLUE}Persistent Volumes:${NC}"
        kubectl get pv | grep "$NAMESPACE" || echo "No PVs found"
        
        echo -e "\n${BLUE}Storage:${NC}"
        kubectl get pvc -n "$NAMESPACE"
        
        # Show access information
        echo -e "\n${GREEN}Access Information:${NC}"
        
        # Get LoadBalancer services
        local lb_services=$(kubectl get services -n "$NAMESPACE" -o jsonpath='{.items[?(@.spec.type=="LoadBalancer")].metadata.name}')
        if [[ -n "$lb_services" ]]; then
            for service in $lb_services; do
                local external_ip=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
                if [[ -n "$external_ip" ]]; then
                    echo "  $service: http://$external_ip"
                else
                    echo "  $service: External IP pending..."
                fi
            done
        fi
        
        # Get ingress information
        if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
            local ingress_hosts=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[*].spec.rules[*].host}')
            if [[ -n "$ingress_hosts" ]]; then
                echo "  Ingress hosts: $ingress_hosts"
            fi
        fi
    fi
    
    echo ""
}

# Main deployment function
main() {
    log_info "Starting ML Pipeline Framework deployment to Kubernetes"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    
    # Check prerequisites
    check_prerequisites
    
    # Cleanup if requested
    cleanup_deployment
    
    # Build Docker images
    build_images
    
    # Update image references
    update_image_references
    
    # Deploy Kubernetes resources
    log_info "Deploying Kubernetes resources..."
    
    # Create namespace and RBAC
    apply_manifests "deploy/kubernetes/namespace.yaml" "namespace and resource quotas"
    apply_manifests "deploy/kubernetes/secrets.yaml" "secrets and RBAC"
    
    # Deploy storage
    apply_manifests "deploy/kubernetes/persistent-volumes.yaml" "persistent volumes"
    
    # Deploy database
    apply_manifests "deploy/kubernetes/postgres.yaml" "PostgreSQL database"
    wait_for_deployment "postgres"
    
    # Deploy MLflow
    apply_manifests "deploy/kubernetes/mlflow.yaml" "MLflow tracking server"
    wait_for_deployment "mlflow"
    
    # Deploy configuration
    apply_manifests "deploy/kubernetes/configmap.yaml" "configuration maps"
    
    # Deploy main application
    apply_manifests "deploy/kubernetes/ml-pipeline-app.yaml" "ML Pipeline application"
    wait_for_deployment "ml-pipeline-app"
    
    # Deploy jobs and cron jobs
    apply_manifests "deploy/kubernetes/jobs.yaml" "jobs and cron jobs"
    
    # Deploy monitoring if enabled
    deploy_monitoring
    
    # Validate deployment
    validate_deployment
    
    # Show status
    show_status
    
    log_info "ML Pipeline Framework deployment completed successfully!"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        echo ""
        log_info "Next steps:"
        echo "1. Check pod status: kubectl get pods -n $NAMESPACE"
        echo "2. View logs: kubectl logs -f deployment/ml-pipeline-app -n $NAMESPACE"
        echo "3. Access MLflow UI via LoadBalancer or Ingress"
        echo "4. Submit training job: kubectl apply -f deploy/kubernetes/jobs.yaml"
    fi
}

# Parse arguments and run main function
parse_args "$@"
main