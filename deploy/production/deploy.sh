#!/bin/bash

# ML Pipeline Framework Production Deployment Script
# This script handles complete production deployment with monitoring, security, and high availability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
VERSION="${VERSION:-latest}"
DOMAIN="${DOMAIN:-ml-pipeline.local}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check minimum Docker version
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' | cut -d. -f1-2)
    if [[ $(echo "${DOCKER_VERSION} < 20.10" | bc -l) ]]; then
        log_warn "Docker version ${DOCKER_VERSION} detected. Recommended version is 20.10+."
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=$((10 * 1024 * 1024)) # 10GB in KB
    
    if [[ ${AVAILABLE_SPACE} -lt ${REQUIRED_SPACE} ]]; then
        log_error "Insufficient disk space. At least 10GB required."
        exit 1
    fi
    
    # Check available memory (minimum 8GB)
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ ${AVAILABLE_MEMORY} -lt 8192 ]]; then
        log_warn "Available memory is ${AVAILABLE_MEMORY}MB. Recommended minimum is 8GB."
    fi
    
    log_success "Prerequisites check completed."
}

# Load environment variables
load_environment() {
    log_info "Loading environment configuration..."
    
    # Check for environment file
    ENV_FILE="${SCRIPT_DIR}/.env.${DEPLOYMENT_ENV}"
    if [[ -f "${ENV_FILE}" ]]; then
        log_info "Loading environment from ${ENV_FILE}"
        set -a
        source "${ENV_FILE}"
        set +a
    else
        log_warn "Environment file ${ENV_FILE} not found. Using defaults."
    fi
    
    # Validate required environment variables
    REQUIRED_VARS=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "ENCRYPTION_KEY"
        "JWT_SECRET"
        "AWS_ACCESS_KEY_ID"
        "AWS_SECRET_ACCESS_KEY"
        "S3_BUCKET"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable ${var} is not set."
            exit 1
        fi
    done
    
    # Set default values for optional variables
    export DOMAIN="${DOMAIN:-ml-pipeline.local}"
    export VERSION="${VERSION:-latest}"
    export DB_NAME="${DB_NAME:-ml_pipeline}"
    export DB_USERNAME="${DB_USERNAME:-ml_user}"
    export GRAFANA_USER="${GRAFANA_USER:-admin}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
    
    log_success "Environment configuration loaded."
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    DIRECTORIES=(
        "data"
        "models"
        "artifacts"
        "logs"
        "postgres-data"
        "redis-data"
        "prometheus-data"
        "grafana-data"
        "alertmanager-data"
        "loki-data"
        "nginx-logs"
        "certs"
        "backups"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "${SCRIPT_DIR}/${dir}"
        # Set appropriate permissions
        if [[ "${dir}" == "certs" ]]; then
            chmod 700 "${SCRIPT_DIR}/${dir}"
        else
            chmod 755 "${SCRIPT_DIR}/${dir}"
        fi
    done
    
    log_success "Directory structure created."
}

# Generate SSL certificates
generate_certificates() {
    log_info "Generating SSL certificates..."
    
    CERTS_DIR="${SCRIPT_DIR}/certs"
    
    if [[ ! -f "${CERTS_DIR}/server.crt" ]]; then
        log_info "Generating self-signed SSL certificate..."
        
        # Generate private key
        openssl genrsa -out "${CERTS_DIR}/server.key" 2048
        
        # Generate certificate signing request
        openssl req -new -key "${CERTS_DIR}/server.key" -out "${CERTS_DIR}/server.csr" \
            -subj "/C=US/ST=CA/L=San Francisco/O=ML Pipeline/CN=${DOMAIN}"
        
        # Generate self-signed certificate
        openssl x509 -req -days 365 -in "${CERTS_DIR}/server.csr" \
            -signkey "${CERTS_DIR}/server.key" -out "${CERTS_DIR}/server.crt"
        
        # Set secure permissions
        chmod 600 "${CERTS_DIR}/server.key"
        chmod 644 "${CERTS_DIR}/server.crt"
        
        log_success "SSL certificates generated."
    else
        log_info "SSL certificates already exist."
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "${PROJECT_ROOT}"
    
    # Build main application image
    log_info "Building ML Pipeline application image..."
    docker build -t "ml-pipeline-framework:${VERSION}" \
        --build-arg VERSION="${VERSION}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        -f Dockerfile .
    
    # Tag as latest if this is a release build
    if [[ "${VERSION}" != "latest" && "${VERSION}" != "dev" ]]; then
        docker tag "ml-pipeline-framework:${VERSION}" "ml-pipeline-framework:latest"
    fi
    
    log_success "Docker images built successfully."
}

# Initialize database
init_database() {
    log_info "Initializing database..."
    
    # Start only PostgreSQL first
    cd "${SCRIPT_DIR}"
    docker-compose -f docker-compose.production.yml up -d postgres
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    until docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${DB_USERNAME}" -d "${DB_NAME}"; do
        sleep 2
    done
    
    # Run database initialization scripts
    if [[ -d "${SCRIPT_DIR}/init-scripts" ]]; then
        log_info "Running database initialization scripts..."
        for script in "${SCRIPT_DIR}"/init-scripts/*.sql; do
            if [[ -f "${script}" ]]; then
                log_info "Executing $(basename "${script}")..."
                docker-compose -f docker-compose.production.yml exec -T postgres \
                    psql -U "${DB_USERNAME}" -d "${DB_NAME}" < "${script}"
            fi
        done
    fi
    
    log_success "Database initialized."
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    cd "${SCRIPT_DIR}"
    
    # Start monitoring services
    docker-compose -f docker-compose.production.yml up -d \
        prometheus grafana alertmanager loki promtail node-exporter
    
    # Wait for services to be ready
    log_info "Waiting for monitoring services to be ready..."
    
    # Wait for Prometheus
    until curl -sf http://localhost:9090/-/healthy >/dev/null 2>&1; do
        sleep 2
    done
    log_info "Prometheus is ready"
    
    # Wait for Grafana
    until curl -sf http://localhost:3000/api/health >/dev/null 2>&1; do
        sleep 2
    done
    log_info "Grafana is ready"
    
    # Import Grafana dashboards
    log_info "Importing Grafana dashboards..."
    if [[ -d "${SCRIPT_DIR}/monitoring/grafana/dashboards" ]]; then
        # Dashboards will be automatically imported via provisioning
        log_info "Grafana dashboards configured for auto-import"
    fi
    
    log_success "Monitoring stack deployed."
}

# Deploy application stack
deploy_application() {
    log_info "Deploying application stack..."
    
    cd "${SCRIPT_DIR}"
    
    # Start all remaining services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for application to be ready
    log_info "Waiting for application to be ready..."
    
    local retries=30
    local count=0
    
    until curl -sf http://localhost:8090/health >/dev/null 2>&1; do
        if [[ ${count} -ge ${retries} ]]; then
            log_error "Application failed to start within expected time"
            docker-compose -f docker-compose.production.yml logs ml-pipeline-app
            exit 1
        fi
        
        sleep 10
        ((count++))
        log_info "Waiting for application... (${count}/${retries})"
    done
    
    log_success "Application is ready."
}

# Run health checks
run_health_checks() {
    log_info "Running comprehensive health checks..."
    
    # Define services and their health check endpoints
    declare -A HEALTH_CHECKS=(
        ["ml-pipeline-app"]="http://localhost:8090/health"
        ["prometheus"]="http://localhost:9090/-/healthy"
        ["grafana"]="http://localhost:3000/api/health"
        ["alertmanager"]="http://localhost:9093/-/healthy"
        ["mlflow"]="http://localhost:5000/health"
    )
    
    local failed_checks=0
    
    for service in "${!HEALTH_CHECKS[@]}"; do
        endpoint="${HEALTH_CHECKS[$service]}"
        log_info "Checking ${service}..."
        
        if curl -sf "${endpoint}" >/dev/null 2>&1; then
            log_success "${service} health check passed"
        else
            log_error "${service} health check failed"
            ((failed_checks++))
        fi
    done
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if docker-compose -f docker-compose.production.yml exec -T postgres \
        pg_isready -U "${DB_USERNAME}" -d "${DB_NAME}" >/dev/null 2>&1; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        ((failed_checks++))
    fi
    
    # Check Redis connectivity
    log_info "Checking Redis connectivity..."
    if docker-compose -f docker-compose.production.yml exec -T redis \
        redis-cli -a "${REDIS_PASSWORD}" ping >/dev/null 2>&1; then
        log_success "Redis connectivity check passed"
    else
        log_error "Redis connectivity check failed"
        ((failed_checks++))
    fi
    
    if [[ ${failed_checks} -eq 0 ]]; then
        log_success "All health checks passed!"
    else
        log_error "${failed_checks} health check(s) failed!"
        exit 1
    fi
}

# Setup backup jobs
setup_backups() {
    log_info "Setting up backup jobs..."
    
    # Create backup script
    cat > "${SCRIPT_DIR}/backup.sh" << 'EOF'
#!/bin/bash

# ML Pipeline Backup Script
set -euo pipefail

BACKUP_DIR="/opt/ml-pipeline/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker-compose exec -T postgres pg_dump -U ${DB_USERNAME} ${DB_NAME} | gzip > "${BACKUP_DIR}/db_backup_${DATE}.sql.gz"

# Models and artifacts backup
tar -czf "${BACKUP_DIR}/models_backup_${DATE}.tar.gz" -C /opt/ml-pipeline models artifacts

# Configuration backup
tar -czf "${BACKUP_DIR}/config_backup_${DATE}.tar.gz" -C /opt/ml-pipeline configs

# Clean up old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "*.gz" -mtime +7 -delete

echo "Backup completed: ${DATE}"
EOF
    
    chmod +x "${SCRIPT_DIR}/backup.sh"
    
    # Add to crontab (run daily at 2 AM)
    if ! crontab -l 2>/dev/null | grep -q "ml-pipeline.*backup"; then
        (crontab -l 2>/dev/null; echo "0 2 * * * ${SCRIPT_DIR}/backup.sh >> ${SCRIPT_DIR}/logs/backup.log 2>&1") | crontab -
        log_info "Backup cron job scheduled"
    fi
    
    log_success "Backup jobs configured."
}

# Display deployment summary
display_summary() {
    log_success "=== ML Pipeline Framework Deployment Complete ==="
    echo
    log_info "🌐 Application URLs:"
    echo "   • ML Pipeline API: https://${DOMAIN}"
    echo "   • Health Check: http://localhost:8090/health"
    echo "   • Metrics: http://localhost:8080/metrics"
    echo "   • MLflow: http://localhost:5000"
    echo "   • Grafana: http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
    echo "   • Prometheus: http://localhost:9090"
    echo "   • Alertmanager: http://localhost:9093"
    echo
    log_info "📊 Monitoring:"
    echo "   • System metrics: Node Exporter (port 9100)"
    echo "   • Application logs: Loki + Promtail"
    echo "   • Dashboards: Grafana with pre-configured dashboards"
    echo "   • Alerts: Prometheus + Alertmanager"
    echo
    log_info "🔧 Management Commands:"
    echo "   • View logs: docker-compose -f docker-compose.production.yml logs -f [service]"
    echo "   • Scale service: docker-compose -f docker-compose.production.yml up -d --scale ml-pipeline-app=3"
    echo "   • Stop all: docker-compose -f docker-compose.production.yml down"
    echo "   • Backup: ${SCRIPT_DIR}/backup.sh"
    echo
    log_info "📁 Important Directories:"
    echo "   • Data: ${SCRIPT_DIR}/data"
    echo "   • Models: ${SCRIPT_DIR}/models"
    echo "   • Logs: ${SCRIPT_DIR}/logs"
    echo "   • Backups: ${SCRIPT_DIR}/backups"
    echo
    log_success "Deployment completed successfully! 🎉"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Deployment failed with exit code ${exit_code}"
        log_info "Cleaning up..."
        cd "${SCRIPT_DIR}"
        docker-compose -f docker-compose.production.yml down --remove-orphans
    fi
    exit ${exit_code}
}

# Main deployment function
main() {
    trap cleanup EXIT
    
    log_info "Starting ML Pipeline Framework production deployment..."
    log_info "Environment: ${DEPLOYMENT_ENV}"
    log_info "Version: ${VERSION}"
    log_info "Domain: ${DOMAIN}"
    echo
    
    check_prerequisites
    load_environment
    create_directories
    generate_certificates
    build_images
    init_database
    deploy_monitoring
    deploy_application
    run_health_checks
    setup_backups
    display_summary
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "health")
        run_health_checks
        ;;
    "stop")
        log_info "Stopping ML Pipeline services..."
        cd "${SCRIPT_DIR}"
        docker-compose -f docker-compose.production.yml down
        log_success "Services stopped."
        ;;
    "restart")
        log_info "Restarting ML Pipeline services..."
        cd "${SCRIPT_DIR}"
        docker-compose -f docker-compose.production.yml restart
        log_success "Services restarted."
        ;;
    "logs")
        cd "${SCRIPT_DIR}"
        docker-compose -f docker-compose.production.yml logs -f "${2:-}"
        ;;
    "backup")
        if [[ -f "${SCRIPT_DIR}/backup.sh" ]]; then
            "${SCRIPT_DIR}/backup.sh"
        else
            log_error "Backup script not found. Run deployment first."
            exit 1
        fi
        ;;
    "update")
        log_info "Updating ML Pipeline to version ${VERSION}..."
        build_images
        cd "${SCRIPT_DIR}"
        docker-compose -f docker-compose.production.yml up -d --force-recreate ml-pipeline-app
        run_health_checks
        log_success "Update completed."
        ;;
    *)
        echo "Usage: $0 {deploy|health|stop|restart|logs [service]|backup|update}"
        echo
        echo "Commands:"
        echo "  deploy  - Full deployment (default)"
        echo "  health  - Run health checks"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - View logs (optionally specify service)"
        echo "  backup  - Run backup manually"
        echo "  update  - Update to new version"
        exit 1
        ;;
esac