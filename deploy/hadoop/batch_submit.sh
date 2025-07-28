#!/bin/bash
set -e

# Hadoop Batch Job Submission Script for ML Pipeline Framework
# This script provides a convenient way to submit various ML jobs to Hadoop cluster

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/job_config.yaml}"
HADOOP_USER="${HADOOP_USER:-ml-pipeline}"
YARN_QUEUE="${YARN_QUEUE:-default}"

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
Hadoop Batch Job Submission Script for ML Pipeline Framework

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    submit      Submit a job to Hadoop cluster
    status      Check status of a job
    kill        Kill a running job
    list        List running ML pipeline jobs
    logs        Get logs for a job
    monitor     Monitor job progress
    
Submit Options:
    -t, --type              Job type (training|prediction|data_processing|evaluation)
    -c, --config            Configuration file path (default: job_config.yaml)
    -q, --queue             YARN queue name (default: default)
    -n, --name              Job name
    -u, --user              Hadoop user (default: ml-pipeline)
    --app-file              Custom application file path
    --num-executors         Number of executors
    --executor-cores        Cores per executor
    --executor-memory       Memory per executor
    --driver-memory         Driver memory
    --dry-run               Show command without executing

Status/Kill/Logs Options:
    -a, --app-id            Application ID

Examples:
    $0 submit -t training -n "churn-model-v1"
    $0 submit -t prediction --dry-run
    $0 status -a application_1234567890_0001
    $0 kill -a application_1234567890_0001
    $0 logs -a application_1234567890_0001
    $0 list
    $0 monitor -a application_1234567890_0001
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running as correct user
    if [[ "$USER" != "$HADOOP_USER" && "$EUID" -ne 0 ]]; then
        log_warn "Running as user '$USER', recommended user is '$HADOOP_USER'"
    fi
    
    # Check Hadoop environment
    if [[ -z "$HADOOP_HOME" ]]; then
        log_error "HADOOP_HOME is not set"
        exit 1
    fi
    
    if [[ -z "$SPARK_HOME" ]]; then
        log_error "SPARK_HOME is not set"
        exit 1
    fi
    
    # Check if yarn command is available
    if ! command -v yarn &> /dev/null; then
        log_error "yarn command not found"
        exit 1
    fi
    
    # Check if spark-submit is available
    if ! command -v spark-submit &> /dev/null; then
        log_error "spark-submit command not found"
        exit 1
    fi
    
    # Check HDFS connectivity
    if ! hdfs dfs -ls / &> /dev/null; then
        log_error "Cannot connect to HDFS"
        exit 1
    fi
    
    # Check YARN connectivity
    if ! yarn application -list &> /dev/null; then
        log_error "Cannot connect to YARN ResourceManager"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Submit job function
submit_job() {
    local job_type="$1"
    local job_name="$2"
    local config_file="$3"
    local queue="$4"
    local dry_run="$5"
    
    log_info "Submitting $job_type job: $job_name"
    
    # Validate job type
    case $job_type in
        training|prediction|data_processing|evaluation)
            ;;
        *)
            log_error "Invalid job type: $job_type"
            exit 1
            ;;
    esac
    
    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    # Prepare HDFS directories
    prepare_hdfs_directories
    
    # Upload necessary files to HDFS
    upload_files_to_hdfs
    
    # Build submission command
    local cmd="python3 $SCRIPT_DIR/submit_job.py $job_type"
    cmd+=" --config $config_file"
    cmd+=" --queue $queue"
    
    # Add optional parameters
    if [[ -n "$JOB_NAME" ]]; then
        cmd+=" --name $JOB_NAME"
    fi
    
    if [[ -n "$NUM_EXECUTORS" ]]; then
        cmd+=" --num-executors $NUM_EXECUTORS"
    fi
    
    if [[ -n "$EXECUTOR_CORES" ]]; then
        cmd+=" --executor-cores $EXECUTOR_CORES"
    fi
    
    if [[ -n "$EXECUTOR_MEMORY" ]]; then
        cmd+=" --executor-memory $EXECUTOR_MEMORY"
    fi
    
    if [[ -n "$DRIVER_MEMORY" ]]; then
        cmd+=" --driver-memory $DRIVER_MEMORY"
    fi
    
    if [[ -n "$APP_FILE" ]]; then
        cmd+=" --app-file $APP_FILE"
    fi
    
    log_debug "Submission command: $cmd"
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run mode - command would be:"
        echo "$cmd"
        return
    fi
    
    # Execute submission
    log_info "Executing job submission..."
    
    if eval "$cmd"; then
        log_info "Job submitted successfully"
        
        # Extract application ID and monitor
        local app_id=$(eval "$cmd" 2>&1 | grep -o 'application_[0-9]*_[0-9]*' | head -1)
        if [[ -n "$app_id" ]]; then
            log_info "Application ID: $app_id"
            log_info "Monitor progress with: $0 monitor -a $app_id"
            
            # Save application ID for easy reference
            echo "$app_id" > "/tmp/ml-pipeline-last-job.txt"
        fi
    else
        log_error "Job submission failed"
        exit 1
    fi
}

# Prepare HDFS directories
prepare_hdfs_directories() {
    log_info "Preparing HDFS directories..."
    
    local directories=(
        "/ml-pipeline"
        "/ml-pipeline/data"
        "/ml-pipeline/data/training"
        "/ml-pipeline/data/validation"
        "/ml-pipeline/data/input"
        "/ml-pipeline/data/output"
        "/ml-pipeline/models"
        "/ml-pipeline/artifacts"
        "/ml-pipeline/logs"
        "/ml-pipeline/checkpoints"
        "/ml-pipeline/metrics"
        "/spark-logs"
        "/spark-warehouse"
    )
    
    for dir in "${directories[@]}"; do
        if ! hdfs dfs -test -d "$dir" 2>/dev/null; then
            log_debug "Creating HDFS directory: $dir"
            hdfs dfs -mkdir -p "$dir"
            hdfs dfs -chmod 755 "$dir"
        fi
    done
    
    log_info "HDFS directories prepared"
}

# Upload files to HDFS
upload_files_to_hdfs() {
    log_info "Uploading files to HDFS..."
    
    # Upload configuration files
    local config_hdfs_path="/ml-pipeline/configs"
    if ! hdfs dfs -test -d "$config_hdfs_path" 2>/dev/null; then
        hdfs dfs -mkdir -p "$config_hdfs_path"
    fi
    
    # Upload current config
    hdfs dfs -put -f "$CONFIG_FILE" "$config_hdfs_path/"
    
    # Upload pipeline configuration if it exists
    if [[ -f "$PROJECT_ROOT/configs/pipeline_config.yaml" ]]; then
        hdfs dfs -put -f "$PROJECT_ROOT/configs/pipeline_config.yaml" "$config_hdfs_path/"
    fi
    
    # Upload Python source code
    local src_hdfs_path="/ml-pipeline/src"
    if ! hdfs dfs -test -d "$src_hdfs_path" 2>/dev/null; then
        hdfs dfs -mkdir -p "$src_hdfs_path"
    fi
    
    # Create and upload source code archive
    local temp_dir="/tmp/ml-pipeline-src-$$"
    mkdir -p "$temp_dir"
    
    # Copy source files
    if [[ -d "$PROJECT_ROOT/src" ]]; then
        cp -r "$PROJECT_ROOT/src" "$temp_dir/"
    fi
    
    if [[ -d "$PROJECT_ROOT/ml_pipeline_framework" ]]; then
        cp -r "$PROJECT_ROOT/ml_pipeline_framework" "$temp_dir/"
    fi
    
    # Copy Hadoop job scripts
    if [[ -d "$SCRIPT_DIR/scripts" ]]; then
        cp -r "$SCRIPT_DIR/scripts" "$temp_dir/"
    fi
    
    # Create archive
    cd "$temp_dir"
    tar -czf ml-pipeline-src.tar.gz .
    
    # Upload to HDFS
    hdfs dfs -put -f ml-pipeline-src.tar.gz "$src_hdfs_path/"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_info "Files uploaded to HDFS"
}

# Check job status
check_status() {
    local app_id="$1"
    
    if [[ -z "$app_id" ]]; then
        log_error "Application ID is required"
        exit 1
    fi
    
    log_info "Checking status for application: $app_id"
    
    python3 "$SCRIPT_DIR/submit_job.py" --check-status "$app_id"
}

# Kill job
kill_job() {
    local app_id="$1"
    
    if [[ -z "$app_id" ]]; then
        log_error "Application ID is required"
        exit 1
    fi
    
    log_info "Killing application: $app_id"
    
    if yarn application -kill "$app_id"; then
        log_info "Application killed successfully"
    else
        log_error "Failed to kill application"
        exit 1
    fi
}

# Get job logs
get_logs() {
    local app_id="$1"
    local log_type="${2:-all}"
    
    if [[ -z "$app_id" ]]; then
        log_error "Application ID is required"
        exit 1
    fi
    
    log_info "Getting logs for application: $app_id"
    
    case $log_type in
        all)
            yarn logs -applicationId "$app_id"
            ;;
        driver)
            yarn logs -applicationId "$app_id" -containerId container_$(echo $app_id | cut -d_ -f2-)_$(echo $app_id | cut -d_ -f3-)_01_000001
            ;;
        executors)
            yarn logs -applicationId "$app_id" | grep -A 50 -B 10 "Container: container.*_000002"
            ;;
        *)
            log_error "Invalid log type: $log_type. Use 'all', 'driver', or 'executors'"
            exit 1
            ;;
    esac
}

# List running jobs
list_jobs() {
    log_info "Listing running ML pipeline jobs..."
    
    python3 "$SCRIPT_DIR/submit_job.py" --list-jobs
}

# Monitor job progress
monitor_job() {
    local app_id="$1"
    local interval="${2:-30}"
    
    if [[ -z "$app_id" ]]; then
        log_error "Application ID is required"
        exit 1
    fi
    
    log_info "Monitoring application: $app_id (update interval: ${interval}s)"
    log_info "Press Ctrl+C to stop monitoring"
    
    while true; do
        clear
        echo "=== ML Pipeline Job Monitor ==="
        echo "Application ID: $app_id"
        echo "Last Update: $(date)"
        echo ""
        
        # Get current status
        local status_output=$(python3 "$SCRIPT_DIR/submit_job.py" --check-status "$app_id" 2>/dev/null)
        
        if [[ $? -eq 0 ]]; then
            echo "$status_output"
            
            # Check if job is finished
            if echo "$status_output" | grep -q '"state": "FINISHED"'; then
                log_info "Job completed successfully!"
                break
            elif echo "$status_output" | grep -q '"state": "FAILED"'; then
                log_error "Job failed!"
                break
            elif echo "$status_output" | grep -q '"state": "KILLED"'; then
                log_warn "Job was killed!"
                break
            fi
        else
            log_error "Failed to get job status"
            break
        fi
        
        echo ""
        echo "Monitoring... (Ctrl+C to stop)"
        sleep "$interval"
    done
}

# Parse command line arguments
parse_args() {
    COMMAND=""
    JOB_TYPE=""
    JOB_NAME=""
    APP_ID=""
    DRY_RUN="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            submit|status|kill|logs|list|monitor)
                COMMAND="$1"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                JOB_TYPE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -q|--queue)
                YARN_QUEUE="$2"
                shift 2
                ;;
            -n|--name)
                JOB_NAME="$2"
                shift 2
                ;;
            -u|--user)
                HADOOP_USER="$2"
                shift 2
                ;;
            -a|--app-id)
                APP_ID="$2"
                shift 2
                ;;
            --app-file)
                APP_FILE="$2"
                shift 2
                ;;
            --num-executors)
                NUM_EXECUTORS="$2"
                shift 2
                ;;
            --executor-cores)
                EXECUTOR_CORES="$2"
                shift 2
                ;;
            --executor-memory)
                EXECUTOR_MEMORY="$2"
                shift 2
                ;;
            --driver-memory)
                DRIVER_MEMORY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
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

# Main function
main() {
    log_info "ML Pipeline Framework - Hadoop Batch Job Submission"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate command
    if [[ -z "$COMMAND" ]]; then
        log_error "Command is required"
        show_help
        exit 1
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Execute command
    case $COMMAND in
        submit)
            if [[ -z "$JOB_TYPE" ]]; then
                log_error "Job type is required for submit command"
                exit 1
            fi
            
            # Generate job name if not provided
            if [[ -z "$JOB_NAME" ]]; then
                JOB_NAME="ml-pipeline-${JOB_TYPE}-$(date +%Y%m%d-%H%M%S)"
            fi
            
            submit_job "$JOB_TYPE" "$JOB_NAME" "$CONFIG_FILE" "$YARN_QUEUE" "$DRY_RUN"
            ;;
        status)
            check_status "$APP_ID"
            ;;
        kill)
            kill_job "$APP_ID"
            ;;
        logs)
            get_logs "$APP_ID"
            ;;
        list)
            list_jobs
            ;;
        monitor)
            monitor_job "$APP_ID"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"