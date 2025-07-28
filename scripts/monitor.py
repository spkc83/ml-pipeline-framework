#!/usr/bin/env python3
"""
ML Pipeline Framework Monitoring Script.

This script monitors the ML pipeline application and sends metrics to Prometheus.
"""

import os
import sys
import time
import logging
import psutil
import requests
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipelineMonitor:
    """Monitor for ML Pipeline Framework application."""
    
    def __init__(self):
        self.monitor_interval = int(os.environ.get('MONITOR_INTERVAL', '60'))
        self.pushgateway_url = os.environ.get('PROMETHEUS_PUSHGATEWAY', 'pushgateway:9091')
        self.job_name = os.environ.get('JOB_NAME', 'ml-pipeline')
        self.pod_name = os.environ.get('POD_NAME', 'unknown')
        self.namespace = os.environ.get('KUBERNETES_NAMESPACE', 'ml-pipeline')
        
        logger.info(f"Monitor initialized:")
        logger.info(f"  Interval: {self.monitor_interval}s")
        logger.info(f"  Pushgateway: {self.pushgateway_url}")
        logger.info(f"  Job: {self.job_name}")
        logger.info(f"  Pod: {self.pod_name}")
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/app')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = {
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory_percent,
                'system_memory_used_gb': memory_used_gb,
                'system_memory_total_gb': memory_total_gb,
                'system_disk_percent': disk_percent,
                'system_disk_free_gb': disk_free_gb,
                'system_disk_total_gb': disk_total_gb,
                'system_process_count': process_count,
            }
            
            logger.debug(f"System metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            metrics = {
                'app_uptime_seconds': self.get_uptime(),
                'app_health_status': self.check_health(),
            }
            
            # Check if specific directories exist and get sizes
            directories = ['/app/logs', '/app/models', '/app/artifacts', '/app/data']
            for directory in directories:
                if Path(directory).exists():
                    size_mb = self.get_directory_size(directory) / (1024**2)
                    dir_name = directory.split('/')[-1]
                    metrics[f'app_directory_size_mb_{dir_name}'] = size_mb
            
            # Count files in directories
            if Path('/app/models').exists():
                model_count = len(list(Path('/app/models').glob('*.pkl'))) + \
                             len(list(Path('/app/models').glob('*.joblib')))
                metrics['app_model_count'] = model_count
            
            logger.debug(f"Application metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}
    
    def get_uptime(self):
        """Get application uptime in seconds."""
        try:
            # Try to get process start time
            current_process = psutil.Process()
            start_time = current_process.create_time()
            uptime = time.time() - start_time
            return uptime
        except Exception:
            return 0
    
    def check_health(self):
        """Check application health status."""
        try:
            # Run basic health check
            result = os.system('python /healthcheck.py > /dev/null 2>&1')
            return 1 if result == 0 else 0
        except Exception:
            return 0
    
    def get_directory_size(self, directory):
        """Get total size of directory in bytes."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            return total_size
        except Exception:
            return 0
    
    def format_prometheus_metrics(self, metrics):
        """Format metrics for Prometheus pushgateway."""
        prometheus_metrics = []
        timestamp = int(time.time() * 1000)
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                prometheus_metrics.append(f"{metric_name} {value} {timestamp}")
        
        return '\n'.join(prometheus_metrics)
    
    def push_metrics(self, metrics):
        """Push metrics to Prometheus pushgateway."""
        try:
            if not metrics:
                logger.warning("No metrics to push")
                return False
            
            # Format metrics for pushgateway
            metric_data = self.format_prometheus_metrics(metrics)
            
            # Prepare URL
            url = f"http://{self.pushgateway_url}/metrics/job/{self.job_name}/instance/{self.pod_name}"
            
            # Push metrics
            headers = {'Content-Type': 'text/plain'}
            response = requests.post(url, data=metric_data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Successfully pushed {len(metrics)} metrics")
                return True
            else:
                logger.error(f"Failed to push metrics: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pushing metrics: {e}")
            return False
    
    def log_metrics(self, metrics):
        """Log metrics to console for debugging."""
        logger.info("Current metrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric_name}: {value}")
    
    def run(self):
        """Main monitoring loop."""
        logger.info("Starting ML Pipeline monitoring...")
        
        while True:
            try:
                # Collect all metrics
                system_metrics = self.collect_system_metrics()
                app_metrics = self.collect_application_metrics()
                
                # Combine metrics
                all_metrics = {**system_metrics, **app_metrics}
                
                # Add metadata
                all_metrics['monitor_timestamp'] = time.time()
                all_metrics['monitor_healthy'] = 1
                
                # Log metrics
                self.log_metrics(all_metrics)
                
                # Push to Prometheus (optional)
                if self.pushgateway_url:
                    self.push_metrics(all_metrics)
                
                # Wait for next iteration
                logger.debug(f"Sleeping for {self.monitor_interval} seconds...")
                time.sleep(self.monitor_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)


def main():
    """Main entry point."""
    monitor = MLPipelineMonitor()
    monitor.run()


if __name__ == "__main__":
    main()