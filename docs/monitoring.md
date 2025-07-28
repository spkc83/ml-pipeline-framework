# Monitoring Guide

Comprehensive monitoring and observability guide for the ML Pipeline Framework v2.0, covering model performance monitoring, data drift detection, business metrics tracking, and operational health monitoring.

## 📋 Table of Contents

- [Overview](#overview)
- [Monitoring Architecture](#monitoring-architecture)
- [Model Performance Monitoring](#model-performance-monitoring)
- [Data Drift Detection](#data-drift-detection)
- [Business Metrics Tracking](#business-metrics-tracking)
- [Infrastructure Monitoring](#infrastructure-monitoring)
- [Alerting & Notifications](#alerting--notifications)
- [Dashboard Configuration](#dashboard-configuration)
- [A/B Testing & Experimentation](#ab-testing--experimentation)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The ML Pipeline Framework v2.0 provides comprehensive monitoring capabilities designed for production ML systems:

- **Model Performance**: Track accuracy, precision, recall, and business metrics
- **Data Quality**: Monitor data drift, schema changes, and quality degradation
- **System Health**: Infrastructure metrics, resource usage, and application health
- **Business Impact**: ROI tracking, cost savings, and fraud detection effectiveness
- **Real-time Alerting**: Proactive notifications for performance degradation

### Monitoring Stack

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│ Metrics Layer   │───▶│ Visualization   │
│                 │    │                 │    │                 │
│ • Application   │    │ • Prometheus    │    │ • Grafana       │
│ • Infrastructure│    │ • Custom Metrics│    │ • Dashboards    │
│ • Business      │    │ • Time Series   │    │ • Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🏗️ Monitoring Architecture

### Components Overview

1. **Metric Collectors**: Application and infrastructure metric collection
2. **Time Series Database**: Prometheus for metrics storage
3. **Visualization**: Grafana dashboards and reports
4. **Alerting**: Alert Manager for notifications
5. **Log Aggregation**: ELK stack for log analysis
6. **Custom Metrics**: Domain-specific ML metrics

### Configuration

```yaml
# configs/monitoring_config.yaml
monitoring:
  enabled: true
  
  # Prometheus configuration
  prometheus:
    port: 8080
    metrics_path: "/metrics"
    scrape_interval: 30
    
  # Custom metrics
  custom_metrics:
    model_performance:
      enabled: true
      collection_interval: 300  # 5 minutes
    data_drift:
      enabled: true
      collection_interval: 3600  # 1 hour
    business_metrics:
      enabled: true
      collection_interval: 900  # 15 minutes
      
  # Alerting
  alerts:
    enabled: true
    channels: ["slack", "email", "pagerduty"]
    
  # Dashboard auto-deployment
  dashboards:
    auto_deploy: true
    grafana_url: "http://grafana:3000"
```

## 📊 Model Performance Monitoring

### Key Performance Metrics

The framework automatically tracks essential ML model metrics:

#### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the Precision-Recall curve

#### Fraud Detection Specific Metrics
- **Precision@1%**: Precision at 1% false positive rate
- **Precision@5%**: Precision at 5% false positive rate
- **Expected Value**: Business value of predictions
- **Lift**: Improvement over random selection
- **Fraud Detection Rate**: Percentage of fraud caught

### Performance Monitoring Setup

```python
# src/monitoring/performance_monitor.py
from prometheus_client import Histogram, Counter, Gauge
import numpy as np

class ModelPerformanceMonitor:
    def __init__(self):
        # Define metrics
        self.accuracy_gauge = Gauge('ml_model_accuracy', 'Model accuracy')
        self.precision_gauge = Gauge('ml_model_precision', 'Model precision')
        self.recall_gauge = Gauge('ml_model_recall', 'Model recall')
        self.f1_gauge = Gauge('ml_model_f1_score', 'Model F1 score')
        self.auc_gauge = Gauge('ml_model_auc_roc', 'Model AUC-ROC')
        
        # Fraud-specific metrics
        self.precision_at_1_gauge = Gauge('ml_fraud_precision_at_1_percent', 'Precision at 1% FPR')
        self.expected_value_gauge = Gauge('ml_fraud_expected_value', 'Expected business value')
        
        # Prediction counters
        self.predictions_total = Counter('ml_predictions_total', 'Total predictions', ['outcome'])
        self.prediction_latency = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
    
    def update_metrics(self, y_true, y_pred, y_proba):
        """Update all performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        
        # Update Prometheus metrics
        self.accuracy_gauge.set(accuracy)
        self.precision_gauge.set(precision)
        self.recall_gauge.set(recall)
        self.f1_gauge.set(f1)
        self.auc_gauge.set(auc)
        
        # Fraud-specific calculations
        precision_at_1 = self.calculate_precision_at_k(y_true, y_proba, k=0.01)
        expected_value = self.calculate_expected_value(y_true, y_pred)
        
        self.precision_at_1_gauge.set(precision_at_1)
        self.expected_value_gauge.set(expected_value)
        
        # Update counters
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == 1 and pred_label == 1:
                self.predictions_total.labels(outcome='true_positive').inc()
            elif true_label == 0 and pred_label == 0:
                self.predictions_total.labels(outcome='true_negative').inc()
            elif true_label == 1 and pred_label == 0:
                self.predictions_total.labels(outcome='false_negative').inc()
            else:
                self.predictions_total.labels(outcome='false_positive').inc()
```

### Performance Degradation Detection

```python
class PerformanceDegradationDetector:
    def __init__(self, baseline_metrics, threshold=0.05):
        self.baseline_metrics = baseline_metrics
        self.threshold = threshold
        
    def check_degradation(self, current_metrics):
        """Check if model performance has degraded significantly."""
        alerts = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics.get(metric, 0)
            degradation = (baseline_value - current_value) / baseline_value
            
            if degradation > self.threshold:
                alerts.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation': degradation,
                    'severity': 'high' if degradation > 0.1 else 'medium'
                })
        
        return alerts
```

## 📈 Data Drift Detection

### Drift Detection Methods

The framework implements multiple drift detection algorithms:

1. **Population Stability Index (PSI)**: Detects distribution changes
2. **Kolmogorov-Smirnov Test**: Statistical distribution comparison
3. **Jensen-Shannon Divergence**: Measure of similarity between distributions
4. **Chi-Square Test**: For categorical features
5. **Wasserstein Distance**: Earth mover's distance

### Drift Monitoring Configuration

```yaml
# Data drift detection settings
data_drift:
  enabled: true
  
  # Detection methods
  methods:
    - name: "psi"
      threshold: 0.1
      enabled: true
    - name: "ks_test"
      threshold: 0.05
      enabled: true
    - name: "js_divergence"
      threshold: 0.1
      enabled: true
      
  # Features to monitor
  features:
    - "transaction_amount"
    - "merchant_category"
    - "transaction_hour"
    - "days_since_last_transaction"
    
  # Monitoring schedule
  schedule:
    frequency: "hourly"
    window_size: "24h"
    reference_window: "7d"
    
  # Alerting thresholds
  alerts:
    warning_threshold: 0.1
    critical_threshold: 0.2
```

### Drift Detection Implementation

```python
# src/monitoring/drift_detector.py
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

class DataDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        
    def calculate_psi(self, current_data, feature, bins=10):
        """Calculate Population Stability Index."""
        # Create bins based on reference data
        _, bin_edges = np.histogram(self.reference_data[feature], bins=bins)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(self.reference_data[feature], bins=bin_edges, density=True)
        curr_dist, _ = np.histogram(current_data[feature], bins=bin_edges, density=True)
        
        # Avoid division by zero
        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        curr_dist = np.where(curr_dist == 0, 0.0001, curr_dist)
        
        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        return psi
    
    def ks_test(self, current_data, feature):
        """Perform Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(
            self.reference_data[feature], 
            current_data[feature]
        )
        return statistic, p_value
    
    def js_divergence(self, current_data, feature, bins=50):
        """Calculate Jensen-Shannon divergence."""
        # Create bins
        min_val = min(self.reference_data[feature].min(), current_data[feature].min())
        max_val = max(self.reference_data[feature].max(), current_data[feature].max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate probability distributions
        ref_hist, _ = np.histogram(self.reference_data[feature], bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(current_data[feature], bins=bin_edges, density=True)
        
        # Normalize to probability distributions
        ref_prob = ref_hist / ref_hist.sum()
        curr_prob = curr_hist / curr_hist.sum()
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, curr_prob, base=2)
        return js_div
    
    def detect_drift(self, current_data, methods=['psi', 'ks_test', 'js_divergence']):
        """Comprehensive drift detection."""
        drift_results = {}
        
        for feature in current_data.columns:
            feature_results = {}
            
            if 'psi' in methods:
                psi_score = self.calculate_psi(current_data, feature)
                feature_results['psi'] = {
                    'score': psi_score,
                    'drift_detected': psi_score > 0.1,
                    'severity': 'high' if psi_score > 0.2 else 'medium' if psi_score > 0.1 else 'low'
                }
            
            if 'ks_test' in methods:
                ks_stat, p_value = self.ks_test(current_data, feature)
                feature_results['ks_test'] = {
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05,
                    'severity': 'high' if p_value < 0.01 else 'medium'
                }
            
            if 'js_divergence' in methods:
                js_score = self.js_divergence(current_data, feature)
                feature_results['js_divergence'] = {
                    'score': js_score,
                    'drift_detected': js_score > 0.1,
                    'severity': 'high' if js_score > 0.2 else 'medium' if js_score > 0.1 else 'low'
                }
            
            drift_results[feature] = feature_results
        
        return drift_results
```

## 💰 Business Metrics Tracking

### Key Business Metrics

For fraud detection and financial applications:

- **ROI (Return on Investment)**: Financial return from the ML system
- **Cost Savings**: Amount saved by preventing fraud
- **False Positive Cost**: Cost of incorrectly flagging legitimate transactions
- **False Negative Cost**: Cost of missing fraudulent transactions
- **Processing Volume**: Number of transactions processed
- **Alert Volume**: Number of alerts generated

### Business Metrics Configuration

```yaml
business_metrics:
  enabled: true
  
  # Cost parameters
  costs:
    false_positive_cost: 50.0     # Cost per false positive
    false_negative_cost: 1000.0   # Cost per false negative
    investigation_cost: 25.0      # Cost per alert investigation
    system_operating_cost: 10000.0  # Monthly operating cost
    
  # Revenue parameters
  revenue:
    fraud_prevention_value: 1000.0  # Value of preventing fraud
    customer_retention_value: 200.0 # Value of not inconveniencing customer
    
  # Reporting
  reporting:
    frequency: "daily"
    include_trends: true
    generate_roi_report: true
```

### Business Metrics Implementation

```python
class BusinessMetricsTracker:
    def __init__(self, cost_config):
        self.cost_config = cost_config
        
        # Prometheus metrics for business tracking
        self.roi_gauge = Gauge('ml_business_roi', 'Return on investment')
        self.cost_savings_gauge = Gauge('ml_business_cost_savings', 'Total cost savings')
        self.processing_volume_counter = Counter('ml_business_processing_volume', 'Transactions processed')
        self.alert_volume_counter = Counter('ml_business_alert_volume', 'Alerts generated')
        
    def calculate_daily_metrics(self, predictions_df):
        """Calculate daily business metrics."""
        # Confusion matrix components
        tp = len(predictions_df[(predictions_df['actual'] == 1) & (predictions_df['predicted'] == 1)])
        tn = len(predictions_df[(predictions_df['actual'] == 0) & (predictions_df['predicted'] == 0)])
        fp = len(predictions_df[(predictions_df['actual'] == 0) & (predictions_df['predicted'] == 1)])
        fn = len(predictions_df[(predictions_df['actual'] == 1) & (predictions_df['predicted'] == 0)])
        
        # Calculate costs and benefits
        fp_cost = fp * self.cost_config['false_positive_cost']
        fn_cost = fn * self.cost_config['false_negative_cost']
        investigation_cost = (tp + fp) * self.cost_config['investigation_cost']
        
        fraud_prevented_value = tp * self.cost_config['fraud_prevention_value']
        customer_satisfaction_value = tn * self.cost_config['customer_retention_value']
        
        # Net benefit calculation
        total_cost = fp_cost + fn_cost + investigation_cost
        total_benefit = fraud_prevented_value + customer_satisfaction_value
        net_benefit = total_benefit - total_cost
        
        # ROI calculation
        operating_cost = self.cost_config['system_operating_cost'] / 30  # Daily cost
        roi = (net_benefit - operating_cost) / operating_cost if operating_cost > 0 else 0
        
        # Update metrics
        self.roi_gauge.set(roi)
        self.cost_savings_gauge.set(net_benefit)
        self.processing_volume_counter.inc(len(predictions_df))
        self.alert_volume_counter.inc(tp + fp)
        
        return {
            'roi': roi,
            'net_benefit': net_benefit,
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'fraud_prevented': tp,
            'false_alarms': fp,
            'missed_fraud': fn
        }
```

## 🖥️ Infrastructure Monitoring

### System Health Metrics

- **CPU Usage**: Application and system CPU utilization
- **Memory Usage**: RAM consumption and memory leaks
- **Disk I/O**: Read/write operations and disk space
- **Network**: Throughput and latency
- **Database**: Query performance and connection pool status
- **Cache**: Hit rates and memory usage

### Health Check Implementation

```python
# src/monitoring/health_monitor.py
from prometheus_client import Gauge, Histogram
import psutil
import time

class InfrastructureMonitor:
    def __init__(self):
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        
        # Application metrics
        self.response_time = Histogram('http_request_duration_seconds', 'Request duration')
        self.active_connections = Gauge('db_active_connections', 'Active database connections')
        
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.disk_usage.set(disk_percent)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk_percent
        }
```

## 🚨 Alerting & Notifications

### Alert Configuration

```yaml
# monitoring/alert_rules.yml
groups:
- name: ml-pipeline-alerts
  rules:
  # Model performance alerts
  - alert: ModelAccuracyDegraded
    expr: ml_model_accuracy < 0.85
    for: 10m
    labels:
      severity: warning
      category: model_performance
    annotations:
      summary: "Model accuracy below threshold"
      description: "Model accuracy is {{ $value }}, below 85% threshold"
      
  - alert: ModelAccuracyCritical
    expr: ml_model_accuracy < 0.75
    for: 5m
    labels:
      severity: critical
      category: model_performance
    annotations:
      summary: "Critical model accuracy degradation"
      description: "Model accuracy is {{ $value }}, requiring immediate attention"
      
  # Data drift alerts
  - alert: DataDriftDetected
    expr: ml_data_drift_psi_score > 0.1
    for: 5m
    labels:
      severity: warning
      category: data_quality
    annotations:
      summary: "Data drift detected"
      description: "PSI score is {{ $value }}, indicating distribution changes"
      
  - alert: SevereDataDrift
    expr: ml_data_drift_psi_score > 0.2
    for: 1m
    labels:
      severity: critical
      category: data_quality
    annotations:
      summary: "Severe data drift detected"
      description: "PSI score is {{ $value }}, requiring immediate model retraining"
      
  # Business metrics alerts
  - alert: ROIBelowThreshold
    expr: ml_business_roi < 1.0
    for: 30m
    labels:
      severity: warning
      category: business_impact
    annotations:
      summary: "ROI below break-even"
      description: "Current ROI is {{ $value }}, below 1.0 break-even point"
      
  # Infrastructure alerts
  - alert: HighCPUUsage
    expr: system_cpu_usage_percent > 80
    for: 10m
    labels:
      severity: warning
      category: infrastructure
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}%"
      
  - alert: HighMemoryUsage
    expr: system_memory_usage_percent > 90
    for: 5m
    labels:
      severity: critical
      category: infrastructure
    annotations:
      summary: "Critical memory usage"
      description: "Memory usage is {{ $value }}%"
```

### Notification Channels

```yaml
# Slack notifications
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ml-alerts'
    title: 'ML Pipeline Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}: {{ .Annotations.description }}{{ end }}'

# Email notifications
email_configs:
  - to: 'ml-team@company.com'
    from: 'alerts@company.com'
    subject: 'ML Pipeline Alert: {{ .GroupLabels.alertname }}'
    body: |
      Alert: {{ .GroupLabels.alertname }}
      Severity: {{ .CommonLabels.severity }}
      {{ range .Alerts }}
      Description: {{ .Annotations.description }}
      {{ end }}

# PagerDuty integration
pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

## 📊 Dashboard Configuration

### Grafana Dashboard Setup

Key dashboards for ML monitoring:

#### 1. Model Performance Dashboard
```json
{
  "dashboard": {
    "title": "ML Model Performance",
    "panels": [
      {
        "title": "Model Accuracy Over Time",
        "type": "stat",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "title": "Precision vs Recall",
        "type": "xy",
        "targets": [
          {
            "expr": "ml_model_precision",
            "legendFormat": "Precision"
          },
          {
            "expr": "ml_model_recall",
            "legendFormat": "Recall"
          }
        ]
      },
      {
        "title": "Business Metrics",
        "type": "table",
        "targets": [
          {
            "expr": "ml_business_roi",
            "legendFormat": "ROI"
          },
          {
            "expr": "ml_business_cost_savings",
            "legendFormat": "Cost Savings"
          }
        ]
      }
    ]
  }
}
```

#### 2. Data Quality Dashboard
```json
{
  "dashboard": {
    "title": "Data Quality Monitoring",
    "panels": [
      {
        "title": "Data Drift Score",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_data_drift_psi_score",
            "legendFormat": "PSI Score"
          }
        ],
        "alert": {
          "conditions": [
            {
              "query": {
                "queryType": "",
                "refId": "A"
              },
              "reducer": {
                "type": "last",
                "params": []
              },
              "evaluator": {
                "params": [0.1],
                "type": "gt"
              }
            }
          ]
        }
      }
    ]
  }
}
```

## 🧪 A/B Testing & Experimentation

### A/B Testing Framework

```python
class ABTestingFramework:
    def __init__(self, experiment_config):
        self.config = experiment_config
        self.experiments = {}
        
    def create_experiment(self, name, model_a, model_b, traffic_split=0.5):
        """Create a new A/B test experiment."""
        experiment = {
            'name': name,
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'start_time': time.time(),
            'results_a': [],
            'results_b': []
        }
        self.experiments[name] = experiment
        return experiment
    
    def route_traffic(self, user_id, experiment_name):
        """Route traffic to model A or B based on user ID."""
        import hashlib
        hash_object = hashlib.md5(f"{user_id}_{experiment_name}".encode())
        hash_value = int(hash_object.hexdigest(), 16)
        
        experiment = self.experiments[experiment_name]
        if (hash_value % 100) / 100 < experiment['traffic_split']:
            return 'model_a'
        else:
            return 'model_b'
    
    def record_result(self, experiment_name, model_variant, prediction, actual, metrics):
        """Record experiment results."""
        experiment = self.experiments[experiment_name]
        result = {
            'timestamp': time.time(),
            'prediction': prediction,
            'actual': actual,
            'metrics': metrics
        }
        
        if model_variant == 'model_a':
            experiment['results_a'].append(result)
        else:
            experiment['results_b'].append(result)
    
    def analyze_experiment(self, experiment_name):
        """Analyze A/B test results."""
        from scipy import stats
        
        experiment = self.experiments[experiment_name]
        results_a = experiment['results_a']
        results_b = experiment['results_b']
        
        if len(results_a) < 30 or len(results_b) < 30:
            return {'status': 'insufficient_data'}
        
        # Extract metrics for comparison
        accuracy_a = [r['metrics']['accuracy'] for r in results_a]
        accuracy_b = [r['metrics']['accuracy'] for r in results_b]
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(accuracy_a, accuracy_b)
        
        # Effect size calculation
        mean_a = np.mean(accuracy_a)
        mean_b = np.mean(accuracy_b)
        pooled_std = np.sqrt((np.std(accuracy_a)**2 + np.std(accuracy_b)**2) / 2)
        effect_size = (mean_b - mean_a) / pooled_std
        
        return {
            'status': 'complete',
            'model_a_performance': mean_a,
            'model_b_performance': mean_b,
            'improvement': mean_b - mean_a,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': effect_size,
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b)
        }
```

## 🔧 Troubleshooting

### Common Monitoring Issues

#### 1. Missing Metrics
```bash
# Check if metrics endpoint is accessible
curl http://localhost:8080/metrics

# Verify Prometheus is scraping
# Check Prometheus targets: http://localhost:9090/targets

# Debug metric collection
kubectl logs -f deployment/ml-pipeline-app | grep -i metric
```

#### 2. Alert Fatigue
```yaml
# Implement alert grouping and inhibition rules
route:
  group_by: ['alertname', 'category']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'category']
```

#### 3. Dashboard Performance Issues
```bash
# Optimize Grafana queries
# Use recording rules for expensive calculations
# Add query caching
# Limit time ranges for heavy dashboards
```

### Monitoring Health Checks

```python
def monitoring_health_check():
    """Comprehensive health check for monitoring system."""
    health_status = {
        'prometheus': check_prometheus_health(),
        'grafana': check_grafana_health(),
        'alertmanager': check_alertmanager_health(),
        'data_collection': check_data_collection(),
        'drift_detection': check_drift_detection()
    }
    
    overall_health = all(health_status.values())
    
    return {
        'overall_healthy': overall_health,
        'components': health_status,
        'timestamp': time.time()
    }
```

## 📞 Support

For monitoring-related issues:

1. **Check Logs**: Review application and monitoring component logs
2. **Verify Configuration**: Validate monitoring configuration files
3. **Test Connectivity**: Ensure all monitoring components can communicate
4. **Community Help**: Seek assistance in community forums
5. **Professional Support**: Contact enterprise support for production issues

---

**Monitoring Success!** 📊 Your ML Pipeline Framework v2.0 now has comprehensive monitoring and observability capabilities for production ML systems.