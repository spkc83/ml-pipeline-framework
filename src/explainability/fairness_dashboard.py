"""Interactive fairness dashboard for bias monitoring and real-time fairness metrics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
warnings.filterwarnings('ignore')


@dataclass
class BiasAlert:
    """Represents a bias alert in the system."""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    threshold_value: float
    actual_value: float
    protected_attribute: str
    affected_groups: List[str]
    description: str
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data


@dataclass
class FairnessMetricSnapshot:
    """Represents a snapshot of fairness metrics at a point in time."""
    timestamp: datetime
    protected_attribute: str
    group_values: Dict[str, Any]
    metric_name: str
    metric_value: float
    threshold: Optional[float]
    status: str  # 'pass', 'warning', 'fail'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data


class FairnessMetricsCalculator:
    """Calculate various fairness metrics for bias monitoring."""
    
    def __init__(self):
        """Initialize fairness metrics calculator."""
        self.supported_metrics = [
            'demographic_parity',
            'equalized_odds',
            'equality_of_opportunity',
            'calibration',
            'individual_fairness',
            'counterfactual_fairness'
        ]
    
    def calculate_demographic_parity(self, y_pred: np.ndarray, protected_attr: pd.Series) -> Dict[str, Any]:
        """Calculate demographic parity (statistical parity).
        
        Args:
            y_pred: Model predictions
            protected_attr: Protected attribute values
            
        Returns:
            Dictionary with demographic parity metrics
        """
        results = {}
        
        # Overall positive rate
        overall_positive_rate = np.mean(y_pred)
        results['overall_positive_rate'] = overall_positive_rate
        
        # Group-specific positive rates
        group_rates = {}
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_positive_rate = np.mean(y_pred[group_mask])
            group_rates[str(group)] = group_positive_rate
        
        results['group_rates'] = group_rates
        
        # Calculate parity differences
        parity_differences = {}
        max_rate = max(group_rates.values())
        min_rate = min(group_rates.values())
        
        for group, rate in group_rates.items():
            parity_differences[group] = rate - overall_positive_rate
        
        results['parity_differences'] = parity_differences
        results['max_difference'] = max_rate - min_rate
        results['disparate_impact_ratio'] = min_rate / max_rate if max_rate > 0 else 0
        
        return results
    
    def calculate_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               protected_attr: pd.Series) -> Dict[str, Any]:
        """Calculate equalized odds.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attr: Protected attribute values
            
        Returns:
            Dictionary with equalized odds metrics
        """
        results = {}
        group_metrics = {}
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # True Positive Rate
            if np.sum(group_y_true == 1) > 0:
                tpr = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_true == 1)
            else:
                tpr = 0.0
            
            # True Negative Rate (Specificity)
            if np.sum(group_y_true == 0) > 0:
                tnr = np.sum((group_y_true == 0) & (group_y_pred == 0)) / np.sum(group_y_true == 0)
            else:
                tnr = 0.0
            
            # False Positive Rate
            if np.sum(group_y_true == 0) > 0:
                fpr = np.sum((group_y_true == 0) & (group_y_pred == 1)) / np.sum(group_y_true == 0)
            else:
                fpr = 0.0
            
            group_metrics[str(group)] = {
                'tpr': tpr,
                'tnr': tnr,
                'fpr': fpr
            }
        
        results['group_metrics'] = group_metrics
        
        # Calculate differences
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        tnrs = [metrics['tnr'] for metrics in group_metrics.values()]
        
        results['tpr_difference'] = max(tprs) - min(tprs) if tprs else 0
        results['tnr_difference'] = max(tnrs) - min(tnrs) if tnrs else 0
        results['max_difference'] = max(results['tpr_difference'], results['tnr_difference'])
        
        return results
    
    def calculate_equality_of_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        protected_attr: pd.Series) -> Dict[str, Any]:
        """Calculate equality of opportunity (equal TPR across groups).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attr: Protected attribute values
            
        Returns:
            Dictionary with equality of opportunity metrics
        """
        results = {}
        group_tprs = {}
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # True Positive Rate
            if np.sum(group_y_true == 1) > 0:
                tpr = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_true == 1)
            else:
                tpr = 0.0
            
            group_tprs[str(group)] = tpr
        
        results['group_tprs'] = group_tprs
        
        # Calculate TPR difference
        tpr_values = list(group_tprs.values())
        results['tpr_difference'] = max(tpr_values) - min(tpr_values) if tpr_values else 0
        results['min_tpr'] = min(tpr_values) if tpr_values else 0
        results['max_tpr'] = max(tpr_values) if tpr_values else 0
        
        return results
    
    def calculate_calibration(self, y_true: np.ndarray, y_prob: np.ndarray,
                            protected_attr: pd.Series, n_bins: int = 10) -> Dict[str, Any]:
        """Calculate calibration metrics across groups.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            protected_attr: Protected attribute values
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration metrics
        """
        results = {}
        group_calibrations = {}
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            # Calculate calibration curve
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            observed_freqs = []
            predicted_freqs = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                if np.sum(in_bin) > 0:
                    observed_freq = np.mean(group_y_true[in_bin])
                    predicted_freq = np.mean(group_y_prob[in_bin])
                    bin_count = np.sum(in_bin)
                else:
                    observed_freq = 0
                    predicted_freq = 0
                    bin_count = 0
                
                observed_freqs.append(observed_freq)
                predicted_freqs.append(predicted_freq)
                bin_counts.append(bin_count)
            
            # Calculate Expected Calibration Error (ECE)
            ece = 0
            total_samples = len(group_y_true)
            
            for i in range(n_bins):
                if bin_counts[i] > 0:
                    weight = bin_counts[i] / total_samples
                    ece += weight * abs(observed_freqs[i] - predicted_freqs[i])
            
            group_calibrations[str(group)] = {
                'ece': ece,
                'observed_freqs': observed_freqs,
                'predicted_freqs': predicted_freqs,
                'bin_counts': bin_counts
            }
        
        results['group_calibrations'] = group_calibrations
        
        # Calculate calibration differences
        eces = [cal['ece'] for cal in group_calibrations.values()]
        results['ece_difference'] = max(eces) - min(eces) if eces else 0
        results['max_ece'] = max(eces) if eces else 0
        
        return results
    
    def calculate_individual_fairness(self, X: np.ndarray, y_pred: np.ndarray,
                                    distance_threshold: float = 0.1) -> Dict[str, Any]:
        """Calculate individual fairness (similar individuals get similar outcomes).
        
        Args:
            X: Feature matrix
            y_pred: Predictions
            distance_threshold: Threshold for considering individuals similar
            
        Returns:
            Dictionary with individual fairness metrics
        """
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate pairwise distances
        distances = euclidean_distances(X_scaled)
        
        # Find pairs of similar individuals
        similar_pairs = np.where((distances < distance_threshold) & (distances > 0))
        
        if len(similar_pairs[0]) == 0:
            return {
                'individual_fairness_score': 1.0,
                'n_similar_pairs': 0,
                'avg_prediction_difference': 0.0
            }
        
        # Calculate prediction differences for similar pairs
        prediction_differences = []
        for i, j in zip(similar_pairs[0], similar_pairs[1]):
            pred_diff = abs(y_pred[i] - y_pred[j])
            prediction_differences.append(pred_diff)
        
        avg_pred_diff = np.mean(prediction_differences)
        
        # Individual fairness score (lower difference = higher fairness)
        if_score = 1.0 / (1.0 + avg_pred_diff)
        
        return {
            'individual_fairness_score': if_score,
            'n_similar_pairs': len(similar_pairs[0]),
            'avg_prediction_difference': avg_pred_diff,
            'max_prediction_difference': max(prediction_differences) if prediction_differences else 0,
            'prediction_differences': prediction_differences
        }


class BiasDetector:
    """Detect bias and generate alerts based on fairness metrics."""
    
    def __init__(self, fairness_thresholds: Optional[Dict[str, float]] = None):
        """Initialize bias detector.
        
        Args:
            fairness_thresholds: Thresholds for different fairness metrics
        """
        self.fairness_thresholds = fairness_thresholds or {
            'demographic_parity': 0.1,  # Maximum allowed difference in positive rates
            'equalized_odds': 0.1,      # Maximum allowed difference in TPR/TNR
            'equality_of_opportunity': 0.1,  # Maximum allowed difference in TPR
            'calibration': 0.05,        # Maximum allowed difference in ECE
            'individual_fairness': 0.8  # Minimum individual fairness score
        }
        self.metrics_calculator = FairnessMetricsCalculator()
        
    def detect_bias(self, y_true: Optional[np.ndarray], y_pred: np.ndarray,
                   y_prob: Optional[np.ndarray], protected_attr: pd.Series,
                   X: Optional[np.ndarray] = None) -> List[BiasAlert]:
        """Detect bias across multiple fairness metrics.
        
        Args:
            y_true: True labels (optional)
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            protected_attr: Protected attribute values
            X: Feature matrix (optional, for individual fairness)
            
        Returns:
            List of bias alerts
        """
        alerts = []
        timestamp = datetime.now()
        
        # Demographic Parity
        dp_results = self.metrics_calculator.calculate_demographic_parity(y_pred, protected_attr)
        if dp_results['max_difference'] > self.fairness_thresholds['demographic_parity']:
            alert = BiasAlert(
                alert_id=f"DP_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                alert_type='demographic_parity',
                severity=self._determine_severity(dp_results['max_difference'], 
                                                self.fairness_thresholds['demographic_parity']),
                metric_name='demographic_parity_difference',
                threshold_value=self.fairness_thresholds['demographic_parity'],
                actual_value=dp_results['max_difference'],
                protected_attribute=protected_attr.name,
                affected_groups=list(dp_results['group_rates'].keys()),
                description=f"Demographic parity violation: {dp_results['max_difference']:.3f} difference in positive rates",
                recommended_action="Review model training data and feature selection for potential bias"
            )
            alerts.append(alert)
        
        # Equalized Odds (if true labels available)
        if y_true is not None:
            eo_results = self.metrics_calculator.calculate_equalized_odds(y_true, y_pred, protected_attr)
            if eo_results['max_difference'] > self.fairness_thresholds['equalized_odds']:
                alert = BiasAlert(
                    alert_id=f"EO_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    alert_type='equalized_odds',
                    severity=self._determine_severity(eo_results['max_difference'],
                                                    self.fairness_thresholds['equalized_odds']),
                    metric_name='equalized_odds_difference',
                    threshold_value=self.fairness_thresholds['equalized_odds'],
                    actual_value=eo_results['max_difference'],
                    protected_attribute=protected_attr.name,
                    affected_groups=list(eo_results['group_metrics'].keys()),
                    description=f"Equalized odds violation: {eo_results['max_difference']:.3f} difference in TPR/TNR",
                    recommended_action="Implement post-processing fairness constraints or re-train with fairness regularization"
                )
                alerts.append(alert)
        
        # Calibration (if probabilities available)
        if y_true is not None and y_prob is not None:
            cal_results = self.metrics_calculator.calculate_calibration(y_true, y_prob, protected_attr)
            if cal_results['ece_difference'] > self.fairness_thresholds['calibration']:
                alert = BiasAlert(
                    alert_id=f"CAL_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    alert_type='calibration',
                    severity=self._determine_severity(cal_results['ece_difference'],
                                                    self.fairness_thresholds['calibration']),
                    metric_name='calibration_difference',
                    threshold_value=self.fairness_thresholds['calibration'],
                    actual_value=cal_results['ece_difference'],
                    protected_attribute=protected_attr.name,
                    affected_groups=list(cal_results['group_calibrations'].keys()),
                    description=f"Calibration violation: {cal_results['ece_difference']:.3f} difference in ECE",
                    recommended_action="Apply calibration techniques like Platt scaling or isotonic regression per group"
                )
                alerts.append(alert)
        
        # Individual Fairness (if features available)
        if X is not None:
            if_results = self.metrics_calculator.calculate_individual_fairness(X, y_pred)
            if if_results['individual_fairness_score'] < self.fairness_thresholds['individual_fairness']:
                alert = BiasAlert(
                    alert_id=f"IF_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    alert_type='individual_fairness',
                    severity=self._determine_severity(1 - if_results['individual_fairness_score'], 0.2),
                    metric_name='individual_fairness_score',
                    threshold_value=self.fairness_thresholds['individual_fairness'],
                    actual_value=if_results['individual_fairness_score'],
                    protected_attribute=protected_attr.name,
                    affected_groups=['all'],
                    description=f"Individual fairness violation: {if_results['individual_fairness_score']:.3f} score",
                    recommended_action="Review feature representation and consider fairness-aware distance metrics"
                )
                alerts.append(alert)
        
        return alerts
    
    def _determine_severity(self, actual_value: float, threshold: float) -> str:
        """Determine alert severity based on how much the threshold is exceeded."""
        ratio = actual_value / threshold
        
        if ratio >= 3.0:
            return 'critical'
        elif ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        else:
            return 'low'


class FairnessDashboard:
    """Interactive dashboard for monitoring fairness metrics and bias detection."""
    
    def __init__(self, fairness_thresholds: Optional[Dict[str, float]] = None):
        """Initialize fairness dashboard.
        
        Args:
            fairness_thresholds: Thresholds for bias detection
        """
        self.bias_detector = BiasDetector(fairness_thresholds)
        self.metrics_calculator = FairnessMetricsCalculator()
        self.metric_history = []
        self.alert_history = []
        
    def update_metrics(self, y_true: Optional[np.ndarray], y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray], protected_attrs: Dict[str, pd.Series],
                      X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Update fairness metrics and detect bias.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            protected_attrs: Dictionary of protected attributes
            X: Feature matrix
            
        Returns:
            Dictionary with updated metrics and alerts
        """
        timestamp = datetime.now()
        update_results = {
            'timestamp': timestamp.isoformat(),
            'metrics': {},
            'alerts': [],
            'summary': {}
        }
        
        total_alerts = 0
        
        for attr_name, attr_values in protected_attrs.items():
            # Calculate all fairness metrics
            attr_metrics = {}
            
            # Demographic Parity
            dp_results = self.metrics_calculator.calculate_demographic_parity(y_pred, attr_values)
            attr_metrics['demographic_parity'] = dp_results
            
            # Store metric snapshot
            dp_snapshot = FairnessMetricSnapshot(
                timestamp=timestamp,
                protected_attribute=attr_name,
                group_values=dp_results['group_rates'],
                metric_name='demographic_parity',
                metric_value=dp_results['max_difference'],
                threshold=self.bias_detector.fairness_thresholds['demographic_parity'],
                status='pass' if dp_results['max_difference'] <= self.bias_detector.fairness_thresholds['demographic_parity'] else 'fail'
            )
            self.metric_history.append(dp_snapshot)
            
            # Equalized Odds (if labels available)
            if y_true is not None:
                eo_results = self.metrics_calculator.calculate_equalized_odds(y_true, y_pred, attr_values)
                attr_metrics['equalized_odds'] = eo_results
                
                eo_snapshot = FairnessMetricSnapshot(
                    timestamp=timestamp,
                    protected_attribute=attr_name,
                    group_values=eo_results['group_metrics'],
                    metric_name='equalized_odds',
                    metric_value=eo_results['max_difference'],
                    threshold=self.bias_detector.fairness_thresholds['equalized_odds'],
                    status='pass' if eo_results['max_difference'] <= self.bias_detector.fairness_thresholds['equalized_odds'] else 'fail'
                )
                self.metric_history.append(eo_snapshot)
            
            # Calibration (if probabilities available)
            if y_true is not None and y_prob is not None:
                cal_results = self.metrics_calculator.calculate_calibration(y_true, y_prob, attr_values)
                attr_metrics['calibration'] = cal_results
                
                cal_snapshot = FairnessMetricSnapshot(
                    timestamp=timestamp,
                    protected_attribute=attr_name,
                    group_values=cal_results['group_calibrations'],
                    metric_name='calibration',
                    metric_value=cal_results['ece_difference'],
                    threshold=self.bias_detector.fairness_thresholds['calibration'],
                    status='pass' if cal_results['ece_difference'] <= self.bias_detector.fairness_thresholds['calibration'] else 'fail'
                )
                self.metric_history.append(cal_snapshot)
            
            # Detect bias for this attribute
            alerts = self.bias_detector.detect_bias(y_true, y_pred, y_prob, attr_values, X)
            update_results['alerts'].extend([alert.to_dict() for alert in alerts])
            self.alert_history.extend(alerts)
            total_alerts += len(alerts)
            
            update_results['metrics'][attr_name] = attr_metrics
        
        # Individual Fairness (global metric)
        if X is not None:
            if_results = self.metrics_calculator.calculate_individual_fairness(X, y_pred)
            update_results['metrics']['individual_fairness'] = if_results
        
        # Summary statistics
        update_results['summary'] = {
            'total_alerts': total_alerts,
            'alert_severity_counts': self._count_alert_severities(update_results['alerts']),
            'protected_attributes_analyzed': list(protected_attrs.keys()),
            'metrics_calculated': list(update_results['metrics'].keys())
        }
        
        return update_results
    
    def _count_alert_severities(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count alerts by severity level."""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for alert in alerts:
            severity = alert.get('severity', 'low')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts
    
    def create_fairness_dashboard(self, figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """Create comprehensive fairness monitoring dashboard.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with dashboard
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # 1. Alert Timeline (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_alert_timeline(ax1)
        
        # 2. Alert Severity Distribution (top row, single column)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_alert_severity_distribution(ax2)
        
        # 3. Current Fairness Status (top row, single column)
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_current_fairness_status(ax3)
        
        # 4. Metric Trends (middle row, spans 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_metric_trends(ax4)
        
        # 5. Group Comparison (middle row, spans 2 columns)
        ax5 = fig.add_subplot(gs[1, 2:])
        self._plot_group_comparison(ax5)
        
        # 6. Calibration Analysis (bottom row, spans 2 columns)
        ax6 = fig.add_subplot(gs[2, :2])
        self._plot_calibration_analysis(ax6)
        
        # 7. Recommendation Summary (bottom row, spans 2 columns)
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_recommendation_summary(ax7)
        
        plt.suptitle('Fairness Monitoring Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_alert_timeline(self, ax):
        """Plot alert timeline."""
        if not self.alert_history:
            ax.text(0.5, 0.5, 'No alerts to display', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Alert Timeline')
            return
        
        # Group alerts by hour for visualization
        alert_times = [alert.timestamp for alert in self.alert_history]
        severities = [alert.severity for alert in self.alert_history]
        
        # Create scatter plot with different colors for severities
        severity_colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
        
        for severity in ['low', 'medium', 'high', 'critical']:
            severity_alerts = [(time, i) for i, (time, sev) in enumerate(zip(alert_times, severities)) if sev == severity]
            if severity_alerts:
                times, indices = zip(*severity_alerts)
                ax.scatter(times, [1] * len(times), c=severity_colors[severity], 
                          label=severity, alpha=0.7, s=100)
        
        ax.set_ylabel('Alerts')
        ax.set_title('Alert Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_alert_severity_distribution(self, ax):
        """Plot alert severity distribution."""
        if not self.alert_history:
            ax.text(0.5, 0.5, 'No alerts', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Alert Severity Distribution')
            return
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for alert in self.alert_history:
            if alert.severity in severity_counts:
                severity_counts[alert.severity] += 1
        
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = ['green', 'yellow', 'orange', 'red']
        
        ax.pie(counts, labels=severities, colors=colors, autopct='%1.1f%%')
        ax.set_title('Alert Severity Distribution')
    
    def _plot_current_fairness_status(self, ax):
        """Plot current fairness status."""
        if not self.metric_history:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Current Fairness Status')
            return
        
        # Get latest metrics
        latest_metrics = {}
        for snapshot in self.metric_history[-10:]:  # Last 10 snapshots
            key = f"{snapshot.protected_attribute}_{snapshot.metric_name}"
            latest_metrics[key] = snapshot.status
        
        # Count pass/fail status
        status_counts = {'pass': 0, 'fail': 0}
        for status in latest_metrics.values():
            if status in status_counts:
                status_counts[status] += 1
        
        if sum(status_counts.values()) > 0:
            ax.pie(status_counts.values(), labels=status_counts.keys(), 
                  colors=['green', 'red'], autopct='%1.1f%%')
        
        ax.set_title('Current Fairness Status')
    
    def _plot_metric_trends(self, ax):
        """Plot metric trends over time."""
        if not self.metric_history:
            ax.text(0.5, 0.5, 'No metric history', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metric Trends')
            return
        
        # Group by metric type
        metric_types = {}
        for snapshot in self.metric_history:
            if snapshot.metric_name not in metric_types:
                metric_types[snapshot.metric_name] = {'times': [], 'values': []}
            metric_types[snapshot.metric_name]['times'].append(snapshot.timestamp)
            metric_types[snapshot.metric_name]['values'].append(snapshot.metric_value)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (metric_name, data) in enumerate(metric_types.items()):
            ax.plot(data['times'], data['values'], 
                   label=metric_name, color=colors[i % len(colors)], marker='o')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Fairness Metric Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_group_comparison(self, ax):
        """Plot group comparison for latest metrics."""
        if not self.metric_history:
            ax.text(0.5, 0.5, 'No data for comparison', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Group Comparison')
            return
        
        # Get latest demographic parity data
        latest_dp = None
        for snapshot in reversed(self.metric_history):
            if snapshot.metric_name == 'demographic_parity':
                latest_dp = snapshot
                break
        
        if latest_dp and isinstance(latest_dp.group_values, dict):
            groups = list(latest_dp.group_values.keys())
            rates = list(latest_dp.group_values.values())
            
            bars = ax.bar(groups, rates)
            ax.set_ylabel('Positive Rate')
            ax.set_title(f'Group Comparison - {latest_dp.protected_attribute}')
            ax.grid(True, alpha=0.3)
            
            # Color bars based on deviation from mean
            mean_rate = np.mean(rates)
            for bar, rate in zip(bars, rates):
                if abs(rate - mean_rate) > 0.1:
                    bar.set_color('red')
                elif abs(rate - mean_rate) > 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
        else:
            ax.text(0.5, 0.5, 'No group data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Group Comparison')
    
    def _plot_calibration_analysis(self, ax):
        """Plot calibration analysis."""
        # Get latest calibration data
        latest_cal = None
        for snapshot in reversed(self.metric_history):
            if snapshot.metric_name == 'calibration':
                latest_cal = snapshot
                break
        
        if latest_cal and isinstance(latest_cal.group_values, dict):
            # Plot calibration curves for each group
            for group_name, cal_data in latest_cal.group_values.items():
                if 'observed_freqs' in cal_data and 'predicted_freqs' in cal_data:
                    ax.plot(cal_data['predicted_freqs'], cal_data['observed_freqs'], 
                           marker='o', label=f'Group {group_name}')
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No calibration data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Calibration Analysis')
    
    def _plot_recommendation_summary(self, ax):
        """Plot recommendation summary."""
        if not self.alert_history:
            ax.text(0.5, 0.5, 'No recommendations', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Recommendations')
            return
        
        # Count recommendation types
        recommendations = {}
        for alert in self.alert_history[-10:]:  # Last 10 alerts
            action = alert.recommended_action[:50] + "..." if len(alert.recommended_action) > 50 else alert.recommended_action
            if action in recommendations:
                recommendations[action] += 1
            else:
                recommendations[action] = 1
        
        if recommendations:
            actions = list(recommendations.keys())
            counts = list(recommendations.values())
            
            bars = ax.barh(range(len(actions)), counts)
            ax.set_yticks(range(len(actions)))
            ax.set_yticklabels(actions, fontsize=8)
            ax.set_xlabel('Frequency')
            ax.set_title('Top Recommendations')
            
            # Color bars by frequency
            max_count = max(counts)
            for bar, count in zip(bars, counts):
                intensity = count / max_count
                bar.set_color(plt.cm.Reds(intensity))
        else:
            ax.text(0.5, 0.5, 'No recommendations', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Recommendations')
    
    def generate_fairness_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive fairness monitoring report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Dictionary with comprehensive fairness report
        """
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'total_metric_snapshots': len(self.metric_history),
                'total_alerts': len(self.alert_history),
                'monitoring_period': self._get_monitoring_period()
            },
            'executive_summary': self._generate_executive_summary(),
            'alert_analysis': self._analyze_alerts(),
            'metric_analysis': self._analyze_metrics(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _get_monitoring_period(self) -> Dict[str, Any]:
        """Get monitoring period information."""
        if not self.metric_history:
            return {'start_date': None, 'end_date': None, 'duration_hours': 0}
        
        timestamps = [snapshot.timestamp for snapshot in self.metric_history]
        start_date = min(timestamps)
        end_date = max(timestamps)
        duration = end_date - start_date
        
        return {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'duration_hours': duration.total_seconds() / 3600
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of fairness monitoring."""
        total_alerts = len(self.alert_history)
        
        if total_alerts == 0:
            return {
                'status': 'No bias detected',
                'alert_summary': 'No fairness violations found during monitoring period',
                'key_findings': ['All fairness metrics within acceptable thresholds']
            }
        
        # Alert severity breakdown
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for alert in self.alert_history:
            if alert.severity in severity_counts:
                severity_counts[alert.severity] += 1
        
        # Determine overall status
        if severity_counts['critical'] > 0:
            status = 'Critical bias detected'
        elif severity_counts['high'] > 0:
            status = 'High bias detected'
        elif severity_counts['medium'] > 0:
            status = 'Moderate bias detected'
        else:
            status = 'Low bias detected'
        
        return {
            'status': status,
            'total_alerts': total_alerts,
            'severity_breakdown': severity_counts,
            'most_common_violation': self._get_most_common_violation(),
            'affected_attributes': list(set(alert.protected_attribute for alert in self.alert_history))
        }
    
    def _get_most_common_violation(self) -> str:
        """Get the most common type of bias violation."""
        if not self.alert_history:
            return 'None'
        
        violation_counts = {}
        for alert in self.alert_history:
            if alert.alert_type in violation_counts:
                violation_counts[alert.alert_type] += 1
            else:
                violation_counts[alert.alert_type] = 1
        
        return max(violation_counts, key=violation_counts.get)
    
    def _analyze_alerts(self) -> Dict[str, Any]:
        """Analyze alert patterns and trends."""
        if not self.alert_history:
            return {'message': 'No alerts to analyze'}
        
        # Time-based analysis
        alert_times = [alert.timestamp for alert in self.alert_history]
        time_analysis = {
            'first_alert': min(alert_times).isoformat(),
            'latest_alert': max(alert_times).isoformat(),
            'alert_frequency': len(self.alert_history) / max(1, (max(alert_times) - min(alert_times)).total_seconds() / 3600)
        }
        
        # Protected attribute analysis
        attr_counts = {}
        for alert in self.alert_history:
            if alert.protected_attribute in attr_counts:
                attr_counts[alert.protected_attribute] += 1
            else:
                attr_counts[alert.protected_attribute] = 1
        
        return {
            'time_analysis': time_analysis,
            'protected_attribute_analysis': attr_counts,
            'alert_types': {alert.alert_type: alert.metric_name for alert in self.alert_history},
            'severity_trend': [alert.severity for alert in self.alert_history[-10:]]
        }
    
    def _analyze_metrics(self) -> Dict[str, Any]:
        """Analyze metric trends and patterns."""
        if not self.metric_history:
            return {'message': 'No metrics to analyze'}
        
        # Group metrics by type
        metric_analysis = {}
        
        for metric_name in ['demographic_parity', 'equalized_odds', 'calibration']:
            metric_snapshots = [s for s in self.metric_history if s.metric_name == metric_name]
            
            if metric_snapshots:
                values = [s.metric_value for s in metric_snapshots]
                metric_analysis[metric_name] = {
                    'current_value': values[-1],
                    'trend': 'improving' if len(values) > 1 and values[-1] < values[0] else 'stable',
                    'average_value': np.mean(values),
                    'violation_rate': sum(1 for s in metric_snapshots if s.status == 'fail') / len(metric_snapshots)
                }
        
        return metric_analysis
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on alert history."""
        recommendations = []
        
        if not self.alert_history:
            recommendations.append({
                'priority': 'low',
                'category': 'monitoring',
                'recommendation': 'Continue monitoring fairness metrics to ensure ongoing compliance',
                'rationale': 'No bias violations detected'
            })
            return recommendations
        
        # Analyze common issues
        alert_types = [alert.alert_type for alert in self.alert_history]
        type_counts = {t: alert_types.count(t) for t in set(alert_types)}
        
        for alert_type, count in type_counts.items():
            if count >= 3:  # Recurring issue
                recommendations.append({
                    'priority': 'high',
                    'category': 'bias_mitigation',
                    'recommendation': f'Address recurring {alert_type} violations through systematic intervention',
                    'rationale': f'{count} instances of {alert_type} violations detected'
                })
        
        # Critical alerts
        critical_alerts = [alert for alert in self.alert_history if alert.severity == 'critical']
        if critical_alerts:
            recommendations.append({
                'priority': 'critical',
                'category': 'immediate_action',
                'recommendation': 'Immediately review and potentially halt model deployment due to critical bias',
                'rationale': f'{len(critical_alerts)} critical bias violations detected'
            })
        
        return recommendations