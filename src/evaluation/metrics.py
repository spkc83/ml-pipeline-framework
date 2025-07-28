import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import json

logger = logging.getLogger(__name__)

# Additional statistical imports
try:
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available for advanced statistical tests")
    SCIPY_AVAILABLE = False
    ks_2samp = chi2_contingency = None

# Plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available for interactive plots")
    PLOTLY_AVAILABLE = False
    go = px = make_subplots = None


class MetricsError(Exception):
    pass


class ThresholdOptimizer:
    """
    Threshold optimization utility for binary classification models.
    """
    
    def __init__(self, metric: str = 'f1', business_value_matrix: Optional[Dict[str, float]] = None):
        """
        Initialize threshold optimizer.
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'business_value')
            business_value_matrix: Dictionary with TP, TN, FP, FN business values
        """
        self.metric = metric
        self.business_value_matrix = business_value_matrix or {
            'TP': 1.0, 'TN': 1.0, 'FP': -1.0, 'FN': -1.0
        }
    
    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Find optimal threshold for the specified metric.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            thresholds: Custom thresholds to evaluate (default: 100 points between 0 and 1)
            
        Returns:
            Dictionary with optimal threshold and performance metrics
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        scores = []
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Business value calculation
            business_value = (tp * self.business_value_matrix['TP'] +
                            tn * self.business_value_matrix['TN'] +
                            fp * self.business_value_matrix['FP'] +
                            fn * self.business_value_matrix['FN'])
            
            metrics_dict = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'business_value': business_value,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
            
            threshold_metrics.append(metrics_dict)
            
            # Select score based on optimization metric
            if self.metric == 'f1':
                scores.append(f1)
            elif self.metric == 'precision':
                scores.append(precision)
            elif self.metric == 'recall':
                scores.append(recall)
            elif self.metric == 'accuracy':
                scores.append(accuracy)
            elif self.metric == 'business_value':
                scores.append(business_value)
            else:
                raise ValueError(f"Unknown optimization metric: {self.metric}")
        
        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        optimal_metrics = threshold_metrics[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_score': optimal_score,
            'optimal_metrics': optimal_metrics,
            'all_thresholds': thresholds,
            'all_scores': scores,
            'all_metrics': threshold_metrics
        }
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """
        Plot threshold analysis showing metric curves.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        optimization_result = self.optimize_threshold(y_true, y_pred_proba)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
        thresholds = optimization_result['all_thresholds']
        metrics_data = optimization_result['all_metrics']
        
        # Plot 1: Primary metrics
        precisions = [m['precision'] for m in metrics_data]
        recalls = [m['recall'] for m in metrics_data]
        f1_scores = [m['f1'] for m in metrics_data]
        accuracies = [m['accuracy'] for m in metrics_data]
        
        ax1.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax1.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax1.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        ax1.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        
        # Mark optimal threshold
        optimal_threshold = optimization_result['optimal_threshold']
        ax1.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal ({self.metric}): {optimal_threshold:.3f}')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Business value and counts
        business_values = [m['business_value'] for m in metrics_data]
        tps = [m['tp'] for m in metrics_data]
        fps = [m['fp'] for m in metrics_data]
        
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(thresholds, business_values, 'g-', linewidth=2, label='Business Value')
        line2 = ax2_twin.plot(thresholds, tps, 'b-', alpha=0.7, label='True Positives')
        line3 = ax2_twin.plot(thresholds, fps, 'r-', alpha=0.7, label='False Positives')
        
        ax2.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Business Value', color='g')
        ax2_twin.set_ylabel('Count', color='b')
        ax2.set_title('Business Impact vs Threshold')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to: {save_path}")
        
        plt.show()


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification and regression models
    with business metrics, calibration analysis, and threshold optimization.
    """
    
    def __init__(self, task_type: str = 'classification', 
                 pos_label: Union[int, str] = 1,
                 business_value_matrix: Optional[Dict[str, float]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            pos_label: Positive class label for binary classification
            business_value_matrix: Business values for TP, TN, FP, FN
        """
        self.task_type = task_type
        self.pos_label = pos_label
        self.business_value_matrix = business_value_matrix or {
            'TP': 1.0, 'TN': 1.0, 'FP': -1.0, 'FN': -1.0
        }
        
        logger.info(f"Initialized MetricsCalculator for {task_type} task")
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_pred_proba: Optional[np.ndarray] = None,
                                       average: str = 'weighted') -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for binary classification)
            average: Averaging strategy for multiclass ('weighted', 'macro', 'micro')
            
        Returns:
            Dictionary with all classification metrics
        """
        logger.info("Calculating classification metrics")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Binary classification specific metrics
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba
            
            # ROC AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba_pos)
            
            # Precision-Recall AUC
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba_pos)
            
            # Brier Score
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba_pos)
            
            # Log Loss
            try:
                if y_pred_proba.ndim == 2:
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                else:
                    # Convert to 2D for log_loss
                    y_pred_proba_2d = np.column_stack([1 - y_pred_proba_pos, y_pred_proba_pos])
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba_2d)
            except:
                logger.warning("Could not calculate log loss")
            
            # KS Statistic
            metrics['ks_statistic'] = self.calculate_ks_statistic(y_true, y_pred_proba_pos)
            
            # Gini coefficient
            metrics['gini_coefficient'] = 2 * metrics['roc_auc'] - 1
        
        # Classification report
        from sklearn.metrics import classification_report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all regression metrics
        """
        logger.info("Calculating regression metrics")
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            metrics['mape'] = mape
        
        # Mean Directional Accuracy (for time series)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = stats.skew(residuals) if SCIPY_AVAILABLE else None
        metrics['residual_kurtosis'] = stats.kurtosis(residuals) if SCIPY_AVAILABLE else None
        
        return metrics
    
    def calculate_ks_statistic(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            KS statistic value
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available for KS statistic calculation")
            return 0.0
        
        try:
            pos_scores = y_pred_proba[y_true == 1]
            neg_scores = y_pred_proba[y_true == 0]
            
            ks_stat, _ = ks_2samp(pos_scores, neg_scores)
            return ks_stat
            
        except Exception as e:
            logger.warning(f"Failed to calculate KS statistic: {e}")
            return 0.0
    
    def calculate_business_metrics_by_decile(self, y_true: np.ndarray, 
                                           y_pred_proba: np.ndarray,
                                           monetary_values: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate business metrics by prediction probability deciles.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            monetary_values: Optional monetary values for each sample
            
        Returns:
            DataFrame with decile-wise business metrics
        """
        logger.info("Calculating business metrics by decile")
        
        # Create deciles based on predicted probabilities
        deciles = pd.qcut(y_pred_proba, q=10, labels=False, duplicates='drop') + 1
        
        decile_metrics = []
        
        for decile in sorted(np.unique(deciles)):
            mask = deciles == decile
            
            if not np.any(mask):
                continue
            
            decile_y_true = y_true[mask]
            decile_y_pred_proba = y_pred_proba[mask]
            decile_monetary = monetary_values[mask] if monetary_values is not None else None
            
            # Basic counts
            n_samples = len(decile_y_true)
            n_positives = np.sum(decile_y_true)
            n_negatives = n_samples - n_positives
            
            # Rates
            positive_rate = n_positives / n_samples if n_samples > 0 else 0
            capture_rate = n_positives / np.sum(y_true) if np.sum(y_true) > 0 else 0
            
            # Probability statistics
            min_prob = np.min(decile_y_pred_proba)
            max_prob = np.max(decile_y_pred_proba)
            avg_prob = np.mean(decile_y_pred_proba)
            
            # Business value calculation
            # Assuming threshold of 0.5 for business value calculation
            decile_y_pred = (decile_y_pred_proba >= 0.5).astype(int)
            tn = np.sum((decile_y_true == 0) & (decile_y_pred == 0))
            fp = np.sum((decile_y_true == 0) & (decile_y_pred == 1))
            fn = np.sum((decile_y_true == 1) & (decile_y_pred == 0))
            tp = np.sum((decile_y_true == 1) & (decile_y_pred == 1))
            
            business_value = (tp * self.business_value_matrix['TP'] +
                            tn * self.business_value_matrix['TN'] +
                            fp * self.business_value_matrix['FP'] +
                            fn * self.business_value_matrix['FN'])
            
            decile_data = {
                'decile': decile,
                'n_samples': n_samples,
                'n_positives': n_positives,
                'n_negatives': n_negatives,
                'positive_rate': positive_rate,
                'capture_rate': capture_rate,
                'min_prob': min_prob,
                'max_prob': max_prob,
                'avg_prob': avg_prob,
                'business_value': business_value,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
            
            # Add monetary metrics if available
            if decile_monetary is not None:
                decile_data['total_monetary_value'] = np.sum(decile_monetary)
                decile_data['avg_monetary_value'] = np.mean(decile_monetary)
                decile_data['monetary_capture'] = np.sum(decile_monetary[decile_y_true == 1])
            
            decile_metrics.append(decile_data)
        
        return pd.DataFrame(decile_metrics)
    
    def generate_calibration_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                n_bins: int = 10, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate calibration plot and calculate calibration metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            save_path: Path to save plot
            
        Returns:
            Dictionary with calibration metrics and plot data
        """
        logger.info("Generating calibration plot")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        max_calibration_error = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Brier Score decomposition
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Calculate reliability (calibration) and resolution
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0
        resolution = 0
        base_rate = np.mean(y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                
                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                resolution += prop_in_bin * (accuracy_in_bin - base_rate) ** 2
        
        uncertainty = base_rate * (1 - base_rate)
        
        # Create calibration plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label=f'Model (ECE: {calibration_error:.3f})')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of predicted probabilities
        ax2.hist(y_pred_proba, bins=50, alpha=0.7, density=True)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration plot saved to: {save_path}")
        
        plt.show()
        
        calibration_metrics = {
            'calibration_error': calibration_error,
            'max_calibration_error': max_calibration_error,
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'n_bins': n_bins
        }
        
        return calibration_metrics
    
    def generate_roc_pr_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ROC and Precision-Recall curves.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
            
        Returns:
            Dictionary with curve data and AUC values
        """
        logger.info("Generating ROC and PR curves")
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Calculate PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        baseline_precision = np.sum(y_true) / len(y_true)
        ax2.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax2.axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.5,
                   label=f'Baseline (No Skill = {baseline_precision:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC/PR curves saved to: {save_path}")
        
        plt.show()
        
        curve_data = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_thresholds': roc_thresholds.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'pr_thresholds': pr_thresholds.tolist(),
            'baseline_precision': baseline_precision
        }
        
        return curve_data
    
    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1', 
                          business_value_matrix: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize classification threshold for specified metric.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'business_value')
            business_value_matrix: Business values for TP, TN, FP, FN
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing threshold for {metric} metric")
        
        bvm = business_value_matrix or self.business_value_matrix
        optimizer = ThresholdOptimizer(metric=metric, business_value_matrix=bvm)
        
        return optimizer.optimize_threshold(y_true, y_pred_proba)
    
    def calculate_lift_and_gain(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               n_deciles: int = 10) -> pd.DataFrame:
        """
        Calculate lift and gain charts data.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_deciles: Number of deciles for analysis
            
        Returns:
            DataFrame with lift and gain metrics
        """
        logger.info("Calculating lift and gain metrics")
        
        # Create dataset
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        })
        
        # Sort by predicted probability (descending)
        df = df.sort_values('y_pred_proba', ascending=False).reset_index(drop=True)
        
        # Create deciles
        df['decile'] = pd.qcut(df.index, q=n_deciles, labels=False) + 1
        
        # Calculate metrics
        total_positives = df['y_true'].sum()
        total_samples = len(df)
        baseline_rate = total_positives / total_samples
        
        lift_gain_data = []
        
        for decile in range(1, n_deciles + 1):
            # Cumulative metrics (top X deciles)
            top_deciles = df[df['decile'] <= decile]
            
            cum_samples = len(top_deciles)
            cum_positives = top_deciles['y_true'].sum()
            cum_positives_rate = cum_positives / cum_samples if cum_samples > 0 else 0
            
            # Lift and gain
            lift = cum_positives_rate / baseline_rate if baseline_rate > 0 else 0
            gain = cum_positives / total_positives if total_positives > 0 else 0
            
            # Current decile metrics
            current_decile = df[df['decile'] == decile]
            decile_samples = len(current_decile)
            decile_positives = current_decile['y_true'].sum()
            decile_rate = decile_positives / decile_samples if decile_samples > 0 else 0
            decile_lift = decile_rate / baseline_rate if baseline_rate > 0 else 0
            
            lift_gain_data.append({
                'decile': decile,
                'cum_samples': cum_samples,
                'cum_positives': cum_positives,
                'cum_positive_rate': cum_positives_rate,
                'cum_lift': lift,
                'cum_gain': gain,
                'decile_samples': decile_samples,
                'decile_positives': decile_positives,
                'decile_positive_rate': decile_rate,
                'decile_lift': decile_lift,
                'population_pct': cum_samples / total_samples
            })
        
        return pd.DataFrame(lift_gain_data)
    
    def plot_lift_and_gain_charts(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot lift and gain charts.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        lift_gain_df = self.calculate_lift_and_gain(y_true, y_pred_proba)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cumulative Gain Chart
        ax1.plot(lift_gain_df['population_pct'] * 100, lift_gain_df['cum_gain'] * 100, 
                'o-', linewidth=2, label='Model')
        ax1.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random Model')
        ax1.set_xlabel('Population Percentage (%)')
        ax1.set_ylabel('Cumulative Gain (%)')
        ax1.set_title('Cumulative Gain Chart')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lift Chart
        ax2.plot(lift_gain_df['decile'], lift_gain_df['decile_lift'], 
                'o-', linewidth=2, label='Decile Lift')
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Baseline (Lift = 1)')
        ax2.set_xlabel('Decile')
        ax2.set_ylabel('Lift')
        ax2.set_title('Lift Chart by Decile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Lift and gain charts saved to: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    monetary_values: Optional[np.ndarray] = None,
                                    output_dir: str = "./metrics_report") -> Dict[str, Any]:
        """
        Generate comprehensive metrics report with all analyses.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            monetary_values: Optional monetary values for business metrics
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with all metrics and analysis results
        """
        logger.info("Generating comprehensive metrics report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {}
        
        try:
            if self.task_type == 'classification':
                # Basic classification metrics
                report['classification_metrics'] = self.calculate_classification_metrics(
                    y_true, y_pred, y_pred_proba
                )
                
                if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                    # ROC and PR curves
                    roc_pr_path = output_path / "roc_pr_curves.png"
                    report['roc_pr_curves'] = self.generate_roc_pr_curves(
                        y_true, y_pred_proba, save_path=str(roc_pr_path)
                    )
                    
                    # Calibration analysis
                    calibration_path = output_path / "calibration_plot.png"
                    report['calibration_metrics'] = self.generate_calibration_plot(
                        y_true, y_pred_proba, save_path=str(calibration_path)
                    )
                    
                    # Business metrics by decile
                    report['business_metrics_by_decile'] = self.calculate_business_metrics_by_decile(
                        y_true, y_pred_proba, monetary_values
                    )
                    
                    # Lift and gain analysis
                    lift_gain_path = output_path / "lift_gain_charts.png"
                    self.plot_lift_and_gain_charts(y_true, y_pred_proba, save_path=str(lift_gain_path))
                    report['lift_gain_data'] = self.calculate_lift_and_gain(y_true, y_pred_proba)
                    
                    # Threshold optimization
                    report['threshold_optimization'] = {}
                    for metric in ['f1', 'precision', 'recall', 'business_value']:
                        opt_result = self.optimize_threshold(y_true, y_pred_proba, metric=metric)
                        report['threshold_optimization'][metric] = opt_result
                    
                    # Threshold analysis plot
                    threshold_path = output_path / "threshold_analysis.png"
                    optimizer = ThresholdOptimizer(metric='f1', business_value_matrix=self.business_value_matrix)
                    optimizer.plot_threshold_analysis(y_true, y_pred_proba, save_path=str(threshold_path))
                
            elif self.task_type == 'regression':
                # Regression metrics
                report['regression_metrics'] = self.calculate_regression_metrics(y_true, y_pred)
                
                # Residual analysis plot
                residuals = y_true - y_pred
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                
                # Residuals vs predicted
                ax1.scatter(y_pred, residuals, alpha=0.6)
                ax1.axhline(y=0, color='red', linestyle='--')
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Residuals')
                ax1.set_title('Residuals vs Predicted')
                ax1.grid(True, alpha=0.3)
                
                # Q-Q plot
                if SCIPY_AVAILABLE:
                    stats.probplot(residuals, dist="norm", plot=ax2)
                    ax2.set_title('Q-Q Plot of Residuals')
                
                # Histogram of residuals
                ax3.hist(residuals, bins=30, density=True, alpha=0.7)
                ax3.set_xlabel('Residuals')
                ax3.set_ylabel('Density')
                ax3.set_title('Distribution of Residuals')
                ax3.grid(True, alpha=0.3)
                
                # Actual vs predicted
                ax4.scatter(y_true, y_pred, alpha=0.6)
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax4.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
                ax4.set_xlabel('Actual Values')
                ax4.set_ylabel('Predicted Values')
                ax4.set_title('Actual vs Predicted')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                residual_path = output_path / "residual_analysis.png"
                plt.savefig(residual_path, dpi=300, bbox_inches='tight')
                plt.show()
            
            # Save summary report
            summary_path = output_path / "metrics_summary.json"
            with open(summary_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_report = self._convert_numpy_for_json(report)
                json.dump(json_report, f, indent=2, default=str)
            
            logger.info(f"Comprehensive metrics report generated in: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise MetricsError(f"Report generation failed: {e}")
        
        return report
    
    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
    
    def calculate_model_stability_metrics(self, y_true_train: np.ndarray, 
                                        y_pred_proba_train: np.ndarray,
                                        y_true_test: np.ndarray, 
                                        y_pred_proba_test: np.ndarray) -> Dict[str, Any]:
        """
        Calculate model stability metrics between train and test sets.
        
        Args:
            y_true_train: Training set true labels
            y_pred_proba_train: Training set predicted probabilities
            y_true_test: Test set true labels
            y_pred_proba_test: Test set predicted probabilities
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info("Calculating model stability metrics")
        
        stability_metrics = {}
        
        # AUC difference
        train_auc = roc_auc_score(y_true_train, y_pred_proba_train)
        test_auc = roc_auc_score(y_true_test, y_pred_proba_test)
        auc_diff = abs(train_auc - test_auc)
        
        stability_metrics['train_auc'] = train_auc
        stability_metrics['test_auc'] = test_auc
        stability_metrics['auc_difference'] = auc_diff
        stability_metrics['auc_stability'] = 1 - auc_diff  # Higher is more stable
        
        # KS statistic difference
        train_ks = self.calculate_ks_statistic(y_true_train, y_pred_proba_train)
        test_ks = self.calculate_ks_statistic(y_true_test, y_pred_proba_test)
        ks_diff = abs(train_ks - test_ks)
        
        stability_metrics['train_ks'] = train_ks
        stability_metrics['test_ks'] = test_ks
        stability_metrics['ks_difference'] = ks_diff
        stability_metrics['ks_stability'] = 1 - ks_diff
        
        # Score distribution similarity (KS test between predicted probabilities)
        if SCIPY_AVAILABLE:
            score_ks_stat, score_ks_pvalue = ks_2samp(y_pred_proba_train, y_pred_proba_test)
            stability_metrics['score_distribution_ks'] = score_ks_stat
            stability_metrics['score_distribution_pvalue'] = score_ks_pvalue
            stability_metrics['score_distribution_stable'] = score_ks_pvalue > 0.05
        
        # Population Stability Index (PSI)
        psi = self._calculate_psi(y_pred_proba_train, y_pred_proba_test)
        stability_metrics['psi'] = psi
        stability_metrics['psi_stable'] = psi < 0.1  # PSI < 0.1 indicates stable model
        
        return stability_metrics
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                      buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Expected distribution (e.g., training set scores)
            actual: Actual distribution (e.g., test set scores)
            buckets: Number of buckets for discretization
            
        Returns:
            PSI value
        """
        # Create buckets based on expected distribution
        bucket_boundaries = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        bucket_boundaries[-1] = bucket_boundaries[-1] + 1e-6  # Ensure last boundary includes max value
        
        # Calculate distributions
        expected_dist = np.histogram(expected, bins=bucket_boundaries)[0] / len(expected)
        actual_dist = np.histogram(actual, bins=bucket_boundaries)[0] / len(actual)
        
        # Avoid division by zero
        expected_dist = np.where(expected_dist == 0, 1e-6, expected_dist)
        actual_dist = np.where(actual_dist == 0, 1e-6, actual_dist)
        
        # Calculate PSI
        psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
        
        return psi