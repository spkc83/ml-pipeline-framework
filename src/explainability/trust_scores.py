"""
Trust Score Calculator for ML Pipeline Framework

This module implements trust scores for individual predictions, uncertainty quantification,
out-of-distribution detection, and reliability analysis for trustworthy ML systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class TrustScoreConfig:
    """Configuration for trust score calculation."""
    
    # Trust score parameters
    k_neighbors: int = 10
    alpha: float = 0.0  # Smoothing parameter for trust scores
    filter_outliers: bool = True
    
    # Uncertainty quantification methods
    enable_epistemic_uncertainty: bool = True
    enable_aleatoric_uncertainty: bool = True
    enable_ensemble_uncertainty: bool = False
    
    # Out-of-distribution detection
    ood_method: str = 'isolation_forest'  # 'isolation_forest', 'mahalanobis', 'knn'
    ood_contamination: float = 0.1
    
    # Calibration parameters
    calibration_method: str = 'isotonic'  # 'isotonic', 'platt'
    calibration_cv_folds: int = 5
    
    # Reliability diagram parameters
    reliability_bins: int = 10
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.6


@dataclass
class TrustScoreResult:
    """Result container for trust score analysis."""
    
    # Individual prediction trust scores
    trust_scores: np.ndarray
    
    # Uncertainty estimates
    epistemic_uncertainty: Optional[np.ndarray] = None
    aleatoric_uncertainty: Optional[np.ndarray] = None
    total_uncertainty: Optional[np.ndarray] = None
    
    # Out-of-distribution scores
    ood_scores: Optional[np.ndarray] = None
    ood_predictions: Optional[np.ndarray] = None
    
    # Confidence-aware predictions
    prediction_confidence: np.ndarray = None
    confidence_intervals: Optional[np.ndarray] = None
    
    # Calibration metrics
    calibration_error: float = 0.0
    reliability_diagram_data: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Summary statistics
    summary_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReliabilityAnalysis:
    """Container for model reliability analysis."""
    
    # Calibration metrics
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    
    # Reliability diagram data
    bin_boundaries: np.ndarray
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_counts: np.ndarray
    
    # Confidence distribution
    confidence_histogram: Dict[str, np.ndarray]
    
    # Performance by confidence level
    performance_by_confidence: Dict[str, Dict[str, float]]


class TrustScoreCalculator:
    """
    Comprehensive trust score calculator for ML model predictions.
    
    Based on "To Trust Or Not To Trust A Classifier" (Jiang et al., 2018)
    with extensions for uncertainty quantification and reliability analysis.
    """
    
    def __init__(self, config: TrustScoreConfig = None):
        """
        Initialize trust score calculator.
        
        Args:
            config: Trust score configuration
        """
        self.config = config or TrustScoreConfig()
        self.is_fitted = False
        
        # Fitted components
        self.scaler = None
        self.nn_same_class = None
        self.nn_diff_class = None
        self.ood_detector = None
        self.calibrator = None
        
        # Training data statistics
        self.class_densities = {}
        self.training_features = None
        self.training_labels = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            model: Any = None) -> 'TrustScoreCalculator':
        """
        Fit trust score calculator on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model: Trained model (optional, for some uncertainty methods)
            
        Returns:
            Fitted TrustScoreCalculator
        """
        logger.info("Fitting trust score calculator")
        
        self.training_features = X_train.copy()
        self.training_labels = y_train.copy()
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Fit nearest neighbor models for each class
        unique_classes = np.unique(y_train)
        self.nn_same_class = {}
        self.nn_diff_class = {}
        
        for class_label in unique_classes:
            # Same class neighbors
            same_class_mask = (y_train == class_label)
            if np.sum(same_class_mask) > self.config.k_neighbors:
                self.nn_same_class[class_label] = NearestNeighbors(
                    n_neighbors=min(self.config.k_neighbors, np.sum(same_class_mask))
                )
                self.nn_same_class[class_label].fit(X_scaled[same_class_mask])
            
            # Different class neighbors
            diff_class_mask = (y_train != class_label)
            if np.sum(diff_class_mask) > self.config.k_neighbors:
                self.nn_diff_class[class_label] = NearestNeighbors(
                    n_neighbors=min(self.config.k_neighbors, np.sum(diff_class_mask))
                )
                self.nn_diff_class[class_label].fit(X_scaled[diff_class_mask])
        
        # Fit out-of-distribution detector
        self._fit_ood_detector(X_scaled)
        
        # Calculate class densities for uncertainty estimation
        self._calculate_class_densities(X_scaled, y_train)
        
        self.is_fitted = True
        logger.info("Trust score calculator fitted successfully")
        
        return self
    
    def calculate_trust_scores(self, X_test: np.ndarray, y_pred: np.ndarray,
                             y_pred_proba: np.ndarray = None) -> TrustScoreResult:
        """
        Calculate trust scores for test predictions.
        
        Args:
            X_test: Test features
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            TrustScoreResult object
        """
        if not self.is_fitted:
            raise ValueError("TrustScoreCalculator must be fitted before calculating trust scores")
        
        logger.info(f"Calculating trust scores for {len(X_test)} samples")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate basic trust scores
        trust_scores = self._calculate_basic_trust_scores(X_test_scaled, y_pred)
        
        # Calculate uncertainty estimates
        uncertainty_results = self._calculate_uncertainty(X_test_scaled, y_pred, y_pred_proba)
        
        # Out-of-distribution detection
        ood_scores, ood_predictions = self._detect_out_of_distribution(X_test_scaled)
        
        # Confidence-aware predictions
        confidence_results = self._calculate_confidence_aware_predictions(
            y_pred, y_pred_proba, trust_scores
        )
        
        # Summary statistics
        summary_stats = self._calculate_summary_statistics(
            trust_scores, uncertainty_results, ood_scores
        )
        
        return TrustScoreResult(
            trust_scores=trust_scores,
            epistemic_uncertainty=uncertainty_results.get('epistemic'),
            aleatoric_uncertainty=uncertainty_results.get('aleatoric'),
            total_uncertainty=uncertainty_results.get('total'),
            ood_scores=ood_scores,
            ood_predictions=ood_predictions,
            prediction_confidence=confidence_results['confidence'],
            confidence_intervals=confidence_results.get('intervals'),
            summary_stats=summary_stats
        )
    
    def analyze_reliability(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          y_pred: np.ndarray = None) -> ReliabilityAnalysis:
        """
        Analyze model reliability and calibration.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            y_pred: Predicted labels (optional)
            
        Returns:
            ReliabilityAnalysis object
        """
        logger.info("Analyzing model reliability and calibration")
        
        if y_pred is None:
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(y_true, y_pred_proba)
        
        # Generate reliability diagram data
        reliability_data = self._generate_reliability_diagram_data(y_true, y_pred_proba)
        
        # Analyze confidence distribution
        confidence_analysis = self._analyze_confidence_distribution(y_pred_proba)
        
        # Performance by confidence level
        performance_by_confidence = self._analyze_performance_by_confidence(
            y_true, y_pred, y_pred_proba
        )
        
        return ReliabilityAnalysis(
            expected_calibration_error=calibration_metrics['ece'],
            maximum_calibration_error=calibration_metrics['mce'],
            brier_score=calibration_metrics['brier_score'],
            bin_boundaries=reliability_data['bin_boundaries'],
            bin_confidences=reliability_data['bin_confidences'],
            bin_accuracies=reliability_data['bin_accuracies'],
            bin_counts=reliability_data['bin_counts'],
            confidence_histogram=confidence_analysis,
            performance_by_confidence=performance_by_confidence
        )
    
    def visualize_reliability_diagram(self, reliability_analysis: ReliabilityAnalysis,
                                    output_path: str = None) -> plt.Figure:
        """
        Create reliability diagram visualization.
        
        Args:
            reliability_analysis: ReliabilityAnalysis object
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Reliability Analysis', fontsize=16, fontweight='bold')
        
        # Reliability diagram
        ax1 = axes[0, 0]
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.plot(reliability_analysis.bin_confidences, reliability_analysis.bin_accuracies, 
                'o-', color='red', linewidth=2, markersize=8, label='Model Calibration')
        
        # Add confidence intervals (error bars based on bin counts)
        bin_errors = np.sqrt(reliability_analysis.bin_accuracies * 
                           (1 - reliability_analysis.bin_accuracies) / 
                           np.maximum(reliability_analysis.bin_counts, 1))
        ax1.errorbar(reliability_analysis.bin_confidences, reliability_analysis.bin_accuracies,
                    yerr=bin_errors, fmt='none', color='red', alpha=0.5)
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Reliability Diagram\nECE: {reliability_analysis.expected_calibration_error:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2 = axes[0, 1]
        bins = reliability_analysis.confidence_histogram['bins']
        counts = reliability_analysis.confidence_histogram['counts']
        ax2.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Number of Predictions')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Performance by confidence level
        ax3 = axes[1, 0]
        conf_levels = list(reliability_analysis.performance_by_confidence.keys())
        accuracies = [reliability_analysis.performance_by_confidence[level]['accuracy'] 
                     for level in conf_levels]
        
        ax3.bar(range(len(conf_levels)), accuracies, alpha=0.7)
        ax3.set_xticks(range(len(conf_levels)))
        ax3.set_xticklabels(conf_levels, rotation=45)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy by Confidence Level')
        ax3.grid(True, alpha=0.3)
        
        # Calibration error by bin
        ax4 = axes[1, 1]
        calibration_errors = np.abs(reliability_analysis.bin_confidences - 
                                   reliability_analysis.bin_accuracies)
        ax4.bar(range(len(calibration_errors)), calibration_errors, alpha=0.7, color='orange')
        ax4.set_xlabel('Confidence Bin')
        ax4.set_ylabel('Calibration Error')
        ax4.set_title(f'Calibration Error by Bin\nMCE: {reliability_analysis.maximum_calibration_error:.4f}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {output_path}")
        
        return fig
    
    def create_trust_score_dashboard(self, trust_result: TrustScoreResult,
                                   output_path: str = None) -> plt.Figure:
        """
        Create comprehensive trust score dashboard.
        
        Args:
            trust_result: TrustScoreResult object
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trust Score Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Trust score distribution
        ax1 = axes[0, 0]
        ax1.hist(trust_result.trust_scores, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Trust Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Trust Score Distribution')
        ax1.axvline(np.mean(trust_result.trust_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(trust_result.trust_scores):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty vs Trust Score
        ax2 = axes[0, 1]
        if trust_result.total_uncertainty is not None:
            ax2.scatter(trust_result.trust_scores, trust_result.total_uncertainty, 
                       alpha=0.6, s=20)
            ax2.set_xlabel('Trust Score')
            ax2.set_ylabel('Total Uncertainty')
            ax2.set_title('Trust Score vs Uncertainty')
            
            # Add correlation coefficient
            corr = np.corrcoef(trust_result.trust_scores, trust_result.total_uncertainty)[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'Uncertainty data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Trust Score vs Uncertainty')
        ax2.grid(True, alpha=0.3)
        
        # Out-of-distribution scores
        ax3 = axes[0, 2]
        if trust_result.ood_scores is not None:
            ax3.hist(trust_result.ood_scores, bins=30, alpha=0.7, edgecolor='black', color='orange')
            ax3.set_xlabel('OOD Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Out-of-Distribution Scores')
            
            # Mark potential OOD samples
            ood_threshold = np.percentile(trust_result.ood_scores, 10)
            ax3.axvline(ood_threshold, color='red', linestyle='--', 
                       label=f'OOD Threshold: {ood_threshold:.3f}')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'OOD scores\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Out-of-Distribution Scores')
        ax3.grid(True, alpha=0.3)
        
        # Confidence distribution
        ax4 = axes[1, 0]
        if trust_result.prediction_confidence is not None:
            ax4.hist(trust_result.prediction_confidence, bins=30, alpha=0.7, 
                    edgecolor='black', color='green')
            ax4.set_xlabel('Prediction Confidence')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Prediction Confidence Distribution')
            
            # Mark confidence thresholds
            ax4.axvline(self.config.high_confidence_threshold, color='red', linestyle='--',
                       label=f'High Conf: {self.config.high_confidence_threshold}')
            ax4.axvline(self.config.low_confidence_threshold, color='orange', linestyle='--',
                       label=f'Low Conf: {self.config.low_confidence_threshold}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Confidence data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Prediction Confidence Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Trust score vs Confidence
        ax5 = axes[1, 1]
        if trust_result.prediction_confidence is not None:
            ax5.scatter(trust_result.trust_scores, trust_result.prediction_confidence,
                       alpha=0.6, s=20, color='purple')
            ax5.set_xlabel('Trust Score')
            ax5.set_ylabel('Prediction Confidence')
            ax5.set_title('Trust Score vs Confidence')
            
            # Add correlation
            corr = np.corrcoef(trust_result.trust_scores, trust_result.prediction_confidence)[0, 1]
            ax5.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax5.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax5.text(0.5, 0.5, 'Confidence data\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Trust Score vs Confidence')
        ax5.grid(True, alpha=0.3)
        
        # Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
        Trust Score Statistics:
        Mean: {trust_result.summary_stats.get('trust_score_mean', 'N/A'):.3f}
        Std: {trust_result.summary_stats.get('trust_score_std', 'N/A'):.3f}
        Min: {trust_result.summary_stats.get('trust_score_min', 'N/A'):.3f}
        Max: {trust_result.summary_stats.get('trust_score_max', 'N/A'):.3f}
        
        High Trust Samples: {trust_result.summary_stats.get('high_trust_ratio', 'N/A'):.1%}
        Low Trust Samples: {trust_result.summary_stats.get('low_trust_ratio', 'N/A'):.1%}
        
        OOD Samples: {trust_result.summary_stats.get('ood_ratio', 'N/A'):.1%}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trust score dashboard saved to {output_path}")
        
        return fig
    
    def get_confidence_aware_predictions(self, X_test: np.ndarray, model: Any,
                                       confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Generate confidence-aware predictions with uncertainty estimates.
        
        Args:
            X_test: Test features
            model: Trained model
            confidence_threshold: Threshold for high-confidence predictions
            
        Returns:
            Dict with confidence-aware predictions and metadata
        """
        confidence_threshold = confidence_threshold or self.config.high_confidence_threshold
        
        # Get base predictions
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        
        # Calculate trust scores
        trust_result = self.calculate_trust_scores(X_test, y_pred, y_pred_proba)
        
        # Identify high/low confidence predictions
        high_confidence_mask = trust_result.prediction_confidence >= confidence_threshold
        low_confidence_mask = trust_result.prediction_confidence < self.config.low_confidence_threshold
        
        # Identify out-of-distribution samples
        ood_mask = None
        if trust_result.ood_scores is not None:
            ood_threshold = np.percentile(trust_result.ood_scores, 10)
            ood_mask = trust_result.ood_scores < ood_threshold
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'trust_scores': trust_result.trust_scores,
            'prediction_confidence': trust_result.prediction_confidence,
            'high_confidence_mask': high_confidence_mask,
            'low_confidence_mask': low_confidence_mask,
            'ood_mask': ood_mask,
            'uncertainty': trust_result.total_uncertainty,
            'summary': {
                'total_samples': len(X_test),
                'high_confidence_count': np.sum(high_confidence_mask),
                'low_confidence_count': np.sum(low_confidence_mask),
                'ood_count': np.sum(ood_mask) if ood_mask is not None else 0,
                'mean_trust_score': np.mean(trust_result.trust_scores),
                'mean_confidence': np.mean(trust_result.prediction_confidence)
            }
        }
    
    def _calculate_basic_trust_scores(self, X_test_scaled: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate basic trust scores using nearest neighbor approach."""
        trust_scores = np.zeros(len(X_test_scaled))
        
        for i, (x, pred_class) in enumerate(zip(X_test_scaled, y_pred)):
            # Distance to nearest same-class examples
            d_same = float('inf')
            if pred_class in self.nn_same_class:
                distances_same, _ = self.nn_same_class[pred_class].kneighbors([x])
                d_same = np.mean(distances_same[0])
            
            # Distance to nearest different-class examples
            d_diff = float('inf')
            if pred_class in self.nn_diff_class:
                distances_diff, _ = self.nn_diff_class[pred_class].kneighbors([x])
                d_diff = np.mean(distances_diff[0])
            
            # Trust score calculation
            if d_same == float('inf') and d_diff == float('inf'):
                trust_scores[i] = 0.5  # Default when no neighbors available
            elif d_same == float('inf'):
                trust_scores[i] = 0.0  # No same-class neighbors
            elif d_diff == float('inf'):
                trust_scores[i] = 1.0  # No different-class neighbors
            else:
                # Original trust score formula with smoothing
                trust_scores[i] = d_diff / (d_same + d_diff + self.config.alpha)
        
        return trust_scores
    
    def _calculate_uncertainty(self, X_test_scaled: np.ndarray, y_pred: np.ndarray,
                             y_pred_proba: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Calculate various uncertainty estimates."""
        uncertainty_results = {}
        
        if y_pred_proba is not None:
            # Aleatoric uncertainty (entropy of predictions)
            if self.config.enable_aleatoric_uncertainty:
                entropy_uncertainty = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-10), axis=1)
                uncertainty_results['aleatoric'] = entropy_uncertainty
            
            # Epistemic uncertainty (variation in predictions)
            if self.config.enable_epistemic_uncertainty:
                # Simplified epistemic uncertainty based on confidence
                max_proba = np.max(y_pred_proba, axis=1)
                epistemic_uncertainty = 1 - max_proba
                uncertainty_results['epistemic'] = epistemic_uncertainty
            
            # Total uncertainty
            if 'aleatoric' in uncertainty_results and 'epistemic' in uncertainty_results:
                uncertainty_results['total'] = (uncertainty_results['aleatoric'] + 
                                               uncertainty_results['epistemic']) / 2
            elif 'aleatoric' in uncertainty_results:
                uncertainty_results['total'] = uncertainty_results['aleatoric']
            elif 'epistemic' in uncertainty_results:
                uncertainty_results['total'] = uncertainty_results['epistemic']
        
        return uncertainty_results
    
    def _fit_ood_detector(self, X_scaled: np.ndarray):
        """Fit out-of-distribution detector."""
        if self.config.ood_method == 'isolation_forest':
            self.ood_detector = IsolationForest(
                contamination=self.config.ood_contamination,
                random_state=42
            )
            self.ood_detector.fit(X_scaled)
        
        elif self.config.ood_method == 'mahalanobis':
            # Store statistics for Mahalanobis distance
            self.ood_detector = {
                'mean': np.mean(X_scaled, axis=0),
                'cov_inv': np.linalg.pinv(np.cov(X_scaled.T))
            }
        
        elif self.config.ood_method == 'knn':
            self.ood_detector = NearestNeighbors(n_neighbors=self.config.k_neighbors)
            self.ood_detector.fit(X_scaled)
    
    def _detect_out_of_distribution(self, X_test_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect out-of-distribution samples."""
        if self.ood_detector is None:
            return None, None
        
        if self.config.ood_method == 'isolation_forest':
            ood_scores = self.ood_detector.decision_function(X_test_scaled)
            ood_predictions = self.ood_detector.predict(X_test_scaled)
            return ood_scores, ood_predictions
        
        elif self.config.ood_method == 'mahalanobis':
            # Calculate Mahalanobis distances
            ood_scores = np.array([
                mahalanobis(x, self.ood_detector['mean'], self.ood_detector['cov_inv'])
                for x in X_test_scaled
            ])
            # Convert to outlier predictions (higher distance = more likely OOD)
            threshold = np.percentile(ood_scores, (1 - self.config.ood_contamination) * 100)
            ood_predictions = (ood_scores > threshold).astype(int)
            return -ood_scores, ood_predictions  # Negative for consistency with isolation forest
        
        elif self.config.ood_method == 'knn':
            # Use average distance to k nearest neighbors as OOD score
            distances, _ = self.ood_detector.kneighbors(X_test_scaled)
            ood_scores = np.mean(distances, axis=1)
            threshold = np.percentile(ood_scores, (1 - self.config.ood_contamination) * 100)
            ood_predictions = (ood_scores > threshold).astype(int)
            return -ood_scores, ood_predictions  # Negative for consistency
        
        return None, None
    
    def _calculate_confidence_aware_predictions(self, y_pred: np.ndarray, 
                                              y_pred_proba: np.ndarray = None,
                                              trust_scores: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Calculate confidence-aware predictions."""
        results = {}
        
        if y_pred_proba is not None:
            # Use maximum probability as confidence
            confidence = np.max(y_pred_proba, axis=1)
        elif trust_scores is not None:
            # Use trust scores as confidence proxy
            confidence = trust_scores
        else:
            # Default uniform confidence
            confidence = np.ones(len(y_pred)) * 0.5
        
        results['confidence'] = confidence
        
        # Calculate confidence intervals (simplified)
        if y_pred_proba is not None and len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
            # For binary classification, use prediction probability as interval
            lower_bound = y_pred_proba[:, 1] - np.sqrt(y_pred_proba[:, 1] * (1 - y_pred_proba[:, 1]) / 100)
            upper_bound = y_pred_proba[:, 1] + np.sqrt(y_pred_proba[:, 1] * (1 - y_pred_proba[:, 1]) / 100)
            results['intervals'] = np.column_stack([lower_bound, upper_bound])
        
        return results
    
    def _calculate_class_densities(self, X_scaled: np.ndarray, y_train: np.ndarray):
        """Calculate class densities for uncertainty estimation."""
        unique_classes = np.unique(y_train)
        
        for class_label in unique_classes:
            class_mask = (y_train == class_label)
            class_data = X_scaled[class_mask]
            
            if len(class_data) > 1:
                # Simple density estimation using covariance
                self.class_densities[class_label] = {
                    'mean': np.mean(class_data, axis=0),
                    'cov': np.cov(class_data.T) + np.eye(class_data.shape[1]) * 1e-6  # Regularization
                }
    
    def _calculate_summary_statistics(self, trust_scores: np.ndarray,
                                    uncertainty_results: Dict[str, np.ndarray],
                                    ood_scores: np.ndarray = None) -> Dict[str, float]:
        """Calculate summary statistics for trust scores and uncertainty."""
        stats = {
            'trust_score_mean': np.mean(trust_scores),
            'trust_score_std': np.std(trust_scores),
            'trust_score_min': np.min(trust_scores),
            'trust_score_max': np.max(trust_scores),
            'high_trust_ratio': np.mean(trust_scores > 0.7),
            'low_trust_ratio': np.mean(trust_scores < 0.3)
        }
        
        if 'total' in uncertainty_results:
            uncertainty = uncertainty_results['total']
            stats.update({
                'uncertainty_mean': np.mean(uncertainty),
                'uncertainty_std': np.std(uncertainty),
                'high_uncertainty_ratio': np.mean(uncertainty > np.percentile(uncertainty, 80))
            })
        
        if ood_scores is not None:
            ood_threshold = np.percentile(ood_scores, 10)
            stats['ood_ratio'] = np.mean(ood_scores < ood_threshold)
        
        return stats
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive calibration metrics."""
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.config.reliability_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score
        }
    
    def _generate_reliability_diagram_data(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate data for reliability diagram."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=self.config.reliability_bins
        )
        
        # Calculate bin counts
        bin_counts = np.zeros(self.config.reliability_bins)
        bin_boundaries = np.linspace(0, 1, self.config.reliability_bins + 1)
        
        for i in range(self.config.reliability_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            bin_counts[i] = np.sum(in_bin)
        
        return {
            'bin_boundaries': bin_boundaries,
            'bin_confidences': mean_predicted_value,
            'bin_accuracies': fraction_of_positives,
            'bin_counts': bin_counts
        }
    
    def _analyze_confidence_distribution(self, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze confidence distribution."""
        bins = np.linspace(0, 1, 21)  # 20 bins
        counts, bin_edges = np.histogram(y_pred_proba, bins=bins)
        
        return {
            'bins': bin_edges,
            'counts': counts
        }
    
    def _analyze_performance_by_confidence(self, y_true: np.ndarray, y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Analyze performance metrics by confidence level."""
        confidence_levels = {
            'Very Low (0.0-0.6)': (0.0, 0.6),
            'Low (0.6-0.7)': (0.6, 0.7),
            'Medium (0.7-0.8)': (0.7, 0.8),
            'High (0.8-0.9)': (0.8, 0.9),
            'Very High (0.9-1.0)': (0.9, 1.0)
        }
        
        performance_by_level = {}
        
        for level_name, (lower, upper) in confidence_levels.items():
            mask = (y_pred_proba >= lower) & (y_pred_proba < upper)
            
            if np.sum(mask) > 0:
                level_accuracy = np.mean(y_true[mask] == y_pred[mask])
                level_count = np.sum(mask)
                level_precision = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1)) / max(np.sum(y_pred[mask] == 1), 1)
                level_recall = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1)) / max(np.sum(y_true[mask] == 1), 1)
                
                performance_by_level[level_name] = {
                    'accuracy': level_accuracy,
                    'count': level_count,
                    'precision': level_precision,
                    'recall': level_recall
                }
            else:
                performance_by_level[level_name] = {
                    'accuracy': 0.0,
                    'count': 0,
                    'precision': 0.0,
                    'recall': 0.0
                }
        
        return performance_by_level


def create_trust_score_calculator(config_dict: Dict[str, Any] = None) -> TrustScoreCalculator:
    """
    Factory function to create trust score calculator.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        TrustScoreCalculator instance
    """
    config = TrustScoreConfig(**config_dict) if config_dict else TrustScoreConfig()
    return TrustScoreCalculator(config)