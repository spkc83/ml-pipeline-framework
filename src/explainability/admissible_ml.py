"""
Admissible ML Generator for ML Pipeline Framework

This module generates model cards, fairness metrics, bias detection, uncertainty
quantification, and comprehensive documentation following Model Cards for Model Reporting standard.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings
from datetime import datetime
import tempfile

# Fairness and bias detection
try:
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import LabelEncoder
    import scipy.stats as stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    warnings.warn("Statistical libraries not available. Some fairness metrics will be limited.")

# Uncertainty quantification
try:
    from scipy.stats import entropy
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False
    warnings.warn("Uncertainty quantification libraries not available.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelCardData:
    """Data structure for Model Cards following Google's Model Cards standard."""
    
    # Model details
    model_name: str
    model_version: str
    model_type: str
    model_description: str
    model_architecture: str
    
    # Intended use
    intended_use: str
    intended_users: List[str]
    out_of_scope_uses: List[str]
    
    # Factors
    relevant_factors: List[str]
    evaluation_factors: List[str]
    
    # Metrics
    model_performance: Dict[str, float]
    fairness_metrics: Dict[str, Dict[str, float]]
    
    # Training data
    training_data_description: str
    training_data_preprocessing: List[str]
    training_data_size: int
    
    # Evaluation data
    evaluation_data_description: str
    evaluation_data_size: int
    
    # Ethical considerations
    ethical_considerations: List[str]
    
    # Caveats and recommendations
    caveats: List[str]
    recommendations: List[str]
    
    # Technical specifications
    technical_specs: Dict[str, Any] = field(default_factory=dict)
    
    # Quantitative analysis
    quantitative_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FairnessMetrics:
    """Container for comprehensive fairness metrics."""
    
    # Demographic parity metrics
    demographic_parity: Dict[str, float]
    demographic_parity_difference: float
    demographic_parity_ratio: float
    
    # Equalized odds metrics
    equalized_odds_difference: Dict[str, float]
    equalized_odds_ratio: Dict[str, float]
    
    # Calibration metrics
    calibration_by_group: Dict[str, Dict[str, float]]
    calibration_difference: float
    
    # Other fairness metrics
    statistical_parity_difference: float
    equal_opportunity_difference: float
    predictive_equality_difference: float
    
    # Group-specific performance
    group_performance: Dict[str, Dict[str, float]]
    
    # Bias indicators
    bias_indicators: Dict[str, str]


@dataclass
class UncertaintyQuantification:
    """Container for uncertainty quantification results."""
    
    # Prediction uncertainty
    prediction_uncertainty: np.ndarray
    epistemic_uncertainty: Optional[np.ndarray] = None
    aleatoric_uncertainty: Optional[np.ndarray] = None
    
    # Calibration metrics
    calibration_error: float
    reliability_diagram_data: Dict[str, np.ndarray]
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Out-of-distribution detection
    ood_scores: Optional[np.ndarray] = None
    ood_threshold: Optional[float] = None


@dataclass
class BiasDetectionResult:
    """Container for bias detection analysis."""
    
    # Protected attribute analysis
    protected_attributes: List[str]
    bias_detected: Dict[str, bool]
    bias_severity: Dict[str, str]  # 'low', 'medium', 'high'
    
    # Intersectional bias
    intersectional_bias: Dict[str, Dict[str, float]]
    
    # Statistical tests
    statistical_tests: Dict[str, Dict[str, float]]  # p-values and test statistics
    
    # Recommendations
    mitigation_recommendations: List[str]


class AdmissibleMLGenerator:
    """
    Generator for admissible ML components including model cards, fairness metrics,
    and comprehensive documentation.
    """
    
    def __init__(self):
        """Initialize the AdmissibleML generator."""
        self.model_card = None
        self.fairness_metrics = None
        self.uncertainty_metrics = None
        self.bias_detection = None
        
        # Configuration
        self.protected_attributes = ['gender', 'age_group', 'race', 'ethnicity']
        self.fairness_thresholds = {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'calibration': 0.1
        }
    
    def generate_model_card(self, model: Any, model_name: str,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          protected_attributes_data: Dict[str, np.ndarray] = None,
                          additional_info: Dict[str, Any] = None) -> ModelCardData:
        """
        Generate comprehensive model card following Model Cards for Model Reporting standard.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Test features  
            y_test: Test labels
            protected_attributes_data: Dict mapping attribute names to values
            additional_info: Additional model information
            
        Returns:
            ModelCardData object
        """
        logger.info(f"Generating model card for {model_name}")
        
        additional_info = additional_info or {}
        
        # Model performance metrics
        performance_metrics = self._calculate_performance_metrics(model, X_test, y_test)
        
        # Fairness metrics if protected attributes provided
        fairness_metrics = {}
        if protected_attributes_data:
            fairness_metrics = self._calculate_fairness_metrics(
                model, X_test, y_test, protected_attributes_data
            )
        
        # Model details
        model_type = type(model).__name__
        model_description = additional_info.get(
            'description', 
            f"Machine learning model of type {model_type} for classification tasks"
        )
        
        # Determine model architecture
        architecture = self._determine_model_architecture(model)
        
        # Create model card
        model_card = ModelCardData(
            model_name=model_name,
            model_version=additional_info.get('version', '1.0.0'),
            model_type=model_type,
            model_description=model_description,
            model_architecture=architecture,
            
            intended_use=additional_info.get(
                'intended_use', 
                "Classification model for automated decision making"
            ),
            intended_users=additional_info.get(
                'intended_users', 
                ["Data scientists", "Business analysts", "Operations teams"]
            ),
            out_of_scope_uses=[
                "Decisions affecting individual legal rights",
                "High-stakes decisions without human oversight",
                "Use on populations significantly different from training data"
            ],
            
            relevant_factors=[
                "Data quality and completeness",
                "Temporal stability of patterns",
                "Population subgroup differences"
            ],
            evaluation_factors=[
                "Overall model performance",
                "Fairness across demographic groups", 
                "Prediction uncertainty and calibration"
            ],
            
            model_performance=performance_metrics,
            fairness_metrics=fairness_metrics,
            
            training_data_description=additional_info.get(
                'training_data_description',
                f"Training dataset with {len(X_train)} samples and {X_train.shape[1]} features"
            ),
            training_data_preprocessing=additional_info.get(
                'preprocessing_steps',
                ["Feature scaling", "Missing value imputation", "Outlier handling"]
            ),
            training_data_size=len(X_train),
            
            evaluation_data_description=f"Hold-out test set with {len(X_test)} samples",
            evaluation_data_size=len(X_test),
            
            ethical_considerations=[
                "Model may exhibit bias against underrepresented groups",
                "Predictions should not be the sole basis for high-stakes decisions",
                "Regular monitoring for performance degradation is required",
                "Explanation and appeal processes should be available"
            ],
            
            caveats=[
                "Model performance may degrade on out-of-distribution data",
                "Protected attribute fairness requires ongoing monitoring",
                "Model uncertainty should be considered in decision making"
            ],
            
            recommendations=[
                "Implement human oversight for high-impact decisions",
                "Monitor model performance and fairness metrics over time",
                "Provide explanation capabilities for affected individuals",
                "Regular retraining with updated data"
            ],
            
            technical_specs={
                'training_time': additional_info.get('training_time'),
                'inference_time_ms': additional_info.get('inference_time_ms'),
                'model_size_mb': additional_info.get('model_size_mb'),
                'memory_requirements_mb': additional_info.get('memory_requirements_mb'),
                'supported_platforms': additional_info.get('supported_platforms', ['Python', 'scikit-learn'])
            },
            
            quantitative_analysis={
                'performance_metrics': performance_metrics,
                'fairness_assessment': fairness_metrics,
                'confidence_analysis': additional_info.get('confidence_analysis', {})
            }
        )
        
        self.model_card = model_card
        return model_card
    
    def calculate_fairness_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                 protected_attributes: Dict[str, np.ndarray]) -> FairnessMetrics:
        """
        Calculate comprehensive fairness metrics across protected attributes.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            protected_attributes: Dict mapping attribute names to values
            
        Returns:
            FairnessMetrics object
        """
        logger.info("Calculating fairness metrics")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred.astype(float)
        
        # Calculate metrics for each protected attribute
        fairness_results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            fairness_results[attr_name] = self._calculate_attribute_fairness(
                y_test, y_pred, y_pred_proba, attr_values
            )
        
        # Aggregate metrics across all attributes
        demographic_parity = {}
        equalized_odds_diff = {}
        equalized_odds_ratio = {}
        calibration_by_group = {}
        group_performance = {}
        
        for attr_name, metrics in fairness_results.items():
            demographic_parity[attr_name] = metrics['demographic_parity_difference']
            equalized_odds_diff[attr_name] = metrics['equalized_odds_difference']
            equalized_odds_ratio[attr_name] = metrics['equalized_odds_ratio']
            calibration_by_group[attr_name] = metrics['calibration_metrics']
            group_performance[attr_name] = metrics['group_performance']
        
        # Calculate overall fairness indicators
        overall_dp_diff = np.mean(list(demographic_parity.values()))
        overall_dp_ratio = np.mean([metrics['demographic_parity_ratio'] 
                                   for metrics in fairness_results.values()])
        
        # Bias indicators
        bias_indicators = {}
        for attr_name, dp_diff in demographic_parity.items():
            if abs(dp_diff) > self.fairness_thresholds['demographic_parity']:
                bias_indicators[attr_name] = 'High bias detected'
            elif abs(dp_diff) > self.fairness_thresholds['demographic_parity'] * 0.5:
                bias_indicators[attr_name] = 'Moderate bias detected'
            else:
                bias_indicators[attr_name] = 'Low bias'
        
        return FairnessMetrics(
            demographic_parity=demographic_parity,
            demographic_parity_difference=overall_dp_diff,
            demographic_parity_ratio=overall_dp_ratio,
            equalized_odds_difference=equalized_odds_diff,
            equalized_odds_ratio=equalized_odds_ratio,
            calibration_by_group=calibration_by_group,
            calibration_difference=np.mean([
                metrics['calibration_difference'] 
                for metrics in fairness_results.values()
            ]),
            statistical_parity_difference=overall_dp_diff,  # Same as demographic parity
            equal_opportunity_difference=np.mean([
                metrics['equal_opportunity_difference']
                for metrics in fairness_results.values()
            ]),
            predictive_equality_difference=np.mean([
                metrics['predictive_equality_difference']
                for metrics in fairness_results.values()
            ]),
            group_performance=group_performance,
            bias_indicators=bias_indicators
        )
    
    def detect_bias(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                   protected_attributes: Dict[str, np.ndarray],
                   significance_level: float = 0.05) -> BiasDetectionResult:
        """
        Detect bias across protected attributes using statistical tests.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            protected_attributes: Dict mapping attribute names to values
            significance_level: Significance level for statistical tests
            
        Returns:
            BiasDetectionResult object
        """
        logger.info("Detecting bias across protected attributes")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred.astype(float)
        
        bias_detected = {}
        bias_severity = {}
        statistical_tests = {}
        intersectional_bias = {}
        
        # Single attribute bias detection
        for attr_name, attr_values in protected_attributes.items():
            unique_groups = np.unique(attr_values)
            
            if len(unique_groups) < 2:
                continue
            
            # Calculate prediction rates by group
            group_pred_rates = []
            group_performance = []
            
            for group in unique_groups:
                group_mask = (attr_values == group)
                group_pred_rate = np.mean(y_pred[group_mask])
                group_pred_rates.append(group_pred_rate)
                
                # Group-specific performance
                if np.sum(group_mask) > 10:  # Minimum group size
                    group_auc = self._calculate_auc(y_test[group_mask], y_pred_proba[group_mask])
                    group_performance.append(group_auc)
            
            # Statistical test for difference in prediction rates
            if len(group_pred_rates) >= 2:
                # Chi-square test for independence
                contingency_table = self._create_contingency_table(y_pred, attr_values)
                chi2_stat, p_value = stats.chi2_contingency(contingency_table)[:2]
                
                statistical_tests[attr_name] = {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'test_type': 'chi2_independence'
                }
                
                # Bias detection based on p-value and effect size
                bias_detected[attr_name] = p_value < significance_level
                
                # Bias severity based on prediction rate differences
                max_diff = max(group_pred_rates) - min(group_pred_rates)
                if max_diff > 0.2:
                    bias_severity[attr_name] = 'high'
                elif max_diff > 0.1:
                    bias_severity[attr_name] = 'medium'
                else:
                    bias_severity[attr_name] = 'low'
        
        # Intersectional bias detection
        if len(protected_attributes) >= 2:
            intersectional_bias = self._detect_intersectional_bias(
                y_pred, protected_attributes
            )
        
        # Generate mitigation recommendations
        mitigation_recommendations = self._generate_bias_mitigation_recommendations(
            bias_detected, bias_severity
        )
        
        return BiasDetectionResult(
            protected_attributes=list(protected_attributes.keys()),
            bias_detected=bias_detected,
            bias_severity=bias_severity,
            intersectional_bias=intersectional_bias,
            statistical_tests=statistical_tests,
            mitigation_recommendations=mitigation_recommendations
        )
    
    def quantify_uncertainty(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                           X_train: np.ndarray = None) -> UncertaintyQuantification:
        """
        Quantify prediction uncertainty using multiple methods.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (for some uncertainty methods)
            
        Returns:
            UncertaintyQuantification object
        """
        logger.info("Quantifying prediction uncertainty")
        
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support probability prediction. Uncertainty analysis limited.")
            return UncertaintyQuantification(
                prediction_uncertainty=np.zeros(len(X_test)),
                calibration_error=0.0,
                reliability_diagram_data={},
                confidence_intervals={}
            )
        
        y_pred_proba = model.predict_proba(X_test)
        
        # Prediction uncertainty (entropy-based)
        prediction_uncertainty = entropy(y_pred_proba.T)
        
        # Calibration analysis
        calibration_results = self._analyze_calibration(y_test, y_pred_proba[:, 1])
        
        # Confidence intervals for key metrics
        confidence_intervals = self._calculate_confidence_intervals(y_test, y_pred_proba[:, 1])
        
        # Out-of-distribution detection if training data available
        ood_scores = None
        ood_threshold = None
        if X_train is not None:
            ood_scores, ood_threshold = self._detect_out_of_distribution(X_train, X_test)
        
        return UncertaintyQuantification(
            prediction_uncertainty=prediction_uncertainty,
            calibration_error=calibration_results['calibration_error'],
            reliability_diagram_data=calibration_results['reliability_data'],
            confidence_intervals=confidence_intervals,
            ood_scores=ood_scores,
            ood_threshold=ood_threshold
        )
    
    def generate_monitoring_metrics(self, model: Any) -> Dict[str, Any]:
        """
        Generate suggested monitoring metrics for production deployment.
        
        Args:
            model: Trained model
            
        Returns:
            Dict with monitoring metric specifications
        """
        logger.info("Generating monitoring metrics recommendations")
        
        monitoring_metrics = {
            'performance_metrics': {
                'primary_metric': 'roc_auc',
                'secondary_metrics': ['precision', 'recall', 'f1_score'],
                'monitoring_frequency': 'daily',
                'alert_thresholds': {
                    'roc_auc_min': 0.7,
                    'precision_min': 0.6,
                    'recall_min': 0.6
                }
            },
            
            'fairness_metrics': {
                'demographic_parity_threshold': 0.1,
                'equalized_odds_threshold': 0.1,
                'monitoring_frequency': 'weekly',
                'protected_attributes': self.protected_attributes
            },
            
            'data_quality_metrics': {
                'missing_value_threshold': 0.05,
                'out_of_range_threshold': 0.02,
                'distribution_shift_threshold': 0.1,
                'monitoring_frequency': 'daily'
            },
            
            'prediction_quality_metrics': {
                'confidence_distribution_monitoring': True,
                'calibration_error_threshold': 0.1,
                'uncertainty_spike_threshold': 2.0,
                'monitoring_frequency': 'daily'
            },
            
            'operational_metrics': {
                'prediction_latency_threshold_ms': 100,
                'throughput_min_requests_per_second': 10,
                'error_rate_threshold': 0.01,
                'monitoring_frequency': 'real_time'
            }
        }
        
        # Add model-specific monitoring recommendations
        model_type = type(model).__name__.lower()
        
        if 'tree' in model_type or 'forest' in model_type:
            monitoring_metrics['model_specific'] = {
                'feature_importance_drift': True,
                'tree_depth_monitoring': True
            }
        elif 'linear' in model_type or 'logistic' in model_type:
            monitoring_metrics['model_specific'] = {
                'coefficient_drift': True,
                'linearity_assumption_validation': True
            }
        elif 'neural' in model_type or 'mlp' in model_type:
            monitoring_metrics['model_specific'] = {
                'gradient_monitoring': True,
                'activation_distribution_monitoring': True
            }
        
        return monitoring_metrics
    
    def export_model_card(self, output_path: str, format: str = 'html') -> None:
        """
        Export model card in specified format.
        
        Args:
            output_path: Output file path
            format: Export format ('html', 'json', 'pdf')
        """
        if self.model_card is None:
            raise ValueError("No model card generated. Call generate_model_card() first.")
        
        logger.info(f"Exporting model card to {output_path} in {format} format")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            self._export_model_card_json(output_path)
        elif format.lower() == 'html':
            self._export_model_card_html(output_path)
        elif format.lower() == 'pdf':
            self._export_model_card_pdf(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _calculate_performance_metrics(self, model: Any, X_test: np.ndarray, 
                                     y_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = self._calculate_auc(y_test, y_pred_proba)
        
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics.update({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        })
        
        return metrics
    
    def _calculate_fairness_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                   protected_attributes: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate fairness metrics for each protected attribute."""
        fairness_metrics = {}
        
        for attr_name, attr_values in protected_attributes.items():
            fairness_metrics[attr_name] = self._calculate_attribute_fairness(
                y_test, model.predict(X_test), 
                model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test),
                attr_values
            )
        
        return fairness_metrics
    
    def _calculate_attribute_fairness(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray, attr_values: np.ndarray) -> Dict[str, float]:
        """Calculate fairness metrics for a single protected attribute."""
        unique_groups = np.unique(attr_values)
        
        if len(unique_groups) < 2:
            return {}
        
        # Calculate metrics by group
        group_metrics = {}
        group_pred_rates = []
        group_tpr = []  # True positive rates
        group_fpr = []  # False positive rates
        
        for group in unique_groups:
            group_mask = (attr_values == group)
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            group_y_proba = y_pred_proba[group_mask]
            
            if len(group_y_true) == 0:
                continue
            
            # Prediction rate (demographic parity)
            pred_rate = np.mean(group_y_pred)
            group_pred_rates.append(pred_rate)
            
            # True positive rate and false positive rate
            if np.sum(group_y_true) > 0 and np.sum(group_y_true == 0) > 0:
                tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                group_tpr.append(tpr)
                group_fpr.append(fpr)
            
            # Group-specific performance
            if len(group_y_true) > 10:
                group_auc = self._calculate_auc(group_y_true, group_y_proba)
                group_metrics[f'group_{group}_auc'] = group_auc
        
        # Calculate fairness metrics
        results = {}
        
        # Demographic parity
        if len(group_pred_rates) >= 2:
            results['demographic_parity_difference'] = max(group_pred_rates) - min(group_pred_rates)
            results['demographic_parity_ratio'] = min(group_pred_rates) / max(group_pred_rates) if max(group_pred_rates) > 0 else 0
        
        # Equalized odds
        if len(group_tpr) >= 2 and len(group_fpr) >= 2:
            tpr_diff = max(group_tpr) - min(group_tpr)
            fpr_diff = max(group_fpr) - min(group_fpr)
            results['equalized_odds_difference'] = max(tpr_diff, fpr_diff)
            results['equalized_odds_ratio'] = min(min(group_tpr), min(group_fpr)) / max(max(group_tpr), max(group_fpr)) if max(max(group_tpr), max(group_fpr)) > 0 else 0
            
            results['equal_opportunity_difference'] = tpr_diff
            results['predictive_equality_difference'] = fpr_diff
        
        # Calibration metrics (simplified)
        calibration_by_group = {}
        calibration_errors = []
        
        for group in unique_groups:
            group_mask = (attr_values == group)
            group_y_true = y_true[group_mask]
            group_y_proba = y_pred_proba[group_mask]
            
            if len(group_y_true) > 10:
                calibration_error = self._calculate_calibration_error(group_y_true, group_y_proba)
                calibration_by_group[f'group_{group}'] = {'calibration_error': calibration_error}
                calibration_errors.append(calibration_error)
        
        results['calibration_metrics'] = calibration_by_group
        if len(calibration_errors) >= 2:
            results['calibration_difference'] = max(calibration_errors) - min(calibration_errors)
        
        results['group_performance'] = group_metrics
        
        return results
    
    def _determine_model_architecture(self, model: Any) -> str:
        """Determine model architecture description."""
        model_type = type(model).__name__.lower()
        
        if 'tree' in model_type:
            if hasattr(model, 'tree_'):
                return f"Single Decision Tree (max_depth: {getattr(model, 'max_depth', 'unlimited')})"
            else:
                return "Decision Tree"
        
        elif 'forest' in model_type:
            n_estimators = getattr(model, 'n_estimators', 'unknown')
            max_depth = getattr(model, 'max_depth', 'unlimited')
            return f"Random Forest ({n_estimators} trees, max_depth: {max_depth})"
        
        elif 'xgb' in model_type or 'gradient' in model_type:
            n_estimators = getattr(model, 'n_estimators', 'unknown')
            return f"Gradient Boosting ({n_estimators} estimators)"
        
        elif 'logistic' in model_type:
            return "Logistic Regression"
        
        elif 'linear' in model_type:
            return "Linear Regression"
        
        elif 'svm' in model_type or 'svc' in model_type:
            kernel = getattr(model, 'kernel', 'unknown')
            return f"Support Vector Machine ({kernel} kernel)"
        
        elif 'neural' in model_type or 'mlp' in model_type:
            hidden_layers = getattr(model, 'hidden_layer_sizes', 'unknown')
            return f"Neural Network (hidden layers: {hidden_layers})"
        
        else:
            return f"Machine Learning Model ({type(model).__name__})"
    
    def _calculate_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate AUC-ROC score."""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_score)
        except Exception:
            return 0.0
    
    def _create_contingency_table(self, y_pred: np.ndarray, attr_values: np.ndarray) -> np.ndarray:
        """Create contingency table for chi-square test."""
        unique_groups = np.unique(attr_values)
        contingency_table = np.zeros((2, len(unique_groups)))
        
        for i, group in enumerate(unique_groups):
            group_mask = (attr_values == group)
            contingency_table[0, i] = np.sum(y_pred[group_mask] == 0)  # Negative predictions
            contingency_table[1, i] = np.sum(y_pred[group_mask] == 1)  # Positive predictions
        
        return contingency_table
    
    def _detect_intersectional_bias(self, y_pred: np.ndarray, 
                                   protected_attributes: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Detect intersectional bias across multiple protected attributes."""
        intersectional_results = {}
        
        # Get combinations of protected attributes
        attr_names = list(protected_attributes.keys())
        
        if len(attr_names) >= 2:
            for i in range(len(attr_names)):
                for j in range(i + 1, len(attr_names)):
                    attr1, attr2 = attr_names[i], attr_names[j]
                    
                    # Create intersectional groups
                    intersectional_groups = {}
                    values1 = protected_attributes[attr1]
                    values2 = protected_attributes[attr2]
                    
                    for val1 in np.unique(values1):
                        for val2 in np.unique(values2):
                            group_mask = (values1 == val1) & (values2 == val2)
                            if np.sum(group_mask) > 10:  # Minimum group size
                                group_key = f"{attr1}_{val1}_{attr2}_{val2}"
                                intersectional_groups[group_key] = np.mean(y_pred[group_mask])
                    
                    if len(intersectional_groups) >= 2:
                        pred_rates = list(intersectional_groups.values())
                        intersectional_results[f"{attr1}_{attr2}"] = {
                            'max_difference': max(pred_rates) - min(pred_rates),
                            'group_rates': intersectional_groups
                        }
        
        return intersectional_results
    
    def _generate_bias_mitigation_recommendations(self, bias_detected: Dict[str, bool],
                                                bias_severity: Dict[str, str]) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []
        
        # General recommendations
        recommendations.extend([
            "Collect more diverse and representative training data",
            "Implement bias testing in model validation pipeline",
            "Consider fairness-aware machine learning algorithms",
            "Establish ongoing bias monitoring in production"
        ])
        
        # Attribute-specific recommendations
        for attr_name, bias_found in bias_detected.items():
            if bias_found:
                severity = bias_severity.get(attr_name, 'medium')
                
                if severity == 'high':
                    recommendations.append(
                        f"High bias detected for {attr_name}: Consider rebalancing training data "
                        f"or applying bias mitigation techniques"
                    )
                elif severity == 'medium':
                    recommendations.append(
                        f"Moderate bias detected for {attr_name}: Monitor closely and consider "
                        f"bias correction methods"
                    )
        
        # Intersectional bias recommendations
        recommendations.append(
            "Evaluate intersectional bias across multiple protected attributes simultaneously"
        )
        
        return recommendations
    
    def _analyze_calibration(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration."""
        if not UNCERTAINTY_AVAILABLE:
            return {'calibration_error': 0.0, 'reliability_data': {}}
        
        try:
            # Reliability diagram data
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                'calibration_error': ece,
                'reliability_data': {
                    'fraction_of_positives': fraction_of_positives,
                    'mean_predicted_value': mean_predicted_value
                }
            }
            
        except Exception as e:
            logger.warning(f"Calibration analysis failed: {e}")
            return {'calibration_error': 0.0, 'reliability_data': {}}
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate calibration error for a subset of data."""
        try:
            calibration_result = self._analyze_calibration(y_true, y_proba)
            return calibration_result['calibration_error']
        except Exception:
            return 0.0
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_proba: np.ndarray,
                                      confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        try:
            from sklearn.utils import resample
            
            n_bootstraps = 1000
            metrics_bootstrap = []
            
            for _ in range(n_bootstraps):
                # Bootstrap sample
                indices = resample(range(len(y_true)), n_samples=len(y_true))
                y_true_boot = y_true[indices]
                y_proba_boot = y_proba[indices]
                
                # Calculate metrics
                auc = self._calculate_auc(y_true_boot, y_proba_boot)
                metrics_bootstrap.append({'auc': auc})
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            auc_values = [m['auc'] for m in metrics_bootstrap]
            
            return {
                'auc': (
                    np.percentile(auc_values, lower_percentile),
                    np.percentile(auc_values, upper_percentile)
                )
            }
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return {}
    
    def _detect_out_of_distribution(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect out-of-distribution samples using simple methods."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Fit isolation forest on training data
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X_train)
            
            # Score test data
            ood_scores = iso_forest.decision_function(X_test)
            ood_threshold = np.percentile(iso_forest.decision_function(X_train), 10)
            
            return ood_scores, ood_threshold
            
        except Exception as e:
            logger.warning(f"OOD detection failed: {e}")
            return np.zeros(len(X_test)), 0.0
    
    def _export_model_card_json(self, output_path: Path):
        """Export model card as JSON."""
        # Convert dataclass to dict
        card_dict = {
            'model_details': {
                'name': self.model_card.model_name,
                'version': self.model_card.model_version,
                'type': self.model_card.model_type,
                'description': self.model_card.model_description,
                'architecture': self.model_card.model_architecture
            },
            'intended_use': {
                'primary_intended_uses': self.model_card.intended_use,
                'intended_users': self.model_card.intended_users,
                'out_of_scope_uses': self.model_card.out_of_scope_uses
            },
            'factors': {
                'relevant_factors': self.model_card.relevant_factors,
                'evaluation_factors': self.model_card.evaluation_factors
            },
            'metrics': {
                'model_performance': self.model_card.model_performance,
                'fairness_metrics': self.model_card.fairness_metrics
            },
            'training_data': {
                'description': self.model_card.training_data_description,
                'preprocessing': self.model_card.training_data_preprocessing,
                'size': self.model_card.training_data_size
            },
            'evaluation_data': {
                'description': self.model_card.evaluation_data_description,
                'size': self.model_card.evaluation_data_size
            },
            'ethical_considerations': self.model_card.ethical_considerations,
            'caveats_and_recommendations': {
                'caveats': self.model_card.caveats,
                'recommendations': self.model_card.recommendations
            },
            'technical_specifications': self.model_card.technical_specs,
            'quantitative_analysis': self.model_card.quantitative_analysis
        }
        
        with open(output_path, 'w') as f:
            json.dump(card_dict, f, indent=2, default=str)
    
    def _export_model_card_html(self, output_path: Path):
        """Export model card as HTML."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Card: {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #dee2e6; border-radius: 8px; }}
                .metric {{ background-color: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; }}
                .recommendation {{ background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 4px; }}
                h1, h2, h3 {{ color: #495057; }}
                ul {{ padding-left: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Card: {model_name}</h1>
                <p><strong>Version:</strong> {model_version}</p>
                <p><strong>Type:</strong> {model_type}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Model Description</h2>
                <p>{model_description}</p>
                <p><strong>Architecture:</strong> {model_architecture}</p>
            </div>
            
            <div class="section">
                <h2>Intended Use</h2>
                <p><strong>Primary Use:</strong> {intended_use}</p>
                <h3>Intended Users</h3>
                <ul>
                    {intended_users_list}
                </ul>
                <h3>Out of Scope Uses</h3>
                <ul>
                    {out_of_scope_list}
                </ul>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {performance_metrics_table}
                </table>
            </div>
            
            <div class="section">
                <h2>Fairness Assessment</h2>
                {fairness_metrics_section}
            </div>
            
            <div class="section">
                <h2>Ethical Considerations</h2>
                <ul>
                    {ethical_considerations_list}
                </ul>
            </div>
            
            <div class="section warning">
                <h2>Caveats and Limitations</h2>
                <ul>
                    {caveats_list}
                </ul>
            </div>
            
            <div class="section recommendation">
                <h2>Recommendations</h2>
                <ul>
                    {recommendations_list}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Format data for template
        intended_users_list = ''.join(f'<li>{user}</li>' for user in self.model_card.intended_users)
        out_of_scope_list = ''.join(f'<li>{use}</li>' for use in self.model_card.out_of_scope_uses)
        
        performance_metrics_table = ''.join(
            f'<tr><td>{metric}</td><td>{value:.4f}</td></tr>' 
            for metric, value in self.model_card.model_performance.items()
        )
        
        fairness_metrics_section = ""
        if self.model_card.fairness_metrics:
            fairness_metrics_section = "<p>Fairness metrics calculated across protected attributes.</p>"
            for attr, metrics in self.model_card.fairness_metrics.items():
                fairness_metrics_section += f"<h3>{attr}</h3><ul>"
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        fairness_metrics_section += f"<li>{metric}: {value:.4f}</li>"
                fairness_metrics_section += "</ul>"
        else:
            fairness_metrics_section = "<p>No fairness metrics calculated. Protected attribute data not provided.</p>"
        
        ethical_considerations_list = ''.join(f'<li>{consideration}</li>' for consideration in self.model_card.ethical_considerations)
        caveats_list = ''.join(f'<li>{caveat}</li>' for caveat in self.model_card.caveats)
        recommendations_list = ''.join(f'<li>{rec}</li>' for rec in self.model_card.recommendations)
        
        html_content = html_template.format(
            model_name=self.model_card.model_name,
            model_version=self.model_card.model_version,
            model_type=self.model_card.model_type,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            model_description=self.model_card.model_description,
            model_architecture=self.model_card.model_architecture,
            intended_use=self.model_card.intended_use,
            intended_users_list=intended_users_list,
            out_of_scope_list=out_of_scope_list,
            performance_metrics_table=performance_metrics_table,
            fairness_metrics_section=fairness_metrics_section,
            ethical_considerations_list=ethical_considerations_list,
            caveats_list=caveats_list,
            recommendations_list=recommendations_list
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_model_card_pdf(self, output_path: Path):
        """Export model card as PDF."""
        # Generate HTML first
        html_path = output_path.with_suffix('.html')
        self._export_model_card_html(html_path)
        
        # Convert to PDF if possible
        if REPORT_GENERATION_AVAILABLE:
            try:
                from weasyprint import HTML
                HTML(filename=str(html_path)).write_pdf(output_path)
                # Clean up temporary HTML
                html_path.unlink()
                logger.info(f"PDF model card exported to {output_path}")
            except Exception as e:
                logger.error(f"PDF export failed: {e}")
        else:
            logger.warning("PDF export not available. HTML version saved instead.")


def create_admissible_ml_generator() -> AdmissibleMLGenerator:
    """Factory function to create AdmissibleML generator."""
    return AdmissibleMLGenerator()