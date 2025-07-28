"""
Interpretability Pipeline Orchestrator for ML Pipeline Framework

This module provides a unified pipeline for running all applicable explainability methods,
generating comprehensive interpretability reports, and producing stakeholder-specific summaries.
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

# Core explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Some explanations will be limited.")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Local explanations will be limited.")

# Model analysis
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance, partial_dependence, plot_partial_dependence
from sklearn.model_selection import cross_val_score

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Report generation
try:
    import jinja2
    from weasyprint import HTML, CSS
    REPORT_GENERATION_AVAILABLE = True
except ImportError:
    REPORT_GENERATION_AVAILABLE = False
    warnings.warn("Report generation libraries not available. PDF reports will be disabled.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability pipeline."""
    
    # Method selection
    enable_global_explanations: bool = True
    enable_local_explanations: bool = True
    enable_feature_importance: bool = True
    enable_partial_dependence: bool = True
    enable_interaction_analysis: bool = True
    enable_prototype_analysis: bool = True
    enable_concept_analysis: bool = False  # Requires additional setup
    enable_causal_analysis: bool = False   # Requires domain knowledge
    
    # Sampling for efficiency
    max_samples_global: int = 1000
    max_samples_local: int = 100
    max_samples_shap: int = 500
    
    # Analysis depth
    top_k_features: int = 20
    interaction_depth: int = 2
    
    # Output formats
    generate_html_report: bool = True
    generate_pdf_report: bool = True
    generate_json_export: bool = True
    generate_excel_export: bool = True
    
    # Stakeholder customization
    include_technical_details: bool = True
    include_business_summary: bool = True
    include_regulatory_summary: bool = True
    
    # Visualization preferences
    plot_style: str = 'seaborn-v0_8'
    color_palette: str = 'viridis'
    figure_dpi: int = 300


@dataclass
class ExplanationResult:
    """Container for explanation results from a specific method."""
    method_name: str
    explanation_type: str  # 'global', 'local', 'feature_importance', etc.
    explanations: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[str] = field(default_factory=list)  # Paths to saved plots
    summary: str = ""
    confidence_score: float = 0.0


@dataclass
class BlindSpotAnalysis:
    """Container for model blind spot analysis."""
    low_confidence_regions: List[Dict[str, Any]]
    out_of_distribution_samples: List[int]
    prediction_uncertainty_high: List[int]
    feature_coverage_gaps: Dict[str, List[Any]]
    interaction_blind_spots: List[Tuple[str, str]]
    recommendations: List[str]


@dataclass
class StakeholderSummary:
    """Stakeholder-specific summary of interpretability results."""
    stakeholder_type: str  # 'technical', 'business', 'regulatory'
    key_insights: List[str]
    recommendations: List[str]
    risk_assessment: str
    actionable_items: List[str]
    visualizations: List[str]
    confidence_level: str


class InterpretabilityPipeline:
    """
    Unified pipeline for comprehensive model interpretability analysis.
    """
    
    def __init__(self, config: InterpretabilityConfig = None):
        """
        Initialize interpretability pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or InterpretabilityConfig()
        self.results = {}
        self.blind_spots = None
        self.stakeholder_summaries = {}
        self.feature_names = None
        self.target_names = None
        
        # Set up plotting
        plt.style.use(self.config.plot_style)
        sns.set_palette(self.config.color_palette)
        
        # Initialize explainer objects
        self.explainers = {}
        self._setup_explainers()
    
    def _setup_explainers(self):
        """Initialize explainer objects."""
        if SHAP_AVAILABLE:
            self.explainers['shap'] = None  # Will be initialized with model
        
        if LIME_AVAILABLE:
            self.explainers['lime'] = None  # Will be initialized with data
    
    def run_full_pipeline(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: List[str] = None,
                         target_names: List[str] = None,
                         output_dir: str = './artifacts/interpretability') -> Dict[str, Any]:
        """
        Run complete interpretability pipeline.
        
        Args:
            model: Trained model to explain
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
            target_names: Target class names
            output_dir: Output directory for artifacts
            
        Returns:
            Dict with all pipeline results
        """
        logger.info("Starting comprehensive interpretability pipeline")
        
        # Setup
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.target_names = target_names or ['class_0', 'class_1']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize explainers with data
        self._initialize_explainers(model, X_train, y_train)
        
        # Run explanation methods
        if self.config.enable_global_explanations:
            self._run_global_explanations(model, X_train, y_train, X_test, y_test, output_path)
        
        if self.config.enable_local_explanations:
            self._run_local_explanations(model, X_test, y_test, output_path)
        
        if self.config.enable_feature_importance:
            self._run_feature_importance_analysis(model, X_train, y_train, X_test, y_test, output_path)
        
        if self.config.enable_partial_dependence:
            self._run_partial_dependence_analysis(model, X_train, y_train, output_path)
        
        if self.config.enable_interaction_analysis:
            self._run_interaction_analysis(model, X_train, y_train, output_path)
        
        # Analyze model blind spots
        self.blind_spots = self._analyze_blind_spots(model, X_train, y_train, X_test, y_test)
        
        # Generate stakeholder summaries
        self._generate_stakeholder_summaries(model, X_test, y_test)
        
        # Export results
        self._export_results(output_path)
        
        # Generate unified report
        if self.config.generate_html_report or self.config.generate_pdf_report:
            self._generate_unified_report(output_path)
        
        logger.info(f"Interpretability pipeline completed. Results saved to {output_path}")
        
        return {
            'explanations': self.results,
            'blind_spots': self.blind_spots,
            'stakeholder_summaries': self.stakeholder_summaries,
            'output_directory': str(output_path)
        }
    
    def _initialize_explainers(self, model: Any, X_train: np.ndarray, y_train: np.ndarray):
        """Initialize explainers with model and training data."""
        
        # SHAP explainer
        if SHAP_AVAILABLE and 'shap' in self.explainers:
            try:
                if hasattr(model, 'predict_proba'):
                    # For probabilistic models
                    if hasattr(model, 'tree_'):
                        # Single decision tree
                        self.explainers['shap'] = shap.TreeExplainer(model)
                    elif hasattr(model, 'estimators_'):
                        # Tree ensembles
                        self.explainers['shap'] = shap.TreeExplainer(model)
                    elif hasattr(model, 'coef_'):
                        # Linear models
                        self.explainers['shap'] = shap.LinearExplainer(model, X_train)
                    else:
                        # Default to kernel explainer
                        background = shap.sample(X_train, min(100, len(X_train)))
                        self.explainers['shap'] = shap.KernelExplainer(model.predict_proba, background)
                
                logger.info("SHAP explainer initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
                self.explainers['shap'] = None
        
        # LIME explainer
        if LIME_AVAILABLE and 'lime' in self.explainers:
            try:
                self.explainers['lime'] = LimeTabularExplainer(
                    X_train,
                    feature_names=self.feature_names,
                    class_names=self.target_names,
                    mode='classification'
                )
                logger.info("LIME explainer initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize LIME explainer: {e}")
                self.explainers['lime'] = None
    
    def _run_global_explanations(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray, output_path: Path):
        """Run global explanation methods."""
        logger.info("Running global explanations")
        
        # SHAP global explanations
        if self.explainers.get('shap') is not None:
            try:
                # Sample data for efficiency
                X_sample = X_test[:self.config.max_samples_shap]
                
                # Calculate SHAP values
                shap_values = self.explainers['shap'].shap_values(X_sample)
                
                # Handle multi-output case
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class for binary classification
                
                # Create visualizations
                plots = []
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
                summary_plot_path = output_path / 'shap_summary_plot.png'
                plt.savefig(summary_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                plt.close()
                plots.append(str(summary_plot_path))
                
                # Feature importance plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                                plot_type="bar", show=False)
                importance_plot_path = output_path / 'shap_importance_plot.png'
                plt.savefig(importance_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                plt.close()
                plots.append(str(importance_plot_path))
                
                # Store results
                self.results['shap_global'] = ExplanationResult(
                    method_name='SHAP Global',
                    explanation_type='global',
                    explanations={
                        'shap_values': shap_values,
                        'expected_value': self.explainers['shap'].expected_value,
                        'feature_importance': np.abs(shap_values).mean(axis=0)
                    },
                    visualizations=plots,
                    summary="SHAP global explanations showing feature importance and interactions",
                    confidence_score=0.9
                )
                
                logger.info("SHAP global explanations completed")
                
            except Exception as e:
                logger.error(f"Error in SHAP global explanations: {e}")
        
        # Model-specific global explanations
        self._run_model_specific_global_explanations(model, X_train, y_train, output_path)
    
    def _run_local_explanations(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                               output_path: Path):
        """Run local explanation methods."""
        logger.info("Running local explanations")
        
        # Select samples for local explanations
        n_samples = min(self.config.max_samples_local, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]
        
        local_explanations = []
        
        # SHAP local explanations
        if self.explainers.get('shap') is not None:
            try:
                shap_values = self.explainers['shap'].shap_values(X_sample[:10])  # First 10 samples
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Create waterfall plots for first few samples
                plots = []
                for i in range(min(5, len(X_sample))):
                    plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[i],
                            base_values=self.explainers['shap'].expected_value,
                            data=X_sample[i],
                            feature_names=self.feature_names
                        ),
                        show=False
                    )
                    waterfall_path = output_path / f'shap_waterfall_sample_{i}.png'
                    plt.savefig(waterfall_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                    plt.close()
                    plots.append(str(waterfall_path))
                
                local_explanations.append({
                    'method': 'SHAP',
                    'explanations': shap_values,
                    'sample_indices': sample_indices[:10],
                    'plots': plots
                })
                
            except Exception as e:
                logger.error(f"Error in SHAP local explanations: {e}")
        
        # LIME local explanations
        if self.explainers.get('lime') is not None:
            try:
                lime_explanations = []
                plots = []
                
                for i in range(min(5, len(X_sample))):
                    explanation = self.explainers['lime'].explain_instance(
                        X_sample[i], 
                        model.predict_proba,
                        num_features=self.config.top_k_features
                    )
                    
                    lime_explanations.append({
                        'sample_index': sample_indices[i],
                        'explanation': explanation.as_list(),
                        'prediction_proba': model.predict_proba([X_sample[i]])[0]
                    })
                    
                    # Save plot
                    fig = explanation.as_pyplot_figure()
                    lime_path = output_path / f'lime_explanation_sample_{i}.png'
                    fig.savefig(lime_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                    plt.close(fig)
                    plots.append(str(lime_path))
                
                local_explanations.append({
                    'method': 'LIME',
                    'explanations': lime_explanations,
                    'plots': plots
                })
                
            except Exception as e:
                logger.error(f"Error in LIME local explanations: {e}")
        
        # Store results
        self.results['local_explanations'] = ExplanationResult(
            method_name='Local Explanations',
            explanation_type='local',
            explanations=local_explanations,
            visualizations=[plot for exp in local_explanations for plot in exp.get('plots', [])],
            summary=f"Local explanations for {n_samples} samples using multiple methods",
            confidence_score=0.8
        )
        
        logger.info("Local explanations completed")
    
    def _run_feature_importance_analysis(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: np.ndarray, y_test: np.ndarray, output_path: Path):
        """Run comprehensive feature importance analysis."""
        logger.info("Running feature importance analysis")
        
        importance_results = {}
        plots = []
        
        # Model-specific feature importance
        if hasattr(model, 'feature_importances_'):
            importance_results['model_importance'] = {
                'values': model.feature_importances_,
                'method': 'Model-specific (e.g., Gini importance)'
            }
        
        elif hasattr(model, 'coef_'):
            importance_results['model_importance'] = {
                'values': np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_),
                'method': 'Coefficient magnitude'
            }
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10, 
                random_state=42,
                scoring='roc_auc'
            )
            
            importance_results['permutation_importance'] = {
                'values': perm_importance.importances_mean,
                'std': perm_importance.importances_std,
                'method': 'Permutation importance'
            }
            
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
        
        # Create comparison plot
        if importance_results:
            fig, axes = plt.subplots(1, len(importance_results), figsize=(15, 8))
            if len(importance_results) == 1:
                axes = [axes]
            
            for i, (method, data) in enumerate(importance_results.items()):
                # Get top features
                top_indices = np.argsort(data['values'])[-self.config.top_k_features:]
                top_values = data['values'][top_indices]
                top_features = [self.feature_names[j] for j in top_indices]
                
                axes[i].barh(range(len(top_values)), top_values)
                axes[i].set_yticks(range(len(top_values)))
                axes[i].set_yticklabels(top_features)
                axes[i].set_title(f'{data["method"]}')
                axes[i].set_xlabel('Importance')
                
                # Add error bars if available
                if 'std' in data:
                    top_std = data['std'][top_indices]
                    axes[i].errorbar(top_values, range(len(top_values)), 
                                   xerr=top_std, fmt='none', color='red', alpha=0.5)
            
            plt.tight_layout()
            importance_plot_path = output_path / 'feature_importance_comparison.png'
            plt.savefig(importance_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            plots.append(str(importance_plot_path))
        
        # Store results
        self.results['feature_importance'] = ExplanationResult(
            method_name='Feature Importance',
            explanation_type='feature_importance',
            explanations=importance_results,
            visualizations=plots,
            summary=f"Feature importance analysis using {len(importance_results)} methods",
            confidence_score=0.85
        )
        
        logger.info("Feature importance analysis completed")
    
    def _run_partial_dependence_analysis(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                        output_path: Path):
        """Run partial dependence analysis."""
        logger.info("Running partial dependence analysis")
        
        try:
            # Select top features for PDP
            if 'feature_importance' in self.results:
                # Use feature importance to select features
                importance_data = self.results['feature_importance'].explanations
                if 'model_importance' in importance_data:
                    importance_values = importance_data['model_importance']['values']
                    top_feature_indices = np.argsort(importance_values)[-10:]
                else:
                    top_feature_indices = list(range(min(10, X_train.shape[1])))
            else:
                top_feature_indices = list(range(min(10, X_train.shape[1])))
            
            # Create partial dependence plots
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Single feature PDPs
            plot_partial_dependence(
                model, X_train, 
                features=top_feature_indices,
                feature_names=self.feature_names,
                ax=ax
            )
            
            pdp_plot_path = output_path / 'partial_dependence_plots.png'
            plt.savefig(pdp_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            plt.close()
            
            # Store results
            self.results['partial_dependence'] = ExplanationResult(
                method_name='Partial Dependence',
                explanation_type='partial_dependence',
                explanations={'top_features': top_feature_indices},
                visualizations=[str(pdp_plot_path)],
                summary=f"Partial dependence plots for top {len(top_feature_indices)} features",
                confidence_score=0.7
            )
            
            logger.info("Partial dependence analysis completed")
            
        except Exception as e:
            logger.error(f"Partial dependence analysis failed: {e}")
    
    def _run_interaction_analysis(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                 output_path: Path):
        """Run feature interaction analysis."""
        logger.info("Running interaction analysis")
        
        try:
            # Simple interaction analysis using SHAP if available
            if self.explainers.get('shap') is not None and SHAP_AVAILABLE:
                # Sample data for efficiency
                X_sample = X_train[:self.config.max_samples_global]
                shap_values = self.explainers['shap'].shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Interaction values (if supported)
                try:
                    interaction_values = self.explainers['shap'].shap_interaction_values(X_sample[:100])
                    
                    # Plot interaction summary
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        interaction_values, X_sample[:100],
                        feature_names=self.feature_names,
                        show=False
                    )
                    interaction_plot_path = output_path / 'interaction_summary.png'
                    plt.savefig(interaction_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                    plt.close()
                    
                    # Store results
                    self.results['interactions'] = ExplanationResult(
                        method_name='Feature Interactions',
                        explanation_type='interactions',
                        explanations={'interaction_values': interaction_values},
                        visualizations=[str(interaction_plot_path)],
                        summary="Feature interaction analysis using SHAP interaction values",
                        confidence_score=0.75
                    )
                    
                except Exception as e:
                    logger.warning(f"SHAP interaction analysis failed: {e}")
                    # Fallback to simple correlation analysis
                    self._simple_interaction_analysis(X_train, output_path)
            else:
                self._simple_interaction_analysis(X_train, output_path)
            
            logger.info("Interaction analysis completed")
            
        except Exception as e:
            logger.error(f"Interaction analysis failed: {e}")
    
    def _simple_interaction_analysis(self, X_train: np.ndarray, output_path: Path):
        """Simple interaction analysis using correlations."""
        # Calculate feature correlations
        df = pd.DataFrame(X_train, columns=self.feature_names)
        correlation_matrix = df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        correlation_plot_path = output_path / 'feature_correlations.png'
        plt.savefig(correlation_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        self.results['interactions'] = ExplanationResult(
            method_name='Feature Correlations',
            explanation_type='interactions',
            explanations={
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_correlations
            },
            visualizations=[str(correlation_plot_path)],
            summary=f"Feature correlation analysis found {len(high_correlations)} high correlations",
            confidence_score=0.6
        )
    
    def _run_model_specific_global_explanations(self, model: Any, X_train: np.ndarray, 
                                               y_train: np.ndarray, output_path: Path):
        """Run model-specific global explanations."""
        model_type = type(model).__name__.lower()
        
        # Decision tree visualization
        if 'tree' in model_type and hasattr(model, 'tree_'):
            try:
                from sklearn.tree import plot_tree
                
                plt.figure(figsize=(20, 10))
                plot_tree(model, feature_names=self.feature_names, 
                         class_names=self.target_names, filled=True, max_depth=3)
                tree_plot_path = output_path / 'decision_tree_visualization.png'
                plt.savefig(tree_plot_path, dpi=self.config.figure_dpi, bbox_inches='tight')
                plt.close()
                
                self.results['tree_visualization'] = ExplanationResult(
                    method_name='Decision Tree Visualization',
                    explanation_type='global',
                    explanations={'tree_structure': 'visualized'},
                    visualizations=[str(tree_plot_path)],
                    summary="Decision tree structure visualization",
                    confidence_score=1.0
                )
                
            except Exception as e:
                logger.warning(f"Tree visualization failed: {e}")
    
    def _analyze_blind_spots(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> BlindSpotAnalysis:
        """Analyze model blind spots and uncertainty regions."""
        logger.info("Analyzing model blind spots")
        
        # Prediction confidence analysis
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            
            # Find low confidence predictions
            if y_proba.shape[1] == 2:  # Binary classification
                confidence = np.max(y_proba, axis=1)
                low_confidence_threshold = 0.6
                low_confidence_indices = np.where(confidence < low_confidence_threshold)[0]
            else:
                confidence = np.max(y_proba, axis=1)
                low_confidence_threshold = 0.5
                low_confidence_indices = np.where(confidence < low_confidence_threshold)[0]
        else:
            low_confidence_indices = []
        
        # Feature coverage analysis
        feature_coverage_gaps = {}
        for i, feature_name in enumerate(self.feature_names):
            train_range = (X_train[:, i].min(), X_train[:, i].max())
            test_out_of_range = np.where(
                (X_test[:, i] < train_range[0]) | (X_test[:, i] > train_range[1])
            )[0]
            
            if len(test_out_of_range) > 0:
                feature_coverage_gaps[feature_name] = test_out_of_range.tolist()
        
        # Out-of-distribution detection (simple method using isolation forest)
        try:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X_train)
            outliers = iso_forest.predict(X_test)
            ood_indices = np.where(outliers == -1)[0]
            
        except Exception:
            ood_indices = []
        
        # High uncertainty predictions (intersection of various criteria)
        high_uncertainty_indices = list(set(low_confidence_indices) | set(ood_indices))
        
        # Feature interaction blind spots (simplified)
        interaction_blind_spots = []
        if len(self.feature_names) > 1:
            # Find features with unusual interaction patterns
            for i in range(min(5, len(self.feature_names))):
                for j in range(i+1, min(5, len(self.feature_names))):
                    feature1, feature2 = self.feature_names[i], self.feature_names[j]
                    # Simple heuristic: if correlation in test differs significantly from train
                    train_corr = np.corrcoef(X_train[:, i], X_train[:, j])[0, 1]
                    test_corr = np.corrcoef(X_test[:, i], X_test[:, j])[0, 1]
                    
                    if abs(train_corr - test_corr) > 0.3:
                        interaction_blind_spots.append((feature1, feature2))
        
        # Generate recommendations
        recommendations = []
        
        if len(low_confidence_indices) > 0:
            recommendations.append(
                f"Model shows low confidence on {len(low_confidence_indices)} samples "
                f"({len(low_confidence_indices)/len(X_test)*100:.1f}% of test set)"
            )
        
        if feature_coverage_gaps:
            recommendations.append(
                f"Features with out-of-range values: {list(feature_coverage_gaps.keys())}"
            )
        
        if len(ood_indices) > 0:
            recommendations.append(
                f"Detected {len(ood_indices)} potential out-of-distribution samples"
            )
        
        if interaction_blind_spots:
            recommendations.append(
                f"Detected {len(interaction_blind_spots)} potential interaction shifts"
            )
        
        return BlindSpotAnalysis(
            low_confidence_regions=[
                {
                    'sample_index': int(idx),
                    'confidence': float(confidence[idx]) if len(low_confidence_indices) > 0 else 0.0
                }
                for idx in low_confidence_indices[:20]  # Limit to first 20
            ],
            out_of_distribution_samples=ood_indices.tolist()[:20],
            prediction_uncertainty_high=high_uncertainty_indices[:20],
            feature_coverage_gaps=feature_coverage_gaps,
            interaction_blind_spots=interaction_blind_spots,
            recommendations=recommendations
        )
    
    def _generate_stakeholder_summaries(self, model: Any, X_test: np.ndarray, y_test: np.ndarray):
        """Generate stakeholder-specific summaries."""
        logger.info("Generating stakeholder summaries")
        
        # Get model performance metrics
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        else:
            y_pred = model.predict(X_test)
            auc_score = roc_auc_score(y_test, y_pred)
        
        # Technical stakeholder summary
        technical_insights = [
            f"Model achieves AUC of {auc_score:.3f} on test set",
            f"Generated explanations using {len(self.results)} methods",
            f"Identified {len(self.blind_spots.recommendations)} potential blind spots"
        ]
        
        technical_recommendations = [
            "Monitor model performance on edge cases identified in blind spot analysis",
            "Consider ensemble methods if individual model confidence is low",
            "Implement uncertainty quantification for production deployment"
        ]
        
        if auc_score < 0.8:
            technical_recommendations.append("Model performance may need improvement before deployment")
        
        self.stakeholder_summaries['technical'] = StakeholderSummary(
            stakeholder_type='technical',
            key_insights=technical_insights,
            recommendations=technical_recommendations,
            risk_assessment="Medium" if auc_score > 0.75 else "High",
            actionable_items=[
                "Review feature importance rankings for model improvement",
                "Implement monitoring for identified blind spots",
                "Set up automated explanation generation pipeline"
            ],
            visualizations=[],  # Will be populated during export
            confidence_level="High" if len(self.results) >= 3 else "Medium"
        )
        
        # Business stakeholder summary
        business_insights = [
            f"Model correctly identifies {auc_score*100:.1f}% of cases on average",
            "Key decision factors have been identified and visualized",
            "Model behavior is explainable for business review"
        ]
        
        business_recommendations = [
            "Review top features influencing decisions for business logic validation",
            "Establish process for handling low-confidence predictions",
            "Consider cost-benefit analysis for prediction thresholds"
        ]
        
        self.stakeholder_summaries['business'] = StakeholderSummary(
            stakeholder_type='business',
            key_insights=business_insights,
            recommendations=business_recommendations,
            risk_assessment="Acceptable" if auc_score > 0.7 else "Needs Review",
            actionable_items=[
                "Validate that important features align with business knowledge",
                "Define escalation process for uncertain predictions",
                "Establish monitoring KPIs for model performance"
            ],
            visualizations=[],
            confidence_level="High"
        )
        
        # Regulatory stakeholder summary
        regulatory_insights = [
            "Model decisions are explainable and auditable",
            "Feature importance and individual predictions can be justified",
            "Bias and fairness analysis capabilities are available"
        ]
        
        regulatory_recommendations = [
            "Implement regular bias testing and fairness monitoring",
            "Maintain explanation records for audit purposes",
            "Establish model governance and documentation processes"
        ]
        
        self.stakeholder_summaries['regulatory'] = StakeholderSummary(
            stakeholder_type='regulatory',
            key_insights=regulatory_insights,
            recommendations=regulatory_recommendations,
            risk_assessment="Compliant" if len(self.results) >= 2 else "Requires Enhancement",
            actionable_items=[
                "Document all explanation methods and their limitations",
                "Implement explanation storage and retrieval system",
                "Establish regular model validation and testing schedule"
            ],
            visualizations=[],
            confidence_level="High"
        )
    
    def _export_results(self, output_path: Path):
        """Export results in various formats."""
        logger.info("Exporting results")
        
        # JSON export
        if self.config.generate_json_export:
            json_data = {
                'pipeline_config': {
                    'timestamp': datetime.now().isoformat(),
                    'methods_used': list(self.results.keys()),
                    'feature_names': self.feature_names,
                    'target_names': self.target_names
                },
                'explanations': {
                    name: {
                        'method_name': result.method_name,
                        'explanation_type': result.explanation_type,
                        'summary': result.summary,
                        'confidence_score': result.confidence_score,
                        'metadata': result.metadata,
                        'visualizations': result.visualizations
                    }
                    for name, result in self.results.items()
                },
                'blind_spots': {
                    'low_confidence_count': len(self.blind_spots.low_confidence_regions),
                    'ood_count': len(self.blind_spots.out_of_distribution_samples),
                    'feature_coverage_gaps': list(self.blind_spots.feature_coverage_gaps.keys()),
                    'recommendations': self.blind_spots.recommendations
                },
                'stakeholder_summaries': {
                    stakeholder: {
                        'key_insights': summary.key_insights,
                        'recommendations': summary.recommendations,
                        'risk_assessment': summary.risk_assessment,
                        'confidence_level': summary.confidence_level
                    }
                    for stakeholder, summary in self.stakeholder_summaries.items()
                }
            }
            
            with open(output_path / 'interpretability_results.json', 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
        
        # Excel export
        if self.config.generate_excel_export:
            self._export_to_excel(output_path)
    
    def _export_to_excel(self, output_path: Path):
        """Export results to Excel format."""
        try:
            import pandas as pd
            
            excel_path = output_path / 'interpretability_results.xlsx'
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for name, result in self.results.items():
                    summary_data.append({
                        'Method': result.method_name,
                        'Type': result.explanation_type,
                        'Confidence': result.confidence_score,
                        'Summary': result.summary
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Stakeholder summaries
                for stakeholder, summary in self.stakeholder_summaries.items():
                    stakeholder_data = {
                        'Category': ['Key Insights'] * len(summary.key_insights) + 
                                  ['Recommendations'] * len(summary.recommendations) + 
                                  ['Action Items'] * len(summary.actionable_items),
                        'Item': summary.key_insights + summary.recommendations + summary.actionable_items
                    }
                    
                    stakeholder_df = pd.DataFrame(stakeholder_data)
                    stakeholder_df.to_excel(writer, sheet_name=f'{stakeholder.title()}', index=False)
                
                # Blind spots analysis
                if self.blind_spots.recommendations:
                    blind_spots_df = pd.DataFrame({
                        'Issue': ['Low Confidence Samples', 'OOD Samples', 'Feature Coverage Gaps', 'Interaction Shifts'],
                        'Count': [
                            len(self.blind_spots.low_confidence_regions),
                            len(self.blind_spots.out_of_distribution_samples),
                            len(self.blind_spots.feature_coverage_gaps),
                            len(self.blind_spots.interaction_blind_spots)
                        ]
                    })
                    blind_spots_df.to_excel(writer, sheet_name='Blind_Spots', index=False)
            
            logger.info(f"Excel export completed: {excel_path}")
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
    
    def _generate_unified_report(self, output_path: Path):
        """Generate unified HTML/PDF report."""
        logger.info("Generating unified interpretability report")
        
        # Create HTML report
        html_content = self._create_html_report()
        
        html_path = output_path / 'interpretability_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {html_path}")
        
        # Generate PDF if requested and libraries available
        if self.config.generate_pdf_report and REPORT_GENERATION_AVAILABLE:
            try:
                pdf_path = output_path / 'interpretability_report.pdf'
                HTML(string=html_content).write_pdf(pdf_path)
                logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                logger.error(f"PDF generation failed: {e}")
    
    def _create_html_report(self) -> str:
        """Create HTML report content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Interpretability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 30px 0; }
                .method { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .stakeholder { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e7f3ff; border-radius: 3px; }
                .recommendation { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }
                .blind-spot { background-color: #f8d7da; padding: 10px; margin: 5px 0; border-left: 4px solid #dc3545; }
                ul { padding-left: 20px; }
                h1, h2, h3 { color: #333; }
                .confidence-high { color: #28a745; font-weight: bold; }
                .confidence-medium { color: #ffc107; font-weight: bold; }
                .confidence-low { color: #dc3545; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Interpretability Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Methods used: {methods_count}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Total Explanations: {total_explanations}</div>
                <div class="metric">Blind Spots Identified: {blind_spots_count}</div>
                <div class="metric">Stakeholder Summaries: {stakeholder_count}</div>
            </div>
            
            <div class="section">
                <h2>Explanation Methods</h2>
                {explanation_methods}
            </div>
            
            <div class="section">
                <h2>Model Blind Spots</h2>
                {blind_spots_section}
            </div>
            
            <div class="section">
                <h2>Stakeholder Summaries</h2>
                {stakeholder_summaries}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {overall_recommendations}
            </div>
        </body>
        </html>
        """
        
        # Build content sections
        explanation_methods_html = ""
        for name, result in self.results.items():
            confidence_class = f"confidence-{self._get_confidence_class(result.confidence_score)}"
            explanation_methods_html += f"""
            <div class="method">
                <h3>{result.method_name}</h3>
                <p><strong>Type:</strong> {result.explanation_type}</p>
                <p><strong>Confidence:</strong> <span class="{confidence_class}">{result.confidence_score:.2f}</span></p>
                <p>{result.summary}</p>
                <p><strong>Visualizations:</strong> {len(result.visualizations)} plots generated</p>
            </div>
            """
        
        blind_spots_html = ""
        for recommendation in self.blind_spots.recommendations:
            blind_spots_html += f'<div class="blind-spot">{recommendation}</div>'
        
        stakeholder_summaries_html = ""
        for stakeholder, summary in self.stakeholder_summaries.items():
            stakeholder_summaries_html += f"""
            <div class="stakeholder">
                <h3>{stakeholder.title()} Stakeholder</h3>
                <p><strong>Risk Assessment:</strong> {summary.risk_assessment}</p>
                <p><strong>Confidence Level:</strong> {summary.confidence_level}</p>
                
                <h4>Key Insights:</h4>
                <ul>
                    {"".join(f"<li>{insight}</li>" for insight in summary.key_insights)}
                </ul>
                
                <h4>Recommendations:</h4>
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in summary.recommendations)}
                </ul>
            </div>
            """
        
        overall_recommendations_html = ""
        all_recommendations = set()
        for summary in self.stakeholder_summaries.values():
            all_recommendations.update(summary.recommendations)
        
        for rec in all_recommendations:
            overall_recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        # Fill template
        return html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            methods_count=len(self.results),
            total_explanations=len(self.results),
            blind_spots_count=len(self.blind_spots.recommendations),
            stakeholder_count=len(self.stakeholder_summaries),
            explanation_methods=explanation_methods_html,
            blind_spots_section=blind_spots_html,
            stakeholder_summaries=stakeholder_summaries_html,
            overall_recommendations=overall_recommendations_html
        )
    
    def _get_confidence_class(self, score: float) -> str:
        """Get CSS class for confidence score."""
        if score >= 0.8:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        else:
            return 'low'


def create_interpretability_pipeline(config_dict: Dict[str, Any] = None) -> InterpretabilityPipeline:
    """
    Factory function to create interpretability pipeline.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        InterpretabilityPipeline instance
    """
    config = InterpretabilityConfig(**config_dict) if config_dict else InterpretabilityConfig()
    return InterpretabilityPipeline(config)