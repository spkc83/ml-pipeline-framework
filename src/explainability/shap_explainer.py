import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# SHAP imports with error handling
try:
    import shap
    from shap import Explainer, TreeExplainer, LinearExplainer, KernelExplainer, DeepExplainer
    from shap import Explanation
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False
    shap = Explainer = TreeExplainer = LinearExplainer = KernelExplainer = DeepExplainer = Explanation = None

# Additional imports
try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = None


class SHAPError(Exception):
    pass


class SHAPExplainer:
    """
    Comprehensive SHAP (SHapley Additive exPlanations) explainer supporting
    global and local explanations across different model types.
    """
    
    def __init__(self, model: Any, X_background: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                 feature_names: Optional[List[str]] = None, class_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            X_background: Background dataset for SHAP explainer (optional)
            feature_names: Names of features
            class_names: Names of classes (for classification)
        """
        if not SHAP_AVAILABLE:
            raise SHAPError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        self.explainer_type = None
        self.shap_values = None
        self.base_value = None
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            # Auto-detect explainer type based on model
            model_type = type(self.model).__name__
            module_name = type(self.model).__module__
            
            logger.info(f"Initializing SHAP explainer for model: {model_type} from {module_name}")
            
            # Tree-based models
            if any(keyword in model_type.lower() for keyword in 
                   ['tree', 'forest', 'xgb', 'lgb', 'catboost', 'gbm', 'gradient']):
                self._initialize_tree_explainer()
            
            # Linear models
            elif any(keyword in model_type.lower() for keyword in 
                     ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
                self._initialize_linear_explainer()
            
            # Deep learning models
            elif any(keyword in module_name.lower() for keyword in 
                     ['torch', 'tensorflow', 'keras']):
                self._initialize_deep_explainer()
            
            # Default to kernel explainer
            else:
                self._initialize_kernel_explainer()
                
        except Exception as e:
            logger.warning(f"Failed to initialize specific explainer: {e}. Falling back to kernel explainer.")
            self._initialize_kernel_explainer()
    
    def _initialize_tree_explainer(self) -> None:
        """Initialize tree explainer for tree-based models."""
        try:
            self.explainer = TreeExplainer(self.model)
            self.explainer_type = "tree"
            logger.info("Initialized TreeExplainer")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to kernel explainer.")
            self._initialize_kernel_explainer()
    
    def _initialize_linear_explainer(self) -> None:
        """Initialize linear explainer for linear models."""
        try:
            # Some linear models need masker
            if self.X_background is not None:
                masker = shap.maskers.Independent(self.X_background)
                self.explainer = LinearExplainer(self.model, masker)
            else:
                self.explainer = LinearExplainer(self.model)
            self.explainer_type = "linear"
            logger.info("Initialized LinearExplainer")
        except Exception as e:
            logger.warning(f"LinearExplainer failed: {e}. Falling back to kernel explainer.")
            self._initialize_kernel_explainer()
    
    def _initialize_deep_explainer(self) -> None:
        """Initialize deep explainer for neural networks."""
        try:
            if self.X_background is None:
                raise ValueError("Background data required for DeepExplainer")
            
            self.explainer = DeepExplainer(self.model, self.X_background)
            self.explainer_type = "deep"
            logger.info("Initialized DeepExplainer")
        except Exception as e:
            logger.warning(f"DeepExplainer failed: {e}. Falling back to kernel explainer.")
            self._initialize_kernel_explainer()
    
    def _initialize_kernel_explainer(self) -> None:
        """Initialize kernel explainer (model-agnostic)."""
        try:
            # Create prediction function
            def predict_fn(X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                else:
                    return self.model.predict(X)
            
            # Use background data or sample
            background = self.X_background
            if background is None and hasattr(self.model, 'X_train_'):
                # Use a sample of training data if available
                background = shap.sample(self.model.X_train_, 100)
            elif background is None:
                logger.warning("No background data provided for KernelExplainer")
                background = np.zeros((1, len(self.feature_names) if self.feature_names else 1))
            
            self.explainer = KernelExplainer(predict_fn, background)
            self.explainer_type = "kernel"
            logger.info("Initialized KernelExplainer")
            
        except Exception as e:
            logger.error(f"Failed to initialize any SHAP explainer: {e}")
            raise SHAPError(f"Could not initialize SHAP explainer: {e}")
    
    def explain(self, X: Union[pd.DataFrame, np.ndarray], 
               max_evals: int = 500) -> Explanation:
        """
        Generate SHAP explanations for given data.
        
        Args:
            X: Data to explain
            max_evals: Maximum evaluations for kernel explainer
            
        Returns:
            SHAP Explanation object
        """
        logger.info(f"Generating SHAP explanations for {len(X)} samples")
        
        try:
            # Convert to numpy if needed
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            
            # Generate explanations
            if self.explainer_type == "kernel":
                explanations = self.explainer(X_array, max_evals=max_evals)
            else:
                explanations = self.explainer(X_array)
            
            # Store for later use
            self.shap_values = explanations.values
            self.base_value = explanations.base_values
            
            # Set feature names if available
            if self.feature_names is not None:
                explanations.feature_names = self.feature_names
            elif isinstance(X, pd.DataFrame):
                explanations.feature_names = list(X.columns)
            
            logger.info("SHAP explanations generated successfully")
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            raise SHAPError(f"SHAP explanation failed: {e}")
    
    def global_explanations(self, X: Union[pd.DataFrame, np.ndarray], 
                          max_evals: int = 500) -> Dict[str, Any]:
        """
        Generate global feature importance explanations.
        
        Args:
            X: Data to explain
            max_evals: Maximum evaluations for kernel explainer
            
        Returns:
            Dictionary with global explanation results
        """
        logger.info("Generating global SHAP explanations")
        
        explanations = self.explain(X, max_evals)
        
        # Calculate global importance metrics
        shap_values = explanations.values
        
        # Mean absolute SHAP values (global importance)
        if shap_values.ndim == 3:  # Multi-class
            # Take mean across classes
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Feature names
        feature_names = (explanations.feature_names if hasattr(explanations, 'feature_names') 
                        else self.feature_names or [f"feature_{i}" for i in range(len(mean_shap))])
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        results = {
            'feature_importance': importance_df,
            'shap_values': shap_values,
            'explanations': explanations,
            'mean_abs_shap': mean_shap
        }
        
        logger.info("Global explanations completed")
        return results
    
    def local_explanations(self, X: Union[pd.DataFrame, np.ndarray], 
                          indices: Optional[List[int]] = None,
                          max_evals: int = 500) -> Dict[str, Any]:
        """
        Generate local explanations for specific instances.
        
        Args:
            X: Data to explain
            indices: Specific indices to explain (None for all)
            max_evals: Maximum evaluations for kernel explainer
            
        Returns:
            Dictionary with local explanation results
        """
        logger.info("Generating local SHAP explanations")
        
        explanations = self.explain(X, max_evals)
        
        if indices is not None:
            # Filter to specific indices
            explanations_subset = explanations[indices]
        else:
            explanations_subset = explanations
        
        results = {
            'explanations': explanations_subset,
            'shap_values': explanations_subset.values,
            'base_values': explanations_subset.base_values,
            'data': explanations_subset.data if hasattr(explanations_subset, 'data') else X
        }
        
        logger.info("Local explanations completed")
        return results
    
    def summary_plot(self, X: Union[pd.DataFrame, np.ndarray], 
                    plot_type: str = "dot", max_display: int = 20,
                    figsize: Tuple[int, int] = (10, 8), save_path: Optional[str] = None) -> None:
        """
        Generate SHAP summary plot.
        
        Args:
            X: Data to explain
            plot_type: Type of plot ("dot", "bar", "violin")
            max_display: Maximum features to display
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating SHAP summary plot ({plot_type})")
        
        explanations = self.explain(X)
        
        plt.figure(figsize=figsize)
        
        try:
            if plot_type == "dot":
                shap.summary_plot(explanations, max_display=max_display, show=False)
            elif plot_type == "bar":
                shap.summary_plot(explanations, plot_type="bar", max_display=max_display, show=False)
            elif plot_type == "violin":
                shap.summary_plot(explanations, plot_type="violin", max_display=max_display, show=False)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            plt.title(f"SHAP Summary Plot ({plot_type.title()})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Summary plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate summary plot: {e}")
            plt.close()
            raise SHAPError(f"Summary plot generation failed: {e}")
    
    def dependence_plot(self, X: Union[pd.DataFrame, np.ndarray], 
                       feature: Union[str, int], interaction_feature: Union[str, int, None] = "auto",
                       figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> None:
        """
        Generate SHAP dependence plot.
        
        Args:
            X: Data to explain
            feature: Feature to plot
            interaction_feature: Feature for interaction coloring ("auto" for automatic selection)
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating SHAP dependence plot for feature: {feature}")
        
        explanations = self.explain(X)
        
        plt.figure(figsize=figsize)
        
        try:
            shap.dependence_plot(
                feature, 
                explanations.values, 
                explanations.data if hasattr(explanations, 'data') else X,
                feature_names=explanations.feature_names if hasattr(explanations, 'feature_names') else None,
                interaction_index=interaction_feature,
                show=False
            )
            
            plt.title(f"SHAP Dependence Plot: {feature}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dependence plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate dependence plot: {e}")
            plt.close()
            raise SHAPError(f"Dependence plot generation failed: {e}")
    
    def waterfall_plot(self, X: Union[pd.DataFrame, np.ndarray], 
                      index: int = 0, max_display: int = 20,
                      figsize: Tuple[int, int] = (10, 8), save_path: Optional[str] = None) -> None:
        """
        Generate SHAP waterfall plot for a single prediction.
        
        Args:
            X: Data to explain
            index: Index of instance to explain
            max_display: Maximum features to display
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating SHAP waterfall plot for index: {index}")
        
        explanations = self.explain(X)
        
        # Select single instance
        single_explanation = explanations[index]
        
        plt.figure(figsize=figsize)
        
        try:
            shap.waterfall_plot(single_explanation, max_display=max_display, show=False)
            
            plt.title(f"SHAP Waterfall Plot (Instance {index})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate waterfall plot: {e}")
            plt.close()
            raise SHAPError(f"Waterfall plot generation failed: {e}")
    
    def force_plot(self, X: Union[pd.DataFrame, np.ndarray], 
                  index: int = 0, matplotlib: bool = False,
                  figsize: Tuple[int, int] = (16, 4), save_path: Optional[str] = None) -> None:
        """
        Generate SHAP force plot for a single prediction.
        
        Args:
            X: Data to explain
            index: Index of instance to explain
            matplotlib: Whether to use matplotlib (vs. interactive)
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating SHAP force plot for index: {index}")
        
        explanations = self.explain(X)
        
        try:
            if matplotlib:
                plt.figure(figsize=figsize)
                shap.force_plot(
                    explanations.base_values[index],
                    explanations.values[index],
                    explanations.data[index] if hasattr(explanations, 'data') else X[index],
                    feature_names=explanations.feature_names if hasattr(explanations, 'feature_names') else None,
                    matplotlib=True,
                    show=False
                )
                
                plt.title(f"SHAP Force Plot (Instance {index})", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Force plot saved to: {save_path}")
                
                plt.show()
            else:
                # Interactive plot
                force_plot = shap.force_plot(
                    explanations.base_values[index],
                    explanations.values[index],
                    explanations.data[index] if hasattr(explanations, 'data') else X[index],
                    feature_names=explanations.feature_names if hasattr(explanations, 'feature_names') else None
                )
                return force_plot
                
        except Exception as e:
            logger.error(f"Failed to generate force plot: {e}")
            if matplotlib:
                plt.close()
            raise SHAPError(f"Force plot generation failed: {e}")
    
    def partial_dependence_plot(self, X: Union[pd.DataFrame, np.ndarray], 
                               features: List[Union[str, int]], 
                               ice: bool = False, kind: str = "average",
                               figsize: Tuple[int, int] = (12, 4), save_path: Optional[str] = None) -> None:
        """
        Generate SHAP-based partial dependence plots.
        
        Args:
            X: Data to explain
            features: Features to plot
            ice: Whether to show individual conditional expectation curves
            kind: Type of plot ("average", "individual", "both")
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating SHAP partial dependence plots for features: {features}")
        
        explanations = self.explain(X)
        
        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(figsize[0] * n_features, figsize[1]))
        if n_features == 1:
            axes = [axes]
        
        try:
            for i, feature in enumerate(features):
                shap.partial_dependence_plot(
                    feature,
                    self.model.predict if hasattr(self.model, 'predict') else self.model,
                    explanations.data if hasattr(explanations, 'data') else X,
                    ice=ice,
                    model_expected_value=True,
                    feature_expected_value=True,
                    ax=axes[i],
                    show=False
                )
                axes[i].set_title(f"Partial Dependence: {feature}")
            
            plt.suptitle("SHAP Partial Dependence Plots", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Partial dependence plots saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate partial dependence plots: {e}")
            plt.close()
            raise SHAPError(f"Partial dependence plot generation failed: {e}")
    
    def generate_explanation_report(self, X: Union[pd.DataFrame, np.ndarray],
                                   output_dir: str = "./shap_explanations",
                                   include_plots: bool = True,
                                   max_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate comprehensive SHAP explanation report.
        
        Args:
            X: Data to explain
            output_dir: Directory to save outputs
            include_plots: Whether to generate and save plots
            max_samples: Maximum samples to use for analysis
            
        Returns:
            Dictionary with explanation results and file paths
        """
        logger.info("Generating comprehensive SHAP explanation report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample data if too large
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices] if isinstance(X, pd.DataFrame) else X[sample_indices]
            logger.info(f"Sampled {max_samples} instances from {len(X)} total")
        else:
            X_sample = X
            sample_indices = np.arange(len(X))
        
        results = {}
        
        try:
            # Global explanations
            global_results = self.global_explanations(X_sample)
            results['global'] = global_results
            
            # Save feature importance
            importance_path = output_path / "feature_importance.csv"
            global_results['feature_importance'].to_csv(importance_path, index=False)
            results['feature_importance_path'] = str(importance_path)
            
            if include_plots:
                # Summary plots
                summary_dot_path = output_path / "summary_plot_dot.png"
                self.summary_plot(X_sample, plot_type="dot", save_path=str(summary_dot_path))
                results['summary_dot_path'] = str(summary_dot_path)
                
                summary_bar_path = output_path / "summary_plot_bar.png"
                self.summary_plot(X_sample, plot_type="bar", save_path=str(summary_bar_path))
                results['summary_bar_path'] = str(summary_bar_path)
                
                # Dependence plots for top features
                top_features = global_results['feature_importance']['feature'].head(5).tolist()
                dependence_paths = []
                
                for feature in top_features:
                    dependence_path = output_path / f"dependence_plot_{feature}.png"
                    self.dependence_plot(X_sample, feature, save_path=str(dependence_path))
                    dependence_paths.append(str(dependence_path))
                
                results['dependence_paths'] = dependence_paths
                
                # Waterfall plots for first few instances
                waterfall_paths = []
                for i in range(min(3, len(X_sample))):
                    waterfall_path = output_path / f"waterfall_plot_instance_{i}.png"
                    self.waterfall_plot(X_sample, index=i, save_path=str(waterfall_path))
                    waterfall_paths.append(str(waterfall_path))
                
                results['waterfall_paths'] = waterfall_paths
            
            # Local explanations for sample instances
            local_results = self.local_explanations(X_sample, indices=list(range(min(10, len(X_sample)))))
            results['local'] = local_results
            
            # Summary statistics
            explanations = local_results['explanations']
            shap_values = explanations.values
            
            summary_stats = {
                'n_samples_explained': len(X_sample),
                'n_features': shap_values.shape[1],
                'mean_abs_shap_per_feature': np.mean(np.abs(shap_values), axis=0).tolist(),
                'total_explanation_magnitude': np.sum(np.abs(shap_values)),
                'explanation_variance': np.var(np.sum(np.abs(shap_values), axis=1))
            }
            results['summary_stats'] = summary_stats
            
            # Save summary to JSON
            summary_path = output_path / "explanation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            results['summary_path'] = str(summary_path)
            
            logger.info(f"SHAP explanation report generated in: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            raise SHAPError(f"Report generation failed: {e}")
        
        return results
    
    def get_explainer_info(self) -> Dict[str, Any]:
        """
        Get information about the SHAP explainer.
        
        Returns:
            Dictionary with explainer information
        """
        return {
            'explainer_type': self.explainer_type,
            'model_type': type(self.model).__name__,
            'has_background_data': self.X_background is not None,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'shap_version': shap.__version__ if SHAP_AVAILABLE else None
        }
    
    def explain_prediction_difference(self, X1: Union[pd.DataFrame, np.ndarray],
                                    X2: Union[pd.DataFrame, np.ndarray],
                                    index1: int = 0, index2: int = 0) -> Dict[str, Any]:
        """
        Explain the difference between two predictions.
        
        Args:
            X1: First dataset
            X2: Second dataset  
            index1: Index in first dataset
            index2: Index in second dataset
            
        Returns:
            Dictionary with difference explanation
        """
        logger.info(f"Explaining prediction difference between instances {index1} and {index2}")
        
        # Get explanations for both instances
        exp1 = self.explain(X1[index1:index1+1] if isinstance(X1, np.ndarray) else X1.iloc[index1:index1+1])
        exp2 = self.explain(X2[index2:index2+1] if isinstance(X2, np.ndarray) else X2.iloc[index2:index2+1])
        
        # Calculate differences
        shap_diff = exp2.values[0] - exp1.values[0]
        value_diff = (X2[index2] if isinstance(X2, np.ndarray) else X2.iloc[index2].values) - \
                    (X1[index1] if isinstance(X1, np.ndarray) else X1.iloc[index1].values)
        
        # Feature names
        feature_names = (exp1.feature_names if hasattr(exp1, 'feature_names') 
                        else self.feature_names or [f"feature_{i}" for i in range(len(shap_diff))])
        
        # Create difference DataFrame
        diff_df = pd.DataFrame({
            'feature': feature_names,
            'shap_difference': shap_diff,
            'value_difference': value_diff,
            'abs_shap_difference': np.abs(shap_diff)
        }).sort_values('abs_shap_difference', ascending=False)
        
        results = {
            'difference_analysis': diff_df,
            'total_shap_difference': np.sum(shap_diff),
            'explanation_1': exp1,
            'explanation_2': exp2,
            'prediction_difference': exp2.base_values[0] + np.sum(exp2.values[0]) - 
                                   (exp1.base_values[0] + np.sum(exp1.values[0]))
        }
        
        return results