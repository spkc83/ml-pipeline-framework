import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

logger = logging.getLogger(__name__)

# Scikit-learn imports
try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    from sklearn.utils import check_array
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn not available for PDP/ICE analysis")
    SKLEARN_AVAILABLE = False
    partial_dependence = PartialDependenceDisplay = check_array = BaseEstimator = None

# Additional plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available for interactive PDP/ICE plots")
    PLOTLY_AVAILABLE = False
    go = px = make_subplots = None


class PDPError(Exception):
    pass


class PartialDependenceAnalyzer:
    """
    Comprehensive Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE)
    analyzer supporting numerical and categorical features with bivariate analysis.
    """
    
    def __init__(self, model: Any, X: Union[pd.DataFrame, np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 categorical_features: Optional[List[Union[str, int]]] = None,
                 target_names: Optional[List[str]] = None):
        """
        Initialize Partial Dependence Analyzer.
        
        Args:
            model: Trained model with predict method
            X: Training/reference data
            feature_names: Names of features
            categorical_features: List of categorical feature names/indices
            target_names: Names of target classes (for classification)
        """
        if not SKLEARN_AVAILABLE:
            raise PDPError("Scikit-learn is required for PDP/ICE analysis. Install with: pip install scikit-learn")
        
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.target_names = target_names
        
        # Infer feature names if not provided
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert X to DataFrame for easier handling
        if not isinstance(self.X, pd.DataFrame):
            self.X = pd.DataFrame(self.X, columns=self.feature_names)
        
        # Identify numerical and categorical features
        self._identify_feature_types()
        
        # Validate model
        if not hasattr(model, 'predict'):
            raise PDPError("Model must have a 'predict' method")
        
        logger.info(f"Initialized PDP analyzer with {len(self.feature_names)} features "
                   f"({len(self.numerical_features)} numerical, {len(self.categorical_features)} categorical)")
    
    def _identify_feature_types(self) -> None:
        """Identify numerical and categorical features."""
        self.numerical_features = []
        self.categorical_feature_names = []
        
        for i, feature in enumerate(self.feature_names):
            if feature in self.categorical_features or i in self.categorical_features:
                self.categorical_feature_names.append(feature)
            else:
                # Check if feature appears to be numerical
                if self.X[feature].dtype in ['int64', 'float64']:
                    # Additional check: if unique values are few, might be categorical
                    n_unique = self.X[feature].nunique()
                    if n_unique <= 10 and self.X[feature].dtype == 'int64':
                        logger.info(f"Feature {feature} appears categorical (few unique values: {n_unique})")
                        self.categorical_feature_names.append(feature)
                    else:
                        self.numerical_features.append(feature)
                else:
                    self.categorical_feature_names.append(feature)
        
        logger.debug(f"Numerical features: {self.numerical_features}")
        logger.debug(f"Categorical features: {self.categorical_feature_names}")
    
    def partial_dependence_1d(self, feature: Union[str, int], 
                             grid_resolution: int = 50,
                             percentiles: Tuple[float, float] = (0.05, 0.95),
                             method: str = "auto") -> Dict[str, Any]:
        """
        Calculate 1D partial dependence for a single feature.
        
        Args:
            feature: Feature name or index
            grid_resolution: Number of points in the grid
            percentiles: Percentiles for grid range (numerical features)
            method: PD calculation method ("auto", "recursion", "brute")
            
        Returns:
            Dictionary with PD results
        """
        logger.info(f"Calculating 1D partial dependence for feature: {feature}")
        
        try:
            # Get feature index
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature)
                feature_name = feature
            else:
                feature_idx = feature
                feature_name = self.feature_names[feature_idx]
            
            # Check if feature is categorical
            is_categorical = feature_name in self.categorical_feature_names
            
            if is_categorical:
                # For categorical features, use unique values
                unique_values = sorted(self.X[feature_name].unique())
                grid_values = unique_values
            else:
                # For numerical features, create grid based on percentiles
                feature_values = self.X[feature_name]
                min_val = feature_values.quantile(percentiles[0])
                max_val = feature_values.quantile(percentiles[1])
                grid_values = np.linspace(min_val, max_val, grid_resolution)
            
            # Calculate partial dependence
            pd_results = partial_dependence(
                self.model,
                self.X,
                features=[feature_idx],
                grid_resolution=len(grid_values) if is_categorical else grid_resolution,
                percentiles=percentiles,
                method=method
            )
            
            partial_dependence_values = pd_results['average'][0]
            grid_values_used = pd_results['grid_values'][0]
            
            # Prepare results
            results = {
                'feature_name': feature_name,
                'feature_index': feature_idx,
                'is_categorical': is_categorical,
                'grid_values': grid_values_used,
                'partial_dependence': partial_dependence_values,
                'method': method,
                'percentiles': percentiles
            }
            
            logger.info(f"Partial dependence calculated for {feature_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate partial dependence for {feature}: {e}")
            raise PDPError(f"PD calculation failed: {e}")
    
    def ice_curves(self, feature: Union[str, int],
                  sample_size: Optional[int] = None,
                  grid_resolution: int = 50,
                  percentiles: Tuple[float, float] = (0.05, 0.95),
                  method: str = "auto") -> Dict[str, Any]:
        """
        Calculate Individual Conditional Expectation (ICE) curves.
        
        Args:
            feature: Feature name or index
            sample_size: Number of samples to use for ICE (None for all)
            grid_resolution: Number of points in the grid
            percentiles: Percentiles for grid range
            method: ICE calculation method
            
        Returns:
            Dictionary with ICE results
        """
        logger.info(f"Calculating ICE curves for feature: {feature}")
        
        try:
            # Sample data if requested
            if sample_size is not None and len(self.X) > sample_size:
                sample_indices = np.random.choice(len(self.X), sample_size, replace=False)
                X_sample = self.X.iloc[sample_indices]
            else:
                X_sample = self.X
                sample_indices = np.arange(len(self.X))
            
            # Get feature info
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature)
                feature_name = feature
            else:
                feature_idx = feature
                feature_name = self.feature_names[feature_idx]
            
            is_categorical = feature_name in self.categorical_feature_names
            
            # Create grid
            if is_categorical:
                unique_values = sorted(self.X[feature_name].unique())
                grid_values = unique_values
            else:
                feature_values = self.X[feature_name]
                min_val = feature_values.quantile(percentiles[0])
                max_val = feature_values.quantile(percentiles[1])
                grid_values = np.linspace(min_val, max_val, grid_resolution)
            
            # Calculate ICE curves
            ice_data = []
            
            for i, (_, row) in enumerate(X_sample.iterrows()):
                instance_predictions = []
                
                for grid_val in grid_values:
                    # Create modified instance
                    modified_row = row.copy()
                    modified_row[feature_name] = grid_val
                    
                    # Get prediction
                    X_modified = pd.DataFrame([modified_row])
                    if hasattr(self.model, 'predict_proba'):
                        pred = self.model.predict_proba(X_modified)[0]
                        # For binary classification, use positive class probability
                        if len(pred) == 2:
                            pred = pred[1]
                        else:
                            pred = pred  # Keep all probabilities for multiclass
                    else:
                        pred = self.model.predict(X_modified)[0]
                    
                    instance_predictions.append(pred)
                
                ice_data.append({
                    'instance_id': sample_indices[i] if sample_size is not None else i,
                    'predictions': instance_predictions
                })
            
            # Calculate average (PDP)
            all_predictions = np.array([ice['predictions'] for ice in ice_data])
            average_predictions = np.mean(all_predictions, axis=0)
            
            results = {
                'feature_name': feature_name,
                'feature_index': feature_idx,
                'is_categorical': is_categorical,
                'grid_values': grid_values,
                'ice_curves': ice_data,
                'average_predictions': average_predictions,
                'individual_predictions': all_predictions,
                'n_instances': len(ice_data),
                'method': method
            }
            
            logger.info(f"ICE curves calculated for {feature_name} ({len(ice_data)} instances)")
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate ICE curves for {feature}: {e}")
            raise PDPError(f"ICE calculation failed: {e}")
    
    def partial_dependence_2d(self, features: Tuple[Union[str, int], Union[str, int]],
                             grid_resolution: int = 20,
                             percentiles: Tuple[float, float] = (0.05, 0.95),
                             method: str = "auto") -> Dict[str, Any]:
        """
        Calculate 2D partial dependence for feature interactions.
        
        Args:
            features: Tuple of two feature names or indices
            grid_resolution: Number of points in each dimension
            percentiles: Percentiles for grid range
            method: PD calculation method
            
        Returns:
            Dictionary with 2D PD results
        """
        logger.info(f"Calculating 2D partial dependence for features: {features}")
        
        try:
            # Get feature indices and names
            feature_indices = []
            feature_names = []
            
            for feature in features:
                if isinstance(feature, str):
                    idx = self.feature_names.index(feature)
                    name = feature
                else:
                    idx = feature
                    name = self.feature_names[feature]
                
                feature_indices.append(idx)
                feature_names.append(name)
            
            # Calculate 2D partial dependence
            pd_results = partial_dependence(
                self.model,
                self.X,
                features=feature_indices,
                grid_resolution=grid_resolution,
                percentiles=percentiles,
                method=method
            )
            
            partial_dependence_values = pd_results['average'][0]
            grid_values = pd_results['grid_values']
            
            results = {
                'feature_names': feature_names,
                'feature_indices': feature_indices,
                'grid_values': grid_values,
                'partial_dependence': partial_dependence_values,
                'grid_resolution': grid_resolution,
                'method': method,
                'percentiles': percentiles
            }
            
            logger.info(f"2D partial dependence calculated for {feature_names}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate 2D partial dependence for {features}: {e}")
            raise PDPError(f"2D PD calculation failed: {e}")
    
    def plot_partial_dependence(self, feature: Union[str, int],
                               include_ice: bool = False,
                               ice_sample_size: int = 100,
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[str] = None) -> None:
        """
        Plot 1D partial dependence with optional ICE curves.
        
        Args:
            feature: Feature to plot
            include_ice: Whether to include ICE curves
            ice_sample_size: Number of ICE curves to show
            figsize: Figure size
            save_path: Path to save plot
        """
        logger.info(f"Plotting partial dependence for feature: {feature}")
        
        # Calculate PD
        pd_results = self.partial_dependence_1d(feature)
        
        # Calculate ICE if requested
        ice_results = None
        if include_ice:
            ice_results = self.ice_curves(feature, sample_size=ice_sample_size)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot ICE curves first (so PD line is on top)
        if ice_results:
            for i, ice_curve in enumerate(ice_results['ice_curves']):
                alpha = min(0.5, 50 / len(ice_results['ice_curves']))  # Adjust transparency
                plt.plot(ice_results['grid_values'], ice_curve['predictions'], 
                        color='lightblue', alpha=alpha, linewidth=0.5,
                        label='ICE curves' if i == 0 else "")
        
        # Plot PD line
        plt.plot(pd_results['grid_values'], pd_results['partial_dependence'],
                color='red', linewidth=3, label='Partial Dependence')
        
        # Customize plot
        plt.xlabel(pd_results['feature_name'])
        plt.ylabel('Partial Dependence')
        plt.title(f"Partial Dependence Plot: {pd_results['feature_name']}")
        
        if include_ice:
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_partial_dependence_2d(self, features: Tuple[Union[str, int], Union[str, int]],
                                  plot_type: str = "contour",
                                  figsize: Tuple[int, int] = (10, 8),
                                  save_path: Optional[str] = None) -> None:
        """
        Plot 2D partial dependence heatmap/contour.
        
        Args:
            features: Tuple of two features
            plot_type: Type of plot ("contour", "heatmap", "3d")
            figsize: Figure size
            save_path: Path to save plot
        """
        logger.info(f"Plotting 2D partial dependence for features: {features}")
        
        # Calculate 2D PD
        pd_results = self.partial_dependence_2d(features)
        
        # Create plot
        if plot_type == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            X_grid, Y_grid = np.meshgrid(pd_results['grid_values'][0], pd_results['grid_values'][1])
            ax.plot_surface(X_grid, Y_grid, pd_results['partial_dependence'].T, 
                           cmap='viridis', alpha=0.8)
            
            ax.set_xlabel(pd_results['feature_names'][0])
            ax.set_ylabel(pd_results['feature_names'][1])
            ax.set_zlabel('Partial Dependence')
            ax.set_title(f"2D Partial Dependence: {' vs '.join(pd_results['feature_names'])}")
            
        else:
            plt.figure(figsize=figsize)
            
            if plot_type == "contour":
                X_grid, Y_grid = np.meshgrid(pd_results['grid_values'][0], pd_results['grid_values'][1])
                cs = plt.contour(X_grid, Y_grid, pd_results['partial_dependence'].T, levels=15)
                plt.clabel(cs, inline=True, fontsize=8)
                plt.contourf(X_grid, Y_grid, pd_results['partial_dependence'].T, 
                           levels=50, alpha=0.6, cmap='viridis')
                
            elif plot_type == "heatmap":
                plt.imshow(pd_results['partial_dependence'].T, 
                          extent=[pd_results['grid_values'][0].min(), pd_results['grid_values'][0].max(),
                                 pd_results['grid_values'][1].min(), pd_results['grid_values'][1].max()],
                          aspect='auto', origin='lower', cmap='viridis')
            
            plt.colorbar(label='Partial Dependence')
            plt.xlabel(pd_results['feature_names'][0])
            plt.ylabel(pd_results['feature_names'][1])
            plt.title(f"2D Partial Dependence: {' vs '.join(pd_results['feature_names'])}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D plot saved to: {save_path}")
        
        plt.show()
    
    def plot_ice_curves(self, feature: Union[str, int],
                       sample_size: int = 100,
                       show_individual: bool = True,
                       show_average: bool = True,
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None) -> None:
        """
        Plot ICE curves with optional average (PD) line.
        
        Args:
            feature: Feature to plot
            sample_size: Number of ICE curves to show
            show_individual: Whether to show individual curves
            show_average: Whether to show average (PD) line
            figsize: Figure size
            save_path: Path to save plot
        """
        logger.info(f"Plotting ICE curves for feature: {feature}")
        
        # Calculate ICE
        ice_results = self.ice_curves(feature, sample_size=sample_size)
        
        plt.figure(figsize=figsize)
        
        # Plot individual ICE curves
        if show_individual:
            for i, ice_curve in enumerate(ice_results['ice_curves']):
                alpha = min(0.6, 100 / len(ice_results['ice_curves']))
                plt.plot(ice_results['grid_values'], ice_curve['predictions'],
                        color='lightblue', alpha=alpha, linewidth=0.8,
                        label='ICE curves' if i == 0 else "")
        
        # Plot average (PD) line
        if show_average:
            plt.plot(ice_results['grid_values'], ice_results['average_predictions'],
                    color='red', linewidth=3, label='Average (PD)')
        
        plt.xlabel(ice_results['feature_name'])
        plt.ylabel('Prediction')
        plt.title(f"ICE Curves: {ice_results['feature_name']} ({len(ice_results['ice_curves'])} instances)")
        
        if show_individual or show_average:
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ICE plot saved to: {save_path}")
        
        plt.show()
    
    def analyze_feature_interactions(self, max_features: int = 10,
                                   interaction_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Analyze feature interactions using variance in ICE curves.
        
        Args:
            max_features: Maximum number of features to analyze
            interaction_threshold: Threshold for significant interactions
            
        Returns:
            Dictionary with interaction analysis results
        """
        logger.info("Analyzing feature interactions using ICE variance")
        
        # Select features to analyze
        features_to_analyze = self.numerical_features[:max_features]
        
        interaction_results = {}
        
        for feature in features_to_analyze:
            try:
                # Calculate ICE curves
                ice_results = self.ice_curves(feature, sample_size=200)
                
                # Calculate variance across ICE curves for each grid point
                ice_matrix = ice_results['individual_predictions']
                ice_variance = np.var(ice_matrix, axis=0)
                mean_variance = np.mean(ice_variance)
                
                # High variance indicates interactions
                interaction_strength = mean_variance / (np.mean(ice_results['average_predictions']) + 1e-8)
                
                interaction_results[feature] = {
                    'mean_ice_variance': mean_variance,
                    'interaction_strength': interaction_strength,
                    'has_interactions': interaction_strength > interaction_threshold,
                    'grid_variances': ice_variance
                }
                
                logger.debug(f"Feature {feature}: interaction strength = {interaction_strength:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze interactions for {feature}: {e}")
                continue
        
        # Sort by interaction strength
        sorted_interactions = sorted(interaction_results.items(), 
                                   key=lambda x: x[1]['interaction_strength'], 
                                   reverse=True)
        
        results = {
            'feature_interactions': dict(sorted_interactions),
            'high_interaction_features': [f for f, r in sorted_interactions 
                                        if r['has_interactions']],
            'interaction_threshold': interaction_threshold,
            'summary': {
                'n_features_analyzed': len(features_to_analyze),
                'n_high_interaction': len([r for r in interaction_results.values() 
                                         if r['has_interactions']]),
                'mean_interaction_strength': np.mean([r['interaction_strength'] 
                                                    for r in interaction_results.values()])
            }
        }
        
        logger.info(f"Interaction analysis completed. {results['summary']['n_high_interaction']} features show high interactions")
        return results
    
    def generate_pdp_report(self, features: Optional[List[Union[str, int]]] = None,
                           output_dir: str = "./pdp_analysis",
                           include_ice: bool = True,
                           include_2d: bool = True,
                           max_2d_combinations: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive PDP/ICE analysis report.
        
        Args:
            features: Features to analyze (None for all numerical features)
            output_dir: Directory to save outputs
            include_ice: Whether to include ICE analysis
            include_2d: Whether to include 2D PDP analysis
            max_2d_combinations: Maximum number of 2D combinations
            
        Returns:
            Dictionary with analysis results and file paths
        """
        logger.info("Generating comprehensive PDP/ICE analysis report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select features
        if features is None:
            features = self.numerical_features[:10]  # Limit to avoid too many plots
        
        results = {}
        
        try:
            # 1D PDP analysis
            pd_results_1d = {}
            for feature in features:
                pd_result = self.partial_dependence_1d(feature)
                pd_results_1d[feature] = pd_result
                
                # Generate plots
                plot_path = output_path / f"pdp_1d_{feature}.png"
                self.plot_partial_dependence(feature, include_ice=include_ice, 
                                            save_path=str(plot_path))
            
            results['1d_partial_dependence'] = pd_results_1d
            
            # ICE analysis
            if include_ice:
                ice_results = {}
                for feature in features:
                    ice_result = self.ice_curves(feature)
                    ice_results[feature] = ice_result
                    
                    # Generate ICE plots
                    ice_plot_path = output_path / f"ice_{feature}.png"
                    self.plot_ice_curves(feature, save_path=str(ice_plot_path))
                
                results['ice_curves'] = ice_results
            
            # 2D PDP analysis
            if include_2d and len(features) >= 2:
                # Select top combinations based on feature importance or correlation
                feature_combinations = list(combinations(features[:5], 2))[:max_2d_combinations]
                
                pd_results_2d = {}
                for feature_pair in feature_combinations:
                    try:
                        pd_result_2d = self.partial_dependence_2d(feature_pair)
                        pd_results_2d[f"{feature_pair[0]}_vs_{feature_pair[1]}"] = pd_result_2d
                        
                        # Generate 2D plots
                        plot_2d_path = output_path / f"pdp_2d_{feature_pair[0]}_vs_{feature_pair[1]}.png"
                        self.plot_partial_dependence_2d(feature_pair, save_path=str(plot_2d_path))
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze 2D PDP for {feature_pair}: {e}")
                        continue
                
                results['2d_partial_dependence'] = pd_results_2d
            
            # Interaction analysis
            interaction_analysis = self.analyze_feature_interactions(max_features=len(features))
            results['interaction_analysis'] = interaction_analysis
            
            # Save summary
            summary = {
                'n_features_analyzed': len(features),
                'features_analyzed': features,
                'include_ice': include_ice,
                'include_2d': include_2d,
                'high_interaction_features': interaction_analysis['high_interaction_features']
            }
            
            summary_path = output_path / "pdp_summary.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            results['summary'] = summary
            results['output_directory'] = str(output_path)
            
            logger.info(f"PDP/ICE analysis report generated in: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate PDP report: {e}")
            raise PDPError(f"Report generation failed: {e}")
        
        return results
    
    def compare_models(self, models: Dict[str, Any], feature: Union[str, int],
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[str] = None) -> None:
        """
        Compare partial dependence across different models.
        
        Args:
            models: Dictionary of model_name -> model pairs
            feature: Feature to compare
            figsize: Figure size
            save_path: Path to save plot
        """
        logger.info(f"Comparing partial dependence across {len(models)} models for feature: {feature}")
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, (model_name, model) in enumerate(models.items()):
            # Create temporary analyzer for this model
            temp_analyzer = PartialDependenceAnalyzer(
                model, self.X, self.feature_names, self.categorical_features
            )
            
            # Calculate PD
            pd_result = temp_analyzer.partial_dependence_1d(feature)
            
            # Plot
            plt.plot(pd_result['grid_values'], pd_result['partial_dependence'],
                    color=colors[i], linewidth=2, label=model_name)
        
        plt.xlabel(pd_result['feature_name'])
        plt.ylabel('Partial Dependence')
        plt.title(f"Model Comparison - Partial Dependence: {pd_result['feature_name']}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to: {save_path}")
        
        plt.show()