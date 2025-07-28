"""
Feature elimination module for backward elimination with comprehensive logging and visualization.

This module provides iterative backward feature elimination with detailed tracking,
Excel reporting, and visualization of feature importance over iterations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class EliminationStep:
    """Data class to store information about each elimination step."""
    iteration: int
    eliminated_feature: Optional[str]
    remaining_features: List[str]
    performance_metric: float
    feature_importances: Dict[str, float]
    execution_time: float
    model_params: Dict[str, Any]
    cross_val_scores: Optional[List[float]] = None
    std_dev: Optional[float] = None


class FeatureEliminator:
    """
    Iterative backward feature elimination with comprehensive tracking and reporting.
    
    This class performs backward elimination by iteratively removing the least important
    feature and tracking performance changes. Results are logged to Excel files and
    visualized through various plots.
    """
    
    def __init__(self, 
                 estimator: Any,
                 scoring: Union[str, Callable] = 'accuracy',
                 cv: int = 5,
                 min_features: int = 1,
                 tolerance: float = 0.001,
                 max_iterations: Optional[int] = None,
                 early_stopping_rounds: Optional[int] = None,
                 feature_importance_method: str = 'auto',
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize FeatureEliminator.
        
        Args:
            estimator: Scikit-learn compatible estimator
            scoring: Scoring metric ('accuracy', 'roc_auc', 'neg_mean_squared_error', etc.)
            cv: Number of cross-validation folds
            min_features: Minimum number of features to retain
            tolerance: Minimum improvement required to continue elimination
            max_iterations: Maximum number of elimination iterations
            early_stopping_rounds: Stop if no improvement for N rounds
            feature_importance_method: Method to get feature importance ('auto', 'permutation', 'coefficients')
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
        """
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.min_features = min_features
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_importance_method = feature_importance_method
        self.random_state = random_state
        self.verbose = verbose
        
        # Results storage
        self.elimination_steps_: List[EliminationStep] = []
        self.best_features_: Optional[List[str]] = None
        self.best_score_: Optional[float] = None
        self.best_iteration_: Optional[int] = None
        self.feature_rankings_: Optional[Dict[str, int]] = None
        
        # Internal state
        self._is_fitted = False
        self._feature_names = None
        self._task_type = None
        
        logger.info(f"Initialized FeatureEliminator with {type(estimator).__name__}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureEliminator':
        """
        Perform backward feature elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting backward feature elimination")
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Store feature names and determine task type
        self._feature_names = list(X.columns)
        self._task_type = self._determine_task_type(y)
        
        # Initialize tracking
        current_features = self._feature_names.copy()
        best_score = float('-inf') if self._is_score_higher_better() else float('inf')
        no_improvement_count = 0
        
        if self.verbose:
            logger.info(f"Starting elimination with {len(current_features)} features")
            logger.info(f"Task type: {self._task_type}")
            logger.info(f"Scoring metric: {self.scoring}")
        
        # Evaluate initial feature set
        initial_step = self._evaluate_feature_set(X, y, current_features, 0, None)
        self.elimination_steps_.append(initial_step)
        best_score = initial_step.performance_metric
        
        iteration = 1
        
        # Main elimination loop
        while (len(current_features) > self.min_features and
               (self.max_iterations is None or iteration <= self.max_iterations)):
            
            if self.verbose:
                logger.info(f"Iteration {iteration}: {len(current_features)} features remaining")
            
            # Find least important feature to eliminate
            feature_to_eliminate = self._find_least_important_feature(
                X[current_features], y, current_features
            )
            
            if feature_to_eliminate is None:
                logger.warning("Could not determine feature to eliminate. Stopping.")
                break
            
            # Create new feature set without the eliminated feature
            new_features = [f for f in current_features if f != feature_to_eliminate]
            
            # Evaluate new feature set
            step = self._evaluate_feature_set(
                X, y, new_features, iteration, feature_to_eliminate
            )
            self.elimination_steps_.append(step)
            
            # Check for improvement
            current_score = step.performance_metric
            score_improved = self._check_improvement(current_score, best_score)
            
            if score_improved:
                best_score = current_score
                self.best_features_ = new_features.copy()
                self.best_score_ = best_score
                self.best_iteration_ = iteration
                no_improvement_count = 0
                
                if self.verbose:
                    logger.info(f"Improvement found: {current_score:.6f} "
                              f"(eliminated {feature_to_eliminate})")
            else:
                no_improvement_count += 1
                if self.verbose:
                    logger.info(f"No improvement: {current_score:.6f} "
                              f"(eliminated {feature_to_eliminate})")
            
            # Early stopping check
            if (self.early_stopping_rounds is not None and
                no_improvement_count >= self.early_stopping_rounds):
                logger.info(f"Early stopping triggered after {iteration} iterations")
                break
            
            current_features = new_features
            iteration += 1
        
        # Finalize results
        self._finalize_results()
        
        total_time = time.time() - start_time
        logger.info(f"Feature elimination completed in {total_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score_:.6f} with {len(self.best_features_)} features")
        
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to include only selected features.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before transform")
        
        if self.best_features_ is None:
            logger.warning("No best features found, returning original data")
            return X
        
        return X[self.best_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the eliminator and transform the data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_rankings(self) -> Dict[str, int]:
        """
        Get feature rankings based on elimination order.
        
        Returns:
            Dictionary mapping features to their ranking (1 = eliminated first)
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before getting rankings")
        
        return self.feature_rankings_.copy()
    
    def get_elimination_summary(self) -> pd.DataFrame:
        """
        Get summary of elimination process.
        
        Returns:
            DataFrame with elimination details
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before getting summary")
        
        summary_data = []
        for step in self.elimination_steps_:
            summary_data.append({
                'iteration': step.iteration,
                'eliminated_feature': step.eliminated_feature,
                'num_features': len(step.remaining_features),
                'performance_metric': step.performance_metric,
                'cv_std': step.std_dev,
                'execution_time': step.execution_time
            })
        
        return pd.DataFrame(summary_data)
    
    def export_to_excel(self, 
                       output_path: Union[str, Path],
                       include_feature_importance: bool = True,
                       include_cv_details: bool = True) -> None:
        """
        Export elimination results to Excel file with multiple sheets.
        
        Args:
            output_path: Path to save Excel file
            include_feature_importance: Include feature importance details
            include_cv_details: Include cross-validation details
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before export")
        
        logger.info(f"Exporting results to {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Summary sheet
            summary_df = self.get_elimination_summary()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Best features sheet
            best_features_df = pd.DataFrame({
                'feature': self.best_features_,
                'ranking': [self.feature_rankings_[f] for f in self.best_features_]
            }).sort_values('ranking', ascending=False)
            best_features_df.to_excel(writer, sheet_name='Best_Features', index=False)
            
            # Feature rankings sheet
            rankings_df = pd.DataFrame([
                {'feature': feature, 'ranking': ranking, 'elimination_order': len(self._feature_names) - ranking + 1}
                for feature, ranking in self.feature_rankings_.items()
            ]).sort_values('ranking', ascending=False)
            rankings_df.to_excel(writer, sheet_name='Feature_Rankings', index=False)
            
            # Detailed results for each iteration
            if include_feature_importance:
                for i, step in enumerate(self.elimination_steps_):
                    if step.feature_importances:
                        importance_df = pd.DataFrame([
                            {'feature': feature, 'importance': importance}
                            for feature, importance in step.feature_importances.items()
                        ]).sort_values('importance', ascending=False)
                        
                        sheet_name = f'Iteration_{step.iteration}_Importance'
                        importance_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Cross-validation details
            if include_cv_details:
                cv_data = []
                for step in self.elimination_steps_:
                    if step.cross_val_scores:
                        for fold, score in enumerate(step.cross_val_scores):
                            cv_data.append({
                                'iteration': step.iteration,
                                'fold': fold + 1,
                                'score': score,
                                'num_features': len(step.remaining_features),
                                'eliminated_feature': step.eliminated_feature
                            })
                
                if cv_data:
                    cv_df = pd.DataFrame(cv_data)
                    cv_df.to_excel(writer, sheet_name='CV_Details', index=False)
            
            # Configuration sheet
            config_df = pd.DataFrame([
                {'parameter': 'estimator', 'value': str(type(self.estimator).__name__)},
                {'parameter': 'scoring', 'value': str(self.scoring)},
                {'parameter': 'cv_folds', 'value': self.cv},
                {'parameter': 'min_features', 'value': self.min_features},
                {'parameter': 'tolerance', 'value': self.tolerance},
                {'parameter': 'max_iterations', 'value': self.max_iterations},
                {'parameter': 'early_stopping_rounds', 'value': self.early_stopping_rounds},
                {'parameter': 'feature_importance_method', 'value': self.feature_importance_method},
                {'parameter': 'task_type', 'value': self._task_type},
                {'parameter': 'best_score', 'value': self.best_score_},
                {'parameter': 'best_iteration', 'value': self.best_iteration_},
                {'parameter': 'total_iterations', 'value': len(self.elimination_steps_) - 1}
            ])
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        logger.info(f"Results exported to {output_path}")
    
    def plot_elimination_curve(self, 
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[Union[str, Path]] = None,
                              show_std: bool = True) -> plt.Figure:
        """
        Plot performance metric vs number of features.
        
        Args:
            figsize: Figure size
            save_path: Path to save plot
            show_std: Show standard deviation bands
            
        Returns:
            Matplotlib figure
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before plotting")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data for plotting
        iterations = [step.iteration for step in self.elimination_steps_]
        num_features = [len(step.remaining_features) for step in self.elimination_steps_]
        scores = [step.performance_metric for step in self.elimination_steps_]
        std_devs = [step.std_dev for step in self.elimination_steps_ if step.std_dev is not None]
        
        # Main curve
        ax.plot(num_features, scores, 'b-', linewidth=2, marker='o', markersize=4, label='CV Score')
        
        # Standard deviation bands
        if show_std and len(std_devs) == len(scores):
            ax.fill_between(num_features, 
                           [s - std for s, std in zip(scores, std_devs)],
                           [s + std for s, std in zip(scores, std_devs)],
                           alpha=0.3, color='blue', label='± 1 std dev')
        
        # Mark best point
        if self.best_iteration_ is not None:
            best_step = self.elimination_steps_[self.best_iteration_]
            ax.plot(len(best_step.remaining_features), best_step.performance_metric,
                   'ro', markersize=8, label=f'Best ({len(best_step.remaining_features)} features)')
        
        ax.set_xlabel('Number of Features')
        ax.set_ylabel(f'Performance ({self.scoring})')
        ax.set_title('Feature Elimination Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Invert x-axis to show elimination progress
        ax.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Elimination curve saved to {save_path}")
        
        return fig
    
    def plot_feature_importance_evolution(self,
                                        top_n: int = 15,
                                        figsize: Tuple[int, int] = (14, 10),
                                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot evolution of feature importance over iterations.
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before plotting")
        
        # Collect importance data
        importance_data = {}
        iterations = []
        
        for step in self.elimination_steps_:
            if step.feature_importances:
                iterations.append(step.iteration)
                for feature, importance in step.feature_importances.items():
                    if feature not in importance_data:
                        importance_data[feature] = []
                    importance_data[feature].append(importance)
        
        if not importance_data:
            logger.warning("No feature importance data available for plotting")
            return None
        
        # Select top N features based on final iteration importance
        final_step = self.elimination_steps_[-1]
        if final_step.feature_importances:
            top_features = sorted(final_step.feature_importances.items(),
                                key=lambda x: x[1], reverse=True)[:top_n]
            top_feature_names = [f[0] for f in top_features]
        else:
            # Fallback to features with most data points
            top_feature_names = sorted(importance_data.keys(),
                                     key=lambda x: len(importance_data[x]),
                                     reverse=True)[:top_n]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_feature_names)))
        
        for i, feature in enumerate(top_feature_names):
            if feature in importance_data:
                # Pad with NaN for missing iterations
                feature_importance = []
                for iteration in iterations:
                    step = self.elimination_steps_[iteration]
                    if step.feature_importances and feature in step.feature_importances:
                        feature_importance.append(step.feature_importances[feature])
                    else:
                        feature_importance.append(np.nan)
                
                ax.plot(iterations, feature_importance, 
                       color=colors[i], marker='o', linewidth=1.5,
                       label=feature, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Feature Importance Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance evolution plot saved to {save_path}")
        
        return fig
    
    def plot_elimination_heatmap(self,
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot heatmap showing when each feature was eliminated.
        
        Args:
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before plotting")
        
        # Create elimination matrix
        elimination_matrix = []
        feature_names = []
        
        for step in self.elimination_steps_:
            if step.iteration > 0:  # Skip initial step
                row = []
                for feature in self._feature_names:
                    if feature in step.remaining_features:
                        row.append(1)  # Feature present
                    elif feature == step.eliminated_feature:
                        row.append(0.5)  # Feature being eliminated
                    else:
                        row.append(0)  # Feature already eliminated
                elimination_matrix.append(row)
                feature_names.append(f"Iter {step.iteration}")
        
        if not elimination_matrix:
            logger.warning("No elimination data available for heatmap")
            return None
        
        # Create DataFrame
        elimination_df = pd.DataFrame(elimination_matrix, 
                                    index=feature_names,
                                    columns=self._feature_names)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(elimination_df, 
                   cmap=['red', 'orange', 'green'], 
                   cbar_kws={'label': 'Feature Status'},
                   ax=ax)
        
        ax.set_title('Feature Elimination Timeline')
        ax.set_xlabel('Features')
        ax.set_ylabel('Elimination Iterations')
        
        # Rotate feature names for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Elimination heatmap saved to {save_path}")
        
        return fig
    
    def generate_comprehensive_report(self,
                                    output_dir: Union[str, Path],
                                    report_name: Optional[str] = None) -> Dict[str, str]:
        """
        Generate comprehensive report with Excel export and all visualizations.
        
        Args:
            output_dir: Directory to save report files
            report_name: Base name for report files
            
        Returns:
            Dictionary mapping report type to file path
        """
        if not self._is_fitted:
            raise ValueError("FeatureEliminator must be fitted before generating report")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"feature_elimination_report_{timestamp}"
        
        report_files = {}
        
        # Excel export
        excel_path = output_dir / f"{report_name}.xlsx"
        self.export_to_excel(excel_path)
        report_files['excel'] = str(excel_path)
        
        # Elimination curve
        curve_path = output_dir / f"{report_name}_elimination_curve.png"
        self.plot_elimination_curve(save_path=curve_path)
        plt.close()
        report_files['elimination_curve'] = str(curve_path)
        
        # Feature importance evolution
        importance_path = output_dir / f"{report_name}_importance_evolution.png"
        self.plot_feature_importance_evolution(save_path=importance_path)
        plt.close()
        report_files['importance_evolution'] = str(importance_path)
        
        # Elimination heatmap
        heatmap_path = output_dir / f"{report_name}_elimination_heatmap.png"
        self.plot_elimination_heatmap(save_path=heatmap_path)
        plt.close()
        report_files['elimination_heatmap'] = str(heatmap_path)
        
        # Summary JSON
        summary_path = output_dir / f"{report_name}_summary.json"
        summary_data = {
            'configuration': {
                'estimator': str(type(self.estimator).__name__),
                'scoring': self.scoring,
                'cv_folds': self.cv,
                'min_features': self.min_features,
                'tolerance': self.tolerance,
                'max_iterations': self.max_iterations,
                'early_stopping_rounds': self.early_stopping_rounds,
                'feature_importance_method': self.feature_importance_method,
                'task_type': self._task_type
            },
            'results': {
                'initial_features': len(self._feature_names),
                'final_features': len(self.best_features_) if self.best_features_ else 0,
                'best_score': self.best_score_,
                'best_iteration': self.best_iteration_,
                'total_iterations': len(self.elimination_steps_) - 1,
                'best_features': self.best_features_
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        report_files['summary'] = str(summary_path)
        
        logger.info(f"Comprehensive report generated in {output_dir}")
        return report_files
    
    # Private methods
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if X.empty or y.empty:
            raise ValueError("X and y cannot be empty")
        
        if len(X.columns) < self.min_features:
            raise ValueError(f"Number of features ({len(X.columns)}) must be >= min_features ({self.min_features})")
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if task is classification or regression."""
        if y.dtype in ['object', 'category', 'bool']:
            return 'classification'
        
        unique_values = y.nunique()
        total_values = len(y)
        
        # If less than 20 unique values or less than 5% of total, likely classification
        if unique_values < 20 or unique_values / total_values < 0.05:
            return 'classification'
        else:
            return 'regression'
    
    def _is_score_higher_better(self) -> bool:
        """Check if higher score is better for the given metric."""
        higher_better_metrics = [
            'accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'r2'
        ]
        lower_better_metrics = [
            'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_log_loss'
        ]
        
        scoring_str = str(self.scoring).lower()
        
        if any(metric in scoring_str for metric in higher_better_metrics):
            return True
        elif any(metric in scoring_str for metric in lower_better_metrics):
            return False
        else:
            # Default assumption
            return not scoring_str.startswith('neg_')
    
    def _check_improvement(self, current_score: float, best_score: float) -> bool:
        """Check if current score is an improvement over best score."""
        if self._is_score_higher_better():
            return current_score > best_score + self.tolerance
        else:
            return current_score < best_score - self.tolerance
    
    def _evaluate_feature_set(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             features: List[str],
                             iteration: int,
                             eliminated_feature: Optional[str]) -> EliminationStep:
        """Evaluate a specific feature set."""
        start_time = time.time()
        
        X_subset = X[features]
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.estimator, X_subset, y, 
            cv=self.cv, scoring=self.scoring
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        # Get feature importance
        feature_importances = self._get_feature_importance(X_subset, y)
        
        # Get model parameters
        model_params = self.estimator.get_params() if hasattr(self.estimator, 'get_params') else {}
        
        execution_time = time.time() - start_time
        
        return EliminationStep(
            iteration=iteration,
            eliminated_feature=eliminated_feature,
            remaining_features=features.copy(),
            performance_metric=mean_score,
            feature_importances=feature_importances,
            execution_time=execution_time,
            model_params=model_params,
            cross_val_scores=cv_scores.tolist(),
            std_dev=std_score
        )
    
    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance using specified method."""
        try:
            if self.feature_importance_method == 'auto':
                # Try built-in importance first
                if hasattr(self.estimator, 'feature_importances_'):
                    self.estimator.fit(X, y)
                    importance_values = self.estimator.feature_importances_
                elif hasattr(self.estimator, 'coef_'):
                    self.estimator.fit(X, y)
                    importance_values = np.abs(self.estimator.coef_).flatten()
                else:
                    return self._get_permutation_importance(X, y)
            
            elif self.feature_importance_method == 'permutation':
                return self._get_permutation_importance(X, y)
            
            elif self.feature_importance_method == 'coefficients':
                if not hasattr(self.estimator, 'coef_'):
                    logger.warning("Estimator does not have coefficients, using permutation importance")
                    return self._get_permutation_importance(X, y)
                
                self.estimator.fit(X, y)
                importance_values = np.abs(self.estimator.coef_).flatten()
            
            else:
                logger.warning(f"Unknown importance method {self.feature_importance_method}, using permutation")
                return self._get_permutation_importance(X, y)
            
            # Convert to dictionary
            return dict(zip(X.columns, importance_values))
            
        except Exception as e:
            logger.warning(f"Error getting feature importance: {e}. Using permutation importance.")
            return self._get_permutation_importance(X, y)
    
    def _get_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance using permutation method."""
        from sklearn.inspection import permutation_importance
        
        # Fit model
        self.estimator.fit(X, y)
        
        # Calculate permutation importance
        result = permutation_importance(
            self.estimator, X, y, 
            scoring=self.scoring,
            n_repeats=5,
            random_state=self.random_state
        )
        
        return dict(zip(X.columns, result.importances_mean))
    
    def _find_least_important_feature(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    features: List[str]) -> Optional[str]:
        """Find the least important feature to eliminate."""
        feature_importances = self._get_feature_importance(X, y)
        
        if not feature_importances:
            return None
        
        # Find feature with minimum importance
        least_important = min(feature_importances.items(), key=lambda x: x[1])
        return least_important[0]
    
    def _finalize_results(self) -> None:
        """Finalize results and create feature rankings."""
        if not self.elimination_steps_:
            return
        
        # If no best features found, use the last step
        if self.best_features_ is None:
            last_step = self.elimination_steps_[-1]
            self.best_features_ = last_step.remaining_features
            self.best_score_ = last_step.performance_metric
            self.best_iteration_ = last_step.iteration
        
        # Create feature rankings
        rankings = {}
        remaining_features = set(self._feature_names)
        
        # Rank eliminated features in reverse order
        for step in reversed(self.elimination_steps_[1:]):  # Skip initial step
            if step.eliminated_feature:
                rankings[step.eliminated_feature] = len(remaining_features)
                remaining_features.remove(step.eliminated_feature)
        
        # Rank remaining features based on final importance
        if self.elimination_steps_:
            final_step = self.elimination_steps_[-1]
            if final_step.feature_importances:
                sorted_remaining = sorted(
                    [(f, imp) for f, imp in final_step.feature_importances.items() 
                     if f in remaining_features],
                    key=lambda x: x[1], reverse=True
                )
                
                current_rank = len(remaining_features)
                for feature, _ in sorted_remaining:
                    rankings[feature] = current_rank
                    current_rank -= 1
        
        self.feature_rankings_ = rankings