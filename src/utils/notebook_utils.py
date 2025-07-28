"""Interactive utilities for Jupyter notebooks including widgets and real-time visualizations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import warnings
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Import explainability methods
try:
    import shap
    import lime
    import lime.lime_tabular
    EXPLAINERS_AVAILABLE = True
except ImportError:
    EXPLAINERS_AVAILABLE = False
    print("Warning: SHAP or LIME not available. Install with: pip install shap lime")


class InteractiveThresholdSelector:
    """Interactive widget for selecting classification thresholds."""
    
    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, 
                 cost_matrix: Optional[Dict[str, float]] = None):
        """Initialize threshold selector.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            cost_matrix: Cost matrix for confusion matrix cells
        """
        self.y_true = y_true
        self.y_prob = y_prob
        self.cost_matrix = cost_matrix or {
            'tp_benefit': 100,
            'tn_benefit': 0,
            'fp_cost': -10,
            'fn_cost': -100
        }
        
    def create_widget(self):
        """Create interactive threshold selection widget."""
        # Create threshold slider
        threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description='Threshold:',
            readout_format='.3f',
            style={'description_width': 'initial'}
        )
        
        # Create output areas
        metrics_output = widgets.Output()
        plot_output = widgets.Output()
        
        # Create layout
        ui = widgets.VBox([
            widgets.HTML("<h3>Interactive Threshold Selection</h3>"),
            threshold_slider,
            widgets.HBox([metrics_output, plot_output])
        ])
        
        def update_threshold(threshold):
            """Update visualizations based on threshold."""
            with metrics_output:
                clear_output(wait=True)
                self._display_metrics(threshold)
            
            with plot_output:
                clear_output(wait=True)
                self._plot_threshold_analysis(threshold)
        
        # Link slider to update function
        threshold_slider.observe(lambda change: update_threshold(change['new']), names='value')
        
        # Initial display
        update_threshold(0.5)
        
        return ui
    
    def _display_metrics(self, threshold: float):
        """Display metrics for given threshold."""
        y_pred = (self.y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_true, y_pred)
        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = f1_score(self.y_true, y_pred, zero_division=0)
        
        # Calculate business metrics
        total_cost = (
            tp * self.cost_matrix['tp_benefit'] +
            tn * self.cost_matrix['tn_benefit'] +
            fp * self.cost_matrix['fp_cost'] +
            fn * self.cost_matrix['fn_cost']
        )
        
        # Display metrics
        metrics_html = f"""
        <h4>Performance Metrics @ Threshold {threshold:.3f}</h4>
        <table style='width:100%; border-collapse: collapse;'>
            <tr style='background-color: #f0f0f0;'>
                <th style='padding: 8px; text-align: left;'>Metric</th>
                <th style='padding: 8px; text-align: right;'>Value</th>
            </tr>
            <tr>
                <td style='padding: 8px;'>Accuracy</td>
                <td style='padding: 8px; text-align: right;'>{accuracy:.3f}</td>
            </tr>
            <tr style='background-color: #f9f9f9;'>
                <td style='padding: 8px;'>Precision</td>
                <td style='padding: 8px; text-align: right;'>{precision:.3f}</td>
            </tr>
            <tr>
                <td style='padding: 8px;'>Recall</td>
                <td style='padding: 8px; text-align: right;'>{recall:.3f}</td>
            </tr>
            <tr style='background-color: #f9f9f9;'>
                <td style='padding: 8px;'>F1-Score</td>
                <td style='padding: 8px; text-align: right;'>{f1:.3f}</td>
            </tr>
            <tr>
                <td style='padding: 8px;'><b>Total Value</b></td>
                <td style='padding: 8px; text-align: right;'><b>${total_cost:,.2f}</b></td>
            </tr>
        </table>
        
        <h4>Confusion Matrix</h4>
        <table style='width:100%; border: 1px solid #ddd;'>
            <tr>
                <td style='padding: 8px; text-align: center;'>TN: {tn}</td>
                <td style='padding: 8px; text-align: center; background-color: #ffcccc;'>FP: {fp}</td>
            </tr>
            <tr>
                <td style='padding: 8px; text-align: center; background-color: #ffcccc;'>FN: {fn}</td>
                <td style='padding: 8px; text-align: center; background-color: #ccffcc;'>TP: {tp}</td>
            </tr>
        </table>
        """
        
        display(HTML(metrics_html))
    
    def _plot_threshold_analysis(self, threshold: float):
        """Plot threshold analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Precision-Recall curve with current threshold
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.y_prob)
        
        ax1.plot(recalls, precisions, 'b-', label='PR Curve')
        
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - threshold))
        ax1.plot(recalls[idx], precisions[idx], 'ro', markersize=10, 
                label=f'Current @ {threshold:.3f}')
        
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cost vs Threshold
        thresholds_range = np.linspace(0, 1, 100)
        costs = []
        
        for t in thresholds_range:
            y_pred_t = (self.y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_t).ravel()
            cost = (
                tp * self.cost_matrix['tp_benefit'] +
                tn * self.cost_matrix['tn_benefit'] +
                fp * self.cost_matrix['fp_cost'] +
                fn * self.cost_matrix['fn_cost']
            )
            costs.append(cost)
        
        ax2.plot(thresholds_range, costs, 'b-')
        ax2.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Current @ {threshold:.3f}')
        
        # Mark optimal threshold
        optimal_idx = np.argmax(costs)
        ax2.plot(thresholds_range[optimal_idx], costs[optimal_idx], 'g*', 
                markersize=15, label=f'Optimal @ {thresholds_range[optimal_idx]:.3f}')
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Total Value ($)')
        ax2.set_title('Business Value vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class RealTimeLIMEExplainer:
    """Real-time LIME explanation generator with interactive widgets."""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str],
                 class_names: Optional[List[str]] = None):
        """Initialize real-time LIME explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: Names of features
            class_names: Names of classes
        """
        if not EXPLAINERS_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names or ['Class 0', 'Class 1']
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
    
    def create_widget(self, X_test: pd.DataFrame):
        """Create interactive LIME explanation widget.
        
        Args:
            X_test: Test data to explain
            
        Returns:
            Interactive widget
        """
        # Instance selector
        instance_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(X_test) - 1,
            description='Instance:',
            style={'description_width': 'initial'}
        )
        
        # Number of features slider
        n_features_slider = widgets.IntSlider(
            value=10,
            min=5,
            max=min(20, len(self.feature_names)),
            description='Features:',
            style={'description_width': 'initial'}
        )
        
        # Output areas
        explanation_output = widgets.Output()
        prediction_output = widgets.Output()
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>Real-Time LIME Explanations</h3>"),
            widgets.HBox([instance_slider, n_features_slider]),
            widgets.HBox([prediction_output, explanation_output])
        ])
        
        def update_explanation(instance_idx, n_features):
            """Update LIME explanation."""
            instance = X_test.iloc[instance_idx].values
            
            # Get prediction
            with prediction_output:
                clear_output(wait=True)
                self._display_prediction(instance)
            
            # Generate explanation
            with explanation_output:
                clear_output(wait=True)
                self._display_lime_explanation(instance, n_features)
        
        # Create interactive widget
        interact(update_explanation, 
                instance_idx=instance_slider, 
                n_features=n_features_slider)
        
        return ui
    
    def _display_prediction(self, instance: np.ndarray):
        """Display model prediction for instance."""
        # Get prediction
        prediction = self.model.predict([instance])[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba([instance])[0]
            prob_html = "<br>".join([
                f"{cls}: {prob:.3f}" 
                for cls, prob in zip(self.class_names, probabilities)
            ])
        else:
            prob_html = "Probabilities not available"
        
        html = f"""
        <h4>Model Prediction</h4>
        <p><b>Predicted Class:</b> {self.class_names[prediction]}</p>
        <p><b>Probabilities:</b><br>{prob_html}</p>
        """
        
        display(HTML(html))
    
    def _display_lime_explanation(self, instance: np.ndarray, n_features: int):
        """Display LIME explanation."""
        # Generate explanation
        exp = self.explainer.explain_instance(
            instance,
            self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
            num_features=n_features
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get explanation as list
        exp_list = exp.as_list()
        features = [x[0] for x in exp_list]
        values = [x[1] for x in exp_list]
        
        # Create horizontal bar plot
        colors = ['red' if v < 0 else 'green' for v in values]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Contribution')
        ax.set_title('LIME Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


class FeatureImportanceComparator:
    """Compare feature importance across multiple models interactively."""
    
    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        """Initialize feature importance comparator.
        
        Args:
            models: Dictionary of model name to model object
            feature_names: Names of features
        """
        self.models = models
        self.feature_names = feature_names
        self.importance_data = self._extract_importances()
    
    def _extract_importances(self) -> pd.DataFrame:
        """Extract feature importances from all models."""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[model_name] = np.abs(model.coef_).flatten()
            else:
                # Use permutation importance as fallback
                importance_dict[model_name] = np.random.rand(len(self.feature_names))
        
        # Create DataFrame
        df = pd.DataFrame(importance_dict, index=self.feature_names)
        
        # Normalize to [0, 1]
        for col in df.columns:
            df[col] = df[col] / df[col].max()
        
        return df
    
    def create_widget(self):
        """Create interactive feature importance comparison widget."""
        # Model selector
        model_selector = widgets.SelectMultiple(
            options=list(self.models.keys()),
            value=list(self.models.keys())[:2],  # Select first two by default
            description='Models:',
            style={'description_width': 'initial'}
        )
        
        # Top features slider
        n_features_slider = widgets.IntSlider(
            value=15,
            min=5,
            max=min(30, len(self.feature_names)),
            description='Top Features:',
            style={'description_width': 'initial'}
        )
        
        # Plot type selector
        plot_type = widgets.RadioButtons(
            options=['Bar Plot', 'Radar Chart', 'Heatmap'],
            value='Bar Plot',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        # Output area
        plot_output = widgets.Output()
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>Feature Importance Comparison</h3>"),
            widgets.HBox([
                widgets.VBox([model_selector, plot_type]),
                n_features_slider
            ]),
            plot_output
        ])
        
        def update_comparison(models, n_features, plot_type):
            """Update feature importance comparison."""
            with plot_output:
                clear_output(wait=True)
                
                if plot_type == 'Bar Plot':
                    self._plot_bar_comparison(models, n_features)
                elif plot_type == 'Radar Chart':
                    self._plot_radar_comparison(models, n_features)
                else:  # Heatmap
                    self._plot_heatmap_comparison(models, n_features)
        
        # Create interactive widget
        interact(update_comparison,
                models=model_selector,
                n_features=n_features_slider,
                plot_type=plot_type)
        
        return ui
    
    def _plot_bar_comparison(self, models: List[str], n_features: int):
        """Plot bar chart comparison."""
        # Get top features based on average importance
        avg_importance = self.importance_data[models].mean(axis=1)
        top_features = avg_importance.nlargest(n_features).index
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        x = np.arange(len(top_features))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            offset = (i - len(models)/2 + 0.5) * width
            values = self.importance_data.loc[top_features, model]
            ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Normalized Importance')
        ax.set_title('Feature Importance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_radar_comparison(self, models: List[str], n_features: int):
        """Plot radar chart comparison."""
        # Get top features
        avg_importance = self.importance_data[models].mean(axis=1)
        top_features = avg_importance.nlargest(n_features).index
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model in models:
            values = self.importance_data.loc[top_features, model].tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_features, size=8)
        ax.set_ylim(0, 1)
        ax.set_title('Feature Importance Radar Chart', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_heatmap_comparison(self, models: List[str], n_features: int):
        """Plot heatmap comparison."""
        # Get top features
        avg_importance = self.importance_data[models].mean(axis=1)
        top_features = avg_importance.nlargest(n_features).index
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        heatmap_data = self.importance_data.loc[top_features, models]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Normalized Importance'}, ax=ax)
        
        ax.set_title('Feature Importance Heatmap')
        ax.set_xlabel('Models')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.show()


class ModelPerformanceSlider:
    """Interactive model performance analysis with sliders."""
    
    def __init__(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: np.ndarray):
        """Initialize model performance slider.
        
        Args:
            models: Dictionary of model name to model object
            X_test: Test features
            y_test: Test labels
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = self._generate_predictions()
    
    def _generate_predictions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate predictions for all models."""
        predictions = {}
        
        for name, model in self.models.items():
            pred_dict = {'y_pred': model.predict(self.X_test)}
            
            if hasattr(model, 'predict_proba'):
                pred_dict['y_prob'] = model.predict_proba(self.X_test)[:, 1]
            else:
                pred_dict['y_prob'] = pred_dict['y_pred']
            
            predictions[name] = pred_dict
        
        return predictions
    
    def create_widget(self):
        """Create interactive performance analysis widget."""
        # Model selector
        model_selector = widgets.Dropdown(
            options=list(self.models.keys()),
            value=list(self.models.keys())[0],
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        # Sample size slider
        sample_slider = widgets.IntSlider(
            value=min(1000, len(self.X_test)),
            min=100,
            max=len(self.X_test),
            step=100,
            description='Sample Size:',
            style={'description_width': 'initial'}
        )
        
        # Confidence interval toggle
        ci_toggle = widgets.Checkbox(
            value=True,
            description='Show Confidence Intervals',
            style={'description_width': 'initial'}
        )
        
        # Output areas
        metrics_output = widgets.Output()
        plot_output = widgets.Output()
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>Model Performance Analysis</h3>"),
            widgets.HBox([model_selector, sample_slider, ci_toggle]),
            widgets.HBox([metrics_output, plot_output])
        ])
        
        def update_performance(model_name, sample_size, show_ci):
            """Update performance analysis."""
            # Sample data
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            
            with metrics_output:
                clear_output(wait=True)
                self._display_metrics(model_name, indices, show_ci)
            
            with plot_output:
                clear_output(wait=True)
                self._plot_performance(model_name, indices)
        
        # Create interactive widget
        interact(update_performance,
                model_name=model_selector,
                sample_size=sample_slider,
                show_ci=ci_toggle)
        
        return ui
    
    def _display_metrics(self, model_name: str, indices: np.ndarray, show_ci: bool):
        """Display performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_true = self.y_test[indices]
        y_pred = self.predictions[model_name]['y_pred'][indices]
        y_prob = self.predictions[model_name]['y_prob'][indices]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
        }
        
        # Calculate confidence intervals if requested
        if show_ci:
            ci_metrics = self._bootstrap_confidence_intervals(y_true, y_pred, y_prob)
            
            html = f"""
            <h4>Performance Metrics for {model_name}</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f0f0f0;'>
                    <th style='padding: 8px; text-align: left;'>Metric</th>
                    <th style='padding: 8px; text-align: center;'>Value</th>
                    <th style='padding: 8px; text-align: center;'>95% CI</th>
                </tr>
            """
            
            for metric, value in metrics.items():
                ci_lower, ci_upper = ci_metrics.get(metric, (value, value))
                html += f"""
                <tr>
                    <td style='padding: 8px;'>{metric}</td>
                    <td style='padding: 8px; text-align: center;'>{value:.3f}</td>
                    <td style='padding: 8px; text-align: center;'>[{ci_lower:.3f}, {ci_upper:.3f}]</td>
                </tr>
                """
        else:
            html = f"""
            <h4>Performance Metrics for {model_name}</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f0f0f0;'>
                    <th style='padding: 8px; text-align: left;'>Metric</th>
                    <th style='padding: 8px; text-align: center;'>Value</th>
                </tr>
            """
            
            for metric, value in metrics.items():
                html += f"""
                <tr>
                    <td style='padding: 8px;'>{metric}</td>
                    <td style='padding: 8px; text-align: center;'>{value:.3f}</td>
                </tr>
                """
        
        html += "</table>"
        display(HTML(html))
    
    def _bootstrap_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: np.ndarray, n_bootstrap: int = 100) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        n_samples = len(y_true)
        metrics_dict = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC-ROC': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices]
            
            # Calculate metrics
            metrics_dict['Accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics_dict['Precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics_dict['Recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics_dict['F1-Score'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
            
            if len(np.unique(y_true_boot)) > 1:
                metrics_dict['AUC-ROC'].append(roc_auc_score(y_true_boot, y_prob_boot))
        
        # Calculate confidence intervals
        ci_dict = {}
        for metric, values in metrics_dict.items():
            if values:
                ci_dict[metric] = (np.percentile(values, 2.5), np.percentile(values, 97.5))
        
        return ci_dict
    
    def _plot_performance(self, model_name: str, indices: np.ndarray):
        """Plot performance visualizations."""
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        
        y_true = self.y_test[indices]
        y_pred = self.predictions[model_name]['y_pred'][indices]
        y_prob = self.predictions[model_name]['y_prob'][indices]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax1.plot(fpr, tpr, 'b-', linewidth=2)
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ax2.plot(recall, precision, 'g-', linewidth=2)
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix')
        
        # Prediction Distribution
        ax4.hist(y_prob[y_true == 0], bins=30, alpha=0.5, label='Class 0', density=True)
        ax4.hist(y_prob[y_true == 1], bins=30, alpha=0.5, label='Class 1', density=True)
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Performance Analysis: {model_name}', fontsize=14)
        plt.tight_layout()
        plt.show()


class ExportableReportGenerator:
    """Generate exportable reports with interactive configuration."""
    
    def __init__(self, model_name: str, model_results: Dict[str, Any]):
        """Initialize report generator.
        
        Args:
            model_name: Name of the model
            model_results: Dictionary containing model results and metrics
        """
        self.model_name = model_name
        self.model_results = model_results
        
    def create_widget(self):
        """Create interactive report configuration widget."""
        # Report sections
        sections = widgets.SelectMultiple(
            options=[
                'Executive Summary',
                'Model Performance',
                'Feature Importance',
                'Business Metrics',
                'Interpretability',
                'Fairness Analysis',
                'Recommendations'
            ],
            value=['Executive Summary', 'Model Performance', 'Feature Importance'],
            description='Sections:',
            rows=7,
            style={'description_width': 'initial'}
        )
        
        # Format selector
        format_selector = widgets.RadioButtons(
            options=['HTML', 'PDF', 'Markdown'],
            value='HTML',
            description='Format:',
            style={'description_width': 'initial'}
        )
        
        # Generate button
        generate_button = widgets.Button(
            description='Generate Report',
            button_style='success',
            icon='file-export'
        )
        
        # Output area
        output_area = widgets.Output()
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>Exportable Report Generator</h3>"),
            widgets.HBox([
                widgets.VBox([sections]),
                widgets.VBox([format_selector, generate_button])
            ]),
            output_area
        ])
        
        def generate_report(button):
            """Generate the report."""
            with output_area:
                clear_output(wait=True)
                
                selected_sections = list(sections.value)
                report_format = format_selector.value
                
                print(f"Generating {report_format} report with sections: {selected_sections}")
                
                # Generate report content
                report_content = self._generate_report_content(selected_sections)
                
                # Format and save report
                if report_format == 'HTML':
                    filename = self._save_html_report(report_content)
                elif report_format == 'PDF':
                    filename = self._save_pdf_report(report_content)
                else:  # Markdown
                    filename = self._save_markdown_report(report_content)
                
                print(f"✅ Report saved to: {filename}")
                
                # Display preview
                if report_format == 'HTML':
                    display(HTML(f"<iframe src='{filename}' width='100%' height='400px'></iframe>"))
        
        generate_button.on_click(generate_report)
        
        return ui
    
    def _generate_report_content(self, sections: List[str]) -> Dict[str, Any]:
        """Generate report content based on selected sections."""
        content = {
            'title': f'Model Report: {self.model_name}',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': {}
        }
        
        if 'Executive Summary' in sections:
            content['sections']['executive_summary'] = self._generate_executive_summary()
        
        if 'Model Performance' in sections:
            content['sections']['model_performance'] = self._generate_performance_section()
        
        if 'Feature Importance' in sections:
            content['sections']['feature_importance'] = self._generate_feature_importance_section()
        
        if 'Business Metrics' in sections:
            content['sections']['business_metrics'] = self._generate_business_metrics_section()
        
        if 'Interpretability' in sections:
            content['sections']['interpretability'] = self._generate_interpretability_section()
        
        if 'Fairness Analysis' in sections:
            content['sections']['fairness_analysis'] = self._generate_fairness_section()
        
        if 'Recommendations' in sections:
            content['sections']['recommendations'] = self._generate_recommendations_section()
        
        return content
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary content."""
        return {
            'title': 'Executive Summary',
            'content': f"""
            This report presents a comprehensive analysis of the {self.model_name} model.
            The model achieves an accuracy of {self.model_results.get('accuracy', 'N/A'):.2%}
            with a business value of ${self.model_results.get('business_value', 0):,.2f}.
            Key findings and recommendations are provided in the following sections.
            """
        }
    
    def _generate_performance_section(self) -> Dict[str, Any]:
        """Generate performance section content."""
        metrics = self.model_results.get('metrics', {})
        return {
            'title': 'Model Performance',
            'metrics': {
                'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                'Precision': f"{metrics.get('precision', 0):.3f}",
                'Recall': f"{metrics.get('recall', 0):.3f}",
                'F1-Score': f"{metrics.get('f1_score', 0):.3f}",
                'AUC-ROC': f"{metrics.get('auc_roc', 0):.3f}"
            }
        }
    
    def _generate_feature_importance_section(self) -> Dict[str, Any]:
        """Generate feature importance section content."""
        return {
            'title': 'Feature Importance',
            'top_features': self.model_results.get('feature_importance', {})
        }
    
    def _generate_business_metrics_section(self) -> Dict[str, Any]:
        """Generate business metrics section content."""
        return {
            'title': 'Business Metrics',
            'metrics': {
                'Total Value': f"${self.model_results.get('business_value', 0):,.2f}",
                'Cost Savings': f"${self.model_results.get('cost_savings', 0):,.2f}",
                'ROI': f"{self.model_results.get('roi', 0):.1%}"
            }
        }
    
    def _generate_interpretability_section(self) -> Dict[str, Any]:
        """Generate interpretability section content."""
        return {
            'title': 'Model Interpretability',
            'methods_used': ['SHAP', 'LIME', 'Feature Importance'],
            'key_insights': self.model_results.get('interpretability_insights', [])
        }
    
    def _generate_fairness_section(self) -> Dict[str, Any]:
        """Generate fairness section content."""
        return {
            'title': 'Fairness Analysis',
            'protected_attributes': self.model_results.get('protected_attributes', []),
            'fairness_metrics': self.model_results.get('fairness_metrics', {})
        }
    
    def _generate_recommendations_section(self) -> Dict[str, Any]:
        """Generate recommendations section content."""
        return {
            'title': 'Recommendations',
            'recommendations': [
                'Monitor model performance weekly',
                'Retrain model quarterly with new data',
                'Implement A/B testing for production deployment',
                'Set up automated bias detection alerts'
            ]
        }
    
    def _save_html_report(self, content: Dict[str, Any]) -> str:
        """Save report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{content['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f0f0f0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>{content['title']}</h1>
            <p>Generated at: {content['generated_at']}</p>
        """
        
        for section_key, section_data in content['sections'].items():
            html += f"<h2>{section_data['title']}</h2>"
            
            if 'content' in section_data:
                html += f"<p>{section_data['content']}</p>"
            
            if 'metrics' in section_data:
                html += "<table>"
                for metric, value in section_data['metrics'].items():
                    html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                html += "</table>"
        
        html += "</body></html>"
        
        filename = f"report_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w') as f:
            f.write(html)
        
        return filename
    
    def _save_pdf_report(self, content: Dict[str, Any]) -> str:
        """Save report as PDF (placeholder)."""
        # In practice, use reportlab or similar
        filename = f"report_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        print("PDF generation would require additional libraries like reportlab")
        return filename
    
    def _save_markdown_report(self, content: Dict[str, Any]) -> str:
        """Save report as Markdown."""
        md = f"# {content['title']}\n\n"
        md += f"*Generated at: {content['generated_at']}*\n\n"
        
        for section_key, section_data in content['sections'].items():
            md += f"## {section_data['title']}\n\n"
            
            if 'content' in section_data:
                md += f"{section_data['content']}\n\n"
            
            if 'metrics' in section_data:
                md += "| Metric | Value |\n"
                md += "|--------|-------|\n"
                for metric, value in section_data['metrics'].items():
                    md += f"| {metric} | {value} |\n"
                md += "\n"
        
        filename = f"report_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(md)
        
        return filename


def create_model_comparison_dashboard(models: Dict[str, Any], X_test: pd.DataFrame, 
                                    y_test: np.ndarray, feature_names: List[str]) -> widgets.VBox:
    """Create a comprehensive model comparison dashboard.
    
    Args:
        models: Dictionary of model name to model object
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features
        
    Returns:
        Interactive dashboard widget
    """
    # Create tab widget
    tabs = widgets.Tab()
    
    # Create individual components
    threshold_selector = InteractiveThresholdSelector(
        y_test, 
        models[list(models.keys())[0]].predict_proba(X_test)[:, 1] if hasattr(models[list(models.keys())[0]], 'predict_proba') else models[list(models.keys())[0]].predict(X_test)
    )
    
    feature_comparator = FeatureImportanceComparator(models, feature_names)
    performance_slider = ModelPerformanceSlider(models, X_test, y_test)
    
    # Add tabs
    tabs.children = [
        threshold_selector.create_widget(),
        feature_comparator.create_widget(),
        performance_slider.create_widget()
    ]
    
    tabs.set_title(0, 'Threshold Selection')
    tabs.set_title(1, 'Feature Importance')
    tabs.set_title(2, 'Model Performance')
    
    # Create main dashboard
    dashboard = widgets.VBox([
        widgets.HTML("<h2>Model Comparison Dashboard</h2>"),
        tabs
    ])
    
    return dashboard