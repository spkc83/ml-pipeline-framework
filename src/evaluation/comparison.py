import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from scipy import stats

from .metrics import MetricsCalculator, MetricsError

logger = logging.getLogger(__name__)

# Statistical test imports
try:
    from scipy.stats import mannwhitneyu, chi2_contingency, friedmanchisquare
    from scipy.stats import ttest_rel, wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available for statistical tests")
    SCIPY_AVAILABLE = False
    mannwhitneyu = chi2_contingency = friedmanchisquare = None
    ttest_rel = wilcoxon = None

# Cross-validation imports
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cross_val_score = StratifiedKFold = KFold = None


class ComparisonError(Exception):
    pass


class ModelComparator:
    """
    Comprehensive model comparison utility for evaluating multiple models
    across various metrics with statistical significance testing and guardrails.
    """
    
    def __init__(self, task_type: str = 'classification',
                 significance_level: float = 0.05,
                 ks_degradation_threshold: float = 0.05,
                 auc_degradation_threshold: float = 0.02):
        """
        Initialize model comparator.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            significance_level: P-value threshold for statistical significance
            ks_degradation_threshold: Maximum allowed KS statistic degradation
            auc_degradation_threshold: Maximum allowed AUC degradation
        """
        self.task_type = task_type
        self.significance_level = significance_level
        self.ks_degradation_threshold = ks_degradation_threshold
        self.auc_degradation_threshold = auc_degradation_threshold
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(task_type=task_type)
        
        self.comparison_history = []
        
        logger.info(f"Initialized ModelComparator for {task_type} task")
    
    def compare_models(self, models_dict: Dict[str, Dict[str, Any]], 
                      baseline_model: Optional[str] = None,
                      include_statistical_tests: bool = True,
                      include_guardrails: bool = True) -> Dict[str, Any]:
        """
        Compare multiple models across various metrics.
        
        Args:
            models_dict: Dictionary with model names as keys and values containing:
                        {'y_true', 'y_pred', 'y_pred_proba', 'model_object'}
            baseline_model: Name of baseline model for comparison (uses first if None)
            include_statistical_tests: Whether to perform statistical significance tests
            include_guardrails: Whether to check degradation guardrails
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        logger.info(f"Comparing {len(models_dict)} models")
        
        if len(models_dict) < 2:
            raise ComparisonError("At least 2 models required for comparison")
        
        # Set baseline model
        if baseline_model is None:
            baseline_model = list(models_dict.keys())[0]
        elif baseline_model not in models_dict:
            raise ComparisonError(f"Baseline model '{baseline_model}' not found in models_dict")
        
        comparison_results = {
            'baseline_model': baseline_model,
            'models_evaluated': list(models_dict.keys()),
            'comparison_timestamp': datetime.now().isoformat(),
            'task_type': self.task_type
        }
        
        try:
            # Calculate metrics for all models
            all_metrics = self._calculate_all_metrics(models_dict)
            comparison_results['individual_metrics'] = all_metrics
            
            # Create comparison tables
            comparison_tables = self._create_comparison_tables(all_metrics)
            comparison_results['comparison_tables'] = comparison_tables
            
            # Statistical significance tests
            if include_statistical_tests and SCIPY_AVAILABLE:
                statistical_tests = self._perform_statistical_tests(models_dict, baseline_model)
                comparison_results['statistical_tests'] = statistical_tests
            
            # Guardrail checks
            if include_guardrails:
                guardrail_results = self._check_guardrails(models_dict, baseline_model)
                comparison_results['guardrail_checks'] = guardrail_results
            
            # Model ranking
            rankings = self._rank_models(all_metrics)
            comparison_results['model_rankings'] = rankings
            
            # Performance degradation analysis
            degradation_analysis = self._analyze_performance_degradation(all_metrics, baseline_model)
            comparison_results['degradation_analysis'] = degradation_analysis
            
            # Business impact comparison
            if self.task_type == 'classification':
                business_impact = self._compare_business_impact(models_dict)
                comparison_results['business_impact_comparison'] = business_impact
            
            # Save comparison to history
            self.comparison_history.append(comparison_results)
            
            logger.info("Model comparison completed successfully")
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise ComparisonError(f"Comparison failed: {e}")
        
        return comparison_results
    
    def _calculate_all_metrics(self, models_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for all models."""
        all_metrics = {}
        
        for model_name, model_data in models_dict.items():
            logger.info(f"Calculating metrics for model: {model_name}")
            
            y_true = model_data['y_true']
            y_pred = model_data['y_pred']
            y_pred_proba = model_data.get('y_pred_proba')
            
            try:
                if self.task_type == 'classification':
                    metrics = self.metrics_calculator.calculate_classification_metrics(
                        y_true, y_pred, y_pred_proba
                    )
                    
                    # Add additional business metrics for binary classification
                    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                        # Business metrics by decile
                        business_metrics = self.metrics_calculator.calculate_business_metrics_by_decile(
                            y_true, y_pred_proba
                        )
                        metrics['business_metrics_by_decile'] = business_metrics.to_dict('records')
                        
                        # Lift and gain
                        lift_gain = self.metrics_calculator.calculate_lift_and_gain(y_true, y_pred_proba)
                        metrics['lift_gain_data'] = lift_gain.to_dict('records')
                        
                        # Calibration metrics
                        calibration = self.metrics_calculator.generate_calibration_plot(
                            y_true, y_pred_proba
                        )
                        metrics['calibration_metrics'] = calibration
                
                elif self.task_type == 'regression':
                    metrics = self.metrics_calculator.calculate_regression_metrics(y_true, y_pred)
                
                all_metrics[model_name] = metrics
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {model_name}: {e}")
                all_metrics[model_name] = {'error': str(e)}
        
        return all_metrics
    
    def _create_comparison_tables(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Create comparison tables for easy visualization."""
        tables = {}
        
        # Extract core metrics for comparison
        if self.task_type == 'classification':
            core_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 
                          'brier_score', 'ks_statistic', 'gini_coefficient']
        else:
            core_metrics = ['mse', 'rmse', 'mae', 'r2_score', 'mape', 'directional_accuracy']
        
        # Core metrics table
        core_data = []
        for model_name, metrics in all_metrics.items():
            if 'error' not in metrics:
                row = {'model': model_name}
                for metric in core_metrics:
                    row[metric] = metrics.get(metric, np.nan)
                core_data.append(row)
        
        if core_data:
            tables['core_metrics'] = pd.DataFrame(core_data).set_index('model')
            
            # Add ranking for each metric (higher is better for most classification metrics)
            if self.task_type == 'classification':
                ascending_metrics = ['brier_score']  # Lower is better
            else:
                ascending_metrics = ['mse', 'rmse', 'mae', 'mape']  # Lower is better
            
            for metric in core_metrics:
                if metric in tables['core_metrics'].columns:
                    ascending = metric in ascending_metrics
                    tables['core_metrics'][f'{metric}_rank'] = tables['core_metrics'][metric].rank(
                        ascending=ascending, method='dense'
                    )
        
        # Business metrics table (classification only)
        if self.task_type == 'classification':
            business_data = []
            for model_name, metrics in all_metrics.items():
                if 'business_metrics_by_decile' in metrics:
                    # Aggregate business metrics across deciles
                    decile_data = pd.DataFrame(metrics['business_metrics_by_decile'])
                    
                    # Top decile performance
                    top_decile = decile_data[decile_data['decile'] == 1].iloc[0] if len(decile_data) > 0 else {}
                    
                    business_row = {
                        'model': model_name,
                        'top_decile_positive_rate': top_decile.get('positive_rate', np.nan),
                        'top_decile_capture_rate': top_decile.get('capture_rate', np.nan),
                        'total_business_value': sum([d['business_value'] for d in metrics['business_metrics_by_decile']]),
                        'avg_confidence': top_decile.get('avg_prob', np.nan)
                    }
                    business_data.append(business_row)
            
            if business_data:
                tables['business_metrics'] = pd.DataFrame(business_data).set_index('model')
        
        return tables
    
    def _perform_statistical_tests(self, models_dict: Dict[str, Dict[str, Any]], 
                                 baseline_model: str) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        logger.info("Performing statistical significance tests")
        
        statistical_results = {}
        baseline_data = models_dict[baseline_model]
        
        for model_name, model_data in models_dict.items():
            if model_name == baseline_model:
                continue
            
            model_tests = {}
            
            try:
                # For classification tasks
                if self.task_type == 'classification' and 'y_pred_proba' in model_data and 'y_pred_proba' in baseline_data:
                    baseline_proba = baseline_data['y_pred_proba']
                    model_proba = model_data['y_pred_proba']
                    
                    # Ensure same length
                    min_len = min(len(baseline_proba), len(model_proba))
                    baseline_proba = baseline_proba[:min_len]
                    model_proba = model_proba[:min_len]
                    
                    # Mann-Whitney U test for predicted probabilities
                    if len(baseline_proba) > 10:  # Minimum sample size
                        u_stat, u_pvalue = mannwhitneyu(baseline_proba, model_proba, 
                                                       alternative='two-sided')
                        model_tests['mannwhitney_u'] = {
                            'statistic': float(u_stat),
                            'p_value': float(u_pvalue),
                            'significant': u_pvalue < self.significance_level
                        }
                    
                    # Paired t-test for AUC differences (if cross-validation data available)
                    # This would require multiple CV folds - simplified here
                    
                # McNemar's test for classification predictions
                if self.task_type == 'classification':
                    y_true = baseline_data['y_true']
                    baseline_pred = baseline_data['y_pred']
                    model_pred = model_data['y_pred']
                    
                    # Ensure same length
                    min_len = min(len(baseline_pred), len(model_pred), len(y_true))
                    y_true = y_true[:min_len]
                    baseline_pred = baseline_pred[:min_len]
                    model_pred = model_pred[:min_len]
                    
                    # Create contingency table for McNemar's test
                    baseline_correct = (baseline_pred == y_true)
                    model_correct = (model_pred == y_true)
                    
                    # McNemar's contingency table
                    both_correct = np.sum(baseline_correct & model_correct)
                    baseline_only = np.sum(baseline_correct & ~model_correct)
                    model_only = np.sum(~baseline_correct & model_correct)
                    both_wrong = np.sum(~baseline_correct & ~model_correct)
                    
                    contingency = np.array([[both_correct, baseline_only],
                                          [model_only, both_wrong]])
                    
                    # McNemar's test (simplified - focusing on disagreement cells)
                    if baseline_only + model_only > 0:
                        mcnemar_stat = (abs(baseline_only - model_only) - 1) ** 2 / (baseline_only + model_only)
                        mcnemar_pvalue = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
                        
                        model_tests['mcnemar_test'] = {
                            'statistic': float(mcnemar_stat),
                            'p_value': float(mcnemar_pvalue),
                            'significant': mcnemar_pvalue < self.significance_level,
                            'contingency_table': contingency.tolist()
                        }
                
                # For regression tasks
                elif self.task_type == 'regression':
                    y_true = baseline_data['y_true']
                    baseline_pred = baseline_data['y_pred']
                    model_pred = model_data['y_pred']
                    
                    # Ensure same length
                    min_len = min(len(baseline_pred), len(model_pred), len(y_true))
                    y_true = y_true[:min_len]
                    baseline_pred = baseline_pred[:min_len]
                    model_pred = model_pred[:min_len]
                    
                    # Calculate residuals
                    baseline_residuals = np.abs(y_true - baseline_pred)
                    model_residuals = np.abs(y_true - model_pred)
                    
                    # Paired t-test on absolute residuals
                    if len(baseline_residuals) > 10:
                        t_stat, t_pvalue = ttest_rel(baseline_residuals, model_residuals)
                        model_tests['paired_ttest_residuals'] = {
                            'statistic': float(t_stat),
                            'p_value': float(t_pvalue),
                            'significant': t_pvalue < self.significance_level
                        }
                    
                    # Wilcoxon signed-rank test (non-parametric alternative)
                    if len(baseline_residuals) > 10:
                        w_stat, w_pvalue = wilcoxon(baseline_residuals, model_residuals)
                        model_tests['wilcoxon_test'] = {
                            'statistic': float(w_stat),
                            'p_value': float(w_pvalue),
                            'significant': w_pvalue < self.significance_level
                        }
                
                statistical_results[model_name] = model_tests
                
            except Exception as e:
                logger.warning(f"Statistical test failed for {model_name}: {e}")
                statistical_results[model_name] = {'error': str(e)}
        
        return statistical_results
    
    def _check_guardrails(self, models_dict: Dict[str, Dict[str, Any]], 
                         baseline_model: str) -> Dict[str, Any]:
        """Check performance degradation guardrails."""
        logger.info("Checking performance degradation guardrails")
        
        guardrail_results = {
            'ks_degradation_threshold': self.ks_degradation_threshold,
            'auc_degradation_threshold': self.auc_degradation_threshold,
            'checks': {}
        }
        
        baseline_data = models_dict[baseline_model]
        
        # Calculate baseline metrics
        if self.task_type == 'classification' and 'y_pred_proba' in baseline_data:
            baseline_y_true = baseline_data['y_true']
            baseline_y_pred_proba = baseline_data['y_pred_proba']
            
            baseline_auc = self.metrics_calculator.calculate_classification_metrics(
                baseline_y_true, baseline_data['y_pred'], baseline_y_pred_proba
            ).get('roc_auc', 0)
            
            baseline_ks = self.metrics_calculator.calculate_ks_statistic(
                baseline_y_true, baseline_y_pred_proba
            )
            
            for model_name, model_data in models_dict.items():
                if model_name == baseline_model or 'y_pred_proba' not in model_data:
                    continue
                
                model_checks = {}
                
                try:
                    # Calculate model metrics
                    model_y_true = model_data['y_true']
                    model_y_pred_proba = model_data['y_pred_proba']
                    
                    model_metrics = self.metrics_calculator.calculate_classification_metrics(
                        model_y_true, model_data['y_pred'], model_y_pred_proba
                    )
                    
                    model_auc = model_metrics.get('roc_auc', 0)
                    model_ks = model_metrics.get('ks_statistic', 0)
                    
                    # AUC degradation check
                    auc_degradation = baseline_auc - model_auc
                    auc_pass = auc_degradation <= self.auc_degradation_threshold
                    
                    model_checks['auc_check'] = {
                        'baseline_auc': baseline_auc,
                        'model_auc': model_auc,
                        'degradation': auc_degradation,
                        'threshold': self.auc_degradation_threshold,
                        'pass': auc_pass
                    }
                    
                    # KS degradation check
                    ks_degradation = baseline_ks - model_ks
                    ks_pass = ks_degradation <= self.ks_degradation_threshold
                    
                    model_checks['ks_check'] = {
                        'baseline_ks': baseline_ks,
                        'model_ks': model_ks,
                        'degradation': ks_degradation,
                        'threshold': self.ks_degradation_threshold,
                        'pass': ks_pass
                    }
                    
                    # Overall guardrail pass
                    model_checks['overall_pass'] = auc_pass and ks_pass
                    
                    guardrail_results['checks'][model_name] = model_checks
                    
                except Exception as e:
                    logger.warning(f"Guardrail check failed for {model_name}: {e}")
                    guardrail_results['checks'][model_name] = {'error': str(e)}
        
        return guardrail_results
    
    def _rank_models(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Rank models based on primary metrics."""
        logger.info("Ranking models")
        
        rankings = {}
        
        if self.task_type == 'classification':
            # Primary metrics for classification
            primary_metrics = ['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy']
            metric_weights = {'roc_auc': 0.3, 'f1_score': 0.25, 'precision': 0.2, 
                            'recall': 0.15, 'accuracy': 0.1}
        else:
            # Primary metrics for regression (lower is better)
            primary_metrics = ['r2_score', 'rmse', 'mae', 'mape']
            metric_weights = {'r2_score': 0.4, 'rmse': 0.3, 'mae': 0.2, 'mape': 0.1}
            reverse_metrics = ['rmse', 'mae', 'mape']  # Lower is better
        
        model_scores = {}
        
        for model_name, metrics in all_metrics.items():
            if 'error' in metrics:
                continue
            
            weighted_score = 0
            available_weight = 0
            
            for metric, weight in metric_weights.items():
                if metric in metrics and not np.isnan(metrics[metric]):
                    metric_value = metrics[metric]
                    
                    # Normalize metric (simple min-max normalization across models)
                    all_values = [m.get(metric, np.nan) for m in all_metrics.values() 
                                if 'error' not in m and metric in m]
                    all_values = [v for v in all_values if not np.isnan(v)]
                    
                    if len(all_values) > 1:
                        if self.task_type == 'regression' and metric in reverse_metrics:
                            # For "lower is better" metrics, invert the normalization
                            normalized_value = (max(all_values) - metric_value) / (max(all_values) - min(all_values))
                        else:
                            # For "higher is better" metrics
                            normalized_value = (metric_value - min(all_values)) / (max(all_values) - min(all_values))
                        
                        weighted_score += normalized_value * weight
                        available_weight += weight
            
            if available_weight > 0:
                model_scores[model_name] = weighted_score / available_weight
        
        # Sort by weighted score (higher is better)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings['weighted_ranking'] = [{'rank': i+1, 'model': model, 'score': score} 
                                      for i, (model, score) in enumerate(sorted_models)]
        
        # Individual metric rankings
        rankings['individual_metric_rankings'] = {}
        for metric in primary_metrics:
            metric_values = [(name, metrics.get(metric, np.nan)) 
                           for name, metrics in all_metrics.items() 
                           if 'error' not in metrics and metric in metrics]
            
            # Sort by metric value
            if self.task_type == 'regression' and metric in reverse_metrics:
                metric_values.sort(key=lambda x: x[1])  # Lower is better
            else:
                metric_values.sort(key=lambda x: x[1], reverse=True)  # Higher is better
            
            rankings['individual_metric_rankings'][metric] = [
                {'rank': i+1, 'model': model, 'value': value} 
                for i, (model, value) in enumerate(metric_values)
            ]
        
        return rankings
    
    def _analyze_performance_degradation(self, all_metrics: Dict[str, Dict[str, Any]], 
                                       baseline_model: str) -> Dict[str, Any]:
        """Analyze performance degradation relative to baseline."""
        logger.info("Analyzing performance degradation")
        
        degradation_analysis = {'baseline_model': baseline_model, 'degradations': {}}
        
        baseline_metrics = all_metrics[baseline_model]
        if 'error' in baseline_metrics:
            return degradation_analysis
        
        for model_name, model_metrics in all_metrics.items():
            if model_name == baseline_model or 'error' in model_metrics:
                continue
            
            model_degradation = {}
            
            # Compare key metrics
            key_metrics = (['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy'] 
                         if self.task_type == 'classification' 
                         else ['r2_score', 'rmse', 'mae', 'mape'])
            
            for metric in key_metrics:
                if metric in baseline_metrics and metric in model_metrics:
                    baseline_value = baseline_metrics[metric]
                    model_value = model_metrics[metric]
                    
                    if not np.isnan(baseline_value) and not np.isnan(model_value):
                        if self.task_type == 'regression' and metric in ['rmse', 'mae', 'mape']:
                            # For "lower is better" metrics
                            degradation = (model_value - baseline_value) / baseline_value
                        else:
                            # For "higher is better" metrics
                            degradation = (baseline_value - model_value) / baseline_value
                        
                        model_degradation[metric] = {
                            'baseline_value': baseline_value,
                            'model_value': model_value,
                            'absolute_difference': model_value - baseline_value,
                            'relative_degradation': degradation,
                            'degradation_percentage': degradation * 100
                        }
            
            degradation_analysis['degradations'][model_name] = model_degradation
        
        return degradation_analysis
    
    def _compare_business_impact(self, models_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare business impact across models."""
        logger.info("Comparing business impact")
        
        business_comparison = {}
        
        for model_name, model_data in models_dict.items():
            if 'y_pred_proba' not in model_data:
                continue
            
            try:
                y_true = model_data['y_true']
                y_pred_proba = model_data['y_pred_proba']
                
                # Calculate business metrics by decile
                business_metrics = self.metrics_calculator.calculate_business_metrics_by_decile(
                    y_true, y_pred_proba
                )
                
                # Aggregate business impact
                total_business_value = business_metrics['business_value'].sum()
                top_decile_capture = business_metrics[business_metrics['decile'] == 1]['capture_rate'].iloc[0] if len(business_metrics) > 0 else 0
                
                business_comparison[model_name] = {
                    'total_business_value': total_business_value,
                    'top_decile_capture_rate': top_decile_capture,
                    'business_metrics_by_decile': business_metrics.to_dict('records')
                }
                
            except Exception as e:
                logger.warning(f"Business impact calculation failed for {model_name}: {e}")
                business_comparison[model_name] = {'error': str(e)}
        
        return business_comparison
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any],
                                 output_dir: str = "./model_comparison") -> str:
        """
        Generate comprehensive model comparison report.
        
        Args:
            comparison_results: Results from compare_models method
            output_dir: Directory to save report
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating model comparison report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"model_comparison_report_{timestamp}.html"
        
        try:
            # Generate HTML report
            html_content = self._generate_html_report(comparison_results)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save raw results as JSON
            json_path = output_path / f"model_comparison_data_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)
            
            # Generate comparison plots
            self._generate_comparison_plots(comparison_results, output_path, timestamp)
            
            logger.info(f"Model comparison report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            raise ComparisonError(f"Report generation failed: {e}")
        
        return str(report_path)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-table {{ margin: 10px 0; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Comparison Report</h1>
                <p><strong>Baseline Model:</strong> {results['baseline_model']}</p>
                <p><strong>Models Evaluated:</strong> {', '.join(results['models_evaluated'])}</p>
                <p><strong>Task Type:</strong> {results['task_type']}</p>
                <p><strong>Generated:</strong> {results['comparison_timestamp']}</p>
            </div>
        """
        
        # Core metrics comparison table
        if 'comparison_tables' in results and 'core_metrics' in results['comparison_tables']:
            html_content += """
            <div class="section">
                <h2>Core Metrics Comparison</h2>
                <div class="metric-table">
            """
            
            core_metrics_df = results['comparison_tables']['core_metrics']
            html_content += core_metrics_df.to_html(classes='metric-table')
            html_content += "</div></div>"
        
        # Model rankings
        if 'model_rankings' in results:
            html_content += """
            <div class="section">
                <h2>Model Rankings</h2>
                <h3>Overall Weighted Ranking</h3>
                <table>
                    <tr><th>Rank</th><th>Model</th><th>Weighted Score</th></tr>
            """
            
            for rank_info in results['model_rankings']['weighted_ranking']:
                html_content += f"""
                    <tr>
                        <td>{rank_info['rank']}</td>
                        <td>{rank_info['model']}</td>
                        <td>{rank_info['score']:.4f}</td>
                    </tr>
                """
            
            html_content += "</table></div>"
        
        # Guardrail checks
        if 'guardrail_checks' in results and results['guardrail_checks']['checks']:
            html_content += """
            <div class="section">
                <h2>Guardrail Checks</h2>
                <table>
                    <tr><th>Model</th><th>AUC Check</th><th>KS Check</th><th>Overall</th></tr>
            """
            
            for model, checks in results['guardrail_checks']['checks'].items():
                if 'error' not in checks:
                    auc_status = '<span class="pass">PASS</span>' if checks['auc_check']['pass'] else '<span class="fail">FAIL</span>'
                    ks_status = '<span class="pass">PASS</span>' if checks['ks_check']['pass'] else '<span class="fail">FAIL</span>'
                    overall_status = '<span class="pass">PASS</span>' if checks['overall_pass'] else '<span class="fail">FAIL</span>'
                    
                    html_content += f"""
                        <tr>
                            <td>{model}</td>
                            <td>{auc_status}</td>
                            <td>{ks_status}</td>
                            <td>{overall_status}</td>
                        </tr>
                    """
            
            html_content += "</table></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_comparison_plots(self, results: Dict[str, Any], 
                                 output_path: Path, timestamp: str) -> None:
        """Generate comparison plots."""
        try:
            # Model performance comparison plot
            if 'comparison_tables' in results and 'core_metrics' in results['comparison_tables']:
                df = results['comparison_tables']['core_metrics']
                
                # Select key metrics for plotting
                if self.task_type == 'classification':
                    plot_metrics = ['roc_auc', 'f1_score', 'precision', 'recall']
                else:
                    plot_metrics = ['r2_score', 'rmse', 'mae']
                
                available_metrics = [m for m in plot_metrics if m in df.columns]
                
                if available_metrics:
                    fig, axes = plt.subplots(1, len(available_metrics), 
                                           figsize=(5*len(available_metrics), 6))
                    if len(available_metrics) == 1:
                        axes = [axes]
                    
                    for i, metric in enumerate(available_metrics):
                        df[metric].plot(kind='bar', ax=axes[i], title=f'{metric.upper()}')
                        axes[i].set_ylabel(metric)
                        axes[i].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    plot_path = output_path / f"model_performance_comparison_{timestamp}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Guardrail visualization
            if 'guardrail_checks' in results and results['guardrail_checks']['checks']:
                models = []
                auc_degradations = []
                ks_degradations = []
                
                for model, checks in results['guardrail_checks']['checks'].items():
                    if 'error' not in checks:
                        models.append(model)
                        auc_degradations.append(checks['auc_check']['degradation'])
                        ks_degradations.append(checks['ks_check']['degradation'])
                
                if models:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # AUC degradation
                    bars1 = ax1.bar(models, auc_degradations)
                    ax1.axhline(y=self.auc_degradation_threshold, color='red', 
                               linestyle='--', label='Threshold')
                    ax1.set_title('AUC Degradation vs Baseline')
                    ax1.set_ylabel('AUC Degradation')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.legend()
                    
                    # Color bars based on pass/fail
                    for i, bar in enumerate(bars1):
                        if auc_degradations[i] <= self.auc_degradation_threshold:
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
                    
                    # KS degradation
                    bars2 = ax2.bar(models, ks_degradations)
                    ax2.axhline(y=self.ks_degradation_threshold, color='red', 
                               linestyle='--', label='Threshold')
                    ax2.set_title('KS Degradation vs Baseline')
                    ax2.set_ylabel('KS Degradation')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.legend()
                    
                    # Color bars based on pass/fail
                    for i, bar in enumerate(bars2):
                        if ks_degradations[i] <= self.ks_degradation_threshold:
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
                    
                    plt.tight_layout()
                    guardrail_path = output_path / f"guardrail_checks_{timestamp}.png"
                    plt.savefig(guardrail_path, dpi=300, bbox_inches='tight')
                    plt.close()
        
        except Exception as e:
            logger.warning(f"Failed to generate comparison plots: {e}")
    
    def cross_validate_models(self, models_dict: Dict[str, Any], 
                            X: np.ndarray, y: np.ndarray,
                            cv_folds: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation comparison of multiple models.
        
        Args:
            models_dict: Dictionary of model names to model objects
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results and statistical tests
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation comparison")
        
        if not SKLEARN_AVAILABLE:
            raise ComparisonError("Scikit-learn is required for cross-validation")
        
        cv_results = {}
        cv_scores = {}
        
        # Set up cross-validation
        if self.task_type == 'classification' and len(np.unique(y)) > 1:
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform CV for each model
        for model_name, model in models_dict.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
                cv_scores[model_name] = scores
                
                cv_results[model_name] = {
                    'scores': scores.tolist(),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores)
                }
                
                logger.info(f"{model_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                logger.warning(f"CV failed for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        # Statistical tests between models
        if len(cv_scores) >= 2 and SCIPY_AVAILABLE:
            statistical_tests = {}
            
            model_names = list(cv_scores.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    scores1, scores2 = cv_scores[model1], cv_scores[model2]
                    
                    # Paired t-test
                    t_stat, t_pvalue = ttest_rel(scores1, scores2)
                    
                    # Wilcoxon signed-rank test
                    w_stat, w_pvalue = wilcoxon(scores1, scores2)
                    
                    test_key = f"{model1}_vs_{model2}"
                    statistical_tests[test_key] = {
                        'paired_ttest': {
                            'statistic': float(t_stat),
                            'p_value': float(t_pvalue),
                            'significant': t_pvalue < self.significance_level
                        },
                        'wilcoxon_test': {
                            'statistic': float(w_stat),
                            'p_value': float(w_pvalue),
                            'significant': w_pvalue < self.significance_level
                        }
                    }
            
            cv_results['statistical_tests'] = statistical_tests
        
        return cv_results
    
    def get_comparison_history(self) -> List[Dict[str, Any]]:
        """Get history of all model comparisons."""
        return self.comparison_history.copy()
    
    def export_comparison_summary(self, comparison_results: Dict[str, Any], 
                                output_path: str) -> None:
        """
        Export comparison summary to CSV.
        
        Args:
            comparison_results: Results from compare_models method
            output_path: Path to save CSV file
        """
        logger.info(f"Exporting comparison summary to: {output_path}")
        
        try:
            summary_data = []
            
            # Extract key metrics for each model
            for model_name, metrics in comparison_results['individual_metrics'].items():
                if 'error' not in metrics:
                    row = {'model': model_name}
                    
                    # Add core metrics
                    if self.task_type == 'classification':
                        core_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                                      'roc_auc', 'pr_auc', 'ks_statistic']
                    else:
                        core_metrics = ['mse', 'rmse', 'mae', 'r2_score', 'mape']
                    
                    for metric in core_metrics:
                        row[metric] = metrics.get(metric, np.nan)
                    
                    # Add ranking if available
                    if 'model_rankings' in comparison_results:
                        rankings = comparison_results['model_rankings']['weighted_ranking']
                        model_rank = next((r['rank'] for r in rankings if r['model'] == model_name), None)
                        row['overall_rank'] = model_rank
                    
                    # Add guardrail status
                    if 'guardrail_checks' in comparison_results and model_name in comparison_results['guardrail_checks']['checks']:
                        guardrail_info = comparison_results['guardrail_checks']['checks'][model_name]
                        if 'error' not in guardrail_info:
                            row['guardrail_pass'] = guardrail_info['overall_pass']
                    
                    summary_data.append(row)
            
            # Create DataFrame and save
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path, index=False)
            
            logger.info(f"Comparison summary exported successfully")
            
        except Exception as e:
            logger.error(f"Failed to export comparison summary: {e}")
            raise ComparisonError(f"Export failed: {e}")