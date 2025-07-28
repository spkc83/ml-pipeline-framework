"""
Unit tests for evaluation modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import os
from pathlib import Path

from src.evaluation.metrics import MetricsCalculator, ThresholdOptimizer
from src.evaluation.comparison import ModelComparator
from src.evaluation.stability import StabilityAnalyzer


class TestMetricsCalculator:
    """Test cases for MetricsCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_calculator = MetricsCalculator(task_type='classification')
        
        # Create sample predictions
        np.random.seed(42)
        self.y_true = np.random.choice([0, 1], 100, p=[0.6, 0.4])
        self.y_pred = np.random.choice([0, 1], 100, p=[0.5, 0.5])
        self.y_pred_proba = np.random.uniform(0, 1, 100)
    
    def test_initialization_classification(self):
        """Test initialization for classification."""
        calc = MetricsCalculator(task_type='classification')
        assert calc.task_type == 'classification'
        assert calc.pos_label == 1
    
    def test_initialization_regression(self):
        """Test initialization for regression."""
        calc = MetricsCalculator(task_type='regression')
        assert calc.task_type == 'regression'
    
    def test_invalid_task_type(self):
        """Test error for invalid task type."""
        with pytest.raises(ValueError, match="Unsupported task type"):
            MetricsCalculator(task_type='invalid')
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics = self.metrics_calculator.calculate_classification_metrics(
            self.y_true, self.y_pred, self.y_pred_proba
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'pr_auc', 'brier_score'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1
        assert 0 <= metrics['brier_score'] <= 1
    
    def test_calculate_classification_metrics_without_proba(self):
        """Test classification metrics without probability predictions."""
        metrics = self.metrics_calculator.calculate_classification_metrics(
            self.y_true, self.y_pred
        )
        
        # Should have basic metrics but not probability-based ones
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' not in metrics
        assert 'pr_auc' not in metrics
        assert 'brier_score' not in metrics
    
    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation."""
        calc = MetricsCalculator(task_type='regression')
        
        y_true = np.random.normal(0, 1, 100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add some noise
        
        metrics = calc.calculate_regression_metrics(y_true, y_pred)
        
        expected_metrics = ['mse', 'rmse', 'mae', 'r2_score', 'mape']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # R² should be high for good predictions
        assert metrics['r2_score'] > 0.8
        # Errors should be relatively low
        assert metrics['mse'] < 1.0
        assert metrics['rmse'] < 1.0
        assert metrics['mae'] < 1.0
    
    def test_calculate_roc_auc(self):
        """Test ROC AUC calculation."""
        roc_auc = self.metrics_calculator.calculate_roc_auc(self.y_true, self.y_pred_proba)
        
        assert isinstance(roc_auc, float)
        assert 0 <= roc_auc <= 1
    
    def test_calculate_pr_auc(self):
        """Test Precision-Recall AUC calculation."""
        pr_auc = self.metrics_calculator.calculate_pr_auc(self.y_true, self.y_pred_proba)
        
        assert isinstance(pr_auc, float)
        assert 0 <= pr_auc <= 1
    
    def test_calculate_brier_score(self):
        """Test Brier score calculation."""
        brier_score = self.metrics_calculator.calculate_brier_score(self.y_true, self.y_pred_proba)
        
        assert isinstance(brier_score, float)
        assert 0 <= brier_score <= 1
    
    def test_calculate_ks_statistic(self):
        """Test KS statistic calculation."""
        ks_stat = self.metrics_calculator.calculate_ks_statistic(self.y_true, self.y_pred_proba)
        
        assert isinstance(ks_stat, float)
        assert 0 <= ks_stat <= 1
    
    def test_calculate_business_metrics_by_decile(self):
        """Test business metrics by decile calculation."""
        business_metrics = self.metrics_calculator.calculate_business_metrics_by_decile(
            self.y_true, self.y_pred_proba
        )
        
        assert isinstance(business_metrics, pd.DataFrame)
        assert len(business_metrics) == 10  # 10 deciles
        
        expected_columns = [
            'decile', 'count', 'positive_count', 'positive_rate',
            'cumulative_positive_count', 'cumulative_positive_rate',
            'lift', 'cumulative_lift'
        ]
        
        for col in expected_columns:
            assert col in business_metrics.columns
    
    def test_calculate_calibration_metrics(self):
        """Test calibration metrics calculation."""
        calibration_metrics = self.metrics_calculator.calculate_calibration_metrics(
            self.y_true, self.y_pred_proba, n_bins=10
        )
        
        assert isinstance(calibration_metrics, dict)
        assert 'ece' in calibration_metrics  # Expected Calibration Error
        assert 'mce' in calibration_metrics  # Maximum Calibration Error
        assert 'bin_boundaries' in calibration_metrics
        assert 'bin_lowers' in calibration_metrics
        assert 'bin_uppers' in calibration_metrics
        assert 'bin_accs' in calibration_metrics
        assert 'bin_confs' in calibration_metrics
        assert 'bin_sizes' in calibration_metrics
    
    def test_calculate_lift_and_gain(self):
        """Test lift and gain calculation."""
        lift_gain = self.metrics_calculator.calculate_lift_and_gain(
            self.y_true, self.y_pred_proba
        )
        
        assert isinstance(lift_gain, dict)
        assert 'lift_at_percentile' in lift_gain
        assert 'cumulative_gain' in lift_gain
        assert 'percentiles' in lift_gain
    
    def test_calculate_population_stability_index(self):
        """Test PSI calculation."""
        # Create baseline and current distributions
        baseline_proba = np.random.uniform(0, 1, 1000)
        current_proba = baseline_proba + np.random.normal(0, 0.1, 1000)  # Slight shift
        current_proba = np.clip(current_proba, 0, 1)
        
        psi = self.metrics_calculator.calculate_population_stability_index(
            baseline_proba, current_proba
        )
        
        assert isinstance(psi, float)
        assert psi >= 0  # PSI is always non-negative
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_roc_curve_plot(self, mock_figure, mock_savefig):
        """Test ROC curve plot generation."""
        output_path = '/tmp/test_roc.png'
        
        self.metrics_calculator.generate_roc_curve_plot(
            self.y_true, self.y_pred_proba, output_path
        )
        
        mock_figure.assert_called()
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_pr_curve_plot(self, mock_figure, mock_savefig):
        """Test PR curve plot generation."""
        output_path = '/tmp/test_pr.png'
        
        self.metrics_calculator.generate_pr_curve_plot(
            self.y_true, self.y_pred_proba, output_path
        )
        
        mock_figure.assert_called()
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_calibration_plot(self, mock_figure, mock_savefig):
        """Test calibration plot generation."""
        output_path = '/tmp/test_calibration.png'
        
        self.metrics_calculator.generate_calibration_plot(
            self.y_true, self.y_pred_proba, output_path
        )
        
        mock_figure.assert_called()
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    def test_generate_comprehensive_report(self, temp_directory):
        """Test comprehensive report generation."""
        report = self.metrics_calculator.generate_comprehensive_report(
            self.y_true, self.y_pred, self.y_pred_proba,
            output_dir=temp_directory
        )
        
        assert isinstance(report, dict)
        assert 'metrics' in report
        assert 'plots' in report
        assert 'business_metrics' in report
        
        # Check that files were created
        output_path = Path(temp_directory)
        assert (output_path / 'metrics_summary.json').exists()
        assert (output_path / 'business_metrics.csv').exists()


class TestThresholdOptimizer:
    """Test cases for ThresholdOptimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true = np.random.choice([0, 1], 100, p=[0.6, 0.4])
        self.y_pred_proba = np.random.uniform(0, 1, 100)
        
        self.optimizer = ThresholdOptimizer()
    
    def test_optimize_threshold_f1(self):
        """Test threshold optimization for F1 score."""
        best_threshold, best_score = self.optimizer.optimize_threshold(
            self.y_true, self.y_pred_proba, metric='f1_score'
        )
        
        assert isinstance(best_threshold, float)
        assert 0 <= best_threshold <= 1
        assert isinstance(best_score, float)
        assert 0 <= best_score <= 1
    
    def test_optimize_threshold_precision(self):
        """Test threshold optimization for precision."""
        best_threshold, best_score = self.optimizer.optimize_threshold(
            self.y_true, self.y_pred_proba, metric='precision'
        )
        
        assert isinstance(best_threshold, float)
        assert 0 <= best_threshold <= 1
        assert isinstance(best_score, float)
        assert 0 <= best_score <= 1
    
    def test_optimize_threshold_recall(self):
        """Test threshold optimization for recall."""
        best_threshold, best_score = self.optimizer.optimize_threshold(
            self.y_true, self.y_pred_proba, metric='recall'
        )
        
        assert isinstance(best_threshold, float)
        assert isinstance(best_score, float)
    
    def test_optimize_threshold_accuracy(self):
        """Test threshold optimization for accuracy."""
        best_threshold, best_score = self.optimizer.optimize_threshold(
            self.y_true, self.y_pred_proba, metric='accuracy'
        )
        
        assert isinstance(best_threshold, float)
        assert isinstance(best_score, float)
    
    def test_optimize_threshold_business_value(self):
        """Test threshold optimization for business value."""
        # Define simple business value function
        cost_matrix = np.array([[0, 1], [5, 0]])  # FN is 5x more expensive than FP
        
        best_threshold, best_value = self.optimizer.optimize_threshold_business_value(
            self.y_true, self.y_pred_proba, cost_matrix
        )
        
        assert isinstance(best_threshold, float)
        assert 0 <= best_threshold <= 1
        assert isinstance(best_value, float)
    
    def test_invalid_metric(self):
        """Test error for invalid metric."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            self.optimizer.optimize_threshold(
                self.y_true, self.y_pred_proba, metric='invalid_metric'
            )
    
    def test_threshold_analysis(self):
        """Test threshold analysis."""
        analysis = self.optimizer.threshold_analysis(
            self.y_true, self.y_pred_proba
        )
        
        assert isinstance(analysis, pd.DataFrame)
        assert 'threshold' in analysis.columns
        assert 'precision' in analysis.columns
        assert 'recall' in analysis.columns
        assert 'f1_score' in analysis.columns
        assert 'accuracy' in analysis.columns


class TestModelComparator:
    """Test cases for ModelComparator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = ModelComparator()
        
        # Create sample model results
        np.random.seed(42)
        n_samples = 100
        self.y_true = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        # Create predictions for multiple models
        self.model_predictions = {
            'model_1': {
                'y_pred': np.random.choice([0, 1], n_samples),
                'y_pred_proba': np.random.uniform(0, 1, n_samples)
            },
            'model_2': {
                'y_pred': np.random.choice([0, 1], n_samples),
                'y_pred_proba': np.random.uniform(0, 1, n_samples)
            },
            'model_3': {
                'y_pred': np.random.choice([0, 1], n_samples),
                'y_pred_proba': np.random.uniform(0, 1, n_samples)
            }
        }
    
    def test_compare_models(self):
        """Test model comparison."""
        comparison_results = self.comparator.compare_models(
            self.y_true, self.model_predictions
        )
        
        assert isinstance(comparison_results, dict)
        assert 'comparison_table' in comparison_results
        assert 'statistical_tests' in comparison_results
        assert 'ranking' in comparison_results
        
        # Check comparison table
        comparison_table = comparison_results['comparison_table']
        assert isinstance(comparison_table, pd.DataFrame)
        assert len(comparison_table) == 3  # 3 models
        assert 'model_name' in comparison_table.columns
        assert 'accuracy' in comparison_table.columns
        assert 'roc_auc' in comparison_table.columns
    
    def test_mcnemar_test(self):
        """Test McNemar's test."""
        model1_pred = self.model_predictions['model_1']['y_pred']
        model2_pred = self.model_predictions['model_2']['y_pred']
        
        statistic, p_value = self.comparator.mcnemar_test(
            self.y_true, model1_pred, model2_pred
        )
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test."""
        model1_proba = self.model_predictions['model_1']['y_pred_proba']
        model2_proba = self.model_predictions['model_2']['y_pred_proba']
        
        statistic, p_value = self.comparator.mann_whitney_u_test(
            model1_proba, model2_proba
        )
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_paired_t_test(self):
        """Test paired t-test."""
        model1_proba = self.model_predictions['model_1']['y_pred_proba']
        model2_proba = self.model_predictions['model_2']['y_pred_proba']
        
        statistic, p_value = self.comparator.paired_t_test(
            model1_proba, model2_proba
        )
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_cross_validation_comparison(self, sample_dataframe):
        """Test cross-validation comparison."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        comparison_results = self.comparator.cross_validation_comparison(
            models, X, y, cv_folds=3, scoring='accuracy'
        )
        
        assert isinstance(comparison_results, dict)
        assert 'mean_scores' in comparison_results
        assert 'std_scores' in comparison_results
        assert 'statistical_significance' in comparison_results
        
        # Check that both models are included
        assert 'rf' in comparison_results['mean_scores']
        assert 'lr' in comparison_results['mean_scores']
    
    def test_ks_degradation_check(self):
        """Test KS degradation check."""
        # Create baseline and current predictions
        baseline_proba = self.model_predictions['model_1']['y_pred_proba']
        current_proba = baseline_proba + np.random.normal(0, 0.1, len(baseline_proba))
        current_proba = np.clip(current_proba, 0, 1)
        
        degradation_result = self.comparator.check_ks_degradation(
            self.y_true, baseline_proba, current_proba, threshold=0.1
        )
        
        assert isinstance(degradation_result, dict)
        assert 'baseline_ks' in degradation_result
        assert 'current_ks' in degradation_result
        assert 'ks_difference' in degradation_result
        assert 'degradation_detected' in degradation_result
        assert isinstance(degradation_result['degradation_detected'], bool)
    
    def test_auc_degradation_check(self):
        """Test AUC degradation check."""
        baseline_proba = self.model_predictions['model_1']['y_pred_proba']
        current_proba = baseline_proba + np.random.normal(0, 0.1, len(baseline_proba))
        current_proba = np.clip(current_proba, 0, 1)
        
        degradation_result = self.comparator.check_auc_degradation(
            self.y_true, baseline_proba, current_proba, threshold=0.05
        )
        
        assert isinstance(degradation_result, dict)
        assert 'baseline_auc' in degradation_result
        assert 'current_auc' in degradation_result
        assert 'auc_difference' in degradation_result
        assert 'degradation_detected' in degradation_result
    
    def test_generate_comparison_report(self, temp_directory):
        """Test comparison report generation."""
        report = self.comparator.generate_comparison_report(
            self.y_true, self.model_predictions, output_dir=temp_directory
        )
        
        assert isinstance(report, dict)
        assert 'comparison_table' in report
        assert 'best_model' in report
        assert 'report_path' in report
        
        # Check that report file was created
        assert os.path.exists(report['report_path'])


class TestStabilityAnalyzer:
    """Test cases for StabilityAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StabilityAnalyzer()
    
    def test_calculate_psi(self):
        """Test PSI calculation."""
        # Create baseline and current distributions
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.1, 1.1, 1000)  # Slight distribution shift
        
        psi = self.analyzer.calculate_psi(baseline, current)
        
        assert isinstance(psi, float)
        assert psi >= 0
        assert psi > 0  # Should detect some drift
    
    def test_calculate_csi(self):
        """Test CSI calculation."""
        # Create baseline and current score distributions
        np.random.seed(42)
        baseline_scores = np.random.uniform(0, 1, 1000)
        current_scores = baseline_scores + np.random.normal(0, 0.05, 1000)
        current_scores = np.clip(current_scores, 0, 1)
        
        csi = self.analyzer.calculate_csi(baseline_scores, current_scores)
        
        assert isinstance(csi, float)
        assert csi >= 0
    
    def test_feature_stability_analysis(self, sample_dataframe):
        """Test feature stability analysis."""
        # Split data to simulate baseline and current periods
        baseline_data = sample_dataframe.iloc[:500]
        current_data = sample_dataframe.iloc[500:]
        
        stability_results = self.analyzer.feature_stability_analysis(
            baseline_data, current_data
        )
        
        assert isinstance(stability_results, dict)
        assert 'feature_psi' in stability_results
        assert 'overall_stability' in stability_results
        assert 'unstable_features' in stability_results
        
        # Check that PSI was calculated for each feature
        for feature in baseline_data.columns:
            if baseline_data[feature].dtype in ['int64', 'float64']:
                assert feature in stability_results['feature_psi']
    
    def test_model_score_stability(self):
        """Test model score stability analysis."""
        # Create baseline and current model scores
        np.random.seed(42)
        baseline_scores = np.random.uniform(0, 1, 1000)
        current_scores = baseline_scores + np.random.normal(0, 0.1, 1000)
        current_scores = np.clip(current_scores, 0, 1)
        
        stability_result = self.analyzer.model_score_stability(
            baseline_scores, current_scores
        )
        
        assert isinstance(stability_result, dict)
        assert 'csi' in stability_result
        assert 'stability_status' in stability_result
        assert 'score_drift_detected' in stability_result
    
    def test_performance_stability_over_time(self):
        """Test performance stability over time."""
        # Create time series of model performance
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        auc_scores = np.random.normal(0.8, 0.05, 12)  # AUC scores with some variation
        
        performance_data = pd.DataFrame({
            'date': dates,
            'auc': auc_scores
        })
        
        stability_result = self.analyzer.performance_stability_over_time(
            performance_data, metric_column='auc', date_column='date'
        )
        
        assert isinstance(stability_result, dict)
        assert 'trend_analysis' in stability_result
        assert 'volatility' in stability_result
        assert 'stability_score' in stability_result
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_stability_report(self, mock_figure, mock_savefig, sample_dataframe, temp_directory):
        """Test stability report generation."""
        # Split data
        baseline_data = sample_dataframe.iloc[:500]
        current_data = sample_dataframe.iloc[500:]
        
        report = self.analyzer.generate_stability_report(
            baseline_data, current_data, output_dir=temp_directory
        )
        
        assert isinstance(report, dict)
        assert 'feature_stability' in report
        assert 'report_summary' in report
        assert 'plots_generated' in report
        
        # Check that plotting functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()


class TestEvaluationIntegration:
    """Integration tests for evaluation components."""
    
    def test_complete_evaluation_workflow(self, sample_dataframe, temp_directory):
        """Test complete evaluation workflow."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics_calc = MetricsCalculator(task_type='classification')
        metrics = metrics_calc.calculate_classification_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Optimize threshold
        optimizer = ThresholdOptimizer()
        best_threshold, best_f1 = optimizer.optimize_threshold(
            y_test, y_pred_proba, metric='f1_score'
        )
        
        # Generate comprehensive report
        report = metrics_calc.generate_comprehensive_report(
            y_test, y_pred, y_pred_proba, output_dir=temp_directory
        )
        
        # Verify results
        assert 'accuracy' in metrics
        assert 0 <= best_threshold <= 1
        assert 0 <= best_f1 <= 1
        assert 'metrics' in report
        assert 'plots' in report
    
    def test_model_comparison_workflow(self, sample_dataframe):
        """Test model comparison workflow."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train multiple models
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        model_predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            model_predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        # Compare models
        comparator = ModelComparator()
        comparison_results = comparator.compare_models(y_test, model_predictions)
        
        # Cross-validation comparison
        cv_results = comparator.cross_validation_comparison(
            models, X_train, y_train, cv_folds=3, scoring='accuracy'
        )
        
        # Verify results
        assert 'comparison_table' in comparison_results
        assert 'statistical_tests' in comparison_results
        assert 'mean_scores' in cv_results
        assert len(comparison_results['comparison_table']) == 2
    
    def test_stability_monitoring_workflow(self, sample_dataframe):
        """Test stability monitoring workflow."""
        # Simulate temporal data split
        train_data = sample_dataframe.iloc[:600]  # Training period
        baseline_data = sample_dataframe.iloc[600:800]  # Baseline period
        current_data = sample_dataframe.iloc[800:]  # Current period
        
        # Analyze feature stability
        analyzer = StabilityAnalyzer()
        feature_stability = analyzer.feature_stability_analysis(
            baseline_data, current_data
        )
        
        # Analyze model score stability (simulate scores)
        np.random.seed(42)
        baseline_scores = np.random.uniform(0, 1, len(baseline_data))
        current_scores = np.random.uniform(0, 1, len(current_data))
        
        score_stability = analyzer.model_score_stability(
            baseline_scores, current_scores
        )
        
        # Verify results
        assert 'feature_psi' in feature_stability
        assert 'overall_stability' in feature_stability
        assert 'csi' in score_stability
        assert 'stability_status' in score_stability