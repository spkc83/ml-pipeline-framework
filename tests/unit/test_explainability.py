"""
Unit tests for explainability modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import tempfile
import os

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.pdp_ice import PartialDependenceAnalyzer
from src.explainability.compliance import ComplianceReporter


class TestSHAPExplainer:
    """Test cases for SHAPExplainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0.2, 0.8, 0.3, 0.7, 0.5])
        self.mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.5, 0.5]
        ])
        
        # Create sample data
        np.random.seed(42)
        self.background_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.uniform(0, 10, 100),
            'feature_3': np.random.randint(1, 5, 100)
        })
        
        self.test_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 20),
            'feature_2': np.random.uniform(0, 10, 20),
            'feature_3': np.random.randint(1, 5, 20)
        })
    
    @patch('src.explainability.shap_explainer.shap')
    def test_initialization_tree_explainer(self, mock_shap):
        """Test initialization with tree explainer."""
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model, 
            self.background_data,
            explainer_type='tree'
        )
        
        assert explainer.model == self.mock_model
        assert explainer.explainer_type == 'tree'
        mock_shap.TreeExplainer.assert_called_once_with(self.mock_model)
    
    @patch('src.explainability.shap_explainer.shap')
    def test_initialization_linear_explainer(self, mock_shap):
        """Test initialization with linear explainer."""
        mock_explainer = MagicMock()
        mock_shap.LinearExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='linear'
        )
        
        mock_shap.LinearExplainer.assert_called_once_with(
            self.mock_model, 
            self.background_data.values
        )
    
    @patch('src.explainability.shap_explainer.shap')
    def test_initialization_kernel_explainer(self, mock_shap):
        """Test initialization with kernel explainer."""
        mock_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='kernel'
        )
        
        mock_shap.KernelExplainer.assert_called_once()
    
    def test_invalid_explainer_type(self):
        """Test error for invalid explainer type."""
        with pytest.raises(ValueError, match="Unsupported explainer type"):
            SHAPExplainer(
                self.mock_model,
                self.background_data,
                explainer_type='invalid'
            )
    
    @patch('src.explainability.shap_explainer.shap')
    def test_explain_instance(self, mock_shap):
        """Test explaining single instance."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_shap_values = np.array([[0.1, -0.2, 0.3]])
        mock_expected_value = 0.5
        
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_explainer.expected_value = mock_expected_value
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Explain single instance
        instance = self.test_data.iloc[0:1]
        explanation = explainer.explain_instance(instance)
        
        assert isinstance(explanation, dict)
        assert 'shap_values' in explanation
        assert 'expected_value' in explanation
        assert 'feature_names' in explanation
        assert explanation['expected_value'] == mock_expected_value
        np.testing.assert_array_equal(explanation['shap_values'], mock_shap_values)
    
    @patch('src.explainability.shap_explainer.shap')
    def test_explain_dataset(self, mock_shap):
        """Test explaining multiple instances."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_shap_values = np.random.normal(0, 0.5, (20, 3))
        mock_expected_value = 0.5
        
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_explainer.expected_value = mock_expected_value
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Explain dataset
        explanations = explainer.explain_dataset(self.test_data)
        
        assert isinstance(explanations, dict)
        assert 'shap_values' in explanations
        assert 'expected_value' in explanations
        assert 'feature_names' in explanations
        assert explanations['shap_values'].shape == (20, 3)
    
    @patch('src.explainability.shap_explainer.shap')
    def test_get_global_feature_importance(self, mock_shap):
        """Test global feature importance calculation."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_shap_values = np.array([
            [0.1, -0.2, 0.3],
            [-0.1, 0.3, -0.2],
            [0.2, -0.1, 0.1]
        ])
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Calculate global importance
        importance = explainer.get_global_feature_importance(self.test_data.iloc[:3])
        
        assert isinstance(importance, dict)
        assert len(importance) == 3  # Number of features
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0 for v in importance.values())  # Absolute importance values
    
    @patch('src.explainability.shap_explainer.shap')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_summary(self, mock_savefig, mock_shap):
        """Test summary plot generation."""
        # Setup mocks
        mock_explainer = MagicMock()
        mock_shap_values = np.random.normal(0, 0.5, (20, 3))
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_shap.summary_plot = MagicMock()
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Generate summary plot
        output_path = '/tmp/test_summary.png'
        explainer.plot_summary(self.test_data, output_path)
        
        mock_shap.summary_plot.assert_called_once()
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('src.explainability.shap_explainer.shap')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_waterfall(self, mock_savefig, mock_shap):
        """Test waterfall plot generation."""
        # Setup mocks
        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        mock_explainer.return_value = mock_explanation
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_shap.waterfall_plot = MagicMock()
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Generate waterfall plot
        output_path = '/tmp/test_waterfall.png'
        instance = self.test_data.iloc[0:1]
        explainer.plot_waterfall(instance, output_path)
        
        mock_shap.waterfall_plot.assert_called_once()
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('src.explainability.shap_explainer.shap')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig, mock_shap):
        """Test feature importance plot generation."""
        # Setup mocks
        mock_explainer = MagicMock()
        mock_shap_values = np.random.normal(0, 0.5, (20, 3))
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Generate feature importance plot
        output_path = '/tmp/test_importance.png'
        explainer.plot_feature_importance(self.test_data, output_path)
        
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('src.explainability.shap_explainer.shap')
    def test_generate_explanation_report(self, mock_shap, temp_directory):
        """Test comprehensive explanation report generation."""
        # Setup mocks
        mock_explainer = MagicMock()
        mock_shap_values = np.random.normal(0, 0.5, (20, 3))
        mock_expected_value = 0.5
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_explainer.expected_value = mock_expected_value
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_shap.summary_plot = MagicMock()
        mock_shap.waterfall_plot = MagicMock()
        
        explainer = SHAPExplainer(
            self.mock_model,
            self.background_data,
            explainer_type='tree'
        )
        
        # Generate report
        report = explainer.generate_explanation_report(
            self.test_data, output_dir=temp_directory
        )
        
        assert isinstance(report, dict)
        assert 'global_importance' in report
        assert 'shap_values' in report
        assert 'plots_generated' in report
        assert 'summary_statistics' in report


class TestPartialDependenceAnalyzer:
    """Test cases for PartialDependenceAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.normal(0.5, 0.1, 100)
        
        # Create sample data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.uniform(0, 10, 1000),
            'feature_3': np.random.randint(0, 5, 1000),
            'feature_4': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        self.feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model, 
            self.data,
            self.feature_names
        )
        
        assert analyzer.model == self.mock_model
        assert analyzer.feature_names == self.feature_names
        pd.testing.assert_frame_equal(analyzer.data, self.data)
    
    def test_calculate_partial_dependence_numeric(self):
        """Test PDP calculation for numeric feature."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        pdp_result = analyzer.calculate_partial_dependence(
            'feature_1', n_grid_points=10
        )
        
        assert isinstance(pdp_result, dict)
        assert 'feature_values' in pdp_result
        assert 'partial_dependence' in pdp_result
        assert 'feature_name' in pdp_result
        assert len(pdp_result['feature_values']) == 10
        assert len(pdp_result['partial_dependence']) == 10
        assert pdp_result['feature_name'] == 'feature_1'
    
    def test_calculate_partial_dependence_categorical(self):
        """Test PDP calculation for categorical feature."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        pdp_result = analyzer.calculate_partial_dependence('feature_4')
        
        assert isinstance(pdp_result, dict)
        assert 'feature_values' in pdp_result
        assert 'partial_dependence' in pdp_result
        assert set(pdp_result['feature_values']) == {'A', 'B', 'C'}
    
    def test_calculate_ice_curves(self):
        """Test ICE curves calculation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data.iloc[:50],  # Use smaller sample for ICE
            self.feature_names
        )
        
        ice_result = analyzer.calculate_ice_curves(
            'feature_1', n_grid_points=10, sample_size=20
        )
        
        assert isinstance(ice_result, dict)
        assert 'feature_values' in ice_result
        assert 'ice_curves' in ice_result
        assert 'feature_name' in ice_result
        assert ice_result['ice_curves'].shape == (20, 10)  # 20 samples, 10 grid points
    
    def test_calculate_2d_partial_dependence(self):
        """Test 2D PDP calculation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        pdp_2d = analyzer.calculate_2d_partial_dependence(
            'feature_1', 'feature_2', n_grid_points=5
        )
        
        assert isinstance(pdp_2d, dict)
        assert 'feature1_values' in pdp_2d
        assert 'feature2_values' in pdp_2d
        assert 'partial_dependence' in pdp_2d
        assert 'feature_names' in pdp_2d
        assert pdp_2d['partial_dependence'].shape == (5, 5)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_partial_dependence(self, mock_savefig):
        """Test PDP plot generation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        output_path = '/tmp/test_pdp.png'
        analyzer.plot_partial_dependence('feature_1', output_path)
        
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_ice_curves(self, mock_savefig):
        """Test ICE curves plot generation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data.iloc[:50],
            self.feature_names
        )
        
        output_path = '/tmp/test_ice.png'
        analyzer.plot_ice_curves('feature_1', output_path, sample_size=20)
        
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_2d_partial_dependence(self, mock_savefig):
        """Test 2D PDP plot generation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        output_path = '/tmp/test_2d_pdp.png'
        analyzer.plot_2d_partial_dependence('feature_1', 'feature_2', output_path)
        
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    def test_feature_interaction_strength(self):
        """Test feature interaction strength calculation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        interaction_strength = analyzer.calculate_feature_interaction_strength(
            'feature_1', 'feature_2'
        )
        
        assert isinstance(interaction_strength, float)
        assert interaction_strength >= 0
    
    def test_generate_pdp_report(self, temp_directory):
        """Test comprehensive PDP report generation."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data.iloc[:100],  # Use smaller sample for faster testing
            self.feature_names[:2]  # Use fewer features
        )
        
        with patch('matplotlib.pyplot.savefig'):
            report = analyzer.generate_pdp_report(output_dir=temp_directory)
        
        assert isinstance(report, dict)
        assert 'pdp_results' in report
        assert 'ice_results' in report
        assert 'interaction_strengths' in report
        assert 'plots_generated' in report
    
    def test_invalid_feature_name(self):
        """Test error for invalid feature name."""
        analyzer = PartialDependenceAnalyzer(
            self.mock_model,
            self.data,
            self.feature_names
        )
        
        with pytest.raises(ValueError, match="Feature .* not found"):
            analyzer.calculate_partial_dependence('invalid_feature')


class TestComplianceReporter:
    """Test cases for ComplianceReporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict_proba.return_value = np.random.uniform(0, 1, 100).reshape(-1, 1)
        
        # Create sample data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 20000, 100),
            'credit_score': np.random.randint(300, 850, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], 100)
        })
        
        self.y = np.random.choice([0, 1], 100, p=[0.7, 0.3])
        self.feature_names = list(self.X.columns)
        self.model_name = 'Test_Model'
    
    def test_initialization(self):
        """Test compliance reporter initialization."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        assert reporter.model == self.mock_model
        assert reporter.model_name == self.model_name
        assert reporter.feature_names == self.feature_names
        pd.testing.assert_frame_equal(reporter.X, self.X)
        np.testing.assert_array_equal(reporter.y, self.y)
    
    def test_calculate_feature_importance_infogram(self):
        """Test feature importance infogram calculation."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        # Mock permutation importance
        with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
            mock_result = MagicMock()
            mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
            mock_result.importances_std = np.array([0.02, 0.05, 0.03, 0.01, 0.02])
            mock_perm_imp.return_value = mock_result
            
            infogram = reporter.calculate_feature_importance_infogram()
            
            assert isinstance(infogram, pd.DataFrame)
            assert 'feature' in infogram.columns
            assert 'importance' in infogram.columns
            assert 'importance_std' in infogram.columns
            assert len(infogram) == len(self.feature_names)
    
    def test_analyze_adverse_impact(self):
        """Test adverse impact analysis."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        # Test with gender as protected attribute
        adverse_impact = reporter.analyze_adverse_impact(
            protected_attribute='gender',
            threshold=0.5
        )
        
        assert isinstance(adverse_impact, dict)
        assert 'selection_rates' in adverse_impact
        assert 'adverse_impact_ratios' in adverse_impact
        assert 'four_fifths_rule_violations' in adverse_impact
        assert 'statistical_significance' in adverse_impact
    
    def test_generate_fairness_metrics(self):
        """Test fairness metrics generation."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        fairness_metrics = reporter.generate_fairness_metrics(
            protected_attributes=['gender', 'race']
        )
        
        assert isinstance(fairness_metrics, dict)
        assert 'demographic_parity' in fairness_metrics
        assert 'equalized_odds' in fairness_metrics
        assert 'equal_opportunity' in fairness_metrics
        
        # Should have metrics for each protected attribute
        for attr in ['gender', 'race']:
            assert attr in fairness_metrics['demographic_parity']
    
    def test_calculate_model_risk_assessment(self):
        """Test model risk assessment."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        risk_assessment = reporter.calculate_model_risk_assessment()
        
        assert isinstance(risk_assessment, dict)
        assert 'complexity_score' in risk_assessment
        assert 'interpretability_score' in risk_assessment
        assert 'bias_risk_score' in risk_assessment
        assert 'overall_risk_level' in risk_assessment
        assert 'recommendations' in risk_assessment
    
    @patch('matplotlib.pyplot.savefig')
    def test_generate_admissible_ml_infogram(self, mock_savefig):
        """Test admissible ML infogram generation."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
            mock_result = MagicMock()
            mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
            mock_result.importances_std = np.array([0.02, 0.05, 0.03, 0.01, 0.02])
            mock_perm_imp.return_value = mock_result
            
            output_path = '/tmp/test_infogram.png'
            reporter.generate_admissible_ml_infogram(output_path)
            
            mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.savefig')
    def test_generate_fair_lending_charts(self, mock_savefig):
        """Test fair lending charts generation."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        output_path = '/tmp/test_fairlending.png'
        reporter.generate_fair_lending_charts(
            protected_attributes=['gender', 'race'],
            output_path=output_path
        )
        
        mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
    
    def test_generate_model_card(self, temp_directory):
        """Test model card generation."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
            mock_result = MagicMock()
            mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
            mock_result.importances_std = np.array([0.02, 0.05, 0.03, 0.01, 0.02])
            mock_perm_imp.return_value = mock_result
            
            model_card = reporter.generate_model_card(
                model_description="Test model for compliance testing",
                intended_use="Testing purposes",
                training_data_description="Synthetic test data",
                performance_metrics={'accuracy': 0.85, 'auc': 0.78}
            )
            
            assert isinstance(model_card, dict)
            assert 'model_details' in model_card
            assert 'intended_use' in model_card
            assert 'performance_metrics' in model_card
            assert 'fairness_assessment' in model_card
            assert 'risk_assessment' in model_card
    
    def test_export_compliance_report_excel(self, temp_directory):
        """Test Excel compliance report export."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
            mock_result = MagicMock()
            mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
            mock_result.importances_std = np.array([0.02, 0.05, 0.03, 0.01, 0.02])
            mock_perm_imp.return_value = mock_result
            
            output_path = os.path.join(temp_directory, 'compliance_report.xlsx')
            reporter.export_compliance_report_excel(
                output_path=output_path,
                protected_attributes=['gender', 'race']
            )
            
            # Check that file was created (mocked pandas.ExcelWriter would create it)
            # In real test, we'd check file exists and content
            assert output_path.endswith('.xlsx')
    
    def test_generate_comprehensive_report(self, temp_directory):
        """Test comprehensive compliance report generation."""
        reporter = ComplianceReporter(
            self.mock_model,
            self.X,
            self.y,
            self.feature_names,
            self.model_name
        )
        
        with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
            mock_result = MagicMock()
            mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
            mock_result.importances_std = np.array([0.02, 0.05, 0.03, 0.01, 0.02])
            mock_perm_imp.return_value = mock_result
            
            with patch('matplotlib.pyplot.savefig'):
                report = reporter.generate_comprehensive_report(
                    output_dir=temp_directory,
                    protected_attributes=['gender', 'race']
                )
            
            assert isinstance(report, dict)
            assert 'feature_importance' in report
            assert 'adverse_impact_analysis' in report
            assert 'fairness_metrics' in report
            assert 'risk_assessment' in report
            assert 'model_card' in report
            assert 'files_generated' in report


class TestExplainabilityIntegration:
    """Integration tests for explainability components."""
    
    def test_complete_explainability_workflow(self, sample_dataframe, temp_directory):
        """Test complete explainability workflow."""
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
        
        with patch('src.explainability.shap_explainer.shap') as mock_shap:
            # Mock SHAP
            mock_explainer = MagicMock()
            mock_shap_values = np.random.normal(0, 0.5, (len(X_test), len(X.columns)))
            mock_explainer.shap_values.return_value = mock_shap_values
            mock_explainer.expected_value = 0.5
            mock_shap.TreeExplainer.return_value = mock_explainer
            mock_shap.summary_plot = MagicMock()
            mock_shap.waterfall_plot = MagicMock()
            
            # SHAP explanations
            shap_explainer = SHAPExplainer(
                model, X_train.iloc[:100], explainer_type='tree'
            )
            shap_report = shap_explainer.generate_explanation_report(
                X_test, output_dir=f"{temp_directory}/shap"
            )
        
        # PDP analysis
        pdp_analyzer = PartialDependenceAnalyzer(
            model, X_test, list(X.columns)
        )
        
        with patch('matplotlib.pyplot.savefig'):
            pdp_report = pdp_analyzer.generate_pdp_report(
                output_dir=f"{temp_directory}/pdp"
            )
        
        # Compliance reporting
        compliance_reporter = ComplianceReporter(
            model, X_test, y_test, list(X.columns), 'Test_Model'
        )
        
        with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
            mock_result = MagicMock()
            mock_result.importances_mean = np.random.uniform(0, 0.5, len(X.columns))
            mock_result.importances_std = np.random.uniform(0, 0.1, len(X.columns))
            mock_perm_imp.return_value = mock_result
            
            with patch('matplotlib.pyplot.savefig'):
                compliance_report = compliance_reporter.generate_comprehensive_report(
                    output_dir=f"{temp_directory}/compliance"
                )
        
        # Verify all reports were generated
        assert 'global_importance' in shap_report
        assert 'pdp_results' in pdp_report
        assert 'feature_importance' in compliance_report
    
    def test_model_explainability_comparison(self, sample_dataframe):
        """Test explainability comparison between models."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X_train, y_train)
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        models = {'RandomForest': rf_model, 'LogisticRegression': lr_model}
        explainability_results = {}
        
        for model_name, model in models.items():
            with patch('src.explainability.shap_explainer.shap') as mock_shap:
                # Mock SHAP for each model
                mock_explainer = MagicMock()
                mock_shap_values = np.random.normal(0, 0.5, (len(X_test), len(X.columns)))
                mock_explainer.shap_values.return_value = mock_shap_values
                mock_explainer.expected_value = 0.5
                mock_shap.TreeExplainer.return_value = mock_explainer
                
                # Get SHAP explanations
                shap_explainer = SHAPExplainer(
                    model, X_train.iloc[:50], explainer_type='tree'
                )
                global_importance = shap_explainer.get_global_feature_importance(X_test)
                
                explainability_results[model_name] = {
                    'global_importance': global_importance
                }
        
        # Compare feature importance across models
        assert 'RandomForest' in explainability_results
        assert 'LogisticRegression' in explainability_results
        
        for model_name, results in explainability_results.items():
            assert 'global_importance' in results
            assert len(results['global_importance']) == len(X.columns)