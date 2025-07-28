"""
Unit tests for model modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.models.base import BaseModel
from src.models.factory import ModelFactory
from src.models.sklearn_model import SklearnModel
from src.models.h2o_model import H2OModel
from src.models.sparkml_model import SparkMLModel
from src.models.statsmodels_model import StatsModelsModel
from src.models.cost_sensitive import CostSensitiveLearning
from src.models.tuning import HyperparameterTuner


class TestBaseModel:
    """Test cases for BaseModel abstract class."""
    
    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        model = BaseModel()
        
        with pytest.raises(NotImplementedError):
            model.train(None, None)
        
        with pytest.raises(NotImplementedError):
            model.predict(None)
        
        with pytest.raises(NotImplementedError):
            model.predict_proba(None)
        
        with pytest.raises(NotImplementedError):
            model.evaluate(None, None)


class TestModelFactory:
    """Test cases for ModelFactory."""
    
    @patch('src.models.sklearn_model.SklearnModel')
    def test_create_sklearn_model(self, mock_sklearn_model):
        """Test creating sklearn model."""
        mock_instance = MagicMock()
        mock_sklearn_model.return_value = mock_instance
        
        model = ModelFactory.create_model(
            algorithm='RandomForestClassifier',
            framework='sklearn',
            n_estimators=100
        )
        
        mock_sklearn_model.assert_called_once()
        assert model == mock_instance
    
    @patch('src.models.h2o_model.H2OModel')
    def test_create_h2o_model(self, mock_h2o_model):
        """Test creating H2O model."""
        mock_instance = MagicMock()
        mock_h2o_model.return_value = mock_instance
        
        model = ModelFactory.create_model(
            algorithm='H2ORandomForestEstimator',
            framework='h2o',
            ntrees=100
        )
        
        mock_h2o_model.assert_called_once()
        assert model == mock_instance
    
    @patch('src.models.sparkml_model.SparkMLModel')
    def test_create_sparkml_model(self, mock_sparkml_model):
        """Test creating SparkML model."""
        mock_instance = MagicMock()
        mock_sparkml_model.return_value = mock_instance
        
        model = ModelFactory.create_model(
            algorithm='RandomForestClassifier',
            framework='sparkml',
            numTrees=100
        )
        
        mock_sparkml_model.assert_called_once()
        assert model == mock_instance
    
    @patch('src.models.statsmodels_model.StatsModelsModel')
    def test_create_statsmodels_model(self, mock_statsmodels_model):
        """Test creating StatsModels model."""
        mock_instance = MagicMock()
        mock_statsmodels_model.return_value = mock_instance
        
        model = ModelFactory.create_model(
            algorithm='OLS',
            framework='statsmodels'
        )
        
        mock_statsmodels_model.assert_called_once()
        assert model == mock_instance
    
    def test_unsupported_framework(self):
        """Test error for unsupported framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            ModelFactory.create_model(
                algorithm='RandomForestClassifier',
                framework='unsupported'
            )
    
    def test_get_available_algorithms_sklearn(self):
        """Test getting available algorithms for sklearn."""
        algorithms = ModelFactory.get_available_algorithms('sklearn')
        
        assert isinstance(algorithms, list)
        assert 'RandomForestClassifier' in algorithms
        assert 'LogisticRegression' in algorithms
        assert 'SVC' in algorithms
    
    def test_get_available_algorithms_invalid_framework(self):
        """Test error for invalid framework in get_available_algorithms."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            ModelFactory.get_available_algorithms('invalid')


class TestSklearnModel:
    """Test cases for SklearnModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sklearn_model = SklearnModel(
            algorithm='RandomForestClassifier',
            n_estimators=10,  # Small for testing
            random_state=42
        )
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.sklearn_model.algorithm == 'RandomForestClassifier'
        assert self.sklearn_model.model is not None
        assert isinstance(self.sklearn_model.model, RandomForestClassifier)
    
    def test_train_classification(self, sample_dataframe):
        """Test training classification model."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        trained_model = self.sklearn_model.train(X, y)
        
        assert trained_model is not None
        assert hasattr(self.sklearn_model.model, 'predict')
        assert self.sklearn_model.is_fitted
    
    def test_predict_classification(self, sample_dataframe):
        """Test prediction for classification."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Train first
        self.sklearn_model.train(X, y)
        
        # Predict
        predictions = self.sklearn_model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)  # Binary classification
    
    def test_predict_proba_classification(self, sample_dataframe):
        """Test probability prediction for classification."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Train first
        self.sklearn_model.train(X, y)
        
        # Predict probabilities
        probabilities = self.sklearn_model.predict_proba(X)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate_classification(self, sample_dataframe):
        """Test model evaluation for classification."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Train first
        self.sklearn_model.train(X, y)
        
        # Evaluate
        metrics = self.sklearn_model.evaluate(X, y)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_regression_model(self, sample_regression_dataframe):
        """Test regression model."""
        regression_model = SklearnModel(
            algorithm='RandomForestRegressor',
            n_estimators=10,
            random_state=42
        )
        
        X = sample_regression_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_regression_dataframe['target']
        
        # Train
        regression_model.train(X, y)
        
        # Predict
        predictions = regression_model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()
    
    def test_get_feature_importance(self, sample_dataframe):
        """Test getting feature importance."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Train first
        self.sklearn_model.train(X, y)
        
        # Get feature importance
        importance = self.sklearn_model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_predict_without_training_raises_error(self, sample_dataframe):
        """Test that prediction without training raises error."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            self.sklearn_model.predict(X)
    
    def test_set_params(self):
        """Test setting model parameters."""
        new_params = {'n_estimators': 50, 'max_depth': 10}
        self.sklearn_model.set_params(**new_params)
        
        assert self.sklearn_model.model.n_estimators == 50
        assert self.sklearn_model.model.max_depth == 10
    
    def test_get_params(self):
        """Test getting model parameters."""
        params = self.sklearn_model.get_params()
        
        assert isinstance(params, dict)
        assert 'n_estimators' in params
        assert 'random_state' in params


class TestH2OModel:
    """Test cases for H2OModel."""
    
    @patch('src.models.h2o_model.h2o')
    def test_initialization(self, mock_h2o):
        """Test H2O model initialization."""
        mock_h2o.init.return_value = None
        mock_h2o.H2ORandomForestEstimator.return_value = MagicMock()
        
        h2o_model = H2OModel(
            algorithm='H2ORandomForestEstimator',
            ntrees=10
        )
        
        assert h2o_model.algorithm == 'H2ORandomForestEstimator'
        mock_h2o.init.assert_called_once()
    
    @patch('src.models.h2o_model.h2o')
    def test_train_with_h2o_frame(self, mock_h2o):
        """Test training with H2O frame."""
        # Mock H2O components
        mock_frame = MagicMock()
        mock_model = MagicMock()
        mock_h2o.H2OFrame.return_value = mock_frame
        mock_h2o.H2ORandomForestEstimator.return_value = mock_model
        
        h2o_model = H2OModel(algorithm='H2ORandomForestEstimator')
        
        # Mock pandas DataFrame
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        X = mock_df.drop('target', axis=1)
        y = mock_df['target']
        
        # Train
        result = h2o_model.train(X, y)
        
        # Verify H2O frame creation and training
        mock_h2o.H2OFrame.assert_called()
        mock_model.train.assert_called_once()
        assert h2o_model.is_fitted
    
    @patch('src.models.h2o_model.h2o')
    def test_predict_with_h2o_frame(self, mock_h2o):
        """Test prediction with H2O frame."""
        # Setup mocks
        mock_frame = MagicMock()
        mock_model = MagicMock()
        mock_predictions = MagicMock()
        mock_predictions.as_data_frame.return_value = pd.DataFrame({'predict': [0, 1, 0]})
        
        mock_h2o.H2OFrame.return_value = mock_frame
        mock_h2o.H2ORandomForestEstimator.return_value = mock_model
        mock_model.predict.return_value = mock_predictions
        
        h2o_model = H2OModel(algorithm='H2ORandomForestEstimator')
        h2o_model.model = mock_model
        h2o_model.is_fitted = True
        
        # Mock test data
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Predict
        predictions = h2o_model.predict(test_data)
        
        # Verify
        mock_h2o.H2OFrame.assert_called_with(test_data)
        mock_model.predict.assert_called_once_with(mock_frame)
        assert isinstance(predictions, np.ndarray)


class TestSparkMLModel:
    """Test cases for SparkMLModel."""
    
    @patch('src.models.sparkml_model.SparkSession')
    def test_initialization(self, mock_spark_session):
        """Test SparkML model initialization."""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.getOrCreate.return_value = mock_spark
        
        sparkml_model = SparkMLModel(
            algorithm='RandomForestClassifier',
            numTrees=10
        )
        
        assert sparkml_model.algorithm == 'RandomForestClassifier'
        assert sparkml_model.spark_session == mock_spark
    
    @patch('src.models.sparkml_model.SparkSession')
    @patch('src.models.sparkml_model.RandomForestClassifier')
    @patch('src.models.sparkml_model.VectorAssembler')
    def test_train_with_spark_dataframe(self, mock_assembler, mock_rf, mock_spark_session):
        """Test training with Spark DataFrame."""
        # Setup mocks
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.getOrCreate.return_value = mock_spark
        
        mock_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_df
        
        mock_assembler_instance = MagicMock()
        mock_assembler.return_value = mock_assembler_instance
        mock_assembler_instance.transform.return_value = mock_df
        
        mock_rf_instance = MagicMock()
        mock_rf.return_value = mock_rf_instance
        mock_model = MagicMock()
        mock_rf_instance.fit.return_value = mock_model
        
        sparkml_model = SparkMLModel(algorithm='RandomForestClassifier')
        
        # Mock pandas DataFrame
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        X = test_data.drop('target', axis=1)
        y = test_data['target']
        
        # Train
        result = sparkml_model.train(X, y)
        
        # Verify
        mock_spark.createDataFrame.assert_called()
        mock_rf_instance.fit.assert_called_once()
        assert sparkml_model.is_fitted


class TestStatsModelsModel:
    """Test cases for StatsModelsModel."""
    
    def test_initialization(self):
        """Test StatsModels initialization."""
        statsmodels_model = StatsModelsModel(algorithm='OLS')
        
        assert statsmodels_model.algorithm == 'OLS'
        assert not statsmodels_model.is_fitted
    
    @patch('src.models.statsmodels_model.sm.OLS')
    def test_train_ols(self, mock_ols):
        """Test training OLS model."""
        # Setup mock
        mock_model = MagicMock()
        mock_fitted_model = MagicMock()
        mock_model.fit.return_value = mock_fitted_model
        mock_ols.return_value = mock_model
        
        statsmodels_model = StatsModelsModel(algorithm='OLS')
        
        # Mock data
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([1, 2, 3])
        
        # Train
        result = statsmodels_model.train(X, y)
        
        # Verify
        mock_ols.assert_called_once()
        mock_model.fit.assert_called_once()
        assert statsmodels_model.is_fitted
    
    @patch('src.models.statsmodels_model.sm.Logit')
    def test_train_logit(self, mock_logit):
        """Test training Logit model."""
        # Setup mock
        mock_model = MagicMock()
        mock_fitted_model = MagicMock()
        mock_model.fit.return_value = mock_fitted_model
        mock_logit.return_value = mock_model
        
        statsmodels_model = StatsModelsModel(algorithm='Logit')
        
        # Mock data
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        # Train
        result = statsmodels_model.train(X, y)
        
        # Verify
        mock_logit.assert_called_once()
        assert statsmodels_model.is_fitted


class TestCostSensitiveLearning:
    """Test cases for CostSensitiveLearning."""
    
    def test_initialization(self):
        """Test initialization."""
        cost_learning = CostSensitiveLearning()
        assert cost_learning is not None
    
    def test_calculate_class_weights(self, sample_imbalanced_dataframe):
        """Test class weight calculation."""
        cost_learning = CostSensitiveLearning()
        y = sample_imbalanced_dataframe['target']
        
        weights = cost_learning.calculate_class_weights(y, method='balanced')
        
        assert isinstance(weights, dict)
        assert 0 in weights
        assert 1 in weights
        # Minority class should have higher weight
        assert weights[1] > weights[0]
    
    def test_calculate_sample_weights(self, sample_imbalanced_dataframe):
        """Test sample weight calculation."""
        cost_learning = CostSensitiveLearning()
        y = sample_imbalanced_dataframe['target']
        
        weights = cost_learning.calculate_sample_weights(y)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(y)
        assert all(w > 0 for w in weights)
    
    def test_create_cost_matrix(self):
        """Test cost matrix creation."""
        cost_learning = CostSensitiveLearning()
        
        # Binary classification
        cost_matrix = cost_learning.create_cost_matrix(
            n_classes=2,
            false_positive_cost=1,
            false_negative_cost=5
        )
        
        assert cost_matrix.shape == (2, 2)
        assert cost_matrix[0, 0] == 0  # True negative cost
        assert cost_matrix[0, 1] == 1  # False positive cost
        assert cost_matrix[1, 0] == 5  # False negative cost
        assert cost_matrix[1, 1] == 0  # True positive cost
    
    def test_auto_configure_sklearn(self, sample_imbalanced_dataframe):
        """Test auto-configuration for sklearn."""
        cost_learning = CostSensitiveLearning()
        y = sample_imbalanced_dataframe['target']
        
        params = cost_learning.auto_configure(framework='sklearn', y=y)
        
        assert isinstance(params, dict)
        assert 'class_weight' in params
    
    def test_auto_configure_xgboost(self, sample_imbalanced_dataframe):
        """Test auto-configuration for XGBoost."""
        cost_learning = CostSensitiveLearning()
        y = sample_imbalanced_dataframe['target']
        
        params = cost_learning.auto_configure(framework='xgboost', y=y)
        
        assert isinstance(params, dict)
        assert 'scale_pos_weight' in params
    
    def test_auto_configure_lightgbm(self, sample_imbalanced_dataframe):
        """Test auto-configuration for LightGBM."""
        cost_learning = CostSensitiveLearning()
        y = sample_imbalanced_dataframe['target']
        
        params = cost_learning.auto_configure(framework='lightgbm', y=y)
        
        assert isinstance(params, dict)
        assert 'class_weight' in params
    
    def test_auto_configure_catboost(self, sample_imbalanced_dataframe):
        """Test auto-configuration for CatBoost."""
        cost_learning = CostSensitiveLearning()
        y = sample_imbalanced_dataframe['target']
        
        params = cost_learning.auto_configure(framework='catboost', y=y)
        
        assert isinstance(params, dict)
        assert 'class_weights' in params


class TestHyperparameterTuner:
    """Test cases for HyperparameterTuner."""
    
    def test_initialization(self):
        """Test tuner initialization."""
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [10, 20], 'max_depth': [3, 5]},
            cv_folds=3
        )
        
        assert tuner.model_class == RandomForestClassifier
        assert tuner.cv_folds == 3
        assert 'n_estimators' in tuner.param_space
    
    def test_grid_search_tuning(self, sample_dataframe):
        """Test grid search tuning."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [5, 10], 'max_depth': [3, 5]},
            cv_folds=3,
            scoring='accuracy'
        )
        
        best_params, best_score, results = tuner.tune(X, y, method='grid_search')
        
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert isinstance(results, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 0 <= best_score <= 1
    
    def test_random_search_tuning(self, sample_dataframe):
        """Test random search tuning."""
        from scipy.stats import randint
        
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': randint(5, 20),
                'max_depth': randint(3, 10)
            },
            cv_folds=3,
            scoring='accuracy'
        )
        
        best_params, best_score, results = tuner.tune(
            X, y, 
            method='random_search',
            n_iter=5
        )
        
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
    
    def test_bayesian_optimization_tuning(self, sample_dataframe):
        """Test Bayesian optimization tuning."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': (5, 20), 'max_depth': (3, 10)},
            cv_folds=3,
            scoring='accuracy'
        )
        
        # This test assumes skopt is available
        try:
            best_params, best_score, results = tuner.tune(
                X, y,
                method='bayesian_optimization',
                n_calls=5
            )
            
            assert isinstance(best_params, dict)
            assert isinstance(best_score, float)
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params
            
        except ImportError:
            pytest.skip("scikit-optimize not available for Bayesian optimization")
    
    def test_invalid_tuning_method(self, sample_dataframe):
        """Test error for invalid tuning method."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [10, 20]},
            cv_folds=3
        )
        
        with pytest.raises(ValueError, match="Unsupported tuning method"):
            tuner.tune(X, y, method='invalid_method')
    
    def test_custom_scoring_function(self, sample_dataframe):
        """Test custom scoring function."""
        from sklearn.metrics import make_scorer, f1_score
        
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        custom_scorer = make_scorer(f1_score, average='weighted')
        
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [5, 10]},
            cv_folds=3,
            scoring=custom_scorer
        )
        
        best_params, best_score, results = tuner.tune(X, y, method='grid_search')
        
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_complete_model_workflow(self, sample_dataframe):
        """Test complete model workflow."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Create model
        model = ModelFactory.create_model(
            algorithm='RandomForestClassifier',
            framework='sklearn',
            n_estimators=10,
            random_state=42
        )
        
        # Train model
        trained_model = model.train(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Evaluate model
        metrics = model.evaluate(X, y)
        
        # Verify results
        assert trained_model is not None
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_model_with_cost_sensitive_learning(self, sample_imbalanced_dataframe):
        """Test model with cost-sensitive learning."""
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        # Configure cost-sensitive learning
        cost_learning = CostSensitiveLearning()
        cost_params = cost_learning.auto_configure(framework='sklearn', y=y)
        
        # Create model with cost-sensitive parameters
        model = ModelFactory.create_model(
            algorithm='RandomForestClassifier',
            framework='sklearn',
            n_estimators=10,
            random_state=42,
            **cost_params
        )
        
        # Train and evaluate
        model.train(X, y)
        metrics = model.evaluate(X, y)
        
        assert 'accuracy' in metrics
        assert model.model.class_weight == 'balanced'
    
    def test_model_with_hyperparameter_tuning(self, sample_dataframe):
        """Test model with hyperparameter tuning."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Set up hyperparameter tuning
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [5, 10], 'max_depth': [3, 5]},
            cv_folds=3,
            scoring='accuracy'
        )
        
        # Tune hyperparameters
        best_params, best_score, results = tuner.tune(X, y, method='grid_search')
        
        # Create model with best parameters
        model = ModelFactory.create_model(
            algorithm='RandomForestClassifier',
            framework='sklearn',
            random_state=42,
            **best_params
        )
        
        # Train final model
        model.train(X, y)
        final_metrics = model.evaluate(X, y)
        
        assert 'accuracy' in final_metrics
        assert final_metrics['accuracy'] >= best_score * 0.9  # Allow some variance
    
    def test_model_serialization(self, sample_dataframe, temp_directory):
        """Test model serialization and deserialization."""
        import pickle
        
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        # Create and train model
        model = ModelFactory.create_model(
            algorithm='RandomForestClassifier',
            framework='sklearn',
            n_estimators=10,
            random_state=42
        )
        model.train(X, y)
        
        # Get predictions before serialization
        predictions_before = model.predict(X)
        
        # Serialize model
        model_path = f"{temp_directory}/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Deserialize model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Get predictions after deserialization
        predictions_after = loaded_model.predict(X)
        
        # Verify predictions are the same
        np.testing.assert_array_equal(predictions_before, predictions_after)
    
    @pytest.mark.parametrize("framework", ['sklearn'])
    def test_different_algorithms(self, framework, sample_dataframe):
        """Test different algorithms for each framework."""
        X = sample_dataframe.drop('target', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['target']
        
        algorithms = {
            'sklearn': ['RandomForestClassifier', 'LogisticRegression']
        }
        
        for algorithm in algorithms[framework]:
            model = ModelFactory.create_model(
                algorithm=algorithm,
                framework=framework,
                random_state=42
            )
            
            # Train model
            model.train(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Verify predictions
            assert len(predictions) == len(X)
            assert all(pred in [0, 1] for pred in predictions)