"""
Unit tests for preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.validator import DataValidator
from src.preprocessing.transformers import (
    MissingValueHandler, CategoricalEncoder, FeatureScaler, OutlierDetector
)
from src.preprocessing.imbalance import (
    SMOTEStrategy, AdasynStrategy, RandomOverSamplingStrategy,
    RandomUnderSamplingStrategy, TomekLinksStrategy, NearMissStrategy,
    BalanceStrategyFactory
)


class TestMissingValueHandler:
    """Test cases for MissingValueHandler."""
    
    def test_init_with_default_strategy(self):
        """Test initialization with default strategy."""
        handler = MissingValueHandler()
        assert handler.strategy == 'mean'
        assert handler.fill_value is None
    
    def test_init_with_custom_strategy(self):
        """Test initialization with custom strategy."""
        handler = MissingValueHandler(strategy='median')
        assert handler.strategy == 'median'
    
    def test_fit_mean_strategy(self, sample_dataframe):
        """Test fitting with mean strategy."""
        handler = MissingValueHandler(strategy='mean')
        
        # Only numeric columns should be processed
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        handler.fit(numeric_data)
        
        assert hasattr(handler, 'imputer_')
        assert isinstance(handler.imputer_, SimpleImputer)
    
    def test_transform_mean_strategy(self, sample_dataframe):
        """Test transformation with mean strategy."""
        handler = MissingValueHandler(strategy='mean')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        
        handler.fit(numeric_data)
        transformed = handler.transform(numeric_data)
        
        # Check that no missing values remain
        assert not transformed.isnull().any().any()
        assert transformed.shape == numeric_data.shape
    
    def test_fit_transform_median_strategy(self, sample_dataframe):
        """Test fit_transform with median strategy."""
        handler = MissingValueHandler(strategy='median')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        
        transformed = handler.fit_transform(numeric_data)
        
        assert not transformed.isnull().any().any()
        assert transformed.shape == numeric_data.shape
    
    def test_constant_strategy(self, sample_dataframe):
        """Test constant fill strategy."""
        handler = MissingValueHandler(strategy='constant', fill_value=0)
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        
        transformed = handler.fit_transform(numeric_data)
        
        assert not transformed.isnull().any().any()
    
    def test_forward_fill_strategy(self, sample_dataframe):
        """Test forward fill strategy."""
        handler = MissingValueHandler(strategy='ffill')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        
        transformed = handler.fit_transform(numeric_data)
        
        # May still have NaN if first values are missing
        assert transformed.shape == numeric_data.shape
    
    def test_invalid_strategy(self):
        """Test error for invalid strategy."""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            MissingValueHandler(strategy='invalid')


class TestCategoricalEncoder:
    """Test cases for CategoricalEncoder."""
    
    def test_init_with_default_strategy(self):
        """Test initialization with default strategy."""
        encoder = CategoricalEncoder()
        assert encoder.strategy == 'onehot'
    
    def test_onehot_encoding(self, sample_dataframe):
        """Test one-hot encoding."""
        encoder = CategoricalEncoder(strategy='onehot')
        categorical_data = sample_dataframe.select_dtypes(include=['object'])
        
        transformed = encoder.fit_transform(categorical_data)
        
        # Should have more columns after one-hot encoding
        assert transformed.shape[1] > categorical_data.shape[1]
        assert transformed.shape[0] == categorical_data.shape[0]
        # All values should be 0 or 1
        assert transformed.isin([0, 1]).all().all()
    
    def test_label_encoding(self, sample_dataframe):
        """Test label encoding."""
        encoder = CategoricalEncoder(strategy='label')
        categorical_data = sample_dataframe.select_dtypes(include=['object'])
        
        transformed = encoder.fit_transform(categorical_data)
        
        # Should have same shape
        assert transformed.shape == categorical_data.shape
        # All values should be numeric
        assert transformed.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    
    def test_target_encoding_with_target(self, sample_dataframe):
        """Test target encoding with target variable."""
        encoder = CategoricalEncoder(strategy='target')
        categorical_data = sample_dataframe.select_dtypes(include=['object'])
        target = sample_dataframe['target']
        
        transformed = encoder.fit_transform(categorical_data, target)
        
        assert transformed.shape == categorical_data.shape
        assert transformed.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    
    def test_target_encoding_without_target(self, sample_dataframe):
        """Test error when target encoding without target."""
        encoder = CategoricalEncoder(strategy='target')
        categorical_data = sample_dataframe.select_dtypes(include=['object'])
        
        with pytest.raises(ValueError, match="Target encoding requires target variable"):
            encoder.fit_transform(categorical_data)
    
    def test_invalid_strategy(self):
        """Test error for invalid strategy."""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            CategoricalEncoder(strategy='invalid')
    
    def test_drop_first_parameter(self, sample_dataframe):
        """Test drop_first parameter in one-hot encoding."""
        encoder = CategoricalEncoder(strategy='onehot', drop_first=True)
        categorical_data = sample_dataframe.select_dtypes(include=['object'])
        
        transformed = encoder.fit_transform(categorical_data)
        
        # Should have fewer columns when dropping first
        encoder_no_drop = CategoricalEncoder(strategy='onehot', drop_first=False)
        transformed_no_drop = encoder_no_drop.fit_transform(categorical_data)
        
        assert transformed.shape[1] < transformed_no_drop.shape[1]


class TestFeatureScaler:
    """Test cases for FeatureScaler."""
    
    def test_init_with_default_method(self):
        """Test initialization with default method."""
        scaler = FeatureScaler()
        assert scaler.method == 'standard'
    
    def test_standard_scaling(self, sample_dataframe):
        """Test standard scaling."""
        scaler = FeatureScaler(method='standard')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        transformed = scaler.fit_transform(numeric_data)
        
        # Standard scaling should result in mean ~0 and std ~1
        assert np.allclose(transformed.mean(), 0, atol=1e-10)
        assert np.allclose(transformed.std(), 1, atol=1e-10)
    
    def test_minmax_scaling(self, sample_dataframe):
        """Test min-max scaling."""
        scaler = FeatureScaler(method='minmax')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        transformed = scaler.fit_transform(numeric_data)
        
        # Min-max scaling should result in values between 0 and 1
        assert transformed.min().min() >= 0
        assert transformed.max().max() <= 1
    
    def test_robust_scaling(self, sample_dataframe):
        """Test robust scaling."""
        scaler = FeatureScaler(method='robust')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        transformed = scaler.fit_transform(numeric_data)
        
        # Robust scaling should handle outliers better
        assert transformed.shape == numeric_data.shape
    
    def test_invalid_method(self):
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="Unsupported scaling method"):
            FeatureScaler(method='invalid')
    
    def test_feature_range_parameter(self, sample_dataframe):
        """Test feature_range parameter for minmax scaling."""
        scaler = FeatureScaler(method='minmax', feature_range=(-1, 1))
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        transformed = scaler.fit_transform(numeric_data)
        
        assert transformed.min().min() >= -1
        assert transformed.max().max() <= 1


class TestOutlierDetector:
    """Test cases for OutlierDetector."""
    
    def test_init_with_default_method(self):
        """Test initialization with default method."""
        detector = OutlierDetector()
        assert detector.method == 'iqr'
        assert detector.threshold == 1.5
    
    def test_iqr_outlier_detection(self, sample_dataframe):
        """Test IQR outlier detection."""
        detector = OutlierDetector(method='iqr')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        outliers = detector.fit_transform(numeric_data)
        
        assert isinstance(outliers, pd.DataFrame)
        assert outliers.shape[1] == numeric_data.shape[1]
        # Should be boolean values
        assert outliers.dtypes.apply(lambda x: x == bool).all()
    
    def test_zscore_outlier_detection(self, sample_dataframe):
        """Test Z-score outlier detection."""
        detector = OutlierDetector(method='zscore', threshold=3.0)
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        outliers = detector.fit_transform(numeric_data)
        
        assert isinstance(outliers, pd.DataFrame)
        assert outliers.dtypes.apply(lambda x: x == bool).all()
    
    def test_isolation_forest_outlier_detection(self, sample_dataframe):
        """Test Isolation Forest outlier detection."""
        detector = OutlierDetector(method='isolation_forest')
        numeric_data = sample_dataframe.select_dtypes(include=[np.number]).dropna()
        
        outliers = detector.fit_transform(numeric_data)
        
        assert isinstance(outliers, pd.DataFrame)
        assert outliers.dtypes.apply(lambda x: x == bool).all()
    
    def test_invalid_method(self):
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="Unsupported outlier detection method"):
            OutlierDetector(method='invalid')


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        pipeline = PreprocessingPipeline()
        assert pipeline.missing_value_strategy == 'mean'
        assert pipeline.scaling_method == 'standard'
        assert pipeline.encoding_strategy == 'onehot'
    
    def test_fit_transform_complete_pipeline(self, sample_dataframe):
        """Test complete pipeline fit_transform."""
        pipeline = PreprocessingPipeline(
            missing_value_strategy='mean',
            scaling_method='standard',
            encoding_strategy='onehot'
        )
        
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        X_transformed = pipeline.fit_transform(X, y)
        
        # Should have no missing values
        assert not X_transformed.isnull().any().any()
        # Should be a DataFrame
        assert isinstance(X_transformed, pd.DataFrame)
        # Should have at least as many columns (due to one-hot encoding)
        assert X_transformed.shape[1] >= X.select_dtypes(include=[np.number]).shape[1]
    
    def test_separate_fit_and_transform(self, sample_dataframe):
        """Test separate fit and transform calls."""
        pipeline = PreprocessingPipeline()
        
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        assert not X_transformed.isnull().any().any()
        assert isinstance(X_transformed, pd.DataFrame)
    
    def test_transform_without_fit_raises_error(self, sample_dataframe):
        """Test that transform without fit raises error."""
        pipeline = PreprocessingPipeline()
        X = sample_dataframe.drop('target', axis=1)
        
        with pytest.raises(ValueError, match="Pipeline has not been fitted"):
            pipeline.transform(X)
    
    def test_get_feature_names(self, sample_dataframe):
        """Test getting feature names after transformation."""
        pipeline = PreprocessingPipeline(encoding_strategy='onehot')
        
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        X_transformed = pipeline.fit_transform(X, y)
        feature_names = pipeline.get_feature_names()
        
        assert len(feature_names) == X_transformed.shape[1]
        assert isinstance(feature_names, list)
    
    def test_pipeline_with_outlier_detection(self, sample_dataframe):
        """Test pipeline with outlier detection enabled."""
        pipeline = PreprocessingPipeline(
            outlier_detection={'method': 'iqr', 'threshold': 1.5}
        )
        
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        X_transformed = pipeline.fit_transform(X, y)
        
        # Should still return valid DataFrame
        assert isinstance(X_transformed, pd.DataFrame)
        assert not X_transformed.isnull().any().any()


class TestDataValidator:
    """Test cases for DataValidator."""
    
    def test_validate_dataframe_basic(self, sample_dataframe):
        """Test basic DataFrame validation."""
        validator = DataValidator()
        results = validator.validate_dataframe(sample_dataframe)
        
        assert isinstance(results, dict)
        assert 'data_quality_score' in results
        assert 'missing_value_percentage' in results
        assert 'duplicate_rows' in results
        assert 'column_count' in results
        assert 'row_count' in results
    
    def test_validate_schema(self, sample_dataframe):
        """Test schema validation."""
        validator = DataValidator()
        
        expected_schema = {
            'feature_1': 'float64',
            'feature_2': 'float64',
            'feature_3': 'object',
            'feature_4': 'int64',
            'target': 'int64'
        }
        
        results = validator.validate_schema(sample_dataframe, expected_schema)
        
        assert isinstance(results, dict)
        assert 'schema_valid' in results
        assert 'schema_errors' in results
    
    def test_detect_outliers(self, sample_dataframe):
        """Test outlier detection."""
        validator = DataValidator()
        outliers = validator.detect_outliers(sample_dataframe)
        
        assert isinstance(outliers, dict)
        # Should have entries for numeric columns
        numeric_columns = sample_dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert col in outliers
    
    def test_calculate_statistics(self, sample_dataframe):
        """Test statistics calculation."""
        validator = DataValidator()
        stats = validator.calculate_statistics(sample_dataframe)
        
        assert isinstance(stats, dict)
        assert 'numeric_stats' in stats
        assert 'categorical_stats' in stats
    
    def test_check_data_drift_same_data(self, sample_dataframe):
        """Test data drift detection with same data."""
        validator = DataValidator()
        
        # Use same data - should have minimal drift
        drift_results = validator.check_data_drift(sample_dataframe, sample_dataframe)
        
        assert isinstance(drift_results, dict)
        assert 'overall_drift_score' in drift_results
        assert drift_results['overall_drift_score'] == 0.0  # No drift expected
    
    def test_check_data_drift_different_data(self, sample_dataframe):
        """Test data drift detection with different data."""
        validator = DataValidator()
        
        # Create modified version
        modified_data = sample_dataframe.copy()
        modified_data['feature_1'] = modified_data['feature_1'] + 10  # Shift distribution
        
        drift_results = validator.check_data_drift(sample_dataframe, modified_data)
        
        assert isinstance(drift_results, dict)
        assert 'overall_drift_score' in drift_results
        assert drift_results['overall_drift_score'] > 0  # Should detect drift


class TestImbalanceHandling:
    """Test cases for imbalance handling strategies."""
    
    def test_smote_strategy(self, sample_imbalanced_dataframe):
        """Test SMOTE strategy."""
        strategy = SMOTEStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should balance the classes
        class_counts = pd.Series(y_resampled).value_counts()
        assert len(class_counts) == 2  # Binary classification
        # Classes should be more balanced
        assert abs(class_counts[0] - class_counts[1]) < len(y) * 0.1
    
    def test_adasyn_strategy(self, sample_imbalanced_dataframe):
        """Test ADASYN strategy."""
        strategy = AdasynStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should increase minority class samples
        assert len(y_resampled) > len(y)
    
    def test_random_oversampling_strategy(self, sample_imbalanced_dataframe):
        """Test Random Oversampling strategy."""
        strategy = RandomOverSamplingStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should balance classes by oversampling
        class_counts = pd.Series(y_resampled).value_counts()
        assert class_counts[0] == class_counts[1]
    
    def test_random_undersampling_strategy(self, sample_imbalanced_dataframe):
        """Test Random Undersampling strategy."""
        strategy = RandomUnderSamplingStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should reduce majority class samples
        assert len(y_resampled) < len(y)
        
        # Classes should be balanced
        class_counts = pd.Series(y_resampled).value_counts()
        assert class_counts[0] == class_counts[1]
    
    def test_tomek_links_strategy(self, sample_imbalanced_dataframe):
        """Test Tomek Links strategy."""
        strategy = TomekLinksStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should remove some samples (Tomek links)
        assert len(y_resampled) <= len(y)
    
    def test_near_miss_strategy(self, sample_imbalanced_dataframe):
        """Test Near Miss strategy."""
        strategy = NearMissStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should undersample majority class
        assert len(y_resampled) < len(y)
    
    def test_balance_strategy_factory(self):
        """Test BalanceStrategyFactory."""
        # Test SMOTE creation
        strategy = BalanceStrategyFactory.create_strategy('smote')
        assert isinstance(strategy, SMOTEStrategy)
        
        # Test ADASYN creation
        strategy = BalanceStrategyFactory.create_strategy('adasyn')
        assert isinstance(strategy, AdasynStrategy)
        
        # Test Random Oversampling creation
        strategy = BalanceStrategyFactory.create_strategy('random_over')
        assert isinstance(strategy, RandomOverSamplingStrategy)
        
        # Test Random Undersampling creation
        strategy = BalanceStrategyFactory.create_strategy('random_under')
        assert isinstance(strategy, RandomUnderSamplingStrategy)
        
        # Test Tomek Links creation
        strategy = BalanceStrategyFactory.create_strategy('tomek_links')
        assert isinstance(strategy, TomekLinksStrategy)
        
        # Test Near Miss creation
        strategy = BalanceStrategyFactory.create_strategy('near_miss')
        assert isinstance(strategy, NearMissStrategy)
    
    def test_balance_strategy_factory_invalid(self):
        """Test BalanceStrategyFactory with invalid strategy."""
        with pytest.raises(ValueError, match="Unsupported balance strategy"):
            BalanceStrategyFactory.create_strategy('invalid_strategy')
    
    def test_strategy_with_custom_parameters(self, sample_imbalanced_dataframe):
        """Test strategy with custom parameters."""
        strategy = BalanceStrategyFactory.create_strategy(
            'smote', 
            sampling_strategy=0.8,  # Don't fully balance
            k_neighbors=3
        )
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        
        # Should work with custom parameters
        assert len(y_resampled) > len(y)
    
    def test_get_sampling_info(self, sample_imbalanced_dataframe):
        """Test getting sampling information."""
        strategy = SMOTEStrategy()
        
        X = sample_imbalanced_dataframe.drop('target', axis=1)
        y = sample_imbalanced_dataframe['target']
        
        X_resampled, y_resampled = strategy.fit_resample(X, y)
        info = strategy.get_sampling_info()
        
        assert isinstance(info, dict)
        assert 'strategy' in info
        assert 'class_counts_before' in info
        assert 'class_counts_after' in info
        assert info['strategy'] == 'SMOTE'


class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""
    
    def test_complete_preprocessing_workflow(self, sample_dataframe):
        """Test complete preprocessing workflow."""
        # Initialize pipeline
        pipeline = PreprocessingPipeline(
            missing_value_strategy='mean',
            scaling_method='standard',
            encoding_strategy='onehot'
        )
        
        # Prepare data
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        # Validate data first
        validator = DataValidator()
        validation_results = validator.validate_dataframe(sample_dataframe)
        assert validation_results['data_quality_score'] > 0
        
        # Apply preprocessing
        X_processed = pipeline.fit_transform(X, y)
        
        # Check results
        assert not X_processed.isnull().any().any()
        assert isinstance(X_processed, pd.DataFrame)
        assert X_processed.shape[0] == X.shape[0]
        
        # Apply imbalance handling if needed
        if validation_results['class_imbalance_ratio'] > 2:
            balance_strategy = BalanceStrategyFactory.create_strategy('smote')
            X_balanced, y_balanced = balance_strategy.fit_resample(X_processed, y)
            
            assert len(y_balanced) >= len(y)
    
    def test_preprocessing_pipeline_consistency(self, sample_dataframe):
        """Test that preprocessing pipeline produces consistent results."""
        pipeline = PreprocessingPipeline()
        
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        # First transformation
        X_transformed1 = pipeline.fit_transform(X, y)
        
        # Second transformation with same pipeline
        X_transformed2 = pipeline.transform(X)
        
        # Should be identical
        pd.testing.assert_frame_equal(X_transformed1, X_transformed2)
    
    def test_pipeline_serialization(self, sample_dataframe, temp_directory):
        """Test pipeline serialization and deserialization."""
        import pickle
        
        pipeline = PreprocessingPipeline()
        X = sample_dataframe.drop('target', axis=1)
        y = sample_dataframe['target']
        
        # Fit pipeline
        pipeline.fit(X, y)
        
        # Serialize
        pipeline_path = f"{temp_directory}/pipeline.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Deserialize
        with open(pipeline_path, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        # Test that loaded pipeline works
        X_original = pipeline.transform(X)
        X_loaded = loaded_pipeline.transform(X)
        
        pd.testing.assert_frame_equal(X_original, X_loaded)