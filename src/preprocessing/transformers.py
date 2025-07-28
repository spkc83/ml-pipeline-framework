import numpy as np
import pandas as pd
import polars as pl
from typing import Any, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer as SklearnPowerTransformer
import warnings

logger = logging.getLogger(__name__)


class DataFrameCompatibleTransformer(ABC):
    """
    Abstract base class for transformers that work with both pandas and Polars DataFrames.
    """
    
    def __init__(self):
        self.is_fitted = False
        self._feature_names = None
        self._input_type = None
    
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], y: Optional[Union[pd.Series, pl.Series]] = None):
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: Union[pd.DataFrame, pl.DataFrame], 
                     y: Optional[Union[pd.Series, pl.Series]] = None) -> Union[pd.DataFrame, pl.DataFrame]:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)
    
    def _check_input_type(self, X: Union[pd.DataFrame, pl.DataFrame]) -> str:
        """Check and store input data type."""
        if isinstance(X, pd.DataFrame):
            input_type = "pandas"
        elif isinstance(X, pl.DataFrame):
            input_type = "polars"
        else:
            raise ValueError(f"Unsupported data type: {type(X)}. Expected pandas or polars DataFrame.")
        
        if self._input_type is None:
            self._input_type = input_type
        elif self._input_type != input_type:
            warnings.warn(f"Input type changed from {self._input_type} to {input_type}")
        
        return input_type
    
    def _to_pandas(self, X: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """Convert input to pandas DataFrame."""
        if isinstance(X, pl.DataFrame):
            return X.to_pandas()
        return X
    
    def _to_original_type(self, X_transformed: pd.DataFrame, 
                         original_type: str) -> Union[pd.DataFrame, pl.DataFrame]:
        """Convert transformed data back to original type."""
        if original_type == "polars":
            return pl.from_pandas(X_transformed)
        return X_transformed


class StandardScaler(DataFrameCompatibleTransformer):
    """
    Standard scaler that works with both pandas and Polars DataFrames.
    Standardizes features by removing the mean and scaling to unit variance.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None
        self.var_ = None
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Optional[Union[pd.Series, pl.Series]] = None):
        """
        Fit the StandardScaler to X.
        
        Args:
            X: Training data
            y: Target values (ignored)
            
        Returns:
            self
        """
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        self._feature_names = list(X_pandas.columns)
        
        # Calculate statistics
        if self.with_mean:
            self.mean_ = X_pandas.mean()
        else:
            self.mean_ = pd.Series(0, index=X_pandas.columns)
        
        if self.with_std:
            self.std_ = X_pandas.std()
            self.var_ = X_pandas.var()
            # Handle zero variance
            self.std_ = self.std_.replace(0, 1)
        else:
            self.std_ = pd.Series(1, index=X_pandas.columns)
            self.var_ = pd.Series(1, index=X_pandas.columns)
        
        self.is_fitted = True
        logger.info(f"StandardScaler fitted with {len(self._feature_names)} features")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Standardize features by removing the mean and scaling to unit variance.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("StandardScaler must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        # Transform
        X_transformed = (X_pandas - self.mean_) / self.std_
        
        return self._to_original_type(X_transformed, input_type)


class MinMaxScaler(DataFrameCompatibleTransformer):
    """
    MinMax scaler that works with both pandas and Polars DataFrames.
    Scales features to a given range, typically [0, 1].
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Optional[Union[pd.Series, pl.Series]] = None):
        """
        Fit the MinMaxScaler to X.
        
        Args:
            X: Training data
            y: Target values (ignored)
            
        Returns:
            self
        """
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        self._feature_names = list(X_pandas.columns)
        
        # Calculate statistics
        self.data_min_ = X_pandas.min()
        self.data_max_ = X_pandas.max()
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle zero range
        self.data_range_ = self.data_range_.replace(0, 1)
        
        # Calculate scale and min for transformation
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        self.is_fitted = True
        logger.info(f"MinMaxScaler fitted with range {self.feature_range}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Scale features to the specified range.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("MinMaxScaler must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        # Transform
        X_transformed = X_pandas * self.scale_ + self.min_
        
        return self._to_original_type(X_transformed, input_type)


class PowerTransformer(DataFrameCompatibleTransformer):
    """
    Power transformer that works with both pandas and Polars DataFrames.
    Applies power transformations to make data more Gaussian-like.
    """
    
    def __init__(self, method: str = 'yeo-johnson', standardize: bool = True):
        super().__init__()
        self.method = method
        self.standardize = standardize
        self._sklearn_transformer = None
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Optional[Union[pd.Series, pl.Series]] = None):
        """
        Fit the PowerTransformer to X.
        
        Args:
            X: Training data
            y: Target values (ignored)
            
        Returns:
            self
        """
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        self._feature_names = list(X_pandas.columns)
        
        # Use sklearn's PowerTransformer
        self._sklearn_transformer = SklearnPowerTransformer(
            method=self.method,
            standardize=self.standardize
        )
        
        self._sklearn_transformer.fit(X_pandas)
        
        self.is_fitted = True
        logger.info(f"PowerTransformer fitted with method '{self.method}'")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Apply power transformation to make data more Gaussian-like.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("PowerTransformer must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        # Transform
        X_transformed_array = self._sklearn_transformer.transform(X_pandas)
        X_transformed = pd.DataFrame(
            X_transformed_array,
            columns=X_pandas.columns,
            index=X_pandas.index
        )
        
        return self._to_original_type(X_transformed, input_type)


class OneHotEncoder(DataFrameCompatibleTransformer):
    """
    One-hot encoder that works with both pandas and Polars DataFrames.
    """
    
    def __init__(self, drop: Optional[str] = None, handle_unknown: str = 'error', 
                 max_categories: Optional[int] = None):
        super().__init__()
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self.categories_ = {}
        self.feature_names_out_ = []
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Optional[Union[pd.Series, pl.Series]] = None):
        """
        Fit the OneHotEncoder to X.
        
        Args:
            X: Training data
            y: Target values (ignored)
            
        Returns:
            self
        """
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        self._feature_names = list(X_pandas.columns)
        
        # Learn categories for each column
        for column in X_pandas.columns:
            unique_values = X_pandas[column].unique()
            unique_values = unique_values[~pd.isna(unique_values)]
            
            # Limit categories if specified
            if self.max_categories and len(unique_values) > self.max_categories:
                value_counts = X_pandas[column].value_counts()
                unique_values = value_counts.head(self.max_categories).index.values
            
            self.categories_[column] = sorted(unique_values)
        
        # Generate output feature names
        self.feature_names_out_ = []
        for column in X_pandas.columns:
            categories = self.categories_[column]
            if self.drop == 'first' and len(categories) > 1:
                categories = categories[1:]
            
            for category in categories:
                self.feature_names_out_.append(f"{column}_{category}")
        
        self.is_fitted = True
        logger.info(f"OneHotEncoder fitted. Output features: {len(self.feature_names_out_)}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Transform X using one-hot encoding.
        
        Args:
            X: Data to transform
            
        Returns:
            One-hot encoded data
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        encoded_dfs = []
        
        for column in X_pandas.columns:
            if column not in self.categories_:
                if self.handle_unknown == 'error':
                    raise ValueError(f"Unknown column: {column}")
                continue
            
            categories = self.categories_[column]
            
            # Create dummy variables
            dummies = pd.get_dummies(X_pandas[column], prefix=column)
            
            # Align with fitted categories
            for category in categories:
                col_name = f"{column}_{category}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Handle unknown categories
            if self.handle_unknown == 'ignore':
                # Keep only known categories
                known_cols = [f"{column}_{cat}" for cat in categories]
                dummies = dummies[known_cols]
            
            # Drop first category if specified
            if self.drop == 'first' and len(categories) > 1:
                first_col = f"{column}_{categories[0]}"
                if first_col in dummies.columns:
                    dummies = dummies.drop(columns=[first_col])
            
            encoded_dfs.append(dummies)
        
        # Concatenate all encoded columns
        if encoded_dfs:
            X_transformed = pd.concat(encoded_dfs, axis=1)
        else:
            X_transformed = pd.DataFrame(index=X_pandas.index)
        
        return self._to_original_type(X_transformed, input_type)


class TargetEncoder(DataFrameCompatibleTransformer):
    """
    Target encoder that works with both pandas and Polars DataFrames.
    Replaces categorical values with target mean.
    """
    
    def __init__(self, smoothing: float = 1.0, min_samples_leaf: int = 1):
        super().__init__()
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.target_mean_ = None
        self.encodings_ = {}
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Union[pd.Series, pl.Series]):
        """
        Fit the TargetEncoder to X and y.
        
        Args:
            X: Training data (categorical features)
            y: Target values
            
        Returns:
            self
        """
        if y is None:
            raise ValueError("TargetEncoder requires target values for fitting")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        if isinstance(y, pl.Series):
            y_pandas = y.to_pandas()
        else:
            y_pandas = y
        
        self._feature_names = list(X_pandas.columns)
        self.target_mean_ = y_pandas.mean()
        
        # Calculate target mean for each category
        for column in X_pandas.columns:
            # Group by category and calculate mean
            grouped = pd.concat([X_pandas[column], y_pandas], axis=1).groupby(column)
            counts = grouped.size()
            means = grouped[y_pandas.name].mean()
            
            # Apply smoothing
            # smoothed_mean = (counts * mean + smoothing * global_mean) / (counts + smoothing)
            smoothed_means = (counts * means + self.smoothing * self.target_mean_) / (counts + self.smoothing)
            
            # Filter by minimum samples
            valid_categories = counts >= self.min_samples_leaf
            smoothed_means = smoothed_means[valid_categories]
            
            self.encodings_[column] = smoothed_means.to_dict()
        
        self.is_fitted = True
        logger.info(f"TargetEncoder fitted for {len(self._feature_names)} features")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Transform X using target encoding.
        
        Args:
            X: Data to transform
            
        Returns:
            Target encoded data
        """
        if not self.is_fitted:
            raise ValueError("TargetEncoder must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        X_transformed = X_pandas.copy()
        
        for column in X_pandas.columns:
            if column in self.encodings_:
                # Map categories to their target means
                X_transformed[column] = X_pandas[column].map(self.encodings_[column])
                
                # Fill unknown categories with global mean
                X_transformed[column] = X_transformed[column].fillna(self.target_mean_)
        
        return self._to_original_type(X_transformed, input_type)


class FrequencyEncoder(DataFrameCompatibleTransformer):
    """
    Frequency encoder that works with both pandas and Polars DataFrames.
    Replaces categorical values with their frequency of occurrence.
    """
    
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.frequencies_ = {}
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Optional[Union[pd.Series, pl.Series]] = None):
        """
        Fit the FrequencyEncoder to X.
        
        Args:
            X: Training data
            y: Target values (ignored)
            
        Returns:
            self
        """
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        self._feature_names = list(X_pandas.columns)
        
        # Calculate frequencies for each column
        for column in X_pandas.columns:
            value_counts = X_pandas[column].value_counts()
            
            if self.normalize:
                value_counts = value_counts / len(X_pandas)
            
            self.frequencies_[column] = value_counts.to_dict()
        
        self.is_fitted = True
        logger.info(f"FrequencyEncoder fitted for {len(self._feature_names)} features")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Transform X using frequency encoding.
        
        Args:
            X: Data to transform
            
        Returns:
            Frequency encoded data
        """
        if not self.is_fitted:
            raise ValueError("FrequencyEncoder must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        X_transformed = X_pandas.copy()
        
        for column in X_pandas.columns:
            if column in self.frequencies_:
                # Map categories to their frequencies
                X_transformed[column] = X_pandas[column].map(self.frequencies_[column])
                
                # Fill unknown categories with 0
                X_transformed[column] = X_transformed[column].fillna(0)
        
        return self._to_original_type(X_transformed, input_type)


class RobustScaler(DataFrameCompatibleTransformer):
    """
    Robust scaler that works with both pandas and Polars DataFrames.
    Uses median and interquartile range for scaling, making it robust to outliers.
    """
    
    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0), 
                 with_centering: bool = True, with_scaling: bool = True):
        super().__init__()
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], 
           y: Optional[Union[pd.Series, pl.Series]] = None):
        """
        Fit the RobustScaler to X.
        
        Args:
            X: Training data
            y: Target values (ignored)
            
        Returns:
            self
        """
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        self._feature_names = list(X_pandas.columns)
        
        # Calculate robust statistics
        q_min, q_max = self.quantile_range
        
        if self.with_centering:
            self.center_ = X_pandas.quantile(0.5)  # Median
        else:
            self.center_ = pd.Series(0, index=X_pandas.columns)
        
        if self.with_scaling:
            q1 = X_pandas.quantile(q_min / 100.0)
            q3 = X_pandas.quantile(q_max / 100.0)
            self.scale_ = q3 - q1
            # Handle zero IQR
            self.scale_ = self.scale_.replace(0, 1)
        else:
            self.scale_ = pd.Series(1, index=X_pandas.columns)
        
        self.is_fitted = True
        logger.info(f"RobustScaler fitted with quantile range {self.quantile_range}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Scale features using robust statistics.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("RobustScaler must be fitted before transforming data")
        
        input_type = self._check_input_type(X)
        X_pandas = self._to_pandas(X)
        
        # Transform
        X_transformed = (X_pandas - self.center_) / self.scale_
        
        return self._to_original_type(X_transformed, input_type)


# Convenience function to get transformer by name
def get_transformer(name: str, **kwargs) -> DataFrameCompatibleTransformer:
    """
    Get a transformer by name.
    
    Args:
        name: Name of the transformer
        **kwargs: Parameters for the transformer
        
    Returns:
        Transformer instance
    """
    transformers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
        'power': PowerTransformer,
        'onehot': OneHotEncoder,
        'target': TargetEncoder,
        'frequency': FrequencyEncoder,
    }
    
    if name not in transformers:
        raise ValueError(f"Unknown transformer: {name}. Available: {list(transformers.keys())}")
    
    return transformers[name](**kwargs)