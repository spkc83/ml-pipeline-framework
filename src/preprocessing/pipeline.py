import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import scipy.stats as stats

from ..utils.config_parser import PreprocessingConfig, CleaningConfig, TransformationConfig
from .validator import DataValidator

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    pass


class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline with configurable data cleaning,
    transformation, and quality assurance capabilities.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration object
        """
        self.config = config
        self.fitted_transformers = {}
        self.feature_statistics = {}
        self.processing_log = []
        self.validator = None
        
        if config and config.data_quality and config.data_quality.enabled:
            self.validator = DataValidator()
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if applicable)
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting preprocessing pipeline fit_transform")
        
        # Store original statistics
        self._store_original_statistics(df)
        
        # Data validation
        if self.validator:
            self._validate_input_data(df)
        
        # Apply preprocessing steps
        df_processed = df.copy()
        
        # Data quality checks
        if self.config and self.config.data_quality and self.config.data_quality.enabled:
            df_processed = self._apply_data_quality_checks(df_processed)
        
        # Data cleaning
        if self.config and self.config.cleaning:
            df_processed = self._apply_cleaning(df_processed, fit=True)
        
        # Data transformation
        if self.config and self.config.transformation:
            df_processed = self._apply_transformation(df_processed, target_column, fit=True)
        
        logger.info(f"Preprocessing completed. Shape: {df.shape} -> {df_processed.shape}")
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Applying preprocessing pipeline transform")
        
        if not self.fitted_transformers:
            raise PreprocessingError("Pipeline not fitted. Call fit_transform first.")
        
        df_processed = df.copy()
        
        # Apply cleaning (using fitted transformers)
        if self.config and self.config.cleaning:
            df_processed = self._apply_cleaning(df_processed, fit=False)
        
        # Apply transformation (using fitted transformers)
        if self.config and self.config.transformation:
            df_processed = self._apply_transformation(df_processed, None, fit=False)
        
        return df_processed
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "median", 
                            columns: Optional[List[str]] = None, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy (mean, median, mode, drop, forward_fill, knn)
            columns: Specific columns to process (None for all)
            fit: Whether to fit the imputer
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        df_result = df.copy()
        columns = columns or df.columns.tolist()
        
        missing_before = df_result[columns].isnull().sum().sum()
        
        if strategy == "drop":
            df_result = df_result.dropna(subset=columns)
        
        elif strategy == "forward_fill":
            df_result[columns] = df_result[columns].fillna(method='ffill')
        
        elif strategy == "backward_fill":
            df_result[columns] = df_result[columns].fillna(method='bfill')
        
        elif strategy in ["mean", "median", "most_frequent"]:
            imputer_key = f"imputer_{strategy}"
            
            if fit:
                # Separate numeric and categorical columns
                numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df[columns].select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols:
                    numeric_imputer = SimpleImputer(strategy=strategy if strategy != "mode" else "most_frequent")
                    df_result[numeric_cols] = numeric_imputer.fit_transform(df_result[numeric_cols])
                    self.fitted_transformers[f"{imputer_key}_numeric"] = numeric_imputer
                
                if categorical_cols:
                    categorical_imputer = SimpleImputer(strategy="most_frequent")
                    df_result[categorical_cols] = categorical_imputer.fit_transform(df_result[categorical_cols].astype(str))
                    self.fitted_transformers[f"{imputer_key}_categorical"] = categorical_imputer
            else:
                # Use fitted imputers
                numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df[columns].select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and f"{imputer_key}_numeric" in self.fitted_transformers:
                    df_result[numeric_cols] = self.fitted_transformers[f"{imputer_key}_numeric"].transform(df_result[numeric_cols])
                
                if categorical_cols and f"{imputer_key}_categorical" in self.fitted_transformers:
                    df_result[categorical_cols] = self.fitted_transformers[f"{imputer_key}_categorical"].transform(df_result[categorical_cols].astype(str))
        
        elif strategy == "knn":
            imputer_key = "knn_imputer"
            numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                if fit:
                    knn_imputer = KNNImputer(n_neighbors=5)
                    df_result[numeric_cols] = knn_imputer.fit_transform(df_result[numeric_cols])
                    self.fitted_transformers[imputer_key] = knn_imputer
                else:
                    if imputer_key in self.fitted_transformers:
                        df_result[numeric_cols] = self.fitted_transformers[imputer_key].transform(df_result[numeric_cols])
        
        missing_after = df_result[columns].isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return df_result
    
    def remove_multicollinearity(self, df: pd.DataFrame, threshold: float = 0.95,
                               target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold for removal
            target_column: Target column to exclude from analysis
            
        Returns:
            DataFrame with multicollinear features removed
        """
        logger.info(f"Removing multicollinear features with threshold: {threshold}")
        
        df_result = df.copy()
        feature_columns = [col for col in df.columns if col != target_column]
        numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            logger.warning("Less than 2 numeric columns found. Skipping multicollinearity removal.")
            return df_result
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_columns].corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            df_result = df_result.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} multicollinear features: {to_drop}")
            self.processing_log.append(f"Removed multicollinear features: {to_drop}")
        else:
            logger.info("No multicollinear features found")
        
        return df_result
    
    def calculate_psi_csi(self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
                         columns: Optional[List[str]] = None, bins: int = 10) -> Dict[str, float]:
        """
        Calculate Population Stability Index (PSI) and Characteristic Stability Index (CSI).
        
        Args:
            reference_df: Reference DataFrame (e.g., training data)
            current_df: Current DataFrame (e.g., validation/test data)
            columns: Columns to analyze (None for all numeric columns)
            bins: Number of bins for discretization
            
        Returns:
            Dictionary with PSI/CSI values for each column
        """
        logger.info("Calculating PSI/CSI values")
        
        if columns is None:
            columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        stability_metrics = {}
        
        for column in columns:
            if column not in current_df.columns:
                logger.warning(f"Column {column} not found in current data")
                continue
            
            try:
                # For numeric columns, create bins
                if reference_df[column].dtype in [np.int64, np.float64]:
                    # Create bins based on reference data
                    _, bin_edges = np.histogram(reference_df[column].dropna(), bins=bins)
                    
                    # Handle edge cases
                    bin_edges[0] = -np.inf
                    bin_edges[-1] = np.inf
                    
                    # Bin both datasets
                    ref_binned = pd.cut(reference_df[column], bins=bin_edges, include_lowest=True)
                    curr_binned = pd.cut(current_df[column], bins=bin_edges, include_lowest=True)
                    
                    # Calculate distributions
                    ref_dist = ref_binned.value_counts(normalize=True, sort=False)
                    curr_dist = curr_binned.value_counts(normalize=True, sort=False)
                
                else:
                    # For categorical columns
                    ref_dist = reference_df[column].value_counts(normalize=True)
                    curr_dist = current_df[column].value_counts(normalize=True)
                
                # Align distributions (handle missing categories)
                all_categories = ref_dist.index.union(curr_dist.index)
                ref_dist = ref_dist.reindex(all_categories, fill_value=1e-10)
                curr_dist = curr_dist.reindex(all_categories, fill_value=1e-10)
                
                # Calculate PSI
                psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
                stability_metrics[column] = psi
                
                logger.debug(f"PSI for {column}: {psi:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to calculate PSI for {column}: {str(e)}")
                stability_metrics[column] = np.nan
        
        return stability_metrics
    
    def encode_features(self, df: pd.DataFrame, encoding_config: Dict[str, Any],
                       target_column: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            encoding_config: Encoding configuration
            target_column: Target column for target encoding
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features")
        
        from .transformers import OneHotEncoder, TargetEncoder, FrequencyEncoder
        
        df_result = df.copy()
        categorical_method = encoding_config.get('categorical', {}).get('method', 'onehot')
        categorical_columns = encoding_config.get('categorical', {}).get('columns', [])
        
        if not categorical_columns:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
        
        if not categorical_columns:
            logger.info("No categorical columns found for encoding")
            return df_result
        
        encoder_key = f"encoder_{categorical_method}"
        
        if categorical_method == "onehot":
            if fit:
                encoder = OneHotEncoder(
                    handle_unknown=encoding_config.get('categorical', {}).get('handle_unknown', 'ignore')
                )
                encoded_data = encoder.fit_transform(df_result[categorical_columns])
                self.fitted_transformers[encoder_key] = encoder
            else:
                if encoder_key in self.fitted_transformers:
                    encoder = self.fitted_transformers[encoder_key]
                    encoded_data = encoder.transform(df_result[categorical_columns])
                else:
                    raise PreprocessingError(f"Encoder {encoder_key} not fitted")
        
        elif categorical_method == "target":
            if target_column is None:
                raise PreprocessingError("Target column required for target encoding")
            
            if fit:
                encoder = TargetEncoder()
                encoded_data = encoder.fit_transform(
                    df_result[categorical_columns], 
                    df_result[target_column]
                )
                self.fitted_transformers[encoder_key] = encoder
            else:
                if encoder_key in self.fitted_transformers:
                    encoder = self.fitted_transformers[encoder_key]
                    encoded_data = encoder.transform(df_result[categorical_columns])
                else:
                    raise PreprocessingError(f"Encoder {encoder_key} not fitted")
        
        elif categorical_method == "frequency":
            if fit:
                encoder = FrequencyEncoder()
                encoded_data = encoder.fit_transform(df_result[categorical_columns])
                self.fitted_transformers[encoder_key] = encoder
            else:
                if encoder_key in self.fitted_transformers:
                    encoder = self.fitted_transformers[encoder_key]
                    encoded_data = encoder.transform(df_result[categorical_columns])
                else:
                    raise PreprocessingError(f"Encoder {encoder_key} not fitted")
        
        elif categorical_method == "label":
            from sklearn.preprocessing import LabelEncoder
            
            if fit:
                encoders = {}
                for col in categorical_columns:
                    le = LabelEncoder()
                    df_result[col] = le.fit_transform(df_result[col].astype(str))
                    encoders[col] = le
                self.fitted_transformers[encoder_key] = encoders
            else:
                if encoder_key in self.fitted_transformers:
                    encoders = self.fitted_transformers[encoder_key]
                    for col in categorical_columns:
                        if col in encoders:
                            df_result[col] = encoders[col].transform(df_result[col].astype(str))
                else:
                    raise PreprocessingError(f"Encoder {encoder_key} not fitted")
            
            return df_result
        
        # For non-label encoding, replace original columns with encoded ones
        if categorical_method != "label":
            # Remove original categorical columns
            df_result = df_result.drop(columns=categorical_columns)
            
            # Add encoded columns
            if isinstance(encoded_data, pd.DataFrame):
                df_result = pd.concat([df_result, encoded_data], axis=1)
            else:
                # Convert to DataFrame if numpy array
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=[f"{col}_encoded" for col in categorical_columns],
                    index=df_result.index
                )
                df_result = pd.concat([df_result, encoded_df], axis=1)
        
        logger.info(f"Encoded {len(categorical_columns)} categorical columns using {categorical_method}")
        return df_result
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                        method: str = "smote", **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance in dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Resampling method (smote, random_over, random_under, adasyn, tomek, smote_tomek, smote_enn)
            **kwargs: Additional parameters for the resampling method
            
        Returns:
            Tuple of resampled (X, y)
        """
        logger.info(f"Handling class imbalance using method: {method}")
        
        # Check original class distribution
        original_distribution = y.value_counts()
        logger.info(f"Original class distribution: {original_distribution.to_dict()}")
        
        if method == "smote":
            sampler = SMOTE(random_state=kwargs.get('random_state', 42))
        elif method == "random_over":
            sampler = RandomOverSampler(random_state=kwargs.get('random_state', 42))
        elif method == "random_under":
            sampler = RandomUnderSampler(random_state=kwargs.get('random_state', 42))
        elif method == "adasyn":
            sampler = ADASYN(random_state=kwargs.get('random_state', 42))
        elif method == "tomek":
            sampler = TomekLinks()
        elif method == "smote_tomek":
            sampler = SMOTETomek(random_state=kwargs.get('random_state', 42))
        elif method == "smote_enn":
            sampler = SMOTEENN(random_state=kwargs.get('random_state', 42))
        else:
            raise PreprocessingError(f"Unknown imbalance handling method: {method}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Convert back to pandas
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            # Check new class distribution
            new_distribution = y_resampled.value_counts()
            logger.info(f"New class distribution: {new_distribution.to_dict()}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Failed to handle imbalance: {str(e)}")
            raise PreprocessingError(f"Failed to handle imbalance: {str(e)}")
    
    def _store_original_statistics(self, df: pd.DataFrame) -> None:
        """Store original dataset statistics."""
        self.feature_statistics['original'] = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data using Great Expectations."""
        try:
            suite_name = "input_validation_suite"
            self.validator.create_expectation_suite(suite_name, overwrite=True)
            self.validator.add_basic_expectations(suite_name, df)
            
            validation_results = self.validator.validate_data(df, suite_name)
            
            if not validation_results['success']:
                logger.warning(f"Data validation failed: {validation_results['failed_expectations']}")
                
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
    
    def _apply_data_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality checks based on configuration."""
        if not self.config.data_quality.checks:
            return df
        
        df_result = df.copy()
        
        for check in self.config.data_quality.checks:
            check_type = check.type
            action = check.action
            threshold = check.threshold
            
            if check_type == "missing_values":
                missing_ratio = df_result.isnull().sum() / len(df_result)
                problematic_cols = missing_ratio[missing_ratio > threshold].index.tolist()
                
                if problematic_cols:
                    if action == "drop":
                        df_result = df_result.drop(columns=problematic_cols)
                        logger.info(f"Dropped columns with high missing values: {problematic_cols}")
                    elif action == "warn":
                        logger.warning(f"High missing values in columns: {problematic_cols}")
            
            elif check_type == "duplicates":
                if action == "remove":
                    initial_rows = len(df_result)
                    df_result = df_result.drop_duplicates()
                    removed_rows = initial_rows - len(df_result)
                    if removed_rows > 0:
                        logger.info(f"Removed {removed_rows} duplicate rows")
            
            elif check_type == "outliers":
                numeric_cols = df_result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = df_result[col].quantile(0.25)
                    Q3 = df_result[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = (df_result[col] < lower_bound) | (df_result[col] > upper_bound)
                    
                    if action == "cap":
                        df_result.loc[df_result[col] < lower_bound, col] = lower_bound
                        df_result.loc[df_result[col] > upper_bound, col] = upper_bound
                        logger.info(f"Capped outliers in column {col}")
                    elif action == "remove":
                        df_result = df_result[~outliers]
                        logger.info(f"Removed outliers from column {col}")
        
        return df_result
    
    def _apply_cleaning(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply data cleaning steps."""
        df_result = df.copy()
        
        # Handle missing values
        if self.config.cleaning.handle_missing:
            missing_config = self.config.cleaning.handle_missing
            strategy = missing_config.get('strategy', 'median')
            columns = missing_config.get('columns')
            
            df_result = self.handle_missing_values(df_result, strategy, columns, fit)
        
        # Handle outliers
        if self.config.cleaning.handle_outliers:
            outlier_config = self.config.cleaning.handle_outliers
            method = outlier_config.get('method', 'clip')
            
            if method == "clip":
                lower_percentile = outlier_config.get('lower_percentile', 0.01)
                upper_percentile = outlier_config.get('upper_percentile', 0.99)
                
                numeric_cols = df_result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    lower_bound = df_result[col].quantile(lower_percentile)
                    upper_bound = df_result[col].quantile(upper_percentile)
                    df_result[col] = df_result[col].clip(lower_bound, upper_bound)
        
        # Parse dates
        if self.config.cleaning.date_parsing:
            date_config = self.config.cleaning.date_parsing
            columns = date_config.get('columns', [])
            date_format = date_config.get('format', '%Y-%m-%d %H:%M:%S')
            
            for col in columns:
                if col in df_result.columns:
                    df_result[col] = pd.to_datetime(df_result[col], format=date_format, errors='coerce')
        
        return df_result
    
    def _apply_transformation(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                            fit: bool = True) -> pd.DataFrame:
        """Apply data transformations."""
        df_result = df.copy()
        
        # Scaling
        if self.config.transformation.scaling:
            scaling_config = self.config.transformation.scaling
            method = scaling_config.get('method', 'standard')
            columns = scaling_config.get('columns')
            
            if columns is None:
                columns = df_result.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in columns:
                    columns.remove(target_column)
            
            scaler_key = f"scaler_{method}"
            
            if fit:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                elif method == "robust":
                    scaler = RobustScaler()
                else:
                    raise PreprocessingError(f"Unknown scaling method: {method}")
                
                df_result[columns] = scaler.fit_transform(df_result[columns])
                self.fitted_transformers[scaler_key] = scaler
            else:
                if scaler_key in self.fitted_transformers:
                    scaler = self.fitted_transformers[scaler_key]
                    df_result[columns] = scaler.transform(df_result[columns])
        
        # Encoding
        if self.config.transformation.encoding:
            df_result = self.encode_features(df_result, self.config.transformation.encoding, target_column, fit)
        
        # Feature selection
        if self.config.transformation.feature_selection and fit:
            selection_config = self.config.transformation.feature_selection
            method = selection_config.get('method', 'variance_threshold')
            
            feature_cols = [col for col in df_result.columns if col != target_column]
            
            if method == "variance_threshold":
                threshold = selection_config.get('threshold', 0.01)
                selector = VarianceThreshold(threshold=threshold)
                
                selected_features = selector.fit_transform(df_result[feature_cols])
                selected_columns = df_result[feature_cols].columns[selector.get_support()].tolist()
                
                df_result = df_result[selected_columns + ([target_column] if target_column else [])]
                self.fitted_transformers['feature_selector'] = selector
                
                logger.info(f"Selected {len(selected_columns)} features using variance threshold")
        
        return df_result