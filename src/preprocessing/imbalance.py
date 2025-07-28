import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

# Imbalanced-learn imports with error handling
try:
    from imblearn.over_sampling import SMOTE as ImbSMOTE, ADASYN as ImbADASYN, RandomOverSampler
    from imblearn.under_sampling import TomekLinks as ImbTomekLinks, NearMiss as ImbNearMiss, RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.base import BaseSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Imbalanced-learn not available. Install with: pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False
    ImbSMOTE = ImbADASYN = ImbTomekLinks = ImbNearMiss = None
    RandomOverSampler = RandomUnderSampler = SMOTEENN = SMOTETomek = BaseSampler = None

# Scikit-learn imports
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils import check_X_y, check_array
    from sklearn.utils.validation import check_is_fitted
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn not available for sampling strategies")
    SKLEARN_AVAILABLE = False
    NearestNeighbors = BaseEstimator = TransformerMixin = None
    check_X_y = check_array = check_is_fitted = None


class ImbalanceError(Exception):
    pass


class SamplingStrategy(Enum):
    """Enumeration of available sampling strategies."""
    RANDOM_OVERSAMPLE = "random_oversample"
    RANDOM_UNDERSAMPLE = "random_undersample"
    SMOTE = "smote"
    ADASYN = "adasyn"
    TOMEK_LINKS = "tomek_links"
    NEAR_MISS = "near_miss"
    SMOTE_TOMEK = "smote_tomek"
    SMOTE_ENN = "smote_enn"


class BaseSamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies to handle imbalanced datasets.
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize base sampling strategy.
        
        Args:
            random_state: Random state for reproducibility
            **kwargs: Additional strategy-specific parameters
        """
        self.random_state = random_state
        self.params = kwargs
        self.is_fitted = False
        self.class_counts_before_ = None
        self.class_counts_after_ = None
        self.sampling_strategy_ = None
    
    @abstractmethod
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Fit the strategy and resample the dataset.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        pass
    
    def _log_sampling_info(self, y_before: np.ndarray, y_after: np.ndarray) -> None:
        """Log information about the sampling process."""
        unique_before, counts_before = np.unique(y_before, return_counts=True)
        unique_after, counts_after = np.unique(y_after, return_counts=True)
        
        self.class_counts_before_ = dict(zip(unique_before, counts_before))
        self.class_counts_after_ = dict(zip(unique_after, counts_after))
        
        logger.info(f"Sampling completed:")
        logger.info(f"  Before: {self.class_counts_before_}")
        logger.info(f"  After:  {self.class_counts_after_}")
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get information about the sampling process."""
        return {
            'class_counts_before': self.class_counts_before_,
            'class_counts_after': self.class_counts_after_,
            'strategy': self.__class__.__name__,
            'parameters': self.params
        }


class TomekLinks(BaseSamplingStrategy):
    """
    Tomek Links undersampling strategy.
    Removes majority class samples that form Tomek links with minority class samples.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', n_jobs: int = 1, **kwargs):
        """
        Initialize Tomek Links sampler.
        
        Args:
            sampling_strategy: Sampling strategy ('auto', 'majority', 'all')
            n_jobs: Number of jobs for parallel processing
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs
        
        if IMBLEARN_AVAILABLE:
            self.sampler = ImbTomekLinks(
                sampling_strategy=sampling_strategy,
                n_jobs=n_jobs
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply Tomek Links undersampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for Tomek Links. Install with: pip install imbalanced-learn")
        
        logger.info("Applying Tomek Links undersampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Tomek Links sampling failed: {str(e)}")
            raise ImbalanceError(f"Tomek Links sampling failed: {str(e)}")


class NearMiss(BaseSamplingStrategy):
    """
    NearMiss undersampling strategy.
    Selects majority class samples based on their distance to minority class samples.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', version: int = 1, 
                 n_neighbors: int = 3, n_jobs: int = 1, **kwargs):
        """
        Initialize NearMiss sampler.
        
        Args:
            sampling_strategy: Sampling strategy ('auto', 'majority', dict)
            version: NearMiss version (1, 2, or 3)
            n_neighbors: Number of neighbors for distance calculation
            n_jobs: Number of jobs for parallel processing
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        self.version = version
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        
        if IMBLEARN_AVAILABLE:
            self.sampler = ImbNearMiss(
                sampling_strategy=sampling_strategy,
                version=version,
                n_neighbors=n_neighbors,
                n_jobs=n_jobs
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply NearMiss undersampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for NearMiss. Install with: pip install imbalanced-learn")
        
        logger.info(f"Applying NearMiss-{self.version} undersampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"NearMiss sampling failed: {str(e)}")
            raise ImbalanceError(f"NearMiss sampling failed: {str(e)}")


class SMOTE(BaseSamplingStrategy):
    """
    SMOTE (Synthetic Minority Oversampling Technique) strategy.
    Generates synthetic minority class samples using nearest neighbors.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', k_neighbors: int = 5, 
                 n_jobs: int = 1, **kwargs):
        """
        Initialize SMOTE sampler.
        
        Args:
            sampling_strategy: Sampling strategy ('auto', 'minority', dict)
            k_neighbors: Number of nearest neighbors for synthesis
            n_jobs: Number of jobs for parallel processing
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        
        if IMBLEARN_AVAILABLE:
            self.sampler = ImbSMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k_neighbors,
                n_jobs=n_jobs
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply SMOTE oversampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for SMOTE. Install with: pip install imbalanced-learn")
        
        logger.info("Applying SMOTE oversampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE sampling failed: {str(e)}")
            raise ImbalanceError(f"SMOTE sampling failed: {str(e)}")


class ADASYN(BaseSamplingStrategy):
    """
    ADASYN (Adaptive Synthetic Sampling) strategy.
    Generates synthetic samples with density-based weighting for minority classes.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', n_neighbors: int = 5, 
                 n_jobs: int = 1, **kwargs):
        """
        Initialize ADASYN sampler.
        
        Args:
            sampling_strategy: Sampling strategy ('auto', 'minority', dict)
            n_neighbors: Number of nearest neighbors for synthesis
            n_jobs: Number of jobs for parallel processing
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        
        if IMBLEARN_AVAILABLE:
            self.sampler = ImbADASYN(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_neighbors=n_neighbors,
                n_jobs=n_jobs
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply ADASYN oversampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for ADASYN. Install with: pip install imbalanced-learn")
        
        logger.info("Applying ADASYN oversampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"ADASYN sampling failed: {str(e)}")
            raise ImbalanceError(f"ADASYN sampling failed: {str(e)}")


class RandomOverSampler(BaseSamplingStrategy):
    """
    Random oversampling strategy.
    Randomly duplicates minority class samples.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', **kwargs):
        """
        Initialize random oversampler.
        
        Args:
            sampling_strategy: Sampling strategy ('auto', 'minority', dict)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        
        if IMBLEARN_AVAILABLE:
            self.sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply random oversampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for random oversampling. Install with: pip install imbalanced-learn")
        
        logger.info("Applying random oversampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Random oversampling failed: {str(e)}")
            raise ImbalanceError(f"Random oversampling failed: {str(e)}")


class RandomUnderSampler(BaseSamplingStrategy):
    """
    Random undersampling strategy.
    Randomly removes majority class samples.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', replacement: bool = False, **kwargs):
        """
        Initialize random undersampler.
        
        Args:
            sampling_strategy: Sampling strategy ('auto', 'majority', dict)
            replacement: Whether to sample with replacement
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        
        if IMBLEARN_AVAILABLE:
            self.sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                replacement=replacement
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply random undersampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for random undersampling. Install with: pip install imbalanced-learn")
        
        logger.info("Applying random undersampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Random undersampling failed: {str(e)}")
            raise ImbalanceError(f"Random undersampling failed: {str(e)}")


class SMOTETomek(BaseSamplingStrategy):
    """
    Combined SMOTE and Tomek Links strategy.
    First applies SMOTE oversampling, then Tomek Links cleaning.
    """
    
    def __init__(self, sampling_strategy: str = 'auto', smote_k_neighbors: int = 5, 
                 tomek_sampling_strategy: str = 'auto', n_jobs: int = 1, **kwargs):
        """
        Initialize SMOTE-Tomek sampler.
        
        Args:
            sampling_strategy: SMOTE sampling strategy
            smote_k_neighbors: Number of neighbors for SMOTE
            tomek_sampling_strategy: Tomek Links sampling strategy
            n_jobs: Number of jobs for parallel processing
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy
        self.smote_k_neighbors = smote_k_neighbors
        self.tomek_sampling_strategy = tomek_sampling_strategy
        self.n_jobs = n_jobs
        
        if IMBLEARN_AVAILABLE:
            self.sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                smote=ImbSMOTE(k_neighbors=smote_k_neighbors, random_state=self.random_state),
                tomek=ImbTomekLinks(sampling_strategy=tomek_sampling_strategy, n_jobs=n_jobs)
            )
        else:
            self.sampler = None
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                              Union[pd.Series, np.ndarray]]:
        """
        Apply SMOTE-Tomek combined sampling.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not IMBLEARN_AVAILABLE:
            raise ImbalanceError("Imbalanced-learn is required for SMOTE-Tomek. Install with: pip install imbalanced-learn")
        
        logger.info("Applying SMOTE-Tomek combined sampling")
        
        # Convert to numpy for processing
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
            
            # Convert back to original format
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            self._log_sampling_info(y_array, y_resampled)
            self.is_fitted = True
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE-Tomek sampling failed: {str(e)}")
            raise ImbalanceError(f"SMOTE-Tomek sampling failed: {str(e)}")


class BalanceStrategyFactory:
    """
    Factory class for creating and configuring imbalance handling strategies.
    """
    
    _strategy_mapping = {
        SamplingStrategy.RANDOM_OVERSAMPLE: RandomOverSampler,
        SamplingStrategy.RANDOM_UNDERSAMPLE: RandomUnderSampler,
        SamplingStrategy.SMOTE: SMOTE,
        SamplingStrategy.ADASYN: ADASYN,
        SamplingStrategy.TOMEK_LINKS: TomekLinks,
        SamplingStrategy.NEAR_MISS: NearMiss,
        SamplingStrategy.SMOTE_TOMEK: SMOTETomek,
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **params) -> BaseSamplingStrategy:
        """
        Create a sampling strategy instance.
        
        Args:
            strategy_name: Name of the sampling strategy
            **params: Parameters for the strategy
            
        Returns:
            BaseSamplingStrategy instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        try:
            strategy_enum = SamplingStrategy(strategy_name.lower())
        except ValueError:
            available_strategies = [s.value for s in SamplingStrategy]
            raise ValueError(f"Unsupported sampling strategy: {strategy_name}. "
                           f"Available strategies: {available_strategies}")
        
        if strategy_enum not in cls._strategy_mapping:
            raise ValueError(f"Strategy {strategy_name} not implemented")
        
        strategy_class = cls._strategy_mapping[strategy_enum]
        
        logger.info(f"Creating sampling strategy: {strategy_name}")
        return strategy_class(**params)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseSamplingStrategy:
        """
        Create strategy from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'method' and optional parameters
            
        Returns:
            BaseSamplingStrategy instance
        """
        if 'method' not in config:
            raise ValueError("Configuration must include 'method' field")
        
        method = config['method']
        params = {k: v for k, v in config.items() if k != 'method'}
        
        return cls.create_strategy(method, **params)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get list of available sampling strategies.
        
        Returns:
            List of strategy names
        """
        return [s.value for s in SamplingStrategy]
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information
        """
        strategy_info = {
            SamplingStrategy.RANDOM_OVERSAMPLE.value: {
                "type": "oversampling",
                "description": "Random duplication of minority class samples",
                "parameters": ["sampling_strategy", "random_state"]
            },
            SamplingStrategy.RANDOM_UNDERSAMPLE.value: {
                "type": "undersampling", 
                "description": "Random removal of majority class samples",
                "parameters": ["sampling_strategy", "replacement", "random_state"]
            },
            SamplingStrategy.SMOTE.value: {
                "type": "oversampling",
                "description": "Synthetic Minority Oversampling Technique using nearest neighbors",
                "parameters": ["sampling_strategy", "k_neighbors", "random_state", "n_jobs"]
            },
            SamplingStrategy.ADASYN.value: {
                "type": "oversampling",
                "description": "Adaptive Synthetic Sampling with density-based weighting",
                "parameters": ["sampling_strategy", "n_neighbors", "random_state", "n_jobs"]
            },
            SamplingStrategy.TOMEK_LINKS.value: {
                "type": "undersampling",
                "description": "Removes majority samples that form Tomek links",
                "parameters": ["sampling_strategy", "n_jobs"]
            },
            SamplingStrategy.NEAR_MISS.value: {
                "type": "undersampling",
                "description": "Selects majority samples based on distance to minority samples",
                "parameters": ["sampling_strategy", "version", "n_neighbors", "n_jobs"]
            },
            SamplingStrategy.SMOTE_TOMEK.value: {
                "type": "combined",
                "description": "SMOTE oversampling followed by Tomek Links cleaning",
                "parameters": ["sampling_strategy", "smote_k_neighbors", "tomek_sampling_strategy", "random_state", "n_jobs"]
            }
        }
        
        return strategy_info.get(strategy_name, {"error": "Strategy not found"})
    
    @classmethod
    def recommend_strategy(cls, X: Union[pd.DataFrame, np.ndarray], 
                          y: Union[pd.Series, np.ndarray],
                          dataset_size: str = "medium") -> str:
        """
        Recommend a sampling strategy based on dataset characteristics.
        
        Args:
            X: Features
            y: Target values
            dataset_size: Size of dataset ("small", "medium", "large")
            
        Returns:
            Recommended strategy name
        """
        # Convert to numpy for analysis
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Calculate imbalance ratio
        unique_classes, counts = np.unique(y_array, return_counts=True)
        imbalance_ratio = min(counts) / max(counts)
        
        # Get dataset characteristics
        n_samples = len(y_array)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
        
        logger.info(f"Dataset analysis: {n_samples} samples, {n_features} features, "
                   f"imbalance ratio: {imbalance_ratio:.3f}")
        
        # Recommendation logic
        if imbalance_ratio > 0.3:
            # Mild imbalance
            return SamplingStrategy.RANDOM_OVERSAMPLE.value
        elif imbalance_ratio > 0.1:
            # Moderate imbalance
            if dataset_size == "small" or n_samples < 1000:
                return SamplingStrategy.SMOTE.value
            else:
                return SamplingStrategy.ADASYN.value
        else:
            # Severe imbalance
            if dataset_size == "large" or n_samples > 50000:
                return SamplingStrategy.SMOTE_TOMEK.value
            else:
                return SamplingStrategy.SMOTE.value
    
    @classmethod
    def auto_balance(cls, X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray],
                    strategy: Optional[str] = None,
                    **params) -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                     Union[pd.Series, np.ndarray]]:
        """
        Automatically balance dataset using recommended or specified strategy.
        
        Args:
            X: Features
            y: Target values
            strategy: Sampling strategy (None for auto-recommendation)
            **params: Additional parameters for the strategy
            
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        if strategy is None:
            strategy = cls.recommend_strategy(X, y)
            logger.info(f"Auto-selected strategy: {strategy}")
        
        sampler = cls.create_strategy(strategy, **params)
        return sampler.fit_resample(X, y)