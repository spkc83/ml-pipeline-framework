"""
Feature Interaction Analyzer for ML Pipeline Framework

This module detects pairwise and higher-order feature interactions, uses H-statistic
for interaction strength measurement, creates interaction network visualizations,
and identifies interaction-based patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform

# Advanced interaction detection
try:
    from sklearn.inspection import partial_dependence
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False
    warnings.warn("sklearn.inspection not available. Some interaction methods will be limited.")

# Network visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be limited.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class InteractionConfig:
    """Configuration for feature interaction analysis."""
    
    # Interaction detection parameters
    max_interaction_order: int = 2  # Maximum order of interactions to detect
    min_interaction_strength: float = 0.01  # Minimum H-statistic threshold
    significance_level: float = 0.05  # Statistical significance threshold
    
    # Sampling for efficiency
    max_samples_h_statistic: int = 1000
    max_samples_visualization: int = 500
    
    # Network analysis
    min_network_edge_weight: float = 0.1
    max_network_nodes: int = 50
    
    # Feature engineering suggestions
    enable_feature_suggestions: bool = True
    min_suggestion_strength: float = 0.2
    
    # Fraud-specific patterns
    enable_fraud_patterns: bool = False
    fraud_column: str = 'is_fraud'


@dataclass
class InteractionResult:
    """Container for feature interaction analysis results."""
    
    # Pairwise interactions
    pairwise_interactions: Dict[Tuple[str, str], float]
    pairwise_significance: Dict[Tuple[str, str], float]
    
    # Higher-order interactions
    higher_order_interactions: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    
    # H-statistics
    h_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Network representation
    interaction_network: Optional[nx.Graph] = None
    
    # Feature engineering suggestions
    feature_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Fraud patterns (if applicable)
    fraud_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Summary statistics
    summary_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionPattern:
    """Container for specific interaction patterns."""
    
    pattern_type: str
    features: List[str]
    strength: float
    description: str
    statistical_significance: float
    business_interpretation: str = ""
    suggested_action: str = ""


class FeatureInteractionAnalyzer:
    """
    Comprehensive feature interaction analyzer with H-statistic calculation,
    network visualization, and pattern detection.
    """
    
    def __init__(self, config: InteractionConfig = None):
        """
        Initialize feature interaction analyzer.
        
        Args:
            config: Interaction analysis configuration
        """
        self.config = config or InteractionConfig()
        self.feature_names = None
        self.target_name = None
        self.interaction_cache = {}
        
    def analyze_interactions(self, X: np.ndarray, y: np.ndarray,
                           model: Any = None,
                           feature_names: List[str] = None,
                           target_name: str = None) -> InteractionResult:
        """
        Perform comprehensive feature interaction analysis.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Trained model (optional, for H-statistics)
            feature_names: Feature names
            target_name: Target name
            
        Returns:
            InteractionResult object
        """
        logger.info("Starting comprehensive feature interaction analysis")
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.target_name = target_name or 'target'
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=self.feature_names)
        df[self.target_name] = y
        
        # Sample data for efficiency if needed
        if len(df) > self.config.max_samples_h_statistic:
            df_sample = df.sample(n=self.config.max_samples_h_statistic, random_state=42)
        else:
            df_sample = df.copy()
        
        # Detect pairwise interactions
        logger.info("Detecting pairwise interactions")
        pairwise_interactions, pairwise_significance = self._detect_pairwise_interactions(
            df_sample, model
        )
        
        # Detect higher-order interactions if requested
        higher_order_interactions = {}
        if self.config.max_interaction_order > 2:
            logger.info("Detecting higher-order interactions")
            higher_order_interactions = self._detect_higher_order_interactions(
                df_sample, model
            )
        
        # Calculate H-statistics if model provided
        h_statistics = {}
        if model is not None:
            logger.info("Calculating H-statistics")
            h_statistics = self._calculate_h_statistics(df_sample, model)
        
        # Create interaction network
        logger.info("Creating interaction network")
        interaction_network = self._create_interaction_network(
            pairwise_interactions, higher_order_interactions
        )
        
        # Generate feature engineering suggestions
        feature_suggestions = []
        if self.config.enable_feature_suggestions:
            feature_suggestions = self._generate_feature_suggestions(
                pairwise_interactions, higher_order_interactions
            )
        
        # Detect fraud patterns if enabled
        fraud_patterns = {}
        if self.config.enable_fraud_patterns and self.config.fraud_column in df.columns:
            fraud_patterns = self._detect_fraud_patterns(df, pairwise_interactions)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(
            pairwise_interactions, higher_order_interactions, h_statistics
        )
        
        result = InteractionResult(
            pairwise_interactions=pairwise_interactions,
            pairwise_significance=pairwise_significance,
            higher_order_interactions=higher_order_interactions,
            h_statistics=h_statistics,
            interaction_network=interaction_network,
            feature_suggestions=feature_suggestions,
            fraud_patterns=fraud_patterns,
            summary_stats=summary_stats
        )
        
        logger.info("Feature interaction analysis completed")
        return result
    
    def visualize_interaction_network(self, result: InteractionResult,
                                    output_path: str = None,
                                    interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Create interaction network visualization.
        
        Args:
            result: InteractionResult object
            output_path: Optional path to save visualization
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if result.interaction_network is None:
            logger.warning("No interaction network available for visualization")
            return None
        
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_network(result, output_path)
        else:
            return self._create_static_network(result, output_path)
    
    def visualize_interaction_heatmap(self, result: InteractionResult,
                                    output_path: str = None) -> plt.Figure:
        """
        Create interaction strength heatmap.
        
        Args:
            result: InteractionResult object
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Create interaction matrix
        n_features = len(self.feature_names)
        interaction_matrix = np.zeros((n_features, n_features))
        
        for (feat1, feat2), strength in result.pairwise_interactions.items():
            idx1 = self.feature_names.index(feat1)
            idx2 = self.feature_names.index(feat2)
            interaction_matrix[idx1, idx2] = strength
            interaction_matrix[idx2, idx1] = strength
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))
        
        sns.heatmap(
            interaction_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            square=True,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cbar_kws={'label': 'Interaction Strength'}
        )
        
        plt.title('Feature Interaction Strength Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Interaction heatmap saved to {output_path}")
        
        return plt.gcf()
    
    def identify_interaction_patterns(self, result: InteractionResult,
                                    pattern_types: List[str] = None) -> List[InteractionPattern]:
        """
        Identify specific interaction patterns.
        
        Args:
            result: InteractionResult object
            pattern_types: Types of patterns to detect
            
        Returns:
            List of InteractionPattern objects
        """
        pattern_types = pattern_types or [
            'synergistic', 'competitive', 'redundant', 'complementary'
        ]
        
        patterns = []
        
        # Analyze pairwise interactions
        for (feat1, feat2), strength in result.pairwise_interactions.items():
            if strength > self.config.min_interaction_strength:
                # Determine pattern type based on strength and other characteristics
                pattern_type = self._classify_interaction_pattern(
                    feat1, feat2, strength, result
                )
                
                significance = result.pairwise_significance.get((feat1, feat2), 1.0)
                
                pattern = InteractionPattern(
                    pattern_type=pattern_type,
                    features=[feat1, feat2],
                    strength=strength,
                    description=f"{pattern_type.title()} interaction between {feat1} and {feat2}",
                    statistical_significance=significance,
                    business_interpretation=self._generate_business_interpretation(
                        feat1, feat2, pattern_type, strength
                    ),
                    suggested_action=self._generate_action_suggestion(
                        feat1, feat2, pattern_type, strength
                    )
                )
                
                patterns.append(pattern)
        
        # Sort by strength
        patterns.sort(key=lambda x: x.strength, reverse=True)
        
        return patterns
    
    def _detect_pairwise_interactions(self, df: pd.DataFrame, 
                                    model: Any = None) -> Tuple[Dict[Tuple[str, str], float], 
                                                              Dict[Tuple[str, str], float]]:
        """Detect pairwise feature interactions."""
        interactions = {}
        significance = {}
        
        feature_cols = [col for col in df.columns if col != self.target_name]
        
        # Method 1: Correlation-based interaction detection
        for feat1, feat2 in combinations(feature_cols, 2):
            # Calculate interaction effect using correlation
            interaction_strength = self._calculate_correlation_interaction(
                df, feat1, feat2, self.target_name
            )
            
            # Statistical significance test
            p_value = self._test_interaction_significance(df, feat1, feat2, self.target_name)
            
            interactions[(feat1, feat2)] = interaction_strength
            significance[(feat1, feat2)] = p_value
        
        # Method 2: Model-based interaction detection (if model provided)
        if model is not None:
            model_interactions = self._calculate_model_based_interactions(df, model)
            
            # Combine with correlation-based interactions
            for (feat1, feat2), strength in model_interactions.items():
                if (feat1, feat2) in interactions:
                    # Average the two measures
                    interactions[(feat1, feat2)] = (interactions[(feat1, feat2)] + strength) / 2
                else:
                    interactions[(feat1, feat2)] = strength
        
        # Filter by minimum strength
        filtered_interactions = {
            k: v for k, v in interactions.items() 
            if abs(v) >= self.config.min_interaction_strength
        }
        
        filtered_significance = {
            k: v for k, v in significance.items() 
            if k in filtered_interactions
        }
        
        return filtered_interactions, filtered_significance
    
    def _detect_higher_order_interactions(self, df: pd.DataFrame, 
                                        model: Any = None) -> Dict[Tuple[str, ...], float]:
        """Detect higher-order feature interactions."""
        interactions = {}
        feature_cols = [col for col in df.columns if col != self.target_name]
        
        # Limit to manageable number of features for higher-order analysis
        if len(feature_cols) > 20:
            # Select top features based on individual correlation with target
            correlations = df[feature_cols].corrwith(df[self.target_name]).abs()
            top_features = correlations.nlargest(20).index.tolist()
        else:
            top_features = feature_cols
        
        # Three-way interactions
        if self.config.max_interaction_order >= 3:
            for feat1, feat2, feat3 in combinations(top_features, 3):
                interaction_strength = self._calculate_three_way_interaction(
                    df, feat1, feat2, feat3, self.target_name
                )
                
                if abs(interaction_strength) >= self.config.min_interaction_strength:
                    interactions[(feat1, feat2, feat3)] = interaction_strength
        
        return interactions
    
    def _calculate_h_statistics(self, df: pd.DataFrame, model: Any) -> Dict[str, Dict[str, float]]:
        """Calculate H-statistics for feature interactions."""
        if not SKLEARN_INSPECTION_AVAILABLE:
            logger.warning("sklearn.inspection not available. H-statistics calculation skipped.")
            return {}
        
        h_stats = {}
        feature_cols = [col for col in df.columns if col != self.target_name]
        X = df[feature_cols].values
        
        try:
            # Calculate H-statistics for each feature pair
            for i, feat1 in enumerate(feature_cols):
                h_stats[feat1] = {}
                
                for j, feat2 in enumerate(feature_cols):
                    if i != j:
                        h_stat = self._calculate_friedman_h_statistic(
                            X, model, i, j
                        )
                        h_stats[feat1][feat2] = h_stat
            
        except Exception as e:
            logger.warning(f"H-statistic calculation failed: {e}")
            return {}
        
        return h_stats
    
    def _calculate_friedman_h_statistic(self, X: np.ndarray, model: Any, 
                                      feature1_idx: int, feature2_idx: int) -> float:
        """Calculate Friedman's H-statistic for feature interaction."""
        try:
            # Sample points for H-statistic calculation
            n_samples = min(100, len(X))
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[sample_indices]
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_sample)[:, 1]
            else:
                predictions = model.predict(X_sample)
            
            # Calculate partial dependence for individual features
            pd1 = self._calculate_partial_dependence_1d(model, X_sample, feature1_idx)
            pd2 = self._calculate_partial_dependence_1d(model, X_sample, feature2_idx)
            
            # Calculate joint partial dependence
            pd_joint = self._calculate_partial_dependence_2d(
                model, X_sample, feature1_idx, feature2_idx
            )
            
            # H-statistic formula: variance of (PD_joint - PD1 - PD2)
            interaction_effect = pd_joint - pd1 - pd2
            h_statistic = np.var(interaction_effect)
            
            return h_statistic
            
        except Exception as e:
            logger.debug(f"H-statistic calculation failed for features {feature1_idx}, {feature2_idx}: {e}")
            return 0.0
    
    def _calculate_partial_dependence_1d(self, model: Any, X: np.ndarray, 
                                       feature_idx: int) -> np.ndarray:
        """Calculate 1D partial dependence."""
        feature_values = X[:, feature_idx]
        grid_values = np.linspace(feature_values.min(), feature_values.max(), 10)
        
        pd_values = np.zeros(len(X))
        
        for i, x in enumerate(X):
            predictions = []
            for grid_val in grid_values:
                x_modified = x.copy()
                x_modified[feature_idx] = grid_val
                
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba([x_modified])[0, 1]
                else:
                    pred = model.predict([x_modified])[0]
                
                predictions.append(pred)
            
            pd_values[i] = np.mean(predictions)
        
        return pd_values
    
    def _calculate_partial_dependence_2d(self, model: Any, X: np.ndarray,
                                       feature1_idx: int, feature2_idx: int) -> np.ndarray:
        """Calculate 2D partial dependence."""
        feature1_values = X[:, feature1_idx]
        feature2_values = X[:, feature2_idx]
        
        grid1 = np.linspace(feature1_values.min(), feature1_values.max(), 5)
        grid2 = np.linspace(feature2_values.min(), feature2_values.max(), 5)
        
        pd_values = np.zeros(len(X))
        
        for i, x in enumerate(X):
            predictions = []
            for val1 in grid1:
                for val2 in grid2:
                    x_modified = x.copy()
                    x_modified[feature1_idx] = val1
                    x_modified[feature2_idx] = val2
                    
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba([x_modified])[0, 1]
                    else:
                        pred = model.predict([x_modified])[0]
                    
                    predictions.append(pred)
            
            pd_values[i] = np.mean(predictions)
        
        return pd_values
    
    def _calculate_correlation_interaction(self, df: pd.DataFrame, 
                                         feat1: str, feat2: str, target: str) -> float:
        """Calculate interaction strength using correlation analysis."""
        # Method 1: Correlation of product with target
        product_corr = df[feat1] * df[feat2]
        interaction_corr = product_corr.corr(df[target])
        
        # Method 2: Conditional correlation
        # Split data by median of feat1 and calculate correlation of feat2 with target
        median_feat1 = df[feat1].median()
        high_feat1 = df[df[feat1] >= median_feat1]
        low_feat1 = df[df[feat1] < median_feat1]
        
        if len(high_feat1) > 10 and len(low_feat1) > 10:
            corr_high = high_feat1[feat2].corr(high_feat1[target])
            corr_low = low_feat1[feat2].corr(low_feat1[target])
            conditional_interaction = abs(corr_high - corr_low)
        else:
            conditional_interaction = 0.0
        
        # Combine measures
        interaction_strength = (abs(interaction_corr) + conditional_interaction) / 2
        
        return interaction_strength
    
    def _test_interaction_significance(self, df: pd.DataFrame,
                                     feat1: str, feat2: str, target: str) -> float:
        """Test statistical significance of interaction."""
        try:
            # Create interaction term
            interaction_term = df[feat1] * df[feat2]
            
            # Fit models with and without interaction
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            
            X_base = df[[feat1, feat2]].values
            X_interaction = np.column_stack([X_base, interaction_term.values])
            y = df[target].values
            
            # Base model
            model_base = LinearRegression()
            model_base.fit(X_base, y)
            mse_base = mean_squared_error(y, model_base.predict(X_base))
            
            # Model with interaction
            model_interaction = LinearRegression()
            model_interaction.fit(X_interaction, y)
            mse_interaction = mean_squared_error(y, model_interaction.predict(X_interaction))
            
            # F-test for model comparison
            n = len(y)
            f_statistic = ((mse_base - mse_interaction) / 1) / (mse_interaction / (n - 4))
            p_value = 1 - stats.f.cdf(f_statistic, 1, n - 4)
            
            return p_value
            
        except Exception as e:
            logger.debug(f"Significance test failed for {feat1}, {feat2}: {e}")
            return 1.0
    
    def _calculate_model_based_interactions(self, df: pd.DataFrame, 
                                          model: Any) -> Dict[Tuple[str, str], float]:
        """Calculate interactions using model predictions."""
        interactions = {}
        feature_cols = [col for col in df.columns if col != self.target_name]
        
        try:
            X = df[feature_cols].values
            
            # Get base predictions
            if hasattr(model, 'predict_proba'):
                base_predictions = model.predict_proba(X)[:, 1]
            else:
                base_predictions = model.predict(X)
            
            # For each feature pair, test interaction effect
            for feat1, feat2 in combinations(feature_cols, 2):
                feat1_idx = feature_cols.index(feat1)
                feat2_idx = feature_cols.index(feat2)
                
                # Calculate interaction effect by varying both features
                interaction_effect = self._calculate_interaction_effect(
                    model, X, feat1_idx, feat2_idx, base_predictions
                )
                
                interactions[(feat1, feat2)] = interaction_effect
        
        except Exception as e:
            logger.warning(f"Model-based interaction calculation failed: {e}")
        
        return interactions
    
    def _calculate_interaction_effect(self, model: Any, X: np.ndarray,
                                    feat1_idx: int, feat2_idx: int,
                                    base_predictions: np.ndarray) -> float:
        """Calculate interaction effect between two features."""
        try:
            # Sample subset for efficiency
            n_samples = min(100, len(X))
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[sample_indices]
            base_sample = base_predictions[sample_indices]
            
            # Calculate effect of changing both features together vs individually
            interaction_effects = []
            
            for i, x in enumerate(X_sample):
                # Original prediction
                original_pred = base_sample[i]
                
                # Perturb feature 1
                x_feat1 = x.copy()
                feat1_range = X[:, feat1_idx].std()
                x_feat1[feat1_idx] += feat1_range * 0.1
                
                if hasattr(model, 'predict_proba'):
                    pred_feat1 = model.predict_proba([x_feat1])[0, 1]
                else:
                    pred_feat1 = model.predict([x_feat1])[0]
                
                # Perturb feature 2
                x_feat2 = x.copy()
                feat2_range = X[:, feat2_idx].std()
                x_feat2[feat2_idx] += feat2_range * 0.1
                
                if hasattr(model, 'predict_proba'):
                    pred_feat2 = model.predict_proba([x_feat2])[0, 1]
                else:
                    pred_feat2 = model.predict([x_feat2])[0]
                
                # Perturb both features
                x_both = x.copy()
                x_both[feat1_idx] += feat1_range * 0.1
                x_both[feat2_idx] += feat2_range * 0.1
                
                if hasattr(model, 'predict_proba'):
                    pred_both = model.predict_proba([x_both])[0, 1]
                else:
                    pred_both = model.predict([x_both])[0]
                
                # Interaction effect = combined effect - sum of individual effects
                individual_effects = (pred_feat1 - original_pred) + (pred_feat2 - original_pred)
                combined_effect = pred_both - original_pred
                interaction_effect = combined_effect - individual_effects
                
                interaction_effects.append(abs(interaction_effect))
            
            return np.mean(interaction_effects)
            
        except Exception as e:
            logger.debug(f"Interaction effect calculation failed: {e}")
            return 0.0
    
    def _calculate_three_way_interaction(self, df: pd.DataFrame,
                                       feat1: str, feat2: str, feat3: str,
                                       target: str) -> float:
        """Calculate three-way interaction strength."""
        try:
            # Use mutual information for three-way interaction
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Create three-way interaction term
            interaction_term = df[feat1] * df[feat2] * df[feat3]
            
            # Calculate mutual information
            if df[target].dtype in ['int64', 'bool']:
                mi_score = mutual_info_classif([interaction_term], df[target])[0]
            else:
                mi_score = mutual_info_regression([interaction_term], df[target])[0]
            
            return mi_score
            
        except Exception as e:
            logger.debug(f"Three-way interaction calculation failed: {e}")
            return 0.0
    
    def _create_interaction_network(self, pairwise_interactions: Dict[Tuple[str, str], float],
                                  higher_order_interactions: Dict[Tuple[str, ...], float]) -> nx.Graph:
        """Create network representation of feature interactions."""
        G = nx.Graph()
        
        # Add nodes (features)
        all_features = set()
        for (feat1, feat2) in pairwise_interactions.keys():
            all_features.add(feat1)
            all_features.add(feat2)
        
        # Limit nodes if too many
        if len(all_features) > self.config.max_network_nodes:
            # Select top features by total interaction strength
            feature_totals = {}
            for feat in all_features:
                total_strength = sum(
                    strength for (f1, f2), strength in pairwise_interactions.items()
                    if feat in (f1, f2)
                )
                feature_totals[feat] = total_strength
            
            sorted_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)
            all_features = set([feat for feat, _ in sorted_features[:self.config.max_network_nodes]])
        
        G.add_nodes_from(all_features)
        
        # Add edges (interactions)
        for (feat1, feat2), strength in pairwise_interactions.items():
            if (feat1 in all_features and feat2 in all_features and 
                abs(strength) >= self.config.min_network_edge_weight):
                G.add_edge(feat1, feat2, weight=abs(strength), interaction_strength=strength)
        
        return G
    
    def _generate_feature_suggestions(self, pairwise_interactions: Dict[Tuple[str, str], float],
                                    higher_order_interactions: Dict[Tuple[str, ...], float]) -> List[Dict[str, Any]]:
        """Generate feature engineering suggestions based on interactions."""
        suggestions = []
        
        # Pairwise interaction features
        for (feat1, feat2), strength in pairwise_interactions.items():
            if strength >= self.config.min_suggestion_strength:
                suggestions.append({
                    'type': 'product',
                    'features': [feat1, feat2],
                    'suggested_name': f'{feat1}_x_{feat2}',
                    'strength': strength,
                    'description': f'Product of {feat1} and {feat2}',
                    'formula': f'{feat1} * {feat2}'
                })
                
                suggestions.append({
                    'type': 'ratio',
                    'features': [feat1, feat2],
                    'suggested_name': f'{feat1}_div_{feat2}',
                    'strength': strength,
                    'description': f'Ratio of {feat1} to {feat2}',
                    'formula': f'{feat1} / ({feat2} + 1e-8)'
                })
        
        # Higher-order interaction features
        for features, strength in higher_order_interactions.items():
            if strength >= self.config.min_suggestion_strength:
                suggestions.append({
                    'type': 'higher_order_product',
                    'features': list(features),
                    'suggested_name': '_x_'.join(features),
                    'strength': strength,
                    'description': f'Product of {", ".join(features)}',
                    'formula': ' * '.join(features)
                })
        
        # Sort by strength
        suggestions.sort(key=lambda x: x['strength'], reverse=True)
        
        return suggestions[:20]  # Limit to top 20 suggestions
    
    def _detect_fraud_patterns(self, df: pd.DataFrame,
                             pairwise_interactions: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        """Detect fraud-specific interaction patterns."""
        if self.config.fraud_column not in df.columns:
            return {}
        
        fraud_patterns = {
            'fraud_specific_interactions': [],
            'fraud_correlation_patterns': {},
            'fraud_interaction_network': None
        }
        
        try:
            # Analyze interactions specific to fraud cases
            fraud_df = df[df[self.config.fraud_column] == 1]
            normal_df = df[df[self.config.fraud_column] == 0]
            
            if len(fraud_df) < 10 or len(normal_df) < 10:
                return fraud_patterns
            
            # Find interactions that are stronger in fraud cases
            for (feat1, feat2), overall_strength in pairwise_interactions.items():
                if feat1 != self.config.fraud_column and feat2 != self.config.fraud_column:
                    # Calculate interaction in fraud vs normal cases
                    fraud_corr = (fraud_df[feat1] * fraud_df[feat2]).corr(fraud_df[self.config.fraud_column])
                    normal_corr = (normal_df[feat1] * normal_df[feat2]).corr(normal_df[self.config.fraud_column])
                    
                    # If interaction is significantly different in fraud cases
                    if abs(fraud_corr - normal_corr) > 0.1:
                        fraud_patterns['fraud_specific_interactions'].append({
                            'features': [feat1, feat2],
                            'fraud_correlation': fraud_corr,
                            'normal_correlation': normal_corr,
                            'difference': fraud_corr - normal_corr,
                            'overall_strength': overall_strength
                        })
            
            # Sort by difference
            fraud_patterns['fraud_specific_interactions'].sort(
                key=lambda x: abs(x['difference']), reverse=True
            )
            
        except Exception as e:
            logger.warning(f"Fraud pattern detection failed: {e}")
        
        return fraud_patterns
    
    def _calculate_summary_statistics(self, pairwise_interactions: Dict[Tuple[str, str], float],
                                    higher_order_interactions: Dict[Tuple[str, ...], float],
                                    h_statistics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate summary statistics for interactions."""
        if not pairwise_interactions:
            return {}
        
        interaction_strengths = list(pairwise_interactions.values())
        
        stats = {
            'total_pairwise_interactions': len(pairwise_interactions),
            'total_higher_order_interactions': len(higher_order_interactions),
            'mean_interaction_strength': np.mean(interaction_strengths),
            'max_interaction_strength': np.max(interaction_strengths),
            'min_interaction_strength': np.min(interaction_strengths),
            'std_interaction_strength': np.std(interaction_strengths),
            'strong_interactions_count': sum(1 for s in interaction_strengths if s > 0.2),
            'weak_interactions_count': sum(1 for s in interaction_strengths if s < 0.05)
        }
        
        # Top interactions
        sorted_interactions = sorted(pairwise_interactions.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        stats['top_interactions'] = sorted_interactions[:10]
        
        # H-statistic summary
        if h_statistics:
            all_h_values = []
            for feat1_stats in h_statistics.values():
                all_h_values.extend(feat1_stats.values())
            
            if all_h_values:
                stats['h_statistic_mean'] = np.mean(all_h_values)
                stats['h_statistic_max'] = np.max(all_h_values)
        
        return stats
    
    def _create_interactive_network(self, result: InteractionResult, output_path: str = None) -> go.Figure:
        """Create interactive network visualization using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Creating static network instead.")
            return self._create_static_network(result, output_path)
        
        G = result.interaction_network
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        # Node sizes based on degree
        node_sizes = [G.degree(node) * 10 + 10 for node in G.nodes()]
        
        # Edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_sizes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Node Degree")
            )
        ))
        
        fig.update_layout(
            title='Feature Interaction Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Feature interaction network - node size indicates number of interactions",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Interactive network saved to {output_path}")
        
        return fig
    
    def _create_static_network(self, result: InteractionResult, output_path: str = None) -> plt.Figure:
        """Create static network visualization using matplotlib."""
        G = result.interaction_network
        
        plt.figure(figsize=(12, 8))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw edges with weights
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in edge_weights], 
                              alpha=0.6, edge_color='gray')
        
        # Draw nodes
        node_sizes = [G.degree(node) * 100 + 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Feature Interaction Network', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Static network saved to {output_path}")
        
        return plt.gcf()
    
    def _classify_interaction_pattern(self, feat1: str, feat2: str, strength: float,
                                    result: InteractionResult) -> str:
        """Classify the type of interaction pattern."""
        if strength > 0.3:
            return 'synergistic'
        elif strength > 0.1:
            return 'complementary'
        elif strength < -0.1:
            return 'competitive'
        else:
            return 'weak'
    
    def _generate_business_interpretation(self, feat1: str, feat2: str, 
                                        pattern_type: str, strength: float) -> str:
        """Generate business interpretation of interaction."""
        interpretations = {
            'synergistic': f"Strong positive interaction: {feat1} and {feat2} work together to significantly impact the outcome",
            'complementary': f"Moderate interaction: {feat1} and {feat2} complement each other's effects",
            'competitive': f"Negative interaction: {feat1} and {feat2} may work against each other",
            'weak': f"Weak interaction: {feat1} and {feat2} have minimal combined effect"
        }
        
        return interpretations.get(pattern_type, "Unknown interaction pattern")
    
    def _generate_action_suggestion(self, feat1: str, feat2: str, 
                                  pattern_type: str, strength: float) -> str:
        """Generate actionable suggestions based on interaction."""
        suggestions = {
            'synergistic': f"Consider creating interaction features combining {feat1} and {feat2}",
            'complementary': f"Monitor both {feat1} and {feat2} together for decision making",
            'competitive': f"Investigate potential data quality issues between {feat1} and {feat2}",
            'weak': f"Low priority for feature engineering involving {feat1} and {feat2}"
        }
        
        return suggestions.get(pattern_type, "No specific action recommended")


def create_feature_interaction_analyzer(config_dict: Dict[str, Any] = None) -> FeatureInteractionAnalyzer:
    """
    Factory function to create feature interaction analyzer.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        FeatureInteractionAnalyzer instance
    """
    config = InteractionConfig(**config_dict) if config_dict else InteractionConfig()
    return FeatureInteractionAnalyzer(config)