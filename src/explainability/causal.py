"""Causal inference and causal explanations for machine learning models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')


class CausalGraph:
    """Represent and manipulate causal graphs."""
    
    def __init__(self):
        """Initialize empty causal graph."""
        self.graph = nx.DiGraph()
        self.edge_strengths = {}
        
    def add_node(self, node: str, node_type: str = 'feature'):
        """Add a node to the causal graph.
        
        Args:
            node: Node name
            node_type: Type of node ('feature', 'outcome', 'confounder')
        """
        self.graph.add_node(node, node_type=node_type)
        
    def add_edge(self, from_node: str, to_node: str, strength: float = 1.0,
                confidence: float = 1.0):
        """Add a causal edge to the graph.
        
        Args:
            from_node: Source node
            to_node: Target node
            strength: Strength of causal relationship
            confidence: Confidence in the relationship
        """
        self.graph.add_edge(from_node, to_node)
        self.edge_strengths[(from_node, to_node)] = {
            'strength': strength,
            'confidence': confidence
        }
        
    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes (direct causes) of a node."""
        return list(self.graph.predecessors(node))
        
    def get_children(self, node: str) -> List[str]:
        """Get child nodes (direct effects) of a node."""
        return list(self.graph.successors(node))
        
    def get_ancestors(self, node: str) -> List[str]:
        """Get all ancestor nodes (all causes) of a node."""
        return list(nx.ancestors(self.graph, node))
        
    def get_descendants(self, node: str) -> List[str]:
        """Get all descendant nodes (all effects) of a node."""
        return list(nx.descendants(self.graph, node))
        
    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find backdoor paths between treatment and outcome."""
        backdoor_paths = []
        
        # Remove direct edge from treatment to outcome temporarily
        if self.graph.has_edge(treatment, outcome):
            self.graph.remove_edge(treatment, outcome)
            paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
            self.graph.add_edge(treatment, outcome)
        else:
            paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
        
        # Filter for backdoor paths (paths that start with an arrow into treatment)
        for path in paths:
            if len(path) > 2:  # Must have at least one intermediate node
                # Check if first edge is into treatment (backdoor criterion)
                if path[1] in self.get_parents(treatment):
                    backdoor_paths.append(path)
                    
        return backdoor_paths
        
    def visualize(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize the causal graph."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Position nodes
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Color nodes by type
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('node_type', 'feature')
            if node_type == 'outcome':
                node_colors.append('lightcoral')
            elif node_type == 'confounder':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgreen')
        
        # Draw graph
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors,
                node_size=1500, font_size=10, font_weight='bold',
                edge_color='gray', arrows=True, arrowsize=20, ax=ax)
        
        # Add edge strength labels
        edge_labels = {}
        for edge, props in self.edge_strengths.items():
            edge_labels[edge] = f"{props['strength']:.2f}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, ax=ax)
        
        ax.set_title('Causal Graph')
        ax.axis('off')
        
        return fig


class CausalInferenceEngine:
    """Engine for various causal inference methods."""
    
    def __init__(self, causal_graph: Optional[CausalGraph] = None):
        """Initialize causal inference engine.
        
        Args:
            causal_graph: Known causal graph (optional)
        """
        self.causal_graph = causal_graph
        
    def estimate_ate(self, data: pd.DataFrame, treatment: str, outcome: str,
                    confounders: Optional[List[str]] = None,
                    method: str = 'regression') -> Dict[str, Any]:
        """Estimate Average Treatment Effect (ATE).
        
        Args:
            data: Dataset
            treatment: Treatment variable name
            outcome: Outcome variable name
            confounders: List of confounding variables
            method: Estimation method ('regression', 'matching', 'ipw')
            
        Returns:
            Dictionary with ATE estimate and statistics
        """
        if confounders is None:
            confounders = []
        
        if method == 'regression':
            return self._regression_ate(data, treatment, outcome, confounders)
        elif method == 'matching':
            return self._matching_ate(data, treatment, outcome, confounders)
        elif method == 'ipw':
            return self._ipw_ate(data, treatment, outcome, confounders)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _regression_ate(self, data: pd.DataFrame, treatment: str, outcome: str,
                       confounders: List[str]) -> Dict[str, Any]:
        """Estimate ATE using regression adjustment."""
        # Prepare features
        features = confounders + [treatment]
        X = data[features]
        y = data[outcome]
        
        # Fit regression model
        if data[outcome].dtype in ['int64', 'bool']:
            model = LogisticRegression()
        else:
            model = LinearRegression()
        
        model.fit(X, y)
        
        # Estimate ATE by comparing predictions with treatment=1 vs treatment=0
        data_treated = data.copy()
        data_treated[treatment] = 1
        
        data_control = data.copy()
        data_control[treatment] = 0
        
        if hasattr(model, 'predict_proba'):
            pred_treated = model.predict_proba(data_treated[features])[:, 1]
            pred_control = model.predict_proba(data_control[features])[:, 1]
        else:
            pred_treated = model.predict(data_treated[features])
            pred_control = model.predict(data_control[features])
        
        ate = np.mean(pred_treated - pred_control)
        
        # Compute confidence interval (simplified)
        se = np.std(pred_treated - pred_control) / np.sqrt(len(data))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'regression',
            'n_treated': np.sum(data[treatment]),
            'n_control': np.sum(1 - data[treatment])
        }
    
    def _matching_ate(self, data: pd.DataFrame, treatment: str, outcome: str,
                     confounders: List[str]) -> Dict[str, Any]:
        """Estimate ATE using propensity score matching (simplified)."""
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate propensity scores
        X_conf = data[confounders]
        t = data[treatment]
        
        ps_model = LogisticRegression()
        ps_model.fit(X_conf, t)
        propensity_scores = ps_model.predict_proba(X_conf)[:, 1]
        
        # Match treated and control units
        treated_mask = data[treatment] == 1
        control_mask = data[treatment] == 0
        
        treated_ps = propensity_scores[treated_mask]
        control_ps = propensity_scores[control_mask]
        treated_outcomes = data[outcome][treated_mask]
        control_outcomes = data[outcome][control_mask]
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control_ps.reshape(-1, 1))
        
        matches = nn.kneighbors(treated_ps.reshape(-1, 1), return_distance=False)
        matched_control_outcomes = control_outcomes.iloc[matches.flatten()]
        
        # Compute ATE
        ate = np.mean(treated_outcomes - matched_control_outcomes)
        se = np.std(treated_outcomes - matched_control_outcomes) / np.sqrt(len(treated_outcomes))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'matching',
            'n_treated': len(treated_outcomes),
            'n_control': len(control_outcomes)
        }
    
    def _ipw_ate(self, data: pd.DataFrame, treatment: str, outcome: str,
                confounders: List[str]) -> Dict[str, Any]:
        """Estimate ATE using Inverse Probability Weighting."""
        # Estimate propensity scores
        X_conf = data[confounders]
        t = data[treatment]
        y = data[outcome]
        
        ps_model = LogisticRegression()
        ps_model.fit(X_conf, t)
        propensity_scores = ps_model.predict_proba(X_conf)[:, 1]
        
        # Compute IPW weights
        weights = np.where(t == 1, 1/propensity_scores, 1/(1-propensity_scores))
        
        # Compute weighted outcomes
        weighted_treated = np.sum(weights * t * y) / np.sum(weights * t)
        weighted_control = np.sum(weights * (1-t) * y) / np.sum(weights * (1-t))
        
        ate = weighted_treated - weighted_control
        
        # Simplified standard error
        se = np.sqrt(np.var(weights * (t * y - (1-t) * y)) / len(data))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'ipw',
            'n_treated': np.sum(t),
            'n_control': np.sum(1-t)
        }
    
    def discover_causal_structure(self, data: pd.DataFrame, 
                                 method: str = 'pc_algorithm') -> CausalGraph:
        """Discover causal structure from data.
        
        Args:
            data: Dataset
            method: Structure learning method
            
        Returns:
            Learned causal graph
        """
        if method == 'pc_algorithm':
            return self._pc_algorithm(data)
        elif method == 'correlation_based':
            return self._correlation_based_structure(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _pc_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """Simplified PC algorithm for causal discovery."""
        variables = data.columns.tolist()
        graph = CausalGraph()
        
        # Add all variables as nodes
        for var in variables:
            graph.add_node(var)
        
        # Start with complete graph and remove edges based on conditional independence
        significance_level = 0.05
        
        for var1, var2 in combinations(variables, 2):
            # Test marginal independence
            if data[var1].dtype in ['int64', 'bool'] and data[var2].dtype in ['int64', 'bool']:
                # Chi-square test for categorical variables
                contingency_table = pd.crosstab(data[var1], data[var2])
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
            else:
                # Correlation test for continuous variables
                corr, p_value = stats.pearsonr(data[var1], data[var2])
            
            # Add edge if variables are dependent
            if p_value < significance_level:
                # Determine direction based on variance (simplified heuristic)
                if np.var(data[var1]) > np.var(data[var2]):
                    graph.add_edge(var1, var2, strength=abs(corr) if 'corr' in locals() else 0.5)
                else:
                    graph.add_edge(var2, var1, strength=abs(corr) if 'corr' in locals() else 0.5)
        
        return graph
    
    def _correlation_based_structure(self, data: pd.DataFrame) -> CausalGraph:
        """Simple correlation-based causal structure discovery."""
        corr_matrix = data.corr().abs()
        threshold = 0.3
        
        graph = CausalGraph()
        variables = data.columns.tolist()
        
        # Add nodes
        for var in variables:
            graph.add_node(var)
        
        # Add edges based on correlation threshold
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i+1:], i+1):
                correlation = corr_matrix.loc[var1, var2]
                if correlation > threshold:
                    # Simple heuristic: variable with higher variance causes the other
                    if np.var(data[var1]) > np.var(data[var2]):
                        graph.add_edge(var1, var2, strength=correlation)
                    else:
                        graph.add_edge(var2, var1, strength=correlation)
        
        return graph


class CausalAnalyzer:
    """High-level causal analysis for machine learning interpretability."""
    
    def __init__(self, model, data: pd.DataFrame):
        """Initialize causal analyzer.
        
        Args:
            model: Trained ML model
            data: Dataset used for training
        """
        self.model = model
        self.data = data
        self.causal_engine = CausalInferenceEngine()
        
    def analyze_feature_causality(self, target_feature: str, 
                                 other_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze causal relationships for a specific feature.
        
        Args:
            target_feature: Feature to analyze
            other_features: Other features to consider (if None, use all)
            
        Returns:
            Dictionary with causal analysis results
        """
        if other_features is None:
            other_features = [col for col in self.data.columns if col != target_feature]
        
        # Discover causal structure
        subset_data = self.data[[target_feature] + other_features]
        causal_graph = self.causal_engine.discover_causal_structure(subset_data)
        
        # Analyze causal relationships
        parents = causal_graph.get_parents(target_feature)
        children = causal_graph.get_children(target_feature)
        
        # Estimate causal effects
        causal_effects = {}
        for parent in parents:
            try:
                confounders = [f for f in other_features if f != parent and f != target_feature]
                effect = self.causal_engine.estimate_ate(
                    self.data, parent, target_feature, confounders[:5]  # Limit confounders
                )
                causal_effects[parent] = effect
            except Exception as e:
                causal_effects[parent] = {'error': str(e)}
        
        return {
            'target_feature': target_feature,
            'causal_parents': parents,
            'causal_children': children,
            'causal_effects': causal_effects,
            'causal_graph': causal_graph
        }
    
    def intervention_analysis(self, intervention_feature: str, 
                            intervention_values: List[Any],
                            target_outcomes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze the effect of interventions on model predictions.
        
        Args:
            intervention_feature: Feature to intervene on
            intervention_values: Values to set the feature to
            target_outcomes: Target variables to measure effects on
            
        Returns:
            Dictionary with intervention analysis results
        """
        if target_outcomes is None:
            # Use model prediction as target
            target_outcomes = ['model_prediction']
        
        results = {}
        
        for value in intervention_values:
            # Create interventional dataset
            interventional_data = self.data.copy()
            interventional_data[intervention_feature] = value
            
            # Get model predictions
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(interventional_data)
                if predictions.shape[1] == 2:
                    predictions = predictions[:, 1]  # Binary classification
            else:
                predictions = self.model.predict(interventional_data)
            
            # Compare with original predictions
            original_predictions = self.model.predict(self.data)
            if hasattr(self.model, 'predict_proba') and len(original_predictions.shape) > 1:
                original_predictions = original_predictions[:, 1]
            
            # Compute intervention effect
            intervention_effect = np.mean(predictions) - np.mean(original_predictions)
            
            results[str(value)] = {
                'intervention_value': value,
                'intervention_effect': intervention_effect,
                'mean_prediction': np.mean(predictions),
                'original_mean_prediction': np.mean(original_predictions),
                'effect_distribution': (predictions - original_predictions).tolist()
            }
        
        return {
            'intervention_feature': intervention_feature,
            'intervention_results': results,
            'strongest_effect': max(results.keys(), 
                                  key=lambda k: abs(results[k]['intervention_effect']))
        }
    
    def counterfactual_analysis(self, instance: pd.Series, 
                              features_to_change: List[str],
                              desired_outcome: Any = None) -> Dict[str, Any]:
        """Generate counterfactual explanations.
        
        Args:
            instance: Instance to explain
            features_to_change: Features allowed to change
            desired_outcome: Desired prediction outcome
            
        Returns:
            Dictionary with counterfactual explanations
        """
        original_prediction = self.model.predict([instance])[0]
        
        if hasattr(self.model, 'predict_proba'):
            original_prob = self.model.predict_proba([instance])[0]
            if len(original_prob) == 2:
                original_prob = original_prob[1]
        
        counterfactuals = []
        
        # Simple grid search for counterfactuals
        for feature in features_to_change:
            # Try different values for this feature
            feature_values = self.data[feature].unique()
            
            for value in feature_values[:10]:  # Limit search
                if value == instance[feature]:
                    continue
                
                # Create counterfactual instance
                cf_instance = instance.copy()
                cf_instance[feature] = value
                
                # Get prediction
                cf_prediction = self.model.predict([cf_instance])[0]
                
                # Check if this achieves desired outcome
                if desired_outcome is None or cf_prediction == desired_outcome:
                    
                    if hasattr(self.model, 'predict_proba'):
                        cf_prob = self.model.predict_proba([cf_instance])[0]
                        if len(cf_prob) == 2:
                            cf_prob = cf_prob[1]
                    else:
                        cf_prob = cf_prediction
                    
                    # Compute distance from original
                    distance = abs(value - instance[feature]) if isinstance(value, (int, float)) else 1
                    
                    counterfactuals.append({
                        'changed_feature': feature,
                        'original_value': instance[feature],
                        'counterfactual_value': value,
                        'original_prediction': original_prediction,
                        'counterfactual_prediction': cf_prediction,
                        'distance': distance,
                        'probability_change': cf_prob - original_prob if hasattr(self.model, 'predict_proba') else None
                    })
        
        # Sort by distance
        counterfactuals.sort(key=lambda x: x['distance'])
        
        return {
            'original_instance': instance.to_dict(),
            'original_prediction': original_prediction,
            'counterfactuals': counterfactuals[:10],  # Return top 10
            'features_analyzed': features_to_change
        }
    
    def generate_causal_report(self, key_features: Optional[List[str]] = None,
                              output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive causal analysis report.
        
        Args:
            key_features: Key features to focus analysis on
            output_path: Path to save report
            
        Returns:
            Dictionary with complete causal analysis
        """
        if key_features is None:
            # Select top features by importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.data.columns if hasattr(self.data, 'columns') else list(range(len(importances)))
                top_indices = np.argsort(importances)[-10:]
                key_features = [feature_names[i] for i in top_indices]
            else:
                key_features = self.data.columns[:10].tolist()
        
        report = {
            'summary': {
                'dataset_size': len(self.data),
                'n_features': len(self.data.columns),
                'key_features_analyzed': key_features,
                'model_type': type(self.model).__name__
            },
            'feature_causality': {},
            'intervention_analyses': {},
            'causal_structure': None
        }
        
        # Discover overall causal structure
        try:
            overall_graph = self.causal_engine.discover_causal_structure(
                self.data[key_features]
            )
            report['causal_structure'] = {
                'n_nodes': len(overall_graph.graph.nodes),
                'n_edges': len(overall_graph.graph.edges),
                'edges': list(overall_graph.graph.edges)
            }
        except Exception as e:
            report['causal_structure'] = {'error': str(e)}
        
        # Analyze each key feature
        for feature in key_features[:5]:  # Limit to avoid long runtime
            try:
                analysis = self.analyze_feature_causality(feature, key_features)
                # Remove graph object for JSON serialization
                analysis_clean = analysis.copy()
                del analysis_clean['causal_graph']
                report['feature_causality'][feature] = analysis_clean
            except Exception as e:
                report['feature_causality'][feature] = {'error': str(e)}
        
        # Intervention analysis for top features
        for feature in key_features[:3]:
            try:
                # Get representative intervention values
                if self.data[feature].dtype in ['int64', 'float64']:
                    values = [self.data[feature].quantile(q) for q in [0.1, 0.5, 0.9]]
                else:
                    values = self.data[feature].value_counts().head(3).index.tolist()
                
                intervention_results = self.intervention_analysis(feature, values)
                report['intervention_analyses'][feature] = intervention_results
            except Exception as e:
                report['intervention_analyses'][feature] = {'error': str(e)}
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report