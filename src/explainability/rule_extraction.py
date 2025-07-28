"""Rule extraction and decision rule generation from machine learning models."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')


@dataclass
class Rule:
    """Represents a single decision rule."""
    conditions: List[str]
    prediction: Any
    confidence: float
    support: int
    feature_importances: Dict[str, float]
    
    def __str__(self):
        conditions_str = " AND ".join(self.conditions)
        return f"IF {conditions_str} THEN {self.prediction} (confidence: {self.confidence:.3f}, support: {self.support})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary format."""
        return {
            'conditions': self.conditions,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'support': self.support,
            'feature_importances': self.feature_importances,
            'rule_text': str(self)
        }


class DecisionTreeRuleExtractor:
    """Extract rules from decision tree models."""
    
    def __init__(self, feature_names: List[str], class_names: Optional[List[str]] = None):
        """Initialize rule extractor.
        
        Args:
            feature_names: Names of features
            class_names: Names of classes (for classification)
        """
        self.feature_names = feature_names
        self.class_names = class_names
        
    def extract_rules(self, tree_model, X: np.ndarray, y: np.ndarray,
                     min_samples_leaf: int = 5) -> List[Rule]:
        """Extract rules from a decision tree.
        
        Args:
            tree_model: Trained decision tree model
            X: Training features
            y: Training labels
            min_samples_leaf: Minimum samples required for a rule
            
        Returns:
            List of extracted rules
        """
        tree = tree_model.tree_
        rules = []
        
        def extract_rules_recursive(node_id: int, conditions: List[str], path_samples: np.ndarray):
            """Recursively extract rules from tree nodes."""
            # Check if leaf node
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node - create rule
                if len(path_samples) >= min_samples_leaf:
                    # Get prediction
                    values = tree.value[node_id][0]
                    if len(values) == 1:  # Regression
                        prediction = values[0]
                    else:  # Classification
                        predicted_class = np.argmax(values)
                        prediction = self.class_names[predicted_class] if self.class_names else predicted_class
                    
                    # Calculate confidence
                    if len(values) > 1:  # Classification
                        confidence = np.max(values) / np.sum(values)
                    else:  # Regression
                        confidence = 1.0  # For regression, use 1.0 as placeholder
                    
                    # Calculate feature importances for this path
                    feature_importances = self._calculate_path_importance(conditions)
                    
                    rule = Rule(
                        conditions=conditions.copy(),
                        prediction=prediction,
                        confidence=confidence,
                        support=len(path_samples),
                        feature_importances=feature_importances
                    )
                    rules.append(rule)
                return
            
            # Internal node - split further
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = self.feature_names[feature_idx]
            
            # Left child (<=)
            left_condition = f"{feature_name} <= {threshold:.3f}"
            left_samples = path_samples[X[path_samples, feature_idx] <= threshold]
            if len(left_samples) > 0:
                extract_rules_recursive(
                    tree.children_left[node_id],
                    conditions + [left_condition],
                    left_samples
                )
            
            # Right child (>)
            right_condition = f"{feature_name} > {threshold:.3f}"
            right_samples = path_samples[X[path_samples, feature_idx] > threshold]
            if len(right_samples) > 0:
                extract_rules_recursive(
                    tree.children_right[node_id],
                    conditions + [right_condition],
                    right_samples
                )
        
        # Start extraction from root
        root_samples = np.arange(len(X))
        extract_rules_recursive(0, [], root_samples)
        
        return rules
    
    def _calculate_path_importance(self, conditions: List[str]) -> Dict[str, float]:
        """Calculate feature importance for a specific rule path."""
        feature_counts = {}
        
        for condition in conditions:
            # Extract feature name from condition
            feature_name = condition.split()[0]
            if feature_name in feature_counts:
                feature_counts[feature_name] += 1
            else:
                feature_counts[feature_name] = 1
        
        # Normalize to get importances
        total_conditions = len(conditions)
        if total_conditions == 0:
            return {}
        
        feature_importances = {
            feature: count / total_conditions 
            for feature, count in feature_counts.items()
        }
        
        return feature_importances


class RandomForestRuleExtractor:
    """Extract rules from random forest models."""
    
    def __init__(self, feature_names: List[str], class_names: Optional[List[str]] = None):
        """Initialize random forest rule extractor.
        
        Args:
            feature_names: Names of features
            class_names: Names of classes
        """
        self.feature_names = feature_names
        self.class_names = class_names
        self.tree_extractor = DecisionTreeRuleExtractor(feature_names, class_names)
        
    def extract_rules(self, rf_model, X: np.ndarray, y: np.ndarray,
                     max_trees: int = 10, min_samples_leaf: int = 10) -> List[Rule]:
        """Extract rules from random forest.
        
        Args:
            rf_model: Trained random forest model
            X: Training features
            y: Training labels
            max_trees: Maximum number of trees to extract rules from
            min_samples_leaf: Minimum samples required for a rule
            
        Returns:
            List of extracted rules
        """
        all_rules = []
        
        # Extract rules from each tree
        n_trees = min(max_trees, len(rf_model.estimators_))
        for i in range(n_trees):
            tree = rf_model.estimators_[i]
            tree_rules = self.tree_extractor.extract_rules(tree, X, y, min_samples_leaf)
            
            # Add tree identifier to rules
            for rule in tree_rules:
                rule.tree_id = i
            
            all_rules.extend(tree_rules)
        
        # Aggregate and filter rules
        aggregated_rules = self._aggregate_rules(all_rules)
        
        return aggregated_rules
    
    def _aggregate_rules(self, rules: List[Rule], 
                        similarity_threshold: float = 0.8) -> List[Rule]:
        """Aggregate similar rules across trees."""
        # Group similar rules
        rule_groups = []
        
        for rule in rules:
            # Find similar existing group
            added_to_group = False
            for group in rule_groups:
                if self._rules_similar(rule, group[0], similarity_threshold):
                    group.append(rule)
                    added_to_group = True
                    break
            
            if not added_to_group:
                rule_groups.append([rule])
        
        # Create aggregated rules
        aggregated_rules = []
        for group in rule_groups:
            if len(group) >= 2:  # Only keep rules that appear in multiple trees
                aggregated_rule = self._create_aggregated_rule(group)
                aggregated_rules.append(aggregated_rule)
        
        # Sort by confidence and support
        aggregated_rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        
        return aggregated_rules
    
    def _rules_similar(self, rule1: Rule, rule2: Rule, threshold: float) -> bool:
        """Check if two rules are similar."""
        # Simple similarity based on overlapping conditions
        conditions1 = set(rule1.conditions)
        conditions2 = set(rule2.conditions)
        
        if len(conditions1) == 0 and len(conditions2) == 0:
            return True
        
        intersection = len(conditions1 & conditions2)
        union = len(conditions1 | conditions2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _create_aggregated_rule(self, rule_group: List[Rule]) -> Rule:
        """Create an aggregated rule from a group of similar rules."""
        # Use most common prediction
        predictions = [rule.prediction for rule in rule_group]
        most_common_prediction = max(set(predictions), key=predictions.count)
        
        # Average confidence
        avg_confidence = np.mean([rule.confidence for rule in rule_group])
        
        # Sum support
        total_support = sum([rule.support for rule in rule_group])
        
        # Use conditions from first rule (they should be similar)
        representative_conditions = rule_group[0].conditions
        
        # Average feature importances
        all_importances = [rule.feature_importances for rule in rule_group]
        avg_importances = {}
        for feature in self.feature_names:
            feature_values = [imp.get(feature, 0) for imp in all_importances]
            if any(v > 0 for v in feature_values):
                avg_importances[feature] = np.mean(feature_values)
        
        return Rule(
            conditions=representative_conditions,
            prediction=most_common_prediction,
            confidence=avg_confidence,
            support=total_support,
            feature_importances=avg_importances
        )


class LinearModelRuleExtractor:
    """Extract rules from linear models using feature coefficients."""
    
    def __init__(self, feature_names: List[str], class_names: Optional[List[str]] = None):
        """Initialize linear model rule extractor.
        
        Args:
            feature_names: Names of features
            class_names: Names of classes
        """
        self.feature_names = feature_names
        self.class_names = class_names
        
    def extract_rules(self, linear_model, X: np.ndarray, y: np.ndarray,
                     coefficient_threshold: float = 0.1) -> List[Rule]:
        """Extract rules from linear model coefficients.
        
        Args:
            linear_model: Trained linear model
            X: Training features
            y: Training labels
            coefficient_threshold: Minimum coefficient magnitude to include
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        # Get coefficients
        if hasattr(linear_model, 'coef_'):
            if len(linear_model.coef_.shape) > 1:
                coefficients = linear_model.coef_[0]
            else:
                coefficients = linear_model.coef_
        else:
            raise ValueError("Model does not have coefficients")
        
        # Get intercept
        intercept = getattr(linear_model, 'intercept_', 0)
        if isinstance(intercept, np.ndarray):
            intercept = intercept[0]
        
        # Create rules based on coefficient signs and magnitudes
        positive_features = []
        negative_features = []
        
        for i, coef in enumerate(coefficients):
            if abs(coef) >= coefficient_threshold:
                feature_name = self.feature_names[i]
                if coef > 0:
                    positive_features.append((feature_name, coef))
                else:
                    negative_features.append((feature_name, abs(coef)))
        
        # Sort by coefficient magnitude
        positive_features.sort(key=lambda x: x[1], reverse=True)
        negative_features.sort(key=lambda x: x[1], reverse=True)
        
        # Create rules for different scenarios
        rules.extend(self._create_linear_rules(positive_features, negative_features, 
                                             intercept, X, y, "high"))
        rules.extend(self._create_linear_rules(negative_features, positive_features, 
                                             intercept, X, y, "low"))
        
        return rules
    
    def _create_linear_rules(self, primary_features: List[Tuple[str, float]],
                           secondary_features: List[Tuple[str, float]],
                           intercept: float, X: np.ndarray, y: np.ndarray,
                           prediction_type: str) -> List[Rule]:
        """Create rules for linear models."""
        rules = []
        
        if not primary_features:
            return rules
        
        # Create rule with top contributing features
        top_features = primary_features[:3]  # Use top 3 features
        
        conditions = []
        feature_importances = {}
        
        total_importance = sum(coef for _, coef in top_features)
        
        for feature_name, coef in top_features:
            # Create condition based on feature statistics
            feature_idx = self.feature_names.index(feature_name)
            feature_values = X[:, feature_idx]
            
            if prediction_type == "high":
                threshold = np.percentile(feature_values, 75)
                condition = f"{feature_name} >= {threshold:.3f}"
            else:
                threshold = np.percentile(feature_values, 25)
                condition = f"{feature_name} <= {threshold:.3f}"
            
            conditions.append(condition)
            feature_importances[feature_name] = coef / total_importance
        
        # Estimate support and confidence
        # This is a simplified estimation
        support = len(X) // 4  # Rough estimate
        confidence = 0.7 + 0.2 * (total_importance / (total_importance + abs(intercept) + 0.1))
        
        prediction = f"{prediction_type} probability" if self.class_names is None else f"{prediction_type} {self.class_names[1] if len(self.class_names) > 1 else self.class_names[0]}"
        
        rule = Rule(
            conditions=conditions,
            prediction=prediction,
            confidence=min(confidence, 1.0),
            support=support,
            feature_importances=feature_importances
        )
        
        rules.append(rule)
        
        return rules


class RuleExtractor:
    """Unified rule extraction interface for different model types."""
    
    def __init__(self, model, feature_names: List[str], 
                 class_names: Optional[List[str]] = None):
        """Initialize rule extractor.
        
        Args:
            model: Trained machine learning model
            feature_names: Names of features
            class_names: Names of classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Initialize appropriate extractor
        if isinstance(model, DecisionTreeClassifier):
            self.extractor = DecisionTreeRuleExtractor(feature_names, class_names)
        elif isinstance(model, RandomForestClassifier):
            self.extractor = RandomForestRuleExtractor(feature_names, class_names)
        elif isinstance(model, LogisticRegression):
            self.extractor = LinearModelRuleExtractor(feature_names, class_names)
        else:
            # Try to use LIME or SHAP for model-agnostic extraction
            self.extractor = self._create_surrogate_extractor()
    
    def _create_surrogate_extractor(self):
        """Create surrogate decision tree for model-agnostic rule extraction."""
        return DecisionTreeRuleExtractor(self.feature_names, self.class_names)
    
    def extract_rules(self, X: np.ndarray, y: np.ndarray, **kwargs) -> List[Rule]:
        """Extract rules from the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments for specific extractors
            
        Returns:
            List of extracted rules
        """
        if hasattr(self.extractor, 'extract_rules'):
            return self.extractor.extract_rules(self.model, X, y, **kwargs)
        else:
            # Model-agnostic extraction using surrogate model
            return self._extract_with_surrogate(X, y, **kwargs)
    
    def _extract_with_surrogate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> List[Rule]:
        """Extract rules using a surrogate decision tree."""
        # Get predictions from the original model
        if hasattr(self.model, 'predict_proba'):
            y_pred = self.model.predict_proba(X)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]  # Use positive class probability
        else:
            y_pred = self.model.predict(X)
        
        # Train surrogate decision tree
        surrogate_tree = DecisionTreeClassifier(
            max_depth=kwargs.get('max_depth', 10),
            min_samples_leaf=kwargs.get('min_samples_leaf', 20),
            random_state=42
        )
        
        # For regression predictions, convert to binary classification
        if len(np.unique(y_pred)) > 10:
            y_surrogate = (y_pred > np.median(y_pred)).astype(int)
        else:
            y_surrogate = y_pred
        
        surrogate_tree.fit(X, y_surrogate)
        
        # Extract rules from surrogate tree
        return self.extractor.extract_rules(surrogate_tree, X, y_surrogate, 
                                          min_samples_leaf=kwargs.get('min_samples_leaf', 20))
    
    def filter_rules(self, rules: List[Rule], 
                    min_confidence: float = 0.7,
                    min_support: int = 10,
                    max_conditions: int = 5) -> List[Rule]:
        """Filter rules based on quality criteria.
        
        Args:
            rules: List of rules to filter
            min_confidence: Minimum confidence threshold
            min_support: Minimum support threshold
            max_conditions: Maximum number of conditions per rule
            
        Returns:
            Filtered list of rules
        """
        filtered_rules = []
        
        for rule in rules:
            if (rule.confidence >= min_confidence and 
                rule.support >= min_support and 
                len(rule.conditions) <= max_conditions):
                filtered_rules.append(rule)
        
        return filtered_rules
    
    def rank_rules(self, rules: List[Rule], 
                  ranking_method: str = 'confidence_support') -> List[Rule]:
        """Rank rules by importance.
        
        Args:
            rules: List of rules to rank
            ranking_method: Method for ranking ('confidence_support', 'support', 'confidence')
            
        Returns:
            Ranked list of rules
        """
        if ranking_method == 'confidence_support':
            # Combine confidence and support
            ranked_rules = sorted(rules, 
                                key=lambda r: r.confidence * np.log(r.support + 1), 
                                reverse=True)
        elif ranking_method == 'support':
            ranked_rules = sorted(rules, key=lambda r: r.support, reverse=True)
        elif ranking_method == 'confidence':
            ranked_rules = sorted(rules, key=lambda r: r.confidence, reverse=True)
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        return ranked_rules
    
    def visualize_rules(self, rules: List[Rule], top_k: int = 10,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize extracted rules.
        
        Args:
            rules: List of rules to visualize
            top_k: Number of top rules to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not rules:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No rules to visualize', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        top_rules = rules[:top_k]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Rule confidence vs support
        confidences = [rule.confidence for rule in top_rules]
        supports = [rule.support for rule in top_rules]
        rule_lengths = [len(rule.conditions) for rule in top_rules]
        
        scatter = ax1.scatter(supports, confidences, s=[l*50 for l in rule_lengths], 
                            alpha=0.6, c=rule_lengths, cmap='viridis')
        ax1.set_xlabel('Support (Number of Examples)')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Rule Quality: Confidence vs Support')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Number of Conditions')
        
        # Plot 2: Feature importance in rules
        all_features = set()
        for rule in top_rules:
            all_features.update(rule.feature_importances.keys())
        
        feature_importance_matrix = []
        rule_labels = []
        
        for i, rule in enumerate(top_rules):
            rule_importances = []
            for feature in sorted(all_features):
                rule_importances.append(rule.feature_importances.get(feature, 0))
            feature_importance_matrix.append(rule_importances)
            rule_labels.append(f"Rule {i+1}")
        
        if feature_importance_matrix:
            im = ax2.imshow(feature_importance_matrix, cmap='Blues', aspect='auto')
            ax2.set_xticks(range(len(sorted(all_features))))
            ax2.set_xticklabels(sorted(all_features), rotation=45, ha='right')
            ax2.set_yticks(range(len(rule_labels)))
            ax2.set_yticklabels(rule_labels)
            ax2.set_title('Feature Importance in Rules')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2, label='Importance')
        
        plt.tight_layout()
        return fig
    
    def generate_rule_report(self, X: np.ndarray, y: np.ndarray,
                           output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive rule extraction report.
        
        Args:
            X: Training features
            y: Training labels
            output_path: Path to save report
            
        Returns:
            Dictionary with rule analysis
        """
        # Extract rules
        all_rules = self.extract_rules(X, y)
        
        # Filter and rank rules
        filtered_rules = self.filter_rules(all_rules)
        ranked_rules = self.rank_rules(filtered_rules)
        
        # Generate report
        report = {
            'model_info': {
                'model_type': type(self.model).__name__,
                'n_features': len(self.feature_names),
                'n_samples': len(X),
                'feature_names': self.feature_names
            },
            'rule_statistics': {
                'total_rules_extracted': len(all_rules),
                'rules_after_filtering': len(filtered_rules),
                'average_confidence': np.mean([r.confidence for r in filtered_rules]) if filtered_rules else 0,
                'average_support': np.mean([r.support for r in filtered_rules]) if filtered_rules else 0,
                'average_conditions_per_rule': np.mean([len(r.conditions) for r in filtered_rules]) if filtered_rules else 0
            },
            'top_rules': [rule.to_dict() for rule in ranked_rules[:20]],
            'feature_usage': self._analyze_feature_usage(filtered_rules),
            'rule_complexity_distribution': self._analyze_rule_complexity(filtered_rules)
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _analyze_feature_usage(self, rules: List[Rule]) -> Dict[str, Any]:
        """Analyze how frequently each feature is used in rules."""
        feature_usage = {feature: 0 for feature in self.feature_names}
        feature_importance_sum = {feature: 0.0 for feature in self.feature_names}
        
        for rule in rules:
            for feature, importance in rule.feature_importances.items():
                if feature in feature_usage:
                    feature_usage[feature] += 1
                    feature_importance_sum[feature] += importance
        
        # Calculate average importance
        feature_avg_importance = {}
        for feature in self.feature_names:
            if feature_usage[feature] > 0:
                feature_avg_importance[feature] = feature_importance_sum[feature] / feature_usage[feature]
            else:
                feature_avg_importance[feature] = 0.0
        
        # Sort by usage frequency
        sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'usage_frequency': dict(sorted_features),
            'average_importance': feature_avg_importance,
            'most_used_features': [f[0] for f in sorted_features[:10]],
            'unused_features': [f[0] for f in sorted_features if f[1] == 0]
        }
    
    def _analyze_rule_complexity(self, rules: List[Rule]) -> Dict[str, Any]:
        """Analyze the complexity distribution of rules."""
        rule_lengths = [len(rule.conditions) for rule in rules]
        
        if not rule_lengths:
            return {'message': 'No rules to analyze'}
        
        complexity_distribution = {}
        for length in rule_lengths:
            complexity_distribution[str(length)] = complexity_distribution.get(str(length), 0) + 1
        
        return {
            'distribution': complexity_distribution,
            'average_complexity': np.mean(rule_lengths),
            'median_complexity': np.median(rule_lengths),
            'max_complexity': max(rule_lengths),
            'min_complexity': min(rule_lengths)
        }