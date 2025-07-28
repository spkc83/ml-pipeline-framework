"""Concept-based explanations using TCAV and related methods."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')


class ConceptActivationVector:
    """Represents a Concept Activation Vector (CAV) for TCAV."""
    
    def __init__(self, concept_name: str, layer_name: str, 
                 direction: np.ndarray, accuracy: float):
        """Initialize CAV.
        
        Args:
            concept_name: Name of the concept
            layer_name: Name of the layer where CAV was learned
            direction: The learned direction vector
            accuracy: Accuracy of the concept classifier
        """
        self.concept_name = concept_name
        self.layer_name = layer_name
        self.direction = direction
        self.accuracy = accuracy
        
    def __repr__(self):
        return f"CAV(concept='{self.concept_name}', layer='{self.layer_name}', acc={self.accuracy:.3f})"


class TCAVAnalyzer:
    """Testing with Concept Activation Vectors (TCAV) implementation."""
    
    def __init__(self, model_fn: Callable, layer_names: List[str], 
                 random_state: int = 42):
        """Initialize TCAV analyzer.
        
        Args:
            model_fn: Function that returns activations for given inputs and layer
            layer_names: Names of layers to analyze
            random_state: Random state for reproducibility
        """
        self.model_fn = model_fn
        self.layer_names = layer_names
        self.random_state = random_state
        self.cavs_ = {}
        
    def learn_concept(self, concept_examples: np.ndarray, 
                     random_examples: np.ndarray, 
                     concept_name: str,
                     layer_name: str,
                     classifier: str = 'logistic') -> ConceptActivationVector:
        """Learn a Concept Activation Vector for a given concept.
        
        Args:
            concept_examples: Examples that contain the concept
            random_examples: Random examples (negative examples)
            concept_name: Name of the concept
            layer_name: Layer to learn CAV for
            classifier: Type of classifier ('logistic' or 'svm')
            
        Returns:
            ConceptActivationVector object
        """
        # Get activations for concept and random examples
        concept_acts = self.model_fn(concept_examples, layer_name)
        random_acts = self.model_fn(random_examples, layer_name)
        
        # Create training data
        X = np.vstack([concept_acts, random_acts])
        y = np.hstack([np.ones(len(concept_acts)), np.zeros(len(random_acts))])
        
        # Train classifier
        if classifier == 'logistic':
            clf = LogisticRegression(random_state=self.random_state)
        elif classifier == 'svm':
            clf = LinearSVC(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        
        # Get the direction vector (normal to decision boundary)
        if hasattr(clf, 'coef_'):
            direction = clf.coef_[0]
        else:
            direction = clf.coef_
            
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        cav = ConceptActivationVector(concept_name, layer_name, direction, accuracy)
        
        # Store CAV
        if layer_name not in self.cavs_:
            self.cavs_[layer_name] = {}
        self.cavs_[layer_name][concept_name] = cav
        
        return cav
    
    def compute_tcav_score(self, test_examples: np.ndarray, 
                          concept_name: str, layer_name: str,
                          target_class: Optional[int] = None) -> Dict[str, float]:
        """Compute TCAV score for a concept on test examples.
        
        Args:
            test_examples: Examples to test
            concept_name: Name of the concept
            layer_name: Layer to use
            target_class: Target class to analyze (if None, use all predictions)
            
        Returns:
            Dictionary with TCAV scores and statistics
        """
        if layer_name not in self.cavs_ or concept_name not in self.cavs_[layer_name]:
            raise ValueError(f"CAV not found for concept '{concept_name}' in layer '{layer_name}'")
        
        cav = self.cavs_[layer_name][concept_name]
        
        # Get activations for test examples
        test_acts = self.model_fn(test_examples, layer_name)
        
        # Compute directional derivatives (gradients)
        gradients = self._compute_gradients(test_examples, layer_name, target_class)
        
        # Compute TCAV score
        tcav_scores = []
        for grad in gradients:
            # Dot product of gradient with CAV direction
            score = np.dot(grad, cav.direction)
            tcav_scores.append(score > 0)  # Positive influence
        
        tcav_score = np.mean(tcav_scores)
        
        return {
            'tcav_score': tcav_score,
            'concept_name': concept_name,
            'layer_name': layer_name,
            'cav_accuracy': cav.accuracy,
            'n_examples': len(test_examples),
            'positive_influence_count': int(np.sum(tcav_scores))
        }
    
    def _compute_gradients(self, examples: np.ndarray, layer_name: str, 
                          target_class: Optional[int] = None) -> np.ndarray:
        """Compute gradients of predictions with respect to layer activations.
        
        This is a simplified implementation. In practice, you would use
        automatic differentiation tools like PyTorch or TensorFlow.
        """
        # Placeholder implementation - in real scenario, use autodiff
        activations = self.model_fn(examples, layer_name)
        
        # Approximate gradients using finite differences
        gradients = []
        epsilon = 1e-5
        
        for i, example in enumerate(examples):
            activation = activations[i]
            grad = np.zeros_like(activation)
            
            # Finite difference approximation
            for j in range(len(activation)):
                # Perturb activation
                perturbed_act = activation.copy()
                perturbed_act[j] += epsilon
                
                # This would require injecting perturbed activations back into model
                # For now, use random gradients as placeholder
                grad[j] = np.random.normal(0, 1)
            
            gradients.append(grad)
        
        return np.array(gradients)
    
    def statistical_significance_test(self, test_examples: np.ndarray,
                                     concept_name: str, layer_name: str,
                                     n_bootstrap: int = 100) -> Dict[str, Any]:
        """Test statistical significance of TCAV scores using bootstrap.
        
        Args:
            test_examples: Examples to test
            concept_name: Name of the concept
            layer_name: Layer to use
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with significance test results
        """
        original_score = self.compute_tcav_score(test_examples, concept_name, layer_name)
        
        # Bootstrap sampling
        bootstrap_scores = []
        n_examples = len(test_examples)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_examples, n_examples, replace=True)
            bootstrap_examples = test_examples[indices]
            
            score = self.compute_tcav_score(bootstrap_examples, concept_name, layer_name)
            bootstrap_scores.append(score['tcav_score'])
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Compute confidence intervals
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        # Test if significantly different from random (0.5)
        p_value = 2 * min(np.mean(bootstrap_scores >= 0.5), np.mean(bootstrap_scores <= 0.5))
        
        return {
            'original_tcav_score': original_score['tcav_score'],
            'bootstrap_mean': np.mean(bootstrap_scores),
            'bootstrap_std': np.std(bootstrap_scores),
            'confidence_interval': (ci_lower, ci_upper),
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'bootstrap_scores': bootstrap_scores.tolist()
        }


class ConceptAnalyzer:
    """High-level concept analysis and visualization."""
    
    def __init__(self, tcav_analyzer: TCAVAnalyzer):
        """Initialize concept analyzer.
        
        Args:
            tcav_analyzer: Fitted TCAV analyzer
        """
        self.tcav_analyzer = tcav_analyzer
        
    def analyze_concept_importance(self, test_examples: np.ndarray,
                                  concept_names: List[str],
                                  layer_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Analyze importance of multiple concepts across layers.
        
        Args:
            test_examples: Examples to analyze
            concept_names: Names of concepts to analyze
            layer_names: Layers to analyze (if None, use all available)
            
        Returns:
            DataFrame with concept importance scores
        """
        if layer_names is None:
            layer_names = self.tcav_analyzer.layer_names
        
        results = []
        
        for layer in layer_names:
            for concept in concept_names:
                try:
                    score_info = self.tcav_analyzer.compute_tcav_score(
                        test_examples, concept, layer
                    )
                    
                    sig_test = self.tcav_analyzer.statistical_significance_test(
                        test_examples, concept, layer, n_bootstrap=50
                    )
                    
                    results.append({
                        'concept': concept,
                        'layer': layer,
                        'tcav_score': score_info['tcav_score'],
                        'cav_accuracy': score_info['cav_accuracy'],
                        'is_significant': sig_test['is_significant'],
                        'p_value': sig_test['p_value'],
                        'confidence_lower': sig_test['confidence_interval'][0],
                        'confidence_upper': sig_test['confidence_interval'][1]
                    })
                    
                except ValueError as e:
                    print(f"Skipping {concept} in {layer}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def visualize_concept_importance(self, importance_df: pd.DataFrame,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize concept importance across layers.
        
        Args:
            importance_df: DataFrame from analyze_concept_importance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Heatmap of TCAV scores
        pivot_data = importance_df.pivot(index='concept', columns='layer', values='tcav_score')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0.5, ax=axes[0, 0])
        axes[0, 0].set_title('TCAV Scores by Concept and Layer')
        
        # Significance indicators
        sig_data = importance_df.pivot(index='concept', columns='layer', values='is_significant')
        sns.heatmap(sig_data, annot=True, fmt='', cmap='RdYlGn', 
                   cbar_kws={'label': 'Significant'}, ax=axes[0, 1])
        axes[0, 1].set_title('Statistical Significance')
        
        # CAV accuracies
        acc_data = importance_df.pivot(index='concept', columns='layer', values='cav_accuracy')
        sns.heatmap(acc_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('CAV Classifier Accuracies')
        
        # P-values
        p_data = importance_df.pivot(index='concept', columns='layer', values='p_value')
        sns.heatmap(p_data, annot=True, fmt='.3f', cmap='viridis_r', ax=axes[1, 1])
        axes[1, 1].set_title('P-values')
        
        plt.tight_layout()
        return fig
    
    def concept_interaction_analysis(self, test_examples: np.ndarray,
                                   concept_pairs: List[Tuple[str, str]],
                                   layer_name: str) -> Dict[str, Any]:
        """Analyze interactions between concept pairs.
        
        Args:
            test_examples: Examples to analyze
            concept_pairs: Pairs of concepts to analyze
            layer_name: Layer to analyze
            
        Returns:
            Dictionary with interaction analysis results
        """
        interactions = {}
        
        for concept1, concept2 in concept_pairs:
            try:
                # Get individual TCAV scores
                score1 = self.tcav_analyzer.compute_tcav_score(
                    test_examples, concept1, layer_name
                )
                score2 = self.tcav_analyzer.compute_tcav_score(
                    test_examples, concept2, layer_name
                )
                
                # Get CAVs
                cav1 = self.tcav_analyzer.cavs_[layer_name][concept1]
                cav2 = self.tcav_analyzer.cavs_[layer_name][concept2]
                
                # Compute cosine similarity between CAVs
                cosine_sim = np.dot(cav1.direction, cav2.direction)
                
                # Analyze correlation in activations
                acts = self.tcav_analyzer.model_fn(test_examples, layer_name)
                proj1 = np.dot(acts, cav1.direction)
                proj2 = np.dot(acts, cav2.direction)
                correlation = np.corrcoef(proj1, proj2)[0, 1]
                
                interactions[f"{concept1}-{concept2}"] = {
                    'tcav_score_1': score1['tcav_score'],
                    'tcav_score_2': score2['tcav_score'],
                    'cav_cosine_similarity': cosine_sim,
                    'activation_correlation': correlation,
                    'orthogonality': abs(cosine_sim),  # How orthogonal the concepts are
                }
                
            except (ValueError, KeyError) as e:
                print(f"Skipping pair {concept1}-{concept2}: {e}")
                continue
        
        return interactions
    
    def generate_concept_report(self, test_examples: np.ndarray,
                              concept_names: List[str],
                              output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive concept analysis report.
        
        Args:
            test_examples: Examples to analyze
            concept_names: Concepts to include in report
            output_path: Path to save report (optional)
            
        Returns:
            Dictionary with complete analysis
        """
        # Basic importance analysis
        importance_df = self.analyze_concept_importance(test_examples, concept_names)
        
        # Concept interactions
        concept_pairs = [(c1, c2) for i, c1 in enumerate(concept_names) 
                        for c2 in concept_names[i+1:]]
        
        interactions = {}
        for layer in self.tcav_analyzer.layer_names:
            interactions[layer] = self.concept_interaction_analysis(
                test_examples, concept_pairs, layer
            )
        
        # Summary statistics
        summary = {
            'n_concepts': len(concept_names),
            'n_layers': len(self.tcav_analyzer.layer_names),
            'n_test_examples': len(test_examples),
            'avg_tcav_score': importance_df['tcav_score'].mean(),
            'significant_concepts': importance_df[importance_df['is_significant']]['concept'].tolist(),
            'best_performing_concepts': importance_df.nlargest(5, 'tcav_score')[['concept', 'layer', 'tcav_score']].to_dict('records')
        }
        
        report = {
            'summary': summary,
            'importance_analysis': importance_df.to_dict('records'),
            'concept_interactions': interactions,
            'methodology': {
                'description': 'TCAV (Testing with Concept Activation Vectors) analysis',
                'layers_analyzed': self.tcav_analyzer.layer_names,
                'bootstrap_samples': 50,
                'significance_threshold': 0.05
            }
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report


class ConceptDriftDetector:
    """Detect concept drift in model behavior over time."""
    
    def __init__(self, reference_cavs: Dict[str, ConceptActivationVector]):
        """Initialize concept drift detector.
        
        Args:
            reference_cavs: Reference CAVs from baseline model
        """
        self.reference_cavs = reference_cavs
        
    def detect_drift(self, current_cavs: Dict[str, ConceptActivationVector],
                    similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """Detect concept drift by comparing CAVs.
        
        Args:
            current_cavs: CAVs from current model
            similarity_threshold: Threshold for drift detection
            
        Returns:
            Dictionary with drift analysis results
        """
        drift_results = {}
        
        for concept_name in self.reference_cavs:
            if concept_name not in current_cavs:
                drift_results[concept_name] = {
                    'status': 'missing',
                    'similarity': 0.0,
                    'drift_detected': True
                }
                continue
            
            ref_cav = self.reference_cavs[concept_name]
            curr_cav = current_cavs[concept_name]
            
            # Compute cosine similarity between CAV directions
            similarity = np.dot(ref_cav.direction, curr_cav.direction)
            
            # Detect drift
            drift_detected = similarity < similarity_threshold
            
            drift_results[concept_name] = {
                'status': 'compared',
                'similarity': similarity,
                'drift_detected': drift_detected,
                'reference_accuracy': ref_cav.accuracy,
                'current_accuracy': curr_cav.accuracy,
                'accuracy_change': curr_cav.accuracy - ref_cav.accuracy
            }
        
        # Summary
        total_concepts = len(drift_results)
        drifted_concepts = sum(1 for r in drift_results.values() if r['drift_detected'])
        
        summary = {
            'total_concepts': total_concepts,
            'drifted_concepts': drifted_concepts,
            'drift_rate': drifted_concepts / total_concepts if total_concepts > 0 else 0,
            'avg_similarity': np.mean([r['similarity'] for r in drift_results.values() 
                                     if r['status'] == 'compared'])
        }
        
        return {
            'summary': summary,
            'detailed_results': drift_results,
            'threshold_used': similarity_threshold
        }