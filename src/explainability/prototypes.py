"""Prototype and representative example selection for interpretability."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
warnings.filterwarnings('ignore')


class PrototypeSelector:
    """Select representative examples and prototypes for interpretability."""
    
    def __init__(self, n_prototypes: int = 10, n_criticisms: int = 5, 
                 distance_metric: str = 'euclidean'):
        """Initialize prototype selector.
        
        Args:
            n_prototypes: Number of prototypes to select
            n_criticisms: Number of criticisms (outliers) to select
            distance_metric: Distance metric for similarity calculations
        """
        self.n_prototypes = n_prototypes
        self.n_criticisms = n_criticisms
        self.distance_metric = distance_metric
        self.prototypes_ = None
        self.criticisms_ = None
        self.prototype_indices_ = None
        self.criticism_indices_ = None
        
    def select_prototypes(self, X: np.ndarray, method: str = 'k_medoids') -> Tuple[np.ndarray, np.ndarray]:
        """Select representative prototypes from the data.
        
        Args:
            X: Feature matrix
            method: Selection method ('k_medoids', 'k_centers', 'diversity')
            
        Returns:
            Tuple of (prototypes, prototype_indices)
        """
        if method == 'k_medoids':
            return self._select_k_medoids(X)
        elif method == 'k_centers':
            return self._select_k_centers(X)
        elif method == 'diversity':
            return self._select_diverse_prototypes(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _select_k_medoids(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select prototypes using k-medoids clustering."""
        # Use KMeans to get initial clusters, then find medoids
        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        prototypes = []
        prototype_indices = []
        
        for cluster_id in range(self.n_prototypes):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = X[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_points) == 0:
                continue
                
            # Find medoid (point with minimum sum of distances to other points in cluster)
            distances = pairwise_distances(cluster_points, metric=self.distance_metric)
            medoid_idx = np.argmin(distances.sum(axis=1))
            
            prototypes.append(cluster_points[medoid_idx])
            prototype_indices.append(cluster_indices[medoid_idx])
        
        self.prototypes_ = np.array(prototypes)
        self.prototype_indices_ = np.array(prototype_indices)
        
        return self.prototypes_, self.prototype_indices_
    
    def _select_k_centers(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select prototypes using k-centers algorithm (greedy farthest-first)."""
        n_samples = len(X)
        prototype_indices = []
        
        # Start with a random point
        first_idx = np.random.randint(n_samples)
        prototype_indices.append(first_idx)
        
        for _ in range(self.n_prototypes - 1):
            # Find point that is farthest from all current prototypes
            distances_to_prototypes = []
            for idx in prototype_indices:
                distances = cdist([X[idx]], X, metric=self.distance_metric)[0]
                distances_to_prototypes.append(distances)
            
            # For each point, find distance to nearest prototype
            min_distances = np.min(distances_to_prototypes, axis=0)
            
            # Select point with maximum distance to nearest prototype
            farthest_idx = np.argmax(min_distances)
            prototype_indices.append(farthest_idx)
        
        self.prototype_indices_ = np.array(prototype_indices)
        self.prototypes_ = X[self.prototype_indices_]
        
        return self.prototypes_, self.prototype_indices_
    
    def _select_diverse_prototypes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select diverse prototypes using maximum margin diversity."""
        n_samples = len(X)
        prototype_indices = []
        
        # Compute all pairwise distances
        all_distances = pairwise_distances(X, metric=self.distance_metric)
        
        # Start with the point that has maximum average distance to all others
        avg_distances = np.mean(all_distances, axis=1)
        first_idx = np.argmax(avg_distances)
        prototype_indices.append(first_idx)
        
        for _ in range(self.n_prototypes - 1):
            # For each remaining point, compute minimum distance to selected prototypes
            remaining_indices = [i for i in range(n_samples) if i not in prototype_indices]
            max_min_distance = -1
            best_idx = None
            
            for candidate_idx in remaining_indices:
                min_distance = min(all_distances[candidate_idx][proto_idx] 
                                 for proto_idx in prototype_indices)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx
            
            prototype_indices.append(best_idx)
        
        self.prototype_indices_ = np.array(prototype_indices)
        self.prototypes_ = X[self.prototype_indices_]
        
        return self.prototypes_, self.prototype_indices_
    
    def select_criticisms(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select criticisms (outliers/edge cases) from the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (criticisms, criticism_indices)
        """
        if self.prototypes_ is None:
            raise ValueError("Must select prototypes first")
        
        # Find points that are poorly represented by prototypes
        distances_to_prototypes = cdist(X, self.prototypes_, metric=self.distance_metric)
        min_distances_to_prototypes = np.min(distances_to_prototypes, axis=1)
        
        # Select points with largest distance to nearest prototype
        criticism_indices = np.argsort(min_distances_to_prototypes)[-self.n_criticisms:]
        
        self.criticism_indices_ = criticism_indices
        self.criticisms_ = X[criticism_indices]
        
        return self.criticisms_, self.criticism_indices_
    
    def explain_prototype(self, prototype_idx: int, X: np.ndarray, 
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Explain why a prototype is representative.
        
        Args:
            prototype_idx: Index of prototype to explain
            X: Original feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary with explanation details
        """
        if self.prototypes_ is None:
            raise ValueError("Must select prototypes first")
        
        prototype = self.prototypes_[prototype_idx]
        
        # Find points represented by this prototype
        distances_to_all_prototypes = cdist(X, self.prototypes_, metric=self.distance_metric)
        nearest_prototype_indices = np.argmin(distances_to_all_prototypes, axis=1)
        represented_mask = nearest_prototype_indices == prototype_idx
        represented_points = X[represented_mask]
        
        explanation = {
            'prototype_index': self.prototype_indices_[prototype_idx],
            'n_represented_points': np.sum(represented_mask),
            'coverage_percentage': np.sum(represented_mask) / len(X) * 100,
            'prototype_values': prototype.tolist(),
        }
        
        if feature_names:
            explanation['feature_names'] = feature_names
            
            # Compute feature importance for this prototype
            if len(represented_points) > 1:
                feature_std = np.std(represented_points, axis=0)
                feature_importance = 1 / (feature_std + 1e-8)  # Higher importance for lower variance
                feature_importance = feature_importance / np.sum(feature_importance)
                
                explanation['feature_importance'] = {
                    name: float(importance) 
                    for name, importance in zip(feature_names, feature_importance)
                }
        
        return explanation
    
    def visualize_prototypes(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                           method: str = 'tsne', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize prototypes and criticisms in 2D space.
        
        Args:
            X: Feature matrix
            y: Target labels (optional)
            method: Dimensionality reduction method ('tsne', 'pca')
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.prototypes_ is None:
            raise ValueError("Must select prototypes first")
        
        # Reduce dimensionality for visualization
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: All points with prototypes highlighted
        if y is not None:
            scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax1)
        else:
            ax1.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, color='lightgray')
        
        # Highlight prototypes
        prototype_2d = X_2d[self.prototype_indices_]
        ax1.scatter(prototype_2d[:, 0], prototype_2d[:, 1], 
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                   label='Prototypes')
        
        # Highlight criticisms if available
        if self.criticisms_ is not None:
            criticism_2d = X_2d[self.criticism_indices_]
            ax1.scatter(criticism_2d[:, 0], criticism_2d[:, 1], 
                       c='orange', s=150, marker='X', edgecolors='black', linewidth=2,
                       label='Criticisms')
        
        ax1.set_title(f'Prototypes and Data Points ({method.upper()})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coverage visualization
        if len(self.prototypes_) > 0:
            # Assign each point to nearest prototype
            distances_to_prototypes = cdist(X, self.prototypes_, metric=self.distance_metric)
            nearest_prototype = np.argmin(distances_to_prototypes, axis=1)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.prototypes_)))
            for i in range(len(self.prototypes_)):
                mask = nearest_prototype == i
                if np.any(mask):
                    ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                              c=[colors[i]], alpha=0.6, label=f'Prototype {i}')
            
            # Plot prototypes
            ax2.scatter(prototype_2d[:, 0], prototype_2d[:, 1], 
                       c='black', s=200, marker='*', edgecolors='white', linewidth=2)
            
            ax2.set_title('Prototype Coverage Areas')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
                       target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive prototype analysis report.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            target_names: Names of target classes
            
        Returns:
            Dictionary with complete analysis
        """
        if self.prototypes_ is None:
            raise ValueError("Must select prototypes first")
        
        report = {
            'summary': {
                'n_prototypes': len(self.prototypes_),
                'n_criticisms': len(self.criticisms_) if self.criticisms_ is not None else 0,
                'total_data_points': len(X),
                'distance_metric': self.distance_metric
            },
            'prototypes': [],
            'criticisms': []
        }
        
        # Analyze each prototype
        for i in range(len(self.prototypes_)):
            prototype_explanation = self.explain_prototype(i, X, feature_names)
            report['prototypes'].append(prototype_explanation)
        
        # Analyze criticisms
        if self.criticisms_ is not None:
            for i, criticism_idx in enumerate(self.criticism_indices_):
                criticism_info = {
                    'criticism_index': int(criticism_idx),
                    'criticism_values': X[criticism_idx].tolist(),
                    'distance_to_nearest_prototype': float(np.min(
                        cdist([X[criticism_idx]], self.prototypes_, metric=self.distance_metric)
                    ))
                }
                if feature_names:
                    criticism_info['feature_names'] = feature_names
                
                report['criticisms'].append(criticism_info)
        
        # Coverage analysis
        if len(self.prototypes_) > 0:
            distances_to_prototypes = cdist(X, self.prototypes_, metric=self.distance_metric)
            min_distances = np.min(distances_to_prototypes, axis=1)
            
            report['coverage_statistics'] = {
                'mean_distance_to_nearest_prototype': float(np.mean(min_distances)),
                'median_distance_to_nearest_prototype': float(np.median(min_distances)),
                'max_distance_to_nearest_prototype': float(np.max(min_distances)),
                'std_distance_to_nearest_prototype': float(np.std(min_distances))
            }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save prototype analysis report to file.
        
        Args:
            report: Report dictionary from generate_report()
            filepath: Path to save the report
        """
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def fit_transform(self, X: np.ndarray, method: str = 'k_medoids') -> Dict[str, np.ndarray]:
        """Fit prototype selector and return prototypes and criticisms.
        
        Args:
            X: Feature matrix
            method: Prototype selection method
            
        Returns:
            Dictionary with prototypes and criticisms
        """
        prototypes, prototype_indices = self.select_prototypes(X, method)
        criticisms, criticism_indices = self.select_criticisms(X)
        
        return {
            'prototypes': prototypes,
            'prototype_indices': prototype_indices,
            'criticisms': criticisms,
            'criticism_indices': criticism_indices
        }


class ExampleBasedExplainer:
    """Explain predictions using similar examples from training data."""
    
    def __init__(self, k_neighbors: int = 5, distance_metric: str = 'euclidean'):
        """Initialize example-based explainer.
        
        Args:
            k_neighbors: Number of nearest neighbors to find
            distance_metric: Distance metric for similarity
        """
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the explainer with training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        self.X_train_ = X_train
        self.y_train_ = y_train
        
    def explain_prediction(self, x_query: np.ndarray, 
                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Explain a prediction using similar training examples.
        
        Args:
            x_query: Query instance to explain
            feature_names: Names of features
            
        Returns:
            Dictionary with explanation
        """
        if self.X_train_ is None:
            raise ValueError("Must fit explainer first")
        
        # Find k nearest neighbors
        distances = cdist([x_query], self.X_train_, metric=self.distance_metric)[0]
        neighbor_indices = np.argsort(distances)[:self.k_neighbors]
        neighbor_distances = distances[neighbor_indices]
        
        explanation = {
            'query_instance': x_query.tolist(),
            'nearest_neighbors': [],
            'class_distribution': {}
        }
        
        # Analyze nearest neighbors
        for i, neighbor_idx in enumerate(neighbor_indices):
            neighbor_info = {
                'index': int(neighbor_idx),
                'distance': float(neighbor_distances[i]),
                'features': self.X_train_[neighbor_idx].tolist(),
                'label': self.y_train_[neighbor_idx]
            }
            
            if feature_names:
                neighbor_info['feature_names'] = feature_names
                # Compute feature differences
                feature_diffs = np.abs(x_query - self.X_train_[neighbor_idx])
                neighbor_info['feature_differences'] = {
                    name: float(diff) 
                    for name, diff in zip(feature_names, feature_diffs)
                }
            
            explanation['nearest_neighbors'].append(neighbor_info)
        
        # Compute class distribution among neighbors
        neighbor_labels = self.y_train_[neighbor_indices]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            explanation['class_distribution'][str(label)] = int(count)
        
        return explanation