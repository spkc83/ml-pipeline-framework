"""Specialized fraud detection explanations and risk analysis."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
import networkx as nx
warnings.filterwarnings('ignore')


class FraudRiskScorer:
    """Calculate fraud risk scores with detailed explanations."""
    
    def __init__(self, model, feature_names: List[str]):
        """Initialize fraud risk scorer.
        
        Args:
            model: Trained fraud detection model
            feature_names: Names of features used by the model
        """
        self.model = model
        self.feature_names = feature_names
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
    def calculate_risk_score(self, transaction: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Calculate comprehensive fraud risk score.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dictionary with risk score and breakdown
        """
        if isinstance(transaction, pd.Series):
            transaction_array = transaction.values.reshape(1, -1)
        else:
            transaction_array = transaction.reshape(1, -1)
        
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            fraud_probability = self.model.predict_proba(transaction_array)[0, 1]
        else:
            fraud_probability = self.model.predict(transaction_array)[0]
        
        # Determine risk level
        if fraud_probability < self.risk_thresholds['low']:
            risk_level = 'low'
        elif fraud_probability < self.risk_thresholds['medium']:
            risk_level = 'medium'
        elif fraud_probability < self.risk_thresholds['high']:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        # Feature importance analysis
        feature_contributions = self._analyze_feature_contributions(transaction_array)
        
        return {
            'fraud_probability': float(fraud_probability),
            'risk_level': risk_level,
            'risk_score': int(fraud_probability * 100),
            'feature_contributions': feature_contributions,
            'risk_factors': self._identify_risk_factors(transaction, feature_contributions),
            'recommendation': self._generate_recommendation(risk_level, fraud_probability)
        }
    
    def _analyze_feature_contributions(self, transaction: np.ndarray) -> Dict[str, float]:
        """Analyze how each feature contributes to fraud probability."""
        contributions = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            for i, feature in enumerate(self.feature_names):
                contributions[feature] = float(importances[i] * transaction[0, i])
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            for i, feature in enumerate(self.feature_names):
                contributions[feature] = float(coefficients[i] * transaction[0, i])
        else:
            # Default to uniform contribution
            for feature in self.feature_names:
                contributions[feature] = 1.0 / len(self.feature_names)
        
        return contributions
    
    def _identify_risk_factors(self, transaction: Union[np.ndarray, pd.Series], 
                              contributions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify key risk factors for the transaction."""
        risk_factors = []
        
        # Sort features by contribution magnitude
        sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, contribution in sorted_features[:5]:  # Top 5 risk factors
            if isinstance(transaction, pd.Series):
                feature_value = transaction[feature]
            else:
                feature_idx = self.feature_names.index(feature)
                feature_value = transaction[feature_idx]
            
            risk_factor = {
                'feature': feature,
                'value': float(feature_value),
                'contribution': float(contribution),
                'impact': 'high' if abs(contribution) > 0.1 else 'medium' if abs(contribution) > 0.05 else 'low',
                'direction': 'increases' if contribution > 0 else 'decreases'
            }
            
            risk_factors.append(risk_factor)
        
        return risk_factors
    
    def _generate_recommendation(self, risk_level: str, probability: float) -> Dict[str, Any]:
        """Generate action recommendations based on risk level."""
        recommendations = {
            'low': {
                'action': 'approve',
                'monitoring': 'standard',
                'review_required': False,
                'message': 'Transaction appears legitimate. Proceed with standard processing.'
            },
            'medium': {
                'action': 'review',
                'monitoring': 'enhanced',
                'review_required': True,
                'message': 'Transaction requires additional verification. Implement enhanced monitoring.'
            },
            'high': {
                'action': 'hold',
                'monitoring': 'intensive',
                'review_required': True,
                'message': 'High fraud risk detected. Hold transaction for manual review.'
            },
            'critical': {
                'action': 'block',
                'monitoring': 'investigation',
                'review_required': True,
                'message': 'Critical fraud risk. Block transaction and initiate investigation.'
            }
        }
        
        recommendation = recommendations[risk_level].copy()
        recommendation['confidence'] = float(max(probability, 1 - probability))
        
        return recommendation


class TransactionPatternAnalyzer:
    """Analyze transaction patterns for fraud detection."""
    
    def __init__(self):
        """Initialize transaction pattern analyzer."""
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_anomalous_patterns(self, transactions: pd.DataFrame,
                                 user_id_col: str = 'user_id',
                                 amount_col: str = 'amount',
                                 timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """Detect anomalous transaction patterns.
        
        Args:
            transactions: Transaction dataset
            user_id_col: Column name for user ID
            amount_col: Column name for transaction amount
            timestamp_col: Column name for timestamp
            
        Returns:
            Dictionary with anomaly analysis results
        """
        # Create features for anomaly detection
        features = self._create_pattern_features(transactions, user_id_col, amount_col, timestamp_col)
        
        # Fit anomaly detector
        features_scaled = self.scaler.fit_transform(features)
        anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
        anomaly_probs = self.anomaly_detector.score_samples(features_scaled)
        
        # Identify anomalous transactions
        anomalies = transactions[anomaly_scores == -1].copy()
        anomalies['anomaly_score'] = anomaly_probs[anomaly_scores == -1]
        
        return {
            'n_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(transactions),
            'anomalous_transactions': anomalies.to_dict('records'),
            'pattern_summary': self._summarize_patterns(anomalies, amount_col, timestamp_col)
        }
    
    def _create_pattern_features(self, transactions: pd.DataFrame,
                                user_id_col: str, amount_col: str, timestamp_col: str) -> np.ndarray:
        """Create features for pattern analysis."""
        features = []
        
        # Convert timestamp to datetime if needed
        if transactions[timestamp_col].dtype == 'object':
            transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])
        
        # Time-based features
        transactions['hour'] = transactions[timestamp_col].dt.hour
        transactions['day_of_week'] = transactions[timestamp_col].dt.dayofweek
        transactions['is_weekend'] = transactions['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        transactions['log_amount'] = np.log1p(transactions[amount_col])
        
        # User-based features (if user_id available)
        if user_id_col in transactions.columns:
            user_stats = transactions.groupby(user_id_col)[amount_col].agg(['mean', 'std', 'count'])
            transactions = transactions.merge(user_stats, left_on=user_id_col, right_index=True, 
                                           suffixes=('', '_user_avg'))
            transactions['amount_vs_user_avg'] = transactions[amount_col] / transactions['mean']
        
        # Select numeric features for anomaly detection
        numeric_features = ['hour', 'day_of_week', 'is_weekend', 'log_amount']
        if 'amount_vs_user_avg' in transactions.columns:
            numeric_features.append('amount_vs_user_avg')
        
        features = transactions[numeric_features].fillna(0).values
        
        return features
    
    def _summarize_patterns(self, anomalies: pd.DataFrame,
                           amount_col: str, timestamp_col: str) -> Dict[str, Any]:
        """Summarize patterns in anomalous transactions."""
        if len(anomalies) == 0:
            return {'message': 'No anomalies detected'}
        
        summary = {
            'amount_statistics': {
                'mean': float(anomalies[amount_col].mean()),
                'median': float(anomalies[amount_col].median()),
                'std': float(anomalies[amount_col].std()),
                'min': float(anomalies[amount_col].min()),
                'max': float(anomalies[amount_col].max())
            },
            'temporal_patterns': {},
            'common_characteristics': []
        }
        
        # Temporal patterns
        if 'hour' in anomalies.columns:
            summary['temporal_patterns']['most_common_hours'] = anomalies['hour'].value_counts().head(3).to_dict()
        
        if 'day_of_week' in anomalies.columns:
            summary['temporal_patterns']['most_common_days'] = anomalies['day_of_week'].value_counts().head(3).to_dict()
        
        # High-value transactions
        high_value_threshold = anomalies[amount_col].quantile(0.9)
        high_value_count = len(anomalies[anomalies[amount_col] > high_value_threshold])
        if high_value_count > 0:
            summary['common_characteristics'].append(f"{high_value_count} high-value transactions (>${high_value_threshold:.2f})")
        
        return summary
    
    def analyze_velocity_patterns(self, transactions: pd.DataFrame,
                                 user_id_col: str = 'user_id',
                                 timestamp_col: str = 'timestamp',
                                 time_window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze transaction velocity patterns.
        
        Args:
            transactions: Transaction dataset
            user_id_col: Column name for user ID
            timestamp_col: Column name for timestamp
            time_window_minutes: Time window for velocity calculation
            
        Returns:
            Dictionary with velocity analysis results
        """
        if transactions[timestamp_col].dtype == 'object':
            transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])
        
        velocity_alerts = []
        
        for user_id in transactions[user_id_col].unique():
            user_txns = transactions[transactions[user_id_col] == user_id].sort_values(timestamp_col)
            
            for i, txn in user_txns.iterrows():
                # Count transactions in time window
                window_start = txn[timestamp_col] - timedelta(minutes=time_window_minutes)
                window_txns = user_txns[
                    (user_txns[timestamp_col] >= window_start) & 
                    (user_txns[timestamp_col] <= txn[timestamp_col])
                ]
                
                velocity = len(window_txns)
                
                # Flag high velocity (more than 10 transactions per hour)
                if velocity > 10:
                    velocity_alerts.append({
                        'user_id': user_id,
                        'timestamp': txn[timestamp_col],
                        'velocity': velocity,
                        'window_minutes': time_window_minutes,
                        'risk_level': 'high' if velocity > 20 else 'medium'
                    })
        
        return {
            'velocity_alerts': velocity_alerts,
            'n_users_with_alerts': len(set(alert['user_id'] for alert in velocity_alerts)),
            'max_velocity': max([alert['velocity'] for alert in velocity_alerts]) if velocity_alerts else 0
        }


class FraudNetworkAnalyzer:
    """Analyze fraud networks and suspicious connections."""
    
    def __init__(self):
        """Initialize fraud network analyzer."""
        self.graph = nx.Graph()
        
    def build_transaction_network(self, transactions: pd.DataFrame,
                                 from_col: str = 'from_account',
                                 to_col: str = 'to_account',
                                 amount_col: str = 'amount') -> nx.Graph:
        """Build a network graph from transactions.
        
        Args:
            transactions: Transaction dataset
            from_col: Column name for source account
            to_col: Column name for destination account
            amount_col: Column name for transaction amount
            
        Returns:
            NetworkX graph
        """
        self.graph.clear()
        
        for _, txn in transactions.iterrows():
            from_account = txn[from_col]
            to_account = txn[to_col]
            amount = txn[amount_col]
            
            # Add edge or update weight
            if self.graph.has_edge(from_account, to_account):
                self.graph[from_account][to_account]['weight'] += amount
                self.graph[from_account][to_account]['count'] += 1
            else:
                self.graph.add_edge(from_account, to_account, weight=amount, count=1)
        
        return self.graph
    
    def detect_suspicious_patterns(self) -> Dict[str, Any]:
        """Detect suspicious patterns in the transaction network.
        
        Returns:
            Dictionary with suspicious pattern analysis
        """
        suspicious_patterns = {
            'circular_flows': self._detect_circular_flows(),
            'high_degree_nodes': self._detect_high_degree_nodes(),
            'isolated_clusters': self._detect_isolated_clusters(),
            'rapid_fire_transactions': self._detect_rapid_fire_patterns()
        }
        
        return suspicious_patterns
    
    def _detect_circular_flows(self) -> List[Dict[str, Any]]:
        """Detect circular money flows (potential money laundering)."""
        cycles = []
        
        # Find cycles of length 3-5
        for cycle_length in range(3, 6):
            try:
                simple_cycles = list(nx.simple_cycles(self.graph.to_directed()))
                for cycle in simple_cycles:
                    if len(cycle) == cycle_length:
                        # Calculate total flow in cycle
                        total_amount = 0
                        for i in range(len(cycle)):
                            from_node = cycle[i]
                            to_node = cycle[(i + 1) % len(cycle)]
                            if self.graph.has_edge(from_node, to_node):
                                total_amount += self.graph[from_node][to_node]['weight']
                        
                        cycles.append({
                            'cycle': cycle,
                            'length': cycle_length,
                            'total_amount': total_amount,
                            'avg_amount': total_amount / cycle_length
                        })
            except:
                continue
        
        # Sort by total amount
        cycles.sort(key=lambda x: x['total_amount'], reverse=True)
        return cycles[:10]  # Return top 10
    
    def _detect_high_degree_nodes(self, threshold: int = 50) -> List[Dict[str, Any]]:
        """Detect nodes with unusually high degree (potential hubs)."""
        high_degree_nodes = []
        
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree > threshold:
                # Calculate total transaction volume
                total_volume = sum(self.graph[node][neighbor]['weight'] 
                                 for neighbor in self.graph.neighbors(node))
                
                high_degree_nodes.append({
                    'node': node,
                    'degree': degree,
                    'total_volume': total_volume,
                    'avg_transaction': total_volume / degree
                })
        
        high_degree_nodes.sort(key=lambda x: x['degree'], reverse=True)
        return high_degree_nodes
    
    def _detect_isolated_clusters(self, min_size: int = 5) -> List[Dict[str, Any]]:
        """Detect isolated clusters (potential fraud rings)."""
        clusters = []
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        for component in components:
            if len(component) >= min_size:
                subgraph = self.graph.subgraph(component)
                
                # Calculate cluster metrics
                total_volume = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
                internal_edges = len(subgraph.edges())
                
                # Check isolation (few external connections)
                external_connections = 0
                for node in component:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in component:
                            external_connections += 1
                
                isolation_ratio = external_connections / len(component)
                
                if isolation_ratio < 0.1:  # Highly isolated
                    clusters.append({
                        'nodes': list(component),
                        'size': len(component),
                        'total_volume': total_volume,
                        'internal_edges': internal_edges,
                        'isolation_ratio': isolation_ratio
                    })
        
        clusters.sort(key=lambda x: x['size'], reverse=True)
        return clusters
    
    def _detect_rapid_fire_patterns(self) -> List[Dict[str, Any]]:
        """Detect rapid-fire transaction patterns."""
        rapid_patterns = []
        
        for edge in self.graph.edges(data=True):
            from_node, to_node, data = edge
            count = data['count']
            total_amount = data['weight']
            
            # Flag edges with many transactions
            if count > 20:  # More than 20 transactions between same accounts
                rapid_patterns.append({
                    'from_account': from_node,
                    'to_account': to_node,
                    'transaction_count': count,
                    'total_amount': total_amount,
                    'avg_amount': total_amount / count
                })
        
        rapid_patterns.sort(key=lambda x: x['transaction_count'], reverse=True)
        return rapid_patterns[:20]
    
    def visualize_network(self, suspicious_nodes: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Visualize the transaction network.
        
        Args:
            suspicious_nodes: List of nodes to highlight
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use only subset of nodes for visualization if graph is too large
        if len(self.graph.nodes()) > 100:
            # Select top nodes by degree
            top_nodes = sorted(self.graph.nodes(), 
                             key=lambda x: self.graph.degree(x), reverse=True)[:50]
            subgraph = self.graph.subgraph(top_nodes)
        else:
            subgraph = self.graph
        
        # Position nodes
        pos = nx.spring_layout(subgraph, k=3, seed=42)
        
        # Color nodes
        node_colors = []
        for node in subgraph.nodes():
            if suspicious_nodes and node in suspicious_nodes:
                node_colors.append('red')
            elif subgraph.degree(node) > 10:
                node_colors.append('orange')
            else:
                node_colors.append('lightblue')
        
        # Size nodes by degree
        node_sizes = [subgraph.degree(node) * 20 + 100 for node in subgraph.nodes()]
        
        # Draw network
        nx.draw(subgraph, pos, node_color=node_colors, node_size=node_sizes,
                with_labels=True, font_size=8, font_weight='bold',
                edge_color='gray', alpha=0.7, ax=ax)
        
        ax.set_title('Transaction Network\n(Red: Suspicious, Orange: High Activity, Blue: Normal)')
        ax.axis('off')
        
        return fig


class FraudExplanationGenerator:
    """Generate comprehensive fraud explanations combining multiple analysis methods."""
    
    def __init__(self, model, feature_names: List[str]):
        """Initialize fraud explanation generator.
        
        Args:
            model: Trained fraud detection model
            feature_names: Names of features used by the model
        """
        self.risk_scorer = FraudRiskScorer(model, feature_names)
        self.pattern_analyzer = TransactionPatternAnalyzer()
        self.network_analyzer = FraudNetworkAnalyzer()
        
    def explain_transaction(self, transaction: Union[np.ndarray, pd.Series],
                          transaction_history: Optional[pd.DataFrame] = None,
                          network_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate comprehensive explanation for a single transaction.
        
        Args:
            transaction: Transaction to explain
            transaction_history: Historical transactions for pattern analysis
            network_data: Network transaction data
            
        Returns:
            Dictionary with comprehensive explanation
        """
        explanation = {
            'transaction_id': getattr(transaction, 'name', 'unknown'),
            'risk_analysis': self.risk_scorer.calculate_risk_score(transaction),
            'pattern_analysis': None,
            'network_analysis': None,
            'summary': {}
        }
        
        # Pattern analysis if history available
        if transaction_history is not None:
            try:
                pattern_results = self.pattern_analyzer.detect_anomalous_patterns(transaction_history)
                explanation['pattern_analysis'] = pattern_results
            except Exception as e:
                explanation['pattern_analysis'] = {'error': str(e)}
        
        # Network analysis if network data available
        if network_data is not None:
            try:
                network_graph = self.network_analyzer.build_transaction_network(network_data)
                network_patterns = self.network_analyzer.detect_suspicious_patterns()
                explanation['network_analysis'] = network_patterns
            except Exception as e:
                explanation['network_analysis'] = {'error': str(e)}
        
        # Generate summary
        explanation['summary'] = self._generate_explanation_summary(explanation)
        
        return explanation
    
    def batch_explain_transactions(self, transactions: pd.DataFrame,
                                  include_network: bool = False) -> Dict[str, Any]:
        """Generate explanations for a batch of transactions.
        
        Args:
            transactions: Batch of transactions to explain
            include_network: Whether to include network analysis
            
        Returns:
            Dictionary with batch explanation results
        """
        batch_results = {
            'transaction_explanations': [],
            'batch_patterns': None,
            'batch_statistics': {},
            'high_risk_transactions': []
        }
        
        # Analyze each transaction
        for idx, transaction in transactions.iterrows():
            try:
                risk_analysis = self.risk_scorer.calculate_risk_score(transaction)
                
                explanation = {
                    'transaction_id': idx,
                    'risk_analysis': risk_analysis
                }
                
                batch_results['transaction_explanations'].append(explanation)
                
                # Track high-risk transactions
                if risk_analysis['risk_level'] in ['high', 'critical']:
                    batch_results['high_risk_transactions'].append({
                        'transaction_id': idx,
                        'risk_score': risk_analysis['risk_score'],
                        'risk_level': risk_analysis['risk_level']
                    })
                    
            except Exception as e:
                batch_results['transaction_explanations'].append({
                    'transaction_id': idx,
                    'error': str(e)
                })
        
        # Batch-level pattern analysis
        try:
            batch_results['batch_patterns'] = self.pattern_analyzer.detect_anomalous_patterns(transactions)
        except Exception as e:
            batch_results['batch_patterns'] = {'error': str(e)}
        
        # Network analysis if requested
        if include_network and 'from_account' in transactions.columns and 'to_account' in transactions.columns:
            try:
                network_graph = self.network_analyzer.build_transaction_network(transactions)
                batch_results['network_patterns'] = self.network_analyzer.detect_suspicious_patterns()
            except Exception as e:
                batch_results['network_patterns'] = {'error': str(e)}
        
        # Batch statistics
        risk_scores = [exp['risk_analysis']['risk_score'] for exp in batch_results['transaction_explanations'] 
                      if 'risk_analysis' in exp]
        
        batch_results['batch_statistics'] = {
            'total_transactions': len(transactions),
            'high_risk_count': len(batch_results['high_risk_transactions']),
            'high_risk_rate': len(batch_results['high_risk_transactions']) / len(transactions),
            'average_risk_score': np.mean(risk_scores) if risk_scores else 0,
            'max_risk_score': max(risk_scores) if risk_scores else 0
        }
        
        return batch_results
    
    def _generate_explanation_summary(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a human-readable summary of the explanation."""
        risk_analysis = explanation['risk_analysis']
        
        summary = {
            'primary_concern': risk_analysis['risk_level'],
            'confidence': risk_analysis['fraud_probability'],
            'key_findings': [],
            'recommended_action': risk_analysis['recommendation']['action']
        }
        
        # Add key risk factors
        top_risk_factors = risk_analysis['risk_factors'][:3]
        for factor in top_risk_factors:
            finding = f"{factor['feature']} {factor['direction']} fraud risk"
            summary['key_findings'].append(finding)
        
        # Add pattern analysis findings
        if explanation['pattern_analysis'] and 'n_anomalies' in explanation['pattern_analysis']:
            if explanation['pattern_analysis']['n_anomalies'] > 0:
                summary['key_findings'].append("Anomalous transaction patterns detected")
        
        # Add network analysis findings
        if explanation['network_analysis']:
            if explanation['network_analysis'].get('circular_flows'):
                summary['key_findings'].append("Potential circular money flows detected")
            if explanation['network_analysis'].get('high_degree_nodes'):
                summary['key_findings'].append("High-activity network nodes involved")
        
        return summary
    
    def generate_fraud_report(self, transactions: pd.DataFrame,
                            output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive fraud analysis report.
        
        Args:
            transactions: Transaction dataset
            output_path: Path to save report
            
        Returns:
            Dictionary with complete fraud analysis
        """
        report = {
            'executive_summary': {},
            'batch_analysis': self.batch_explain_transactions(transactions, include_network=True),
            'pattern_insights': {},
            'network_insights': {},
            'recommendations': []
        }
        
        # Executive summary
        batch_stats = report['batch_analysis']['batch_statistics']
        report['executive_summary'] = {
            'total_transactions_analyzed': batch_stats['total_transactions'],
            'fraud_detection_rate': f"{batch_stats['high_risk_rate']:.1%}",
            'average_risk_score': f"{batch_stats['average_risk_score']:.1f}/100",
            'highest_risk_score': f"{batch_stats['max_risk_score']}/100",
            'critical_alerts': len([t for t in report['batch_analysis']['high_risk_transactions'] 
                                  if t['risk_level'] == 'critical'])
        }
        
        # Pattern insights
        if 'batch_patterns' in report['batch_analysis'] and report['batch_analysis']['batch_patterns']:
            pattern_data = report['batch_analysis']['batch_patterns']
            report['pattern_insights'] = {
                'anomaly_rate': f"{pattern_data.get('anomaly_rate', 0):.1%}",
                'pattern_summary': pattern_data.get('pattern_summary', {})
            }
        
        # Network insights
        if 'network_patterns' in report['batch_analysis']:
            network_data = report['batch_analysis']['network_patterns']
            report['network_insights'] = {
                'suspicious_cycles': len(network_data.get('circular_flows', [])),
                'high_activity_nodes': len(network_data.get('high_degree_nodes', [])),
                'isolated_clusters': len(network_data.get('isolated_clusters', []))
            }
        
        # Generate recommendations
        if batch_stats['high_risk_rate'] > 0.1:
            report['recommendations'].append("High fraud rate detected. Implement additional verification steps.")
        
        if report['network_insights'].get('suspicious_cycles', 0) > 0:
            report['recommendations'].append("Circular transaction patterns detected. Investigate for money laundering.")
        
        if batch_stats['average_risk_score'] > 50:
            report['recommendations'].append("Elevated average risk score. Review fraud detection thresholds.")
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report