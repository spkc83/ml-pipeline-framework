"""Enhanced fraud detection visualizations including animated patterns, 3D spaces, and business dashboards."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
import networkx as nx
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
warnings.filterwarnings('ignore')

# Try to import additional visualization libraries
try:
    import plotly.offline as pyo
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not available for dimensionality reduction")


class AnimatedFraudPatternVisualizer:
    """Create animated visualizations showing fraud pattern evolution over time."""
    
    def __init__(self, transactions: pd.DataFrame, fraud_column: str = 'is_fraud',
                 timestamp_column: str = 'timestamp'):
        """Initialize animated fraud pattern visualizer.
        
        Args:
            transactions: Transaction dataset
            fraud_column: Column indicating fraud (1 for fraud, 0 for legitimate)
            timestamp_column: Column with timestamp information
        """
        self.transactions = transactions.copy()
        self.fraud_column = fraud_column
        self.timestamp_column = timestamp_column
        
        # Ensure timestamp is datetime
        if self.transactions[timestamp_column].dtype == 'object':
            self.transactions[timestamp_column] = pd.to_datetime(self.transactions[timestamp_column])
    
    def create_animated_scatter(self, x_column: str, y_column: str, 
                               time_window_hours: int = 24,
                               save_path: Optional[str] = None) -> FuncAnimation:
        """Create animated scatter plot showing fraud patterns over time.
        
        Args:
            x_column: Column for x-axis
            y_column: Column for y-axis  
            time_window_hours: Time window for each frame in hours
            save_path: Path to save animation (optional)
            
        Returns:
            Animation object
        """
        # Create time bins
        min_time = self.transactions[self.timestamp_column].min()
        max_time = self.transactions[self.timestamp_column].max()
        time_bins = pd.date_range(start=min_time, end=max_time, 
                                 freq=f'{time_window_hours}H')
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            """Animation function for each frame."""
            ax.clear()
            
            if frame < len(time_bins) - 1:
                # Get data for current time window
                start_time = time_bins[frame]
                end_time = time_bins[frame + 1]
                
                window_data = self.transactions[
                    (self.transactions[self.timestamp_column] >= start_time) &
                    (self.transactions[self.timestamp_column] < end_time)
                ]
                
                if len(window_data) > 0:
                    # Separate fraud and legitimate transactions
                    fraud_data = window_data[window_data[self.fraud_column] == 1]
                    legit_data = window_data[window_data[self.fraud_column] == 0]
                    
                    # Plot legitimate transactions
                    if len(legit_data) > 0:
                        ax.scatter(legit_data[x_column], legit_data[y_column], 
                                 c='blue', alpha=0.6, s=30, label='Legitimate')
                    
                    # Plot fraud transactions
                    if len(fraud_data) > 0:
                        ax.scatter(fraud_data[x_column], fraud_data[y_column], 
                                 c='red', alpha=0.8, s=60, marker='X', label='Fraud')
                    
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                    ax.set_title(f'Fraud Patterns: {start_time.strftime("%Y-%m-%d %H:%M")} - {end_time.strftime("%Y-%m-%d %H:%M")}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    fraud_rate = len(fraud_data) / len(window_data) if len(window_data) > 0 else 0
                    ax.text(0.02, 0.98, f'Transactions: {len(window_data)}\nFraud Rate: {fraud_rate:.2%}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(time_bins)-1, 
                           interval=1000, repeat=True, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
        
        return anim
    
    def create_fraud_heatmap_animation(self, amount_bins: int = 20, time_bins: int = 24,
                                     save_path: Optional[str] = None) -> FuncAnimation:
        """Create animated heatmap showing fraud intensity over amount and time.
        
        Args:
            amount_bins: Number of amount bins
            time_bins: Number of time bins per day
            save_path: Path to save animation
            
        Returns:
            Animation object
        """
        # Add hour column
        self.transactions['hour'] = self.transactions[self.timestamp_column].dt.hour
        
        # Create amount bins
        amount_min = self.transactions['amount'].min()
        amount_max = self.transactions['amount'].max()
        amount_edges = np.linspace(amount_min, amount_max, amount_bins + 1)
        self.transactions['amount_bin'] = pd.cut(self.transactions['amount'], 
                                               bins=amount_edges, labels=False)
        
        # Get unique days
        self.transactions['date'] = self.transactions[self.timestamp_column].dt.date
        unique_dates = sorted(self.transactions['date'].unique())
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            """Animation function for each frame."""
            ax.clear()
            
            if frame < len(unique_dates):
                current_date = unique_dates[frame]
                day_data = self.transactions[self.transactions['date'] == current_date]
                
                if len(day_data) > 0:
                    # Create heatmap data
                    heatmap_data = np.zeros((amount_bins, 24))
                    
                    for _, row in day_data.iterrows():
                        if not pd.isna(row['amount_bin']) and 0 <= row['hour'] < 24:
                            amount_idx = int(row['amount_bin'])
                            hour_idx = int(row['hour'])
                            if row[self.fraud_column] == 1:
                                heatmap_data[amount_idx, hour_idx] += 1
                    
                    # Create heatmap
                    im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', origin='lower')
                    
                    # Set labels
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Amount Bin')
                    ax.set_title(f'Fraud Intensity Heatmap: {current_date}')
                    
                    # Set ticks
                    ax.set_xticks(range(0, 24, 4))
                    ax.set_xticklabels(range(0, 24, 4))
                    
                    y_labels = [f'${amount_edges[i]:.0f}-${amount_edges[i+1]:.0f}' 
                               for i in range(0, amount_bins, 5)]
                    ax.set_yticks(range(0, amount_bins, 5))
                    ax.set_yticklabels(y_labels)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Fraud Count')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(unique_dates), 
                           interval=2000, repeat=True, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=0.5)
        
        return anim


class Interactive3DFeatureSpace:
    """Create interactive 3D visualizations of feature spaces for fraud detection."""
    
    def __init__(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
        """Initialize 3D feature space visualizer.
        
        Args:
            X: Feature matrix
            y: Target labels (0 for legitimate, 1 for fraud)
            feature_names: Names of features
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. 3D visualizations will be limited.")
    
    def create_3d_scatter(self, features: List[str], sample_size: int = 5000) -> go.Figure:
        """Create interactive 3D scatter plot.
        
        Args:
            features: List of 3 feature names for x, y, z axes
            sample_size: Number of points to display
            
        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D visualizations")
        
        if len(features) != 3:
            raise ValueError("Exactly 3 features required for 3D visualization")
        
        # Sample data for performance
        indices = np.random.choice(len(self.X), min(sample_size, len(self.X)), replace=False)
        X_sample = self.X.iloc[indices]
        y_sample = self.y[indices]
        
        # Create traces for fraud and legitimate transactions
        fraud_mask = y_sample == 1
        legit_mask = y_sample == 0
        
        fig = go.Figure()
        
        # Legitimate transactions
        if np.any(legit_mask):
            fig.add_trace(go.Scatter3d(
                x=X_sample.loc[legit_mask, features[0]],
                y=X_sample.loc[legit_mask, features[1]],
                z=X_sample.loc[legit_mask, features[2]],
                mode='markers',
                marker=dict(
                    size=3,
                    color='blue',
                    opacity=0.6
                ),
                name='Legitimate',
                text=[f'Legitimate<br>{features[0]}: {x:.2f}<br>{features[1]}: {y:.2f}<br>{features[2]}: {z:.2f}'
                     for x, y, z in zip(X_sample.loc[legit_mask, features[0]],
                                       X_sample.loc[legit_mask, features[1]],
                                       X_sample.loc[legit_mask, features[2]])],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Fraud transactions
        if np.any(fraud_mask):
            fig.add_trace(go.Scatter3d(
                x=X_sample.loc[fraud_mask, features[0]],
                y=X_sample.loc[fraud_mask, features[1]],
                z=X_sample.loc[fraud_mask, features[2]],
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    opacity=0.8,
                    symbol='x'
                ),
                name='Fraud',
                text=[f'Fraud<br>{features[0]}: {x:.2f}<br>{features[1]}: {y:.2f}<br>{features[2]}: {z:.2f}'
                     for x, y, z in zip(X_sample.loc[fraud_mask, features[0]],
                                       X_sample.loc[fraud_mask, features[1]],
                                       X_sample.loc[fraud_mask, features[2]])],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='3D Feature Space: Fraud vs Legitimate Transactions',
            scene=dict(
                xaxis_title=features[0],
                yaxis_title=features[1],
                zaxis_title=features[2],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_3d_density_plot(self, features: List[str]) -> go.Figure:
        """Create 3D density plot showing fraud concentration.
        
        Args:
            features: List of 3 feature names
            
        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D visualizations")
        
        # Get fraud data
        fraud_data = self.X[self.y == 1]
        
        if len(fraud_data) == 0:
            raise ValueError("No fraud data available")
        
        # Create 3D histogram
        fig = go.Figure(data=go.Histogram2dContour(
            x=fraud_data[features[0]],
            y=fraud_data[features[1]],
            colorscale='Reds',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Fraud Density: {features[0]} vs {features[1]}',
            xaxis_title=features[0],
            yaxis_title=features[1]
        )
        
        return fig
    
    def create_dimensionality_reduction_3d(self, method: str = 'tsne') -> go.Figure:
        """Create 3D visualization using dimensionality reduction.
        
        Args:
            method: Reduction method ('tsne', 'pca')
            
        Returns:
            Plotly figure
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for dimensionality reduction")
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D visualizations")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        elif method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        X_reduced = reducer.fit_transform(X_scaled)
        
        # Create traces
        fraud_mask = self.y == 1
        legit_mask = self.y == 0
        
        fig = go.Figure()
        
        # Legitimate transactions
        if np.any(legit_mask):
            fig.add_trace(go.Scatter3d(
                x=X_reduced[legit_mask, 0],
                y=X_reduced[legit_mask, 1],
                z=X_reduced[legit_mask, 2],
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.6),
                name='Legitimate'
            ))
        
        # Fraud transactions  
        if np.any(fraud_mask):
            fig.add_trace(go.Scatter3d(
                x=X_reduced[fraud_mask, 0],
                y=X_reduced[fraud_mask, 1],
                z=X_reduced[fraud_mask, 2],
                mode='markers',
                marker=dict(size=6, color='red', opacity=0.8),
                name='Fraud'
            ))
        
        # Add explained variance for PCA
        title = f'3D {method.upper()} Visualization'
        if method == 'pca' and hasattr(reducer, 'explained_variance_ratio_'):
            var_explained = reducer.explained_variance_ratio_
            title += f' (Explained Variance: {var_explained.sum():.1%})'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                zaxis_title=f'{method.upper()} Component 3'
            ),
            width=800,
            height=600
        )
        
        return fig


class FraudNetworkGraphVisualizer:
    """Visualize fraud networks and suspicious transaction patterns."""
    
    def __init__(self, transactions: pd.DataFrame, 
                 from_col: str = 'from_account', to_col: str = 'to_account',
                 amount_col: str = 'amount', fraud_col: str = 'is_fraud'):
        """Initialize fraud network visualizer.
        
        Args:
            transactions: Transaction dataset
            from_col: Source account column
            to_col: Destination account column
            amount_col: Transaction amount column
            fraud_col: Fraud indicator column
        """
        self.transactions = transactions
        self.from_col = from_col
        self.to_col = to_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col
        
    def create_fraud_network_graph(self, min_fraud_transactions: int = 2) -> go.Figure:
        """Create interactive network graph of fraud-related accounts.
        
        Args:
            min_fraud_transactions: Minimum fraud transactions to include account
            
        Returns:
            Plotly network graph
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for network visualizations")
        
        # Filter fraud transactions
        fraud_transactions = self.transactions[self.transactions[self.fraud_col] == 1]
        
        if len(fraud_transactions) == 0:
            raise ValueError("No fraud transactions found")
        
        # Count fraud transactions per account
        from_counts = fraud_transactions[self.from_col].value_counts()
        to_counts = fraud_transactions[self.to_col].value_counts()
        all_counts = from_counts.add(to_counts, fill_value=0)
        
        # Filter accounts with minimum fraud transactions
        fraud_accounts = all_counts[all_counts >= min_fraud_transactions].index
        
        # Create network graph
        G = nx.Graph()
        
        for _, row in fraud_transactions.iterrows():
            from_acc = row[self.from_col]
            to_acc = row[self.to_col]
            amount = row[self.amount_col]
            
            if from_acc in fraud_accounts or to_acc in fraud_accounts:
                if G.has_edge(from_acc, to_acc):
                    G[from_acc][to_acc]['weight'] += amount
                    G[from_acc][to_acc]['count'] += 1
                else:
                    G.add_edge(from_acc, to_acc, weight=amount, count=1)
        
        if len(G.nodes()) == 0:
            raise ValueError("No network structure found")
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            count = G[edge[0]][edge[1]]['count']
            edge_info.append(f'Transactions: {count}<br>Total Amount: ${weight:,.2f}')
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Calculate node metrics
            degree = G.degree(node)
            fraud_count = all_counts.get(node, 0)
            
            node_text.append(f'Account: {node}<br>Connections: {degree}<br>Fraud Transactions: {fraud_count}')
            node_size.append(10 + degree * 2)
            node_color.append(fraud_count)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='Fraud Transactions'),
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Fraud Network Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Nodes sized by degree centrality, colored by fraud count",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def create_transaction_flow_sankey(self, top_accounts: int = 20) -> go.Figure:
        """Create Sankey diagram showing transaction flows.
        
        Args:
            top_accounts: Number of top accounts to include
            
        Returns:
            Plotly Sankey diagram
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for Sankey diagrams")
        
        # Get top accounts by transaction volume
        from_volumes = self.transactions.groupby(self.from_col)[self.amount_col].sum()
        to_volumes = self.transactions.groupby(self.to_col)[self.amount_col].sum()
        all_volumes = from_volumes.add(to_volumes, fill_value=0)
        top_accounts_list = all_volumes.nlargest(top_accounts).index.tolist()
        
        # Filter transactions
        filtered_transactions = self.transactions[
            (self.transactions[self.from_col].isin(top_accounts_list)) &
            (self.transactions[self.to_col].isin(top_accounts_list))
        ]
        
        # Create node labels and indices
        all_accounts = list(set(filtered_transactions[self.from_col].tolist() + 
                               filtered_transactions[self.to_col].tolist()))
        account_to_idx = {account: idx for idx, account in enumerate(all_accounts)}
        
        # Aggregate flows
        flows = filtered_transactions.groupby([self.from_col, self.to_col]).agg({
            self.amount_col: 'sum',
            self.fraud_col: 'sum'
        }).reset_index()
        
        # Create Sankey data
        source = [account_to_idx[acc] for acc in flows[self.from_col]]
        target = [account_to_idx[acc] for acc in flows[self.to_col]]
        value = flows[self.amount_col].tolist()
        
        # Color links based on fraud presence
        link_colors = ['red' if fraud > 0 else 'blue' for fraud in flows[self.fraud_col]]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_accounts,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title_text="Transaction Flow Network (Red: Contains Fraud, Blue: Clean)",
            font_size=10
        )
        
        return fig


class HierarchicalFeatureImportanceVisualizer:
    """Create hierarchical visualizations of feature importance with clustering."""
    
    def __init__(self, feature_importance: Dict[str, float], 
                 feature_categories: Optional[Dict[str, str]] = None):
        """Initialize hierarchical feature importance visualizer.
        
        Args:
            feature_importance: Dictionary of feature names to importance scores
            feature_categories: Optional categorization of features
        """
        self.feature_importance = feature_importance
        self.feature_categories = feature_categories or {}
        
    def create_treemap(self) -> go.Figure:
        """Create interactive treemap of feature importance.
        
        Returns:
            Plotly treemap figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for treemap visualization")
        
        # Prepare data
        features = list(self.feature_importance.keys())
        importances = list(self.feature_importance.values())
        
        # Create categories if not provided
        if not self.feature_categories:
            # Simple categorization based on feature name patterns
            categories = []
            for feature in features:
                if 'amount' in feature.lower():
                    categories.append('Amount Features')
                elif 'time' in feature.lower() or 'hour' in feature.lower():
                    categories.append('Temporal Features')
                elif 'merchant' in feature.lower():
                    categories.append('Merchant Features')
                elif 'customer' in feature.lower() or 'user' in feature.lower():
                    categories.append('Customer Features')
                else:
                    categories.append('Other Features')
        else:
            categories = [self.feature_categories.get(f, 'Other') for f in features]
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=features,
            values=importances,
            parents=categories,
            textinfo="label+value+percent parent",
            textfont_size=12,
            marker_colorscale='RdBu',
            marker_colorbar_title="Importance Score"
        ))
        
        fig.update_layout(
            title="Hierarchical Feature Importance Treemap",
            font_size=12
        )
        
        return fig
    
    def create_sunburst_chart(self) -> go.Figure:
        """Create sunburst chart of feature importance hierarchy.
        
        Returns:
            Plotly sunburst figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for sunburst visualization")
        
        # Prepare hierarchical data
        features = list(self.feature_importance.keys())
        importances = list(self.feature_importance.values())
        
        # Create category mapping
        if not self.feature_categories:
            category_map = {}
            for feature in features:
                if 'amount' in feature.lower():
                    category_map[feature] = 'Amount'
                elif 'time' in feature.lower():
                    category_map[feature] = 'Temporal'
                elif 'merchant' in feature.lower():
                    category_map[feature] = 'Merchant'
                else:
                    category_map[feature] = 'Other'
        else:
            category_map = self.feature_categories
        
        # Build sunburst data
        ids = ['All Features'] + list(set(category_map.values())) + features
        labels = ['All Features'] + list(set(category_map.values())) + features
        parents = [''] + ['All Features'] * len(set(category_map.values())) + \
                 [category_map[f] for f in features]
        values = [sum(importances)] + \
                [sum(imp for f, imp in self.feature_importance.items() 
                     if category_map[f] == cat) for cat in set(category_map.values())] + \
                importances
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        ))
        
        fig.update_layout(
            title="Feature Importance Hierarchy (Sunburst)",
            font_size=12
        )
        
        return fig
    
    def create_dendrogram(self) -> go.Figure:
        """Create dendrogram clustering features by importance patterns.
        
        Returns:
            Plotly dendrogram figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dendrogram visualization")
        
        # Create correlation matrix for clustering (simplified)
        features = list(self.feature_importance.keys())
        importances = np.array(list(self.feature_importance.values()))
        
        # Simple distance matrix based on importance differences
        n_features = len(features)
        distance_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                distance_matrix[i, j] = abs(importances[i] - importances[j])
        
        # Create dendrogram
        fig = ff.create_dendrogram(
            distance_matrix,
            labels=features,
            orientation='left'
        )
        
        fig.update_layout(
            title="Feature Importance Clustering Dendrogram",
            xaxis_title="Distance",
            yaxis_title="Features"
        )
        
        return fig


class BusinessMetricsDashboard:
    """Create comprehensive business metrics dashboard for fraud detection."""
    
    def __init__(self, transactions: pd.DataFrame, predictions: np.ndarray,
                 fraud_column: str = 'is_fraud', amount_column: str = 'amount'):
        """Initialize business metrics dashboard.
        
        Args:
            transactions: Transaction dataset
            predictions: Model predictions
            fraud_column: Column indicating actual fraud
            amount_column: Transaction amount column
        """
        self.transactions = transactions.copy()
        self.predictions = predictions
        self.fraud_column = fraud_column
        self.amount_column = amount_column
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        self.cm = confusion_matrix(transactions[fraud_column], predictions)
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()
        
    def create_business_dashboard(self, cost_matrix: Optional[Dict[str, float]] = None) -> go.Figure:
        """Create comprehensive business metrics dashboard.
        
        Args:
            cost_matrix: Cost/benefit values for confusion matrix cells
            
        Returns:
            Plotly dashboard figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dashboard creation")
        
        # Default cost matrix
        if cost_matrix is None:
            cost_matrix = {
                'tp_benefit': 100,   # Benefit of catching fraud
                'tn_benefit': 1,     # Benefit of correct approval
                'fp_cost': -10,      # Cost of false alarm
                'fn_cost': -500      # Cost of missing fraud
            }
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Confusion Matrix', 'Business Value Impact', 'Cost-Benefit Analysis',
                'Fraud Detection Rates', 'Transaction Volume Analysis', 'Risk Distribution',
                'Monthly Fraud Trends', 'Amount Distribution', 'Model Performance KPIs'
            ],
            specs=[
                [{"type": "table"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Confusion Matrix
        cm_df = pd.DataFrame(
            self.cm,
            index=['Actual Legit', 'Actual Fraud'],
            columns=['Pred Legit', 'Pred Fraud']
        )
        
        fig.add_trace(
            go.Table(
                header=dict(values=['', 'Predicted Legit', 'Predicted Fraud']),
                cells=dict(values=[
                    ['Actual Legit', 'Actual Fraud'],
                    [self.tn, self.fn],
                    [self.fp, self.tp]
                ])
            ),
            row=1, col=1
        )
        
        # 2. Business Value Impact
        business_impact = {
            'True Positives': self.tp * cost_matrix['tp_benefit'],
            'True Negatives': self.tn * cost_matrix['tn_benefit'],
            'False Positives': self.fp * cost_matrix['fp_cost'],
            'False Negatives': self.fn * cost_matrix['fn_cost']
        }
        
        fig.add_trace(
            go.Bar(
                x=list(business_impact.keys()),
                y=list(business_impact.values()),
                marker_color=['green', 'lightgreen', 'orange', 'red']
            ),
            row=1, col=2
        )
        
        # 3. Cost-Benefit Analysis Pie Chart
        total_benefit = business_impact['True Positives'] + business_impact['True Negatives']
        total_cost = abs(business_impact['False Positives']) + abs(business_impact['False Negatives'])
        
        fig.add_trace(
            go.Pie(
                labels=['Benefits', 'Costs'],
                values=[total_benefit, total_cost],
                marker_colors=['green', 'red']
            ),
            row=1, col=3
        )
        
        # 4. Detection Rates
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        fig.add_trace(
            go.Bar(
                x=['Precision', 'Recall', 'F1-Score'],
                y=[precision, recall, f1_score],
                marker_color='blue'
            ),
            row=2, col=1
        )
        
        # 5. Transaction Volume Analysis
        fraud_amounts = self.transactions[self.transactions[self.fraud_column] == 1][self.amount_column]
        legit_amounts = self.transactions[self.transactions[self.fraud_column] == 0][self.amount_column]
        
        fig.add_trace(
            go.Histogram(
                x=legit_amounts,
                name='Legitimate',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=fraud_amounts,
                name='Fraud',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # 6. Risk Distribution Box Plot
        fig.add_trace(
            go.Box(
                y=legit_amounts,
                name='Legitimate',
                boxpoints='outliers'
            ),
            row=2, col=3
        )
        
        fig.add_trace(
            go.Box(
                y=fraud_amounts,
                name='Fraud',
                boxpoints='outliers'
            ),
            row=2, col=3
        )
        
        # 7. Monthly Trends (if timestamp available)
        if 'timestamp' in self.transactions.columns:
            self.transactions['month'] = pd.to_datetime(self.transactions['timestamp']).dt.to_period('M')
            monthly_fraud = self.transactions.groupby('month')[self.fraud_column].sum()
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_fraud.index.astype(str),
                    y=monthly_fraud.values,
                    mode='lines+markers',
                    name='Monthly Fraud Count'
                ),
                row=3, col=1
            )
        
        # 8. Amount Distribution by Prediction
        predicted_fraud_amounts = self.transactions[self.predictions == 1][self.amount_column]
        predicted_legit_amounts = self.transactions[self.predictions == 0][self.amount_column]
        
        fig.add_trace(
            go.Histogram(
                x=predicted_legit_amounts,
                name='Predicted Legit',
                opacity=0.7,
                nbinsx=30
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=predicted_fraud_amounts,
                name='Predicted Fraud',
                opacity=0.7,
                nbinsx=30
            ),
            row=3, col=2
        )
        
        # 9. Key Performance Indicators
        total_value = sum(business_impact.values())
        fraud_detection_rate = recall
        false_alarm_rate = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=total_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Business Value ($)"},
                gauge={'axis': {'range': [None, max(0, total_value * 1.5)]},
                      'bar': {'color': "darkblue"},
                      'steps': [
                          {'range': [0, total_value * 0.5], 'color': "lightgray"},
                          {'range': [total_value * 0.5, total_value], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': total_value * 0.8}}
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Fraud Detection Business Metrics Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_roi_analysis(self, implementation_cost: float = 50000,
                           maintenance_cost_monthly: float = 5000) -> go.Figure:
        """Create ROI analysis visualization.
        
        Args:
            implementation_cost: One-time implementation cost
            maintenance_cost_monthly: Monthly maintenance cost
            
        Returns:
            ROI analysis figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for ROI analysis")
        
        # Calculate monthly benefits
        fraud_prevented_value = self.tp * 500  # Assume $500 average fraud amount
        false_alarm_cost = self.fp * 10        # Assume $10 cost per false alarm
        monthly_benefit = fraud_prevented_value - false_alarm_cost
        
        # Calculate ROI over time
        months = range(1, 25)  # 2 years
        cumulative_benefits = []
        cumulative_costs = []
        roi_values = []
        
        for month in months:
            cum_benefit = monthly_benefit * month
            cum_cost = implementation_cost + (maintenance_cost_monthly * month)
            roi = ((cum_benefit - cum_cost) / cum_cost) * 100 if cum_cost > 0 else 0
            
            cumulative_benefits.append(cum_benefit)
            cumulative_costs.append(cum_cost)
            roi_values.append(roi)
        
        # Create ROI visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cumulative Benefits vs Costs', 'ROI Over Time (%)',
                'Break-even Analysis', 'Monthly Cash Flow'
            ]
        )
        
        # Benefits vs Costs
        fig.add_trace(
            go.Scatter(x=months, y=cumulative_benefits, name='Cumulative Benefits', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=months, y=cumulative_costs, name='Cumulative Costs', line=dict(color='red')),
            row=1, col=1
        )
        
        # ROI over time
        fig.add_trace(
            go.Scatter(x=months, y=roi_values, name='ROI %', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # Break-even point
        net_values = [b - c for b, c in zip(cumulative_benefits, cumulative_costs)]
        fig.add_trace(
            go.Scatter(x=months, y=net_values, name='Net Value', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Monthly cash flow
        monthly_cash_flow = [monthly_benefit - maintenance_cost_monthly] * len(months)
        monthly_cash_flow[0] -= implementation_cost  # First month includes implementation
        
        fig.add_trace(
            go.Bar(x=months, y=monthly_cash_flow, name='Monthly Cash Flow'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Fraud Detection System ROI Analysis",
            showlegend=True
        )
        
        return fig


def create_comprehensive_fraud_visualization_suite(
    transactions: pd.DataFrame,
    model_predictions: np.ndarray,
    feature_importance: Dict[str, float],
    X: pd.DataFrame,
    fraud_column: str = 'is_fraud'
) -> Dict[str, go.Figure]:
    """Create a comprehensive suite of fraud detection visualizations.
    
    Args:
        transactions: Transaction dataset
        model_predictions: Model predictions
        feature_importance: Feature importance scores
        X: Feature matrix
        fraud_column: Column indicating fraud
        
    Returns:
        Dictionary of visualization names to Plotly figures
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for comprehensive visualization suite")
    
    visualizations = {}
    
    # 3D Feature Space
    if len(X.columns) >= 3:
        viz_3d = Interactive3DFeatureSpace(X, transactions[fraud_column].values, X.columns.tolist())
        visualizations['3d_feature_space'] = viz_3d.create_3d_scatter(X.columns[:3].tolist())
    
    # Feature Importance Hierarchy
    hierarchy_viz = HierarchicalFeatureImportanceVisualizer(feature_importance)
    visualizations['feature_treemap'] = hierarchy_viz.create_treemap()
    visualizations['feature_sunburst'] = hierarchy_viz.create_sunburst_chart()
    
    # Business Dashboard
    business_viz = BusinessMetricsDashboard(transactions, model_predictions, fraud_column)
    visualizations['business_dashboard'] = business_viz.create_business_dashboard()
    visualizations['roi_analysis'] = business_viz.create_roi_analysis()
    
    # Network Analysis (if account columns exist)
    if 'from_account' in transactions.columns and 'to_account' in transactions.columns:
        network_viz = FraudNetworkGraphVisualizer(transactions)
        visualizations['fraud_network'] = network_viz.create_fraud_network_graph()
        visualizations['transaction_flow'] = network_viz.create_transaction_flow_sankey()
    
    return visualizations