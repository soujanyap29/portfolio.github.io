"""
Graph Construction Module
Converts segmented images and features into biological graphs for GNN analysis.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import json


class GraphConstructor:
    """
    Build biological graphs from segmented images and extracted features.
    """
    
    def __init__(self, distance_threshold: float = 50.0, k_neighbors: int = 5):
        """
        Initialize graph constructor.
        
        Args:
            distance_threshold: Maximum distance for edge creation
            k_neighbors: Number of nearest neighbors to connect
        """
        self.distance_threshold = distance_threshold
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()
    
    def build_spatial_graph(self, features: pd.DataFrame, 
                           method: str = 'knn') -> nx.Graph:
        """
        Build graph based on spatial relationships.
        
        Args:
            features: DataFrame with extracted features including centroids
            method: Graph construction method ('knn', 'distance', 'delaunay')
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes with features
        for idx, row in features.iterrows():
            node_id = int(row['label'])
            node_attrs = row.to_dict()
            G.add_node(node_id, **node_attrs)
        
        # Extract centroid coordinates
        if 'centroid_x' in features.columns and 'centroid_y' in features.columns:
            coords = features[['centroid_x', 'centroid_y']].values
        else:
            print("Warning: Centroid coordinates not found in features")
            return G
        
        # Build edges based on method
        if method == 'knn':
            self._add_knn_edges(G, coords, features)
        elif method == 'distance':
            self._add_distance_edges(G, coords, features)
        elif method == 'delaunay':
            self._add_delaunay_edges(G, coords, features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return G
    
    def _add_knn_edges(self, G: nx.Graph, coords: np.ndarray, features: pd.DataFrame):
        """Add edges to k-nearest neighbors."""
        # Compute distance matrix
        dist_matrix = distance_matrix(coords, coords)
        
        for i in range(len(coords)):
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i]
            nearest_indices = np.argsort(distances)[1:self.k_neighbors + 1]
            
            node_i = int(features.iloc[i]['label'])
            for j in nearest_indices:
                node_j = int(features.iloc[j]['label'])
                dist = distances[j]
                
                if dist <= self.distance_threshold:
                    G.add_edge(node_i, node_j, 
                             distance=float(dist),
                             weight=1.0 / (dist + 1e-6))
    
    def _add_distance_edges(self, G: nx.Graph, coords: np.ndarray, features: pd.DataFrame):
        """Add edges based on distance threshold."""
        dist_matrix = distance_matrix(coords, coords)
        
        for i in range(len(coords)):
            node_i = int(features.iloc[i]['label'])
            for j in range(i + 1, len(coords)):
                node_j = int(features.iloc[j]['label'])
                dist = dist_matrix[i, j]
                
                if dist <= self.distance_threshold:
                    G.add_edge(node_i, node_j,
                             distance=float(dist),
                             weight=1.0 / (dist + 1e-6))
    
    def _add_delaunay_edges(self, G: nx.Graph, coords: np.ndarray, features: pd.DataFrame):
        """Add edges based on Delaunay triangulation."""
        try:
            from scipy.spatial import Delaunay
            
            tri = Delaunay(coords)
            
            # Add edges from triangulation
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        node_i = int(features.iloc[simplex[i]]['label'])
                        node_j = int(features.iloc[simplex[j]]['label'])
                        
                        dist = np.linalg.norm(coords[simplex[i]] - coords[simplex[j]])
                        
                        if dist <= self.distance_threshold:
                            G.add_edge(node_i, node_j,
                                     distance=float(dist),
                                     weight=1.0 / (dist + 1e-6))
        except Exception as e:
            print(f"Delaunay triangulation failed: {e}. Using KNN instead.")
            self._add_knn_edges(G, coords, features)
    
    def add_morphological_edges(self, G: nx.Graph, features: pd.DataFrame,
                               similarity_threshold: float = 0.8):
        """
        Add edges based on morphological similarity.
        
        Args:
            G: Existing graph
            features: DataFrame with features
            similarity_threshold: Minimum similarity for edge creation
        """
        # Select morphological features
        morph_features = ['area', 'perimeter', 'eccentricity', 'solidity', 
                         'circularity', 'major_axis_length', 'minor_axis_length']
        
        available_features = [f for f in morph_features if f in features.columns]
        
        if not available_features:
            print("Warning: No morphological features available")
            return
        
        # Normalize features
        X = features[available_features].values
        X_norm = self.scaler.fit_transform(X)
        
        # Compute similarity matrix
        similarity_matrix = 1 - distance_matrix(X_norm, X_norm) / np.max(distance_matrix(X_norm, X_norm))
        
        for i in range(len(features)):
            node_i = int(features.iloc[i]['label'])
            for j in range(i + 1, len(features)):
                node_j = int(features.iloc[j]['label'])
                similarity = similarity_matrix[i, j]
                
                if similarity >= similarity_threshold:
                    # Add or update edge
                    if G.has_edge(node_i, node_j):
                        G[node_i][node_j]['morphology_similarity'] = float(similarity)
                    else:
                        G.add_edge(node_i, node_j,
                                 morphology_similarity=float(similarity),
                                 weight=float(similarity))
    
    def convert_to_pyg(self, G: nx.Graph, feature_cols: Optional[List[str]] = None):
        """
        Convert NetworkX graph to PyTorch Geometric format.
        
        Args:
            G: NetworkX graph
            feature_cols: List of feature column names to include
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
            
            # Map node IDs to indices
            node_list = sorted(G.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            
            # Extract node features
            if feature_cols is None:
                # Use all numeric features
                feature_cols = []
                sample_node = node_list[0]
                for key, value in G.nodes[sample_node].items():
                    if isinstance(value, (int, float)) and key != 'label':
                        feature_cols.append(key)
            
            # Build feature matrix
            x_list = []
            for node in node_list:
                node_features = []
                for col in feature_cols:
                    val = G.nodes[node].get(col, 0.0)
                    node_features.append(float(val))
                x_list.append(node_features)
            
            x = torch.tensor(x_list, dtype=torch.float)
            
            # Build edge index
            edge_index = []
            edge_attr = []
            for u, v, data in G.edges(data=True):
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                edge_index.append([u_idx, v_idx])
                edge_index.append([v_idx, u_idx])  # Undirected
                
                # Edge attributes
                weight = data.get('weight', 1.0)
                edge_attr.append([weight])
                edge_attr.append([weight])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.num_nodes = len(node_list)
            
            return data
            
        except ImportError:
            print("PyTorch Geometric not available")
            return None
    
    def convert_to_dgl(self, G: nx.Graph, feature_cols: Optional[List[str]] = None):
        """
        Convert NetworkX graph to DGL format.
        
        Args:
            G: NetworkX graph
            feature_cols: List of feature column names to include
            
        Returns:
            DGL graph object
        """
        try:
            import torch
            import dgl
            
            # Create DGL graph from NetworkX
            dgl_g = dgl.from_networkx(G, node_attrs=feature_cols, edge_attrs=['weight'])
            
            return dgl_g
            
        except ImportError:
            print("DGL not available")
            return None
    
    def save_graph(self, G: nx.Graph, output_path: str, format: str = 'pickle'):
        """
        Save graph to file.
        
        Args:
            G: NetworkX graph
            output_path: Output file path
            format: Save format ('pickle', 'gml', 'graphml', 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(G, f)
        elif format == 'gml':
            nx.write_gml(G, output_path)
        elif format == 'graphml':
            nx.write_graphml(G, output_path)
        elif format == 'json':
            data = nx.node_link_data(G)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Saved graph to {output_path}")
    
    def load_graph(self, input_path: str, format: str = 'pickle') -> nx.Graph:
        """
        Load graph from file.
        
        Args:
            input_path: Input file path
            format: File format
            
        Returns:
            NetworkX graph
        """
        if format == 'pickle':
            with open(input_path, 'rb') as f:
                G = pickle.load(f)
        elif format == 'gml':
            G = nx.read_gml(input_path)
        elif format == 'graphml':
            G = nx.read_graphml(input_path)
        elif format == 'json':
            with open(input_path, 'r') as f:
                data = json.load(f)
            G = nx.node_link_graph(data)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return G
    
    def get_graph_statistics(self, G: nx.Graph) -> Dict:
        """
        Compute graph statistics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
        }
        
        if stats['is_connected']:
            stats['diameter'] = nx.diameter(G)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        
        stats['avg_degree'] = sum(dict(G.degree()).values()) / float(G.number_of_nodes())
        stats['avg_clustering'] = nx.average_clustering(G)
        
        return stats


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        features_path = sys.argv[1]
        features = pd.read_csv(features_path)
        
        print(f"Building graph from {len(features)} regions...")
        constructor = GraphConstructor(distance_threshold=100.0, k_neighbors=5)
        
        # Build spatial graph
        G = constructor.build_spatial_graph(features, method='knn')
        
        # Add morphological edges
        constructor.add_morphological_edges(G, features)
        
        # Get statistics
        stats = constructor.get_graph_statistics(G)
        print("\n=== Graph Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Save graph
        output_path = Path(features_path).parent / f"{Path(features_path).stem}_graph.pkl"
        constructor.save_graph(G, str(output_path))
        
    else:
        print("Usage: python graph_construction.py <path_to_features.csv>")
