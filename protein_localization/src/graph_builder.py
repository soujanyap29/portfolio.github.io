"""
Graph construction module for building biological graphs from segmented images.
Creates node-edge representations compatible with GNN frameworks.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from scipy.spatial import distance_matrix
import torch
from torch_geometric.data import Data
from pathlib import Path
import pickle


class BiologicalGraphBuilder:
    """
    Constructs biological graphs from segmented images and extracted features.
    Nodes represent puncta/compartments, edges represent spatial relationships.
    """
    
    def __init__(self, distance_threshold: float = 50.0, k_neighbors: int = 5):
        """
        Initialize the graph builder.
        
        Args:
            distance_threshold: Maximum distance for edge creation
            k_neighbors: Number of nearest neighbors to connect
        """
        self.distance_threshold = distance_threshold
        self.k_neighbors = k_neighbors
    
    def build_graph(self, features: List[Dict], method: str = 'knn') -> nx.Graph:
        """
        Build a NetworkX graph from extracted features.
        
        Args:
            features: List of feature dictionaries
            method: 'knn' for k-nearest neighbors or 'threshold' for distance threshold
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes with features
        for feat in features:
            node_id = feat['label']
            G.add_node(node_id, **feat)
        
        # Extract coordinates for edge computation
        coords = np.array([[feat['centroid_y'], feat['centroid_x']] for feat in features])
        
        if len(coords) < 2:
            print(f"Warning: Only {len(coords)} nodes, no edges created")
            return G
        
        # Compute distance matrix
        dist_matrix = distance_matrix(coords, coords)
        
        # Add edges based on method
        if method == 'knn':
            self._add_knn_edges(G, dist_matrix, features)
        elif method == 'threshold':
            self._add_threshold_edges(G, dist_matrix, features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _add_knn_edges(self, G: nx.Graph, dist_matrix: np.ndarray, features: List[Dict]):
        """Add edges to k-nearest neighbors."""
        for i, feat_i in enumerate(features):
            node_i = feat_i['label']
            
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i, :]
            nearest_indices = np.argsort(distances)[1:self.k_neighbors+1]
            
            for j in nearest_indices:
                if j < len(features):
                    node_j = features[j]['label']
                    distance = distances[j]
                    
                    if distance < self.distance_threshold:
                        G.add_edge(node_i, node_j, weight=distance, distance=distance)
    
    def _add_threshold_edges(self, G: nx.Graph, dist_matrix: np.ndarray, features: List[Dict]):
        """Add edges based on distance threshold."""
        for i, feat_i in enumerate(features):
            for j, feat_j in enumerate(features):
                if i < j:  # Avoid duplicate edges
                    distance = dist_matrix[i, j]
                    if distance < self.distance_threshold:
                        node_i = feat_i['label']
                        node_j = feat_j['label']
                        G.add_edge(node_i, node_j, weight=distance, distance=distance)
    
    def networkx_to_pyg(self, G: nx.Graph, feature_keys: Optional[List[str]] = None) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            G: NetworkX graph
            feature_keys: List of feature keys to include as node features
            
        Returns:
            PyTorch Geometric Data object
        """
        if feature_keys is None:
            feature_keys = [
                'centroid_y', 'centroid_x', 'distance_from_center',
                'area', 'perimeter', 'eccentricity', 'solidity',
                'mean_intensity', 'max_intensity', 'intensity_std'
            ]
        
        # Create node feature matrix
        node_features = []
        node_labels = []
        node_mapping = {}
        
        for idx, (node, data) in enumerate(G.nodes(data=True)):
            node_mapping[node] = idx
            
            # Extract features
            features = [data.get(key, 0.0) for key in feature_keys]
            node_features.append(features)
            
            # Store original label
            node_labels.append(data.get('label', node))
        
        # Create edge index
        edge_index = []
        edge_attr = []
        
        for u, v, data in G.edges(data=True):
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                edge_index.append([node_mapping[v], node_mapping[u]])  # Make undirected
                
                edge_weight = data.get('weight', 1.0)
                edge_attr.append([edge_weight])
                edge_attr.append([edge_weight])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_features)
        )
        
        # Store node labels as additional attribute
        data.node_labels = torch.tensor(node_labels, dtype=torch.long)
        
        return data
    
    def process_results(self, processed_results: List[Dict], 
                       output_dir: str) -> List[Dict]:
        """
        Process multiple segmentation results to create graphs.
        
        Args:
            processed_results: List of preprocessing results
            output_dir: Directory to save graphs
            
        Returns:
            List of graph data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        graph_results = []
        
        for result in processed_results:
            file_name = result['file_name']
            features = result['features']
            
            if len(features) == 0:
                print(f"Skipping {file_name}: no features")
                continue
            
            print(f"\nBuilding graph for {file_name}")
            
            # Build NetworkX graph
            G = self.build_graph(features, method='knn')
            
            # Convert to PyTorch Geometric
            pyg_data = self.networkx_to_pyg(G)
            
            # Save graphs
            base_name = Path(file_name).stem
            
            # Save NetworkX graph
            nx_path = output_path / f"{base_name}_graph.gpickle"
            nx.write_gpickle(G, nx_path)
            
            # Save PyG data
            pyg_path = output_path / f"{base_name}_pyg.pt"
            torch.save(pyg_data, pyg_path)
            
            graph_result = {
                'file_name': file_name,
                'networkx_graph': G,
                'pyg_data': pyg_data,
                'nx_path': str(nx_path),
                'pyg_path': str(pyg_path),
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
            }
            
            graph_results.append(graph_result)
        
        print(f"\n✓ Created {len(graph_results)} graphs")
        return graph_results


class GraphFeatureExtractor:
    """Extract graph-level features for traditional ML."""
    
    @staticmethod
    def extract_graph_features(G: nx.Graph) -> Dict:
        """
        Extract global graph features.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of graph features
        """
        features = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
        }
        
        # Add clustering coefficient if graph has edges
        if G.number_of_edges() > 0:
            features['avg_clustering'] = nx.average_clustering(G)
            
            # Connected components
            features['n_connected_components'] = nx.number_connected_components(G)
            
            # Average shortest path length (for largest component)
            if nx.is_connected(G):
                features['avg_shortest_path'] = nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    features['avg_shortest_path'] = nx.average_shortest_path_length(subgraph)
                else:
                    features['avg_shortest_path'] = 0
        else:
            features['avg_clustering'] = 0
            features['n_connected_components'] = G.number_of_nodes()
            features['avg_shortest_path'] = 0
        
        return features


def build_graphs_pipeline(processed_results: List[Dict], 
                          output_dir: str,
                          distance_threshold: float = 50.0,
                          k_neighbors: int = 5) -> List[Dict]:
    """
    Main graph building pipeline entry point.
    
    Args:
        processed_results: List of preprocessing results
        output_dir: Directory to save graphs
        distance_threshold: Maximum distance for edge creation
        k_neighbors: Number of nearest neighbors
        
    Returns:
        List of graph results
    """
    builder = BiologicalGraphBuilder(
        distance_threshold=distance_threshold,
        k_neighbors=k_neighbors
    )
    graph_results = builder.process_results(processed_results, output_dir)
    return graph_results
