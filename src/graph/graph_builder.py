"""
Graph Construction Module
Convert segmented images to biological graphs for GNN processing
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
import pickle
from pathlib import Path


class GraphBuilder:
    """Build biological graphs from segmented images"""
    
    def __init__(self, distance_threshold: float = 50.0, k_neighbors: int = 5):
        """
        Args:
            distance_threshold: Maximum distance for edge creation
            k_neighbors: Number of nearest neighbors to connect
        """
        self.distance_threshold = distance_threshold
        self.k_neighbors = k_neighbors
        
    def build_graph_from_features(self, features: Dict) -> nx.Graph:
        """
        Build a NetworkX graph from extracted features
        
        Args:
            features: Dictionary containing region and spatial features
            
        Returns:
            NetworkX graph with node and edge attributes
        """
        G = nx.Graph()
        
        region_features = features['region_features']
        spatial_features = features['spatial_features']
        centroids = spatial_features['centroids']
        
        # Add nodes
        for i, region in enumerate(region_features):
            node_attrs = {
                'centroid': centroids[i] if i < len(centroids) else (0, 0),
                'area': region['area'],
                'perimeter': region['perimeter'],
                'eccentricity': region['eccentricity'],
                'mean_intensity': region['mean_intensity'],
                'aspect_ratio': region['aspect_ratio'],
                'compactness': region['compactness'],
                'label': i  # Node label preserved for visualization
            }
            G.add_node(i, **node_attrs)
        
        # Add edges based on spatial proximity
        if len(centroids) > 0:
            # Calculate pairwise distances
            distances = cdist(centroids, centroids)
            
            # Method 1: K-nearest neighbors
            for i in range(len(centroids)):
                # Get k nearest neighbors (excluding self)
                neighbor_indices = np.argsort(distances[i])[1:self.k_neighbors+1]
                
                for j in neighbor_indices:
                    if distances[i, j] < self.distance_threshold:
                        G.add_edge(i, j, 
                                 distance=distances[i, j],
                                 weight=1.0 / (distances[i, j] + 1e-6))
        
        return G
    
    def build_spatial_graph(self, centroids: np.ndarray, 
                           features: List[Dict]) -> nx.Graph:
        """Build graph based on spatial relationships"""
        G = nx.Graph()
        
        # Add nodes with features
        for i, (centroid, feat) in enumerate(zip(centroids, features)):
            G.add_node(i, pos=centroid, **feat, label=i)
        
        # Add edges
        if len(centroids) > 1:
            distances = cdist(centroids, centroids)
            
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    if distances[i, j] < self.distance_threshold:
                        G.add_edge(i, j, distance=distances[i, j])
        
        return G
    
    def to_pytorch_geometric(self, G: nx.Graph, 
                            feature_keys: Optional[List[str]] = None) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object
        
        Args:
            G: NetworkX graph
            feature_keys: List of node attribute keys to use as features
            
        Returns:
            PyTorch Geometric Data object
        """
        if feature_keys is None:
            feature_keys = ['area', 'perimeter', 'eccentricity', 
                          'mean_intensity', 'aspect_ratio', 'compactness']
        
        # Extract node features
        node_features = []
        node_labels = []
        
        for node in G.nodes():
            features = []
            for key in feature_keys:
                val = G.nodes[node].get(key, 0.0)
                features.append(float(val))
            node_features.append(features)
            node_labels.append(G.nodes[node].get('label', node))
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edges
        edge_index = []
        edge_attr = []
        
        for edge in G.edges():
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])  # Add reverse edge for undirected
            
            # Edge attributes
            edge_data = G.edges[edge]
            distance = edge_data.get('distance', 1.0)
            edge_attr.append([distance])
            edge_attr.append([distance])
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(node_features)
        data.node_labels = node_labels  # Preserve for visualization
        
        return data


class GraphDataset:
    """Dataset of graphs for protein localization"""
    
    def __init__(self, preprocessed_data_path: str):
        """
        Args:
            preprocessed_data_path: Path to preprocessed pickle file
        """
        self.data_path = Path(preprocessed_data_path)
        self.graphs = []
        self.labels = []
        self.graph_builder = GraphBuilder()
        
    def load_and_build_graphs(self):
        """Load preprocessed data and build graphs"""
        with open(self.data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        print(f"Building graphs from {len(preprocessed_data)} images...")
        
        for i, data in enumerate(preprocessed_data):
            try:
                # Build NetworkX graph
                G = self.graph_builder.build_graph_from_features(data)
                
                # Convert to PyTorch Geometric
                pyg_data = self.graph_builder.to_pytorch_geometric(G)
                
                self.graphs.append(pyg_data)
                
                # Placeholder label (would need actual labels from annotations)
                # For now, use dummy labels
                self.labels.append(0)
                
            except Exception as e:
                print(f"Error building graph for {data.get('filename', i)}: {e}")
                continue
        
        print(f"Successfully built {len(self.graphs)} graphs")
        
    def save_graphs(self, output_path: str):
        """Save graphs to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_dict = {
            'graphs': self.graphs,
            'labels': self.labels
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Saved {len(self.graphs)} graphs to {output_path}")
    
    def get_graph(self, idx: int) -> Tuple[Data, int]:
        """Get a single graph and its label"""
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def get_feature_statistics(self) -> Dict:
        """Calculate statistics of node features across all graphs"""
        all_features = []
        
        for graph in self.graphs:
            all_features.append(graph.x.numpy())
        
        if len(all_features) > 0:
            all_features = np.vstack(all_features)
            
            stats = {
                'mean': all_features.mean(axis=0),
                'std': all_features.std(axis=0),
                'min': all_features.min(axis=0),
                'max': all_features.max(axis=0)
            }
        else:
            stats = {}
        
        return stats


class CompartmentGraph:
    """Specialized graph for sub-cellular compartment analysis"""
    
    COMPARTMENT_TYPES = [
        'soma',
        'dendrite',
        'axon',
        'nucleus',
        'synapse',
        'mitochondria'
    ]
    
    def __init__(self):
        self.compartment_labels = {}
        
    def assign_compartment_labels(self, G: nx.Graph, 
                                  compartment_predictions: np.ndarray) -> nx.Graph:
        """
        Assign compartment type labels to graph nodes
        
        Args:
            G: Input graph
            compartment_predictions: Array of compartment type indices
            
        Returns:
            Graph with compartment labels
        """
        for i, node in enumerate(G.nodes()):
            if i < len(compartment_predictions):
                compartment_idx = compartment_predictions[i]
                compartment_type = self.COMPARTMENT_TYPES[compartment_idx % len(self.COMPARTMENT_TYPES)]
                G.nodes[node]['compartment'] = compartment_type
        
        return G
    
    def create_hierarchical_graph(self, G: nx.Graph) -> nx.DiGraph:
        """Create hierarchical representation (soma -> dendrites -> synapses)"""
        H = nx.DiGraph()
        
        # Group nodes by compartment
        compartments = {}
        for node, data in G.nodes(data=True):
            comp_type = data.get('compartment', 'unknown')
            if comp_type not in compartments:
                compartments[comp_type] = []
            compartments[comp_type].append(node)
        
        # Create hierarchy: soma at top, then branches
        hierarchy = ['soma', 'nucleus', 'dendrite', 'axon', 'synapse', 'mitochondria']
        
        for i in range(len(hierarchy) - 1):
            parent_type = hierarchy[i]
            child_type = hierarchy[i + 1]
            
            if parent_type in compartments and child_type in compartments:
                for parent in compartments[parent_type]:
                    for child in compartments[child_type]:
                        # Add directed edge if spatial proximity
                        if G.has_edge(parent, child):
                            H.add_edge(parent, child)
        
        return H


if __name__ == "__main__":
    # Example usage
    dataset = GraphDataset("/mnt/d/5TH_SEM/CELLULAR/output/preprocessed_data.pkl")
    dataset.load_and_build_graphs()
    dataset.save_graphs("/mnt/d/5TH_SEM/CELLULAR/output/graphs.pkl")
    
    # Print statistics
    stats = dataset.get_feature_statistics()
    print("\nFeature Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
