"""
Graph construction from segmented images
Converts segmented cellular structures into graph representations
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from scipy.spatial import distance_matrix


class GraphConstructor:
    """Convert segmented images to graph representations"""
    
    def __init__(self):
        self.graph = None
        self.node_features = []
        
    def create_graph_from_regions(self, features: List[dict], 
                                  distance_threshold: float = 50.0) -> nx.Graph:
        """
        Create a graph from segmented regions
        
        Args:
            features: List of feature dictionaries from segmented regions
            distance_threshold: Maximum distance for connecting nodes
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes
        for feature in features:
            node_id = feature['label']
            G.add_node(node_id, 
                      pos=feature['centroid'],
                      area=feature['area'],
                      mean_intensity=feature['mean_intensity'],
                      eccentricity=feature['eccentricity'],
                      solidity=feature['solidity'])
        
        # Add edges based on distance
        if len(features) > 1:
            # Get centroids
            centroids = np.array([f['centroid'] for f in features])
            
            # Calculate distance matrix
            dist_matrix = distance_matrix(centroids, centroids)
            
            # Create edges for nearby nodes
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    if dist_matrix[i, j] < distance_threshold:
                        weight = 1.0 / (dist_matrix[i, j] + 1e-6)
                        G.add_edge(features[i]['label'], 
                                 features[j]['label'],
                                 weight=weight,
                                 distance=dist_matrix[i, j])
        
        self.graph = G
        return G
    
    def get_node_feature_matrix(self, G: nx.Graph) -> np.ndarray:
        """
        Extract node features as a matrix
        
        Args:
            G: NetworkX graph
            
        Returns:
            Feature matrix (nodes x features)
        """
        feature_names = ['area', 'mean_intensity', 'eccentricity', 'solidity']
        
        nodes = sorted(G.nodes())
        feature_matrix = []
        
        for node in nodes:
            node_data = G.nodes[node]
            features = [node_data.get(fn, 0.0) for fn in feature_names]
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def get_adjacency_matrix(self, G: nx.Graph) -> np.ndarray:
        """
        Get adjacency matrix from graph
        
        Args:
            G: NetworkX graph
            
        Returns:
            Adjacency matrix
        """
        return nx.adjacency_matrix(G).todense()
    
    def add_node_labels(self, G: nx.Graph, labels: Dict[int, str]) -> nx.Graph:
        """
        Add classification labels to nodes
        
        Args:
            G: NetworkX graph
            labels: Dictionary mapping node IDs to labels
            
        Returns:
            Graph with labels
        """
        for node_id, label in labels.items():
            if node_id in G.nodes():
                G.nodes[node_id]['class_label'] = label
        
        return G
    
    def get_graph_statistics(self, G: nx.Graph) -> dict:
        """
        Calculate graph statistics
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G)
        }
        
        if G.number_of_nodes() > 0:
            if nx.is_connected(G):
                stats['diameter'] = nx.diameter(G)
                stats['average_shortest_path'] = nx.average_shortest_path_length(G)
            
            stats['average_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
            stats['clustering_coefficient'] = nx.average_clustering(G)
        
        return stats
    
    def save_graph(self, G: nx.Graph, filepath: str, format: str = 'gml'):
        """
        Save graph to file
        
        Args:
            G: NetworkX graph
            filepath: Output file path
            format: File format ('gml', 'graphml', 'gexf')
        """
        if format == 'gml':
            nx.write_gml(G, filepath)
        elif format == 'graphml':
            nx.write_graphml(G, filepath)
        elif format == 'gexf':
            nx.write_gexf(G, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load_graph(self, filepath: str, format: str = 'gml') -> nx.Graph:
        """
        Load graph from file
        
        Args:
            filepath: Input file path
            format: File format ('gml', 'graphml', 'gexf')
            
        Returns:
            NetworkX graph
        """
        if format == 'gml':
            G = nx.read_gml(filepath)
        elif format == 'graphml':
            G = nx.read_graphml(filepath)
        elif format == 'gexf':
            G = nx.read_gexf(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self.graph = G
        return G


if __name__ == "__main__":
    # Test with synthetic features
    print("Creating test features...")
    
    # Simulate 5 segmented regions
    test_features = []
    np.random.seed(42)
    
    for i in range(5):
        feature = {
            'label': i + 1,
            'area': np.random.randint(100, 500),
            'centroid': (np.random.uniform(0, 100), np.random.uniform(0, 100)),
            'mean_intensity': np.random.uniform(0.3, 0.9),
            'max_intensity': np.random.uniform(0.8, 1.0),
            'min_intensity': np.random.uniform(0.1, 0.3),
            'eccentricity': np.random.uniform(0, 1),
            'solidity': np.random.uniform(0.7, 1.0),
            'bbox': (0, 0, 50, 50)
        }
        test_features.append(feature)
    
    print("\nConstructing graph...")
    constructor = GraphConstructor()
    G = constructor.create_graph_from_regions(test_features, distance_threshold=50.0)
    
    print("\nGraph statistics:")
    stats = constructor.get_graph_statistics(G)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nNode feature matrix shape:", constructor.get_node_feature_matrix(G).shape)
    print("Adjacency matrix shape:", constructor.get_adjacency_matrix(G).shape)
