"""
Graph construction module for protein sub-cellular localization
Converts segmented images to graphs for GNN processing
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
import pickle
import json


class GraphConstructor:
    """Construct graphs from segmented images and features"""
    
    def __init__(self, proximity_threshold: float = 50.0, max_edges_per_node: int = 10):
        """
        Initialize graph constructor
        
        Args:
            proximity_threshold: Maximum distance for edge creation
            max_edges_per_node: Maximum number of edges per node
        """
        self.proximity_threshold = proximity_threshold
        self.max_edges_per_node = max_edges_per_node
    
    def construct_graph(self, features: pd.DataFrame, masks: np.ndarray) -> nx.Graph:
        """
        Construct a graph from features and masks
        
        Args:
            features: DataFrame with extracted features
            masks: Segmentation masks
        
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes with features
        for idx, row in features.iterrows():
            node_id = int(row['label']) if 'label' in row else idx
            
            # Store all features as node attributes
            node_attrs = row.to_dict()
            G.add_node(node_id, **node_attrs)
        
        # Add edges based on spatial proximity
        if 'centroid_y' in features.columns and 'centroid_x' in features.columns:
            self._add_spatial_edges(G, features)
        
        # Add edges based on adjacency in masks
        self._add_adjacency_edges(G, masks)
        
        return G
    
    def _add_spatial_edges(self, G: nx.Graph, features: pd.DataFrame):
        """Add edges based on spatial proximity"""
        labels = features['label'].values if 'label' in features.columns else features.index.values
        centroids = features[['centroid_y', 'centroid_x']].values
        
        # Calculate pairwise distances
        distances = cdist(centroids, centroids, metric='euclidean')
        
        # Create edges for nearby nodes
        for i, label_i in enumerate(labels):
            # Get distances to all other nodes
            node_distances = [(labels[j], distances[i, j]) for j in range(len(labels)) if i != j]
            
            # Sort by distance
            node_distances.sort(key=lambda x: x[1])
            
            # Add edges to k nearest neighbors within threshold
            added_edges = 0
            for neighbor_label, dist in node_distances:
                if dist <= self.proximity_threshold and added_edges < self.max_edges_per_node:
                    if G.has_node(label_i) and G.has_node(neighbor_label):
                        G.add_edge(label_i, neighbor_label, 
                                 weight=1.0 / (dist + 1),  # Inverse distance weighting
                                 distance=dist,
                                 edge_type='spatial')
                        added_edges += 1
    
    def _add_adjacency_edges(self, G: nx.Graph, masks: np.ndarray):
        """Add edges for adjacent regions in the segmentation mask"""
        from skimage import measure
        
        # Find boundaries
        boundaries = self._find_boundaries(masks)
        
        # For each boundary pixel, check if it touches different regions
        for y, x in boundaries:
            # Check neighborhood
            neighbors = self._get_neighboring_labels(masks, y, x)
            
            # Add edges between neighboring regions
            for i, label1 in enumerate(neighbors):
                for label2 in neighbors[i+1:]:
                    if label1 != label2 and label1 != 0 and label2 != 0:
                        if G.has_node(label1) and G.has_node(label2):
                            if not G.has_edge(label1, label2):
                                G.add_edge(label1, label2, edge_type='adjacent')
    
    @staticmethod
    def _find_boundaries(masks: np.ndarray) -> List[Tuple[int, int]]:
        """Find boundary pixels in the mask"""
        from scipy.ndimage import sobel
        
        # Calculate gradient
        grad_y = sobel(masks, axis=0)
        grad_x = sobel(masks, axis=1)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # Threshold to find boundaries
        boundary_mask = gradient_magnitude > 0
        
        # Get coordinates
        y_coords, x_coords = np.where(boundary_mask)
        return list(zip(y_coords, x_coords))
    
    @staticmethod
    def _get_neighboring_labels(masks: np.ndarray, y: int, x: int, radius: int = 2) -> List[int]:
        """Get labels of neighboring regions around a pixel"""
        h, w = masks.shape
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        
        neighborhood = masks[y_min:y_max, x_min:x_max]
        unique_labels = np.unique(neighborhood)
        
        return [int(label) for label in unique_labels if label != 0]
    
    def construct_bipartite_graph(self, features: pd.DataFrame, 
                                 puncta_labels: List[int], 
                                 compartment_labels: List[int]) -> nx.Graph:
        """
        Construct a bipartite graph with puncta and compartments
        
        Args:
            features: DataFrame with features
            puncta_labels: Labels for puncta nodes
            compartment_labels: Labels for compartment nodes
        
        Returns:
            Bipartite NetworkX graph
        """
        G = nx.Graph()
        
        # Add puncta nodes
        for label in puncta_labels:
            if label in features['label'].values:
                row = features[features['label'] == label].iloc[0]
                G.add_node(label, bipartite='puncta', **row.to_dict())
        
        # Add compartment nodes
        for label in compartment_labels:
            if label in features['label'].values:
                row = features[features['label'] == label].iloc[0]
                G.add_node(label, bipartite='compartment', **row.to_dict())
        
        # Add edges between puncta and their containing compartments
        for puncta_label in puncta_labels:
            if G.has_node(puncta_label):
                puncta_centroid = features[features['label'] == puncta_label][['centroid_y', 'centroid_x']].values[0]
                
                # Find closest compartment
                min_dist = float('inf')
                closest_compartment = None
                
                for comp_label in compartment_labels:
                    if G.has_node(comp_label):
                        comp_centroid = features[features['label'] == comp_label][['centroid_y', 'centroid_x']].values[0]
                        dist = np.linalg.norm(puncta_centroid - comp_centroid)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_compartment = comp_label
                
                if closest_compartment is not None:
                    G.add_edge(puncta_label, closest_compartment, 
                             weight=1.0 / (min_dist + 1),
                             distance=min_dist)
        
        return G


class PyTorchGeometricConverter:
    """Convert NetworkX graphs to PyTorch Geometric format"""
    
    @staticmethod
    def to_pytorch_geometric(G: nx.Graph, feature_columns: List[str] = None) -> Dict:
        """
        Convert NetworkX graph to PyTorch Geometric Data object format
        
        Args:
            G: NetworkX graph
            feature_columns: List of feature columns to use
        
        Returns:
            Dictionary with 'x' (node features), 'edge_index', and 'edge_attr'
        """
        import torch
        
        # Create node mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Extract node features
        if feature_columns is None:
            # Use all numeric attributes
            sample_node = node_list[0]
            feature_columns = [k for k, v in G.nodes[sample_node].items() 
                             if isinstance(v, (int, float)) and k != 'label']
        
        node_features = []
        for node in node_list:
            features = [G.nodes[node].get(col, 0.0) for col in feature_columns]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edges
        edge_list = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            edge_list.append([node_to_idx[v], node_to_idx[u]])  # Undirected
            
            weight = data.get('weight', 1.0)
            edge_weights.append(weight)
            edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1) if edge_weights else torch.zeros((0, 1), dtype=torch.float)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_mapping': node_to_idx,
            'feature_columns': feature_columns
        }
    
    @staticmethod
    def create_data_object(graph_dict: Dict, y: Optional[int] = None):
        """
        Create a PyTorch Geometric Data object
        
        Args:
            graph_dict: Dictionary from to_pytorch_geometric
            y: Optional label for the graph
        
        Returns:
            torch_geometric.data.Data object
        """
        try:
            from torch_geometric.data import Data
            import torch
            
            data = Data(
                x=graph_dict['x'],
                edge_index=graph_dict['edge_index'],
                edge_attr=graph_dict['edge_attr']
            )
            
            if y is not None:
                data.y = torch.tensor([y], dtype=torch.long)
            
            return data
        except ImportError:
            print("PyTorch Geometric not installed. Returning dictionary format.")
            return graph_dict


class DGLConverter:
    """Convert NetworkX graphs to DGL format"""
    
    @staticmethod
    def to_dgl(G: nx.Graph, feature_columns: List[str] = None):
        """
        Convert NetworkX graph to DGL graph
        
        Args:
            G: NetworkX graph
            feature_columns: List of feature columns to use
        
        Returns:
            DGL graph
        """
        try:
            import dgl
            import torch
            
            # Convert to DGL
            dgl_graph = dgl.from_networkx(G)
            
            # Add node features
            node_list = list(G.nodes())
            
            if feature_columns is None:
                sample_node = node_list[0]
                feature_columns = [k for k, v in G.nodes[sample_node].items() 
                                 if isinstance(v, (int, float)) and k != 'label']
            
            node_features = []
            for node in node_list:
                features = [G.nodes[node].get(col, 0.0) for col in feature_columns]
                node_features.append(features)
            
            dgl_graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float)
            
            # Add edge features if available
            edge_weights = []
            for u, v in G.edges():
                edge_weights.append(G[u][v].get('weight', 1.0))
            
            if edge_weights:
                dgl_graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float)
            
            return dgl_graph
            
        except ImportError:
            print("DGL not installed. Returning None.")
            return None


class GraphStorage:
    """Store and load graphs"""
    
    def __init__(self, output_dir: str = './graphs'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def save_graph(self, G: nx.Graph, filename: str):
        """Save graph to file"""
        import os
        filepath = os.path.join(self.output_dir, filename)
        
        # Save as pickle
        with open(filepath + '.gpickle', 'wb') as f:
            pickle.dump(G, f)
        
        # Save as GraphML for compatibility
        try:
            nx.write_graphml(G, filepath + '.graphml')
        except:
            pass
        
        # Save node and edge info as JSON
        graph_info = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'nodes': list(G.nodes()),
            'edges': [(u, v) for u, v in G.edges()]
        }
        
        with open(filepath + '_info.json', 'w') as f:
            json.dump(graph_info, f, indent=2)
        
        print(f"Graph saved to {filepath}")
    
    def load_graph(self, filename: str) -> nx.Graph:
        """Load graph from file"""
        import os
        filepath = os.path.join(self.output_dir, filename)
        
        # Try loading from pickle
        try:
            with open(filepath + '.gpickle', 'rb') as f:
                return pickle.load(f)
        except:
            pass
        
        # Try loading from GraphML
        try:
            return nx.read_graphml(filepath + '.graphml')
        except:
            pass
        
        raise FileNotFoundError(f"Could not load graph from {filepath}")


if __name__ == "__main__":
    # Test graph construction
    print("Testing graph construction module...")
    
    # Create dummy features
    features = pd.DataFrame({
        'label': [1, 2, 3, 4],
        'centroid_y': [10, 30, 50, 70],
        'centroid_x': [10, 30, 50, 70],
        'area': [100, 150, 120, 130],
        'mean_intensity': [50, 60, 55, 65]
    })
    
    # Create dummy masks
    masks = np.zeros((100, 100), dtype=int)
    masks[5:15, 5:15] = 1
    masks[25:35, 25:35] = 2
    masks[45:55, 45:55] = 3
    masks[65:75, 65:75] = 4
    
    # Construct graph
    constructor = GraphConstructor(proximity_threshold=50)
    G = constructor.construct_graph(features, masks)
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Convert to PyTorch Geometric
    converter = PyTorchGeometricConverter()
    pg_data = converter.to_pytorch_geometric(G)
    
    print(f"PyTorch Geometric format: {pg_data['x'].shape}")
