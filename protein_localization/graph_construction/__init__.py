"""
Graph construction module for creating biological graphs from segmented images
"""

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalGraphBuilder:
    """Build biological graphs from segmented neuronal components"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_config = config.get('graph', {})
        
    def build_graph(self, features: Dict) -> nx.Graph:
        """
        Build a graph from extracted features
        
        Args:
            features: Dictionary of region features
            
        Returns:
            NetworkX graph with nodes and edges
        """
        logger.info("Building biological graph...")
        
        G = nx.Graph()
        
        # Add nodes
        n_regions = len(features['region_ids'])
        for i in range(n_regions):
            node_attrs = {
                'centroid': features['centroids'][i],
                'area': features['areas'][i],
                'perimeter': features['perimeters'][i],
                'eccentricity': features['eccentricities'][i],
                'mean_intensity': features['mean_intensities'][i],
                'max_intensity': features['max_intensities'][i],
                'min_intensity': features['min_intensities'][i],
                'std_intensity': features['std_intensities'][i]
            }
            G.add_node(features['region_ids'][i], **node_attrs)
        
        # Add edges based on spatial proximity
        self._add_proximity_edges(G, features)
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _add_proximity_edges(self, G: nx.Graph, features: Dict) -> None:
        """
        Add edges between spatially proximate nodes
        
        Args:
            G: NetworkX graph
            features: Dictionary of region features
        """
        centroids = np.array(features['centroids'])
        threshold = self.graph_config.get('proximity_threshold', 50)
        
        # Compute pairwise distances
        distances = distance_matrix(centroids, centroids)
        
        # Add edges for nodes within threshold
        region_ids = features['region_ids']
        for i in range(len(region_ids)):
            for j in range(i + 1, len(region_ids)):
                dist = distances[i, j]
                if dist < threshold:
                    # Edge attributes
                    intensity_sim = 1.0 - abs(
                        features['mean_intensities'][i] - 
                        features['mean_intensities'][j]
                    ) / max(
                        features['mean_intensities'][i],
                        features['mean_intensities'][j]
                    )
                    
                    G.add_edge(
                        region_ids[i],
                        region_ids[j],
                        distance=dist,
                        intensity_similarity=intensity_sim
                    )
    
    def graph_to_pytorch_geometric(self, G: nx.Graph, labels: List[int] = None):
        """
        Convert NetworkX graph to PyTorch Geometric format
        
        Args:
            G: NetworkX graph
            labels: Node labels for classification
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            logger.warning("PyTorch Geometric not available")
            return None
        
        # Extract node features
        node_features = []
        node_ids = sorted(G.nodes())
        
        for node in node_ids:
            attrs = G.nodes[node]
            features = [
                attrs['area'],
                attrs['perimeter'],
                attrs['eccentricity'],
                attrs['mean_intensity'],
                attrs['max_intensity'],
                attrs['min_intensity'],
                attrs['std_intensity']
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edges
        edge_index = []
        edge_attr = []
        
        for edge in G.edges():
            src, dst = edge
            src_idx = node_ids.index(src)
            dst_idx = node_ids.index(dst)
            
            # Add both directions for undirected graph
            edge_index.append([src_idx, dst_idx])
            edge_index.append([dst_idx, src_idx])
            
            edge_data = G.edges[edge]
            edge_features = [
                edge_data.get('distance', 0),
                edge_data.get('intensity_similarity', 0)
            ]
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Add labels if provided
        y = None
        if labels is not None:
            y = torch.tensor(labels, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        logger.info(f"Converted to PyTorch Geometric: {data}")
        return data
    
    def save_graph(self, G: nx.Graph, output_path: str) -> None:
        """Save graph to file"""
        nx.write_gpickle(G, output_path)
        logger.info(f"Graph saved to: {output_path}")
    
    def load_graph(self, input_path: str) -> nx.Graph:
        """Load graph from file"""
        G = nx.read_gpickle(input_path)
        logger.info(f"Graph loaded from: {input_path}")
        return G
