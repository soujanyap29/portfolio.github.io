"""
Superpixel generation and graph construction for GNN.
"""

import numpy as np
from skimage.segmentation import slic, felzenszwalb
from skimage import measure, color
from scipy.spatial import Delaunay
import networkx as nx
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperpixelGenerator:
    """Generate superpixels from images."""
    
    def __init__(self, method: str = 'slic', n_segments: int = 100, compactness: float = 10):
        """
        Initialize superpixel generator.
        
        Args:
            method: Superpixel method ('slic' or 'felzenszwalb')
            n_segments: Target number of superpixels
            compactness: Compactness parameter for SLIC
        """
        self.method = method
        self.n_segments = n_segments
        self.compactness = compactness
    
    def generate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate superpixels.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Superpixel labels
        """
        if self.method == 'slic':
            segments = slic(
                image,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=0
            )
        elif self.method == 'felzenszwalb':
            segments = felzenszwalb(
                image,
                scale=100,
                sigma=0.5,
                min_size=50
            )
        else:
            raise ValueError(f"Unknown superpixel method: {self.method}")
        
        logger.info(f"Generated {len(np.unique(segments))} superpixels")
        return segments
    
    def extract_features(self, image: np.ndarray, segments: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract features for each superpixel.
        
        Args:
            image: Original image
            segments: Superpixel labels
            
        Returns:
            Dictionary mapping superpixel ID to feature vector
        """
        features = {}
        
        for segment_id in np.unique(segments):
            mask = (segments == segment_id)
            
            # Intensity features
            region_pixels = image[mask]
            if len(region_pixels.shape) == 1:
                # Grayscale
                mean_intensity = np.mean(region_pixels)
                std_intensity = np.std(region_pixels)
                max_intensity = np.max(region_pixels)
                min_intensity = np.min(region_pixels)
            else:
                # RGB
                mean_intensity = np.mean(region_pixels, axis=0)
                std_intensity = np.std(region_pixels, axis=0)
                max_intensity = np.max(region_pixels, axis=0)
                min_intensity = np.min(region_pixels, axis=0)
            
            # Geometric features using regionprops
            props = measure.regionprops(mask.astype(int))
            if props:
                prop = props[0]
                area = prop.area
                perimeter = prop.perimeter
                eccentricity = prop.eccentricity
                centroid = prop.centroid
            else:
                area = np.sum(mask)
                perimeter = 0
                eccentricity = 0
                centroid = (0, 0)
            
            # Combine features
            feature_vector = np.concatenate([
                np.atleast_1d(mean_intensity).flatten(),
                np.atleast_1d(std_intensity).flatten(),
                np.atleast_1d(max_intensity).flatten(),
                np.atleast_1d(min_intensity).flatten(),
                [area, perimeter, eccentricity],
                centroid
            ])
            
            features[segment_id] = feature_vector.astype(np.float32)
        
        return features


class GraphConstructor:
    """Construct graphs from superpixels."""
    
    @staticmethod
    def build_adjacency_graph(segments: np.ndarray) -> nx.Graph:
        """
        Build graph based on spatial adjacency.
        
        Args:
            segments: Superpixel labels
            
        Returns:
            NetworkX graph
        """
        graph = nx.Graph()
        
        # Add all nodes
        unique_segments = np.unique(segments)
        graph.add_nodes_from(unique_segments)
        
        # Find adjacent superpixels
        # Check 4-connectivity
        h, w = segments.shape
        for i in range(h):
            for j in range(w):
                current = segments[i, j]
                
                # Check right neighbor
                if j < w - 1:
                    neighbor = segments[i, j + 1]
                    if current != neighbor:
                        graph.add_edge(current, neighbor)
                
                # Check bottom neighbor
                if i < h - 1:
                    neighbor = segments[i + 1, j]
                    if current != neighbor:
                        graph.add_edge(current, neighbor)
        
        logger.info(f"Built adjacency graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    @staticmethod
    def build_delaunay_graph(features: Dict[int, np.ndarray]) -> nx.Graph:
        """
        Build graph based on Delaunay triangulation of centroids.
        
        Args:
            features: Dictionary of node features (must contain centroid info)
            
        Returns:
            NetworkX graph
        """
        graph = nx.Graph()
        
        # Extract centroids (last 2 elements of feature vector)
        nodes = list(features.keys())
        centroids = np.array([features[node][-2:] for node in nodes])
        
        # Add nodes
        graph.add_nodes_from(nodes)
        
        # Delaunay triangulation
        if len(centroids) >= 3:
            tri = Delaunay(centroids)
            
            # Add edges from triangulation
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        node1 = nodes[simplex[i]]
                        node2 = nodes[simplex[j]]
                        graph.add_edge(node1, node2)
        
        logger.info(f"Built Delaunay graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    @staticmethod
    def build_knn_graph(features: Dict[int, np.ndarray], k: int = 5) -> nx.Graph:
        """
        Build k-nearest neighbors graph based on feature similarity.
        
        Args:
            features: Dictionary of node features
            k: Number of nearest neighbors
            
        Returns:
            NetworkX graph
        """
        from sklearn.neighbors import NearestNeighbors
        
        graph = nx.Graph()
        
        # Prepare data
        nodes = list(features.keys())
        feature_matrix = np.array([features[node] for node in nodes])
        
        # Add nodes
        graph.add_nodes_from(nodes)
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(feature_matrix)
        distances, indices = nbrs.kneighbors(feature_matrix)
        
        # Add edges
        for i, node in enumerate(nodes):
            for j in range(1, k + 1):  # Skip first (itself)
                neighbor = nodes[indices[i, j]]
                weight = 1.0 / (distances[i, j] + 1e-6)
                graph.add_edge(node, neighbor, weight=weight)
        
        logger.info(f"Built k-NN graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    @staticmethod
    def to_pytorch_geometric(graph: nx.Graph, 
                            features: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert NetworkX graph to PyTorch Geometric format.
        
        Args:
            graph: NetworkX graph
            features: Node features
            
        Returns:
            Tuple of (edge_index, node_features)
        """
        # Create node mapping
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Extract edges
        edge_list = []
        for u, v in graph.edges():
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            edge_list.append([node_to_idx[v], node_to_idx[u]])  # Undirected
        
        edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0))
        
        # Extract node features
        node_features = np.array([features[node] for node in nodes])
        
        return edge_index, node_features


def visualize_superpixels(image: np.ndarray, segments: np.ndarray, save_path: str = None):
    """
    Visualize superpixels.
    
    Args:
        image: Original image
        segments: Superpixel labels
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    
    # Create visualization
    if len(image.shape) == 2:
        # Convert grayscale to RGB for visualization
        image_rgb = color.gray2rgb(image)
    else:
        image_rgb = image
    
    # Normalize if needed
    if image_rgb.max() > 1:
        image_rgb = image_rgb / image_rgb.max()
    
    marked = mark_boundaries(image_rgb, segments, color=(1, 0, 0), mode='thick')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(marked)
    axes[1].set_title(f'Superpixels (n={len(np.unique(segments))})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved superpixel visualization to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    from skimage import data
    
    # Load example image
    image = data.astronaut()
    
    # Generate superpixels
    sp_gen = SuperpixelGenerator(method='slic', n_segments=100)
    segments = sp_gen.generate(image)
    
    # Extract features
    features = sp_gen.extract_features(image, segments)
    print(f"Generated {len(features)} superpixels")
    print(f"Feature vector size: {features[0].shape}")
    
    # Build graph
    constructor = GraphConstructor()
    graph = constructor.build_adjacency_graph(segments)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Convert to PyTorch Geometric format
    edge_index, node_features = constructor.to_pytorch_geometric(graph, features)
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Node features shape: {node_features.shape}")
