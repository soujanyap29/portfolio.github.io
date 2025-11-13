"""
Graph Builder Module
Constructs graph representations from preprocessed images
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from skimage import segmentation, measure, feature
from scipy.spatial import distance
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build graph representations from images"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize graph builder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default graph construction configuration"""
        return {
            'superpixel_method': 'slic',
            'num_segments': 500,
            'compactness': 10,
            'edge_threshold': 0.5,
            'edge_feature': 'distance',
            'node_features': [
                'intensity_mean',
                'intensity_std',
                'texture_contrast',
                'texture_homogeneity',
                'morphology_area',
                'morphology_eccentricity'
            ]
        }
    
    def build_graph(self, image: np.ndarray, filename: str = "") -> Dict:
        """
        Build graph from image
        
        Args:
            image: Preprocessed image
            filename: Optional filename for logging
            
        Returns:
            Dictionary containing graph data
        """
        logger.debug(f"Building graph for {filename or 'image'}")
        
        # Handle multi-channel images
        if image.ndim == 3:
            # Use first channel or average
            if image.shape[2] == 1:
                image_2d = image[:, :, 0]
            else:
                image_2d = np.mean(image, axis=2)
        else:
            image_2d = image
        
        # Step 1: Segment image into superpixels
        segments = self._segment_image(image_2d)
        
        # Step 2: Extract node features
        node_features = self._extract_node_features(image_2d, segments)
        
        # Step 3: Build adjacency and edges
        edges, edge_features = self._build_edges(segments, node_features)
        
        # Step 4: Create graph structure
        graph_data = {
            'num_nodes': len(node_features),
            'node_features': node_features,
            'edges': edges,
            'edge_features': edge_features,
            'segments': segments,
            'metadata': {
                'filename': filename,
                'image_shape': image.shape,
                'num_segments': len(node_features)
            }
        }
        
        return graph_data
    
    def _segment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image into superpixels
        
        Args:
            image: 2D image array
            
        Returns:
            Segmentation map (label image)
        """
        method = self.config.get('superpixel_method', 'slic')
        n_segments = self.config.get('num_segments', 500)
        
        if method == 'slic':
            segments = segmentation.slic(
                image,
                n_segments=n_segments,
                compactness=self.config.get('compactness', 10),
                start_label=0
            )
        
        elif method == 'felzenszwalb':
            segments = segmentation.felzenszwalb(
                image,
                scale=100,
                sigma=0.5,
                min_size=50
            )
        
        elif method == 'watershed':
            # Simple gradient-based watershed
            from skimage import filters, morphology
            edges = filters.sobel(image)
            markers = np.zeros_like(image, dtype=int)
            markers[image < np.percentile(image, 25)] = 1
            markers[image > np.percentile(image, 75)] = 2
            segments = segmentation.watershed(edges, markers)
        
        else:
            logger.warning(f"Unknown segmentation method: {method}, using SLIC")
            segments = segmentation.slic(image, n_segments=n_segments)
        
        return segments
    
    def _extract_node_features(self, image: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """
        Extract features for each node (superpixel)
        
        Args:
            image: 2D image array
            segments: Segmentation map
            
        Returns:
            Node feature matrix (num_nodes, num_features)
        """
        props = measure.regionprops(segments + 1, intensity_image=image)
        
        feature_list = []
        feature_names = self.config.get('node_features', [])
        
        for region in props:
            features = []
            
            if 'intensity_mean' in feature_names:
                features.append(region.mean_intensity)
            
            if 'intensity_std' in feature_names:
                # Calculate standard deviation
                mask = segments == (region.label - 1)
                features.append(np.std(image[mask]))
            
            if 'texture_contrast' in feature_names or 'texture_homogeneity' in feature_names:
                # Extract texture features using GLCM
                mask = segments == (region.label - 1)
                region_image = image[mask].reshape(region.bbox[2] - region.bbox[0],
                                                   region.bbox[3] - region.bbox[1])
                texture_features = self._extract_texture_features(region_image)
                
                if 'texture_contrast' in feature_names:
                    features.append(texture_features['contrast'])
                if 'texture_homogeneity' in feature_names:
                    features.append(texture_features['homogeneity'])
            
            if 'morphology_area' in feature_names:
                features.append(region.area)
            
            if 'morphology_eccentricity' in feature_names:
                features.append(region.eccentricity)
            
            if 'morphology_solidity' in feature_names:
                features.append(region.solidity)
            
            if 'morphology_extent' in feature_names:
                features.append(region.extent)
            
            feature_list.append(features)
        
        node_features = np.array(feature_list, dtype=np.float32)
        
        # Normalize features
        node_features = self._normalize_features(node_features)
        
        return node_features
    
    def _extract_texture_features(self, region_image: np.ndarray) -> Dict:
        """Extract texture features from region"""
        try:
            # Simple texture measures
            contrast = np.std(region_image)
            homogeneity = 1.0 / (1.0 + contrast)
            
            return {
                'contrast': contrast,
                'homogeneity': homogeneity
            }
        except:
            return {'contrast': 0.0, 'homogeneity': 0.5}
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (features - min_vals) / range_vals
        
        return normalized
    
    def _build_edges(self, segments: np.ndarray, node_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges between adjacent superpixels
        
        Args:
            segments: Segmentation map
            node_features: Node features
            
        Returns:
            Tuple of (edge_index, edge_features)
        """
        # Find adjacent segments
        adjacency = self._find_adjacency(segments)
        
        # Build edge list
        edges = []
        edge_features = []
        
        for node_i, neighbors in adjacency.items():
            for node_j in neighbors:
                if node_i < node_j:  # Avoid duplicate edges
                    edges.append([node_i, node_j])
                    
                    # Calculate edge features
                    edge_feat = self._calculate_edge_features(
                        node_i, node_j, node_features, segments
                    )
                    edge_features.append(edge_feat)
        
        if not edges:
            # Create at least one self-loop if no edges
            edges = [[0, 0]]
            edge_features = [[1.0]]
        
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.float32)
        
        return edge_index, edge_attr
    
    def _find_adjacency(self, segments: np.ndarray) -> Dict[int, List[int]]:
        """Find adjacent segments"""
        adjacency = {}
        
        # Check horizontal neighbors
        for i in range(segments.shape[0]):
            for j in range(segments.shape[1] - 1):
                s1, s2 = segments[i, j], segments[i, j + 1]
                if s1 != s2:
                    adjacency.setdefault(s1, set()).add(s2)
                    adjacency.setdefault(s2, set()).add(s1)
        
        # Check vertical neighbors
        for i in range(segments.shape[0] - 1):
            for j in range(segments.shape[1]):
                s1, s2 = segments[i, j], segments[i + 1, j]
                if s1 != s2:
                    adjacency.setdefault(s1, set()).add(s2)
                    adjacency.setdefault(s2, set()).add(s1)
        
        # Convert sets to lists
        adjacency = {k: list(v) for k, v in adjacency.items()}
        
        return adjacency
    
    def _calculate_edge_features(self, node_i: int, node_j: int,
                                 node_features: np.ndarray, 
                                 segments: np.ndarray) -> List[float]:
        """Calculate edge features between two nodes"""
        edge_type = self.config.get('edge_feature', 'distance')
        
        if edge_type == 'distance':
            # Euclidean distance in feature space
            dist = np.linalg.norm(node_features[node_i] - node_features[node_j])
            return [dist]
        
        elif edge_type == 'similarity':
            # Cosine similarity
            similarity = np.dot(node_features[node_i], node_features[node_j])
            return [similarity]
        
        else:
            return [1.0]
    
    def build_batch(self, images: Dict[str, np.ndarray],
                   output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """
        Build graphs for a batch of images
        
        Args:
            images: Dictionary mapping filenames to images
            output_dir: Optional directory to save graphs
            
        Returns:
            Dictionary of graph data
        """
        logger.info(f"Building graphs for {len(images)} images...")
        
        graphs = {}
        
        for filename, image in tqdm(images.items(), desc="Building graphs"):
            try:
                graph_data = self.build_graph(image, filename)
                graphs[filename] = graph_data
                
                # Save if output directory specified
                if output_dir:
                    self._save_graph(graph_data, filename, output_dir)
                    
            except Exception as e:
                logger.error(f"Error building graph for {filename}: {str(e)}")
                continue
        
        logger.info(f"Successfully built {len(graphs)} graphs")
        return graphs
    
    def _save_graph(self, graph_data: Dict, filename: str, output_dir: str):
        """Save graph data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Change extension to .pkl
        output_file = output_path / f"{Path(filename).stem}_graph.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(graph_data, f)


def build_graph_from_image(image: np.ndarray, config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to build graph from single image
    
    Args:
        image: Input image
        config: Optional configuration
        
    Returns:
        Graph data dictionary
    """
    builder = GraphBuilder(config)
    return builder.build_graph(image)


if __name__ == "__main__":
    import argparse
    from data_loader import TIFFDataLoader
    from preprocessor import ImagePreprocessor
    
    parser = argparse.ArgumentParser(description="Build graphs from images")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing preprocessed images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save graphs")
    parser.add_argument("--num_segments", type=int, default=500,
                       help="Number of superpixel segments")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'num_segments': args.num_segments,
    }
    
    # Load images
    loader = TIFFDataLoader(args.input_dir)
    images_dict = loader.load_all(validate=False)
    images = {k: v[0] for k, v in images_dict.items()}
    
    # Build graphs
    builder = GraphBuilder(config)
    graphs = builder.build_batch(images, args.output_dir)
    
    logger.info(f"Saved {len(graphs)} graphs to {args.output_dir}")
