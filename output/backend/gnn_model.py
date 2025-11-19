"""
Graph Neural Network (GNN) model for protein localization
Uses superpixel-based graph construction
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from typing import Tuple, List
import networkx as nx


class GraphConstructor:
    """Construct graph from superpixel segmentation"""
    
    @staticmethod
    def extract_superpixel_features(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """
        Extract features for each superpixel
        
        Args:
            image: Original image
            segments: Superpixel segmentation
            
        Returns:
            Feature matrix (n_superpixels x n_features)
        """
        features = []
        n_segments = segments.max() + 1
        
        for seg_id in range(n_segments):
            mask = segments == seg_id
            
            if not mask.any():
                continue
            
            # Intensity features
            if len(image.shape) == 2:
                intensities = image[mask]
            else:
                intensities = image[mask].mean(axis=1)
            
            mean_intensity = np.mean(intensities)
            std_intensity = np.std(intensities)
            min_intensity = np.min(intensities)
            max_intensity = np.max(intensities)
            
            # Geometric features
            region_props = regionprops(mask.astype(int))[0] if mask.any() else None
            
            if region_props:
                area = region_props.area
                perimeter = region_props.perimeter
                eccentricity = region_props.eccentricity
                solidity = region_props.solidity
                centroid_y, centroid_x = region_props.centroid
            else:
                area = perimeter = eccentricity = solidity = 0
                centroid_y = centroid_x = 0
            
            # Texture features (simple)
            if len(intensities) > 1:
                entropy = -np.sum(np.histogram(intensities, bins=10)[0] * 
                                 np.log(np.histogram(intensities, bins=10)[0] + 1e-10))
            else:
                entropy = 0
            
            feature_vector = [
                mean_intensity, std_intensity, min_intensity, max_intensity,
                area, perimeter, eccentricity, solidity,
                centroid_x / image.shape[1], centroid_y / image.shape[0],
                entropy
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    @staticmethod
    def build_adjacency(segments: np.ndarray, k_neighbors: int = 5) -> np.ndarray:
        """
        Build adjacency matrix based on superpixel neighbors
        
        Args:
            segments: Superpixel segmentation
            k_neighbors: Number of nearest neighbors to connect
            
        Returns:
            Adjacency matrix
        """
        n_segments = segments.max() + 1
        
        # Get centroids
        centroids = []
        for seg_id in range(n_segments):
            mask = segments == seg_id
            if mask.any():
                y_coords, x_coords = np.where(mask)
                centroid = [np.mean(x_coords), np.mean(y_coords)]
                centroids.append(centroid)
            else:
                centroids.append([0, 0])
        
        centroids = np.array(centroids)
        
        # Compute distances
        distances = cdist(centroids, centroids, metric='euclidean')
        
        # Build adjacency based on spatial proximity
        adjacency = np.zeros((n_segments, n_segments))
        
        for i in range(n_segments):
            # Get k nearest neighbors
            nearest = np.argsort(distances[i])[1:k_neighbors+1]
            adjacency[i, nearest] = 1
            adjacency[nearest, i] = 1
        
        # Add direct neighbors (touching superpixels)
        for i in range(segments.shape[0] - 1):
            for j in range(segments.shape[1] - 1):
                current = segments[i, j]
                right = segments[i, j+1]
                down = segments[i+1, j]
                
                if current != right:
                    adjacency[current, right] = 1
                    adjacency[right, current] = 1
                
                if current != down:
                    adjacency[current, down] = 1
                    adjacency[down, current] = 1
        
        return adjacency
    
    @staticmethod
    def create_graph_data(features: np.ndarray, adjacency: np.ndarray, 
                         label: int = None) -> Data:
        """
        Create PyTorch Geometric Data object
        
        Args:
            features: Node features
            adjacency: Adjacency matrix
            label: Graph label (optional)
            
        Returns:
            PyG Data object
        """
        # Convert to edge index format
        edge_index = []
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[1]):
                if adjacency[i, j] > 0:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Convert features to tensor
        x = torch.tensor(features, dtype=torch.float)
        
        # Create data object
        if label is not None:
            y = torch.tensor([label], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
        else:
            data = Data(x=x, edge_index=edge_index)
        
        return data


class GCNModel(nn.Module):
    """Graph Convolutional Network for protein localization"""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 num_classes: int, num_layers: int = 3, dropout: float = 0.5):
        """
        Initialize GCN model
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, num_classes))
    
    def forward(self, data):
        """Forward pass"""
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = torch.mean(x, dim=0, keepdim=True)
        
        return F.log_softmax(x, dim=1)


class GATModel(nn.Module):
    """Graph Attention Network for protein localization"""
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_classes: int, num_layers: int = 3, 
                 heads: int = 4, dropout: float = 0.5):
        """Initialize GAT model"""
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        
        self.convs.append(GATConv(hidden_channels * heads, num_classes, heads=1))
    
    def forward(self, data):
        """Forward pass"""
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = torch.mean(x, dim=0, keepdim=True)
        
        return F.log_softmax(x, dim=1)


class GraphSAGEModel(nn.Module):
    """GraphSAGE model for protein localization"""
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_classes: int, num_layers: int = 3, dropout: float = 0.5):
        """Initialize GraphSAGE model"""
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, num_classes))
    
    def forward(self, data):
        """Forward pass"""
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = torch.mean(x, dim=0, keepdim=True)
        
        return F.log_softmax(x, dim=1)


class GNNClassifier:
    """Wrapper for GNN models"""
    
    def __init__(self, model_type: str = "GCN", in_channels: int = 11,
                 hidden_channels: int = 128, num_classes: int = 8,
                 num_layers: int = 3, dropout: float = 0.5):
        """
        Initialize GNN classifier
        
        Args:
            model_type: Type of GNN ('GCN', 'GAT', or 'GraphSAGE')
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            num_classes: Number of classes
            num_layers: Number of layers
            dropout: Dropout rate
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == "GCN":
            self.model = GCNModel(in_channels, hidden_channels, num_classes, 
                                 num_layers, dropout)
        elif model_type == "GAT":
            self.model = GATModel(in_channels, hidden_channels, num_classes,
                                 num_layers, dropout=dropout)
        elif model_type == "GraphSAGE":
            self.model = GraphSAGEModel(in_channels, hidden_channels, num_classes,
                                       num_layers, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
    
    def predict(self, data: Data) -> Tuple[int, np.ndarray]:
        """
        Predict protein localization
        
        Args:
            data: PyG Data object
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data)
            probabilities = torch.exp(out).cpu().numpy()[0]
            predicted_class = probabilities.argmax()
        
        return predicted_class, probabilities
