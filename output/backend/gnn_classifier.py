"""
Graph Neural Network (GNN) implementation for superpixel-based classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric.data import Data
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
import networkx as nx


class GraphConstructor:
    """Construct graph from superpixel segmentation"""
    
    def __init__(self):
        pass
    
    def extract_superpixel_features(self, image, segments):
        """
        Extract features for each superpixel
        
        Args:
            image: Original image
            segments: Superpixel segmentation
            
        Returns:
            Feature matrix (num_superpixels x feature_dim)
        """
        props = regionprops(segments, intensity_image=image)
        features = []
        
        for prop in props:
            # Intensity features
            mean_intensity = prop.mean_intensity
            max_intensity = prop.max_intensity
            min_intensity = prop.min_intensity
            
            # Geometry features
            area = prop.area
            perimeter = prop.perimeter
            eccentricity = prop.eccentricity
            solidity = prop.solidity
            
            # Position features (normalized)
            centroid_y, centroid_x = prop.centroid
            centroid_y /= image.shape[0]
            centroid_x /= image.shape[1]
            
            # Texture features (simplified)
            std_intensity = np.std(prop.intensity_image)
            
            feature_vector = [
                mean_intensity, max_intensity, min_intensity,
                area, perimeter, eccentricity, solidity,
                centroid_x, centroid_y, std_intensity
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def build_adjacency_graph(self, segments):
        """
        Build adjacency graph from superpixels
        
        Args:
            segments: Superpixel segmentation
            
        Returns:
            Edge index for graph
        """
        # Find adjacent superpixels
        edges = []
        unique_segments = np.unique(segments)
        
        # Check 4-connectivity
        for i in range(segments.shape[0] - 1):
            for j in range(segments.shape[1] - 1):
                current = segments[i, j]
                right = segments[i, j + 1]
                down = segments[i + 1, j]
                
                if current != right:
                    edges.append((current - 1, right - 1))  # Adjust for 0-indexing
                if current != down:
                    edges.append((current - 1, down - 1))
        
        # Remove duplicates and create edge index
        edges = list(set(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def construct_graph(self, image, segments):
        """
        Construct complete graph from image and segmentation
        
        Args:
            image: Original image
            segments: Superpixel segmentation
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract features
        node_features = self.extract_superpixel_features(image, segments)
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Build edges
        edge_index = self.build_adjacency_graph(segments)
        
        # Create graph data
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        return graph_data


class GNNClassifier(nn.Module):
    """Graph Neural Network for protein localization classification"""
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=5, 
                 num_layers=3, dropout=0.5, model_type='gcn'):
        super(GNNClassifier, self).__init__()
        
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        if model_type == 'gcn':
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        elif model_type == 'gat':
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (mean of all nodes)
        x = torch.mean(x, dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x


class GNNPredictor:
    """Wrapper for GNN predictions"""
    
    def __init__(self, input_dim=10, hidden_dim=64, num_classes=5, 
                 model_path=None, model_type='gcn'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GNNClassifier(input_dim, hidden_dim, num_classes, 
                                   model_type=model_type)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.graph_constructor = GraphConstructor()
    
    def predict(self, image, segments):
        """
        Predict protein localization using GNN
        
        Args:
            image: Original image
            segments: Superpixel segmentation
            
        Returns:
            probabilities: Class probabilities
            predicted_class: Predicted class index
            confidence: Confidence score
        """
        # Construct graph
        graph_data = self.graph_constructor.construct_graph(image, segments)
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(graph_data)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        return (probabilities.cpu().numpy()[0],
                predicted_class.cpu().item(),
                confidence.cpu().item())
    
    def extract_graph_embeddings(self, image, segments):
        """
        Extract graph embeddings for fusion
        
        Args:
            image: Original image
            segments: Superpixel segmentation
            
        Returns:
            Graph embedding vector
        """
        graph_data = self.graph_constructor.construct_graph(image, segments)
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            x, edge_index = graph_data.x, graph_data.edge_index
            for i, conv in enumerate(self.model.convs):
                x = conv(x, edge_index)
                if i < len(self.model.convs) - 1:
                    x = F.relu(x)
            embedding = torch.mean(x, dim=0)
        
        return embedding.cpu().numpy()
