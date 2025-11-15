"""
Combined CNN + Graph-CNN model for protein localization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CombinedModel(nn.Module):
    """Combined CNN and Graph-CNN model"""
    
    def __init__(self,
                 cnn_model: nn.Module,
                 graph_model: nn.Module,
                 fusion_method: str = 'concat',
                 num_classes: int = 10):
        """
        Initialize combined model
        
        Args:
            cnn_model: CNN feature extractor
            graph_model: Graph-CNN model
            fusion_method: Method to fuse features ('concat', 'add', 'attention')
            num_classes: Number of output classes
        """
        super(CombinedModel, self).__init__()
        
        self.cnn_model = cnn_model
        self.graph_model = graph_model
        self.fusion_method = fusion_method
        
        # Determine feature dimensions
        self.cnn_feat_dim = self._get_cnn_output_dim()
        self.graph_feat_dim = self._get_graph_output_dim()
        
        # Fusion layers
        if fusion_method == 'concat':
            combined_dim = self.cnn_feat_dim + self.graph_feat_dim
            self.fusion = nn.Identity()
        elif fusion_method == 'add':
            # Project to same dimension
            combined_dim = max(self.cnn_feat_dim, self.graph_feat_dim)
            self.cnn_proj = nn.Linear(self.cnn_feat_dim, combined_dim)
            self.graph_proj = nn.Linear(self.graph_feat_dim, combined_dim)
        elif fusion_method == 'attention':
            combined_dim = max(self.cnn_feat_dim, self.graph_feat_dim)
            self.cnn_proj = nn.Linear(self.cnn_feat_dim, combined_dim)
            self.graph_proj = nn.Linear(self.graph_feat_dim, combined_dim)
            self.attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ReLU(),
                nn.Linear(combined_dim // 2, 2),
                nn.Softmax(dim=-1)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(combined_dim // 2, num_classes)
        )
    
    def _get_cnn_output_dim(self) -> int:
        """Determine CNN output dimension"""
        # Extract features and get dimension
        if hasattr(self.cnn_model, 'extract_features'):
            dummy_input = torch.randn(1, 1, 224, 224)
            with torch.no_grad():
                features = self.cnn_model.extract_features(dummy_input)
            return features.shape[1]
        else:
            return 512  # Default
    
    def _get_graph_output_dim(self) -> int:
        """Determine graph model output dimension"""
        return 64  # Default hidden dimension
    
    def forward(self, 
                image: torch.Tensor,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            image: Input image [batch_size, channels, height, width]
            node_features: Graph node features
            edge_index: Graph edge indices
            batch: Batch assignment for graph
        
        Returns:
            Class predictions
        """
        # Extract CNN features
        if hasattr(self.cnn_model, 'extract_features'):
            cnn_features = self.cnn_model.extract_features(image)
        else:
            # Extract from intermediate layer
            cnn_features = self.cnn_model.features(image)
            cnn_features = torch.flatten(cnn_features, 1)
        
        # Extract graph features
        graph_features = self.graph_model(node_features, edge_index, batch)
        
        # Ensure same batch size
        if cnn_features.shape[0] != graph_features.shape[0]:
            # Repeat CNN features if needed
            if cnn_features.shape[0] == 1:
                cnn_features = cnn_features.repeat(graph_features.shape[0], 1)
            else:
                # Average pool graph features if needed
                graph_features = torch.mean(graph_features, dim=0, keepdim=True)
                graph_features = graph_features.repeat(cnn_features.shape[0], 1)
        
        # Fuse features
        if self.fusion_method == 'concat':
            combined = torch.cat([cnn_features, graph_features], dim=1)
        elif self.fusion_method == 'add':
            cnn_proj = self.cnn_proj(cnn_features)
            graph_proj = self.graph_proj(graph_features)
            combined = cnn_proj + graph_proj
        elif self.fusion_method == 'attention':
            cnn_proj = self.cnn_proj(cnn_features)
            graph_proj = self.graph_proj(graph_features)
            
            # Calculate attention weights
            combined_for_attn = cnn_proj + graph_proj
            attn_weights = self.attention(combined_for_attn)
            
            # Apply attention
            combined = attn_weights[:, 0:1] * cnn_proj + attn_weights[:, 1:2] * graph_proj
        
        # Classify
        output = self.classifier(combined)
        
        return output


class HierarchicalModel(nn.Module):
    """Hierarchical model with image and graph branches"""
    
    def __init__(self,
                 cnn_model: nn.Module,
                 graph_model: nn.Module,
                 num_classes: int = 10):
        """Initialize hierarchical model"""
        super(HierarchicalModel, self).__init__()
        
        self.cnn_branch = cnn_model
        self.graph_branch = graph_model
        
        # Separate classifiers for each branch
        self.cnn_classifier = nn.Linear(512, num_classes)
        self.graph_classifier = nn.Linear(64, num_classes)
        
        # Combined classifier
        self.combined_classifier = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes * 2, num_classes)
        )
    
    def forward(self,
                image: torch.Tensor,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None):
        """Forward pass with multi-task learning"""
        
        # CNN branch
        if hasattr(self.cnn_branch, 'extract_features'):
            cnn_features = self.cnn_branch.extract_features(image)
        else:
            cnn_features = self.cnn_branch(image)
        
        cnn_pred = self.cnn_classifier(cnn_features)
        
        # Graph branch
        graph_features = self.graph_branch(node_features, edge_index, batch)
        graph_pred = self.graph_classifier(graph_features)
        
        # Combined prediction
        combined_features = torch.cat([cnn_pred, graph_pred], dim=1)
        combined_pred = self.combined_classifier(combined_features)
        
        return combined_pred, cnn_pred, graph_pred


def create_combined_model(cnn_type: str = 'vgg16',
                          graph_type: str = 'gcn',
                          fusion_method: str = 'concat',
                          num_classes: int = 10,
                          **kwargs):
    """
    Create a combined CNN + Graph-CNN model
    
    Args:
        cnn_type: Type of CNN ('vgg16', 'resnet50', 'custom')
        graph_type: Type of graph model ('gcn', 'gat', 'sage')
        fusion_method: Feature fusion method
        num_classes: Number of classes
        **kwargs: Additional model parameters
    
    Returns:
        Combined model
    """
    from .vgg16 import create_cnn_model, VGG16FeatureExtractor
    from .graph_cnn import create_graph_model
    
    # Create CNN feature extractor
    if cnn_type == 'vgg16':
        cnn_model = VGG16FeatureExtractor(pretrained=kwargs.get('pretrained', True),
                                          in_channels=kwargs.get('in_channels', 1))
    else:
        cnn_model = create_cnn_model(cnn_type, **kwargs)
    
    # Create graph model
    graph_model = create_graph_model(
        graph_type,
        in_channels=kwargs.get('graph_in_channels', 20),
        hidden_channels=kwargs.get('hidden_channels', 64),
        out_channels=64,  # Feature dimension
        num_layers=kwargs.get('num_layers', 3),
        dropout=kwargs.get('dropout', 0.5)
    )
    
    # Create combined model
    combined = CombinedModel(
        cnn_model=cnn_model,
        graph_model=graph_model,
        fusion_method=fusion_method,
        num_classes=num_classes
    )
    
    return combined


if __name__ == "__main__":
    print("Testing combined model...")
    
    from .vgg16 import VGG16FeatureExtractor
    from .graph_cnn import GraphCNN
    
    # Create models
    cnn = VGG16FeatureExtractor(pretrained=False, in_channels=1)
    graph = GraphCNN(in_channels=20, hidden_channels=64, out_channels=64, num_layers=3)
    
    # Create combined model
    model = CombinedModel(cnn, graph, fusion_method='concat', num_classes=5)
    model.eval()
    
    # Test with dummy data
    image = torch.randn(2, 1, 224, 224)
    node_features = torch.randn(100, 20)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)
    
    with torch.no_grad():
        output = model(image, node_features, edge_index, batch)
    
    print(f"Combined model output shape: {output.shape}")
    print("Combined model test complete!")
