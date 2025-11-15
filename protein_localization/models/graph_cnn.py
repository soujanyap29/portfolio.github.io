"""
Graph Convolutional Neural Network for protein localization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphCNN(nn.Module):
    """Graph Convolutional Neural Network"""
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 10,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Initialize Graph CNN
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden units
            out_channels: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(GraphCNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Try to use PyTorch Geometric layers if available
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            self.use_pyg = True
            self.pool = global_mean_pool
            
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            # Input layer
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Output layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, out_channels)
            )
            
        except ImportError:
            print("PyTorch Geometric not available. Using simple MLP.")
            self.use_pyg = False
            
            # Fallback to simple MLP
            layers = []
            prev_channels = in_channels
            
            for i in range(num_layers):
                if i == num_layers - 1:
                    layers.append(nn.Linear(prev_channels, out_channels))
                else:
                    layers.append(nn.Linear(prev_channels, hidden_channels))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    layers.append(nn.BatchNorm1d(hidden_channels))
                    prev_channels = hidden_channels
            
            self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, edge_index=None, batch=None, edge_weight=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_weight: Edge weights [num_edges]
        
        Returns:
            Output predictions
        """
        if self.use_pyg and edge_index is not None:
            # Use GCN layers
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = bn(x)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Pool node features to graph-level
            if batch is not None:
                x = self.pool(x, batch)
            else:
                x = torch.mean(x, dim=0, keepdim=True)
            
            # Classify
            x = self.classifier(x)
        else:
            # Use simple MLP
            x = self.mlp(x)
        
        return x


class GATModel(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 10,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.5):
        """
        Initialize GAT
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            out_channels: Number of output classes
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GATModel, self).__init__()
        
        try:
            from torch_geometric.nn import GATConv, global_mean_pool
            
            self.convs = nn.ModuleList()
            self.num_layers = num_layers
            self.dropout = dropout
            self.pool = global_mean_pool
            
            # Input layer
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                        heads=heads, dropout=dropout))
            
            # Output layer
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=1, dropout=dropout))
            
            # Classification head
            self.classifier = nn.Linear(hidden_channels, out_channels)
            
        except ImportError:
            raise ImportError("PyTorch Geometric required for GAT model")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool
        if batch is not None:
            x = self.pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classify
        x = self.classifier(x)
        
        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE model"""
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 10,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """Initialize GraphSAGE"""
        super(GraphSAGEModel, self).__init__()
        
        try:
            from torch_geometric.nn import SAGEConv, global_mean_pool
            
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.num_layers = num_layers
            self.dropout = dropout
            self.pool = global_mean_pool
            
            # Input layer
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Output layer
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Classifier
            self.classifier = nn.Linear(hidden_channels, out_channels)
            
        except ImportError:
            raise ImportError("PyTorch Geometric required for GraphSAGE model")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass"""
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool
        if batch is not None:
            x = self.pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classify
        x = self.classifier(x)
        
        return x


def create_graph_model(model_type: str = 'gcn', **kwargs):
    """
    Factory function to create graph models
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'sage')
        **kwargs: Model parameters
    
    Returns:
        Model instance
    """
    if model_type.lower() == 'gcn':
        return GraphCNN(**kwargs)
    elif model_type.lower() == 'gat':
        return GATModel(**kwargs)
    elif model_type.lower() == 'sage':
        return GraphSAGEModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing Graph CNN models...")
    
    # Test with dummy data
    in_channels = 20
    hidden_channels = 64
    out_channels = 5
    num_nodes = 100
    num_edges = 200
    
    # Create dummy data
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Test GCN
    model = GraphCNN(in_channels, hidden_channels, out_channels)
    model.eval()
    
    with torch.no_grad():
        output = model(x, edge_index, batch)
    
    print(f"GCN output shape: {output.shape}")
    print("Graph CNN models test complete!")
