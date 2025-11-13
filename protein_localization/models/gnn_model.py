"""
Graph Neural Network (GNN) Model for protein localization prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool


class GNNModel(nn.Module):
    """Graph Neural Network for graph-level classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 output_dim: int = 10, num_layers: int = 3,
                 dropout: float = 0.3, conv_type: str = 'gcn'):
        """
        Initialize GNN model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution (gcn, gat, sage)
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for mean+max pooling
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment (num_nodes,)
            
        Returns:
            Logits (batch_size, output_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Graph convolution layers
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Residual connection
            if i > 0:
                x = x + x_res
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Output layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x


class NodeGNNModel(nn.Module):
    """Graph Neural Network for node-level classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 output_dim: int = 10, num_layers: int = 3,
                 dropout: float = 0.3, conv_type: str = 'gcn'):
        """
        Initialize node-level GNN model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution
        """
        super(NodeGNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Node logits (num_nodes, output_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Graph convolution layers
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Residual connection
            if i > 0:
                x = x + x_res
        
        # Output
        x = self.output(x)
        
        return x


def create_gnn_model(config: dict, input_dim: int) -> nn.Module:
    """
    Factory function to create GNN model from config
    
    Args:
        config: Model configuration
        input_dim: Input feature dimension
        
    Returns:
        GNN model
    """
    architecture = config.get('architecture', {})
    gnn_config = config.get('gnn', {})
    labels_config = config.get('labels', {})
    
    hidden_dim = architecture.get('hidden_dim', 128)
    num_layers = architecture.get('num_layers', 3)
    dropout = architecture.get('dropout', 0.3)
    output_dim = labels_config.get('num_classes', 10)
    conv_type = gnn_config.get('conv_type', 'gcn')
    
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type
    )
    
    return model
