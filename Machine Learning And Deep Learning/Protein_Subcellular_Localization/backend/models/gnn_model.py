"""
Graph Neural Network models for protein localization.
Supports GCN, GraphSAGE, and GAT architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCNModel(nn.Module):
    """Graph Convolutional Network."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_classes: int = 5,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(GCNModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        
        logger.info(f"Initialized GCN: {num_layers} layers, {hidden_dim} hidden dim")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE model."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_classes: int = 5,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of SAGE layers
            dropout: Dropout probability
        """
        super(GraphSAGEModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        
        logger.info(f"Initialized GraphSAGE: {num_layers} layers, {hidden_dim} hidden dim")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x


class GATModel(nn.Module):
    """Graph Attention Network."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_classes: int = 5,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 num_heads: int = 4):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GAT layers
            dropout: Dropout probability
            num_heads: Number of attention heads
        """
        super(GATModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
        
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        
        logger.info(f"Initialized GAT: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x


class GNNTrainer:
    """Trainer for GNN models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            model: GNN model
            device: Device to train on
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        logger.info(f"Initialized GNN trainer on device: {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in dataloader:
            data = data.to(self.device)
            
            # Forward pass
            outputs = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(outputs, data.y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.criterion(outputs, data.y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              save_path: Optional[str] = None) -> dict:
        """Full training loop."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved best model to {save_path}")
        
        return history
    
    def predict(self, node_features: np.ndarray, edge_index: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict class for a graph.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge index array
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.model.eval()
        
        # Convert to tensors
        x = torch.from_numpy(node_features).float().to(self.device)
        edge_idx = torch.from_numpy(edge_index).long().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x, edge_idx)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities


def create_gnn_model(model_type: str, input_dim: int, num_classes: int = 5, **kwargs) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: Type of GNN ('GCN', 'GraphSAGE', 'GAT')
        input_dim: Input feature dimension
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        GNN model
    """
    if model_type.upper() == 'GCN':
        return GCNModel(input_dim, num_classes=num_classes, **kwargs)
    elif model_type.upper() == 'GRAPHSAGE':
        return GraphSAGEModel(input_dim, num_classes=num_classes, **kwargs)
    elif model_type.upper() == 'GAT':
        return GATModel(input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown GNN model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    input_dim = 20
    num_classes = 5
    
    # Create models
    gcn = create_gnn_model('GCN', input_dim, num_classes)
    sage = create_gnn_model('GraphSAGE', input_dim, num_classes)
    gat = create_gnn_model('GAT', input_dim, num_classes)
    
    # Test forward pass
    x = torch.randn(10, input_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    
    print("GCN output:", gcn(x, edge_index).shape)
    print("GraphSAGE output:", sage(x, edge_index).shape)
    print("GAT output:", gat(x, edge_index).shape)
