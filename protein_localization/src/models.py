"""
Model training module for protein localization classification.
Supports Graph-CNN, VGG-16, and hybrid architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import torchvision.models as models
import json


class GraphCNN(nn.Module):
    """
    Graph Convolutional Network for protein localization classification.
    """
    
    def __init__(self, num_node_features: int, num_classes: int, 
                 hidden_channels: int = 64, num_layers: int = 3):
        """
        Initialize Graph-CNN.
        
        Args:
            num_node_features: Number of input features per node
            num_classes: Number of output classes
            hidden_channels: Hidden layer dimensions
            num_layers: Number of GCN layers
        """
        super(GraphCNN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, data):
        """Forward pass."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class VGG16Classifier(nn.Module):
    """
    VGG-16 based classifier for image-based protein localization.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize VGG-16 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Use pretrained weights
        """
        super(VGG16Classifier, self).__init__()
        
        # Load pretrained VGG16
        self.vgg16 = models.vgg16(pretrained=pretrained)
        
        # Modify first layer to accept single channel
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Modify classifier
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        return self.vgg16(x)


class HybridModel(nn.Module):
    """
    Hybrid model combining CNN and Graph-CNN.
    """
    
    def __init__(self, num_node_features: int, num_classes: int,
                 cnn_channels: int = 64, graph_channels: int = 64):
        """
        Initialize hybrid model.
        
        Args:
            num_node_features: Number of input features per node
            num_classes: Number of output classes
            cnn_channels: CNN hidden channels
            graph_channels: Graph-CNN hidden channels
        """
        super(HybridModel, self).__init__()
        
        # CNN branch (simplified)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Graph branch
        self.graph_conv1 = GCNConv(num_node_features, graph_channels)
        self.graph_conv2 = GCNConv(graph_channels, graph_channels)
        
        # Fusion layer
        self.fc = nn.Linear(64 + graph_channels, num_classes)
        
    def forward(self, img, graph_data):
        """Forward pass with both image and graph."""
        # CNN branch
        cnn_features = self.cnn(img)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Graph branch
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.graph_conv1(x, edge_index))
        x = F.relu(self.graph_conv2(x, edge_index))
        graph_features = global_mean_pool(x, batch)
        
        # Fusion
        combined = torch.cat([cnn_features, graph_features], dim=1)
        output = self.fc(combined)
        
        return F.log_softmax(output, dim=1)


class ModelTrainer:
    """
    Handles model training and evaluation.
    """
    
    def __init__(self, model_type: str = 'graph_cnn', num_classes: int = 5,
                 device: str = None):
        """
        Initialize trainer.
        
        Args:
            model_type: 'graph_cnn', 'vgg16', or 'hybrid'
            num_classes: Number of output classes
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        print(f"Using device: {self.device}")
    
    def create_model(self, num_node_features: int):
        """Create model based on type."""
        if self.model_type == 'graph_cnn':
            self.model = GraphCNN(
                num_node_features=num_node_features,
                num_classes=self.num_classes,
                hidden_channels=64,
                num_layers=3
            )
        elif self.model_type == 'vgg16':
            self.model = VGG16Classifier(num_classes=self.num_classes)
        elif self.model_type == 'hybrid':
            self.model = HybridModel(
                num_node_features=num_node_features,
                num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        print(f"Created {self.model_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            output = self.model(data)
            loss = criterion(output, data.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, criterion):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = criterion(output, data.y)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs: int = 50, lr: float = 0.001):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✓ Training complete. Best validation accuracy: {best_val_acc:.4f}")
    
    def compute_metrics(self, y_true, y_pred) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        # Compute specificity for binary classification
        if self.num_classes == 2:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def save_model(self, path: str):
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str, num_node_features: int):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model_type = checkpoint['model_type']
        self.num_classes = checkpoint['num_classes']
        self.history = checkpoint.get('history', {})
        
        self.create_model(num_node_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


def train_model_pipeline(graph_data_list: List[Data], 
                        output_dir: str,
                        model_type: str = 'graph_cnn',
                        num_classes: int = 5,
                        epochs: int = 50,
                        batch_size: int = 32) -> Dict:
    """
    Main model training pipeline entry point.
    
    Args:
        graph_data_list: List of PyG Data objects
        output_dir: Directory to save models
        model_type: Type of model to train
        num_classes: Number of classes
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        Dictionary with training results
    """
    # Create dummy labels for demonstration (in real scenario, these come from data)
    for data in graph_data_list:
        data.y = torch.randint(0, num_classes, (1,))
    
    # Split data
    train_data, val_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize trainer
    num_node_features = graph_data_list[0].x.shape[1] if len(graph_data_list) > 0 else 10
    trainer = ModelTrainer(model_type=model_type, num_classes=num_classes)
    trainer.create_model(num_node_features)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Evaluate
    _, _, val_preds, val_labels = trainer.evaluate(val_loader, nn.NLLLoss())
    metrics = trainer.compute_metrics(val_labels, val_preds)
    
    # Save model
    model_path = Path(output_dir) / 'models' / f'{model_type}_model.pt'
    trainer.save_model(str(model_path))
    
    # Save metrics
    metrics_path = Path(output_dir) / 'models' / f'{model_type}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Model training complete")
    print(f"Metrics: {metrics}")
    
    return {
        'trainer': trainer,
        'metrics': metrics,
        'history': trainer.history
    }
