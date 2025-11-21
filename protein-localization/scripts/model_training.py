"""
Model Training Module
Implements Graph-CNN and hybrid CNN+GNN models for protein localization classification.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle


class GraphDataset(Dataset):
    """Dataset for graph-based data."""
    
    def __init__(self, graphs: List, labels: List):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network for node/graph classification.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 output_dim: int = 3, num_layers: int = 2, dropout: float = 0.5):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.pool = global_mean_pool
            self.fc = nn.Linear(hidden_dim, output_dim)
            
        except ImportError:
            print("Warning: PyTorch Geometric not available. Using fallback model.")
            self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, data):
        """Forward pass."""
        try:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            x = self.pool(x, batch)
            x = self.fc(x)
            
            return F.log_softmax(x, dim=1)
            
        except AttributeError:
            # Fallback for when PyG is not available
            x = data if isinstance(data, torch.Tensor) else data.x
            x = torch.mean(x, dim=0, keepdim=True)  # Simple pooling
            return F.log_softmax(self.fc(x), dim=1)


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting features from images (VGG-16 style).
    """
    
    def __init__(self, input_channels: int = 1, pretrained: bool = False):
        super(CNNFeatureExtractor, self).__init__()
        
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.fc = nn.Linear(512 * 7 * 7, 512)
    
    def forward(self, x):
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        return x


class HybridCNNGNN(nn.Module):
    """
    Hybrid model combining CNN and GNN.
    """
    
    def __init__(self, cnn_input_channels: int = 1, 
                 gnn_input_dim: int = 20, 
                 hidden_dim: int = 128,
                 output_dim: int = 3):
        super(HybridCNNGNN, self).__init__()
        
        self.cnn = CNNFeatureExtractor(cnn_input_channels)
        self.gnn = GraphConvolutionalNetwork(gnn_input_dim, hidden_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(512 + hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, image, graph_data):
        """Forward pass with both image and graph."""
        # Extract CNN features
        cnn_features = self.cnn(image)
        
        # Extract GNN features
        gnn_features = self.gnn(graph_data)
        
        # Fuse features
        fused = torch.cat([cnn_features, gnn_features], dim=1)
        fused = F.relu(self.fusion(fused))
        
        # Classification
        output = self.classifier(fused)
        return F.log_softmax(output, dim=1)


class ModelTrainer:
    """
    Trainer for graph-based models.
    """
    
    def __init__(self, model_type: str = 'gcn', device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('gcn', 'cnn', 'hybrid')
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_type = model_type
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def create_model(self, input_dim: int, output_dim: int, **kwargs):
        """
        Create model based on type.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Number of output classes
            **kwargs: Additional model arguments
        """
        if self.model_type == 'gcn':
            self.model = GraphConvolutionalNetwork(
                input_dim, 
                hidden_dim=kwargs.get('hidden_dim', 64),
                output_dim=output_dim,
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.5)
            )
        elif self.model_type == 'cnn':
            self.model = CNNFeatureExtractor(
                input_channels=kwargs.get('input_channels', 1)
            )
        elif self.model_type == 'hybrid':
            self.model = HybridCNNGNN(
                cnn_input_channels=kwargs.get('input_channels', 1),
                gnn_input_dim=input_dim,
                hidden_dim=kwargs.get('hidden_dim', 128),
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        print(f"Created {self.model_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            data, labels = batch
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, criterion):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, labels)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_data, val_data, epochs: int = 100, 
              lr: float = 0.001, batch_size: int = 32):
        """
        Train model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
        """
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\nTraining for {epochs} epochs...")
        best_val_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, self.optimizer)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint('best_model.pt')
        
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    
    def compute_metrics(self, test_loader) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                data, labels = batch
                data = data.to(self.device)
                
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        # Compute specificity per class
        specificity_per_class = []
        for i in range(cm.shape[0]):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(np.mean(specificity_per_class)),
            'confusion_matrix': cm.tolist(),
        }
        
        return metrics
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'model_type': self.model_type,
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {})


if __name__ == "__main__":
    print("Model Training Module")
    print("This module provides GCN, CNN, and Hybrid models for protein localization.")
    print("Import and use the ModelTrainer class for training.")
