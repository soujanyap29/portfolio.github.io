"""
Graph Convolutional Neural Network for protein localization classification
Includes optional VGG-16 feature extraction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class GraphCNN(nn.Module):
    """Graph Convolutional Neural Network"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64):
        """
        Initialize Graph CNN
        
        Args:
            num_features: Number of input features per node
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
        """
        super(GraphCNN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        self.fc1 = nn.Linear(hidden_dim // 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Class logits
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x


class HybridCNN(nn.Module):
    """Hybrid CNN combining VGG-16 features and Graph CNN"""
    
    def __init__(self, num_graph_features: int, num_classes: int):
        """
        Initialize Hybrid CNN
        
        Args:
            num_graph_features: Number of graph node features
            num_classes: Number of output classes
        """
        super(HybridCNN, self).__init__()
        
        # VGG-16 feature extractor (simplified)
        self.vgg_features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Graph CNN
        self.gcn1 = GCNConv(num_graph_features, 64)
        self.gcn2 = GCNConv(64, 32)
        
        # Fusion layer
        self.fc1 = nn.Linear(128 * 4 * 4 + 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, image, graph_data):
        """
        Forward pass with both image and graph data
        
        Args:
            image: Input image tensor
            graph_data: PyTorch Geometric Data object
            
        Returns:
            Class logits
        """
        # VGG features
        vgg_out = self.vgg_features(image)
        vgg_out = vgg_out.view(vgg_out.size(0), -1)
        
        # Graph features
        x, edge_index = graph_data.x, graph_data.edge_index
        
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        
        graph_out = global_mean_pool(x, graph_data.batch)
        
        # Fusion
        combined = torch.cat([vgg_out, graph_out], dim=1)
        
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class ModelTrainer:
    """Train and evaluate Graph CNN models"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = None
        self.criterion = None
        
    def setup_training(self, learning_rate: float = 0.001):
        """
        Setup optimizer and loss function
        
        Args:
            learning_rate: Learning rate
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, data.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (accuracy, predictions, true_labels)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                pred = output.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        accuracy = correct / total
        return accuracy, all_preds, all_labels
    
    def train(self, train_loader, test_loader, num_epochs: int = 50):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'test_accuracy': []
        }
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_acc, _, _ = self.evaluate(test_loader)
            
            history['train_loss'].append(train_loss)
            history['test_accuracy'].append(test_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {train_loss:.4f}, "
                      f"Test Acc: {test_acc:.4f}")
        
        return history
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Graph CNN Model Implementation")
    print("=" * 50)
    
    # Test model creation
    num_features = 4
    num_classes = 5
    
    model = GraphCNN(num_features, num_classes)
    print(f"\nModel architecture:")
    print(model)
    
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dummy data
    print("\nTesting with dummy data...")
    x = torch.randn(10, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    output = model(data)
    print(f"Output shape: {output.shape}")
    print("Model test successful!")
