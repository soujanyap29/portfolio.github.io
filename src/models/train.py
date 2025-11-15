"""
Model Training and Testing Module
Includes Graph-CNN and hybrid CNN+GNN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import DataLoader
import torchvision.models as tv_models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import json


class GraphCNN(nn.Module):
    """Graph Convolutional Network for protein localization classification"""
    
    def __init__(self, num_features: int, num_classes: int, 
                 hidden_channels: int = 64, num_layers: int = 3):
        super(GraphCNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for enhanced feature learning"""
    
    def __init__(self, num_features: int, num_classes: int,
                 hidden_channels: int = 64, num_heads: int = 4, num_layers: int = 3):
        super(GraphAttentionNetwork, self).__init__()
        
        self.num_layers = num_layers
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_channels, heads=num_heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
        
        # Last layer with single head
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1))
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Final conv layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class HybridCNNGNN(nn.Module):
    """Hybrid CNN + Graph-CNN architecture"""
    
    def __init__(self, num_graph_features: int, num_classes: int):
        super(HybridCNNGNN, self).__init__()
        
        # VGG-16 for image features
        vgg = tv_models.vgg16(pretrained=True)
        self.image_features = nn.Sequential(*list(vgg.features.children()))
        
        # Freeze some VGG layers
        for param in list(self.image_features.parameters())[:-4]:
            param.requires_grad = False
        
        # Image feature adapter
        self.image_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Graph CNN for graph features
        self.graph_conv1 = GCNConv(num_graph_features, 64)
        self.graph_conv2 = GCNConv(64, 64)
        self.graph_conv3 = GCNConv(64, 64)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, graph_x, edge_index, batch=None):
        # Image branch
        img_feat = self.image_features(image)
        img_feat = self.image_adapter(img_feat)
        
        # Graph branch
        graph_feat = F.relu(self.graph_conv1(graph_x, edge_index))
        graph_feat = F.relu(self.graph_conv2(graph_feat, edge_index))
        graph_feat = self.graph_conv3(graph_feat, edge_index)
        
        if batch is not None:
            graph_feat = global_mean_pool(graph_feat, batch)
        else:
            graph_feat = graph_feat.mean(dim=0, keepdim=True)
        
        # Fusion
        combined = torch.cat([img_feat, graph_feat], dim=1)
        output = self.fusion(combined)
        
        return output


class ModelTrainer:
    """Train and evaluate models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        
        accuracy = correct / total
        return accuracy, np.array(all_preds), np.array(all_labels)
    
    def compute_metrics(self, y_true, y_pred, num_classes: int) -> Dict:
        """Compute all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Specificity (per class)
        specificity = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            if (tn + fp) > 0:
                spec = tn / (tn + fp)
            else:
                spec = 0
            specificity.append(spec)
        
        metrics['specificity'] = np.mean(specificity)
        metrics['specificity_per_class'] = specificity
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs: int = 100,
              learning_rate: float = 0.001, num_classes: int = 6):
        """Full training loop"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_acc, val_preds, val_labels = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        test_acc, test_preds, test_labels = self.evaluate(val_loader)
        metrics = self.compute_metrics(test_labels, test_preds, num_classes)
        
        return metrics
    
    def save_model(self, path: str):
        """Save trained model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        
        print(f"Model loaded from {path}")


def prepare_data_loaders(graphs_path: str, batch_size: int = 32, 
                        test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and test data loaders"""
    
    with open(graphs_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    graphs = data_dict['graphs']
    labels = data_dict['labels']
    
    # Add labels to graph data
    for graph, label in zip(graphs, labels):
        graph.y = torch.tensor([label], dtype=torch.long)
    
    # Train-test split
    train_graphs, test_graphs = train_test_split(graphs, test_size=test_size, random_state=42)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Preparing data...")
    train_loader, test_loader = prepare_data_loaders(
        "/mnt/d/5TH_SEM/CELLULAR/output/graphs.pkl"
    )
    
    # Get feature dimension from first batch
    sample_data = next(iter(train_loader))
    num_features = sample_data.x.shape[1]
    num_classes = 6  # Example: soma, dendrite, axon, nucleus, synapse, mitochondria
    
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = GraphCNN(num_features=num_features, num_classes=num_classes)
    trainer = ModelTrainer(model)
    
    # Train
    print("\nTraining model...")
    metrics = trainer.train(train_loader, test_loader, num_epochs=100)
    
    # Save model
    trainer.save_model("/mnt/d/5TH_SEM/CELLULAR/output/models/graph_cnn.pth")
    
    # Save metrics
    metrics_path = Path("/mnt/d/5TH_SEM/CELLULAR/output/models/metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    print("\nFinal Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
