"""
Deep learning models for protein localization classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torchvision import models as vision_models
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphCNN(nn.Module):
    """Graph Convolutional Neural Network for protein localization"""
    
    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64):
        super(GraphCNN, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class VGG16Classifier(nn.Module):
    """VGG-16 based image classifier"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(VGG16Classifier, self).__init__()
        
        # Load pretrained VGG16
        self.vgg = vision_models.vgg16(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.vgg(x)


class HybridModel(nn.Module):
    """Hybrid model combining CNN and Graph CNN"""
    
    def __init__(self, num_node_features: int, num_classes: int, 
                 cnn_hidden: int = 512, graph_hidden: int = 64):
        super(HybridModel, self).__init__()
        
        # CNN branch (simplified VGG-like)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.cnn_fc = nn.Linear(256 * 7 * 7, cnn_hidden)
        
        # Graph CNN branch
        self.graph_conv1 = GCNConv(num_node_features, graph_hidden)
        self.graph_conv2 = GCNConv(graph_hidden, graph_hidden)
        
        # Fusion layer
        self.fusion = nn.Linear(cnn_hidden + graph_hidden, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, img, x_graph, edge_index, batch=None):
        # CNN branch
        cnn_out = self.cnn(img)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = F.relu(self.cnn_fc(cnn_out))
        cnn_out = self.dropout(cnn_out)
        
        # Graph CNN branch
        graph_out = self.graph_conv1(x_graph, edge_index)
        graph_out = F.relu(graph_out)
        graph_out = self.graph_conv2(graph_out, edge_index)
        graph_out = F.relu(graph_out)
        
        if batch is None:
            batch = torch.zeros(graph_out.size(0), dtype=torch.long)
        graph_out = global_mean_pool(graph_out, batch)
        
        # Fusion
        combined = torch.cat([cnn_out, graph_out], dim=1)
        out = self.fusion(combined)
        
        return F.log_softmax(out, dim=1)


class ModelTrainer:
    """Trainer class for protein localization models"""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.train_config = config.get('training', {})
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_config.get('learning_rate', 0.001)
        )
        self.criterion = nn.NLLLoss()
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total
        return accuracy, all_preds, all_labels
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from: {path}")
