"""
Training and evaluation module for protein localization models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import json


class ProteinLocalizationDataset(Dataset):
    """Dataset for protein localization"""
    
    def __init__(self, images, graphs, labels, transform=None):
        """
        Initialize dataset
        
        Args:
            images: List of image tensors
            graphs: List of graph data objects
            labels: List of labels
            transform: Optional transforms
        """
        self.images = images
        self.graphs = graphs
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        graph = self.graphs[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'graph': graph,
            'label': label
        }


class ModelTrainer:
    """Trainer for protein localization models"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Handle graph data
            if 'graph' in batch and batch['graph'] is not None:
                # Extract graph features
                try:
                    graphs = batch['graph']
                    node_features = graphs.x.to(self.device) if hasattr(graphs, 'x') else None
                    edge_index = graphs.edge_index.to(self.device) if hasattr(graphs, 'edge_index') else None
                    graph_batch = graphs.batch.to(self.device) if hasattr(graphs, 'batch') else None
                except:
                    node_features = None
                    edge_index = None
                    graph_batch = None
            else:
                node_features = None
                edge_index = None
                graph_batch = None
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if node_features is not None:
                outputs = self.model(images, node_features, edge_index, graph_batch)
            else:
                outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Handle graph data
                if 'graph' in batch and batch['graph'] is not None:
                    try:
                        graphs = batch['graph']
                        node_features = graphs.x.to(self.device) if hasattr(graphs, 'x') else None
                        edge_index = graphs.edge_index.to(self.device) if hasattr(graphs, 'edge_index') else None
                        graph_batch = graphs.batch.to(self.device) if hasattr(graphs, 'batch') else None
                    except:
                        node_features = None
                        edge_index = None
                        graph_batch = None
                else:
                    node_features = None
                    edge_index = None
                    graph_batch = None
                
                # Forward pass
                if node_features is not None:
                    outputs = self.model(images, node_features, edge_index, graph_batch)
                else:
                    outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics['accuracy'], metrics
    
    def calculate_metrics(self, y_true: List, y_pred: List) -> Dict:
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specificity
        specificity_per_class = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        specificity = np.mean(specificity_per_class)
        
        metrics = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'specificity': specificity * 100,
            'confusion_matrix': cm.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'specificity_per_class': specificity_per_class
        }
        
        return metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              early_stopping_patience: int = 10,
              save_dir: str = './models'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Precision: {metrics['precision']:.2f}% | Recall: {metrics['recall']:.2f}%")
            print(f"F1-Score: {metrics['f1_score']:.2f}% | Specificity: {metrics['specificity']:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print("Saved best model!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_model(os.path.join(save_dir, 'final_model.pth'))
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\nTraining complete!")
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']


def create_data_loaders(dataset: Dataset,
                       train_split: float = 0.8,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        dataset: Dataset to split
        train_split: Fraction for training
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        Train and validation data loaders
    """
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Training module ready!")
    print("Use ModelTrainer class to train models")
