"""
Training script for protein localization models
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    """Dataset for graph data"""
    
    def __init__(self, graph_dir: str, label_dir: Optional[str] = None):
        """
        Initialize dataset
        
        Args:
            graph_dir: Directory containing graph pickle files
            label_dir: Optional directory containing labels
        """
        self.graph_dir = Path(graph_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        
        # Load graph files
        self.graph_files = sorted(list(self.graph_dir.glob("*_graph.pkl")))
        logger.info(f"Found {len(self.graph_files)} graph files")
        
        # Load labels if available
        self.labels = self._load_labels() if self.label_dir else None
    
    def _load_labels(self) -> Dict:
        """Load labels from file"""
        # Implement label loading logic
        # For now, return dummy labels
        return {f.stem.replace('_graph', ''): np.random.randint(0, 10) 
                for f in self.graph_files}
    
    def __len__(self):
        return len(self.graph_files)
    
    def __getitem__(self, idx):
        # Load graph data
        with open(self.graph_files[idx], 'rb') as f:
            graph_data = pickle.load(f)
        
        # Convert to PyTorch Geometric Data
        x = torch.tensor(graph_data['node_features'], dtype=torch.float)
        edge_index = torch.tensor(graph_data['edges'], dtype=torch.long)
        
        # Get label if available
        if self.labels:
            filename = self.graph_files[idx].stem.replace('_graph', '')
            y = torch.tensor(self.labels.get(filename, 0), dtype=torch.long)
        else:
            # Dummy label for unsupervised case
            y = torch.tensor(0, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        return data


class Trainer:
    """Trainer for protein localization models"""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training config
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 100)
        self.batch_size = training_config.get('batch_size', 16)
        self.lr = training_config.get('learning_rate', 0.001)
        
        # Setup optimizer
        optimizer_name = training_config.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=self.lr,
                weight_decay=training_config.get('weight_decay', 0.0001)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=training_config.get('weight_decay', 0.0001)
            )
        
        # Setup scheduler
        scheduler_name = training_config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        es_config = training_config.get('early_stopping', {})
        self.early_stopping_enabled = es_config.get('enabled', True)
        self.patience = es_config.get('patience', 15)
        self.min_delta = es_config.get('min_delta', 0.001)
        
        # Tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('training', {}).get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': correct / total
            })
        
        metrics = {
            'loss': total_loss / total,
            'acc': correct / total
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.num_graphs
        
        metrics = {
            'loss': total_loss / total,
            'acc': correct / total
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['acc']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])
                
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Acc: {val_metrics['acc']:.4f}")
                
                # Early stopping check
                if self.early_stopping_enabled:
                    if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_metrics['loss']
                        self.patience_counter = 0
                        self.save_checkpoint('best_model.pth')
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                            break
            
            # Step scheduler
            self.scheduler.step()
        
        logger.info("Training completed!")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs')) / 'models'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, output_dir / filename)
        logger.info(f"Saved checkpoint to {output_dir / filename}")


def collate_fn(data_list):
    """Custom collate function for PyG Data objects"""
    return Batch.from_data_list(data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train protein localization model")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing graph data")
    parser.add_argument("--model_type", type=str, default="gnn",
                       choices=['gnn', 'cnn'],
                       help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Create dataset
    dataset = GraphDataset(args.data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Create model
    if args.model_type == 'gnn':
        from models.gnn_model import create_gnn_model
        # Get input dimension from first sample
        sample = dataset[0]
        input_dim = sample.x.shape[1]
        model = create_gnn_model(config, input_dim)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not yet implemented for graph data")
    
    # Create trainer
    trainer = Trainer(model, config, device=config['training'].get('device', 'cuda'))
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    
    logger.info("Training completed successfully!")
