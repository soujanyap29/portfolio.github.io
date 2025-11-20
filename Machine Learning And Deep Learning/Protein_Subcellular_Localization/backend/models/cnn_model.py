"""
VGG16 CNN model for protein localization classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinLocalizationCNN(nn.Module):
    """VGG16-based CNN for protein localization."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True, freeze_layers: int = 15):
        """
        Initialize VGG16 model.
        
        Args:
            num_classes: Number of localization classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_layers: Number of initial layers to freeze
        """
        super(ProteinLocalizationCNN, self).__init__()
        
        # Load pretrained VGG16
        self.vgg16 = models.vgg16(pretrained=pretrained)
        
        # Freeze early layers
        if freeze_layers > 0:
            for i, param in enumerate(self.vgg16.features.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
        
        # Modify classifier for our task
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)
        
        logger.info(f"Initialized VGG16 with {num_classes} classes, {freeze_layers} frozen layers")
    
    def forward(self, x):
        """Forward pass."""
        return self.vgg16(x)
    
    def get_features(self, x):
        """Extract features before final classification."""
        # Get features from VGG16
        features = self.vgg16.features(x)
        features = self.vgg16.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Pass through most of classifier except final layer
        for i in range(6):
            features = self.vgg16.classifier[i](features)
        
        return features


class ProteinDataset(Dataset):
    """Dataset for protein localization images."""
    
    def __init__(self, images: List[np.ndarray], labels: List[int], transform=None):
        """
        Initialize dataset.
        
        Args:
            images: List of preprocessed images
            labels: List of class labels
            transform: Optional transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL for transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label


class CNNTrainer:
    """Trainer for CNN model."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.0001):
        """
        Initialize trainer.
        
        Args:
            model: The CNN model
            device: Device to train on
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              save_path: Optional[str] = None) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_path: Path to save best model
            
        Returns:
            Training history
        """
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
    
    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict class for a single image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.model.eval()
        
        # Convert to tensor
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities


def load_model(model_path: str, num_classes: int = 5, device: str = 'cpu') -> ProteinLocalizationCNN:
    """
    Load a trained model.
    
    Args:
        model_path: Path to saved model weights
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = ProteinLocalizationCNN(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model


if __name__ == "__main__":
    # Example usage
    model = ProteinLocalizationCNN(num_classes=5, pretrained=True, freeze_layers=15)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
