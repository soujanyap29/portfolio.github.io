"""
CNN Model for protein localization prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNModel(nn.Module):
    """CNN for image-level classification"""
    
    def __init__(self, output_dim: int = 10, base_model: str = 'resnet50',
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize CNN model
        
        Args:
            output_dim: Number of output classes
            base_model: Base architecture (resnet50, resnet18, efficientnet)
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone weights
        """
        super(CNNModel, self).__init__()
        
        self.output_dim = output_dim
        self.base_model_name = base_model
        
        # Load base model
        if base_model == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif base_model == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif base_model == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            Logits (batch_size, output_dim)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output


class UNetSegmentation(nn.Module):
    """U-Net for semantic segmentation of protein localization"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        """
        Initialize U-Net
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
        """
        super(UNetSegmentation, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels, out_channels):
        """Convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, in_channels, height, width)
            
        Returns:
            Segmentation maps (batch_size, out_channels, height, width)
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        
        return out


def create_cnn_model(config: dict) -> nn.Module:
    """
    Factory function to create CNN model from config
    
    Args:
        config: Model configuration
        
    Returns:
        CNN model
    """
    architecture = config.get('architecture', {})
    cnn_config = config.get('cnn', {})
    labels_config = config.get('labels', {})
    
    output_dim = labels_config.get('num_classes', 10)
    base_model = cnn_config.get('base_model', 'resnet50')
    pretrained = cnn_config.get('pretrained', True)
    freeze_backbone = cnn_config.get('freeze_backbone', False)
    
    model = CNNModel(
        output_dim=output_dim,
        base_model=base_model,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    return model
