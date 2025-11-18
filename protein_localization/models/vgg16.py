"""
VGG-16 model for image-based protein localization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class VGG16Classifier(nn.Module):
    """VGG-16 based classifier for protein localization"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 pretrained: bool = True,
                 in_channels: int = 3,
                 freeze_features: bool = False):
        """
        Initialize VGG-16 classifier
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
            freeze_features: Whether to freeze feature extractor
        """
        super(VGG16Classifier, self).__init__()
        
        # Load pre-trained VGG16
        if pretrained:
            self.vgg = models.vgg16(weights='IMAGENET1K_V1')
        else:
            self.vgg = models.vgg16(weights=None)
        
        # Modify first conv layer if in_channels != 3
        if in_channels != 3:
            first_conv = self.vgg.features[0]
            self.vgg.features[0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding
            )
            
            if pretrained and in_channels == 1:
                # Initialize with average of RGB weights
                with torch.no_grad():
                    self.vgg.features[0].weight[:, 0, :, :] = first_conv.weight.mean(dim=1)
        
        # Freeze feature extractor if requested
        if freeze_features:
            for param in self.vgg.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.vgg(x)
    
    def extract_features(self, x):
        """Extract feature representations"""
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class VGG16FeatureExtractor(nn.Module):
    """VGG-16 feature extractor (without classification head)"""
    
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        """Initialize VGG-16 feature extractor"""
        super(VGG16FeatureExtractor, self).__init__()
        
        if pretrained:
            vgg = models.vgg16(weights='IMAGENET1K_V1')
        else:
            vgg = models.vgg16(weights=None)
        
        # Modify first conv layer if needed
        if in_channels != 3:
            first_conv = vgg.features[0]
            vgg.features[0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding
            )
            
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    vgg.features[0].weight[:, 0, :, :] = first_conv.weight.mean(dim=1)
        
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        
    def forward(self, x):
        """Extract features"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CustomCNN(nn.Module):
    """Custom CNN for microscopy images"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 in_channels: int = 1,
                 dropout: float = 0.5):
        """
        Initialize custom CNN
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            dropout: Dropout probability
        """
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet50Classifier(nn.Module):
    """ResNet-50 based classifier"""
    
    def __init__(self,
                 num_classes: int = 10,
                 pretrained: bool = True,
                 in_channels: int = 3):
        """Initialize ResNet-50 classifier"""
        super(ResNet50Classifier, self).__init__()
        
        if pretrained:
            self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Modify first conv layer if needed
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace classification head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        return self.resnet(x)


def create_cnn_model(model_type: str = 'vgg16', **kwargs):
    """
    Factory function to create CNN models
    
    Args:
        model_type: Type of model ('vgg16', 'resnet50', 'custom')
        **kwargs: Model parameters
    
    Returns:
        Model instance
    """
    if model_type.lower() == 'vgg16':
        return VGG16Classifier(**kwargs)
    elif model_type.lower() == 'resnet50':
        return ResNet50Classifier(**kwargs)
    elif model_type.lower() == 'custom':
        return CustomCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing CNN models...")
    
    # Test VGG-16
    model = VGG16Classifier(num_classes=5, in_channels=1, pretrained=False)
    x = torch.randn(2, 1, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"VGG-16 output shape: {output.shape}")
    
    # Test Custom CNN
    model2 = CustomCNN(num_classes=5, in_channels=1)
    model2.eval()
    
    with torch.no_grad():
        output2 = model2(x)
    
    print(f"Custom CNN output shape: {output2.shape}")
    print("CNN models test complete!")
