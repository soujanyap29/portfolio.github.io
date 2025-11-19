"""
VGG16-based CNN classifier for protein localization
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np


class VGG16Classifier(nn.Module):
    """VGG16-based classifier fine-tuned for protein localization"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(VGG16Classifier, self).__init__()
        
        # Load pretrained VGG16
        self.vgg16 = models.vgg16(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.vgg16.parameters())[:-8]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.vgg16(x)
    
    def extract_features(self, x):
        """Extract feature vector before final classification"""
        features = self.vgg16.features(x)
        features = self.vgg16.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.vgg16.classifier[:-1](features)
        return features


class CNNPredictor:
    """Wrapper for VGG16 predictions"""
    
    def __init__(self, num_classes=5, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGG16Classifier(num_classes=num_classes)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for CNN input
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed tensor
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    def predict(self, image):
        """
        Predict protein localization class
        
        Args:
            image: Input image
            
        Returns:
            predictions: Class probabilities
            predicted_class: Predicted class index
            confidence: Confidence score
        """
        img_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        return (probabilities.cpu().numpy()[0], 
                predicted_class.cpu().item(), 
                confidence.cpu().item())
    
    def extract_features(self, image):
        """
        Extract deep features for fusion with GNN
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        img_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            features = self.model.extract_features(img_tensor)
        
        return features.cpu().numpy()[0]
    
    def batch_predict(self, images):
        """
        Batch prediction for multiple images
        
        Args:
            images: List of images
            
        Returns:
            List of (probabilities, predicted_class, confidence) tuples
        """
        results = []
        for img in images:
            results.append(self.predict(img))
        return results
