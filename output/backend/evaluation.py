"""
Model fusion and evaluation metrics
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
import json


class ModelFusion:
    """Combine predictions from CNN and GNN models"""
    
    def __init__(self, cnn_weight=0.6, gnn_weight=0.4, method='weighted'):
        """
        Initialize fusion module
        
        Args:
            cnn_weight: Weight for CNN predictions
            gnn_weight: Weight for GNN predictions
            method: 'weighted', 'max', or 'voting'
        """
        self.cnn_weight = cnn_weight
        self.gnn_weight = gnn_weight
        self.method = method
        
        # Normalize weights
        total = cnn_weight + gnn_weight
        self.cnn_weight /= total
        self.gnn_weight /= total
    
    def fuse_predictions(self, cnn_probs, gnn_probs):
        """
        Fuse predictions from both models
        
        Args:
            cnn_probs: Probability distribution from CNN
            gnn_probs: Probability distribution from GNN
            
        Returns:
            fused_probs: Combined probability distribution
            predicted_class: Final predicted class
            confidence: Confidence score
        """
        if self.method == 'weighted':
            # Weighted average of probabilities
            fused_probs = (self.cnn_weight * cnn_probs + 
                          self.gnn_weight * gnn_probs)
        
        elif self.method == 'max':
            # Maximum probability at each position
            fused_probs = np.maximum(cnn_probs, gnn_probs)
        
        elif self.method == 'voting':
            # Hard voting
            cnn_class = np.argmax(cnn_probs)
            gnn_class = np.argmax(gnn_probs)
            
            if cnn_class == gnn_class:
                predicted_class = cnn_class
                fused_probs = (cnn_probs + gnn_probs) / 2
            else:
                # Use weighted probabilities
                fused_probs = (self.cnn_weight * cnn_probs + 
                              self.gnn_weight * gnn_probs)
        
        # Get final prediction
        predicted_class = np.argmax(fused_probs)
        confidence = fused_probs[predicted_class]
        
        return fused_probs, predicted_class, confidence


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def __init__(self, class_names):
        self.class_names = class_names
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, 
                                               average='weighted', 
                                               zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, 
                                        average='weighted', 
                                        zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, 
                                       average='weighted', 
                                       zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, 
                                             average=None, 
                                             zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, 
                                       average=None, 
                                       zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, 
                               average=None, 
                               zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Specificity calculation
        specificity_per_class = []
        for i in range(len(self.class_names)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        metrics['specificity'] = np.mean(specificity_per_class)
        metrics['specificity_per_class'] = {
            class_name: float(spec) 
            for class_name, spec in zip(self.class_names, specificity_per_class)
        }
        
        return metrics
    
    def generate_report(self, y_true, y_pred, probabilities=None):
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            probabilities: Prediction probabilities (optional)
            
        Returns:
            Report dictionary
        """
        report = {}
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        report['metrics'] = metrics
        
        # Classification report
        class_report = classification_report(y_true, y_pred, 
                                            target_names=self.class_names,
                                            output_dict=True,
                                            zero_division=0)
        report['classification_report'] = class_report
        
        # Add probabilities if provided
        if probabilities is not None:
            report['probabilities'] = probabilities.tolist()
        
        return report
    
    def save_report(self, report, filepath):
        """
        Save report to JSON file
        
        Args:
            report: Report dictionary
            filepath: Output filepath
        """
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def calculate_single_prediction_metrics(self, probabilities, predicted_class):
        """
        Calculate metrics for single prediction
        
        Args:
            probabilities: Probability distribution
            predicted_class: Predicted class index
            
        Returns:
            Dictionary with prediction info
        """
        info = {
            'predicted_class': self.class_names[predicted_class],
            'predicted_class_index': int(predicted_class),
            'confidence': float(probabilities[predicted_class]),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }
        }
        
        return info
