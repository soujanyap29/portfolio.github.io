"""
Model fusion and evaluation utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFusion:
    """Fuse predictions from CNN and GNN models."""
    
    def __init__(self, method: str = 'weighted_average', cnn_weight: float = 0.6, gnn_weight: float = 0.4):
        """
        Initialize model fusion.
        
        Args:
            method: Fusion method ('weighted_average', 'voting', 'max')
            cnn_weight: Weight for CNN predictions
            gnn_weight: Weight for GNN predictions
        """
        self.method = method
        self.cnn_weight = cnn_weight
        self.gnn_weight = gnn_weight
        
        # Normalize weights
        total_weight = cnn_weight + gnn_weight
        self.cnn_weight = cnn_weight / total_weight
        self.gnn_weight = gnn_weight / total_weight
        
        logger.info(f"Initialized fusion: {method}, CNN weight: {self.cnn_weight:.2f}, GNN weight: {self.gnn_weight:.2f}")
    
    def fuse(self, cnn_probs: np.ndarray, gnn_probs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Fuse predictions from both models.
        
        Args:
            cnn_probs: CNN probability distribution
            gnn_probs: GNN probability distribution
            
        Returns:
            Tuple of (fused_class, fused_probabilities)
        """
        if self.method == 'weighted_average':
            fused_probs = self.cnn_weight * cnn_probs + self.gnn_weight * gnn_probs
            fused_class = np.argmax(fused_probs)
        
        elif self.method == 'voting':
            cnn_class = np.argmax(cnn_probs)
            gnn_class = np.argmax(gnn_probs)
            
            if cnn_class == gnn_class:
                fused_class = cnn_class
            else:
                # Use weighted voting
                if cnn_probs[cnn_class] * self.cnn_weight > gnn_probs[gnn_class] * self.gnn_weight:
                    fused_class = cnn_class
                else:
                    fused_class = gnn_class
            
            fused_probs = self.cnn_weight * cnn_probs + self.gnn_weight * gnn_probs
        
        elif self.method == 'max':
            fused_probs = np.maximum(cnn_probs, gnn_probs)
            fused_class = np.argmax(fused_probs)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        
        return fused_class, fused_probs


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int], class_names: List[str] = None) -> Dict:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Specificity (per class)
        specificity_per_class = []
        for i in range(len(cm)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        metrics['specificity_per_class'] = specificity_per_class
        metrics['specificity_macro'] = np.mean(specificity_per_class)
        
        # Classification report
        if class_names:
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict, model_name: str = "Model"):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Metrics dictionary
            model_name: Name of the model
        """
        print(f"\n{'=' * 50}")
        print(f"{model_name} Performance Metrics")
        print(f"{'=' * 50}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-score (macro):   {metrics['f1_macro']:.4f}")
        print(f"  Specificity (macro):{metrics['specificity_macro']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, (prec, rec, f1, spec) in enumerate(zip(
            metrics['precision_per_class'],
            metrics['recall_per_class'],
            metrics['f1_per_class'],
            metrics['specificity_per_class']
        )):
            print(f"  Class {i}:")
            print(f"    Precision:   {prec:.4f}")
            print(f"    Recall:      {rec:.4f}")
            print(f"    F1-score:    {f1:.4f}")
            print(f"    Specificity: {spec:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        for row in cm:
            print(f"  {row}")
    
    @staticmethod
    def compare_models(metrics_dict: Dict[str, Dict]) -> Dict:
        """
        Compare metrics across multiple models.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            
        Returns:
            Comparison summary
        """
        comparison = {
            'model_names': list(metrics_dict.keys()),
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'specificities': []
        }
        
        for model_name in comparison['model_names']:
            metrics = metrics_dict[model_name]
            comparison['accuracies'].append(metrics['accuracy'])
            comparison['precisions'].append(metrics['precision_macro'])
            comparison['recalls'].append(metrics['recall_macro'])
            comparison['f1_scores'].append(metrics['f1_macro'])
            comparison['specificities'].append(metrics['specificity_macro'])
        
        # Find best model for each metric
        comparison['best_accuracy'] = comparison['model_names'][np.argmax(comparison['accuracies'])]
        comparison['best_precision'] = comparison['model_names'][np.argmax(comparison['precisions'])]
        comparison['best_recall'] = comparison['model_names'][np.argmax(comparison['recalls'])]
        comparison['best_f1'] = comparison['model_names'][np.argmax(comparison['f1_scores'])]
        comparison['best_specificity'] = comparison['model_names'][np.argmax(comparison['specificities'])]
        
        return comparison
    
    @staticmethod
    def print_comparison(comparison: Dict):
        """Print model comparison."""
        print(f"\n{'=' * 50}")
        print("Model Comparison")
        print(f"{'=' * 50}")
        
        for i, model_name in enumerate(comparison['model_names']):
            print(f"\n{model_name}:")
            print(f"  Accuracy:    {comparison['accuracies'][i]:.4f}")
            print(f"  Precision:   {comparison['precisions'][i]:.4f}")
            print(f"  Recall:      {comparison['recalls'][i]:.4f}")
            print(f"  F1-score:    {comparison['f1_scores'][i]:.4f}")
            print(f"  Specificity: {comparison['specificities'][i]:.4f}")
        
        print(f"\nBest Models:")
        print(f"  Accuracy:    {comparison['best_accuracy']}")
        print(f"  Precision:   {comparison['best_precision']}")
        print(f"  Recall:      {comparison['best_recall']}")
        print(f"  F1-score:    {comparison['best_f1']}")
        print(f"  Specificity: {comparison['best_specificity']}")


if __name__ == "__main__":
    # Example usage
    
    # Simulate predictions
    cnn_probs = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
    gnn_probs = np.array([0.1, 0.4, 0.2, 0.2, 0.1])
    
    # Test fusion
    fusion = ModelFusion(method='weighted_average', cnn_weight=0.6, gnn_weight=0.4)
    fused_class, fused_probs = fusion.fuse(cnn_probs, gnn_probs)
    
    print(f"CNN probs: {cnn_probs}")
    print(f"GNN probs: {gnn_probs}")
    print(f"Fused class: {fused_class}")
    print(f"Fused probs: {fused_probs}")
    
    # Test metrics
    y_true = [0, 1, 2, 1, 0, 2, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 2, 2, 0]
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(y_true, y_pred)
    calculator.print_metrics(metrics, "Test Model")
