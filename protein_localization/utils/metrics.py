"""
Evaluation metrics module for model performance assessment
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, y_true: List[int], y_pred: List[int],
                             class_names: List[str] = None) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class and weighted metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specificity for each class
        specificity_per_class = []
        for i in range(len(cm)):
            # True negatives: all correct predictions except for this class
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            # False positives: predictions for this class that were wrong
            fp = cm[:, i].sum() - cm[i, i]
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_per_class.append(specificity)
        
        # Average specificity
        specificity_macro = np.mean(specificity_per_class)
        
        # Create results dictionary
        metrics = {
            'accuracy': float(accuracy),
            'precision': {
                'macro': float(precision_macro),
                'weighted': float(precision_weighted),
                'per_class': [float(p) for p in precision_per_class]
            },
            'recall': {
                'macro': float(recall_macro),
                'weighted': float(recall_weighted),
                'per_class': [float(r) for r in recall_per_class]
            },
            'f1_score': {
                'macro': float(f1_macro),
                'weighted': float(f1_weighted),
                'per_class': [float(f) for f in f1_per_class]
            },
            'specificity': {
                'macro': float(specificity_macro),
                'per_class': [float(s) for s in specificity_per_class]
            },
            'confusion_matrix': cm.tolist(),
            'support': [int(s) for s in support]
        }
        
        # Add class names if provided
        if class_names:
            metrics['class_names'] = class_names
        
        self.metrics = metrics
        
        logger.info("Metrics calculated successfully")
        return metrics
    
    def print_metrics(self, metrics: Dict = None) -> None:
        """
        Pretty print all metrics
        
        Args:
            metrics: Metrics dictionary (uses stored if not provided)
        """
        if metrics is None:
            metrics = self.metrics
        
        if not metrics:
            logger.warning("No metrics available to print")
            return
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        # Overall metrics
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        
        # Macro-averaged metrics
        print(f"\nMacro-Averaged Metrics:")
        print(f"  Precision:   {metrics['precision']['macro']:.4f}")
        print(f"  Recall:      {metrics['recall']['macro']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']['macro']:.4f}")
        print(f"  Specificity: {metrics['specificity']['macro']:.4f}")
        
        # Weighted-averaged metrics
        print(f"\nWeighted-Averaged Metrics:")
        print(f"  Precision: {metrics['precision']['weighted']:.4f}")
        print(f"  Recall:    {metrics['recall']['weighted']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']['weighted']:.4f}")
        
        # Per-class metrics
        n_classes = len(metrics['precision']['per_class'])
        class_names = metrics.get('class_names', [f'Class {i}' for i in range(n_classes)])
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12} {'Support':<10}")
        print("-" * 75)
        
        for i in range(n_classes):
            print(f"{class_names[i]:<15} "
                  f"{metrics['precision']['per_class'][i]:<12.4f} "
                  f"{metrics['recall']['per_class'][i]:<12.4f} "
                  f"{metrics['f1_score']['per_class'][i]:<12.4f} "
                  f"{metrics['specificity']['per_class'][i]:<12.4f} "
                  f"{metrics['support'][i]:<10}")
        
        print("\n" + "="*60)
    
    def save_metrics(self, filepath: str, metrics: Dict = None) -> None:
        """
        Save metrics to file
        
        Args:
            filepath: Path to save metrics
            metrics: Metrics dictionary (uses stored if not provided)
        """
        import json
        
        if metrics is None:
            metrics = self.metrics
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {filepath}")
    
    def load_metrics(self, filepath: str) -> Dict:
        """
        Load metrics from file
        
        Args:
            filepath: Path to metrics file
            
        Returns:
            Metrics dictionary
        """
        import json
        
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        self.metrics = metrics
        logger.info(f"Metrics loaded from: {filepath}")
        
        return metrics


def calculate_colocalization_metrics(channel1: np.ndarray, 
                                     channel2: np.ndarray) -> Dict:
    """
    Calculate colocalization metrics (Pearson, Manders)
    
    Args:
        channel1: First channel intensity values
        channel2: Second channel intensity values
        
    Returns:
        Dictionary with colocalization metrics
    """
    logger.info("Calculating colocalization metrics...")
    
    # Flatten arrays
    c1 = channel1.flatten()
    c2 = channel2.flatten()
    
    # Pearson correlation coefficient
    pearson_r = np.corrcoef(c1, c2)[0, 1]
    
    # Manders' colocalization coefficients
    # M1: fraction of channel1 overlapping with channel2
    # M2: fraction of channel2 overlapping with channel1
    
    # Define threshold (mean intensity)
    threshold1 = np.mean(c1)
    threshold2 = np.mean(c2)
    
    # Pixels above threshold
    c1_above = c1 > threshold1
    c2_above = c2 > threshold2
    
    # Manders' M1 coefficient
    if c1_above.sum() > 0:
        m1 = np.sum(c1[c1_above & c2_above]) / np.sum(c1[c1_above])
    else:
        m1 = 0.0
    
    # Manders' M2 coefficient
    if c2_above.sum() > 0:
        m2 = np.sum(c2[c1_above & c2_above]) / np.sum(c2[c2_above])
    else:
        m2 = 0.0
    
    # Overlap coefficient
    overlap = np.sum(c1 * c2) / np.sqrt(np.sum(c1**2) * np.sum(c2**2))
    
    metrics = {
        'pearson_r': float(pearson_r),
        'manders_m1': float(m1),
        'manders_m2': float(m2),
        'overlap_coefficient': float(overlap)
    }
    
    logger.info("Colocalization metrics calculated")
    return metrics


def print_colocalization_metrics(metrics: Dict) -> None:
    """Print colocalization metrics"""
    print("\nColocalization Metrics:")
    print(f"  Pearson r:          {metrics['pearson_r']:.4f}")
    print(f"  Manders M1:         {metrics['manders_m1']:.4f}")
    print(f"  Manders M2:         {metrics['manders_m2']:.4f}")
    print(f"  Overlap Coefficient: {metrics['overlap_coefficient']:.4f}")
