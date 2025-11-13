"""
Evaluation script for protein localization predictions
"""

import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for model predictions"""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.class_names = config.get('labels', {}).get('class_names', [])
        self.visualizer = Visualizer(config.get('visualization'))
    
    def evaluate(self, predictions: Dict[str, Dict], 
                ground_truth: Optional[Dict[str, int]] = None) -> Dict:
        """
        Evaluate predictions
        
        Args:
            predictions: Dictionary of predictions
            ground_truth: Optional ground truth labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if ground_truth is None:
            logger.warning("No ground truth provided, skipping evaluation")
            return self._generate_summary_stats(predictions)
        
        # Extract predictions and labels
        y_pred = []
        y_true = []
        
        for filename, pred in predictions.items():
            if filename in ground_truth:
                y_pred.append(pred['prediction'])
                y_true.append(ground_truth[filename])
        
        if not y_pred:
            logger.error("No matching predictions and ground truth found")
            return {}
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        
        # Generate visualizations
        self._generate_visualizations(y_true, y_pred, metrics)
        
        # Generate report
        self._generate_report(metrics)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, 
                target_names=self.class_names[:len(np.unique(y_true))],
                zero_division=0
            )
        }
        
        return metrics
    
    def _generate_summary_stats(self, predictions: Dict[str, Dict]) -> Dict:
        """Generate summary statistics without ground truth"""
        pred_values = [p['prediction'] for p in predictions.values()]
        
        stats = {
            'total_predictions': len(predictions),
            'unique_classes': len(np.unique(pred_values)),
            'class_distribution': dict(zip(*np.unique(pred_values, return_counts=True)))
        }
        
        return stats
    
    def _generate_visualizations(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray, metrics: Dict):
        """Generate visualization plots"""
        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs')) / 'visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        self.visualizer.plot_confusion_matrix(
            cm, 
            self.class_names[:cm.shape[0]],
            save_path=str(output_dir / 'confusion_matrix.png')
        )
        
        # Class distribution
        self.visualizer.plot_class_distribution(
            y_true,
            self.class_names,
            title="Ground Truth Distribution",
            save_path=str(output_dir / 'ground_truth_distribution.png')
        )
        
        self.visualizer.plot_class_distribution(
            y_pred,
            self.class_names,
            title="Prediction Distribution",
            save_path=str(output_dir / 'prediction_distribution.png')
        )
        
        logger.info(f"Saved visualizations to {output_dir}")
    
    def _generate_report(self, metrics: Dict):
        """Generate and save evaluation report"""
        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs')) / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / 'evaluation_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Protein Sub-Cellular Localization - Evaluation Report\n")
            f.write("="*70 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:         {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Precision (micro): {metrics['precision_micro']:.4f}\n")
            f.write(f"Recall (macro):    {metrics['recall_macro']:.4f}\n")
            f.write(f"Recall (micro):    {metrics['recall_micro']:.4f}\n")
            f.write(f"F1-Score (macro):  {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (micro):  {metrics['f1_micro']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-"*70 + "\n")
            f.write(metrics['classification_report'])
            f.write("\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-"*70 + "\n")
            f.write(str(metrics['confusion_matrix']))
            f.write("\n")
        
        logger.info(f"Saved evaluation report to {report_file}")
        
        # Print to console
        logger.info("\n" + "="*70)
        logger.info("Evaluation Results")
        logger.info("="*70)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_macro']:.4f}")
        logger.info("="*70)


def load_ground_truth(label_dir: str) -> Dict[str, int]:
    """
    Load ground truth labels
    
    Args:
        label_dir: Directory containing label files
        
    Returns:
        Dictionary mapping filenames to labels
    """
    # Implement label loading logic based on your data format
    # For now, return empty dict
    logger.warning("Ground truth loading not implemented, returning empty dict")
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate protein localization predictions")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--predictions_dir", type=str, required=True,
                       help="Directory containing prediction results")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                       help="Directory containing ground truth labels")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load predictions
    pred_file = Path(args.predictions_dir) / 'predictions.pkl'
    if not pred_file.exists():
        logger.error(f"Predictions file not found: {pred_file}")
        sys.exit(1)
    
    with open(pred_file, 'rb') as f:
        predictions = pickle.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load ground truth if available
    ground_truth = None
    if args.ground_truth_dir:
        ground_truth = load_ground_truth(args.ground_truth_dir)
        logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Evaluate
    metrics = evaluator.evaluate(predictions, ground_truth)
    
    logger.info("Evaluation completed successfully!")
