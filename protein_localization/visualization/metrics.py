"""
Evaluation metrics and confusion matrix visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, roc_curve, auc)
from typing import Dict, List
import os


class MetricsEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_all_metrics(self, y_true: List, y_pred: List, 
                             class_names: List[str] = None) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        
        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specificity for each class
        specificity = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity.append(spec * 100)
        
        metrics = {
            'accuracy': accuracy,
            'precision_per_class': (precision * 100).tolist(),
            'recall_per_class': (recall * 100).tolist(),
            'f1_per_class': (f1 * 100).tolist(),
            'specificity_per_class': specificity,
            'support': support.tolist(),
            'precision_avg': precision_avg * 100,
            'recall_avg': recall_avg * 100,
            'f1_avg': f1_avg * 100,
            'specificity_avg': np.mean(specificity),
            'confusion_matrix': cm.tolist(),
            'num_classes': len(np.unique(y_true))
        }
        
        if class_names:
            metrics['class_names'] = class_names
        
        return metrics
    
    def plot_confusion_matrix(self,
                             cm: np.ndarray,
                             class_names: List[str] = None,
                             title: str = "Confusion Matrix",
                             filename: str = None,
                             normalize: bool = False):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            filename: Output filename
            normalize: Whether to normalize
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax)
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_metrics_comparison(self,
                               metrics: Dict,
                               title: str = "Performance Metrics",
                               filename: str = None):
        """
        Plot bar chart comparing different metrics
        
        Args:
            metrics: Dictionary with metrics
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        values = [
            metrics['accuracy'],
            metrics['precision_avg'],
            metrics['recall_avg'],
            metrics['f1_avg'],
            metrics['specificity_avg']
        ]
        
        colors = sns.color_palette("husl", len(metric_names))
        bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_per_class_metrics(self,
                               metrics: Dict,
                               title: str = "Per-Class Metrics",
                               filename: str = None):
        """
        Plot per-class performance metrics
        
        Args:
            metrics: Dictionary with metrics
            title: Plot title
            filename: Output filename
        """
        import pandas as pd
        
        num_classes = metrics['num_classes']
        class_names = metrics.get('class_names', [f'Class {i}' for i in range(num_classes)])
        
        # Prepare data
        data = {
            'Class': class_names,
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1-Score': metrics['f1_per_class'],
            'Specificity': metrics['specificity_per_class']
        }
        
        df = pd.DataFrame(data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(class_names))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, df['Precision'], width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, df['Recall'], width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, df['Specificity'], width, label='Specificity', alpha=0.8)
        
        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Score (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def generate_report(self,
                       metrics: Dict,
                       filename: str = "evaluation_report.txt"):
        """
        Generate text report of metrics
        
        Args:
            metrics: Dictionary with metrics
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PROTEIN LOCALIZATION MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:     {metrics['accuracy']:.2f}%\n")
            f.write(f"Precision:    {metrics['precision_avg']:.2f}%\n")
            f.write(f"Recall:       {metrics['recall_avg']:.2f}%\n")
            f.write(f"F1-Score:     {metrics['f1_avg']:.2f}%\n")
            f.write(f"Specificity:  {metrics['specificity_avg']:.2f}%\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            
            class_names = metrics.get('class_names', 
                                     [f'Class {i}' for i in range(metrics['num_classes'])])
            
            for i, class_name in enumerate(class_names):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision:    {metrics['precision_per_class'][i]:.2f}%\n")
                f.write(f"  Recall:       {metrics['recall_per_class'][i]:.2f}%\n")
                f.write(f"  F1-Score:     {metrics['f1_per_class'][i]:.2f}%\n")
                f.write(f"  Specificity:  {metrics['specificity_per_class'][i]:.2f}%\n")
                f.write(f"  Support:      {metrics['support'][i]}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Saved report: {filepath}")


if __name__ == "__main__":
    print("Metrics evaluator module ready!")
    
    # Test with dummy data
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 2, 3, 2]
    
    evaluator = MetricsEvaluator()
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, 
                                             class_names=['Soma', 'Dendrite', 'Axon', 'Puncta'])
    
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1-Score: {metrics['f1_avg']:.2f}%")
