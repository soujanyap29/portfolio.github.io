"""
Evaluation metrics for protein localization classification
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationMetrics:
    """Compute and visualize evaluation metrics"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       class_names: list = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Dictionary of metrics
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Specificity (manually computed)
        cm = confusion_matrix(y_true, y_pred)
        specificity_per_class = []
        
        for i in range(len(cm)):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        specificity = np.mean(specificity_per_class)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'specificity_per_class': specificity_per_class,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: list, 
                             output_path: str, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            output_path: Path to save plot
            title: Plot title
        """
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_probability_distribution(probabilities: np.ndarray, 
                                     class_names: list, 
                                     output_path: str,
                                     predicted_class: int = None):
        """
        Plot probability distribution for a prediction
        
        Args:
            probabilities: Probability array
            class_names: List of class names
            output_path: Path to save plot
            predicted_class: Index of predicted class (for highlighting)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#3498db' if i != predicted_class else '#e74c3c' 
                 for i in range(len(probabilities))]
        
        bars = ax.bar(range(len(probabilities)), probabilities, color=colors, alpha=0.8)
        
        ax.set_xlabel('Protein Localization Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title('Classification Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Add legend
        if predicted_class is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', alpha=0.8, label='Predicted Class'),
                Patch(facecolor='#3498db', alpha=0.8, label='Other Classes')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, float], 
                               output_path: str):
        """
        Plot comparison of evaluation metrics
        
        Args:
            metrics_dict: Dictionary of metrics
            output_path: Path to save plot
        """
        # Extract main metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [
            metrics_dict.get('accuracy', 0),
            metrics_dict.get('precision', 0),
            metrics_dict.get('recall', 0),
            metrics_dict.get('f1_score', 0),
            metrics_dict.get('specificity', 0)
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Evaluation Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                      class_names: list, output_path: str):
        """
        Generate and save detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            output_path: Path to save report
        """
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                      zero_division=0)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROTEIN LOCALIZATION CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Classification report saved to {output_path}")


def compute_colocalization_metrics(image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
    """
    Compute colocalization metrics between two channels
    
    Args:
        image1: First channel image
        image2: Second channel image
        
    Returns:
        Dictionary of colocalization metrics
    """
    # Flatten images
    img1_flat = image1.flatten()
    img2_flat = image2.flatten()
    
    # Pearson correlation coefficient
    pearson = np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    # Manders' colocalization coefficients
    # M1: fraction of image1 overlapping with image2
    # M2: fraction of image2 overlapping with image1
    
    threshold1 = np.mean(img1_flat) + np.std(img1_flat)
    threshold2 = np.mean(img2_flat) + np.std(img2_flat)
    
    coloc_mask1 = img1_flat > threshold1
    coloc_mask2 = img2_flat > threshold2
    
    M1 = np.sum(img1_flat[coloc_mask2]) / np.sum(img1_flat) if np.sum(img1_flat) > 0 else 0
    M2 = np.sum(img2_flat[coloc_mask1]) / np.sum(img2_flat) if np.sum(img2_flat) > 0 else 0
    
    metrics = {
        'pearson_coefficient': pearson,
        'manders_M1': M1,
        'manders_M2': M2
    }
    
    return metrics
