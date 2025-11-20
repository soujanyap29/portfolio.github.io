"""
High-resolution visualization utilities (≥300 DPI).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import networkx as nx
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


class Visualizer:
    """High-resolution scientific visualizations."""
    
    def __init__(self, output_dir: str, dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: Resolution in dots per inch
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray,
                             class_names: List[str],
                             filename: str,
                             title: str = "Confusion Matrix"):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def plot_probability_distribution(self,
                                     probabilities: np.ndarray,
                                     class_names: List[str],
                                     filename: str,
                                     title: str = "Class Probability Distribution"):
        """Plot probability distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(class_names))
        bars = ax.bar(x, probabilities, color='steelblue', alpha=0.8)
        
        # Highlight the predicted class
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('crimson')
        
        ax.set_xlabel('Class', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved probability distribution to {save_path}")
    
    def plot_metrics_comparison(self,
                               metrics_dict: Dict[str, Dict],
                               filename: str,
                               title: str = "Model Metrics Comparison"):
        """Plot metrics comparison across models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = list(metrics_dict.keys())
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        titles = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        
        for idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [metrics_dict[model][metric] for model in model_names]
            
            bars = ax.bar(model_names, values, color=['steelblue', 'orange', 'green'][:len(model_names)], alpha=0.8)
            
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(metric_title, fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics comparison to {save_path}")
    
    def plot_training_history(self,
                             history: Dict,
                             filename: str,
                             title: str = "Training History"):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training history to {save_path}")
    
    def plot_graph(self,
                   graph: nx.Graph,
                   node_features: Optional[Dict] = None,
                   filename: str = "graph_visualization.png",
                   title: str = "Superpixel Graph"):
        """Plot graph with smooth curved edges."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = ['steelblue' for _ in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos,
                              node_color=node_colors,
                              node_size=300,
                              alpha=0.8,
                              ax=ax)
        
        # Draw edges with curved paths
        nx.draw_networkx_edges(graph, pos,
                              edge_color='gray',
                              width=1.5,
                              alpha=0.5,
                              connectionstyle="arc3,rad=0.1",
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos,
                               font_size=8,
                               font_color='white',
                               font_weight='bold',
                               ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved graph visualization to {save_path}")
    
    def plot_box_plot(self,
                     data_dict: Dict[str, List[float]],
                     filename: str,
                     title: str = "Performance Distribution",
                     ylabel: str = "Score"):
        """Plot box plot for comparing distributions."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [data_dict[key] for key in data_dict.keys()]
        labels = list(data_dict.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        colors = ['steelblue', 'orange', 'green']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved box plot to {save_path}")
    
    def plot_image_overlay(self,
                          original: np.ndarray,
                          overlay: np.ndarray,
                          filename: str,
                          title: str = "Image with Overlay"):
        """Plot original image with overlay."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # With overlay
        if len(original.shape) == 2:
            axes[1].imshow(original, cmap='gray')
        else:
            axes[1].imshow(original)
        
        axes[1].imshow(overlay, alpha=0.5, cmap='jet')
        axes[1].set_title('With Overlay', fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved image overlay to {save_path}")


if __name__ == "__main__":
    # Example usage
    visualizer = Visualizer(output_dir="/tmp/visualizations")
    
    # Test confusion matrix
    cm = np.array([[10, 2, 1], [3, 15, 2], [1, 1, 18]])
    class_names = ['Class A', 'Class B', 'Class C']
    visualizer.plot_confusion_matrix(cm, class_names, "test_cm.png")
    
    # Test probability distribution
    probs = np.array([0.1, 0.6, 0.2, 0.05, 0.05])
    class_names = ['Soma', 'Dendrites', 'Axon', 'Synapses', 'Nucleus']
    visualizer.plot_probability_distribution(probs, class_names, "test_probs.png")
    
    print("Visualizations created successfully!")
