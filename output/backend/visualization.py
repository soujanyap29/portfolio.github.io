"""
Scientific visualization module for publication-quality plots
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.patches import Rectangle
import cv2


class ScientificVisualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, dpi=300, style='seaborn-v0_8-paper'):
        self.dpi = dpi
        plt.style.use('default')  # Use default style as fallback
        sns.set_palette("husl")
    
    def plot_image_with_segmentation(self, original_image, segmentation, 
                                    save_path, title="Image Segmentation"):
        """
        Create overlay visualization of image and segmentation
        
        Args:
            original_image: Original TIFF image
            segmentation: Segmentation mask
            save_path: Output filepath
            title: Plot title
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(original_image.shape) == 2:
            axes[0].imshow(original_image, cmap='gray')
        else:
            axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(segmentation, cmap='nipy_spectral')
        axes[1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        if len(original_image.shape) == 2:
            display_img = np.stack([original_image] * 3, axis=-1)
        else:
            display_img = original_image.copy()
        
        # Normalize for display
        if display_img.max() <= 1.0:
            display_img = (display_img * 255).astype(np.uint8)
        
        # Create colored overlay
        from skimage.segmentation import mark_boundaries
        overlay = mark_boundaries(display_img / 255.0, segmentation, 
                                 color=(1, 0, 0), mode='thick')
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names, save_path, 
                             title="Confusion Matrix"):
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Output filepath
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_probability_distribution(self, probabilities, class_names, 
                                     save_path, title="Prediction Probabilities"):
        """
        Plot probability distribution as bar chart
        
        Args:
            probabilities: Probability values
            class_names: List of class names
            save_path: Output filepath
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", len(class_names))
        bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_xlabel('Localization Class', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, save_path, 
                               title="Model Performance Metrics"):
        """
        Plot comparison of different metrics
        
        Args:
            metrics_dict: Dictionary of metrics
            save_path: Output filepath
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [
            metrics_dict.get('accuracy', 0),
            metrics_dict.get('precision', 0),
            metrics_dict.get('recall', 0),
            metrics_dict.get('f1_score', 0),
            metrics_dict.get('specificity', 0)
        ]
        
        colors = sns.color_palette("Set2", len(metric_names))
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_graph_visualization(self, graph_data, save_path, 
                                title="Superpixel Graph Structure"):
        """
        Visualize graph structure from GNN
        
        Args:
            graph_data: PyTorch Geometric Data object
            save_path: Output filepath
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Convert to NetworkX graph
        G = nx.Graph()
        edge_index = graph_data.edge_index.cpu().numpy()
        edges = [(edge_index[0, i], edge_index[1, i]) 
                for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes with rounded appearance
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, alpha=0.9,
                              edgecolors='darkblue', linewidths=2, ax=ax)
        
        # Draw edges with smooth curves
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=1.5, alpha=0.6, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, 
                               font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_intensity_profile(self, image, save_path, 
                              title="Intensity Profile"):
        """
        Plot intensity profile across image
        
        Args:
            image: Input image
            save_path: Output filepath
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Display image
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title('Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot intensity profile
        if len(image.shape) == 2:
            profile = np.mean(image, axis=0)
        else:
            profile = np.mean(np.mean(image, axis=2), axis=0)
        
        axes[1].plot(profile, linewidth=2, color='darkblue')
        axes[1].fill_between(range(len(profile)), profile, alpha=0.3)
        axes[1].set_xlabel('Position', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Intensity', fontsize=12, fontweight='bold')
        axes[1].set_title('Intensity Profile', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_metrics(self, per_class_metrics, class_names, 
                              save_path, title="Per-Class Performance"):
        """
        Plot per-class performance metrics
        
        Args:
            per_class_metrics: Dictionary of per-class metrics
            class_names: List of class names
            save_path: Output filepath
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        precision = [per_class_metrics[cls]['precision'] for cls in class_names]
        recall = [per_class_metrics[cls]['recall'] for cls in class_names]
        f1 = [per_class_metrics[cls]['f1_score'] for cls in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, precision, width, label='Precision', 
                      alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall', 
                      alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                      alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Localization Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
