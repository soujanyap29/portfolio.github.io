"""
Visualization Module
Publication-ready plots for protein localization analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.stats import sem


class VisualizationSuite:
    """Complete visualization suite for protein localization"""
    
    def __init__(self, output_dir: str = "/mnt/d/5TH_SEM/CELLULAR/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
    def plot_image_overlay(self, image: np.ndarray, masks: np.ndarray,
                          title: str = "Image Overlay", save_name: str = "image_overlay.png"):
        """Plot raw TIFF with segmentation outlines"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Raw image
        if image.ndim == 3 and image.shape[-1] <= 3:
            axes[0].imshow(image)
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Raw Image')
        axes[0].axis('off')
        
        # Segmentation masks
        axes[1].imshow(masks, cmap='nipy_spectral')
        axes[1].set_title('Segmentation Masks')
        axes[1].axis('off')
        
        # Overlay
        if image.ndim == 3 and image.shape[-1] <= 3:
            axes[2].imshow(image)
        else:
            axes[2].imshow(image, cmap='gray')
        
        # Draw outlines
        from skimage import segmentation
        boundaries = segmentation.find_boundaries(masks, mode='outer')
        axes[2].imshow(boundaries, cmap='Reds', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_compartment_mask_map(self, masks: np.ndarray, compartment_labels: Optional[np.ndarray] = None,
                                  save_name: str = "compartment_map.png"):
        """Colored segmentation map per region"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if compartment_labels is not None:
            # Color by compartment type
            colored_masks = np.zeros_like(masks)
            for i, label in enumerate(compartment_labels):
                colored_masks[masks == i+1] = label
            ax.imshow(colored_masks, cmap='tab10')
        else:
            ax.imshow(masks, cmap='nipy_spectral')
        
        ax.set_title('Compartment Mask Map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        if compartment_labels is not None:
            cbar.set_label('Compartment Type', fontsize=12)
        else:
            cbar.set_label('Region ID', fontsize=12)
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_grouped_bar_with_points(self, data: Dict[str, List[float]],
                                     ylabel: str = "Intensity",
                                     title: str = "Grouped Bar Plot",
                                     save_name: str = "grouped_bar.png"):
        """Grouped bar plot with mean ± SEM and individual data points"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(data.keys())
        x = np.arange(len(categories))
        width = 0.6
        
        means = [np.mean(data[cat]) for cat in categories]
        sems = [sem(data[cat]) if len(data[cat]) > 1 else 0 for cat in categories]
        
        # Bar plot
        bars = ax.bar(x, means, width, yerr=sems, capsize=5,
                     alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
        
        # Individual points with jitter
        for i, cat in enumerate(categories):
            points = data[cat]
            jitter = np.random.normal(0, 0.05, len(points))
            ax.scatter(x[i] + jitter, points, alpha=0.6, s=50, color='darkblue')
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines('right').set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_box_violin(self, data: Dict[str, List[float]],
                       ylabel: str = "Distribution",
                       title: str = "Box and Violin Plot",
                       save_name: str = "box_violin.png"):
        """Combined box and violin plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
        
        # Box plot
        df.boxplot(ax=ax1, patch_artist=True)
        ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax1.set_title('Box Plot', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        df_melted = df.melt(var_name='Compartment', value_name='Value')
        sns.violinplot(data=df_melted, x='Compartment', y='Value', ax=ax2)
        ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax2.set_title('Violin Plot', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_colocalization_scatter(self, channel_a: np.ndarray, channel_b: np.ndarray,
                                   title: str = "Colocalization Analysis",
                                   save_name: str = "colocalization.png"):
        """Scatter and hexbin plots for colocalization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Flatten arrays
        a_flat = channel_a.flatten()
        b_flat = channel_b.flatten()
        
        # Scatter plot
        ax1.scatter(a_flat, b_flat, alpha=0.3, s=1, c='steelblue')
        ax1.set_xlabel('Channel A Intensity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Channel B Intensity', fontsize=12, fontweight='bold')
        ax1.set_title('Scatter Plot', fontsize=12, fontweight='bold')
        
        # Add correlation
        corr = np.corrcoef(a_flat, b_flat)[0, 1]
        ax1.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hexbin plot
        hb = ax2.hexbin(a_flat, b_flat, gridsize=50, cmap='YlOrRd', mincnt=1)
        ax2.set_xlabel('Channel A Intensity', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Channel B Intensity', fontsize=12, fontweight='bold')
        ax2.set_title('Hexbin Density Plot', fontsize=12, fontweight='bold')
        plt.colorbar(hb, ax=ax2, label='Counts')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_colocalization_metrics(self, metrics: Dict[str, float],
                                    save_name: str = "coloc_metrics.png"):
        """Plot colocalization metrics (Manders, Pearson)"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.barh(metric_names, metric_values, color=colors[:len(metrics)])
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title('Colocalization Metrics', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_intensity_profile(self, distances: np.ndarray, intensities: np.ndarray,
                              title: str = "Intensity Profile",
                              save_name: str = "intensity_profile.png"):
        """Intensity vs distance from soma"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(distances, intensities, alpha=0.5, s=20, c='steelblue')
        
        # Add trend line
        z = np.polyfit(distances, intensities, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(distances.min(), distances.max(), 100)
        ax.plot(x_trend, p(x_trend), "r-", linewidth=2, label='Trend')
        
        ax.set_xlabel('Distance from Soma (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Intensity', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_graph_visualization(self, G: nx.Graph, pos: Optional[Dict] = None,
                                 node_colors: Optional[List] = None,
                                 title: str = "Biological Graph",
                                 save_name: str = "graph_viz.png"):
        """Graph visualization with rounded nodes and labels"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        if pos is None:
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges first
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax)
        
        # Draw nodes with rounded style
        if node_colors is None:
            node_colors = ['steelblue'] * len(G.nodes())
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=500, alpha=0.9, ax=ax,
                              edgecolors='black', linewidths=2)
        
        # Draw node labels
        labels = {node: G.nodes[node].get('label', node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8,
                               font_color='white', font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             save_name: str = "confusion_matrix.png"):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_history(self, history: Dict[str, List[float]],
                             save_name: str = "training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_acc' in history:
            ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_plots(self, data_dict: Dict):
        """Generate all publication-ready plots"""
        print("Generating visualizations...")
        
        # Example: Generate available plots based on data
        if 'image' in data_dict and 'masks' in data_dict:
            self.plot_image_overlay(data_dict['image'], data_dict['masks'])
            self.plot_compartment_mask_map(data_dict['masks'])
        
        if 'compartment_data' in data_dict:
            self.plot_grouped_bar_with_points(data_dict['compartment_data'])
            self.plot_box_violin(data_dict['compartment_data'])
        
        if 'channel_a' in data_dict and 'channel_b' in data_dict:
            self.plot_colocalization_scatter(data_dict['channel_a'], data_dict['channel_b'])
        
        if 'graph' in data_dict:
            self.plot_graph_visualization(data_dict['graph'])
        
        if 'confusion_matrix' in data_dict:
            self.plot_confusion_matrix(data_dict['confusion_matrix'],
                                      data_dict.get('class_names', []))
        
        if 'history' in data_dict:
            self.plot_training_history(data_dict['history'])
        
        print(f"All plots saved to {self.output_dir}")


if __name__ == "__main__":
    # Example usage
    viz = VisualizationSuite()
    
    # Generate example plots with dummy data
    dummy_image = np.random.rand(512, 512)
    dummy_masks = np.random.randint(0, 10, (512, 512))
    
    viz.plot_image_overlay(dummy_image, dummy_masks)
    viz.plot_compartment_mask_map(dummy_masks)
    
    print("Example visualizations generated")
