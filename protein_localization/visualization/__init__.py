"""
Visualization module for scientific-quality plots and figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-ready style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)


class ScientificVisualizer:
    """Create scientific publication-ready visualizations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.dpi = self.viz_config.get('dpi', 300)
        self.figsize = tuple(self.viz_config.get('figure_size', [10, 8]))
        
    def plot_segmentation_overlay(self, img: np.ndarray, masks: np.ndarray,
                                   output_path: str) -> None:
        """
        Plot original image with segmentation overlay
        
        Args:
            img: Original image
            masks: Segmentation masks
            output_path: Path to save figure
        """
        logger.info("Creating segmentation overlay...")
        
        # Handle multi-dimensional images
        if img.ndim > 2:
            img_2d = np.max(img, axis=tuple(range(img.ndim - 2)))
        else:
            img_2d = img
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_2d, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation masks
        axes[1].imshow(masks, cmap='nipy_spectral')
        axes[1].set_title('Segmentation Masks')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_2d, cmap='gray', alpha=0.7)
        axes[2].imshow(masks, cmap='nipy_spectral', alpha=0.3)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Segmentation overlay saved to: {output_path}")
    
    def plot_compartment_map(self, masks: np.ndarray, output_path: str) -> None:
        """
        Plot compartment mask map
        
        Args:
            masks: Segmentation masks
            output_path: Path to save figure
        """
        logger.info("Creating compartment map...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(masks, cmap='tab20')
        ax.set_title('Compartment Mask Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Compartment ID', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Compartment map saved to: {output_path}")
    
    def plot_grouped_bar(self, data: Dict[str, List[float]], 
                        output_path: str, ylabel: str = "Value") -> None:
        """
        Create grouped bar plot with mean ± SEM
        
        Args:
            data: Dictionary mapping groups to values
            output_path: Path to save figure
            ylabel: Y-axis label
        """
        logger.info("Creating grouped bar plot...")
        
        groups = list(data.keys())
        means = [np.mean(data[g]) for g in groups]
        sems = [np.std(data[g]) / np.sqrt(len(data[g])) for g in groups]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(groups))
        bars = ax.bar(x, means, yerr=sems, capsize=5, 
                      color=sns.color_palette("husl", len(groups)))
        
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_xlabel('Groups', fontsize=14, fontweight='bold')
        ax.set_title('Mean ± SEM', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Grouped bar plot saved to: {output_path}")
    
    def plot_box_violin(self, data: Dict[str, List[float]], 
                       output_path: str, ylabel: str = "Value") -> None:
        """
        Create box and violin plots
        
        Args:
            data: Dictionary mapping groups to values
            output_path: Path to save figure
            ylabel: Y-axis label
        """
        logger.info("Creating box/violin plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data for seaborn
        plot_data = []
        for group, values in data.items():
            for value in values:
                plot_data.append({'Group': group, 'Value': value})
        
        import pandas as pd
        df = pd.DataFrame(plot_data)
        
        # Box plot
        sns.boxplot(data=df, x='Group', y='Value', ax=axes[0])
        axes[0].set_title('Box Plot', fontsize=16, fontweight='bold')
        axes[0].set_ylabel(ylabel, fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Violin plot
        sns.violinplot(data=df, x='Group', y='Value', ax=axes[1])
        axes[1].set_title('Violin Plot', fontsize=16, fontweight='bold')
        axes[1].set_ylabel(ylabel, fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Box/violin plots saved to: {output_path}")
    
    def plot_colocalization_scatter(self, channel1: np.ndarray, 
                                    channel2: np.ndarray,
                                    output_path: str) -> None:
        """
        Create colocalization scatter plot
        
        Args:
            channel1: Intensity values from channel 1
            channel2: Intensity values from channel 2
            output_path: Path to save figure
        """
        logger.info("Creating colocalization scatter plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(channel1.flatten(), channel2.flatten())[0, 1]
        
        # Scatter plot with hexbin for density
        hexbin = ax.hexbin(channel1.flatten(), channel2.flatten(), 
                          gridsize=50, cmap='YlOrRd', mincnt=1)
        
        ax.set_xlabel('Channel 1 Intensity', fontsize=14, fontweight='bold')
        ax.set_ylabel('Channel 2 Intensity', fontsize=14, fontweight='bold')
        ax.set_title(f'Colocalization Analysis\nPearson r = {correlation:.3f}',
                    fontsize=16, fontweight='bold')
        
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        
        # Add diagonal line
        lims = [0, max(channel1.max(), channel2.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Colocalization scatter saved to: {output_path}")
    
    def plot_intensity_profile(self, distances: np.ndarray, 
                              intensities: np.ndarray,
                              output_path: str) -> None:
        """
        Plot intensity vs distance from soma
        
        Args:
            distances: Distances from soma
            intensities: Intensity values
            output_path: Path to save figure
        """
        logger.info("Creating intensity profile plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by distance
        sorted_idx = np.argsort(distances)
        distances_sorted = distances[sorted_idx]
        intensities_sorted = intensities[sorted_idx]
        
        # Plot with moving average
        window = max(1, len(distances) // 20)
        ma = np.convolve(intensities_sorted, np.ones(window)/window, mode='valid')
        ma_dist = distances_sorted[:len(ma)]
        
        ax.scatter(distances, intensities, alpha=0.3, s=10, label='Data points')
        ax.plot(ma_dist, ma, 'r-', linewidth=2, label='Moving average')
        
        ax.set_xlabel('Distance from Soma (pixels)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Intensity', fontsize=14, fontweight='bold')
        ax.set_title('Intensity Profile', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Intensity profile saved to: {output_path}")
    
    def plot_graph_visualization(self, G: nx.Graph, output_path: str,
                                node_labels: bool = True) -> None:
        """
        Create scientific graph visualization with rounded nodes
        
        Args:
            G: NetworkX graph
            output_path: Path to save figure
            node_labels: Whether to show node labels
        """
        logger.info("Creating graph visualization...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Use spring layout for nice positioning
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, ax=ax)
        
        # Draw nodes with custom styling
        node_sizes = [G.nodes[node].get('area', 100) for node in G.nodes()]
        node_colors = [G.nodes[node].get('mean_intensity', 0.5) for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='viridis',
            alpha=0.8,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )
        
        # Draw node labels if requested
        if node_labels and G.number_of_nodes() < 50:
            nx.draw_networkx_labels(G, pos, font_size=8, 
                                   font_weight='bold', ax=ax)
        
        ax.set_title('Biological Graph', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean Intensity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graph visualization saved to: {output_path}")
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                             class_names: List[str], output_path: str) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            output_path: Path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        logger.info("Creating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to: {output_path}")
