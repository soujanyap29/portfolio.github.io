"""
Scientific visualization module for publication-ready plots
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import networkx as nx
from typing import List, Tuple, Dict
import cv2


class ScientificVisualizer:
    """Generate publication-quality visualizations"""
    
    def __init__(self, dpi: int = 300, style: str = 'seaborn-v0_8-paper'):
        """
        Initialize visualizer
        
        Args:
            dpi: Resolution for saved figures
            style: Matplotlib style
        """
        self.dpi = dpi
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set publication-quality defaults
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def plot_image_overlay(self, image: np.ndarray, mask: np.ndarray,
                          output_path: str, title: str = "Image Segmentation Overlay"):
        """
        Plot original image with segmentation overlay
        
        Args:
            image: Original image
            mask: Segmentation mask
            output_path: Output file path
            title: Plot title
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title('Original TIFF Image', fontweight='bold')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='nipy_spectral')
        axes[1].set_title('Segmentation Mask', fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image.copy()
        
        axes[2].imshow(image_rgb)
        axes[2].imshow(mask, alpha=0.4, cmap='nipy_spectral')
        axes[2].set_title('Overlay', fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_compartment_map(self, mask: np.ndarray, output_path: str,
                            title: str = "Cellular Compartment Map"):
        """
        Create colored compartment mask map
        
        Args:
            mask: Segmentation mask
            output_path: Output path
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(mask, cmap='tab20', interpolation='nearest')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Compartment ID', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_grouped_bars(self, data: Dict[str, List[float]], 
                         output_path: str,
                         ylabel: str = "Value", 
                         title: str = "Grouped Bar Plot with Error Bars"):
        """
        Create grouped bar plot with mean ± SEM and individual points
        
        Args:
            data: Dictionary of {group_name: [values]}
            output_path: Output path
            ylabel: Y-axis label
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        groups = list(data.keys())
        n_groups = len(groups)
        
        means = [np.mean(data[g]) for g in groups]
        sems = [np.std(data[g]) / np.sqrt(len(data[g])) for g in groups]
        
        x_pos = np.arange(n_groups)
        colors = plt.cm.Set3(np.linspace(0, 1, n_groups))
        
        # Plot bars
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5, alpha=0.8,
                     color=colors, edgecolor='black', linewidth=1.5)
        
        # Add individual data points
        for i, group in enumerate(groups):
            values = data[group]
            x_scatter = np.random.normal(i, 0.04, size=len(values))
            ax.scatter(x_scatter, values, color='black', alpha=0.6, s=30, zorder=3)
        
        ax.set_xlabel('Groups', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_violin_box(self, data: Dict[str, List[float]], 
                       output_path: str,
                       ylabel: str = "Value",
                       title: str = "Distribution Analysis"):
        """
        Create violin plot with box plot overlay
        
        Args:
            data: Dictionary of {group_name: [values]}
            output_path: Output path
            ylabel: Y-axis label
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        groups = list(data.keys())
        values = [data[g] for g in groups]
        
        # Violin plot
        parts = ax.violinplot(values, positions=range(len(groups)), 
                             showmeans=True, showmedians=True)
        
        # Color violin plots
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(groups)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        # Overlay box plot
        bp = ax.boxplot(values, positions=range(len(groups)), widths=0.3,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='white', alpha=0.5),
                       medianprops=dict(color='red', linewidth=2))
        
        ax.set_xlabel('Groups', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_colocalization_scatter(self, channel1: np.ndarray, channel2: np.ndarray,
                                    output_path: str,
                                    title: str = "Colocalization Analysis"):
        """
        Create scatter/hexbin plot for colocalization
        
        Args:
            channel1: First channel image
            channel2: Second channel image
            output_path: Output path
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Flatten images
        c1_flat = channel1.flatten()
        c2_flat = channel2.flatten()
        
        # Scatter plot
        axes[0].scatter(c1_flat, c2_flat, alpha=0.3, s=1)
        axes[0].set_xlabel('Channel 1 Intensity', fontweight='bold')
        axes[0].set_ylabel('Channel 2 Intensity', fontweight='bold')
        axes[0].set_title('Scatter Plot', fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(c1_flat, c2_flat, 1)
        p = np.poly1d(z)
        axes[0].plot(c1_flat, p(c1_flat), "r--", alpha=0.8, linewidth=2)
        
        # Hexbin plot
        hexbin = axes[1].hexbin(c1_flat, c2_flat, gridsize=50, cmap='YlOrRd', mincnt=1)
        axes[1].set_xlabel('Channel 1 Intensity', fontweight='bold')
        axes[1].set_ylabel('Channel 2 Intensity', fontweight='bold')
        axes[1].set_title('Hexbin Density Plot', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=axes[1])
        cbar.set_label('Counts', rotation=270, labelpad=20, fontweight='bold')
        
        # Compute and display Pearson correlation
        corr = np.corrcoef(c1_flat, c2_flat)[0, 1]
        fig.suptitle(f'{title}\nPearson Correlation: {corr:.3f}', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_intensity_profile(self, image: np.ndarray, line_coords: Tuple,
                              output_path: str,
                              title: str = "Intensity Profile"):
        """
        Plot intensity profile along a line
        
        Args:
            image: Image array
            line_coords: ((x1, y1), (x2, y2)) line coordinates
            output_path: Output path
            title: Plot title
        """
        from skimage.measure import profile_line
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Show image with line
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        
        (x1, y1), (x2, y2) = line_coords
        axes[0].plot([x1, x2], [y1, y2], 'r-', linewidth=2, label='Profile Line')
        axes[0].scatter([x1, x2], [y1, y2], c='red', s=100, zorder=5)
        axes[0].set_title('Image with Profile Line', fontweight='bold')
        axes[0].axis('off')
        axes[0].legend()
        
        # Plot intensity profile
        if len(image.shape) == 2:
            profile = profile_line(image, (y1, x1), (y2, x2))
        else:
            # Average across channels
            profile = profile_line(image.mean(axis=2), (y1, x1), (y2, x2))
        
        axes[1].plot(profile, linewidth=2, color='#2E86AB')
        axes[1].fill_between(range(len(profile)), profile, alpha=0.3, color='#2E86AB')
        axes[1].set_xlabel('Distance along line (pixels)', fontweight='bold')
        axes[1].set_ylabel('Intensity', fontweight='bold')
        axes[1].set_title('Intensity Profile', fontweight='bold')
        axes[1].grid(alpha=0.3, linestyle='--')
        
        plt.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_graph_visualization(self, adjacency: np.ndarray, features: np.ndarray,
                                output_path: str, node_labels: List[str] = None,
                                title: str = "Graph Network Visualization"):
        """
        Visualize graph network with smooth edges and rounded nodes
        
        Args:
            adjacency: Adjacency matrix
            features: Node features (for coloring)
            output_path: Output path
            node_labels: Optional node labels
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency)
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Node colors based on first feature
        node_colors = features[:, 0] if features.shape[1] > 0 else np.ones(len(G.nodes))
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                      node_size=500, cmap='viridis',
                                      alpha=0.9, edgecolors='black',
                                      linewidths=2, ax=ax)
        
        # Draw edges with varying transparency based on weight
        edges = nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.3,
                                      edge_color='gray', ax=ax,
                                      connectionstyle="arc3,rad=0.1")
        
        # Draw labels if provided
        if node_labels and len(node_labels) == len(G.nodes):
            labels = {i: node_labels[i] for i in G.nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=8,
                                   font_weight='bold', ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=node_colors.min(),
                                                     vmax=node_colors.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Node Feature Value', rotation=270, labelpad=20, fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
        ax.axis('off')
        ax.margins(0.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
