"""
Visualization module for creating publication-ready scientific figures.
Includes image overlays, statistical plots, and graph visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import sem


class ProteinVisualization:
    """
    Creates scientific visualizations for protein localization analysis.
    """
    
    def __init__(self, output_dir: str, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualization.
        
        Args:
            output_dir: Directory to save figures
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-ready style
        plt.style.use('default')
        sns.set_palette("husl")
        self.dpi = 300
        
    def plot_segmentation_overlay(self, image: np.ndarray, masks: np.ndarray,
                                  filename: str, title: str = "Segmentation Overlay"):
        """
        Plot raw image with segmentation overlay.
        
        Args:
            image: Raw image array
            masks: Segmentation masks
            filename: Output filename
            title: Plot title
        """
        # Handle different dimensions
        if image.ndim == 4:
            img_2d = image[0, image.shape[1]//2, :, :]
        elif image.ndim == 3:
            img_2d = image[image.shape[0]//2, :, :]
        else:
            img_2d = image
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Raw image
        axes[0].imshow(img_2d, cmap='gray')
        axes[0].set_title('Raw Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(masks, cmap='nipy_spectral')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_2d, cmap='gray', alpha=0.7)
        axes[2].imshow(masks, cmap='nipy_spectral', alpha=0.3)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved segmentation overlay: {output_path}")
    
    def plot_compartment_masks(self, masks: np.ndarray, filename: str):
        """
        Plot compartment mask map.
        
        Args:
            masks: Segmentation masks
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(masks, cmap='tab20', interpolation='nearest')
        ax.set_title('Compartment Mask Map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Compartment ID', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved compartment masks: {output_path}")
    
    def plot_feature_distributions(self, features_list: List[Dict], 
                                   filename: str = "feature_distributions.png"):
        """
        Plot distribution of extracted features.
        
        Args:
            features_list: List of feature dictionaries
            filename: Output filename
        """
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Select numerical features
        numeric_cols = ['area', 'perimeter', 'eccentricity', 'solidity', 
                       'mean_intensity', 'distance_from_center']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(numeric_cols) == 0:
            print("No numeric features to plot")
            return
        
        # Create subplots
        n_features = len(numeric_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature distributions: {output_path}")
    
    def plot_grouped_bar(self, data_dict: Dict[str, List[float]], 
                        filename: str = "grouped_bar.png",
                        title: str = "Mean ± SEM"):
        """
        Create grouped bar plot with error bars.
        
        Args:
            data_dict: Dictionary of category -> values
            filename: Output filename
            title: Plot title
        """
        categories = list(data_dict.keys())
        means = [np.mean(values) for values in data_dict.values()]
        sems = [sem(values) if len(values) > 1 else 0 for values in data_dict.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Customize
        ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean ± SEM', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved grouped bar plot: {output_path}")
    
    def plot_box_violin(self, data_dict: Dict[str, List[float]], 
                       filename: str = "box_violin.png",
                       plot_type: str = 'box'):
        """
        Create box or violin plot.
        
        Args:
            data_dict: Dictionary of category -> values
            filename: Output filename
            plot_type: 'box' or 'violin'
        """
        # Prepare data
        data = []
        labels = []
        for category, values in data_dict.items():
            data.extend(values)
            labels.extend([category] * len(values))
        
        df = pd.DataFrame({'Category': labels, 'Value': data})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'box':
            sns.boxplot(data=df, x='Category', y='Value', ax=ax)
        else:
            sns.violinplot(data=df, x='Category', y='Value', ax=ax)
        
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{plot_type.title()} Plot', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {plot_type} plot: {output_path}")
    
    def plot_intensity_profile(self, features_list: List[Dict],
                              filename: str = "intensity_profile.png"):
        """
        Plot intensity vs distance from soma/center.
        
        Args:
            features_list: List of feature dictionaries
            filename: Output filename
        """
        df = pd.DataFrame(features_list)
        
        if 'distance_from_center' not in df.columns or 'mean_intensity' not in df.columns:
            print("Required columns not found for intensity profile")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(df['distance_from_center'], df['mean_intensity'], 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Fit trend line
        z = np.polyfit(df['distance_from_center'], df['mean_intensity'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(df['distance_from_center'].min(), 
                            df['distance_from_center'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
        
        ax.set_xlabel('Distance from Center (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Intensity', fontsize=12, fontweight='bold')
        ax.set_title('Intensity Profile vs Distance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved intensity profile: {output_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             filename: str = "confusion_matrix.png"):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix: {output_path}")
    
    def plot_graph(self, G: nx.Graph, filename: str = "graph_visualization.png",
                  node_size: int = 300, show_labels: bool = True):
        """
        Plot NetworkX graph with scientific style.
        
        Args:
            G: NetworkX graph
            filename: Output filename
            node_size: Size of nodes
            show_labels: Whether to show node labels
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax)
        
        # Draw nodes
        node_colors = [G.nodes[node].get('mean_intensity', 1.0) for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(
            G, pos, node_size=node_size, node_color=node_colors,
            cmap='viridis', alpha=0.9, edgecolors='black', 
            linewidths=2, ax=ax
        )
        
        # Draw labels if requested
        if show_labels and G.number_of_nodes() <= 50:
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                                   font_weight='bold', ax=ax)
        
        # Add colorbar
        plt.colorbar(nodes, ax=ax, label='Mean Intensity', fraction=0.046, pad=0.04)
        
        ax.set_title(f'Biological Graph\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved graph visualization: {output_path}")
    
    def plot_training_history(self, history: Dict, 
                             filename: str = "training_history.png"):
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Dictionary with train/val loss and accuracy
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
            axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
            axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training history: {output_path}")


def create_visualizations(processed_results: List[Dict],
                         graph_results: List[Dict],
                         training_results: Dict,
                         output_dir: str):
    """
    Create all visualizations for the pipeline.
    
    Args:
        processed_results: Preprocessing results
        graph_results: Graph construction results
        training_results: Training results
        output_dir: Output directory
    """
    viz = ProteinVisualization(output_dir)
    
    # Process each result
    for i, (proc_result, graph_result) in enumerate(zip(processed_results, graph_results)):
        base_name = Path(proc_result['file_name']).stem
        
        # Load image and masks (you'd need to reload or pass these)
        # For now, create dummy visualization
        print(f"\nCreating visualizations for {base_name}")
        
        # Plot graph
        if 'networkx_graph' in graph_result:
            G = graph_result['networkx_graph']
            viz.plot_graph(G, filename=f"{base_name}_graph.png")
        
        # Plot feature distributions
        if 'features' in proc_result:
            viz.plot_feature_distributions(
                proc_result['features'],
                filename=f"{base_name}_features.png"
            )
            
            viz.plot_intensity_profile(
                proc_result['features'],
                filename=f"{base_name}_intensity_profile.png"
            )
    
    # Plot training history
    if 'history' in training_results:
        viz.plot_training_history(
            training_results['history'],
            filename="training_history.png"
        )
    
    # Plot confusion matrix
    if 'metrics' in training_results and 'confusion_matrix' in training_results['metrics']:
        cm = np.array(training_results['metrics']['confusion_matrix'])
        class_names = [f"Class {i}" for i in range(len(cm))]
        viz.plot_confusion_matrix(cm, class_names)
    
    print("\n✓ All visualizations created")
