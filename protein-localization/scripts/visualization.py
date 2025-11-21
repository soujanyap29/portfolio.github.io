"""
Visualization Module
Generates publication-ready plots and analytics for protein localization analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import networkx as nx
from scipy import stats


class Visualizer:
    """
    Generate comprehensive visualizations for protein localization analysis.
    """
    
    def __init__(self, output_dir: str = "/mnt/d/5TH_SEM/CELLULAR/output", 
                 style: str = 'seaborn-v0_8-paper'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            sns.set_style("whitegrid")
        
        # Set color palette
        self.colors = sns.color_palette("husl", 10)
        
        # Configure matplotlib for publication quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def plot_segmentation_overlay(self, image: np.ndarray, masks: np.ndarray,
                                  title: str = "Segmentation", save_name: str = "segmentation"):
        """
        Plot original image with segmentation overlay.
        
        Args:
            image: Original image (can be 2D, 3D, or multi-channel)
            masks: Segmentation masks
            title: Plot title
            save_name: Base name for saved file
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare 2D image for visualization
        if image.ndim == 2:
            # Already 2D
            img_2d = image
        elif image.ndim == 3:
            # Could be (C, H, W) for multi-channel or (Z, H, W) for z-stack
            if image.shape[0] <= 4:
                # Likely multi-channel (C, H, W) - take max projection or first channel
                if image.shape[0] == 1:
                    img_2d = image[0]
                else:
                    # For multi-channel, create composite by averaging or taking first channel
                    img_2d = np.mean(image, axis=0)
            else:
                # Likely z-stack (Z, H, W) - take max projection
                img_2d = np.max(image, axis=0)
        elif image.ndim == 4:
            # (C, Z, H, W) or (Z, C, H, W) - take max projection
            if image.shape[0] <= 4:
                # Likely (C, Z, H, W)
                img_2d = np.max(np.mean(image, axis=0), axis=0)
            else:
                # Likely (Z, C, H, W)
                img_2d = np.max(np.mean(image, axis=1), axis=0)
        else:
            # Fallback: flatten to 2D
            img_2d = image.reshape(image.shape[-2], image.shape[-1])
        
        # Ensure img_2d is actually 2D
        if img_2d.ndim != 2:
            img_2d = img_2d.squeeze()
        
        # Original image
        axes[0].imshow(img_2d, cmap='gray')
        axes[0].set_title('Original Image')
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
        
        plt.suptitle(title)
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved segmentation visualization to {save_path}")
    
    def plot_compartment_masks(self, masks: np.ndarray, save_name: str = "compartments"):
        """
        Plot color-coded compartment mask map.
        
        Args:
            masks: Segmentation masks
            save_name: Base name for saved file
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(masks, cmap='tab20')
        plt.colorbar(label='Region ID')
        plt.title('Compartment Mask Map')
        plt.axis('off')
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved compartment masks to {save_path}")
    
    def plot_intensity_heatmap(self, image: np.ndarray, channel_names: Optional[List[str]] = None,
                               save_name: str = "intensity_heatmap"):
        """
        Plot intensity heatmaps per channel.
        
        Args:
            image: Image array
            channel_names: Names for each channel
            save_name: Base name for saved file
        """
        # Handle multi-channel images
        if image.ndim > 2:
            if image.shape[0] < 10:  # Likely channels
                n_channels = image.shape[0]
                fig, axes = plt.subplots(1, n_channels, figsize=(5*n_channels, 4))
                if n_channels == 1:
                    axes = [axes]
                
                for i, ax in enumerate(axes):
                    im = ax.imshow(image[i], cmap='hot')
                    channel_name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i+1}"
                    ax.set_title(channel_name)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, label='Intensity')
            else:
                # Use max projection
                img_2d = np.max(image, axis=0)
                plt.figure(figsize=(8, 6))
                plt.imshow(img_2d, cmap='hot')
                plt.colorbar(label='Intensity')
                plt.title('Intensity Heatmap (Max Projection)')
                plt.axis('off')
        else:
            plt.figure(figsize=(8, 6))
            plt.imshow(image, cmap='hot')
            plt.colorbar(label='Intensity')
            plt.title('Intensity Heatmap')
            plt.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved intensity heatmap to {save_path}")
    
    def plot_feature_distributions(self, features: pd.DataFrame, 
                                   feature_cols: Optional[List[str]] = None,
                                   save_name: str = "feature_distributions"):
        """
        Plot distributions of extracted features.
        
        Args:
            features: DataFrame with features
            feature_cols: List of features to plot
            save_name: Base name for saved file
        """
        if feature_cols is None:
            # Select numeric columns
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != 'label'][:12]  # Limit to 12
        
        n_features = len(feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes
        
        for i, feature in enumerate(feature_cols):
            if i < len(axes):
                axes[i].hist(features[feature].dropna(), bins=30, color=self.colors[i % len(self.colors)], alpha=0.7)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'Distribution of {feature}')
        
        # Hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved feature distributions to {save_path}")
    
    def plot_grouped_bar(self, data: pd.DataFrame, x_col: str, y_col: str,
                        group_col: Optional[str] = None, 
                        save_name: str = "grouped_bar"):
        """
        Plot grouped bar plots with mean ± SEM and individual data points.
        
        Args:
            data: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            group_col: Column for grouping
            save_name: Base name for saved file
        """
        plt.figure(figsize=(10, 6))
        
        if group_col:
            # Grouped bar plot
            grouped = data.groupby([x_col, group_col])[y_col].agg(['mean', 'sem'])
            grouped.unstack().plot(kind='bar', y='mean', yerr='sem', capsize=4, alpha=0.7)
            
            # Add individual points
            for i, (x_val, grp_val) in enumerate(data.groupby([x_col, group_col]).groups.keys()):
                subset = data[(data[x_col] == x_val) & (data[group_col] == grp_val)]
                x_pos = i
                plt.scatter([x_pos]*len(subset), subset[y_col], alpha=0.5, s=20, color='black')
        else:
            # Simple bar plot
            grouped = data.groupby(x_col)[y_col].agg(['mean', 'sem'])
            grouped.plot(kind='bar', y='mean', yerr='sem', capsize=4, alpha=0.7, legend=False)
            
            # Add individual points
            for i, x_val in enumerate(data[x_col].unique()):
                subset = data[data[x_col] == x_val]
                plt.scatter([i]*len(subset), subset[y_col], alpha=0.5, s=20, color='black')
        
        plt.xlabel(x_col)
        plt.ylabel(f'{y_col} (mean ± SEM)')
        plt.title(f'{y_col} by {x_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved grouped bar plot to {save_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None,
                             save_name: str = "confusion_matrix"):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            save_name: Base name for saved file
        """
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved confusion matrix to {save_path}")
    
    def plot_training_history(self, history: Dict, save_name: str = "training_history"):
        """
        Plot training history.
        
        Args:
            history: Dictionary with training metrics
            save_name: Base name for saved file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        if 'train_loss' in history:
            ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        if 'train_acc' in history:
            ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        if 'val_acc' in history:
            ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved training history to {save_path}")
    
    def plot_graph(self, G: nx.Graph, labels: Optional[Dict] = None,
                  node_color_attribute: Optional[str] = None,
                  save_name: str = "graph_visualization"):
        """
        Plot graph with rounded nodes and clear labeling.
        
        Args:
            G: NetworkX graph
            labels: Node labels dictionary
            node_color_attribute: Attribute to use for node coloring
            save_name: Base name for saved file
        """
        plt.figure(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Node colors
        if node_color_attribute and node_color_attribute in next(iter(G.nodes(data=True)))[1]:
            node_colors = [G.nodes[n].get(node_color_attribute, 0) for n in G.nodes()]
        else:
            node_colors = self.colors[0]
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=500, alpha=0.7, cmap='viridis')
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
        
        if labels:
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
        else:
            # Use node IDs as labels
            node_labels = {n: str(n) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
        
        plt.title(f'Graph Visualization ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
        plt.axis('off')
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved graph visualization to {save_path}")
    
    def plot_metrics_summary(self, metrics: Dict, save_name: str = "metrics_summary"):
        """
        Plot summary of evaluation metrics.
        
        Args:
            metrics: Dictionary with metrics
            save_name: Base name for saved file
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar plot of main metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('specificity', 0)
        ]
        
        axes[0].bar(metric_names, metric_values, color=self.colors[:5], alpha=0.7)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Metrics')
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                       cbar_kws={'label': 'Count'})
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            axes[1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved metrics summary to {save_path}")


if __name__ == "__main__":
    print("Visualization Module")
    print("Use the Visualizer class to generate publication-ready figures.")
