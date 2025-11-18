"""
Visualization module for protein sub-cellular localization
Creates publication-ready visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
import networkx as nx
from typing import Dict, List, Tuple, Optional
import os
from skimage import color


class SegmentationVisualizer:
    """Visualize segmentation results"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_segmentation_overlay(self,
                                  image: np.ndarray,
                                  masks: np.ndarray,
                                  title: str = "Segmentation Overlay",
                                  filename: str = None,
                                  alpha: float = 0.5):
        """
        Plot segmentation overlay on original image
        
        Args:
            image: Original image
            masks: Segmentation masks
            title: Plot title
            filename: Output filename
            alpha: Transparency of overlay
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Normalize image for display
        img_display = self._normalize_for_display(image)
        
        # Original image
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation masks
        axes[1].imshow(masks, cmap='nipy_spectral')
        axes[1].set_title('Segmentation Masks')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_display, cmap='gray')
        axes[2].imshow(masks, cmap='nipy_spectral', alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_compartment_map(self,
                            masks: np.ndarray,
                            compartment_types: Dict[str, np.ndarray],
                            title: str = "Compartment Map",
                            filename: str = None):
        """
        Plot color-coded compartment maps
        
        Args:
            masks: Segmentation masks
            compartment_types: Dictionary of compartment types and their masks
            title: Plot title
            filename: Output filename
        """
        n_types = len(compartment_types)
        fig, axes = plt.subplots(1, n_types + 1, figsize=(5 * (n_types + 1), 5))
        
        if n_types == 0:
            axes = [axes]
        
        # All compartments
        axes[0].imshow(masks, cmap='nipy_spectral')
        axes[0].set_title('All Compartments')
        axes[0].axis('off')
        
        # Individual compartment types
        colors = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        for idx, (comp_type, comp_mask) in enumerate(compartment_types.items()):
            ax_idx = idx + 1
            if ax_idx < len(axes):
                axes[ax_idx].imshow(comp_mask, cmap=colors[idx % len(colors)])
                axes[ax_idx].set_title(comp_type.capitalize())
                axes[ax_idx].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_boundaries(self,
                       image: np.ndarray,
                       masks: np.ndarray,
                       title: str = "Cell Boundaries",
                       filename: str = None):
        """Plot cell boundaries"""
        from skimage.segmentation import find_boundaries
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Normalize image
        img_display = self._normalize_for_display(image)
        
        # Find boundaries
        boundaries = find_boundaries(masks, mode='thick')
        
        # Display
        ax.imshow(img_display, cmap='gray')
        ax.imshow(boundaries, cmap='Reds', alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    @staticmethod
    def _normalize_for_display(image: np.ndarray) -> np.ndarray:
        """Normalize image for display"""
        if len(image.shape) == 4:
            image = image[0, 0, :, :]
        elif len(image.shape) == 3:
            if image.shape[0] < 10:
                image = image[image.shape[0] // 2, :, :]
            else:
                image = image[:, :, 0] if image.shape[2] < 10 else image
        
        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            return image


class StatisticalPlotter:
    """Create statistical plots"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_grouped_bar(self,
                        data: Dict[str, List[float]],
                        title: str = "Grouped Bar Plot",
                        ylabel: str = "Value",
                        filename: str = None,
                        show_individuals: bool = True):
        """
        Plot grouped bar plot with mean ± SEM
        
        Args:
            data: Dictionary of {group_name: [values]}
            title: Plot title
            ylabel: Y-axis label
            filename: Output filename
            show_individuals: Whether to show individual datapoints
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = list(data.keys())
        means = [np.mean(data[g]) for g in groups]
        sems = [np.std(data[g]) / np.sqrt(len(data[g])) for g in groups]
        
        x = np.arange(len(groups))
        bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7, 
                     color=sns.color_palette("husl", len(groups)))
        
        # Add individual datapoints
        if show_individuals:
            for i, group in enumerate(groups):
                points = data[group]
                x_jitter = np.random.normal(i, 0.04, len(points))
                ax.scatter(x_jitter, points, alpha=0.6, s=20, color='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_box_and_violin(self,
                           data: Dict[str, List[float]],
                           title: str = "Distribution Comparison",
                           ylabel: str = "Value",
                           filename: str = None):
        """Create box plot and violin plot side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data for seaborn
        import pandas as pd
        df_data = []
        for group, values in data.items():
            for val in values:
                df_data.append({'Group': group, 'Value': val})
        df = pd.DataFrame(df_data)
        
        # Box plot
        sns.boxplot(data=df, x='Group', y='Value', ax=axes[0])
        axes[0].set_title('Box Plot', fontweight='bold')
        axes[0].set_ylabel(ylabel)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=df, x='Group', y='Value', ax=axes[1])
        axes[1].set_title('Violin Plot', fontweight='bold')
        axes[1].set_ylabel(ylabel)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_scatter(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    labels: Optional[List] = None,
                    title: str = "Scatter Plot",
                    xlabel: str = "X",
                    ylabel: str = "Y",
                    filename: str = None):
        """Create scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = sns.color_palette("husl", len(unique_labels))
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(x[mask], y[mask], c=[colors[i]], label=f'Class {label}', 
                          alpha=0.6, s=30)
            ax.legend()
        else:
            ax.scatter(x, y, alpha=0.6, s=30)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_hexbin(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   title: str = "Hexbin Plot",
                   xlabel: str = "Channel 1",
                   ylabel: str = "Channel 2",
                   filename: str = None):
        """Create hexbin plot for co-localization"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hexbin = ax.hexbin(x, y, gridsize=50, cmap='viridis', mincnt=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        
        cb = plt.colorbar(hexbin, ax=ax)
        cb.set_label('Count')
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()


class ColocalizationAnalyzer:
    """Analyze and visualize co-localization"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_manders_coefficients(self,
                                      channel1: np.ndarray,
                                      channel2: np.ndarray,
                                      threshold1: float = None,
                                      threshold2: float = None) -> Dict[str, float]:
        """
        Calculate Manders' colocalization coefficients
        
        Args:
            channel1: First channel image
            channel2: Second channel image
            threshold1: Threshold for channel 1 (auto if None)
            threshold2: Threshold for channel 2 (auto if None)
        
        Returns:
            Dictionary with M1 and M2 coefficients
        """
        # Auto-threshold if not provided
        if threshold1 is None:
            threshold1 = np.mean(channel1) + np.std(channel1)
        if threshold2 is None:
            threshold2 = np.mean(channel2) + np.std(channel2)
        
        # Apply thresholds
        mask1 = channel1 > threshold1
        mask2 = channel2 > threshold2
        
        # Calculate Manders coefficients
        M1 = np.sum(channel1[mask2]) / np.sum(channel1) if np.sum(channel1) > 0 else 0
        M2 = np.sum(channel2[mask1]) / np.sum(channel2) if np.sum(channel2) > 0 else 0
        
        return {'M1': M1, 'M2': M2}
    
    def calculate_pearson_correlation(self,
                                     channel1: np.ndarray,
                                     channel2: np.ndarray) -> float:
        """
        Calculate Pearson correlation coefficient
        
        Args:
            channel1: First channel image
            channel2: Second channel image
        
        Returns:
            Pearson correlation coefficient
        """
        flat1 = channel1.flatten()
        flat2 = channel2.flatten()
        
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        
        return correlation
    
    def plot_colocalization_metrics(self,
                                   manders: Dict[str, float],
                                   pearson: float,
                                   filename: str = None):
        """Plot colocalization metrics"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics = ['M1', 'M2', 'Pearson']
        values = [manders['M1'], manders['M2'], pearson]
        
        bars = ax.bar(metrics, values, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Co-localization Metrics', fontweight='bold')
        ax.set_ylim([-1, 1])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()


class IntensityProfiler:
    """Create intensity profile plots"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_intensity_profile(self,
                              image: np.ndarray,
                              line_coords: Tuple[Tuple[int, int], Tuple[int, int]],
                              title: str = "Intensity Profile",
                              filename: str = None):
        """
        Plot intensity profile along a line
        
        Args:
            image: Input image
            line_coords: ((y1, x1), (y2, x2))
            title: Plot title
            filename: Output filename
        """
        from skimage.measure import profile_line
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract profile
        profile = profile_line(image, line_coords[0], line_coords[1])
        
        # Plot image with line
        axes[0].imshow(image, cmap='gray')
        axes[0].plot([line_coords[0][1], line_coords[1][1]],
                    [line_coords[0][0], line_coords[1][0]],
                    'r-', linewidth=2)
        axes[0].set_title('Image with Profile Line')
        axes[0].axis('off')
        
        # Plot profile
        axes[1].plot(profile, linewidth=2)
        axes[1].set_xlabel('Distance (pixels)')
        axes[1].set_ylabel('Intensity')
        axes[1].set_title('Intensity Profile')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()


if __name__ == "__main__":
    print("Visualization module ready!")
    print("Available visualizers:")
    print("- SegmentationVisualizer")
    print("- StatisticalPlotter")
    print("- ColocalizationAnalyzer")
    print("- IntensityProfiler")
