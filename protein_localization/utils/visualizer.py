"""
Visualization utilities for protein localization analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color, segmentation
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization tools for images, graphs, and results"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize visualizer"""
        self.config = config or self._default_config()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _default_config(self) -> Dict:
        """Default visualization configuration"""
        return {
            'dpi': 300,
            'figure_size': (12, 8),
            'colormap': 'viridis',
            'save_format': 'png'
        }
    
    def visualize_image(self, image: np.ndarray, title: str = "Image",
                       save_path: Optional[str] = None):
        """Visualize a single image"""
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        if image.ndim == 2:
            ax.imshow(image, cmap=self.config['colormap'])
        else:
            ax.imshow(image)
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()
    
    def visualize_preprocessing(self, original: np.ndarray, processed: np.ndarray,
                               title: str = "Preprocessing Comparison",
                               save_path: Optional[str] = None):
        """Compare original and preprocessed images"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(original, cmap=self.config['colormap'])
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(processed, cmap=self.config['colormap'])
        axes[1].set_title("Preprocessed", fontsize=12)
        axes[1].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()
    
    def visualize_segmentation(self, image: np.ndarray, segments: np.ndarray,
                              title: str = "Segmentation",
                              save_path: Optional[str] = None):
        """Visualize image segmentation"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        if image.ndim == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(segmentation.mark_boundaries(
            color.gray2rgb(image) if image.ndim == 2 else image, 
            segments
        ))
        axes[1].set_title("Boundaries", fontsize=12)
        axes[1].axis('off')
        
        # Colored segments
        axes[2].imshow(color.label2rgb(segments, image, kind='avg'))
        axes[2].set_title("Segments", fontsize=12)
        axes[2].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()
    
    def visualize_predictions(self, image: np.ndarray, predictions: np.ndarray,
                             class_names: List[str], 
                             title: str = "Predictions",
                             save_path: Optional[str] = None):
        """Visualize prediction results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        if image.ndim == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(predictions, cmap='tab10')
        axes[1].set_title("Predicted Localization", fontsize=12)
        axes[1].axis('off')
        
        # Add colorbar with class names
        cbar = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)
        cbar.set_label("Localization Class", fontsize=10)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training Accuracy', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(self, labels: np.ndarray, class_names: List[str],
                               title: str = "Class Distribution",
                               save_path: Optional[str] = None):
        """Plot class distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        unique, counts = np.unique(labels, return_counts=True)
        
        bars = ax.bar(range(len(unique)), counts, color=sns.color_palette("husl", len(unique)))
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       bbox_inches='tight', format=self.config['save_format'])
        else:
            plt.show()
        
        plt.close()


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay of mask on image
    
    Args:
        image: Base image
        mask: Segmentation mask
        alpha: Transparency
        
    Returns:
        Overlay image
    """
    if image.ndim == 2:
        image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = (image * 255).astype(np.uint8)
    
    # Create colored mask
    mask_colored = color.label2rgb(mask, bg_label=0)
    mask_colored = (mask_colored * 255).astype(np.uint8)
    
    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
    
    return overlay
