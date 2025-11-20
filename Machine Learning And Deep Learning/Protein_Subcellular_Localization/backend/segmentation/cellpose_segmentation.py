"""
Cellpose segmentation integration for neuronal images.
"""

import numpy as np
from cellpose import models, core
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """Segment neuronal images using Cellpose."""
    
    def __init__(self, 
                 model_type: str = 'cyto',
                 gpu: bool = False,
                 diameter: int = 30):
        """
        Initialize Cellpose segmenter.
        
        Args:
            model_type: Type of Cellpose model ('cyto', 'nuclei', 'cyto2')
            gpu: Whether to use GPU acceleration
            diameter: Expected cell diameter in pixels
        """
        self.model_type = model_type
        self.gpu = gpu and core.use_gpu()
        self.diameter = diameter
        
        # Initialize model
        try:
            self.model = models.Cellpose(model_type=model_type, gpu=self.gpu)
            logger.info(f"Initialized Cellpose model: {model_type}, GPU: {self.gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize Cellpose: {e}")
            raise
    
    def segment(self, 
                image: np.ndarray,
                channels: list = [0, 0],
                flow_threshold: float = 0.4,
                cellprob_threshold: float = 0.0) -> Tuple[np.ndarray, dict]:
        """
        Segment an image using Cellpose.
        
        Args:
            image: Input image (grayscale or RGB)
            channels: [cytoplasm_channel, nucleus_channel] - use [0,0] for grayscale
            flow_threshold: Flow error threshold (default 0.4)
            cellprob_threshold: Cell probability threshold (default 0.0)
            
        Returns:
            Tuple of (masks, additional_info)
        """
        try:
            # Run segmentation
            masks, flows, styles, diams = self.model.eval(
                image,
                diameter=self.diameter,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            
            # Package additional information
            info = {
                'flows': flows,
                'styles': styles,
                'diameters': diams,
                'n_cells': len(np.unique(masks)) - 1  # Exclude background
            }
            
            logger.info(f"Segmented {info['n_cells']} cells/regions")
            return masks, info
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
    
    def visualize_segmentation(self,
                               image: np.ndarray,
                               masks: np.ndarray,
                               save_path: Optional[str] = None,
                               dpi: int = 300) -> np.ndarray:
        """
        Visualize segmentation results.
        
        Args:
            image: Original image
            masks: Segmentation masks
            save_path: Optional path to save the visualization
            dpi: Resolution for saved image
            
        Returns:
            Visualization as numpy array
        """
        from cellpose import plot
        
        fig = plt.figure(figsize=(12, 6))
        
        # Original image
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Segmentation overlay
        ax2 = fig.add_subplot(1, 2, 2)
        if len(image.shape) == 2:
            # Grayscale - convert to RGB for overlay
            img_rgb = np.stack([image] * 3, axis=-1)
        else:
            img_rgb = image
        
        # Normalize if needed
        if img_rgb.max() > 1:
            img_rgb = img_rgb / img_rgb.max()
        
        # Create overlay
        overlay = plot.mask_overlay(img_rgb, masks)
        ax2.imshow(overlay)
        ax2.set_title(f'Segmentation ({len(np.unique(masks)) - 1} regions)')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved segmentation visualization to {save_path}")
        
        # Convert figure to array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis_array
    
    def extract_region_features(self, image: np.ndarray, masks: np.ndarray) -> dict:
        """
        Extract features for each segmented region.
        
        Args:
            image: Original image
            masks: Segmentation masks
            
        Returns:
            Dictionary of region features
        """
        from skimage import measure
        
        features = {}
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        
        for label in unique_labels:
            # Create binary mask for this region
            region_mask = (masks == label)
            
            # Extract region properties
            props = measure.regionprops(region_mask.astype(int), intensity_image=image)
            
            if props:
                prop = props[0]
                features[label] = {
                    'area': prop.area,
                    'perimeter': prop.perimeter,
                    'centroid': prop.centroid,
                    'eccentricity': prop.eccentricity,
                    'mean_intensity': prop.mean_intensity,
                    'max_intensity': prop.max_intensity,
                    'min_intensity': prop.min_intensity,
                    'bbox': prop.bbox
                }
        
        return features
    
    def save_masks(self, masks: np.ndarray, output_path: str):
        """
        Save segmentation masks.
        
        Args:
            masks: Segmentation masks
            output_path: Output file path
        """
        # Save as PNG
        plt.figure(figsize=(10, 10))
        plt.imshow(masks, cmap='nipy_spectral')
        plt.colorbar()
        plt.title('Segmentation Masks')
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved masks to {output_path}")


def batch_segment(segmenter: CellposeSegmenter,
                  images: list,
                  output_dir: str) -> list:
    """
    Segment a batch of images.
    
    Args:
        segmenter: CellposeSegmenter instance
        images: List of (filepath, original_image, preprocessed_image) tuples
        output_dir: Directory to save segmentation results
        
    Returns:
        List of segmentation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for filepath, original, processed in images:
        filename = Path(filepath).stem
        
        # Segment
        masks, info = segmenter.segment(original)
        
        # Visualize and save
        vis_path = output_path / f"{filename}_segment.png"
        segmenter.visualize_segmentation(original, masks, save_path=str(vis_path))
        
        # Extract features
        features = segmenter.extract_region_features(original, masks)
        
        results.append({
            'filepath': filepath,
            'filename': filename,
            'masks': masks,
            'info': info,
            'features': features,
            'vis_path': str(vis_path)
        })
        
        logger.info(f"Processed {filename}: {info['n_cells']} regions")
    
    return results


if __name__ == "__main__":
    # Example usage
    segmenter = CellposeSegmenter(model_type='cyto', gpu=False, diameter=30)
    
    # Create a test image
    test_image = np.random.rand(512, 512) * 255
    test_image = test_image.astype(np.uint8)
    
    # Segment
    masks, info = segmenter.segment(test_image)
    print(f"Detected {info['n_cells']} regions")
    
    # Visualize
    segmenter.visualize_segmentation(test_image, masks, save_path="/tmp/test_segmentation.png")
