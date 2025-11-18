"""
Preprocessing module for TIFF image loading and segmentation
"""

import numpy as np
import tifffile
from cellpose import models, io
from skimage import measure
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TIFFProcessor:
    """Process 4D TIFF images for protein localization analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.seg_config = config.get('segmentation', {})
        
    def load_tiff(self, file_path: str) -> np.ndarray:
        """
        Load TIFF file (supports 2D, 3D, and 4D)
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            numpy array of image data
        """
        logger.info(f"Loading TIFF from: {file_path}")
        try:
            img = tifffile.imread(file_path)
            logger.info(f"Image shape: {img.shape}, dtype: {img.dtype}")
            return img
        except Exception as e:
            logger.error(f"Error loading TIFF: {e}")
            raise
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img
    
    def segment_cellpose(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment image using Cellpose
        
        Args:
            img: Input image (2D, 3D, or 4D)
            
        Returns:
            masks: Segmentation masks
            flows: Flow fields from Cellpose
        """
        logger.info("Running Cellpose segmentation...")
        
        # Initialize Cellpose model
        model = models.Cellpose(
            gpu=False,  # Set to True if GPU available
            model_type=self.seg_config.get('model_type', 'cyto')
        )
        
        # Handle different image dimensions
        if img.ndim == 4:
            # 4D: Take max projection or process each channel
            img_2d = np.max(img, axis=(0, 1))
        elif img.ndim == 3:
            # 3D: Take max projection
            img_2d = np.max(img, axis=0)
        else:
            img_2d = img
        
        # Normalize for Cellpose
        img_normalized = self.normalize_image(img_2d)
        
        # Run segmentation
        masks, flows, styles, diams = model.eval(
            img_normalized,
            diameter=self.seg_config.get('diameter', 30),
            channels=self.seg_config.get('channels', [0, 0]),
            flow_threshold=self.seg_config.get('flow_threshold', 0.4),
            cellprob_threshold=self.seg_config.get('cellprob_threshold', 0.0)
        )
        
        logger.info(f"Found {len(np.unique(masks)) - 1} regions")
        return masks, flows
    
    def extract_features(self, img: np.ndarray, masks: np.ndarray) -> Dict:
        """
        Extract spatial and morphological features from segmented regions
        
        Args:
            img: Original image
            masks: Segmentation masks
            
        Returns:
            Dictionary of features per region
        """
        logger.info("Extracting features from segmented regions...")
        
        # Handle multi-dimensional images
        if img.ndim > 2:
            img_2d = np.max(img, axis=tuple(range(img.ndim - 2)))
        else:
            img_2d = img
        
        # Measure region properties
        props = measure.regionprops(masks, intensity_image=img_2d)
        
        features = {
            'region_ids': [],
            'centroids': [],
            'areas': [],
            'perimeters': [],
            'eccentricities': [],
            'mean_intensities': [],
            'max_intensities': [],
            'min_intensities': [],
            'std_intensities': []
        }
        
        for prop in props:
            features['region_ids'].append(prop.label)
            features['centroids'].append(prop.centroid)
            features['areas'].append(prop.area)
            features['perimeters'].append(prop.perimeter)
            features['eccentricities'].append(prop.eccentricity)
            features['mean_intensities'].append(prop.mean_intensity)
            features['max_intensities'].append(prop.max_intensity)
            features['min_intensities'].append(prop.min_intensity)
            
            # Calculate std intensity
            region_pixels = img_2d[prop.coords[:, 0], prop.coords[:, 1]]
            features['std_intensities'].append(np.std(region_pixels))
        
        logger.info(f"Extracted features for {len(features['region_ids'])} regions")
        return features
    
    def process_single_tiff(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete processing pipeline for a single TIFF file
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            img: Original image
            masks: Segmentation masks
            features: Extracted features
        """
        img = self.load_tiff(file_path)
        masks, flows = self.segment_cellpose(img)
        features = self.extract_features(img, masks)
        
        return img, masks, features
