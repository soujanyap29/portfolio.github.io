"""
Preprocessing module for TIFF image loading, segmentation, and feature extraction.
Handles recursive scanning of directories, Cellpose segmentation, and feature extraction.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tifffile
from skimage import measure, morphology
from scipy.ndimage import distance_transform_edt
import warnings

warnings.filterwarnings('ignore')


class TIFFPreprocessor:
    """
    Handles TIFF image preprocessing including loading, segmentation, and feature extraction.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing TIFF files
            output_dir: Directory to save processed outputs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def scan_tiff_files(self) -> List[Path]:
        """
        Recursively scan for all TIFF files in input directory.
        
        Returns:
            List of Path objects for all TIFF files found
        """
        tiff_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tiff_files.extend(self.input_dir.rglob(ext))
        
        print(f"Found {len(tiff_files)} TIFF files")
        return sorted(tiff_files)
    
    def load_tiff(self, file_path: Path) -> np.ndarray:
        """
        Load a TIFF file (supports 3D and 4D).
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            Numpy array containing the image data
        """
        try:
            img = tifffile.imread(str(file_path))
            print(f"Loaded {file_path.name}: shape {img.shape}, dtype {img.dtype}")
            return img
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def segment_cellpose(self, image: np.ndarray, diameter: float = 30.0, 
                        use_gpu: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Segment image using Cellpose or fallback method.
        
        Args:
            image: Input image array
            diameter: Expected cell diameter
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Tuple of (masks, metadata)
        """
        try:
            from cellpose import models
            
            # Initialize Cellpose model
            model = models.Cellpose(gpu=use_gpu, model_type='cyto')
            
            # Handle different image dimensions
            if image.ndim == 4:
                # 4D image: process middle z-slice of first timepoint
                img_2d = image[0, image.shape[1]//2, :, :]
            elif image.ndim == 3:
                # 3D image: process middle slice
                img_2d = image[image.shape[0]//2, :, :]
            else:
                img_2d = image
            
            # Run segmentation
            masks, flows, styles, diams = model.eval(img_2d, diameter=diameter, channels=[0, 0])
            
            metadata = {
                'n_cells': len(np.unique(masks)) - 1,
                'diameter': diams,
                'method': 'cellpose'
            }
            
            print(f"Cellpose segmentation: {metadata['n_cells']} cells detected")
            return masks, metadata
            
        except ImportError:
            print("Cellpose not available, using fallback segmentation")
            return self._fallback_segmentation(image)
        except Exception as e:
            print(f"Cellpose error: {e}, using fallback")
            return self._fallback_segmentation(image)
    
    def _fallback_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Simple threshold-based segmentation fallback.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (masks, metadata)
        """
        # Handle different dimensions
        if image.ndim == 4:
            img_2d = image[0, image.shape[1]//2, :, :]
        elif image.ndim == 3:
            img_2d = image[image.shape[0]//2, :, :]
        else:
            img_2d = image
        
        # Normalize
        img_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
        
        # Threshold
        threshold = np.percentile(img_norm, 75)
        binary = img_norm > threshold
        
        # Clean up
        binary = morphology.remove_small_objects(binary, min_size=50)
        binary = morphology.remove_small_holes(binary, area_threshold=50)
        
        # Label regions
        masks = measure.label(binary)
        
        metadata = {
            'n_cells': len(np.unique(masks)) - 1,
            'method': 'threshold'
        }
        
        print(f"Fallback segmentation: {metadata['n_cells']} regions detected")
        return masks, metadata
    
    def extract_features(self, image: np.ndarray, masks: np.ndarray) -> List[Dict]:
        """
        Extract comprehensive features from segmented regions.
        
        Args:
            image: Original image
            masks: Segmentation masks
            
        Returns:
            List of feature dictionaries for each region
        """
        # Handle different dimensions - get 2D representation
        if image.ndim == 4:
            img_2d = image[0, image.shape[1]//2, :, :]
        elif image.ndim == 3:
            img_2d = image[image.shape[0]//2, :, :]
        else:
            img_2d = image
        
        features_list = []
        regions = measure.regionprops(masks, intensity_image=img_2d)
        
        # Calculate distance from center for each region
        center_y, center_x = np.array(masks.shape) / 2
        
        for region in regions:
            # Spatial features
            centroid_y, centroid_x = region.centroid
            dist_from_center = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
            
            features = {
                # Identity
                'label': region.label,
                
                # Spatial coordinates
                'centroid_y': centroid_y,
                'centroid_x': centroid_x,
                'distance_from_center': dist_from_center,
                
                # Morphological descriptors
                'area': region.area,
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'extent': region.extent,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'orientation': region.orientation,
                
                # Intensity statistics
                'mean_intensity': region.mean_intensity,
                'max_intensity': region.max_intensity,
                'min_intensity': region.min_intensity,
                'intensity_std': np.std(img_2d[masks == region.label]),
            }
            
            # Add bounding box
            min_row, min_col, max_row, max_col = region.bbox
            features.update({
                'bbox_min_row': min_row,
                'bbox_min_col': min_col,
                'bbox_max_row': max_row,
                'bbox_max_col': max_col,
            })
            
            features_list.append(features)
        
        print(f"Extracted features from {len(features_list)} regions")
        return features_list
    
    def process_single_tiff(self, file_path: Path) -> Optional[Dict]:
        """
        Process a single TIFF file through the complete pipeline.
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            Dictionary containing processed data or None if failed
        """
        # Load image
        image = self.load_tiff(file_path)
        if image is None:
            return None
        
        # Segment
        masks, seg_metadata = self.segment_cellpose(image)
        
        # Extract features
        features = self.extract_features(image, masks)
        
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'image_shape': image.shape,
            'masks': masks,
            'features': features,
            'segmentation_metadata': seg_metadata,
            'n_regions': len(features)
        }
        
        return result
    
    def process_all_tiffs(self) -> List[Dict]:
        """
        Process all TIFF files in input directory.
        
        Returns:
            List of processed results
        """
        tiff_files = self.scan_tiff_files()
        results = []
        
        for tiff_file in tiff_files:
            print(f"\nProcessing: {tiff_file.name}")
            result = self.process_single_tiff(tiff_file)
            if result is not None:
                results.append(result)
        
        print(f"\n✓ Successfully processed {len(results)} files")
        return results


def preprocess_pipeline(input_dir: str, output_dir: str) -> List[Dict]:
    """
    Main preprocessing pipeline entry point.
    
    Args:
        input_dir: Directory containing input TIFF files
        output_dir: Directory to save outputs
        
    Returns:
        List of processed results
    """
    preprocessor = TIFFPreprocessor(input_dir, output_dir)
    results = preprocessor.process_all_tiffs()
    return results
