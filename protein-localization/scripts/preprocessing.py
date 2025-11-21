"""
Preprocessing Module
Handles image segmentation using Cellpose and feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
import cv2


class ImagePreprocessor:
    """
    Preprocess microscopy images with segmentation and feature extraction.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            use_gpu: Whether to use GPU for Cellpose
        """
        self.use_gpu = use_gpu
        self.model = None
        
    def load_cellpose_model(self, model_type: str = 'cyto2'):
        """
        Load Cellpose segmentation model.
        
        Args:
            model_type: Type of Cellpose model ('cyto', 'cyto2', 'nuclei')
        """
        try:
            from cellpose import models
            # Try newer Cellpose API first (v2.0+)
            try:
                self.model = models.CellposeModel(gpu=self.use_gpu, model_type=model_type)
                print(f"Loaded Cellpose model: {model_type} (v2.0+ API)")
            except (AttributeError, TypeError):
                # Fall back to older API
                try:
                    self.model = models.Cellpose(gpu=self.use_gpu, model_type=model_type)
                    print(f"Loaded Cellpose model: {model_type} (legacy API)")
                except AttributeError:
                    # Try direct model instantiation
                    self.model = models.Cellpose(model_type=model_type, gpu=self.use_gpu)
                    print(f"Loaded Cellpose model: {model_type} (direct API)")
        except ImportError:
            print("Warning: Cellpose not available. Using fallback segmentation.")
            self.model = None
        except Exception as e:
            print(f"Warning: Could not load Cellpose ({e}). Using fallback segmentation.")
            self.model = None
    
    def segment_image(self, image: np.ndarray, diameter: float = 30.0,
                     channels: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Segment image to detect cells and compartments.
        
        Args:
            image: Input image array
            diameter: Expected cell diameter in pixels
            channels: Channel configuration [cytoplasm, nucleus] or None
            
        Returns:
            Tuple of (mask array, segmentation info dict)
        """
        # Handle multi-dimensional images
        if image.ndim > 2:
            # Use first 2D slice or max projection
            if image.ndim == 3:
                if image.shape[0] < 10:  # Likely channels
                    img_2d = np.max(image, axis=0)
                else:  # Likely Z-stack
                    img_2d = np.max(image, axis=0)
            else:
                img_2d = np.max(image[0], axis=0) if image.ndim == 4 else image
        else:
            img_2d = image
        
        # Normalize image
        img_normalized = self._normalize_image(img_2d)
        
        if self.model is not None:
            # Use Cellpose
            try:
                masks, flows, styles, diams = self.model.eval(
                    img_normalized,
                    diameter=diameter,
                    channels=channels if channels else [0, 0]
                )
                
                info = {
                    'n_cells': len(np.unique(masks)) - 1,
                    'diameter': diams,
                    'method': 'cellpose'
                }
                return masks, info
            except Exception as e:
                print(f"Cellpose segmentation failed: {e}. Using fallback.")
                return self._fallback_segmentation(img_normalized)
        else:
            # Fallback segmentation
            return self._fallback_segmentation(img_normalized)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range."""
        img = image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
        return (img * 255).astype(np.uint8)
    
    def _fallback_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fallback segmentation using traditional methods.
        
        Args:
            image: Normalized 2D image
            
        Returns:
            Tuple of (mask array, info dict)
        """
        # Otsu thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Label connected components
        masks = measure.label(binary)
        
        info = {
            'n_cells': len(np.unique(masks)) - 1,
            'method': 'threshold'
        }
        
        return masks, info
    
    def extract_features(self, image: np.ndarray, masks: np.ndarray,
                        channels: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Extract features from segmented regions.
        
        Args:
            image: Original image array
            masks: Segmentation mask array
            channels: List of channel indices to analyze
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        # Get region properties
        props = measure.regionprops(masks, intensity_image=image if image.ndim == 2 else None)
        
        for region in props:
            features = self._extract_region_features(region, image, masks)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        return df
    
    def _extract_region_features(self, region, image: np.ndarray, masks: np.ndarray) -> Dict:
        """
        Extract comprehensive features from a single region.
        
        Args:
            region: Region property object from skimage
            image: Original image
            masks: Segmentation masks
            
        Returns:
            Dictionary of features
        """
        features = {
            # Identity
            'label': region.label,
            
            # Spatial coordinates
            'centroid_x': region.centroid[1] if len(region.centroid) > 1 else 0,
            'centroid_y': region.centroid[0],
            
            # Morphological features
            'area': region.area,
            'perimeter': region.perimeter,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,
            'orientation': region.orientation,
            'major_axis_length': region.major_axis_length,
            'minor_axis_length': region.minor_axis_length,
            'convex_area': region.convex_area,
            'equivalent_diameter': region.equivalent_diameter,
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-10),
        }
        
        # Intensity features
        if hasattr(region, 'intensity_image') and region.intensity_image is not None:
            features.update({
                'mean_intensity': region.mean_intensity,
                'max_intensity': region.max_intensity,
                'min_intensity': region.min_intensity,
            })
        
        # Texture features (if image is 2D)
        if image.ndim == 2:
            try:
                bbox = region.bbox
                roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                if roi.size > 0:
                    # Normalize ROI for texture analysis
                    roi_norm = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-10) * 7).astype(np.uint8)
                    
                    # GLCM texture features
                    glcm = graycomatrix(roi_norm, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 8, symmetric=True, normed=True)
                    
                    features.update({
                        'texture_contrast': graycoprops(glcm, 'contrast').mean(),
                        'texture_dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
                        'texture_homogeneity': graycoprops(glcm, 'homogeneity').mean(),
                        'texture_energy': graycoprops(glcm, 'energy').mean(),
                        'texture_correlation': graycoprops(glcm, 'correlation').mean(),
                    })
            except:
                pass
        
        return features
    
    def save_features(self, features: pd.DataFrame, output_dir: str, basename: str):
        """
        Save extracted features to CSV and JSON.
        
        Args:
            features: DataFrame with features
            output_dir: Output directory path
            basename: Base name for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_path / f"{basename}_features.csv"
        features.to_csv(csv_file, index=False)
        print(f"Saved features to {csv_file}")
        
        # Save as JSON
        json_file = output_path / f"{basename}_features.json"
        features_dict = features.to_dict(orient='records')
        with open(json_file, 'w') as f:
            json.dump(features_dict, f, indent=2)
        print(f"Saved features to {json_file}")
    
    def process_image(self, image: np.ndarray, output_dir: Optional[str] = None,
                     basename: str = "image") -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image: Input image array
            output_dir: Directory to save features (optional)
            basename: Base name for output files
            
        Returns:
            Tuple of (masks, features DataFrame, info dict)
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_cellpose_model()
        
        # Segment image
        print(f"Segmenting {basename}...")
        masks, seg_info = self.segment_image(image)
        
        # Extract features
        print(f"Extracting features from {seg_info['n_cells']} regions...")
        features = self.extract_features(image, masks)
        
        # Save features if output directory provided
        if output_dir:
            self.save_features(features, output_dir, basename)
        
        info = {
            'basename': basename,
            'segmentation': seg_info,
            'n_features': len(features.columns),
            'n_regions': len(features)
        }
        
        return masks, features, info


if __name__ == "__main__":
    # Example usage
    import sys
    from tiff_loader import load_tiff_from_path
    
    if len(sys.argv) > 1:
        tiff_path = sys.argv[1]
        image, metadata = load_tiff_from_path(tiff_path)
        
        if image is not None:
            print(f"Processing {metadata['filename']}")
            preprocessor = ImagePreprocessor(use_gpu=False)
            masks, features, info = preprocessor.process_image(
                image,
                output_dir="/mnt/d/5TH_SEM/CELLULAR/output/features",
                basename=Path(tiff_path).stem
            )
            
            print(f"\n=== Processing Results ===")
            print(f"Detected regions: {info['n_regions']}")
            print(f"Extracted features: {info['n_features']}")
            print(f"\nFeature columns: {list(features.columns)}")
    else:
        print("Usage: python preprocessing.py <path_to_tiff>")
