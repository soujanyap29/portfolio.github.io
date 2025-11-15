"""
Preprocessing Module for Protein Sub-Cellular Localization
Handles TIFF loading, segmentation, and feature extraction
"""

import os
import glob
import numpy as np
import tifffile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from cellpose import models
import cv2
from skimage import measure, morphology
from scipy import ndimage
import pickle


class TIFFProcessor:
    """Process 4D TIFF images from multiple subdirectories"""
    
    def __init__(self, input_dir: str = "/mnt/d/5TH_SEM/CELLULAR/input"):
        self.input_dir = Path(input_dir)
        self.tiff_files = []
        
    def scan_directories(self) -> List[Path]:
        """Recursively scan for TIFF files"""
        patterns = ['**/*.tif', '**/*.tiff']
        tiff_files = []
        
        for pattern in patterns:
            tiff_files.extend(self.input_dir.glob(pattern))
        
        self.tiff_files = sorted(list(set(tiff_files)))
        print(f"Found {len(self.tiff_files)} TIFF files")
        return self.tiff_files
    
    def load_tiff(self, filepath: Path) -> np.ndarray:
        """Load a TIFF file"""
        try:
            img = tifffile.imread(str(filepath))
            print(f"Loaded {filepath.name}: shape {img.shape}")
            return img
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img


class CellposeSegmenter:
    """Segment neuronal structures using Cellpose"""
    
    def __init__(self, model_type: str = 'cyto2', gpu: bool = True):
        # Use CellposeModel for Cellpose 2.0+ compatibility
        self.model = models.CellposeModel(model_type=model_type, gpu=gpu)
        self.diameter = None  # Auto-detect
        
    def segment(self, img: np.ndarray, channels: List[int] = [0, 0]) -> Tuple[np.ndarray, Dict]:
        """
        Segment image using Cellpose
        
        Args:
            img: Input image (can be 2D, 3D, or 4D)
            channels: [cytoplasm, nucleus] channel indices
            
        Returns:
            masks: Segmentation masks
            metadata: Additional segmentation info
        """
        # Handle different image dimensions
        if img.ndim == 4:  # 4D TIFF (time, z, y, x)
            # Use maximum intensity projection
            img_2d = np.max(img, axis=(0, 1))
        elif img.ndim == 3:  # 3D (z, y, x) or (y, x, channels)
            if img.shape[-1] <= 3:  # Likely RGB/multi-channel
                img_2d = img
            else:
                img_2d = np.max(img, axis=0)  # Z-projection
        else:
            img_2d = img
            
        # Run segmentation
        masks, flows, styles, diams = self.model.eval(
            img_2d,
            diameter=self.diameter,
            channels=channels,
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        
        metadata = {
            'diameters': diams,
            'num_cells': masks.max(),
            'image_shape': img_2d.shape
        }
        
        return masks, metadata


class FeatureExtractor:
    """Extract features from segmented images"""
    
    def __init__(self):
        self.features = {}
        
    def extract_region_properties(self, image: np.ndarray, masks: np.ndarray) -> List[Dict]:
        """Extract morphological and intensity features for each region"""
        regions = measure.regionprops(masks, intensity_image=image)
        
        features_list = []
        for region in regions:
            features = {
                # Spatial coordinates
                'centroid': region.centroid,
                'bbox': region.bbox,
                
                # Morphological features
                'area': region.area,
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'orientation': region.orientation,
                
                # Intensity features
                'mean_intensity': region.mean_intensity,
                'max_intensity': region.max_intensity,
                'min_intensity': region.min_intensity,
                
                # Shape descriptors
                'equivalent_diameter': region.equivalent_diameter,
                'convex_area': region.convex_area,
            }
            
            # Additional computed features
            if features['minor_axis_length'] > 0:
                features['aspect_ratio'] = features['major_axis_length'] / features['minor_axis_length']
            else:
                features['aspect_ratio'] = 0
                
            features['compactness'] = (4 * np.pi * features['area']) / (features['perimeter'] ** 2) if features['perimeter'] > 0 else 0
            
            features_list.append(features)
            
        return features_list
    
    def extract_channel_intensities(self, image: np.ndarray, masks: np.ndarray) -> Dict:
        """Extract channel-wise intensity distributions"""
        channel_features = {}
        
        if image.ndim == 3 and image.shape[-1] <= 3:  # Multi-channel image
            for ch in range(image.shape[-1]):
                channel_img = image[:, :, ch]
                regions = measure.regionprops(masks, intensity_image=channel_img)
                
                channel_features[f'channel_{ch}'] = [
                    {
                        'mean_intensity': r.mean_intensity,
                        'integrated_intensity': r.mean_intensity * r.area,
                    }
                    for r in regions
                ]
        else:
            regions = measure.regionprops(masks, intensity_image=image)
            channel_features['channel_0'] = [
                {
                    'mean_intensity': r.mean_intensity,
                    'integrated_intensity': r.mean_intensity * r.area,
                }
                for r in regions
            ]
            
        return channel_features
    
    def extract_spatial_features(self, masks: np.ndarray) -> Dict:
        """Extract spatial relationship features"""
        num_regions = masks.max()
        
        # Calculate center of mass for each region
        centroids = []
        for i in range(1, num_regions + 1):
            region_mask = (masks == i)
            centroid = ndimage.center_of_mass(region_mask)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        if len(centroids) > 0:
            distances = squareform(pdist(centroids))
        else:
            distances = np.array([])
        
        spatial_features = {
            'num_regions': num_regions,
            'centroids': centroids,
            'distance_matrix': distances,
            'mean_distance': distances.mean() if distances.size > 0 else 0,
            'min_distance': distances[distances > 0].min() if distances.size > 0 and (distances > 0).any() else 0
        }
        
        return spatial_features


class PreprocessingPipeline:
    """Complete preprocessing pipeline"""
    
    def __init__(self, input_dir: str = "/mnt/d/5TH_SEM/CELLULAR/input",
                 output_dir: str = "/mnt/d/5TH_SEM/CELLULAR/output"):
        self.processor = TIFFProcessor(input_dir)
        self.segmenter = CellposeSegmenter()
        self.feature_extractor = FeatureExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_all(self) -> List[Dict]:
        """Process all TIFF files in the input directory"""
        tiff_files = self.processor.scan_directories()
        results = []
        
        for tiff_file in tiff_files:
            print(f"\nProcessing: {tiff_file.name}")
            result = self.process_single(tiff_file)
            if result is not None:
                results.append(result)
                
        # Save results
        self.save_results(results)
        return results
    
    def process_single(self, filepath: Path) -> Optional[Dict]:
        """Process a single TIFF file"""
        # Load image
        img = self.processor.load_tiff(filepath)
        if img is None:
            return None
            
        # Normalize
        img_norm = self.processor.normalize_image(img)
        
        # Segment
        masks, seg_metadata = self.segmenter.segment(img_norm)
        
        # Extract features
        if img.ndim == 4:
            img_2d = np.max(img, axis=(0, 1))
        elif img.ndim == 3:
            img_2d = np.max(img, axis=0) if img.shape[-1] > 3 else img
        else:
            img_2d = img
            
        region_features = self.feature_extractor.extract_region_properties(img_2d, masks)
        channel_features = self.feature_extractor.extract_channel_intensities(img_2d, masks)
        spatial_features = self.feature_extractor.extract_spatial_features(masks)
        
        result = {
            'filename': filepath.name,
            'filepath': str(filepath),
            'image_shape': img.shape,
            'masks': masks,
            'segmentation_metadata': seg_metadata,
            'region_features': region_features,
            'channel_features': channel_features,
            'spatial_features': spatial_features,
            'num_regions': len(region_features)
        }
        
        return result
    
    def save_results(self, results: List[Dict]):
        """Save preprocessing results"""
        output_file = self.output_dir / 'preprocessed_data.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    # Example usage
    pipeline = PreprocessingPipeline()
    results = pipeline.process_all()
    print(f"\nProcessed {len(results)} images successfully")
