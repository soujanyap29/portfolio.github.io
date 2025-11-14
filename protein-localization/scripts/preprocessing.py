"""
Image preprocessing and segmentation using Cellpose-like algorithms
"""

import numpy as np
from typing import Tuple, List
from scipy import ndimage
from skimage import filters, morphology, measure


class ImagePreprocessor:
    """Preprocess and segment cellular structures in images"""
    
    def __init__(self):
        self.segmented_regions = []
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image
    
    def denoise_image(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian filtering to denoise the image
        
        Args:
            image: Input image
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Denoised image
        """
        return filters.gaussian(image, sigma=sigma)
    
    def segment_cells(self, image: np.ndarray, threshold_method: str = 'otsu') -> np.ndarray:
        """
        Segment cellular structures using thresholding
        
        Args:
            image: Input image (2D or 3D)
            threshold_method: Method for thresholding ('otsu', 'li', 'yen')
            
        Returns:
            Binary segmentation mask
        """
        # Handle 3D/4D images by taking max projection if needed
        if image.ndim > 2:
            if image.ndim == 4:
                # For 4D TIFF, take max projection across time and z
                image = np.max(image, axis=(0, 1))
            elif image.ndim == 3:
                image = np.max(image, axis=0)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Denoise
        image = self.denoise_image(image)
        
        # Apply threshold
        if threshold_method == 'otsu':
            threshold = filters.threshold_otsu(image)
        elif threshold_method == 'li':
            threshold = filters.threshold_li(image)
        elif threshold_method == 'yen':
            threshold = filters.threshold_yen(image)
        else:
            threshold = 0.5
        
        binary = image > threshold
        
        # Clean up segmentation
        binary = morphology.remove_small_objects(binary, min_size=50)
        binary = morphology.remove_small_holes(binary, area_threshold=50)
        
        return binary
    
    def label_regions(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Label connected regions in binary mask
        
        Args:
            binary_mask: Binary segmentation mask
            
        Returns:
            Tuple of (labeled_image, num_regions)
        """
        labeled_image = measure.label(binary_mask)
        num_regions = labeled_image.max()
        
        return labeled_image, num_regions
    
    def extract_features(self, image: np.ndarray, labeled_regions: np.ndarray) -> List[dict]:
        """
        Extract features from segmented regions
        
        Args:
            image: Original image
            labeled_regions: Labeled segmentation mask
            
        Returns:
            List of feature dictionaries for each region
        """
        features = []
        regions = measure.regionprops(labeled_regions, intensity_image=image)
        
        for region in regions:
            feature_dict = {
                'label': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'mean_intensity': region.mean_intensity,
                'max_intensity': region.max_intensity,
                'min_intensity': region.min_intensity,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'bbox': region.bbox
            }
            features.append(feature_dict)
        
        self.segmented_regions = features
        return features
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (labeled_regions, features)
        """
        # Segment
        binary_mask = self.segment_cells(image)
        
        # Label regions
        labeled_regions, num_regions = self.label_regions(binary_mask)
        
        print(f"Found {num_regions} regions")
        
        # Extract features
        # For 4D images, use max projection
        if image.ndim > 2:
            if image.ndim == 4:
                image_2d = np.max(image, axis=(0, 1))
            else:
                image_2d = np.max(image, axis=0)
        else:
            image_2d = image
        
        image_2d = self.normalize_image(image_2d)
        features = self.extract_features(image_2d, labeled_regions)
        
        return labeled_regions, features


if __name__ == "__main__":
    # Test with synthetic data
    print("Creating synthetic test image...")
    
    # Create a simple test image
    test_image = np.zeros((100, 100))
    test_image[20:40, 20:40] = 1.0
    test_image[60:80, 60:80] = 0.8
    test_image[20:40, 70:85] = 0.6
    
    # Add noise
    test_image += np.random.normal(0, 0.1, test_image.shape)
    
    print("Processing test image...")
    preprocessor = ImagePreprocessor()
    labeled_regions, features = preprocessor.process_image(test_image)
    
    print(f"\nExtracted {len(features)} regions")
    print("\nFeatures of first region:")
    if features:
        for key, value in features[0].items():
            print(f"  {key}: {value}")
