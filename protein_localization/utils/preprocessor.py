"""
Image Preprocessor Module
Handles preprocessing of TIFF images including normalization, denoising, and enhancement
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from skimage import filters, exposure, restoration, morphology
from skimage.util import img_as_float, img_as_ubyte
from scipy import ndimage
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess TIFF images for analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default preprocessing configuration"""
        return {
            'normalize': True,
            'denoise': True,
            'denoise_method': 'gaussian',
            'enhance_contrast': True,
            'contrast_method': 'clahe',
            'remove_artifacts': True,
            'z_stack_projection': 'max',
            'target_size': None,
        }
    
    def preprocess(self, image: np.ndarray, filename: str = "") -> np.ndarray:
        """
        Apply complete preprocessing pipeline
        
        Args:
            image: Input image array
            filename: Optional filename for logging
            
        Returns:
            Preprocessed image
        """
        logger.debug(f"Preprocessing {filename or 'image'}")
        
        # Handle 3D images (z-stacks)
        if image.ndim == 3 and image.shape[0] > 3:
            image = self._project_z_stack(image)
        
        # Handle multi-channel images
        if image.ndim == 3 and image.shape[2] <= 3:
            # Process each channel separately
            processed_channels = []
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                processed = self._preprocess_single_channel(channel)
                processed_channels.append(processed)
            image = np.stack(processed_channels, axis=2)
        else:
            # Single channel image
            image = self._preprocess_single_channel(image)
        
        # Resize if needed
        if self.config.get('target_size'):
            image = self._resize(image, self.config['target_size'])
        
        return image
    
    def _preprocess_single_channel(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single channel image"""
        # Convert to float for processing
        image = img_as_float(image)
        
        # Remove artifacts
        if self.config.get('remove_artifacts'):
            image = self._remove_artifacts(image)
        
        # Denoise
        if self.config.get('denoise'):
            image = self._denoise(image, method=self.config.get('denoise_method', 'gaussian'))
        
        # Normalize
        if self.config.get('normalize'):
            image = self._normalize(image)
        
        # Enhance contrast
        if self.config.get('enhance_contrast'):
            image = self._enhance_contrast(image, method=self.config.get('contrast_method', 'clahe'))
        
        return image
    
    def _project_z_stack(self, image: np.ndarray) -> np.ndarray:
        """
        Project 3D z-stack to 2D
        
        Args:
            image: 3D image array (z, y, x)
            
        Returns:
            2D projected image
        """
        method = self.config.get('z_stack_projection', 'max')
        
        if method == 'max':
            return np.max(image, axis=0)
        elif method == 'mean':
            return np.mean(image, axis=0)
        elif method == 'median':
            return np.median(image, axis=0)
        else:
            logger.warning(f"Unknown projection method: {method}, using max")
            return np.max(image, axis=0)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
            image = np.clip(image, 0, 1)
        
        return image
    
    def _denoise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Apply denoising
        
        Args:
            image: Input image
            method: Denoising method (gaussian, bilateral, nlmeans)
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            return filters.gaussian(image, sigma=1.0)
        
        elif method == 'bilateral':
            # Convert to uint8 for bilateral filtering
            image_uint8 = img_as_ubyte(image)
            denoised = cv2.bilateralFilter(image_uint8, 9, 75, 75)
            return img_as_float(denoised)
        
        elif method == 'nlmeans':
            sigma_est = restoration.estimate_sigma(image)
            return restoration.denoise_nl_means(
                image, 
                h=1.15 * sigma_est, 
                fast_mode=True,
                patch_size=5,
                patch_distance=6
            )
        
        else:
            logger.warning(f"Unknown denoising method: {method}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image: Input image
            method: Enhancement method (clahe, histogram_equalization)
            
        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            # Convert to uint8 for CLAHE
            image_uint8 = img_as_ubyte(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_uint8)
            return img_as_float(enhanced)
        
        elif method == 'histogram_equalization':
            return exposure.equalize_hist(image)
        
        elif method == 'adaptive':
            return exposure.equalize_adapthist(image)
        
        else:
            logger.warning(f"Unknown contrast method: {method}")
            return image
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove common imaging artifacts
        
        Args:
            image: Input image
            
        Returns:
            Image with artifacts removed
        """
        # Remove small bright spots (hot pixels)
        image = morphology.opening(image, morphology.disk(1))
        
        # Remove background illumination variation
        background = filters.gaussian(image, sigma=50)
        image = image - background
        image = np.clip(image, 0, None)
        
        return image
    
    def _resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: (height, width)
            
        Returns:
            Resized image
        """
        if image.ndim == 2:
            return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            # Multi-channel image
            channels = []
            for i in range(image.shape[2]):
                resized = cv2.resize(image[:, :, i], target_size[::-1], 
                                   interpolation=cv2.INTER_LINEAR)
                channels.append(resized)
            return np.stack(channels, axis=2)
    
    def preprocess_batch(self, images: Dict[str, np.ndarray], 
                        output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Preprocess a batch of images
        
        Args:
            images: Dictionary mapping filenames to image arrays
            output_dir: Optional directory to save preprocessed images
            
        Returns:
            Dictionary of preprocessed images
        """
        logger.info(f"Preprocessing {len(images)} images...")
        
        processed_images = {}
        
        for filename, image in tqdm(images.items(), desc="Preprocessing"):
            try:
                processed = self.preprocess(image, filename)
                processed_images[filename] = processed
                
                # Save if output directory specified
                if output_dir:
                    self._save_processed(processed, filename, output_dir)
                    
            except Exception as e:
                logger.error(f"Error preprocessing {filename}: {str(e)}")
                continue
        
        logger.info(f"Successfully preprocessed {len(processed_images)} images")
        return processed_images
    
    def _save_processed(self, image: np.ndarray, filename: str, output_dir: str):
        """Save preprocessed image"""
        import tifffile
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to appropriate format for saving
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        output_file = output_path / filename
        tifffile.imwrite(str(output_file), image)


def preprocess_image(image: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
    """
    Convenience function to preprocess a single image
    
    Args:
        image: Input image
        config: Optional configuration dictionary
        
    Returns:
        Preprocessed image
    """
    preprocessor = ImagePreprocessor(config)
    return preprocessor.preprocess(image)


if __name__ == "__main__":
    import argparse
    from data_loader import TIFFDataLoader
    
    parser = argparse.ArgumentParser(description="Preprocess TIFF images")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing raw TIFF images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save preprocessed images")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize images")
    parser.add_argument("--denoise", action="store_true", default=True,
                       help="Apply denoising")
    parser.add_argument("--enhance_contrast", action="store_true", default=True,
                       help="Enhance contrast")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'normalize': args.normalize,
        'denoise': args.denoise,
        'enhance_contrast': args.enhance_contrast,
    }
    
    # Load images
    loader = TIFFDataLoader(args.input_dir)
    images_dict = loader.load_all(validate=True)
    images = {k: v[0] for k, v in images_dict.items()}
    
    # Preprocess
    preprocessor = ImagePreprocessor(config)
    processed = preprocessor.preprocess_batch(images, args.output_dir)
    
    logger.info(f"Saved {len(processed)} preprocessed images to {args.output_dir}")
