"""
Image preprocessing utilities for TIFF microscopy images.
"""

import numpy as np
import tifffile
from PIL import Image
from skimage import transform, util
from scipy import ndimage
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TIFFLoader:
    """Load and preprocess TIFF microscopy images."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize TIFF loader.
        
        Args:
            target_size: Target image dimensions (height, width)
        """
        self.target_size = target_size
    
    def load_tiff(self, filepath: str) -> np.ndarray:
        """
        Load a TIFF image file and handle multi-dimensional stacks.
        
        Args:
            filepath: Path to TIFF file
            
        Returns:
            Loaded image as numpy array (2D or 3D)
        """
        try:
            image = tifffile.imread(filepath)
            logger.info(f"Loaded TIFF image: {filepath}, shape: {image.shape}")
            
            # Handle multi-dimensional TIFF stacks
            # Common formats: (z, channels, height, width) or (z, height, width) or (height, width, channels)
            if len(image.shape) == 4:
                # Take maximum projection along z-axis and first channel
                # Shape: (z, channels, height, width) -> (height, width)
                image = np.max(image[:, 0, :, :], axis=0)
                logger.info(f"Converted 4D stack to 2D using max projection: {image.shape}")
            elif len(image.shape) == 3:
                # If it's a stack, take max projection
                if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                    # Likely (z, height, width) format
                    image = np.max(image, axis=0)
                    logger.info(f"Converted 3D stack to 2D using max projection: {image.shape}")
                # Otherwise assume it's (height, width, channels) - keep as is
            
            return image
        except Exception as e:
            logger.error(f"Error loading TIFF file {filepath}: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            # Already float, normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size using scikit-image.
        
        Args:
            image: Input image (2D or 3D)
            
        Returns:
            Resized image
        """
        # Ensure we have a valid 2D or 3D image
        if len(image.shape) == 2:
            # Grayscale image - add channel dimension for consistent processing
            resized = transform.resize(
                image, 
                self.target_size, 
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
        elif len(image.shape) == 3:
            # Multi-channel image or RGB
            # Resize spatial dimensions while preserving channels
            target_shape = (self.target_size[0], self.target_size[1], image.shape[2])
            resized = transform.resize(
                image,
                target_shape,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected 2D or 3D array.")
        
        return resized.astype(image.dtype)
    
    def preprocess(self, filepath: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to TIFF file
            normalize: Whether to normalize the image
            
        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # Load image
        original = self.load_tiff(filepath)
        
        # Keep a copy of original
        processed = original.copy()
        
        # Normalize
        if normalize:
            processed = self.normalize_image(processed)
        
        # Resize
        processed = self.resize_image(processed)
        
        # Ensure 3 channels for CNN
        if len(processed.shape) == 2:
            processed = np.stack([processed] * 3, axis=-1)
        elif processed.shape[-1] == 1:
            processed = np.repeat(processed, 3, axis=-1)
        
        return original, processed
    
    def batch_load(self, directory: str, extensions: List[str] = ['.tif', '.tiff']) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Load all TIFF images from a directory.
        
        Args:
            directory: Directory path
            extensions: List of valid file extensions
            
        Returns:
            List of tuples (filepath, original_image, preprocessed_image)
        """
        dir_path = Path(directory)
        results = []
        
        # Recursively find all TIFF files
        for ext in extensions:
            for filepath in dir_path.rglob(f'*{ext}'):
                try:
                    original, processed = self.preprocess(str(filepath))
                    results.append((str(filepath), original, processed))
                    logger.info(f"Preprocessed: {filepath.name}")
                except Exception as e:
                    logger.warning(f"Failed to process {filepath}: {e}")
        
        logger.info(f"Successfully loaded {len(results)} images from {directory}")
        return results


class ImageAugmentor:
    """Optional image augmentation for training."""
    
    @staticmethod
    def random_flip(image: np.ndarray) -> np.ndarray:
        """Random horizontal flip."""
        if np.random.rand() > 0.5:
            return np.fliplr(image)
        return image
    
    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: int = 15) -> np.ndarray:
        """Random rotation using scipy."""
        angle = np.random.uniform(-max_angle, max_angle)
        rotated = ndimage.rotate(image, angle, reshape=False, mode='reflect')
        return rotated
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust brightness."""
        if factor is None:
            factor = np.random.uniform(0.8, 1.2)
        adjusted = image * factor
        return np.clip(adjusted, 0, 1)


def create_test_image(output_path: str, size: Tuple[int, int] = (512, 512)):
    """
    Create a test TIFF image for demonstration.
    
    Args:
        output_path: Where to save the test image
        size: Image dimensions
    """
    # Create a synthetic microscopy-like image
    image = np.random.randint(0, 65535, size=size, dtype=np.uint16)
    
    # Add some structure (simulating cells) using skimage
    from skimage.draw import disk
    for _ in range(10):
        center_x = np.random.randint(50, size[0] - 50)
        center_y = np.random.randint(50, size[1] - 50)
        radius = np.random.randint(20, 40)
        
        # Draw circle using disk function
        rr, cc = disk((center_x, center_y), radius, shape=image.shape)
        image[rr, cc] = 50000
    
    # Save as TIFF
    tifffile.imwrite(output_path, image)
    logger.info(f"Created test TIFF image at {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = TIFFLoader(target_size=(224, 224))
    
    # Create a test image
    test_path = "/tmp/test_microscopy.tif"
    create_test_image(test_path)
    
    # Load and preprocess
    original, processed = loader.preprocess(test_path)
    print(f"Original shape: {original.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
