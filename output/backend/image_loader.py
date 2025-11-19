"""
Image preprocessing and TIFF loading utilities with Cellpose integration
"""
import numpy as np
import tifffile
from PIL import Image
import cv2
from typing import Tuple, Union, Optional
import os

try:
    from cellpose import models, transforms
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: Cellpose not available. Install with: pip install cellpose")


class TIFFLoader:
    """Handle TIFF image loading and preprocessing with Cellpose support"""
    
    def __init__(self, use_cellpose: bool = True):
        """
        Initialize TIFF loader
        
        Args:
            use_cellpose: Whether to use Cellpose for preprocessing (default: True)
        """
        self.use_cellpose = use_cellpose and CELLPOSE_AVAILABLE
        if self.use_cellpose:
            # Initialize Cellpose model for nuclei/cells (cyto2 model is general purpose)
            self.cellpose_model = models.Cellpose(gpu=False, model_type='cyto2')
    
    @staticmethod
    def load_tiff(file_path: str) -> np.ndarray:
        """
        Load TIFF image file
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            numpy array of image data
        """
        try:
            image = tifffile.imread(file_path)
            return image
        except Exception as e:
            print(f"Error loading TIFF file {file_path}: {e}")
            return None
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        if image is None:
            return None
        
        image = image.astype(np.float32)
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())
        return image
    
    def preprocess_with_cellpose(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess image using Cellpose's advanced normalization and resizing
        
        Args:
            image: Input image
            target_size: Optional target size (width, height). If None, returns normalized image.
            
        Returns:
            Preprocessed image
        """
        if not self.use_cellpose:
            # Fallback to standard preprocessing
            return self.resize_image(image, target_size) if target_size else self.normalize_image(image)
        
        if image is None:
            return None
        
        # Ensure image is in correct format for Cellpose
        if len(image.shape) == 2:
            # Grayscale image
            img_normalized = image.astype(np.float32)
        elif len(image.shape) == 3 and image.shape[2] in [1, 3]:
            # RGB or single channel
            if image.shape[2] == 1:
                img_normalized = image[:, :, 0].astype(np.float32)
            else:
                img_normalized = image.astype(np.float32)
        else:
            # Convert to grayscale if unexpected shape
            img_normalized = np.mean(image, axis=-1).astype(np.float32) if len(image.shape) > 2 else image.astype(np.float32)
        
        # Use Cellpose's normalization (percentile-based, robust to outliers)
        img_normalized = transforms.normalize99(img_normalized)
        
        # Resize if target size is specified
        if target_size is not None:
            # Validate size parameter
            if not target_size or len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError(f"Invalid size parameter: {target_size}. Size must be a tuple of two positive integers (width, height).")
            
            # Use transforms.resize_image for better quality
            if len(img_normalized.shape) == 2:
                # For 2D images
                img_resized = transforms.resize_image(img_normalized, Ly=target_size[1], Lx=target_size[0])
            else:
                # For 3D images, resize while preserving channels
                img_resized = cv2.resize(img_normalized, target_size, interpolation=cv2.INTER_LINEAR)
            
            return img_resized
        
        return img_normalized
    
    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to specified dimensions (fallback method using OpenCV)
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        if image is None:
            return None
        
        # Validate size parameter
        if not size or len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"Invalid size parameter: {size}. Size must be a tuple of two positive integers (width, height).")
        
        # Handle multi-channel images
        try:
            if len(image.shape) == 2:
                image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            else:
                image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            raise ValueError(f"OpenCV resize error with size {size} and image shape {image.shape}: {str(e)}")
        
        return image
    
    @staticmethod
    def preprocess_for_model(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Complete preprocessing pipeline for model input
        
        Args:
            image: Raw TIFF image
            size: Target size for model
            
        Returns:
            Preprocessed image ready for model
        """
        # Normalize
        image = TIFFLoader.normalize_image(image)
        
        # Resize
        image = TIFFLoader.resize_image(image, size)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @staticmethod
    def scan_directory(directory: str, recursive: bool = True) -> list:
        """
        Scan directory for TIFF files
        
        Args:
            directory: Path to directory
            recursive: Whether to scan recursively
            
        Returns:
            List of TIFF file paths
        """
        tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        tiff_files = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return tiff_files
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.endswith(ext) for ext in tiff_extensions):
                        tiff_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if any(file.endswith(ext) for ext in tiff_extensions):
                    tiff_files.append(os.path.join(directory, file))
        
        return tiff_files


class ImageAugmentation:
    """Image augmentation utilities"""
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """Adjust image contrast"""
        return np.clip(image * alpha, 0, 1)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, beta: float = 0.1) -> np.ndarray:
        """Adjust image brightness"""
        return np.clip(image + beta, 0, 1)
    
    @staticmethod
    def gaussian_noise(image: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)
