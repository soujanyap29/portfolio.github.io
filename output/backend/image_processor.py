"""
Image preprocessing and TIFF handling module
"""
import numpy as np
import tifffile
from PIL import Image
import cv2
from skimage import exposure, transform
import os


class ImageProcessor:
    """Handle TIFF image loading and preprocessing"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_tiff(self, filepath):
        """
        Load TIFF microscopy image
        
        Args:
            filepath: Path to TIFF file
            
        Returns:
            numpy array of image data
        """
        try:
            img = tifffile.imread(filepath)
            return img
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            return None
    
    def normalize_image(self, img):
        """
        Normalize image intensities
        
        Args:
            img: Input image array
            
        Returns:
            Normalized image
        """
        # Handle different bit depths
        if img.dtype == np.uint16:
            img = (img / 65535.0 * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            pass
        else:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Apply histogram equalization for better contrast
        if len(img.shape) == 2:  # Grayscale
            img = exposure.equalize_adapthist(img, clip_limit=0.03)
        
        return img
    
    def resize_image(self, img, size=None):
        """
        Resize image to target size
        
        Args:
            img: Input image
            size: Target size (height, width)
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        return transform.resize(img, size, anti_aliasing=True, preserve_range=True)
    
    def preprocess(self, filepath):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to TIFF file
            
        Returns:
            Preprocessed image array
        """
        img = self.load_tiff(filepath)
        if img is None:
            return None
        
        # Normalize
        img = self.normalize_image(img)
        
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Resize
        img = self.resize_image(img)
        
        return img
    
    def save_image(self, img, filepath):
        """
        Save processed image
        
        Args:
            img: Image array
            filepath: Output filepath
        """
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        Image.fromarray(img).save(filepath)
    
    def scan_directory(self, directory):
        """
        Recursively scan directory for TIFF files
        
        Args:
            directory: Root directory to scan
            
        Returns:
            List of TIFF file paths
        """
        tiff_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tiff_files.append(os.path.join(root, file))
        
        return tiff_files
