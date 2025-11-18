"""
Recursive TIFF file loader and preprocessing script
Loads 4D TIFF images from nested subdirectories
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple
import tifffile


class TIFFLoader:
    """Recursively loads TIFF files from a directory structure"""
    
    def __init__(self, root_dir: str):
        """
        Initialize the TIFF loader
        
        Args:
            root_dir: Root directory containing TIFF files in subdirectories
        """
        self.root_dir = Path(root_dir)
        self.tiff_files = []
        
    def scan_directory(self) -> List[Path]:
        """
        Recursively scan all subdirectories for TIFF files
        
        Returns:
            List of paths to TIFF files
        """
        self.tiff_files = []
        
        for file_path in self.root_dir.rglob('*.tif*'):
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                self.tiff_files.append(file_path)
        
        print(f"Found {len(self.tiff_files)} TIFF files")
        return self.tiff_files
    
    def load_single_tiff(self, file_path: Path) -> np.ndarray:
        """
        Load a single TIFF file
        
        Args:
            file_path: Path to the TIFF file
            
        Returns:
            Numpy array containing the image data
        """
        try:
            image = tifffile.imread(str(file_path))
            return image
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_tiffs(self) -> List[Tuple[Path, np.ndarray]]:
        """
        Load all TIFF files found in the directory
        
        Returns:
            List of tuples (file_path, image_data)
        """
        if not self.tiff_files:
            self.scan_directory()
        
        loaded_images = []
        for file_path in self.tiff_files:
            image = self.load_single_tiff(file_path)
            if image is not None:
                loaded_images.append((file_path, image))
        
        return loaded_images
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Get information about an image
        
        Args:
            image: Numpy array of image data
            
        Returns:
            Dictionary with image information
        """
        return {
            'shape': image.shape,
            'dtype': image.dtype,
            'min': image.min(),
            'max': image.max(),
            'mean': image.mean(),
            'ndim': image.ndim
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "D:\\5TH_SEM\\CELLULAR\\input"
    
    print(f"Scanning directory: {root_dir}")
    
    # Check if directory exists
    if not os.path.exists(root_dir):
        print(f"Warning: Directory {root_dir} does not exist")
        print("This script is designed to work with actual TIFF files")
        print("For demonstration purposes, the structure is ready")
    else:
        loader = TIFFLoader(root_dir)
        tiff_files = loader.scan_directory()
        
        if tiff_files:
            print("\nFirst 5 TIFF files found:")
            for i, file_path in enumerate(tiff_files[:5]):
                print(f"{i+1}. {file_path}")
            
            # Load first image as example
            if tiff_files:
                print("\nLoading first image...")
                first_image = loader.load_single_tiff(tiff_files[0])
                if first_image is not None:
                    info = loader.get_image_info(first_image)
                    print("\nImage information:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
