"""
TIFF Loader Module
Handles loading of multi-dimensional TIFF microscopy images from multiple sub-folders.
Supports 3D, 4D, and multi-channel TIFF stacks.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import tifffile
from tqdm import tqdm


class TIFFLoader:
    """
    Load TIFF microscopy images from nested directory structures.
    """
    
    def __init__(self, root_dir: str, recursive: bool = True):
        """
        Initialize TIFF loader.
        
        Args:
            root_dir: Root directory to search for TIFF files
            recursive: Whether to search recursively in subdirectories
        """
        self.root_dir = Path(root_dir)
        self.recursive = recursive
        self.tiff_files: List[Path] = []
        
    def scan_directory(self) -> List[Path]:
        """
        Scan directory for TIFF files.
        
        Returns:
            List of paths to TIFF files
        """
        patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        self.tiff_files = []
        
        if self.recursive:
            for pattern in patterns:
                self.tiff_files.extend(self.root_dir.rglob(pattern))
        else:
            for pattern in patterns:
                self.tiff_files.extend(self.root_dir.glob(pattern))
        
        self.tiff_files = sorted(list(set(self.tiff_files)))
        print(f"Found {len(self.tiff_files)} TIFF files in {self.root_dir}")
        return self.tiff_files
    
    def load_tiff(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load a single TIFF file with metadata.
        
        Args:
            filepath: Path to TIFF file
            
        Returns:
            Tuple of (image array, metadata dict)
        """
        try:
            with tifffile.TiffFile(filepath) as tif:
                # Load image data
                image = tif.asarray()
                
                # Extract metadata
                metadata = {
                    'filename': filepath.name,
                    'filepath': str(filepath),
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'ndim': image.ndim,
                }
                
                # Add additional metadata if available
                if tif.imagej_metadata:
                    metadata['imagej'] = tif.imagej_metadata
                
                # Detect dimensions
                if image.ndim == 2:
                    metadata['dimensions'] = 'XY'
                elif image.ndim == 3:
                    # Could be XYZ or XYC (channels)
                    if image.shape[0] < 10:  # Likely channels
                        metadata['dimensions'] = 'CXY'
                    else:
                        metadata['dimensions'] = 'ZXY'
                elif image.ndim == 4:
                    metadata['dimensions'] = 'CZXY or TZXY'
                
                return image, metadata
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def load_all(self, max_files: Optional[int] = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        Load all discovered TIFF files.
        
        Args:
            max_files: Maximum number of files to load (None for all)
            
        Returns:
            List of (image, metadata) tuples
        """
        if not self.tiff_files:
            self.scan_directory()
        
        files_to_load = self.tiff_files[:max_files] if max_files else self.tiff_files
        loaded_data = []
        
        print(f"Loading {len(files_to_load)} TIFF files...")
        for filepath in tqdm(files_to_load, desc="Loading TIFF files"):
            image, metadata = self.load_tiff(filepath)
            if image is not None:
                loaded_data.append((image, metadata))
        
        print(f"Successfully loaded {len(loaded_data)} files")
        return loaded_data
    
    def load_batch(self, batch_size: int = 10) -> List[Tuple[np.ndarray, Dict]]:
        """
        Load TIFF files in batches (generator-style for memory efficiency).
        
        Args:
            batch_size: Number of files to load per batch
            
        Yields:
            Batches of (image, metadata) tuples
        """
        if not self.tiff_files:
            self.scan_directory()
        
        for i in range(0, len(self.tiff_files), batch_size):
            batch_files = self.tiff_files[i:i + batch_size]
            batch_data = []
            
            for filepath in batch_files:
                image, metadata = self.load_tiff(filepath)
                if image is not None:
                    batch_data.append((image, metadata))
            
            yield batch_data
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about discovered TIFF files without loading them.
        
        Returns:
            Dictionary with file statistics
        """
        if not self.tiff_files:
            self.scan_directory()
        
        stats = {
            'total_files': len(self.tiff_files),
            'file_sizes': [],
            'directories': set(),
        }
        
        for filepath in self.tiff_files:
            stats['file_sizes'].append(filepath.stat().st_size / (1024**2))  # MB
            stats['directories'].add(str(filepath.parent))
        
        stats['total_size_mb'] = sum(stats['file_sizes'])
        stats['avg_size_mb'] = np.mean(stats['file_sizes']) if stats['file_sizes'] else 0
        stats['unique_directories'] = len(stats['directories'])
        stats['directories'] = sorted(list(stats['directories']))
        
        return stats


def load_tiff_from_path(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to load a single TIFF file.
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        Tuple of (image array, metadata dict)
    """
    loader = TIFFLoader(os.path.dirname(filepath), recursive=False)
    return loader.load_tiff(Path(filepath))


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "/mnt/d/5TH_SEM/CELLULAR/input"
    
    print(f"Scanning directory: {input_dir}")
    loader = TIFFLoader(input_dir, recursive=True)
    
    # Get statistics
    stats = loader.get_statistics()
    print("\n=== TIFF File Statistics ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"Average size: {stats['avg_size_mb']:.2f} MB")
    print(f"Unique directories: {stats['unique_directories']}")
    
    # Load first file as example
    if stats['total_files'] > 0:
        print("\n=== Loading first file ===")
        data = loader.load_all(max_files=1)
        if data:
            image, metadata = data[0]
            print(f"Shape: {image.shape}")
            print(f"Dtype: {image.dtype}")
            print(f"Dimensions: {metadata['dimensions']}")
