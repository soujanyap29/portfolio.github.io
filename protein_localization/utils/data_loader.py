"""
Data Loader Module for TIFF Images
Handles loading and validation of TIFF images from OpenCell database
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import tifffile
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TIFFDataLoader:
    """Load and validate TIFF images from specified directory"""
    
    def __init__(self, input_dir: str, supported_formats: List[str] = ['.tif', '.tiff']):
        """
        Initialize the data loader
        
        Args:
            input_dir: Directory containing TIFF images
            supported_formats: List of supported file extensions
        """
        self.input_dir = Path(input_dir)
        self.supported_formats = supported_formats
        self.image_files = []
        self.metadata = {}
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    def scan_directory(self) -> List[Path]:
        """
        Scan directory for TIFF files
        
        Returns:
            List of paths to TIFF files
        """
        logger.info(f"Scanning directory: {self.input_dir}")
        
        image_files = []
        for ext in self.supported_formats:
            pattern = str(self.input_dir / f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            image_files.extend([Path(f) for f in files])
        
        self.image_files = sorted(image_files)
        logger.info(f"Found {len(self.image_files)} TIFF files")
        
        return self.image_files
    
    def load_tiff(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load a single TIFF file
        
        Args:
            filepath: Path to TIFF file
            
        Returns:
            Tuple of (image array, metadata dictionary)
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
                    'size_bytes': image.nbytes,
                }
                
                # Add TIFF-specific metadata if available
                if tif.pages:
                    page = tif.pages[0]
                    if hasattr(page, 'tags'):
                        metadata['pixel_size'] = self._extract_pixel_size(page.tags)
                        metadata['description'] = self._extract_description(page.tags)
                
                self.metadata[filepath.name] = metadata
                
                return image, metadata
                
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def _extract_pixel_size(self, tags) -> Optional[float]:
        """Extract pixel size from TIFF tags"""
        try:
            if 'XResolution' in tags:
                x_res = tags['XResolution'].value
                if isinstance(x_res, tuple):
                    return x_res[1] / x_res[0] if x_res[0] != 0 else None
            return None
        except:
            return None
    
    def _extract_description(self, tags) -> Optional[str]:
        """Extract image description from TIFF tags"""
        try:
            if 'ImageDescription' in tags:
                return tags['ImageDescription'].value
            return None
        except:
            return None
    
    def load_all(self, validate: bool = True) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        Load all TIFF files in the directory
        
        Args:
            validate: Whether to validate images after loading
            
        Returns:
            Dictionary mapping filenames to (image, metadata) tuples
        """
        if not self.image_files:
            self.scan_directory()
        
        images = {}
        logger.info(f"Loading {len(self.image_files)} TIFF files...")
        
        for filepath in tqdm(self.image_files, desc="Loading images"):
            try:
                image, metadata = self.load_tiff(filepath)
                images[filepath.name] = (image, metadata)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {str(e)}")
                continue
        
        if validate:
            self.validate_images(images)
        
        return images
    
    def validate_images(self, images: Dict[str, Tuple[np.ndarray, Dict]]) -> bool:
        """
        Perform sanity checks on loaded images
        
        Args:
            images: Dictionary of loaded images
            
        Returns:
            True if all validations pass
        """
        logger.info("Performing sanity checks on loaded images...")
        
        issues = []
        
        for filename, (image, metadata) in images.items():
            # Check for empty images
            if image.size == 0:
                issues.append(f"{filename}: Empty image")
                continue
            
            # Check for NaN or Inf values
            if np.isnan(image).any():
                issues.append(f"{filename}: Contains NaN values")
            
            if np.isinf(image).any():
                issues.append(f"{filename}: Contains Inf values")
            
            # Check dimensions
            if image.ndim < 2:
                issues.append(f"{filename}: Invalid dimensions (ndim={image.ndim})")
            
            # Check data type
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                issues.append(f"{filename}: Unusual data type ({image.dtype})")
            
            # Check for extremely dark or bright images
            if image.max() == image.min():
                issues.append(f"{filename}: Constant intensity (no contrast)")
        
        if issues:
            logger.warning(f"Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more issues")
            return False
        
        logger.info("All sanity checks passed!")
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded images
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.metadata:
            logger.warning("No images loaded yet")
            return {}
        
        shapes = [meta['shape'] for meta in self.metadata.values()]
        dtypes = [meta['dtype'] for meta in self.metadata.values()]
        sizes = [meta['size_bytes'] for meta in self.metadata.values()]
        
        stats = {
            'num_images': len(self.metadata),
            'unique_shapes': list(set([str(s) for s in shapes])),
            'unique_dtypes': list(set(dtypes)),
            'total_size_mb': sum(sizes) / (1024 * 1024),
            'avg_size_mb': np.mean(sizes) / (1024 * 1024),
        }
        
        return stats
    
    def print_summary(self):
        """Print summary of loaded data"""
        stats = self.get_statistics()
        
        if not stats:
            logger.info("No data loaded")
            return
        
        logger.info("=" * 50)
        logger.info("Dataset Summary")
        logger.info("=" * 50)
        logger.info(f"Number of images: {stats['num_images']}")
        logger.info(f"Unique shapes: {stats['unique_shapes']}")
        logger.info(f"Data types: {stats['unique_dtypes']}")
        logger.info(f"Total size: {stats['total_size_mb']:.2f} MB")
        logger.info(f"Average size: {stats['avg_size_mb']:.2f} MB")
        logger.info("=" * 50)


def load_tiff(filepath: str) -> np.ndarray:
    """
    Simple convenience function to load a single TIFF file
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        Image array
    """
    return tifffile.imread(filepath)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and validate TIFF images")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing TIFF images")
    parser.add_argument("--validate", action="store_true",
                       help="Perform validation checks")
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = TIFFDataLoader(args.input_dir)
    
    # Scan and load images
    loader.scan_directory()
    images = loader.load_all(validate=args.validate)
    
    # Print summary
    loader.print_summary()
    
    logger.info(f"Successfully loaded {len(images)} images")
