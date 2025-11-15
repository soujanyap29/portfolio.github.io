"""
Preprocessing pipeline for protein sub-cellular localization
Handles directory scanning, TIFF loading, and segmentation
"""
import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import tifffile
from cellpose import models
from skimage import measure
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DirectoryHandler:
    """Handles recursive scanning of directories for TIFF files"""
    
    def __init__(self, root_dir: str, extensions: List[str] = ['.tif', '.tiff']):
        self.root_dir = root_dir
        self.extensions = [ext.lower() for ext in extensions]
    
    def scan_directory(self) -> List[str]:
        """Recursively scan directory for TIFF files"""
        tiff_files = []
        
        if not os.path.exists(self.root_dir):
            print(f"Warning: Directory {self.root_dir} does not exist")
            return tiff_files
        
        for ext in self.extensions:
            # Recursive glob search
            pattern = os.path.join(self.root_dir, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            tiff_files.extend(files)
            
            # Also check uppercase
            pattern = os.path.join(self.root_dir, f"**/*{ext.upper()}")
            files = glob.glob(pattern, recursive=True)
            tiff_files.extend(files)
        
        # Remove duplicates
        tiff_files = list(set(tiff_files))
        print(f"Found {len(tiff_files)} TIFF files in {self.root_dir}")
        
        return sorted(tiff_files)


class TIFFLoader:
    """Loads and processes TIFF images"""
    
    @staticmethod
    def load_tiff(filepath: str) -> np.ndarray:
        """Load a TIFF file and return as numpy array"""
        try:
            image = tifffile.imread(filepath)
            return image
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict:
        """Get information about the image"""
        if image is None:
            return {}
        
        return {
            'shape': image.shape,
            'dtype': image.dtype,
            'min': np.min(image),
            'max': np.max(image),
            'mean': np.mean(image),
            'std': np.std(image)
        }


class CellposeSegmenter:
    """Performs segmentation using Cellpose"""
    
    def __init__(self, model_type: str = 'cyto2', diameter: int = None, channels: List[int] = [0, 0]):
        """
        Initialize Cellpose segmenter
        
        Args:
            model_type: Type of Cellpose model ('cyto', 'cyto2', 'nuclei')
            diameter: Expected cell diameter (None for auto-detect)
            channels: Channel configuration [cytoplasm, nucleus]
        """
        self.model_type = model_type
        self.diameter = diameter
        self.channels = channels
        self.model = None
    
    def initialize_model(self):
        """Initialize the Cellpose model"""
        try:
            self.model = models.Cellpose(model_type=self.model_type)
            print(f"Cellpose model '{self.model_type}' initialized successfully")
        except Exception as e:
            print(f"Error initializing Cellpose: {str(e)}")
            print("Attempting to use CPU model...")
            self.model = models.Cellpose(model_type=self.model_type, gpu=False)
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Segment an image using Cellpose
        
        Args:
            image: Input image array
        
        Returns:
            masks: Labeled segmentation mask
            info: Dictionary with segmentation information
        """
        if self.model is None:
            self.initialize_model()
        
        try:
            # Handle different image dimensions
            if len(image.shape) == 4:  # 4D TIFF (T, Z, Y, X)
                # Process first time point and Z-slice
                img_2d = image[0, 0, :, :]
            elif len(image.shape) == 3:
                # If it's (Z, Y, X), take middle Z-slice
                if image.shape[0] < 10:  # Likely Z-stack
                    img_2d = image[image.shape[0] // 2, :, :]
                else:  # Likely (Y, X, C)
                    img_2d = image[:, :, 0] if image.shape[2] < 10 else image
            else:
                img_2d = image
            
            # Normalize image
            img_2d = self._normalize_image(img_2d)
            
            # Run segmentation
            masks, flows, styles, diams = self.model.eval(
                img_2d, 
                diameter=self.diameter,
                channels=self.channels,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            
            info = {
                'num_cells': len(np.unique(masks)) - 1,  # Exclude background
                'diameter': diams,
                'image_shape': img_2d.shape
            }
            
            return masks, info
            
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            return None, {}
    
    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        if image.dtype == np.uint8:
            return image
        
        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        
        return normalized
    
    def segment_3d_stack(self, image: np.ndarray, num_slices: int = None) -> np.ndarray:
        """
        Segment a 3D stack slice by slice
        
        Args:
            image: 3D image stack (Z, Y, X)
            num_slices: Number of slices to process (None for all)
        
        Returns:
            masks_3d: 3D labeled mask
        """
        if len(image.shape) != 3:
            raise ValueError("Image must be 3D for stack segmentation")
        
        z_dim = image.shape[0]
        if num_slices is None:
            num_slices = z_dim
        
        masks_3d = np.zeros(image.shape, dtype=np.int32)
        
        for z in tqdm(range(min(num_slices, z_dim)), desc="Segmenting Z-slices"):
            img_slice = self._normalize_image(image[z, :, :])
            mask, _ = self.segment_image(img_slice)
            if mask is not None:
                # Offset labels to make them unique across slices
                mask[mask > 0] += z * 10000
                masks_3d[z, :, :] = mask
        
        return masks_3d


def segment_all_structures(image: np.ndarray, masks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Identify different cellular structures from segmented masks
    
    Args:
        image: Original image
        masks: Segmentation masks
    
    Returns:
        Dictionary with different structure types
    """
    structures = {
        'soma': np.zeros_like(masks),
        'dendrites': np.zeros_like(masks),
        'axons': np.zeros_like(masks),
        'compartments': masks,
        'puncta': np.zeros_like(masks)
    }
    
    # Simple heuristic: larger regions are soma, smaller are puncta
    regions = measure.regionprops(masks)
    
    if len(regions) > 0:
        areas = [r.area for r in regions]
        area_threshold_high = np.percentile(areas, 75)
        area_threshold_low = np.percentile(areas, 25)
        
        for region in regions:
            label = region.label
            if region.area > area_threshold_high:
                structures['soma'][masks == label] = label
            elif region.area < area_threshold_low:
                structures['puncta'][masks == label] = label
            else:
                structures['dendrites'][masks == label] = label
    
    return structures


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing module...")
    
    # Test directory scanning
    handler = DirectoryHandler("/mnt/d/5TH_SEM/CELLULAR/input")
    files = handler.scan_directory()
    print(f"Found {len(files)} files")
    
    if len(files) > 0:
        # Test loading and segmentation
        loader = TIFFLoader()
        img = loader.load_tiff(files[0])
        
        if img is not None:
            print(f"Loaded image shape: {img.shape}")
            
            segmenter = CellposeSegmenter()
            masks, info = segmenter.segment_image(img)
            
            if masks is not None:
                print(f"Segmentation complete. Found {info['num_cells']} cells")
