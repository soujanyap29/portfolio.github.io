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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
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
    
    def __init__(self, model_type: str = 'cyto2', diameter: int = None, channels: List[int] = [0, 0], 
                 use_gpu: bool = True, fast_mode: bool = True):
        """
        Initialize Cellpose segmenter
        
        Args:
            model_type: Type of Cellpose model ('cyto', 'cyto2', 'nuclei')
            diameter: Expected cell diameter (None for auto-detect)
            channels: Channel configuration [cytoplasm, nucleus]
            use_gpu: Whether to use GPU (default True)
            fast_mode: Use faster processing with fewer iterations (default True)
        """
        self.model_type = model_type
        self.diameter = diameter
        self.channels = channels
        self.use_gpu = use_gpu
        self.fast_mode = fast_mode
        self.model = None
        # Initialize model immediately to avoid reinitialization overhead
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Cellpose model"""
        if self.model is not None:
            return  # Already initialized
            
        try:
            self.model = models.Cellpose(model_type=self.model_type, gpu=self.use_gpu)
            print(f"Cellpose model '{self.model_type}' initialized (GPU: {self.use_gpu})")
        except Exception as e:
            print(f"Error initializing Cellpose: {str(e)}")
            print("Attempting to use CPU model...")
            self.model = models.Cellpose(model_type=self.model_type, gpu=False)
            self.use_gpu = False
    
    def segment_image(self, image: np.ndarray, extract_2d: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Segment an image using Cellpose
        
        Args:
            image: Input image array
            extract_2d: If True, extract representative 2D slice from multidimensional images (faster)
        
        Returns:
            masks: Labeled segmentation mask
            info: Dictionary with segmentation information
        """
        if self.model is None:
            self.initialize_model()
        
        try:
            # Handle different image dimensions - extract 2D slice efficiently
            if extract_2d and len(image.shape) >= 3:
                img_2d = self._extract_2d_slice(image)
            elif len(image.shape) == 2:
                img_2d = image
            else:
                img_2d = self._extract_2d_slice(image)
            
            # Normalize image
            img_2d = self._normalize_image(img_2d)
            
            # Run segmentation with optimized parameters for speed
            if self.fast_mode:
                # Faster settings: higher thresholds, fixed diameter
                masks, flows, styles, diams = self.model.eval(
                    img_2d, 
                    diameter=self.diameter if self.diameter else 30,  # Use fixed diameter to skip detection
                    channels=self.channels,
                    flow_threshold=0.6,  # Higher threshold = faster
                    cellprob_threshold=0.2,  # Higher threshold = faster
                    do_3D=False,
                    resample=False  # Skip resampling for speed
                )
            else:
                # Standard settings
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
    def _extract_2d_slice(image: np.ndarray) -> np.ndarray:
        """
        Efficiently extract a representative 2D slice from multidimensional image
        
        Args:
            image: Input image of any dimension
        
        Returns:
            2D numpy array
        """
        if len(image.shape) == 4:  # 4D TIFF (T, Z, Y, X) or (T, Y, X, C)
            # Take first timepoint, middle Z-slice or first channel
            if image.shape[1] < 20:  # Likely (T, Z, Y, X)
                return image[0, image.shape[1] // 2, :, :]
            else:  # Likely (T, Y, X, C)
                return image[0, :, :, 0]
        elif len(image.shape) == 3:
            # If it's (Z, Y, X), take middle Z-slice
            if image.shape[0] < 20:  # Likely Z-stack
                return image[image.shape[0] // 2, :, :]
            elif image.shape[2] < 20:  # Likely (Y, X, C)
                return image[:, :, 0]
            else:  # Assume (Z, Y, X)
                return image[image.shape[0] // 2, :, :]
        else:
            return image
    
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


def process_single_file(filepath: str, model_type: str = 'cyto2', 
                       fast_mode: bool = True, use_gpu: bool = False) -> Tuple[str, np.ndarray, Dict, bool]:
    """
    Process a single TIFF file (for parallel processing)
    
    Args:
        filepath: Path to TIFF file
        model_type: Cellpose model type
        fast_mode: Use fast processing mode
        use_gpu: Use GPU (False for parallel CPU processing)
    
    Returns:
        Tuple of (filepath, masks, info, success)
    """
    try:
        # Load image
        loader = TIFFLoader()
        image = loader.load_tiff(filepath)
        if image is None:
            return filepath, None, {}, False
        
        # Segment with CPU (safer for parallel processing)
        segmenter = CellposeSegmenter(model_type=model_type, fast_mode=fast_mode, use_gpu=use_gpu)
        masks, info = segmenter.segment_image(image)
        
        if masks is None:
            return filepath, None, {}, False
        
        return filepath, masks, info, True
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return filepath, None, {}, False


class BatchProcessor:
    """
    Efficient batch processor for multiple TIFF files
    """
    
    def __init__(self, model_type: str = 'cyto2', fast_mode: bool = True, 
                 use_gpu: bool = True, n_workers: int = None):
        """
        Initialize batch processor
        
        Args:
            model_type: Cellpose model type
            fast_mode: Use fast processing mode (recommended for large datasets)
            use_gpu: Use GPU for processing
            n_workers: Number of parallel workers (None = single process)
        """
        self.model_type = model_type
        self.fast_mode = fast_mode
        self.use_gpu = use_gpu
        self.n_workers = n_workers
        
        # For single-process mode, initialize one segmenter
        if n_workers is None:
            self.segmenter = CellposeSegmenter(
                model_type=model_type, 
                fast_mode=fast_mode, 
                use_gpu=use_gpu
            )
            self.loader = TIFFLoader()
    
    def process_files(self, file_list: List[str], max_files: int = None) -> Dict:
        """
        Process multiple files efficiently
        
        Args:
            file_list: List of file paths
            max_files: Maximum number of files to process (None = all)
        
        Returns:
            Dictionary with results
        """
        if max_files:
            file_list = file_list[:max_files]
        
        results = {
            'images': [],
            'masks': [],
            'info': [],
            'filenames': [],
            'success_count': 0,
            'fail_count': 0
        }
        
        if self.n_workers is None or self.n_workers == 1:
            # Single-process mode (faster for GPU)
            results = self._process_sequential(file_list)
        else:
            # Multi-process mode (better for CPU)
            results = self._process_parallel(file_list)
        
        return results
    
    def _process_sequential(self, file_list: List[str]) -> Dict:
        """Process files sequentially (GPU mode)"""
        results = {
            'images': [],
            'masks': [],
            'info': [],
            'filenames': [],
            'success_count': 0,
            'fail_count': 0
        }
        
        for filepath in tqdm(file_list, desc='Processing'):
            try:
                filename = Path(filepath).stem
                
                # Load image
                image = self.loader.load_tiff(filepath)
                if image is None:
                    results['fail_count'] += 1
                    continue
                
                # Segment
                masks, info = self.segmenter.segment_image(image)
                if masks is None:
                    results['fail_count'] += 1
                    continue
                
                results['images'].append(image)
                results['masks'].append(masks)
                results['info'].append(info)
                results['filenames'].append(filename)
                results['success_count'] += 1
                
            except Exception as e:
                print(f"Error: {Path(filepath).name}: {str(e)}")
                results['fail_count'] += 1
        
        return results
    
    def _process_parallel(self, file_list: List[str]) -> Dict:
        """Process files in parallel (CPU mode)"""
        results = {
            'images': [],
            'masks': [],
            'info': [],
            'filenames': [],
            'success_count': 0,
            'fail_count': 0
        }
        
        # Process in parallel using multiple CPU cores
        process_func = partial(
            process_single_file,
            model_type=self.model_type,
            fast_mode=self.fast_mode,
            use_gpu=False  # Disable GPU for parallel processing
        )
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_func, f): f for f in file_list}
            
            for future in tqdm(as_completed(futures), total=len(file_list), desc='Processing'):
                filepath, masks, info, success = future.result()
                
                if success:
                    # Need to reload image in main process
                    image = TIFFLoader.load_tiff(filepath)
                    results['images'].append(image)
                    results['masks'].append(masks)
                    results['info'].append(info)
                    results['filenames'].append(Path(filepath).stem)
                    results['success_count'] += 1
                else:
                    results['fail_count'] += 1
        
        return results


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing module...")
    
    # Test directory scanning
    handler = DirectoryHandler("/mnt/d/5TH_SEM/CELLULAR/input")
    files = handler.scan_directory()
    print(f"Found {len(files)} files")
    
    if len(files) > 0:
        # Test fast batch processing (recommended for large datasets)
        print("\nTesting fast batch processing (first 5 files)...")
        batch_processor = BatchProcessor(fast_mode=True, use_gpu=True)
        results = batch_processor.process_files(files[:5])
        
        print(f"\nResults:")
        print(f"  Successful: {results['success_count']}")
        print(f"  Failed: {results['fail_count']}")
        
        if results['success_count'] > 0:
            print(f"  Total cells detected: {sum(info['num_cells'] for info in results['info'])}")

