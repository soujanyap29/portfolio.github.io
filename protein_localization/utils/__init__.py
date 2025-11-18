"""
Utility functions for the protein localization pipeline
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_tiff_files(root_dir: str, recursive: bool = True) -> list:
    """
    Recursively find all TIFF files in directory
    
    Args:
        root_dir: Root directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of paths to TIFF files
    """
    tiff_files = []
    extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    
    if recursive:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    tiff_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(root_dir):
            if any(file.endswith(ext) for ext in extensions):
                tiff_files.append(os.path.join(root_dir, file))
    
    return sorted(tiff_files)


def validate_tiff_file(file_path: str) -> bool:
    """Validate that a file is a valid TIFF file"""
    try:
        import tifffile
        with tifffile.TiffFile(file_path) as tif:
            return len(tif.pages) > 0
    except Exception:
        return False
