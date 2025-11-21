"""
Protein Localization Pipeline - Core Modules

This package contains the core modules for the protein localization pipeline.
"""

__version__ = "1.0.0"

# Import main classes for easy access
try:
    from .tiff_loader import TIFFLoader, load_tiff_from_path
    from .preprocessing import ImagePreprocessor
    from .graph_construction import GraphConstructor
    from .model_training import ModelTrainer, GraphDataset
    from .visualization import Visualizer
    
    __all__ = [
        'TIFFLoader',
        'load_tiff_from_path',
        'ImagePreprocessor',
        'GraphConstructor',
        'ModelTrainer',
        'GraphDataset',
        'Visualizer',
    ]
except ImportError:
    # If dependencies are not installed, fail gracefully
    pass
