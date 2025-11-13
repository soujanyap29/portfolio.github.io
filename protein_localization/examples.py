"""
Example usage script demonstrating the pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import ProteinLocalizationPipeline


def example_preprocessing():
    """Example: Run preprocessing on sample data"""
    print("\n" + "="*70)
    print("EXAMPLE: Preprocessing Pipeline")
    print("="*70)
    
    # Initialize pipeline
    pipeline = ProteinLocalizationPipeline('config.yaml')
    
    # Run preprocessing
    # Note: Replace 'data/raw' with your actual data directory
    pipeline.run_preprocessing_pipeline(input_dir='data/raw')
    
    print("\nPreprocessing completed!")
    print("Next: Train the model using:")
    print("  python train.py --data_dir data/graphs --epochs 50")


def example_full_pipeline():
    """Example: Run full pipeline with trained model"""
    print("\n" + "="*70)
    print("EXAMPLE: Full Pipeline")
    print("="*70)
    
    # Initialize pipeline
    pipeline = ProteinLocalizationPipeline('config.yaml')
    
    # Run full pipeline
    # Note: Requires a trained model
    pipeline.run_full_pipeline(
        input_dir='data/raw',
        model_path='outputs/models/best_model.pth'
    )
    
    print("\nFull pipeline completed!")


def example_single_image():
    """Example: Process a single image"""
    print("\n" + "="*70)
    print("EXAMPLE: Single Image Processing")
    print("="*70)
    
    import numpy as np
    from utils.data_loader import load_tiff
    from utils.preprocessor import preprocess_image
    from utils.graph_builder import build_graph_from_image
    
    # Load image
    image_path = "data/raw/sample.tif"  # Replace with your image
    print(f"Loading image from {image_path}...")
    
    try:
        image = load_tiff(image_path)
        print(f"✓ Image loaded: shape={image.shape}, dtype={image.dtype}")
        
        # Preprocess
        print("Preprocessing...")
        processed = preprocess_image(image)
        print(f"✓ Preprocessed: shape={processed.shape}")
        
        # Build graph
        print("Building graph...")
        graph = build_graph_from_image(processed)
        print(f"✓ Graph built: {graph['num_nodes']} nodes, {graph['edges'].shape[1]} edges")
        
        print("\nSingle image processing completed!")
        
    except FileNotFoundError:
        print(f"✗ File not found: {image_path}")
        print("Please provide a valid TIFF image path")


def main():
    """Main example runner"""
    print("\n" + "="*70)
    print("Protein Sub-Cellular Localization - Examples")
    print("="*70)
    print("\nAvailable examples:")
    print("1. Preprocessing pipeline")
    print("2. Full pipeline (requires trained model)")
    print("3. Single image processing")
    print("4. Exit")
    
    choice = input("\nSelect example (1-4): ").strip()
    
    if choice == '1':
        example_preprocessing()
    elif choice == '2':
        example_full_pipeline()
    elif choice == '3':
        example_single_image()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
