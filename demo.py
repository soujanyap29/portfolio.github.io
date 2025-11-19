"""
Demo script for Protein Sub-Cellular Localization System
This creates mock data and demonstrates the system functionality
"""
import numpy as np
import os
import sys
from PIL import Image

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'output', 'backend'))

def create_mock_tiff_image(filepath, size=(512, 512)):
    """
    Create a mock TIFF image for testing
    
    Args:
        filepath: Output path for TIFF file
        size: Image dimensions
    """
    # Create synthetic microscopy-like image
    img = np.random.rand(*size) * 100 + 50  # Base intensity
    
    # Add some structures (mock cellular compartments)
    y, x = np.ogrid[:size[0], :size[1]]
    
    # Mock nucleus (bright center)
    center = (size[0]//2, size[1]//2)
    nucleus_mask = (x - center[1])**2 + (y - center[0])**2 < (size[0]//4)**2
    img[nucleus_mask] += 100
    
    # Mock mitochondria (scattered bright spots)
    for _ in range(10):
        cx, cy = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        mito_mask = (x - cy)**2 + (y - cx)**2 < 20**2
        img[mito_mask] += 80
    
    # Normalize to 8-bit
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Save as TIFF
    Image.fromarray(img).save(filepath, format='TIFF')
    print(f"Created mock TIFF image: {filepath}")


def demo_single_image_analysis():
    """Demonstrate single image analysis"""
    print("\n" + "="*70)
    print("DEMO: Single Image Analysis")
    print("="*70)
    
    # Create mock data
    test_dir = "/tmp/test_protein_localization"
    os.makedirs(test_dir, exist_ok=True)
    
    test_image_path = os.path.join(test_dir, "test_neuron.tif")
    create_mock_tiff_image(test_image_path)
    
    try:
        from pipeline import ProteinLocalizationPipeline
        
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = ProteinLocalizationPipeline()
        
        # Analyze image
        print("\nAnalyzing mock image...")
        print("Note: Since models are not trained, this will use random predictions")
        print("In production, pre-trained model weights would be loaded.")
        
        # Note: This will fail without actual trained models
        # but demonstrates the pipeline structure
        result = pipeline.analyze_single_image(test_image_path, save_results=True)
        
        print("\nAnalysis complete!")
        print(f"Predicted class: {result['fused_prediction']['class']}")
        print(f"Confidence: {result['fused_prediction']['confidence']:.2%}")
        print(f"\nFull results saved to output directory")
        
    except Exception as e:
        print(f"\nNote: Full analysis requires trained models.")
        print(f"This demo shows the system structure.")
        print(f"Error details: {e}")
        print("\nTo run the full system:")
        print("1. Train models on labeled microscopy data")
        print("2. Save model weights")
        print("3. Update config.py with model paths")
        print("4. Run the web interface: cd output/frontend && python app.py")


def demo_preprocessing():
    """Demonstrate preprocessing capabilities"""
    print("\n" + "="*70)
    print("DEMO: Image Preprocessing")
    print("="*70)
    
    from image_processor import ImageProcessor
    
    # Create mock image
    test_dir = "/tmp/test_protein_localization"
    os.makedirs(test_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "test_preprocessing.tif")
    create_mock_tiff_image(test_image_path, size=(256, 256))
    
    # Initialize processor
    processor = ImageProcessor(target_size=(224, 224))
    
    # Load and preprocess
    print("\n1. Loading TIFF image...")
    original = processor.load_tiff(test_image_path)
    print(f"   Original shape: {original.shape}, dtype: {original.dtype}")
    
    print("\n2. Preprocessing (normalize, resize, convert to RGB)...")
    preprocessed = processor.preprocess(test_image_path)
    print(f"   Preprocessed shape: {preprocessed.shape}, dtype: {preprocessed.dtype}")
    
    print("\n3. Saving processed image...")
    output_path = os.path.join(test_dir, "preprocessed.png")
    processor.save_image(preprocessed, output_path)
    print(f"   Saved to: {output_path}")
    
    print("\n✓ Preprocessing demo complete!")


def demo_segmentation():
    """Demonstrate segmentation"""
    print("\n" + "="*70)
    print("DEMO: Image Segmentation")
    print("="*70)
    
    from image_processor import ImageProcessor
    from segmentation import SegmentationEngine
    
    # Create and preprocess image
    test_dir = "/tmp/test_protein_localization"
    os.makedirs(test_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "test_segmentation.tif")
    create_mock_tiff_image(test_image_path, size=(256, 256))
    
    processor = ImageProcessor(target_size=(224, 224))
    img = processor.preprocess(test_image_path)
    
    # Segment using SLIC
    print("\nPerforming SLIC superpixel segmentation...")
    seg_engine = SegmentationEngine(method='slic')
    segments = seg_engine.segment(img, n_segments=50, compactness=10, sigma=1)
    
    print(f"Generated {segments.max()} superpixels")
    
    # Visualize
    print("\nCreating visualization...")
    vis = seg_engine.visualize_segmentation(img, segments)
    
    output_path = os.path.join(test_dir, "segmentation_vis.png")
    from PIL import Image as PILImage
    PILImage.fromarray((vis * 255).astype(np.uint8)).save(output_path)
    print(f"Visualization saved to: {output_path}")
    
    print("\n✓ Segmentation demo complete!")


def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n" + "="*70)
    print("DEMO: Scientific Visualization")
    print("="*70)
    
    from visualization import ScientificVisualizer
    import config
    
    visualizer = ScientificVisualizer(dpi=300)
    
    test_dir = "/tmp/test_protein_localization"
    os.makedirs(test_dir, exist_ok=True)
    
    # Mock probability distribution
    print("\n1. Creating probability distribution plot...")
    probabilities = [0.1, 0.15, 0.45, 0.2, 0.1]
    output_path = os.path.join(test_dir, "probabilities.png")
    visualizer.plot_probability_distribution(
        probabilities,
        config.LOCALIZATION_CLASSES,
        output_path
    )
    print(f"   Saved to: {output_path}")
    
    # Mock confusion matrix
    print("\n2. Creating confusion matrix...")
    cm = np.array([
        [45, 2, 1, 1, 1],
        [2, 48, 0, 0, 0],
        [1, 0, 42, 3, 4],
        [0, 1, 3, 44, 2],
        [2, 1, 4, 2, 41]
    ])
    output_path = os.path.join(test_dir, "confusion_matrix.png")
    visualizer.plot_confusion_matrix(
        cm,
        config.LOCALIZATION_CLASSES,
        output_path
    )
    print(f"   Saved to: {output_path}")
    
    # Mock metrics
    print("\n3. Creating metrics comparison plot...")
    metrics = {
        'accuracy': 0.92,
        'precision': 0.91,
        'recall': 0.90,
        'f1_score': 0.905,
        'specificity': 0.94
    }
    output_path = os.path.join(test_dir, "metrics.png")
    visualizer.plot_metrics_comparison(metrics, output_path)
    print(f"   Saved to: {output_path}")
    
    print("\n✓ Visualization demo complete!")


def show_system_info():
    """Display system information"""
    print("\n" + "="*70)
    print("PROTEIN SUB-CELLULAR LOCALIZATION SYSTEM")
    print("="*70)
    print("\nSystem Components:")
    print("✓ Backend Modules:")
    print("  - Image Processor (TIFF loading, preprocessing)")
    print("  - Segmentation Engine (U-Net, SLIC, Watershed)")
    print("  - CNN Classifier (VGG16-based)")
    print("  - GNN Classifier (Graph Neural Network)")
    print("  - Model Fusion (Weighted combination)")
    print("  - Evaluation Metrics (Accuracy, Precision, Recall, F1, Specificity)")
    print("  - Scientific Visualizer (Publication-quality plots)")
    print("  - Pipeline Orchestrator (End-to-end processing)")
    
    print("\n✓ Frontend:")
    print("  - Flask web application")
    print("  - Responsive HTML/CSS/JavaScript interface")
    print("  - Single image upload")
    print("  - Batch processing")
    print("  - Real-time result display")
    
    print("\n✓ Documentation:")
    print("  - README.md (Usage guide)")
    print("  - JOURNAL_DOCUMENT.md (Complete scientific paper)")
    print("  - Inline code documentation")
    
    print("\n" + "="*70)
    print("TO RUN THE WEB INTERFACE:")
    print("="*70)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Start server:")
    print("   cd output/frontend")
    print("   python app.py")
    print("\n3. Open browser:")
    print("   http://localhost:5000")
    print("="*70)


def main():
    """Run all demos"""
    show_system_info()
    
    print("\n\nRunning Demonstrations...\n")
    
    # Run demos
    demo_preprocessing()
    demo_segmentation()
    demo_visualization()
    
    # Note about full analysis
    print("\n" + "="*70)
    print("NOTE: Full Analysis Demo")
    print("="*70)
    print("The complete analysis pipeline requires trained model weights.")
    print("To train models:")
    print("1. Collect labeled microscopy dataset")
    print("2. Train VGG16 and GNN models")
    print("3. Save model weights")
    print("4. Update config.py with weight paths")
    print("\nWithout trained models, predictions will be random.")
    print("However, all other components (preprocessing, segmentation,")
    print("visualization, web interface) work fully.")
    print("="*70)
    
    print("\n\n✅ All demos completed successfully!")
    print(f"\nDemo outputs saved to: /tmp/test_protein_localization")
    print("\nNext steps:")
    print("1. Review the generated visualizations")
    print("2. Explore the code in output/backend/")
    print("3. Read JOURNAL_DOCUMENT.md for detailed documentation")
    print("4. Start the web interface to try the full system")


if __name__ == "__main__":
    main()
