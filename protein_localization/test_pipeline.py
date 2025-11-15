#!/usr/bin/env python3
"""
Test script to verify pipeline installation and basic functionality
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from preprocessing import segmentation, feature_extraction
        print("✓ Preprocessing modules")
    except Exception as e:
        print(f"✗ Preprocessing modules: {e}")
        return False
    
    try:
        from graph_construction import graph_builder
        print("✓ Graph construction module")
    except Exception as e:
        print(f"✗ Graph construction module: {e}")
        return False
    
    try:
        from models import graph_cnn, vgg16, combined_model, trainer
        print("✓ Model modules")
    except Exception as e:
        print(f"✗ Model modules: {e}")
        return False
    
    try:
        from visualization import plotters, graph_viz, metrics
        print("✓ Visualization modules")
    except Exception as e:
        print(f"✗ Visualization modules: {e}")
        return False
    
    try:
        from interface import app
        print("✓ Interface module")
    except Exception as e:
        print(f"✗ Interface module: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with dummy data"""
    print("\nTesting basic functionality...")
    
    import numpy as np
    import torch
    
    try:
        # Test feature extraction
        from preprocessing.feature_extraction import FeatureExtractor
        
        dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        dummy_masks = np.zeros((100, 100), dtype=int)
        dummy_masks[20:40, 20:40] = 1
        dummy_masks[60:80, 60:80] = 2
        
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(dummy_image, dummy_masks)
        
        print(f"✓ Feature extraction: {len(features)} regions, {len(features.columns)} features")
    except Exception as e:
        print(f"✗ Feature extraction: {e}")
        return False
    
    try:
        # Test graph construction
        from graph_construction.graph_builder import GraphConstructor
        
        constructor = GraphConstructor()
        graph = constructor.construct_graph(features, dummy_masks)
        
        print(f"✓ Graph construction: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"✗ Graph construction: {e}")
        return False
    
    try:
        # Test model creation
        from models.graph_cnn import GraphCNN
        
        model = GraphCNN(in_channels=10, hidden_channels=32, out_channels=5, num_layers=2)
        print(f"✓ Model creation: {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"✗ Model creation: {e}")
        return False
    
    return True


def test_pytorch():
    """Test PyTorch availability"""
    print("\nTesting PyTorch...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    return True


def test_dependencies():
    """Test key dependencies"""
    print("\nTesting dependencies...")
    
    dependencies = [
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'seaborn',
        'networkx',
        'tifffile',
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} not installed")
            all_ok = False
    
    # Optional dependencies
    optional_deps = [
        'cellpose',
        'torch_geometric',
        'dgl',
        'gradio',
    ]
    
    print("\nOptional dependencies:")
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"○ {dep} not installed (optional)")
    
    return all_ok


def main():
    """Run all tests"""
    print("="*60)
    print("PROTEIN LOCALIZATION PIPELINE - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n✅ All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python main.py interface")
        print("  2. Open: http://localhost:7860")
        print("  3. Upload a TIFF file and process!")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
