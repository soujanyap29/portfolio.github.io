"""
Test Structure Module
Debug and verification scripts for the protein localization pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    errors = []
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        errors.append(f"pandas: {e}")
    
    try:
        import tifffile
        print("✓ tifffile")
    except ImportError as e:
        errors.append(f"tifffile: {e}")
    
    try:
        import cv2
        print("✓ opencv")
    except ImportError as e:
        errors.append(f"opencv: {e}")
    
    try:
        from skimage import measure
        print("✓ scikit-image")
    except ImportError as e:
        errors.append(f"scikit-image: {e}")
    
    try:
        import torch
        print(f"✓ pytorch ({torch.__version__})")
    except ImportError as e:
        errors.append(f"pytorch: {e}")
    
    try:
        import networkx
        print("✓ networkx")
    except ImportError as e:
        errors.append(f"networkx: {e}")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        errors.append(f"matplotlib: {e}")
    
    try:
        import seaborn
        print("✓ seaborn")
    except ImportError as e:
        errors.append(f"seaborn: {e}")
    
    try:
        import streamlit
        print("✓ streamlit")
    except ImportError as e:
        errors.append(f"streamlit: {e}")
    
    # Optional imports
    try:
        from cellpose import models
        print("✓ cellpose")
    except ImportError:
        print("⚠ cellpose (optional)")
    
    try:
        import torch_geometric
        print("✓ torch_geometric")
    except ImportError:
        print("⚠ torch_geometric (optional)")
    
    try:
        import dgl
        print("✓ dgl")
    except ImportError:
        print("⚠ dgl (optional)")
    
    if errors:
        print("\n❌ Some imports failed:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✓ All imports successful!")
        return True


def test_modules():
    """Test that local modules can be imported."""
    print("\nTesting local modules...")
    
    errors = []
    
    try:
        from tiff_loader import TIFFLoader
        print("✓ tiff_loader")
    except Exception as e:
        errors.append(f"tiff_loader: {e}")
    
    try:
        from preprocessing import ImagePreprocessor
        print("✓ preprocessing")
    except Exception as e:
        errors.append(f"preprocessing: {e}")
    
    try:
        from graph_construction import GraphConstructor
        print("✓ graph_construction")
    except Exception as e:
        errors.append(f"graph_construction: {e}")
    
    try:
        from model_training import ModelTrainer
        print("✓ model_training")
    except Exception as e:
        errors.append(f"model_training: {e}")
    
    try:
        from visualization import Visualizer
        print("✓ visualization")
    except Exception as e:
        errors.append(f"visualization: {e}")
    
    try:
        from pipeline import ProteinLocalizationPipeline
        print("✓ pipeline")
    except Exception as e:
        errors.append(f"pipeline: {e}")
    
    if errors:
        print("\n❌ Some modules failed:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✓ All modules imported successfully!")
        return True


def test_tiff_loader():
    """Test TIFF loader with synthetic data."""
    print("\nTesting TIFF loader...")
    
    try:
        from tiff_loader import TIFFLoader
        import tempfile
        import tifffile
        
        # Create temporary directory with test TIFF
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test TIFF
            test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            test_path = Path(tmpdir) / "test.tif"
            tifffile.imwrite(test_path, test_image)
            
            # Test loader
            loader = TIFFLoader(tmpdir, recursive=True)
            loader.scan_directory()
            
            if len(loader.tiff_files) == 1:
                print("✓ TIFF scanning works")
            else:
                print(f"❌ Expected 1 file, found {len(loader.tiff_files)}")
                return False
            
            # Test loading
            data = loader.load_all()
            if len(data) == 1:
                image, metadata = data[0]
                if image.shape == (100, 100):
                    print("✓ TIFF loading works")
                    return True
                else:
                    print(f"❌ Wrong image shape: {image.shape}")
                    return False
            else:
                print(f"❌ Expected 1 loaded file, got {len(data)}")
                return False
                
    except Exception as e:
        print(f"❌ TIFF loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing with synthetic data."""
    print("\nTesting preprocessing...")
    
    try:
        from preprocessing import ImagePreprocessor
        
        # Create synthetic image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test preprocessor
        preprocessor = ImagePreprocessor(use_gpu=False)
        masks, features, info = preprocessor.process_image(test_image)
        
        if masks is not None and len(features) > 0:
            print(f"✓ Preprocessing works (found {len(features)} regions)")
            return True
        else:
            print("❌ Preprocessing failed to segment image")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_construction():
    """Test graph construction with synthetic data."""
    print("\nTesting graph construction...")
    
    try:
        from graph_construction import GraphConstructor
        
        # Create synthetic features
        features = pd.DataFrame({
            'label': [1, 2, 3, 4, 5],
            'centroid_x': [10, 20, 30, 40, 50],
            'centroid_y': [10, 20, 30, 40, 50],
            'area': [100, 150, 120, 180, 140],
            'perimeter': [40, 50, 45, 55, 48],
        })
        
        # Test graph construction
        constructor = GraphConstructor(distance_threshold=50, k_neighbors=2)
        G = constructor.build_spatial_graph(features, method='knn')
        
        if G.number_of_nodes() == 5:
            print(f"✓ Graph construction works ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            return True
        else:
            print(f"❌ Expected 5 nodes, got {G.number_of_nodes()}")
            return False
            
    except Exception as e:
        print(f"❌ Graph construction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization with synthetic data."""
    print("\nTesting visualization...")
    
    try:
        from visualization import Visualizer
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = Visualizer(output_dir=tmpdir)
            
            # Test simple plot
            test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            test_masks = np.random.randint(0, 5, (100, 100), dtype=np.int32)
            
            visualizer.plot_segmentation_overlay(test_image, test_masks, save_name="test")
            
            # Check if file was created
            output_file = Path(tmpdir) / "test.png"
            if output_file.exists():
                print("✓ Visualization works")
                return True
            else:
                print("❌ Visualization did not create output file")
                return False
                
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Check if required directories exist."""
    print("\nTesting directory structure...")
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    required_dirs = [
        'scripts',
        'frontend',
        'docs',
        'models',
        'output'
    ]
    
    missing = []
    for dir_name in required_dirs:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (missing)")
            missing.append(dir_name)
    
    if missing:
        print(f"\n⚠ Missing directories: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All required directories exist")
        return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("PROTEIN LOCALIZATION PIPELINE - STRUCTURE TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Imports", test_imports()))
    results.append(("Modules", test_modules()))
    results.append(("TIFF Loader", test_tiff_loader()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Graph Construction", test_graph_construction()))
    results.append(("Visualization", test_visualization()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print("\n" + "=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
