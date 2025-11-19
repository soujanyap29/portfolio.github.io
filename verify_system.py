"""
Quick verification script for Protein Sub-Cellular Localization System
Checks that all files are in place without requiring dependencies
"""
import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {os.path.basename(filepath)} ({size} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        files = os.listdir(dirpath)
        print(f"✓ {description}: {dirpath} ({len(files)} items)")
        return True
    else:
        print(f"✗ {description}: {dirpath} NOT FOUND")
        return False

def main():
    print("="*70)
    print("PROTEIN SUB-CELLULAR LOCALIZATION SYSTEM - VERIFICATION")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True
    
    # Check core files
    print("\n📋 Core Files:")
    all_ok &= check_file(os.path.join(base_dir, "README.md"), "README")
    all_ok &= check_file(os.path.join(base_dir, "requirements.txt"), "Requirements")
    all_ok &= check_file(os.path.join(base_dir, "JOURNAL_DOCUMENT.md"), "Journal Document")
    all_ok &= check_file(os.path.join(base_dir, ".gitignore"), "Git Ignore")
    all_ok &= check_file(os.path.join(base_dir, "demo.py"), "Demo Script")
    
    # Check backend modules
    print("\n🔧 Backend Modules:")
    backend_dir = os.path.join(base_dir, "output", "backend")
    all_ok &= check_file(os.path.join(backend_dir, "config.py"), "Configuration")
    all_ok &= check_file(os.path.join(backend_dir, "image_processor.py"), "Image Processor")
    all_ok &= check_file(os.path.join(backend_dir, "segmentation.py"), "Segmentation")
    all_ok &= check_file(os.path.join(backend_dir, "cnn_classifier.py"), "CNN Classifier")
    all_ok &= check_file(os.path.join(backend_dir, "gnn_classifier.py"), "GNN Classifier")
    all_ok &= check_file(os.path.join(backend_dir, "evaluation.py"), "Evaluation")
    all_ok &= check_file(os.path.join(backend_dir, "visualization.py"), "Visualization")
    all_ok &= check_file(os.path.join(backend_dir, "pipeline.py"), "Pipeline")
    
    # Check frontend
    print("\n🌐 Frontend:")
    frontend_dir = os.path.join(base_dir, "output", "frontend")
    all_ok &= check_file(os.path.join(frontend_dir, "app.py"), "Flask App")
    all_ok &= check_file(os.path.join(frontend_dir, "templates", "index.html"), "HTML Template")
    all_ok &= check_directory(os.path.join(frontend_dir, "static"), "Static Assets")
    
    # Check output directories
    print("\n📁 Output Directories:")
    results_dir = os.path.join(base_dir, "output", "results")
    all_ok &= check_directory(os.path.join(results_dir, "segmented"), "Segmented")
    all_ok &= check_directory(os.path.join(results_dir, "predictions"), "Predictions")
    all_ok &= check_directory(os.path.join(results_dir, "reports"), "Reports")
    all_ok &= check_directory(os.path.join(results_dir, "graphs"), "Graphs")
    
    print("\n" + "="*70)
    if all_ok:
        print("✅ ALL COMPONENTS VERIFIED SUCCESSFULLY!")
        print("="*70)
        print("\n📚 Documentation:")
        print("   - README.md: Complete usage guide")
        print("   - JOURNAL_DOCUMENT.md: Scientific paper (42KB)")
        
        print("\n🚀 To Run the System:")
        print("   1. Install dependencies:")
        print("      pip install -r requirements.txt")
        print("\n   2. Start web server:")
        print("      cd output/frontend")
        print("      python app.py")
        print("\n   3. Open browser:")
        print("      http://localhost:5000")
        
        print("\n🧪 To Run Demo:")
        print("   python demo.py")
        print("   (requires dependencies installed)")
        
        print("\n📊 System Features:")
        print("   ✓ VGG16-based CNN classifier")
        print("   ✓ Graph Neural Network (GNN)")
        print("   ✓ SLIC superpixel segmentation")
        print("   ✓ Model fusion (weighted combination)")
        print("   ✓ Publication-quality visualizations (300+ DPI)")
        print("   ✓ Web interface for image upload")
        print("   ✓ Batch processing capability")
        print("   ✓ Comprehensive metrics (accuracy, precision, recall, F1)")
        print("   ✓ Complete journal documentation")
        
        print("\n📖 Localization Classes:")
        print("   1. Nucleus")
        print("   2. Cytoplasm")
        print("   3. Mitochondria")
        print("   4. Endoplasmic Reticulum")
        print("   5. Membrane")
        
        return 0
    else:
        print("❌ SOME COMPONENTS ARE MISSING")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
