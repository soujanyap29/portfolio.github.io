#!/usr/bin/env python3
"""
Setup script for Protein Sub-Cellular Localization project.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directories")
    
    directories = [
        "/mnt/d/5TH_SEM/CELLULAR/input",
        "/mnt/d/5TH_SEM/CELLULAR/output",
        "/mnt/d/5TH_SEM/CELLULAR/output/segmented",
        "/mnt/d/5TH_SEM/CELLULAR/output/predictions",
        "/mnt/d/5TH_SEM/CELLULAR/output/reports",
        "/mnt/d/5TH_SEM/CELLULAR/output/graphs",
        "backend/models/saved"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create {directory}: {e}")
    
    return True


def install_dependencies(requirements_file="requirements.txt"):
    """Install Python dependencies."""
    print_header("Installing Dependencies")
    
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("\n✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False


def verify_imports():
    """Verify that key packages can be imported."""
    print_header("Verifying Imports")
    
    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "sklearn",
        "cv2",
        "flask",
        "networkx"
    ]
    
    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - failed to import")
            all_ok = False
    
    # Check optional packages
    print("\nOptional packages:")
    optional = ["cellpose", "torch_geometric"]
    for package in optional:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - not installed (optional)")
    
    return all_ok


def run_tests():
    """Run basic tests."""
    print_header("Running Basic Tests")
    
    try:
        # Test image preprocessing
        sys.path.append('backend')
        from utils.image_preprocessing import TIFFLoader
        
        loader = TIFFLoader()
        print("✅ Image preprocessing module OK")
        
        # Test visualization
        from utils.visualization import Visualizer
        visualizer = Visualizer(output_dir="/tmp/test_viz")
        print("✅ Visualization module OK")
        
        return True
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False


def display_next_steps():
    """Display next steps for the user."""
    print_header("Setup Complete!")
    
    print("""
Next Steps:
    
1. Add TIFF microscopy images to:
   /mnt/d/5TH_SEM/CELLULAR/input/
    
2. Run the automated pipeline:
   cd notebooks
   jupyter notebook automated_pipeline.ipynb
    
3. Or start the web interface:
   cd frontend
   python app.py
    
4. Access the web interface at:
   http://localhost:5000
    
For more information, see README.md
    
""")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup Protein Sub-Cellular Localization project"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip verification tests"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  Protein Sub-Cellular Localization - Setup")
    print("  Student: Soujanya")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            print("\n⚠️  Warning: Some dependencies failed to install")
            print("You may need to install them manually")
    else:
        print("\nSkipping dependency installation (--skip-install)")
    
    # Verify imports
    if not args.skip_tests:
        verify_imports()
        
        # Run tests
        run_tests()
    else:
        print("\nSkipping tests (--skip-tests)")
    
    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    main()
