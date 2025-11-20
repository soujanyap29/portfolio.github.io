#!/usr/bin/env python3
"""
Startup script for the Protein Localization Web Interface.
This script checks dependencies and starts the Flask server.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'flask',
        'flask_cors',
        'numpy',
        'tifffile',
        'scikit-image',
        'scipy',
        'cellpose',
        'matplotlib',
        'networkx',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Default directories
    input_dir = "/mnt/d/5TH_SEM/CELLULAR/input"
    output_dir = "/mnt/d/5TH_SEM/CELLULAR/output"
    
    print(f"Input directory: {input_dir}")
    if not os.path.exists(input_dir):
        print(f"⚠️  Directory does not exist, creating it...")
        os.makedirs(input_dir, exist_ok=True)
        print(f"✅ Created: {input_dir}")
    else:
        print(f"✅ Exists")
    
    print(f"\nOutput directory: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"⚠️  Directory does not exist, creating it...")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/segmented", exist_ok=True)
        os.makedirs(f"{output_dir}/predictions", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        os.makedirs(f"{output_dir}/graphs", exist_ok=True)
        print(f"✅ Created: {output_dir}")
    else:
        print(f"✅ Exists")
    
    return True

def start_server():
    """Start the Flask server."""
    print("\n" + "="*60)
    print("Starting Flask Server...")
    print("="*60)
    print("\n🌐 Server will be available at: http://localhost:5000")
    print("📝 Press Ctrl+C to stop the server\n")
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    os.chdir(frontend_dir)
    
    # Start Flask app
    from frontend.app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

def main():
    """Main function."""
    print("="*60)
    print("Protein Sub-Cellular Localization - Web Interface")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)
    
    # Check directories
    check_directories()
    
    # Start server
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
