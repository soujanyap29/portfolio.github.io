#!/usr/bin/env python3
"""
Test script to verify the pipeline structure and imports
This script checks the project structure without requiring external dependencies
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists and is readable"""
    if os.path.exists(filepath):
        print(f"✓ Found: {filepath}")
        return True
    else:
        print(f"✗ Missing: {filepath}")
        return False

def check_directory_structure():
    """Verify the project directory structure"""
    print("=" * 60)
    print("Checking Protein Localization Pipeline Structure")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        'scripts/tiff_loader.py',
        'scripts/preprocessing.py',
        'scripts/graph_construction.py',
        'scripts/model_training.py',
        'scripts/visualization.py',
        'scripts/pipeline.py',
        'frontend/index.html',
        'frontend/style.css',
        'frontend/app.js',
        'requirements.txt',
        'README.md',
        'docs/QUICKSTART.md'
    ]
    
    print("\nChecking files:")
    all_exist = True
    for file in required_files:
        filepath = base_dir / file
        if not check_file_exists(filepath):
            all_exist = False
    
    print("\n" + "=" * 60)
    if all_exist:
        print("✓ All required files are present!")
    else:
        print("✗ Some files are missing!")
    print("=" * 60)
    
    return all_exist

def check_script_syntax():
    """Check Python files for syntax errors"""
    print("\nChecking Python files for syntax errors:")
    
    base_dir = Path(__file__).parent.parent
    script_dir = base_dir / 'scripts'
    
    python_files = [
        'tiff_loader.py',
        'preprocessing.py',
        'graph_construction.py',
        'model_training.py',
        'visualization.py',
        'pipeline.py'
    ]
    
    all_valid = True
    for filename in python_files:
        filepath = script_dir / filename
        try:
            with open(filepath, 'r') as f:
                code = f.read()
                compile(code, filename, 'exec')
            print(f"✓ {filename}: Syntax OK")
        except SyntaxError as e:
            print(f"✗ {filename}: Syntax Error - {e}")
            all_valid = False
        except Exception as e:
            print(f"✗ {filename}: Error - {e}")
            all_valid = False
    
    return all_valid

def count_lines_of_code():
    """Count lines of code in the project"""
    print("\nCode Statistics:")
    
    base_dir = Path(__file__).parent.parent
    
    total_lines = 0
    file_stats = {}
    
    # Python files
    for py_file in (base_dir / 'scripts').glob('*.py'):
        with open(py_file, 'r') as f:
            lines = len(f.readlines())
            file_stats[py_file.name] = lines
            total_lines += lines
    
    # Frontend files
    for html_file in (base_dir / 'frontend').glob('*.html'):
        with open(html_file, 'r') as f:
            lines = len(f.readlines())
            file_stats[html_file.name] = lines
            total_lines += lines
    
    for css_file in (base_dir / 'frontend').glob('*.css'):
        with open(css_file, 'r') as f:
            lines = len(f.readlines())
            file_stats[css_file.name] = lines
            total_lines += lines
    
    for js_file in (base_dir / 'frontend').glob('*.js'):
        with open(js_file, 'r') as f:
            lines = len(f.readlines())
            file_stats[js_file.name] = lines
            total_lines += lines
    
    print(f"\nTotal lines of code: {total_lines}")
    print("\nBreakdown:")
    for filename, lines in sorted(file_stats.items()):
        print(f"  {filename:30s}: {lines:5d} lines")

def check_documentation():
    """Check documentation files"""
    print("\nChecking documentation:")
    
    base_dir = Path(__file__).parent.parent
    
    # Check README
    readme = base_dir / 'README.md'
    if readme.exists():
        size = readme.stat().st_size
        print(f"✓ README.md: {size} bytes")
    else:
        print("✗ README.md not found")
    
    # Check quickstart
    quickstart = base_dir / 'docs' / 'QUICKSTART.md'
    if quickstart.exists():
        size = quickstart.stat().st_size
        print(f"✓ QUICKSTART.md: {size} bytes")
    else:
        print("✗ QUICKSTART.md not found")

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("PROTEIN LOCALIZATION PIPELINE - STRUCTURE VERIFICATION")
    print("=" * 60 + "\n")
    
    # Check structure
    structure_ok = check_directory_structure()
    
    # Check syntax
    syntax_ok = check_script_syntax()
    
    # Count code
    count_lines_of_code()
    
    # Check docs
    check_documentation()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if structure_ok and syntax_ok:
        print("✓ All checks passed!")
        print("✓ Project structure is complete")
        print("✓ All Python files have valid syntax")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run pipeline: python scripts/pipeline.py")
        print("3. Open web interface: open frontend/index.html")
    else:
        print("✗ Some checks failed")
        print("Please review the errors above")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
