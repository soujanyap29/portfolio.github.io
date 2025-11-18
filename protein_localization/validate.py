#!/usr/bin/env python3
"""
Validation script to verify the protein localization pipeline installation.
Run this after installation to ensure everything is working correctly.
"""

import sys
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_file_structure():
    """Check if all required files exist."""
    print_section("1. Checking File Structure")
    
    required_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        'setup.py',
        'demo.py',
        'install.sh',
        'src/__init__.py',
        'src/preprocessing.py',
        'src/graph_builder.py',
        'src/models.py',
        'src/visualization.py',
        'frontend/app.py',
        'frontend/templates/index.html',
        'notebooks/final_pipeline.ipynb',
        'docs/DOCUMENTATION.md',
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_exist = False
    
    return all_exist

def check_python_syntax():
    """Check Python syntax of all modules."""
    print_section("2. Checking Python Syntax")
    
    python_files = [
        'src/__init__.py',
        'src/preprocessing.py',
        'src/graph_builder.py',
        'src/models.py',
        'src/visualization.py',
        'frontend/app.py',
        'demo.py',
    ]
    
    all_valid = True
    for file in python_files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"  ✓ {file}")
        except SyntaxError as e:
            print(f"  ✗ {file} - SYNTAX ERROR: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"  ✗ {file} - NOT FOUND")
            all_valid = False
    
    return all_valid

def check_documentation():
    """Check if documentation files are present and non-empty."""
    print_section("3. Checking Documentation")
    
    docs = {
        'README.md': 'Main documentation',
        'QUICKSTART.md': 'Quick start guide',
        'docs/DOCUMENTATION.md': 'Comprehensive docs',
    }
    
    all_good = True
    for file, desc in docs.items():
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            lines = len(path.read_text().splitlines())
            print(f"  ✓ {file} - {desc} ({lines} lines, {size} bytes)")
        else:
            print(f"  ✗ {file} - MISSING")
            all_good = False
    
    return all_good

def check_imports():
    """Try to import core modules (without dependencies)."""
    print_section("4. Checking Module Structure")
    
    sys.path.insert(0, 'src')
    
    modules = {
        'preprocessing': ['TIFFPreprocessor'],
        'graph_builder': ['BiologicalGraphBuilder'],
        'models': ['GraphCNN', 'ModelTrainer'],
        'visualization': ['ProteinVisualization'],
    }
    
    all_imports_ok = True
    for module_name, classes in modules.items():
        try:
            # Check if file exists and has valid syntax
            module_file = Path(f'src/{module_name}.py')
            if module_file.exists():
                with open(module_file) as f:
                    content = f.read()
                    # Check for class definitions
                    for cls in classes:
                        if f'class {cls}' in content:
                            print(f"  ✓ {module_name}.{cls} defined")
                        else:
                            print(f"  ✗ {module_name}.{cls} not found")
                            all_imports_ok = False
            else:
                print(f"  ✗ {module_name}.py not found")
                all_imports_ok = False
        except Exception as e:
            print(f"  ✗ Error checking {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def count_code_stats():
    """Count lines of code."""
    print_section("5. Code Statistics")
    
    categories = {
        'Core modules': ['src/preprocessing.py', 'src/graph_builder.py', 
                        'src/models.py', 'src/visualization.py'],
        'Frontend': ['frontend/app.py', 'frontend/templates/index.html'],
        'Demo': ['demo.py'],
        'Setup': ['setup.py'],
    }
    
    total_lines = 0
    for category, files in categories.items():
        cat_lines = 0
        for file in files:
            path = Path(file)
            if path.exists():
                lines = len(path.read_text().splitlines())
                cat_lines += lines
        print(f"  {category}: {cat_lines} lines")
        total_lines += cat_lines
    
    print(f"\n  Total: {total_lines} lines of code")
    return True

def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print("  PROTEIN LOCALIZATION PIPELINE - VALIDATION")
    print("=" * 70)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Python Syntax", check_python_syntax),
        ("Documentation", check_documentation),
        ("Module Structure", check_imports),
        ("Code Statistics", count_code_stats),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("  ✓ ALL CHECKS PASSED")
        print("=" * 70)
        print("\n  The pipeline is properly installed and ready to use!")
        print("\n  Next steps:")
        print("    1. Run: python demo.py")
        print("    2. Or: jupyter lab notebooks/final_pipeline.ipynb")
        print("    3. Or: cd frontend && python app.py")
        print()
        return 0
    else:
        print("  ✗ SOME CHECKS FAILED")
        print("=" * 70)
        print("\n  Please review the errors above and:")
        print("    1. Ensure all files are present")
        print("    2. Check file permissions")
        print("    3. Verify Python syntax")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
