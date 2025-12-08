#!/usr/bin/env python3
"""
System Tests for Smart Traffic Management System
Validates installation and basic functionality
"""

import sys
import os
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_mark(success):
    """Return checkmark or X"""
    return "✓" if success else "✗"

def test_python_version():
    """Test Python version"""
    print("\n1. Checking Python version...")
    version = sys.version_info
    required = (3, 8)
    
    success = version >= required
    status = check_mark(success)
    print(f"   {status} Python {version.major}.{version.minor}.{version.micro}")
    
    if not success:
        print(f"   Required: Python {required[0]}.{required[1]}+")
    
    return success

def test_sumo_installation():
    """Test SUMO installation"""
    print("\n2. Checking SUMO installation...")
    
    # Check SUMO_HOME
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print(f"   ✗ SUMO_HOME not set")
        return False
    
    print(f"   ✓ SUMO_HOME: {sumo_home}")
    
    # Check sumo binary
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"   ✓ SUMO installed: {version}")
            return True
        else:
            print(f"   ✗ SUMO not working properly")
            return False
    except:
        print(f"   ✗ SUMO binary not found")
        return False

def test_python_packages():
    """Test required Python packages"""
    print("\n3. Checking Python packages...")
    
    packages = {
        'traci': 'TraCI (SUMO interface)',
        'numpy': 'NumPy (numerical computing)',
        'pandas': 'Pandas (data analysis)',
        'sqlite3': 'SQLite3 (database)'
    }
    
    all_installed = True
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"   ✓ {description}")
        except ImportError:
            print(f"   ✗ {description} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def test_database_files():
    """Test database schema files"""
    print("\n4. Checking database files...")
    
    schema_file = 'database/schemas/complete_schema.sql'
    
    if os.path.exists(schema_file):
        print(f"   ✓ Database schema found")
        return True
    else:
        print(f"   ✗ Database schema not found: {schema_file}")
        return False

def test_sumo_files():
    """Test SUMO configuration files"""
    print("\n5. Checking SUMO files...")
    
    files = {
        'sumo/networks/city_network.net.xml': 'Network file',
        'sumo/routes/vehicles.rou.xml': 'Routes file',
        'sumo/scenarios/basic_traffic.sumocfg': 'Config file'
    }
    
    all_exist = True
    
    for file_path, description in files.items():
        if os.path.exists(file_path):
            print(f"   ✓ {description}")
        else:
            print(f"   ✗ {description} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_python_modules():
    """Test Python module files"""
    print("\n6. Checking Python modules...")
    
    modules = {
        'python/core/adaptive_signals.py': 'Adaptive signals',
        'python/core/emergency_priority.py': 'Emergency priority',
        'python/core/v2x_communication.py': 'V2X communication',
        'python/core/siot_trust.py': 'SIoT trust'
    }
    
    all_exist = True
    
    for file_path, description in modules.items():
        if os.path.exists(file_path):
            print(f"   ✓ {description}")
        else:
            print(f"   ✗ {description} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_configuration():
    """Test configuration file"""
    print("\n7. Checking configuration...")
    
    config_file = 'configs/scenario_config.json'
    
    if not os.path.exists(config_file):
        print(f"   ✗ Config file not found")
        return False
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_keys = ['simulation', 'adaptive_signals', 'v2x_communication']
        for key in required_keys:
            if key in config:
                print(f"   ✓ {key} configured")
            else:
                print(f"   ✗ {key} not configured")
                return False
        
        return True
    except Exception as e:
        print(f"   ✗ Error reading config: {e}")
        return False

def test_basic_simulation():
    """Test basic SUMO simulation"""
    print("\n8. Testing basic simulation...")
    
    try:
        # Try to start SUMO in headless mode
        cmd = [
            'sumo',
            '-c', 'sumo/scenarios/basic_traffic.sumocfg',
            '--end', '10',  # Run for 10 seconds only
            '--no-step-log',
            '--no-warnings'
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode == 0:
            print(f"   ✓ Basic simulation runs successfully")
            return True
        else:
            print(f"   ✗ Simulation failed")
            print(f"   Error: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ✗ Simulation timed out")
        return False
    except Exception as e:
        print(f"   ✗ Error running simulation: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print_header("SMART TRAFFIC MANAGEMENT SYSTEM - INSTALLATION TEST")
    
    tests = [
        ("Python Version", test_python_version),
        ("SUMO Installation", test_sumo_installation),
        ("Python Packages", test_python_packages),
        ("Database Files", test_database_files),
        ("SUMO Files", test_sumo_files),
        ("Python Modules", test_python_modules),
        ("Configuration", test_configuration),
        ("Basic Simulation", test_basic_simulation)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n   ✗ Test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    print()
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = check_mark(result)
        print(f"  {symbol} {name}: {status}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nYou can now run:")
        print("  python run_simulation.py --duration 300")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nRefer to INSTALL.md for installation instructions.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
