#!/usr/bin/env python3
"""
OpenStreetMap to SUMO Network Converter
Part of Smart Traffic Management System

Converts OSM map data to SUMO network format.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_sumo_installation():
    """Verify SUMO is installed and SUMO_HOME is set"""
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("Error: SUMO_HOME environment variable not set")
        print("Please install SUMO and set SUMO_HOME")
        return False
    
    netconvert = os.path.join(sumo_home, 'bin', 'netconvert')
    if not os.path.exists(netconvert):
        netconvert = 'netconvert'  # Try system PATH
    
    try:
        subprocess.run([netconvert, '--version'], 
                      capture_output=True, check=True)
        return True
    except:
        print("Error: netconvert not found")
        return False

def convert_osm_to_sumo(osm_file, output_dir=None, options=None):
    """
    Convert OSM file to SUMO network.
    
    Args:
        osm_file: Path to OSM file
        output_dir: Output directory (default: same as input)
        options: Additional netconvert options
    """
    if not os.path.exists(osm_file):
        print(f"Error: OSM file not found: {osm_file}")
        return False
    
    # Determine output directory and file names
    if output_dir is None:
        output_dir = os.path.dirname(osm_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(osm_file).stem
    net_file = os.path.join(output_dir, f"{base_name}.net.xml")
    
    # Build netconvert command
    cmd = [
        'netconvert',
        '--osm-files', osm_file,
        '-o', net_file,
        '--geometry.remove',
        '--ramps.guess',
        '--junctions.join',
        '--tls.guess-signals',
        '--tls.discard-simple',
        '--tls.join',
        '--tls.default-type', 'actuated',
        '--roundabouts.guess',
        '--remove-edges.isolated',
        '--no-internal-links', 'false',
        '--no-turnarounds', 'false',
        '--junctions.corner-detail', '5',
        '--output.street-names',
        '--output.original-names'
    ]
    
    # Add custom options
    if options:
        cmd.extend(options)
    
    print(f"Converting {osm_file} to SUMO network...")
    print(f"Output: {net_file}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Conversion successful!")
            print(f"Network file created: {net_file}")
            
            # Generate additional files
            generate_additional_files(net_file, output_dir, base_name)
            
            return True
        else:
            print("✗ Conversion failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def generate_additional_files(net_file, output_dir, base_name):
    """Generate route and configuration files"""
    
    print("\nGenerating additional files...")
    
    # 1. Generate random trips
    route_file = os.path.join(output_dir, f"{base_name}.rou.xml")
    trips_file = os.path.join(output_dir, f"{base_name}.trips.xml")
    
    try:
        # Find randomTrips.py
        sumo_home = os.environ.get('SUMO_HOME')
        random_trips = os.path.join(sumo_home, 'tools', 'randomTrips.py')
        
        if not os.path.exists(random_trips):
            print("Warning: randomTrips.py not found, skipping route generation")
        else:
            cmd = [
                'python', random_trips,
                '-n', net_file,
                '-o', trips_file,
                '-e', '3600',
                '--fringe-factor', '100',
                '--min-distance', '300'
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ Generated trips: {trips_file}")
            
            # Convert trips to routes
            cmd = [
                'duarouter',
                '-n', net_file,
                '-r', trips_file,
                '-o', route_file,
                '--ignore-errors',
                '--no-warnings'
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ Generated routes: {route_file}")
    except Exception as e:
        print(f"Warning: Route generation failed: {e}")
    
    # 2. Generate SUMO config file
    config_file = os.path.join(output_dir, f"{base_name}.sumocfg")
    
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{base_name}.net.xml"/>
        <route-files value="{base_name}.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    
    <processing>
        <time-to-teleport value="300"/>
    </processing>
    
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Generated config: {config_file}")

def download_osm_area(place_name, output_file):
    """
    Download OSM data for a specific place using Overpass API.
    
    Args:
        place_name: Name of place (e.g., "Manhattan, New York")
        output_file: Output OSM file path
    """
    print(f"Downloading OSM data for: {place_name}")
    print("This feature requires overpy package:")
    print("  pip install overpy")
    
    try:
        import overpy
    except ImportError:
        print("Error: overpy not installed")
        return False
    
    api = overpy.Overpass()
    
    # Query for road network
    query = f"""
    [out:xml][timeout:300];
    area[name="{place_name}"]->.searchArea;
    (
        way["highway"](area.searchArea);
        node(w);
    );
    out body;
    """
    
    try:
        result = api.query(query)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(result.toXML())
        
        print(f"✓ Downloaded OSM data: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error downloading OSM data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenStreetMap data to SUMO network format'
    )
    
    parser.add_argument(
        'input',
        help='Input OSM file or place name (with --download)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory (default: same as input file)'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download OSM data for specified place'
    )
    
    parser.add_argument(
        '--no-traffic-lights',
        action='store_true',
        help='Disable traffic light generation'
    )
    
    parser.add_argument(
        '--plain-output',
        action='store_true',
        help='Also generate plain XML outputs'
    )
    
    args = parser.parse_args()
    
    # Check SUMO installation
    if not check_sumo_installation():
        sys.exit(1)
    
    # Handle download mode
    if args.download:
        osm_file = args.input.replace(' ', '_') + '.osm'
        if not download_osm_area(args.input, osm_file):
            sys.exit(1)
    else:
        osm_file = args.input
    
    # Build options
    options = []
    if args.no_traffic_lights:
        options.extend(['--tls.discard-loaded', '--tls.discard-simple'])
    
    if args.plain_output:
        plain_prefix = os.path.join(
            args.output_dir or os.path.dirname(osm_file),
            Path(osm_file).stem
        )
        options.extend(['--plain-output-prefix', plain_prefix])
    
    # Convert
    if convert_osm_to_sumo(osm_file, args.output_dir, options):
        print("\n✓ All done!")
        print("\nYou can now run the simulation with:")
        net_name = Path(osm_file).stem
        print(f"  sumo-gui -c {net_name}.sumocfg")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
