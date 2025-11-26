#!/usr/bin/env python3
"""
Map Converter - OSM to SUMO Network Converter

Converts OpenStreetMap (.osm) files to SUMO network files (.net.xml)
and generates all necessary configuration files for simulation.

Usage:
    python map_converter.py --input maps/your_map.osm --output sumo_config/
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional, List
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MapConverter')

# Highway types to keep in the network conversion
HIGHWAY_TYPES = [
    'highway.motorway',
    'highway.motorway_link',
    'highway.trunk',
    'highway.trunk_link',
    'highway.primary',
    'highway.primary_link',
    'highway.secondary',
    'highway.secondary_link',
    'highway.tertiary',
    'highway.tertiary_link',
    'highway.residential',
    'highway.living_street',
    'highway.unclassified'
]


class MapConverter:
    """
    Converts OpenStreetMap data to SUMO simulation files.
    
    This class handles the conversion of OSM data and generation
    of all necessary SUMO configuration files.
    """
    
    def __init__(self, input_file: str, output_dir: str, network_name: str = "network"):
        """
        Initialize the map converter.
        
        Args:
            input_file: Path to input OSM file
            output_dir: Directory for output files
            network_name: Name prefix for generated files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.network_name = network_name
        
        # Check SUMO_HOME
        self.sumo_home = os.environ.get('SUMO_HOME', '')
        if not self.sumo_home:
            logger.warning("SUMO_HOME not set. Trying default paths.")
            self._find_sumo_home()
        
        # Output file paths
        self.net_file = self.output_dir / f"{network_name}.net.xml"
        self.route_file = self.output_dir / "routes.rou.xml"
        self.vtype_file = self.output_dir / "vehicles.vtype.xml"
        self.add_file = self.output_dir / "additional.add.xml"
        self.config_file = self.output_dir / "simulation.sumocfg"
        
    def _find_sumo_home(self) -> None:
        """Try to find SUMO installation."""
        common_paths = [
            '/usr/share/sumo',
            '/usr/local/share/sumo',
            'C:\\Program Files (x86)\\Eclipse\\Sumo',
            'C:\\Program Files\\Eclipse\\Sumo'
        ]
        for path in common_paths:
            if os.path.exists(path):
                self.sumo_home = path
                os.environ['SUMO_HOME'] = path
                logger.info(f"Found SUMO at: {path}")
                return
        logger.error("Could not find SUMO installation")
    
    def convert_osm_to_net(self, keep_all_routes: bool = True,
                           add_traffic_lights: bool = True) -> bool:
        """
        Convert OSM file to SUMO network using netconvert.
        
        Args:
            keep_all_routes: Whether to keep all route connections
            add_traffic_lights: Whether to generate traffic lights
            
        Returns:
            bool: True if conversion successful
        """
        logger.info(f"Converting {self.input_file} to SUMO network")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build netconvert command
        cmd = [
            'netconvert',
            '--osm-files', str(self.input_file),
            '-o', str(self.net_file),
            '--geometry.remove',
            '--roundabouts.guess',
            '--ramps.guess',
            '--junctions.join',
            '--tls.guess-signals',
            '--tls.discard-simple',
            '--tls.join',
            '--tls.default-type', 'actuated',
            '--output.street-names',
            '--output.original-names',
            '--keep-edges.by-type', ','.join(HIGHWAY_TYPES),
        ]
        
        if add_traffic_lights:
            cmd.extend(['--tls.guess', 'true'])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"netconvert failed: {result.stderr}")
                return False
            logger.info(f"Network file created: {self.net_file}")
            return True
        except FileNotFoundError:
            logger.error("netconvert not found. Please ensure SUMO is installed.")
            return False
    
    def generate_vehicle_types(self) -> None:
        """Generate vehicle type definitions."""
        logger.info("Generating vehicle type definitions")
        
        vtypes = """<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <!-- Standard car -->
    <vType id="car" 
           accel="2.6" 
           decel="4.5" 
           sigma="0.5" 
           length="4.5" 
           minGap="2.5" 
           maxSpeed="50" 
           speedFactor="1.0" 
           speedDev="0.1"
           color="1,1,0"
           guiShape="passenger"/>
    
    <!-- Bus -->
    <vType id="bus" 
           accel="1.2" 
           decel="3.0" 
           sigma="0.5" 
           length="12.0" 
           minGap="3.0" 
           maxSpeed="40" 
           speedFactor="0.9"
           color="0,0,1"
           guiShape="bus"/>
    
    <!-- Truck -->
    <vType id="truck" 
           accel="1.0" 
           decel="2.5" 
           sigma="0.5" 
           length="15.0" 
           minGap="3.5" 
           maxSpeed="35" 
           speedFactor="0.8"
           color="0.5,0.5,0.5"
           guiShape="truck"/>
    
    <!-- Two-wheeler / Motorcycle -->
    <vType id="motorcycle" 
           accel="3.0" 
           decel="5.0" 
           sigma="0.6" 
           length="2.0" 
           minGap="1.5" 
           maxSpeed="60" 
           speedFactor="1.1"
           color="1,0,0"
           guiShape="motorcycle"/>
    
    <!-- Ambulance -->
    <vType id="ambulance" 
           accel="3.5" 
           decel="5.0" 
           sigma="0.3" 
           length="6.0" 
           minGap="2.0" 
           maxSpeed="70" 
           speedFactor="1.2"
           color="1,1,1"
           guiShape="emergency"
           vClass="emergency"/>
    
    <!-- Police -->
    <vType id="police" 
           accel="3.5" 
           decel="5.0" 
           sigma="0.3" 
           length="5.0" 
           minGap="2.0" 
           maxSpeed="70" 
           speedFactor="1.2"
           color="0,0,0.8"
           guiShape="emergency"
           vClass="emergency"/>
    
    <!-- Fire truck -->
    <vType id="firetruck" 
           accel="2.5" 
           decel="4.0" 
           sigma="0.3" 
           length="10.0" 
           minGap="3.0" 
           maxSpeed="60" 
           speedFactor="1.1"
           color="1,0,0"
           guiShape="firebrigade"
           vClass="emergency"/>
    
    <!-- Auto-rickshaw (three-wheeler) -->
    <vType id="auto" 
           accel="2.0" 
           decel="3.5" 
           sigma="0.6" 
           length="3.0" 
           minGap="2.0" 
           maxSpeed="40" 
           speedFactor="0.85"
           color="0,1,0"
           guiShape="passenger/van"/>
</additional>
"""
        
        with open(self.vtype_file, 'w') as f:
            f.write(vtypes)
        logger.info(f"Vehicle types file created: {self.vtype_file}")
    
    def generate_routes(self, num_vehicles: int = 500,
                        duration: float = 3600.0,
                        emergency_probability: float = 0.02) -> bool:
        """
        Generate random routes for vehicles.
        
        Args:
            num_vehicles: Approximate number of vehicles
            duration: Simulation duration
            emergency_probability: Probability of emergency vehicle
            
        Returns:
            bool: True if successful
        """
        logger.info("Generating vehicle routes")
        
        # Use SUMO's randomTrips.py to generate trips
        random_trips_path = os.path.join(self.sumo_home, 'tools', 'randomTrips.py')
        
        if not os.path.exists(random_trips_path):
            logger.warning("randomTrips.py not found, generating basic routes")
            self._generate_basic_routes()
            return True
        
        trips_file = self.output_dir / "trips.xml"
        
        cmd = [
            sys.executable,
            random_trips_path,
            '-n', str(self.net_file),
            '-o', str(trips_file),
            '-e', str(duration),
            '-p', str(duration / num_vehicles),
            '--validate',
            '--random'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"randomTrips warning: {result.stderr}")
            
            # Convert trips to routes using duarouter
            self._convert_trips_to_routes(trips_file)
            
            # Modify routes to include different vehicle types
            self._diversify_vehicle_types(emergency_probability)
            
            return True
        except Exception as e:
            logger.error(f"Error generating routes: {e}")
            self._generate_basic_routes()
            return True
    
    def _convert_trips_to_routes(self, trips_file: Path) -> None:
        """Convert trips to routes using duarouter."""
        cmd = [
            'duarouter',
            '-n', str(self.net_file),
            '-t', str(trips_file),
            '-o', str(self.route_file),
            '--ignore-errors',
            '--no-warnings'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"Routes file created: {self.route_file}")
        except Exception as e:
            logger.warning(f"duarouter failed: {e}")
    
    def _generate_basic_routes(self) -> None:
        """Generate basic route file template."""
        routes = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Include vehicle types -->
    <include href="vehicles.vtype.xml"/>
    
    <!-- NOTE: This is a template. For realistic simulation, 
         use randomTrips.py to generate routes based on your network.
         
         Example flows can be added here like:
         <flow id="flow_0" type="car" from="edge_start" to="edge_end" 
               begin="0" end="3600" probability="0.1"/>
    -->
    
    <!-- Sample vehicle for testing -->
    <vehicle id="test_car" type="car" depart="0">
        <route edges=""/>
    </vehicle>
    
</routes>
"""
        with open(self.route_file, 'w') as f:
            f.write(routes)
        logger.info(f"Basic routes template created: {self.route_file}")
    
    def _diversify_vehicle_types(self, emergency_probability: float) -> None:
        """Modify routes to include diverse vehicle types."""
        if not self.route_file.exists():
            return
            
        try:
            tree = ET.parse(self.route_file)
            root = tree.getroot()
            
            import random
            vehicle_types = ['car', 'car', 'car', 'bus', 'truck', 'motorcycle', 'auto']
            emergency_types = ['ambulance', 'police', 'firetruck']
            
            for vehicle in root.findall('.//vehicle'):
                if random.random() < emergency_probability:
                    vehicle.set('type', random.choice(emergency_types))
                else:
                    vehicle.set('type', random.choice(vehicle_types))
            
            tree.write(self.route_file)
            logger.info("Vehicle types diversified in routes")
        except ET.ParseError as e:
            logger.warning(f"Could not parse routes file: {e}")
    
    def generate_additional_config(self) -> None:
        """Generate additional SUMO configuration (traffic lights, detectors)."""
        logger.info("Generating additional configuration")
        
        additional = """<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    
    <!-- Traffic light programs will be loaded from network file -->
    <!-- Additional detectors and POIs can be defined here -->
    
    <!-- Example: Edge-based mean data collection -->
    <!--
    <edgeData id="edge_data" file="edge_output.xml" period="300"/>
    -->
    
    <!-- Example: Lane area detector -->
    <!--
    <laneAreaDetector id="det_1" lane="lane_id_0" pos="0" endPos="50" freq="60" file="detector_output.xml"/>
    -->
    
    <!-- Rerouter for dynamic routing (example) -->
    <!--
    <rerouter id="rerouter_1" edges="edge_id">
        <interval begin="0" end="3600">
            <closingReroute id="edge_to_close"/>
            <destProbReroute id="alternative_edge" probability="1.0"/>
        </interval>
    </rerouter>
    -->
    
</additional>
"""
        
        with open(self.add_file, 'w') as f:
            f.write(additional)
        logger.info(f"Additional config created: {self.add_file}")
    
    def generate_sumo_config(self, duration: float = 3600.0,
                              step_length: float = 0.1) -> None:
        """
        Generate main SUMO configuration file.
        
        Args:
            duration: Simulation duration in seconds
            step_length: Simulation step length in seconds
        """
        logger.info("Generating SUMO configuration file")
        
        config = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{self.net_file.name}"/>
        <route-files value="{self.route_file.name}"/>
        <additional-files value="{self.vtype_file.name},{self.add_file.name}"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="{int(duration)}"/>
        <step-length value="{step_length}"/>
    </time>
    
    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <time-to-teleport value="300"/>
        <max-depart-delay value="60"/>
        <lanechange.duration value="3"/>
    </processing>
    
    <routing>
        <device.rerouting.probability value="0.3"/>
        <device.rerouting.period value="60"/>
        <device.rerouting.adaptation-steps value="18"/>
    </routing>
    
    <output>
        <summary-output value="summary.xml"/>
        <tripinfo-output value="tripinfo.xml"/>
        <statistic-output value="statistics.xml"/>
    </output>
    
    <gui_only>
        <gui-settings-file value=""/>
        <start value="true"/>
        <quit-on-end value="true"/>
    </gui_only>
    
    <report>
        <verbose value="false"/>
        <no-warnings value="false"/>
        <log value="simulation.log"/>
    </report>
    
</configuration>
"""
        
        with open(self.config_file, 'w') as f:
            f.write(config)
        logger.info(f"SUMO config created: {self.config_file}")
    
    def convert_all(self, duration: float = 3600.0,
                    num_vehicles: int = 500,
                    keep_all_routes: bool = True,
                    add_traffic_lights: bool = True) -> bool:
        """
        Perform complete conversion from OSM to SUMO.
        
        Args:
            duration: Simulation duration
            num_vehicles: Number of vehicles to generate
            keep_all_routes: Keep all route connections
            add_traffic_lights: Generate traffic lights
            
        Returns:
            bool: True if all conversions successful
        """
        logger.info(f"Starting complete conversion of {self.input_file}")
        
        # Step 1: Convert OSM to network
        if not self.convert_osm_to_net(keep_all_routes, add_traffic_lights):
            return False
        
        # Step 2: Generate vehicle types
        self.generate_vehicle_types()
        
        # Step 3: Generate routes
        self.generate_routes(num_vehicles, duration)
        
        # Step 4: Generate additional config
        self.generate_additional_config()
        
        # Step 5: Generate main config
        self.generate_sumo_config(duration)
        
        logger.info("=" * 50)
        logger.info("Conversion complete! Generated files:")
        logger.info(f"  Network: {self.net_file}")
        logger.info(f"  Vehicle Types: {self.vtype_file}")
        logger.info(f"  Routes: {self.route_file}")
        logger.info(f"  Additional: {self.add_file}")
        logger.info(f"  Config: {self.config_file}")
        logger.info("=" * 50)
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert OpenStreetMap to SUMO network"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input OSM file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../sumo_config',
        help='Output directory for SUMO files'
    )
    parser.add_argument(
        '--network-name', '-n',
        type=str,
        default='network',
        help='Name prefix for network files'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=3600.0,
        help='Simulation duration in seconds'
    )
    parser.add_argument(
        '--vehicles', '-v',
        type=int,
        default=500,
        help='Number of vehicles to generate'
    )
    parser.add_argument(
        '--keep-all-routes',
        action='store_true',
        default=True,
        help='Keep all route connections'
    )
    parser.add_argument(
        '--add-traffic-lights',
        action='store_true',
        default=True,
        help='Generate traffic lights at intersections'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    # Check input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Run conversion
    converter = MapConverter(input_path, output_path, args.network_name)
    success = converter.convert_all(
        duration=args.duration,
        num_vehicles=args.vehicles,
        keep_all_routes=args.keep_all_routes,
        add_traffic_lights=args.add_traffic_lights
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
