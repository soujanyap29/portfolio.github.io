#!/usr/bin/env python3
"""
Smart Traffic Management System - Main Simulation Runner

This script orchestrates the entire traffic simulation, integrating:
- SUMO simulation via TraCI
- V2V (Vehicle-to-Vehicle) communication
- V2I (Vehicle-to-Infrastructure) communication
- SIoT (Social Internet of Things) behavior
- Emergency vehicle handling
- Adaptive traffic light control

Usage:
    python main.py [options]

Options:
    --gui           Run with SUMO GUI
    --duration      Simulation duration in seconds (default: 3600)
    --config        Path to SUMO configuration file
    --compare       Run comparative analysis (fixed, actuated, smart)
    --output        Output directory for results
    --test-mode     Run in test mode (shorter duration)
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple
import logging

# Ensure SUMO tools are in path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # Try common SUMO installation paths
    common_paths = [
        '/usr/share/sumo/tools',
        '/usr/local/share/sumo/tools',
        'C:\\Program Files (x86)\\Eclipse\\Sumo\\tools',
        'C:\\Program Files\\Eclipse\\Sumo\\tools'
    ]
    for path in common_paths:
        if os.path.exists(path):
            sys.path.append(path)
            os.environ['SUMO_HOME'] = os.path.dirname(path)
            break

try:
    import traci
    import sumolib
except ImportError:
    print("Error: Could not import SUMO libraries.")
    print("Please ensure SUMO is installed and SUMO_HOME environment variable is set.")
    sys.exit(1)

# Import local modules
from vehicle_controller import VehicleController
from traffic_light_controller import TrafficLightController
from v2v_communication import V2VCommunication
from v2i_communication import V2ICommunication
from siot_manager import SIoTManager
from emergency_handler import EmergencyHandler
from metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SmartTraffic')


class SmartTrafficSimulation:
    """
    Main simulation controller for the Smart Traffic Management System.
    
    This class coordinates all components of the intelligent traffic system
    including V2V/V2I communication, SIoT behavior, and adaptive control.
    """
    
    def __init__(self, config_file: str, gui: bool = False, output_dir: str = "results"):
        """
        Initialize the simulation.
        
        Args:
            config_file: Path to SUMO configuration file
            gui: Whether to run with GUI
            output_dir: Directory for output files
        """
        self.config_file = config_file
        self.gui = gui
        self.output_dir = output_dir
        
        # Initialize component managers
        self.vehicle_controller = VehicleController()
        self.traffic_light_controller = TrafficLightController()
        self.v2v_comm = V2VCommunication(communication_range=100.0)
        self.v2i_comm = V2ICommunication()
        self.siot_manager = SIoTManager()
        self.emergency_handler = EmergencyHandler()
        self.metrics_collector = MetricsCollector(output_dir)
        
        # Simulation state
        self.step = 0
        self.running = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def start(self) -> None:
        """Start the SUMO simulation and TraCI connection."""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--step-length", "0.1",
            "--collision.action", "warn",
            "--collision.check-junctions",
            "--start",
            "--quit-on-end",
        ]
        
        logger.info(f"Starting SUMO with command: {' '.join(sumo_cmd)}")
        
        try:
            traci.start(sumo_cmd)
            self.running = True
            logger.info("TraCI connection established successfully")
            
            # Initialize traffic lights
            self._initialize_traffic_lights()
            
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            raise
    
    def _initialize_traffic_lights(self) -> None:
        """Initialize traffic light controller with all traffic lights in the network."""
        tl_ids = traci.trafficlight.getIDList()
        for tl_id in tl_ids:
            self.traffic_light_controller.register_traffic_light(tl_id)
            self.v2i_comm.register_rsu(tl_id)
        logger.info(f"Initialized {len(tl_ids)} traffic lights")
    
    def step_simulation(self) -> bool:
        """
        Execute one simulation step.
        
        Returns:
            bool: True if simulation should continue, False if ended
        """
        if not self.running:
            return False
            
        try:
            # Get current simulation time
            sim_time = traci.simulation.getTime()
            
            # Get all vehicles in simulation
            vehicle_ids = traci.vehicle.getIDList()
            
            # Update vehicle states
            vehicle_states = self._get_vehicle_states(vehicle_ids)
            
            # Process V2V communication
            v2v_messages = self.v2v_comm.process_communication(vehicle_states)
            
            # Process V2I communication
            v2i_messages = self.v2i_comm.process_communication(vehicle_states)
            
            # Update SIoT relationships and trust
            self.siot_manager.update_relationships(vehicle_states, v2v_messages)
            
            # Handle emergency vehicles
            emergency_vehicles = self._get_emergency_vehicles(vehicle_ids)
            if emergency_vehicles:
                self.emergency_handler.handle_emergency_vehicles(
                    emergency_vehicles,
                    vehicle_states,
                    self.traffic_light_controller
                )
            
            # Apply adaptive traffic light control
            self.traffic_light_controller.update_signals(vehicle_states, v2i_messages)
            
            # Apply vehicle control actions based on communication
            self._apply_vehicle_actions(vehicle_ids, v2v_messages, v2i_messages)
            
            # Collect metrics
            self.metrics_collector.collect_step_metrics(
                sim_time,
                vehicle_states,
                v2v_messages,
                v2i_messages,
                self.siot_manager.get_trust_scores()
            )
            
            # Advance simulation
            traci.simulationStep()
            self.step += 1
            
            # Check if simulation ended
            if traci.simulation.getMinExpectedNumber() <= 0:
                return False
                
            return True
            
        except traci.exceptions.FatalTraCIError as e:
            logger.error(f"TraCI Error: {e}")
            return False
    
    def _get_vehicle_states(self, vehicle_ids: List[str]) -> Dict:
        """
        Get current state of all vehicles.
        
        Args:
            vehicle_ids: List of vehicle IDs
            
        Returns:
            Dictionary mapping vehicle IDs to their states
        """
        states = {}
        for veh_id in vehicle_ids:
            try:
                states[veh_id] = {
                    'id': veh_id,
                    'position': traci.vehicle.getPosition(veh_id),
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'acceleration': traci.vehicle.getAcceleration(veh_id),
                    'lane_id': traci.vehicle.getLaneID(veh_id),
                    'lane_index': traci.vehicle.getLaneIndex(veh_id),
                    'route': traci.vehicle.getRoute(veh_id),
                    'route_index': traci.vehicle.getRouteIndex(veh_id),
                    'type': traci.vehicle.getTypeID(veh_id),
                    'angle': traci.vehicle.getAngle(veh_id),
                    'signals': traci.vehicle.getSignals(veh_id),
                }
            except traci.exceptions.TraCIException as e:
                logger.warning(f"Could not get state for vehicle {veh_id}: {e}")
        return states
    
    def _get_emergency_vehicles(self, vehicle_ids: List[str]) -> List[str]:
        """
        Identify emergency vehicles in the simulation.
        
        Args:
            vehicle_ids: List of all vehicle IDs
            
        Returns:
            List of emergency vehicle IDs
        """
        emergency_types = {'ambulance', 'police', 'firetruck', 'emergency'}
        emergency_vehicles = []
        
        for veh_id in vehicle_ids:
            try:
                veh_type = traci.vehicle.getTypeID(veh_id).lower()
                if any(em_type in veh_type for em_type in emergency_types):
                    emergency_vehicles.append(veh_id)
            except traci.exceptions.TraCIException:
                pass
                
        return emergency_vehicles
    
    def _apply_vehicle_actions(self, vehicle_ids: List[str],
                                v2v_messages: Dict, v2i_messages: Dict) -> None:
        """
        Apply control actions to vehicles based on communication.
        
        Args:
            vehicle_ids: List of vehicle IDs
            v2v_messages: V2V communication messages
            v2i_messages: V2I communication messages
        """
        for veh_id in vehicle_ids:
            try:
                # Get recommendations from V2I
                if veh_id in v2i_messages:
                    msg = v2i_messages[veh_id]
                    
                    # Apply recommended speed if available
                    if 'recommended_speed' in msg:
                        current_speed = traci.vehicle.getSpeed(veh_id)
                        rec_speed = msg['recommended_speed']
                        # Gradually adjust speed
                        if abs(current_speed - rec_speed) > 1.0:
                            new_speed = current_speed + 0.3 * (rec_speed - current_speed)
                            traci.vehicle.setSpeed(veh_id, max(0, new_speed))
                
                # Process V2V alerts
                if veh_id in v2v_messages:
                    alerts = v2v_messages[veh_id].get('alerts', [])
                    for alert in alerts:
                        if alert['type'] == 'emergency_braking':
                            # Increase following gap
                            traci.vehicle.setMinGap(veh_id, 3.0)
                        elif alert['type'] == 'emergency_vehicle':
                            # Change lane if possible
                            self._yield_to_emergency(veh_id)
                            
            except traci.exceptions.TraCIException as e:
                logger.debug(f"Could not apply action to {veh_id}: {e}")
    
    def _yield_to_emergency(self, veh_id: str) -> None:
        """Make a vehicle yield to emergency vehicles."""
        try:
            current_lane = traci.vehicle.getLaneIndex(veh_id)
            lane_count = traci.edge.getLaneNumber(traci.vehicle.getRoadID(veh_id))
            
            # Try to move to rightmost lane
            if current_lane > 0:
                traci.vehicle.changeLane(veh_id, current_lane - 1, 5.0)
            elif lane_count > 1:
                # If already in rightmost, slow down
                traci.vehicle.setSpeed(veh_id, 5.0)
        except traci.exceptions.TraCIException:
            pass
    
    def run(self, duration: float = 3600.0) -> Dict:
        """
        Run the simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            Dictionary containing simulation results
        """
        logger.info(f"Running simulation for {duration} seconds")
        start_time = time.time()
        
        try:
            self.start()
            
            while self.running and traci.simulation.getTime() < duration:
                if not self.step_simulation():
                    break
                    
                # Log progress periodically
                if self.step % 1000 == 0:
                    sim_time = traci.simulation.getTime()
                    vehicle_count = traci.vehicle.getIDCount()
                    logger.info(f"Step {self.step}, Time: {sim_time:.1f}s, Vehicles: {vehicle_count}")
            
            # Generate final metrics
            results = self.metrics_collector.generate_summary()
            
            return results
            
        finally:
            self.stop()
            elapsed = time.time() - start_time
            logger.info(f"Simulation completed in {elapsed:.2f} seconds (real time)")
    
    def stop(self) -> None:
        """Stop the simulation and close TraCI connection."""
        if self.running:
            try:
                traci.close()
                logger.info("TraCI connection closed")
            except Exception as e:
                logger.warning(f"Error closing TraCI: {e}")
            finally:
                self.running = False


def run_comparative_analysis(config_file: str, output_dir: str, duration: float) -> Dict:
    """
    Run comparative analysis of different traffic control strategies.
    
    Args:
        config_file: Path to SUMO configuration file
        output_dir: Output directory for results
        duration: Simulation duration in seconds
        
    Returns:
        Dictionary containing results for all strategies
    """
    strategies = {
        'fixed_time': {'adaptive': False, 'v2v': False, 'v2i': False, 'siot': False},
        'actuated': {'adaptive': True, 'v2v': False, 'v2i': False, 'siot': False},
        'smart_system': {'adaptive': True, 'v2v': True, 'v2i': True, 'siot': True}
    }
    
    all_results = {}
    
    for strategy_name, settings in strategies.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running strategy: {strategy_name}")
        logger.info(f"{'='*50}")
        
        strategy_output = os.path.join(output_dir, strategy_name)
        sim = SmartTrafficSimulation(config_file, gui=False, output_dir=strategy_output)
        
        # Configure strategy
        sim.v2v_comm.enabled = settings['v2v']
        sim.v2i_comm.enabled = settings['v2i']
        sim.siot_manager.enabled = settings['siot']
        sim.traffic_light_controller.adaptive = settings['adaptive']
        
        results = sim.run(duration)
        all_results[strategy_name] = results
        
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Traffic Management System Simulation"
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Run with SUMO GUI'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=3600.0,
        help='Simulation duration in seconds (default: 3600)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../sumo_config/simulation.sumocfg',
        help='Path to SUMO configuration file'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparative analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (100 second simulation)'
    )
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    # Set duration for test mode
    duration = 100.0 if args.test_mode else args.duration
    
    if args.compare:
        results = run_comparative_analysis(config_path, output_path, duration)
        logger.info("\n=== Comparative Analysis Complete ===")
        for strategy, metrics in results.items():
            logger.info(f"\n{strategy}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
    else:
        sim = SmartTrafficSimulation(config_path, args.gui, output_path)
        results = sim.run(duration)
        logger.info("\n=== Simulation Results ===")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
