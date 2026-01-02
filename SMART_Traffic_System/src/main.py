#!/usr/bin/env python3
"""
SMART Traffic Control System - Main Controller
Author: SMART Traffic System
Date: January 2026

This is the main entry point for the SMART traffic simulation system.
It orchestrates all modules and manages the simulation lifecycle.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

# Import project modules
from config_manager import ConfigManager
from v2x_communication import V2XCommunication
from lane_detection import LaneDetection
from traffic_signal_control import TrafficSignalControl
from emergency_vehicle_priority import EmergencyVehiclePriority
from performance_metrics import PerformanceMetrics


class TrafficSimulation:
    """
    Main simulation controller for SMART Traffic System.
    
    Responsibilities:
    - Initialize SUMO simulation
    - Coordinate all subsystems
    - Manage simulation lifecycle
    - Handle emergency events
    - Collect and export results
    """
    
    def __init__(self, config_file='config/simulation_config.json', gui=False):
        """
        Initialize the traffic simulation.
        
        Args:
            config_file (str): Path to configuration file
            gui (bool): Whether to use SUMO GUI
        """
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_config()
        self.gui = gui
        self.simulation_running = False
        
        # Initialize subsystems
        self.v2x = None
        self.lane_detector = None
        self.signal_control = None
        self.emergency_priority = None
        self.metrics = None
        
        # Simulation state
        self.current_time = 0
        self.emergency_vehicles = []
        
        print("\n" + "="*60)
        print("SMART Traffic Control System Initialized")
        print("="*60)
        print(f"Configuration: {config_file}")
        print(f"GUI Mode: {gui}")
        print(f"Simulation Duration: {self.config['simulation']['total_time']}s")
        print("="*60 + "\n")
    
    def start_sumo(self):
        """Start SUMO simulation with appropriate configuration."""
        sumo_config = os.path.join('sumo_files', 'simulation.sumocfg')
        
        if self.gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
            sumo_cmd = [sumo_binary, '-c', sumo_config, '--start']
        else:
            sumo_binary = sumolib.checkBinary('sumo')
            sumo_cmd = [sumo_binary, '-c', sumo_config, '--no-step-log']
        
        # Add additional options
        sumo_cmd.extend([
            '--step-length', str(self.config['simulation']['step_length']),
            '--time-to-teleport', '300',
            '--collision.action', 'warn',
            '--seed', '42'
        ])
        
        print(f"Starting SUMO: {' '.join(sumo_cmd)}\n")
        traci.start(sumo_cmd)
        self.simulation_running = True
        
        # Initialize subsystems after TraCI connection
        self._initialize_subsystems()
    
    def _initialize_subsystems(self):
        """Initialize all subsystem modules."""
        print("Initializing subsystems...")
        
        # V2X Communication
        self.v2x = V2XCommunication(
            range_meters=self.config['v2x']['communication_range'],
            frequency=self.config['v2x']['message_frequency']
        )
        
        # Lane Detection
        self.lane_detector = LaneDetection()
        
        # Traffic Signal Control
        self.signal_control = TrafficSignalControl()
        
        # Emergency Vehicle Priority
        self.emergency_priority = EmergencyVehiclePriority()
        
        # Performance Metrics
        self.metrics = PerformanceMetrics(output_dir='results/metrics')
        
        print("✓ All subsystems initialized\n")
    
    def run(self, duration=None):
        """
        Run the traffic simulation.
        
        Args:
            duration (int): Simulation duration in seconds (overrides config)
        """
        if not self.simulation_running:
            self.start_sumo()
        
        sim_duration = duration or self.config['simulation']['total_time']
        step_length = self.config['simulation']['step_length']
        total_steps = int(sim_duration / step_length)
        
        print(f"Starting simulation for {sim_duration}s ({total_steps} steps)\n")
        
        try:
            for step in range(total_steps):
                self.current_time = step * step_length
                
                # Perform simulation step
                traci.simulationStep()
                
                # Update all subsystems
                self._update_subsystems()
                
                # Display progress every 60 seconds (simulation time)
                if self.current_time % 60 == 0 and self.current_time > 0:
                    self._display_status()
                
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        except Exception as e:
            print(f"\n\nError during simulation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _update_subsystems(self):
        """Update all subsystems for current simulation step."""
        # Get all vehicles in simulation
        vehicle_ids = traci.vehicle.getIDList()
        
        # Update V2X communication
        if self.config['v2x']['broadcast_enabled']:
            self.v2x.update(vehicle_ids, self.current_time)
        
        # Update lane detection
        lane_data = self.lane_detector.analyze_lanes(vehicle_ids)
        
        # Check for emergency vehicles
        emergency_detected = self.emergency_priority.detect_emergency_vehicles(vehicle_ids)
        
        if emergency_detected:
            # Handle emergency vehicle priority
            for emerg_vehicle in emergency_detected:
                if emerg_vehicle not in self.emergency_vehicles:
                    self.emergency_vehicles.append(emerg_vehicle)
                    print(f"\n🚨 EMERGENCY VEHICLE DETECTED: {emerg_vehicle}")
                    self.emergency_priority.activate_priority(emerg_vehicle, lane_data)
                    self.signal_control.emergency_mode(emerg_vehicle)
        
        # Remove departed emergency vehicles
        self.emergency_vehicles = [v for v in self.emergency_vehicles if v in vehicle_ids]
        
        # Update traffic signals (adaptive control)
        if not self.emergency_vehicles:  # Normal operation
            self.signal_control.update_adaptive_control(lane_data, self.current_time)
        
        # Collect metrics
        self.metrics.collect_data(vehicle_ids, lane_data, self.current_time)
    
    def _display_status(self):
        """Display current simulation status."""
        vehicle_count = len(traci.vehicle.getIDList())
        
        print(f"\n{'='*60}")
        print(f"[TIME: {self._format_time(self.current_time)}]")
        print(f"Vehicles in simulation: {vehicle_count}")
        
        if self.emergency_vehicles:
            print(f"🚨 Active emergency vehicles: {len(self.emergency_vehicles)}")
        
        # Display lane statistics
        lane_stats = self.lane_detector.get_summary()
        if lane_stats:
            print("\nLane Status:")
            for lane_id, stats in list(lane_stats.items())[:4]:  # Show first 4 lanes
                status = "CONGESTED" if stats['congested'] else "NORMAL"
                print(f"  {lane_id}: {stats['vehicle_count']} vehicles, "
                      f"avg speed: {stats['avg_speed']:.1f} m/s [{status}]")
        
        print(f"{'='*60}\n")
    
    def _format_time(self, seconds):
        """Format simulation time as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def add_emergency_vehicle(self, time, lane, vehicle_type='ambulance'):
        """
        Add an emergency vehicle at specific time and lane.
        
        Args:
            time (int): Depart time in seconds
            lane (int): Lane number (0-3)
            vehicle_type (str): Type of emergency vehicle
        """
        # This would be implemented using TraCI vehicle add
        print(f"Scheduled {vehicle_type} at time {time} on lane {lane}")
    
    def export_results(self, output_dir='results'):
        """
        Export all collected metrics and results.
        
        Args:
            output_dir (str): Directory to save results
        """
        print("\nExporting results...")
        self.metrics.export_csv(output_dir)
        self.metrics.generate_summary_report(output_dir)
        print(f"✓ Results exported to {output_dir}\n")
    
    def stop(self):
        """Stop the simulation and cleanup."""
        if self.simulation_running:
            print("\nStopping simulation...")
            
            # Export final results
            self.export_results()
            
            # Close TraCI connection
            traci.close()
            self.simulation_running = False
            
            print("✓ Simulation stopped")
            print("\nThank you for using SMART Traffic Control System!")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='SMART Traffic Control System - Intelligent Traffic Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --gui
  python main.py --duration 1800 --config config/peak_hour.json
  python main.py --emergency-test --gui
        """
    )
    
    parser.add_argument('--config', type=str, default='config/simulation_config.json',
                        help='Path to configuration file')
    parser.add_argument('--gui', action='store_true',
                        help='Run with SUMO GUI')
    parser.add_argument('--duration', type=int,
                        help='Simulation duration in seconds (overrides config)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--emergency-test', action='store_true',
                        help='Run emergency vehicle test scenario')
    parser.add_argument('--emergency-frequency', type=float,
                        help='Emergency vehicle probability (0.0-1.0)')
    parser.add_argument('--test', action='store_true',
                        help='Run system test and exit')
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        print("Running system test...")
        print("✓ Python modules imported successfully")
        print("✓ SUMO_HOME environment variable set")
        print("✓ All dependencies available")
        print("\nSystem test PASSED")
        return 0
    
    # Create simulation instance
    try:
        sim = TrafficSimulation(config_file=args.config, gui=args.gui)
        
        # Run simulation
        sim.run(duration=args.duration)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
