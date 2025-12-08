"""
Main Simulation Runner
Part of Smart Traffic Management System

Orchestrates complete simulation with all components:
- SUMO traffic simulation
- Adaptive signal control
- Emergency vehicle priority
- V2X communication
- SIoT trust management

Usage:
    python run_simulation.py --config configs/scenario_config.json --duration 3600
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python/core'))

from adaptive_signals import AdaptiveSignalController
from emergency_priority import EmergencyVehicleManager
from v2x_communication import V2XCommunication, MessageType
from siot_trust import SIoTTrustManager

import traci

class SmartTrafficSimulation:
    """
    Main simulation orchestrator that integrates all system components.
    """
    
    def __init__(self, config_file):
        """
        Initialize simulation with configuration.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config = self.load_config(config_file)
        
        # Initialize components
        self.adaptive_signals = AdaptiveSignalController()
        self.emergency_manager = EmergencyVehicleManager()
        self.v2x = V2XCommunication(
            communication_range=self.config.get('communication_range', 300)
        )
        self.siot = SIoTTrustManager()
        
        # Simulation state
        self.step = 0
        self.start_time = None
        self.is_running = False
        
    def load_config(self, config_file):
        """Load simulation configuration from JSON"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from: {config_file}")
            return config
        except FileNotFoundError:
            print(f"Config file not found: {config_file}")
            print("Using default configuration")
            return self.get_default_config()
            
    def get_default_config(self):
        """Return default configuration"""
        return {
            'sumo_config': 'sumo/scenarios/basic_traffic.sumocfg',
            'duration': 3600,
            'use_gui': False,
            'communication_range': 300,
            'enable_adaptive_signals': True,
            'enable_emergency_priority': True,
            'enable_v2x': True,
            'enable_siot': True,
            'rsu_positions': {
                'rsu_1': [500, 500],
                'rsu_2': [1500, 500],
                'rsu_3': [500, 1500],
                'rsu_4': [1500, 1500]
            }
        }
        
    def setup_rsus(self):
        """Setup Roadside Units (RSUs) for V2I communication"""
        rsu_positions = self.config.get('rsu_positions', {})
        
        for rsu_id, position in rsu_positions.items():
            self.v2x.register_rsu(rsu_id, tuple(position))
            print(f"Registered RSU: {rsu_id} at {position}")
            
    def start_simulation(self):
        """Start SUMO and initialize TraCI connection"""
        sumo_config = self.config.get('sumo_config', 'sumo/scenarios/basic_traffic.sumocfg')
        use_gui = self.config.get('use_gui', False)
        
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", sumo_config, "--start", "--quit-on-end"]
        
        print("\n" + "="*80)
        print("SMART TRAFFIC MANAGEMENT SYSTEM")
        print("="*80)
        print(f"Simulation Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"SUMO Configuration: {sumo_config}")
        print(f"Duration: {self.config.get('duration', 3600)} seconds")
        print(f"GUI Enabled: {use_gui}")
        print("\nEnabled Features:")
        print(f"  ✓ Adaptive Signals: {self.config.get('enable_adaptive_signals', True)}")
        print(f"  ✓ Emergency Priority: {self.config.get('enable_emergency_priority', True)}")
        print(f"  ✓ V2X Communication: {self.config.get('enable_v2x', True)}")
        print(f"  ✓ SIoT Trust Management: {self.config.get('enable_siot', True)}")
        print("="*80 + "\n")
        
        traci.start(sumo_cmd)
        self.start_time = time.time()
        self.is_running = True
        
        # Setup RSUs
        self.setup_rsus()
        
    def step_simulation(self):
        """Execute one simulation step"""
        traci.simulationStep()
        
        current_time = self.step * 0.1  # SUMO step = 0.1 seconds
        
        # Get current vehicles
        all_vehicles = traci.vehicle.getIDList()
        
        # Adaptive signal control (every 1 second)
        if self.config.get('enable_adaptive_signals', True) and self.step % 10 == 0:
            junctions = traci.trafficlight.getIDList()
            for junction in junctions:
                self.adaptive_signals.adapt_signal_timing(junction, self.step)
        
        # Emergency vehicle management
        if self.config.get('enable_emergency_priority', True):
            emergency_vehicles = self.emergency_manager.detect_emergency_vehicles()
            for ev_id in emergency_vehicles:
                self.emergency_manager.manage_emergency_vehicle(ev_id, self.step)
        
        # V2X Communication (every 0.5 seconds)
        if self.config.get('enable_v2x', True) and self.step % 5 == 0:
            # Broadcast SPaT from RSUs
            for rsu_id in self.v2x.rsu_positions.keys():
                junctions = traci.trafficlight.getIDList()
                if junctions:
                    self.v2x.broadcast_spat(junctions[0], rsu_id)
            
            # V2V position broadcasts from vehicles
            for vehicle_id in all_vehicles[:10]:  # Limit to avoid overload
                try:
                    position = traci.vehicle.getPosition(vehicle_id)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    
                    content = {
                        'position': position,
                        'speed': speed,
                        'timestamp': current_time
                    }
                    
                    self.v2x.broadcast_v2v(vehicle_id, MessageType.V2V_POSITION, content)
                except:
                    pass
        
        # SIoT Trust Management (every 5 seconds)
        if self.config.get('enable_siot', True) and self.step % 50 == 0:
            # Discover new relationships
            self.siot.discover_relationships()
            
            # Update trust scores based on behavior
            self.siot.update_trust_scores(self.step)
            
            # Save periodic snapshot (every minute)
            if self.step % 600 == 0:
                self.siot.save_trust_snapshot()
        
        # Progress reporting (every 60 seconds)
        if self.step % 600 == 0:
            elapsed = self.step / 10
            progress = (elapsed / self.config.get('duration', 3600)) * 100
            print(f"[{elapsed:.0f}s] Progress: {progress:.1f}% | "
                  f"Vehicles: {len(all_vehicles)} | "
                  f"Messages: {self.v2x.stats.get('v2v_sent', 0)}")
        
        self.step += 1
        
    def run(self):
        """Main simulation loop"""
        try:
            self.start_simulation()
            
            duration = self.config.get('duration', 3600)
            max_steps = duration * 10  # 0.1s step size
            
            while self.step < max_steps and self.is_running:
                if traci.simulation.getMinExpectedNumber() <= 0:
                    print("\nAll vehicles completed. Ending simulation.")
                    break
                    
                self.step_simulation()
                
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up and generate reports"""
        print("\n" + "="*80)
        print("SIMULATION CLEANUP AND REPORTING")
        print("="*80)
        
        # Close TraCI
        try:
            traci.close()
        except:
            pass
        
        # Close database connections
        if hasattr(self.adaptive_signals, 'db_connection'):
            self.adaptive_signals.db_connection.close()
        if hasattr(self.emergency_manager, 'db_connection'):
            self.emergency_manager.db_connection.close()
        if hasattr(self.v2x, 'db_connection'):
            self.v2x.db_connection.close()
        if hasattr(self.siot, 'db_connection'):
            self.siot.db_connection.close()
        
        # Generate reports
        print("\nGenerating component reports...\n")
        
        if self.config.get('enable_adaptive_signals', True):
            self.adaptive_signals.generate_report()
            
        if self.config.get('enable_emergency_priority', True):
            self.emergency_manager.generate_report()
            
        if self.config.get('enable_v2x', True):
            self.v2x.generate_report()
            
        if self.config.get('enable_siot', True):
            self.siot.generate_report()
        
        # Final summary
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        print("\n" + "="*80)
        print("SIMULATION COMPLETED")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Real Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Simulated Time: {self.step / 10:.2f} seconds")
        print(f"Simulation Steps: {self.step}")
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Smart Traffic Management System Simulation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/scenario_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable SUMO GUI'
    )
    
    args = parser.parse_args()
    
    # Create simulation
    sim = SmartTrafficSimulation(args.config)
    
    # Override config with command line args
    if args.duration:
        sim.config['duration'] = args.duration
    if args.gui:
        sim.config['use_gui'] = True
    
    # Run simulation
    sim.run()


if __name__ == "__main__":
    main()
