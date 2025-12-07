"""
Smart Traffic Management System - Main Integration
Orchestrates all components of the traffic simulation system
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))
sys.path.append(str(Path(__file__).parent / 'database'))
sys.path.append(str(Path(__file__).parent / 'simulation'))

from vehicle_agents import create_vehicle, VehicleType, VEHICLE_CONFIGS
from communication import V2XCommunicationManager, MessageType
# from schema import DatabaseManager, ETLPipeline  # Optional: requires sqlalchemy


class TrafficSimulationSystem:
    """
    Main system controller integrating all components
    Maps to: Operating Systems - System-level orchestration
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        # self.db_manager = DatabaseManager()  # Optional database integration
        self.comm_manager = V2XCommunicationManager()
        self.vehicles: Dict[str, object] = {}
        self.simulation_id: str = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.running: bool = False
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load simulation configuration"""
        default_config = {
            'scenario': 'city_center',
            'duration': 3600,
            'vehicle_mix': {
                'car': 100,
                'bus': 10,
                'truck': 15,
                'motorcycle': 30,
                'bicycle': 20,
                'pedestrian': 50
            },
            'communication_range': 100.0,
            'trust_threshold': 0.3,
            'step_length': 0.1
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        return default_config
    
    def initialize_vehicles(self) -> None:
        """
        Create vehicle agents based on configuration
        Demonstrates OOP: Factory pattern and polymorphism
        """
        print(f"\n{'='*60}")
        print(f"Initializing Smart Traffic Management System")
        print(f"Simulation ID: {self.simulation_id}")
        print(f"Scenario: {self.config['scenario']}")
        print(f"{'='*60}\n")
        
        vehicle_count = 0
        for vehicle_type_str, count in self.config['vehicle_mix'].items():
            try:
                vehicle_type = VehicleType(vehicle_type_str)
                for i in range(count):
                    vehicle_id = f"{vehicle_type_str}_{i}"
                    vehicle = create_vehicle(vehicle_type, vehicle_id)
                    self.vehicles[vehicle_id] = vehicle
                    self.comm_manager.register_agent(vehicle_id)
                    vehicle_count += 1
                    
                print(f"✓ Created {count} {vehicle_type_str} vehicles")
            except ValueError as e:
                print(f"✗ Error creating {vehicle_type_str}: {e}")
        
        print(f"\nTotal vehicles created: {vehicle_count}")
        
    def run_simulation_step(self, time_step: float) -> None:
        """
        Execute one simulation step
        Updates all vehicle states and handles communication
        """
        # Update each vehicle
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle.update_state(time_step)
            
            # Get messages for this vehicle
            messages = self.comm_manager.get_messages(vehicle_id)
            for message in messages:
                vehicle.handle_message(message.to_dict())
            
            # Log vehicle state
            position = vehicle.get_position()
            speed = vehicle.get_speed()
            
            # In production, would log to database
            # self.db_manager.log_vehicle_state(
            #     self.simulation_id, vehicle_id, position, speed, {}
            # )
    
    def simulate_v2v_communication(self) -> None:
        """
        Simulate V2V communication between vehicles
        Demonstrates Computer Networks concepts
        """
        # Sample communication scenario
        if len(self.vehicles) >= 2:
            vehicle_ids = list(self.vehicles.keys())
            sender = vehicle_ids[0]
            receiver = vehicle_ids[1]
            
            # Send traffic update message
            self.comm_manager.send_v2v_message(
                sender_id=sender,
                receiver_id=receiver,
                msg_type=MessageType.TRAFFIC_UPDATE,
                payload={
                    'recommended_speed': 30.0,
                    'road_condition': 'clear',
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def generate_statistics(self) -> Dict:
        """
        Generate simulation statistics
        Maps to: DBMS - Analytics queries
        """
        stats = {
            'simulation_id': self.simulation_id,
            'scenario': self.config['scenario'],
            'total_vehicles': len(self.vehicles),
            'vehicle_breakdown': {},
            'communication_stats': self.comm_manager.channel.get_statistics()
        }
        
        # Count vehicles by type
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle_type = vehicle.vehicle_type.value
            if vehicle_type not in stats['vehicle_breakdown']:
                stats['vehicle_breakdown'][vehicle_type] = 0
            stats['vehicle_breakdown'][vehicle_type] += 1
        
        return stats
    
    def export_results(self, output_dir: str = 'results') -> None:
        """Export simulation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self.generate_statistics()
        
        output_file = os.path.join(
            output_dir, 
            f"simulation_{self.simulation_id}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Results exported to: {output_file}")
    
    def print_summary(self) -> None:
        """Print simulation summary"""
        stats = self.generate_statistics()
        
        print(f"\n{'='*60}")
        print(f"Simulation Summary")
        print(f"{'='*60}")
        print(f"Scenario: {stats['scenario']}")
        print(f"Total Vehicles: {stats['total_vehicles']}")
        print(f"\nVehicle Breakdown:")
        for vehicle_type, count in stats['vehicle_breakdown'].items():
            print(f"  - {vehicle_type}: {count}")
        
        print(f"\nCommunication Statistics:")
        comm_stats = stats['communication_stats']
        print(f"  - Total Messages: {comm_stats['total_messages']}")
        print(f"  - Success Rate: {comm_stats['success_rate']:.2%}")
        print(f"  - Avg Latency: {comm_stats['avg_latency_ms']:.2f} ms")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   Smart Traffic Management System                          ║
    ║   Full-Stack Multi-Agent Traffic Simulation                ║
    ║                                                            ║
    ║   Features:                                                ║
    ║   • OSM Integration                                        ║
    ║   • SUMO/TraCI Simulation                                  ║
    ║   • V2V/V2I Communication                                  ║
    ║   • Real-time Dashboard                                    ║
    ║   • Multi-vehicle Types                                    ║
    ║   • Trust-based Routing                                    ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Create and initialize system
    system = TrafficSimulationSystem()
    
    # Initialize vehicles
    system.initialize_vehicles()
    
    # Simulate a few communication events
    print("\nSimulating V2V communication...")
    for _ in range(5):
        system.simulate_v2v_communication()
    print("✓ Communication simulation complete")
    
    # Generate and display statistics
    system.print_summary()
    
    # Export results
    system.export_results()
    
    print("✓ Simulation complete!")
    print("\nNext steps:")
    print("  1. Open frontend/dashboard.html to view the dashboard")
    print("  2. Review docs/DOCUMENTATION_INDEX.md for complete documentation")
    print("  3. Run with SUMO: python simulation/sumo_controller.py")
    

if __name__ == "__main__":
    main()
