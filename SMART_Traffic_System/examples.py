#!/usr/bin/env python3
"""
Quick Start Example for SMART Traffic System
This script demonstrates basic usage of the system
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import TrafficSimulation


def example_basic_simulation():
    """Run a basic 5-minute simulation."""
    print("\n" + "="*60)
    print("Example 1: Basic 5-Minute Simulation")
    print("="*60)
    
    # Create simulation
    sim = TrafficSimulation(config_file='config/simulation_config.json', gui=False)
    
    # Run for 5 minutes (300 seconds)
    sim.run(duration=300)
    
    print("\nSimulation complete!")


def example_with_gui():
    """Run simulation with GUI."""
    print("\n" + "="*60)
    print("Example 2: Simulation with GUI")
    print("="*60)
    
    # Create simulation with GUI
    sim = TrafficSimulation(config_file='config/simulation_config.json', gui=True)
    
    # Run for 10 minutes
    sim.run(duration=600)
    
    print("\nSimulation complete!")


def example_emergency_test():
    """Run a test scenario with emergency vehicles."""
    print("\n" + "="*60)
    print("Example 3: Emergency Vehicle Test")
    print("="*60)
    
    # Create simulation
    sim = TrafficSimulation(config_file='config/simulation_config.json', gui=True)
    
    # Schedule emergency vehicles
    print("\nScheduling emergency vehicles:")
    print("  - Ambulance at 100s on lane 2")
    print("  - Fire truck at 300s on lane 1")
    print("  - Police at 500s on lane 0")
    
    sim.add_emergency_vehicle(time=100, lane=2, vehicle_type='ambulance')
    sim.add_emergency_vehicle(time=300, lane=1, vehicle_type='fire_truck')
    sim.add_emergency_vehicle(time=500, lane=0, vehicle_type='police')
    
    # Run simulation
    sim.run(duration=900)  # 15 minutes
    
    print("\nEmergency vehicle test complete!")


def example_custom_scenario():
    """Create a custom traffic scenario."""
    print("\n" + "="*60)
    print("Example 4: Custom Traffic Scenario")
    print("="*60)
    
    # Modify configuration for peak hour
    import json
    
    config = {
        'simulation': {
            'step_length': 0.1,
            'total_time': 1800,  # 30 minutes
            'gui_enabled': True
        },
        'traffic': {
            'vehicle_density': 'high',  # Peak hour traffic
            'peak_hour_multiplier': 2.0,
            'emergency_vehicle_probability': 0.02
        },
        'v2x': {
            'communication_range': 400,  # Extended range
            'message_frequency': 15,
            'broadcast_enabled': True
        }
    }
    
    # Save custom config
    with open('config/custom_scenario.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Custom configuration created: config/custom_scenario.json")
    
    # Run with custom config
    sim = TrafficSimulation(config_file='config/custom_scenario.json', gui=True)
    sim.run(duration=1800)
    
    print("\nCustom scenario complete!")


def main():
    """Main function with menu."""
    examples = {
        '1': ('Basic Simulation (5 min, no GUI)', example_basic_simulation),
        '2': ('Simulation with GUI (10 min)', example_with_gui),
        '3': ('Emergency Vehicle Test (15 min, GUI)', example_emergency_test),
        '4': ('Custom Peak Hour Scenario (30 min, GUI)', example_custom_scenario),
    }
    
    print("\n" + "="*60)
    print("SMART Traffic System - Quick Start Examples")
    print("="*60)
    print("\nAvailable examples:")
    
    for key, (description, _) in examples.items():
        print(f"  {key}. {description}")
    
    print("  Q. Quit")
    
    choice = input("\nSelect an example (1-4, Q to quit): ").strip().upper()
    
    if choice == 'Q':
        print("Goodbye!")
        return
    
    if choice in examples:
        _, example_func = examples[choice]
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nExample interrupted by user")
        except Exception as e:
            print(f"\n\nError running example: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    # Check if running from correct directory
    if not os.path.exists('config/simulation_config.json'):
        print("ERROR: Please run this script from the SMART_Traffic_System directory")
        print("Usage: python examples.py")
        sys.exit(1)
    
    main()
