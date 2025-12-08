"""
Adaptive Traffic Signal Control System
Part of Smart Traffic Management System

This module implements congestion-aware adaptive signal timing using TraCI.
Monitors lane occupancy and dynamically adjusts signal phases to optimize traffic flow.

Course Mappings:
- Operating Systems: Process scheduling, resource allocation
- Computer Networks: Real-time communication, latency optimization
- DBMS: Event logging and metric collection
"""

import traci
import sys
import time
import sqlite3
from datetime import datetime
import json

class AdaptiveSignalController:
    """
    Adaptive traffic signal controller that adjusts timing based on real-time congestion.
    
    Features:
    - Lane occupancy monitoring
    - Dynamic phase duration adjustment
    - Emergency vehicle detection and priority
    - Congestion-based green wave coordination
    """
    
    def __init__(self, config_file='../../configs/signal_config.json'):
        """Initialize the adaptive signal controller"""
        self.junctions = []
        self.occupancy_threshold = 0.7  # 70% occupancy triggers adaptation
        self.min_green_time = 20  # Minimum green phase duration (seconds)
        self.max_green_time = 90  # Maximum green phase duration (seconds)
        self.extension_time = 15  # Time to extend green phase (seconds)
        self.db_connection = None
        self.load_config(config_file)
        self.init_database()
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.occupancy_threshold = config.get('occupancy_threshold', 0.7)
                self.min_green_time = config.get('min_green_time', 20)
                self.max_green_time = config.get('max_green_time', 90)
                self.extension_time = config.get('extension_time', 15)
        except FileNotFoundError:
            print(f"Config file not found: {config_file}. Using defaults.")
            
    def init_database(self):
        """Initialize SQLite database for logging"""
        self.db_connection = sqlite3.connect('../../database/traffic_events.db')
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_adaptations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                junction_id TEXT,
                lane_id TEXT,
                occupancy REAL,
                current_phase INTEGER,
                action TEXT,
                new_duration INTEGER,
                reason TEXT
            )
        ''')
        self.db_connection.commit()
        
    def get_lane_occupancy(self, lane_id):
        """
        Calculate lane occupancy as percentage of lane capacity.
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            float: Occupancy percentage (0.0 to 1.0)
        """
        try:
            num_vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            # Average vehicle length + gap = 7.5 meters
            capacity = lane_length / 7.5
            occupancy = num_vehicles / capacity if capacity > 0 else 0
            return min(occupancy, 1.0)
        except traci.exceptions.TraCIException:
            return 0.0
            
    def get_junction_lanes(self, junction_id):
        """Get all incoming lanes for a junction"""
        try:
            # Get all edges connected to junction
            incoming_edges = traci.trafficlight.getControlledLinks(junction_id)
            lanes = set()
            for link_list in incoming_edges:
                for link in link_list:
                    lanes.add(link[0])  # Incoming lane
            return list(lanes)
        except traci.exceptions.TraCIException:
            return []
            
    def detect_congestion(self, junction_id):
        """
        Detect congestion at a junction and identify which approaches are congested.
        
        Returns:
            dict: Lane IDs mapped to occupancy levels
        """
        lanes = self.get_junction_lanes(junction_id)
        congestion_map = {}
        
        for lane in lanes:
            occupancy = self.get_lane_occupancy(lane)
            if occupancy > self.occupancy_threshold:
                congestion_map[lane] = occupancy
                
        return congestion_map
        
    def adapt_signal_timing(self, junction_id, step):
        """
        Adapt signal timing based on current traffic conditions.
        
        Args:
            junction_id: Traffic light junction ID
            step: Current simulation step
        """
        # Get current traffic light state
        current_state = traci.trafficlight.getRedYellowGreenState(junction_id)
        current_phase = traci.trafficlight.getPhase(junction_id)
        current_duration = traci.trafficlight.getPhaseDuration(junction_id)
        next_switch = traci.trafficlight.getNextSwitch(junction_id)
        time_until_switch = next_switch - step * 0.1  # Convert to seconds
        
        # Detect congestion
        congested_lanes = self.detect_congestion(junction_id)
        
        if not congested_lanes:
            return  # No congestion, no adaptation needed
            
        # Check if green phase can be extended for congested lanes
        green_lanes = [i for i, state in enumerate(current_state) if state == 'G']
        
        for lane, occupancy in congested_lanes.items():
            lane_index = self.get_lane_signal_index(junction_id, lane)
            
            if lane_index in green_lanes and time_until_switch < 5:
                # Lane is green and about to switch - extend it
                new_duration = min(current_duration + self.extension_time, self.max_green_time)
                
                if new_duration > current_duration:
                    traci.trafficlight.setPhaseDuration(junction_id, new_duration)
                    
                    # Log the adaptation
                    self.log_adaptation(
                        junction_id, lane, occupancy, current_phase,
                        'EXTEND_GREEN', new_duration,
                        f'Congestion detected: {occupancy:.2%} occupancy'
                    )
                    
                    print(f"[{step}] Extended green phase at {junction_id} for lane {lane} "
                          f"(occupancy: {occupancy:.2%}) to {new_duration}s")
                    
    def get_lane_signal_index(self, junction_id, lane_id):
        """Get the signal index for a specific lane at a junction"""
        try:
            controlled_links = traci.trafficlight.getControlledLinks(junction_id)
            for idx, link_list in enumerate(controlled_links):
                for link in link_list:
                    if link[0] == lane_id:
                        return idx
        except:
            pass
        return -1
        
    def handle_emergency_vehicle(self, junction_id, emergency_vehicle_id):
        """
        Provide green wave for emergency vehicles by adapting signals.
        
        Args:
            junction_id: Junction to control
            emergency_vehicle_id: ID of the emergency vehicle
        """
        try:
            # Get emergency vehicle lane
            lane_id = traci.vehicle.getLaneID(emergency_vehicle_id)
            edge_id = traci.vehicle.getRoadID(emergency_vehicle_id)
            
            # Force green for emergency vehicle direction
            current_state = list(traci.trafficlight.getRedYellowGreenState(junction_id))
            lane_index = self.get_lane_signal_index(junction_id, lane_id)
            
            if lane_index >= 0:
                # Set emergency vehicle lane to green, others to red
                new_state = ['r'] * len(current_state)
                new_state[lane_index] = 'G'
                traci.trafficlight.setRedYellowGreenState(junction_id, ''.join(new_state))
                traci.trafficlight.setPhaseDuration(junction_id, 30)  # Hold for 30 seconds
                
                # Log emergency priority
                self.log_adaptation(
                    junction_id, lane_id, 1.0, -1,
                    'EMERGENCY_PRIORITY', 30,
                    f'Emergency vehicle {emergency_vehicle_id} priority'
                )
                
                print(f"[EMERGENCY] Green wave activated at {junction_id} for {emergency_vehicle_id}")
                
        except traci.exceptions.TraCIException as e:
            print(f"Error handling emergency vehicle: {e}")
            
    def log_adaptation(self, junction_id, lane_id, occupancy, phase, action, duration, reason):
        """Log signal adaptation to database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO signal_adaptations 
            (timestamp, junction_id, lane_id, occupancy, current_phase, action, new_duration, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), junction_id, lane_id, occupancy, phase, action, duration, reason))
        self.db_connection.commit()
        
    def run(self, sumo_config, duration=3600):
        """
        Run the adaptive signal control simulation.
        
        Args:
            sumo_config: Path to SUMO configuration file
            duration: Simulation duration in seconds
        """
        # Start SUMO with TraCI
        sumo_binary = "sumo"  # Use "sumo-gui" for visualization
        sumo_cmd = [sumo_binary, "-c", sumo_config, "--start", "--quit-on-end"]
        
        print("Starting adaptive traffic signal control...")
        print(f"Occupancy threshold: {self.occupancy_threshold:.2%}")
        print(f"Green time range: {self.min_green_time}s - {self.max_green_time}s")
        print(f"Extension time: {self.extension_time}s\n")
        
        traci.start(sumo_cmd)
        
        # Get all traffic lights
        self.junctions = traci.trafficlight.getIDList()
        print(f"Controlling {len(self.junctions)} traffic lights: {self.junctions}\n")
        
        step = 0
        max_steps = duration * 10  # 0.1s step size
        
        try:
            while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                
                # Adapt signals every 10 steps (1 second)
                if step % 10 == 0:
                    # Check for emergency vehicles
                    all_vehicles = traci.vehicle.getIDList()
                    emergency_vehicles = [v for v in all_vehicles if 'ambulance' in v.lower()]
                    
                    if emergency_vehicles:
                        for ev in emergency_vehicles:
                            # Find nearest junction
                            ev_edge = traci.vehicle.getRoadID(ev)
                            for junction in self.junctions:
                                self.handle_emergency_vehicle(junction, ev)
                    
                    # Normal adaptive control
                    for junction in self.junctions:
                        self.adapt_signal_timing(junction, step)
                
                # Progress indicator
                if step % 600 == 0:  # Every minute
                    elapsed = step / 10
                    print(f"[{elapsed:.0f}s] Simulation progress: {elapsed/duration*100:.1f}%")
                
                step += 1
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            traci.close()
            self.db_connection.close()
            print("\nSimulation completed. Logs saved to database.")
            
    def generate_report(self):
        """Generate summary report of signal adaptations"""
        cursor = self.db_connection.cursor()
        
        # Total adaptations
        cursor.execute("SELECT COUNT(*) FROM signal_adaptations")
        total_adaptations = cursor.fetchone()[0]
        
        # Adaptations by type
        cursor.execute("SELECT action, COUNT(*) FROM signal_adaptations GROUP BY action")
        adaptations_by_type = cursor.fetchall()
        
        # Average occupancy when adapting
        cursor.execute("SELECT AVG(occupancy) FROM signal_adaptations WHERE action='EXTEND_GREEN'")
        avg_occupancy = cursor.fetchone()[0] or 0
        
        print("\n" + "="*60)
        print("ADAPTIVE SIGNAL CONTROL REPORT")
        print("="*60)
        print(f"Total signal adaptations: {total_adaptations}")
        print(f"\nAdaptations by type:")
        for action, count in adaptations_by_type:
            print(f"  {action}: {count}")
        print(f"\nAverage occupancy at adaptation: {avg_occupancy:.2%}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    controller = AdaptiveSignalController()
    
    if len(sys.argv) > 1:
        sumo_config = sys.argv[1]
    else:
        sumo_config = "../../sumo/scenarios/basic_traffic.sumocfg"
    
    controller.run(sumo_config, duration=3600)
    controller.generate_report()
