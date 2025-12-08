"""
Emergency Vehicle Priority System
Part of Smart Traffic Management System

Implements real-time emergency vehicle detection and priority signal control.
Provides green wave corridors and forces other vehicles to yield.

Course Mappings:
- Operating Systems: Priority scheduling, interrupt handling
- Computer Networks: Real-time message prioritization
- OOPS: Inheritance, polymorphism in vehicle classes
"""

import traci
import sys
import sqlite3
import time
from collections import defaultdict

class EmergencyVehicleManager:
    """
    Manages emergency vehicle priority throughout the traffic network.
    
    Features:
    - Real-time emergency vehicle detection
    - Green wave corridor creation
    - Traffic halting for emergency passage
    - Route prediction and pre-emptive signal control
    """
    
    def __init__(self):
        """Initialize emergency vehicle management system"""
        self.active_emergencies = {}  # vehicle_id -> status
        self.green_wave_junctions = defaultdict(list)  # vehicle_id -> junctions
        self.db_connection = None
        self.init_database()
        
    def init_database(self):
        """Initialize database for emergency event logging"""
        self.db_connection = sqlite3.connect('../../database/emergency_events.db')
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                vehicle_id TEXT,
                event_type TEXT,
                junction_id TEXT,
                lane_id TEXT,
                speed REAL,
                halted_vehicles INTEGER,
                response_time REAL,
                notes TEXT
            )
        ''')
        self.db_connection.commit()
        
    def detect_emergency_vehicles(self):
        """
        Detect emergency vehicles in the simulation.
        
        Returns:
            list: IDs of active emergency vehicles
        """
        all_vehicles = traci.vehicle.getIDList()
        emergency_vehicles = []
        
        for veh_id in all_vehicles:
            try:
                veh_type = traci.vehicle.getTypeID(veh_id)
                # Emergency vehicles have type "emergency" or name contains "ambulance"
                if veh_type == 'emergency' or 'ambulance' in veh_id.lower():
                    emergency_vehicles.append(veh_id)
                    
                    if veh_id not in self.active_emergencies:
                        self.active_emergencies[veh_id] = {
                            'start_time': time.time(),
                            'start_pos': traci.vehicle.getPosition(veh_id),
                            'route': traci.vehicle.getRoute(veh_id)
                        }
                        print(f"[EMERGENCY] Detected emergency vehicle: {veh_id}")
                        self.log_event(veh_id, 'DETECTED', None, None, 0, 0, 0, 
                                     f"Emergency vehicle entered simulation")
            except:
                pass
                
        return emergency_vehicles
        
    def get_upcoming_junctions(self, vehicle_id):
        """
        Get list of junctions the emergency vehicle will encounter.
        
        Args:
            vehicle_id: Emergency vehicle ID
            
        Returns:
            list: Junction IDs along the route
        """
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            
            # Find current position in route
            try:
                current_index = route.index(current_edge)
                remaining_route = route[current_index:]
            except ValueError:
                remaining_route = route
            
            # Get junctions from edges
            junctions = []
            for edge_id in remaining_route[:5]:  # Look ahead 5 edges
                # Get junction at end of edge
                try:
                    edge_to_node = traci.edge.getToJunction(edge_id)
                    # Check if it's a traffic light
                    tl_list = traci.trafficlight.getIDList()
                    if edge_to_node in tl_list:
                        junctions.append(edge_to_node)
                except:
                    pass
                    
            return junctions
        except:
            return []
            
    def create_green_wave(self, vehicle_id, junctions):
        """
        Create green wave corridor for emergency vehicle.
        
        Args:
            vehicle_id: Emergency vehicle ID
            junctions: List of junctions to control
        """
        try:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            
            for junction_id in junctions:
                if junction_id in traci.trafficlight.getIDList():
                    # Get current state
                    current_state = list(traci.trafficlight.getRedYellowGreenState(junction_id))
                    
                    # Find signal index for emergency vehicle's approach
                    controlled_links = traci.trafficlight.getControlledLinks(junction_id)
                    
                    for idx, link_list in enumerate(controlled_links):
                        for link in link_list:
                            incoming_lane = link[0]
                            incoming_edge = incoming_lane.rsplit('_', 1)[0]
                            
                            if incoming_edge == edge_id or incoming_lane == lane_id:
                                # Set this direction to green
                                current_state[idx] = 'G'
                            else:
                                # Set other directions to red
                                current_state[idx] = 'r'
                    
                    # Apply new state
                    traci.trafficlight.setRedYellowGreenState(junction_id, ''.join(current_state))
                    traci.trafficlight.setPhaseDuration(junction_id, 60)
                    
                    self.green_wave_junctions[vehicle_id].append(junction_id)
                    
                    print(f"[GREEN WAVE] Activated at {junction_id} for {vehicle_id}")
                    self.log_event(vehicle_id, 'GREEN_WAVE', junction_id, lane_id, 
                                 traci.vehicle.getSpeed(vehicle_id), 0, 0.0,
                                 "Green wave corridor activated")
                    
        except Exception as e:
            print(f"Error creating green wave: {e}")
            
    def halt_conflicting_traffic(self, vehicle_id, radius=100):
        """
        Force nearby vehicles to slow down and yield to emergency vehicle.
        
        Args:
            vehicle_id: Emergency vehicle ID
            radius: Radius in meters to affect surrounding vehicles
        """
        try:
            ev_pos = traci.vehicle.getPosition(vehicle_id)
            ev_edge = traci.vehicle.getRoadID(vehicle_id)
            
            all_vehicles = traci.vehicle.getIDList()
            halted_count = 0
            
            for veh_id in all_vehicles:
                if veh_id == vehicle_id:
                    continue
                    
                try:
                    veh_pos = traci.vehicle.getPosition(veh_id)
                    veh_edge = traci.vehicle.getRoadID(veh_id)
                    
                    # Calculate distance
                    distance = ((ev_pos[0] - veh_pos[0])**2 + (ev_pos[1] - veh_pos[1])**2)**0.5
                    
                    if distance < radius:
                        # Check if vehicle is on conflicting approach
                        if veh_edge != ev_edge:
                            # Force vehicle to slow down
                            traci.vehicle.setSpeed(veh_id, 0)  # Full stop
                            traci.vehicle.setColor(veh_id, (255, 255, 0, 255))  # Yellow color
                            halted_count += 1
                except:
                    pass
            
            if halted_count > 0:
                print(f"[HALT] {halted_count} vehicles halted for {vehicle_id}")
                
            return halted_count
            
        except Exception as e:
            print(f"Error halting traffic: {e}")
            return 0
            
    def manage_emergency_vehicle(self, vehicle_id, step):
        """
        Manage single emergency vehicle - coordinate all priority actions.
        
        Args:
            vehicle_id: Emergency vehicle ID
            step: Current simulation step
        """
        try:
            # Get vehicle status
            speed = traci.vehicle.getSpeed(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            
            # Create green wave for upcoming junctions
            upcoming_junctions = self.get_upcoming_junctions(vehicle_id)
            if upcoming_junctions:
                self.create_green_wave(vehicle_id, upcoming_junctions[:2])  # Next 2 junctions
            
            # Halt conflicting traffic
            halted = self.halt_conflicting_traffic(vehicle_id, radius=100)
            
            # Log status every 5 seconds
            if step % 50 == 0:
                elapsed_time = time.time() - self.active_emergencies[vehicle_id]['start_time']
                self.log_event(vehicle_id, 'STATUS_UPDATE', None, lane_id, 
                             speed, halted, elapsed_time,
                             f"Speed: {speed:.2f} m/s, Halted: {halted} vehicles")
                
        except Exception as e:
            print(f"Error managing emergency vehicle {vehicle_id}: {e}")
            
    def release_green_wave(self, vehicle_id):
        """
        Release green wave junctions back to normal operation.
        
        Args:
            vehicle_id: Emergency vehicle ID
        """
        junctions = self.green_wave_junctions.get(vehicle_id, [])
        
        for junction_id in junctions:
            try:
                # Reset to program 0 (default)
                traci.trafficlight.setProgram(junction_id, "0")
                print(f"[RELEASE] Junction {junction_id} returned to normal operation")
            except:
                pass
                
        if vehicle_id in self.green_wave_junctions:
            del self.green_wave_junctions[vehicle_id]
            
    def release_halted_vehicles(self):
        """Release all halted vehicles back to normal operation"""
        all_vehicles = traci.vehicle.getIDList()
        
        for veh_id in all_vehicles:
            try:
                # Check if vehicle is halted (speed limit set to 0)
                if traci.vehicle.getSpeed(veh_id) == 0:
                    # Release speed limit
                    traci.vehicle.setSpeed(veh_id, -1)  # -1 means resume normal driving
                    # Reset color
                    traci.vehicle.setColor(veh_id, (255, 255, 0, 255))  # Yellow (default)
            except:
                pass
                
    def log_event(self, vehicle_id, event_type, junction_id, lane_id, speed, 
                  halted_vehicles, response_time, notes):
        """Log emergency event to database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO emergency_events 
            (timestamp, vehicle_id, event_type, junction_id, lane_id, speed, 
             halted_vehicles, response_time, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), vehicle_id, event_type, junction_id, lane_id, speed,
              halted_vehicles, response_time, notes))
        self.db_connection.commit()
        
    def run(self, sumo_config, duration=3600):
        """
        Run emergency vehicle priority system.
        
        Args:
            sumo_config: Path to SUMO configuration file
            duration: Simulation duration in seconds
        """
        sumo_binary = "sumo"
        sumo_cmd = [sumo_binary, "-c", sumo_config, "--start", "--quit-on-end"]
        
        print("Starting Emergency Vehicle Priority System...")
        print("="*60)
        
        traci.start(sumo_cmd)
        
        step = 0
        max_steps = duration * 10
        
        try:
            while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                
                # Detect and manage emergency vehicles
                emergency_vehicles = self.detect_emergency_vehicles()
                
                for ev_id in emergency_vehicles:
                    self.manage_emergency_vehicle(ev_id, step)
                
                # Check for completed emergency vehicles
                completed = [vid for vid in self.active_emergencies.keys() 
                           if vid not in emergency_vehicles]
                
                for vid in completed:
                    elapsed = time.time() - self.active_emergencies[vid]['start_time']
                    print(f"[COMPLETE] Emergency vehicle {vid} completed route in {elapsed:.1f}s")
                    self.log_event(vid, 'COMPLETED', None, None, 0, 0, elapsed,
                                 f"Emergency response completed")
                    self.release_green_wave(vid)
                    del self.active_emergencies[vid]
                
                # Release halted vehicles periodically
                if step % 50 == 0:
                    self.release_halted_vehicles()
                
                step += 1
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            traci.close()
            self.db_connection.close()
            print("\nEmergency vehicle management completed.")
            self.generate_report()
            
    def generate_report(self):
        """Generate emergency vehicle performance report"""
        cursor = self.db_connection.cursor()
        
        # Total events
        cursor.execute("SELECT COUNT(*) FROM emergency_events")
        total_events = cursor.fetchone()[0]
        
        # Emergency vehicles
        cursor.execute("SELECT COUNT(DISTINCT vehicle_id) FROM emergency_events")
        total_vehicles = cursor.fetchone()[0]
        
        # Average response time
        cursor.execute("""
            SELECT AVG(response_time) FROM emergency_events 
            WHERE event_type='COMPLETED'
        """)
        avg_response = cursor.fetchone()[0] or 0
        
        # Total vehicles halted
        cursor.execute("SELECT SUM(halted_vehicles) FROM emergency_events")
        total_halted = cursor.fetchone()[0] or 0
        
        print("\n" + "="*60)
        print("EMERGENCY VEHICLE PRIORITY REPORT")
        print("="*60)
        print(f"Total emergency vehicles: {total_vehicles}")
        print(f"Total priority events: {total_events}")
        print(f"Average response time: {avg_response:.2f} seconds")
        print(f"Total vehicles halted: {total_halted}")
        print("="*60)


if __name__ == "__main__":
    manager = EmergencyVehicleManager()
    
    if len(sys.argv) > 1:
        sumo_config = sys.argv[1]
    else:
        sumo_config = "../../sumo/scenarios/basic_traffic.sumocfg"
    
    manager.run(sumo_config, duration=3600)
