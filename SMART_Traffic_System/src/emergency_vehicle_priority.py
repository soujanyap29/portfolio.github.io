"""
Emergency Vehicle Priority Module for SMART Traffic System
Handles detection and priority management for emergency vehicles
"""

import traci
from collections import defaultdict


class EmergencyVehiclePriority:
    """
    Manages emergency vehicle detection and priority handling.
    
    Features:
    - Emergency vehicle detection
    - Lane clearance coordination
    - Green corridor creation
    - Priority level management
    - Automatic normalization
    """
    
    EMERGENCY_TYPES = {
        'ambulance': {'priority': 1, 'name': 'Ambulance'},
        'fire_truck': {'priority': 2, 'name': 'Fire Truck'},
        'police': {'priority': 3, 'name': 'Police'}
    }
    
    def __init__(self):
        """Initialize emergency vehicle priority system."""
        self.active_emergencies = {}
        self.lane_clearance_initiated = set()
        self.detection_range = 500  # meters
        
        print("✓ Emergency Vehicle Priority initialized")
    
    def detect_emergency_vehicles(self, vehicle_ids):
        """
        Detect emergency vehicles in the simulation.
        
        Args:
            vehicle_ids (list): List of all vehicle IDs
        
        Returns:
            list: List of detected emergency vehicle IDs
        """
        emergency_vehicles = []
        
        for veh_id in vehicle_ids:
            try:
                veh_type = traci.vehicle.getTypeID(veh_id)
                
                if self._is_emergency_type(veh_type):
                    emergency_vehicles.append(veh_id)
                    
                    # Track emergency vehicle if new
                    if veh_id not in self.active_emergencies:
                        self.active_emergencies[veh_id] = {
                            'type': veh_type,
                            'priority': self._get_priority_level(veh_type),
                            'detected_time': traci.simulation.getTime(),
                            'route': traci.vehicle.getRoute(veh_id)
                        }
            
            except traci.exceptions.TraCIException:
                pass
        
        # Remove vehicles that have left simulation
        to_remove = [v_id for v_id in self.active_emergencies if v_id not in vehicle_ids]
        for v_id in to_remove:
            del self.active_emergencies[v_id]
            if v_id in self.lane_clearance_initiated:
                self.lane_clearance_initiated.remove(v_id)
        
        return emergency_vehicles
    
    def _is_emergency_type(self, vehicle_type):
        """
        Check if vehicle type is emergency.
        
        Args:
            vehicle_type (str): Vehicle type ID
        
        Returns:
            bool: True if emergency vehicle
        """
        return any(emerg_type in vehicle_type.lower() 
                  for emerg_type in self.EMERGENCY_TYPES.keys())
    
    def _get_priority_level(self, vehicle_type):
        """
        Get priority level for vehicle type.
        
        Args:
            vehicle_type (str): Vehicle type ID
        
        Returns:
            int: Priority level (lower is higher priority)
        """
        for emerg_type, info in self.EMERGENCY_TYPES.items():
            if emerg_type in vehicle_type.lower():
                return info['priority']
        return 99  # Default low priority
    
    def activate_priority(self, emergency_vehicle_id, lane_data):
        """
        Activate priority handling for emergency vehicle.
        
        Args:
            emergency_vehicle_id (str): Emergency vehicle ID
            lane_data (dict): Current lane traffic data
        """
        if emergency_vehicle_id in self.lane_clearance_initiated:
            return  # Already activated
        
        try:
            # Get emergency vehicle information
            emerg_lane = traci.vehicle.getLaneID(emergency_vehicle_id)
            emerg_edge = traci.vehicle.getRoadID(emergency_vehicle_id)
            
            print(f"\n{'='*60}")
            print(f"EMERGENCY PRIORITY ACTIVATED")
            print(f"Vehicle: {emergency_vehicle_id}")
            print(f"Type: {self.active_emergencies[emergency_vehicle_id]['type']}")
            print(f"Lane: {emerg_lane}")
            print(f"{'='*60}")
            
            # Initiate lane clearance
            self._clear_lane(emergency_vehicle_id, emerg_lane, lane_data)
            
            # Mark as initiated
            self.lane_clearance_initiated.add(emergency_vehicle_id)
            
        except Exception as e:
            print(f"Error activating emergency priority: {e}")
    
    def _clear_lane(self, emergency_vehicle_id, lane_id, lane_data):
        """
        Clear vehicles from emergency vehicle's lane.
        
        Args:
            emergency_vehicle_id (str): Emergency vehicle ID
            lane_id (str): Lane to clear
            lane_data (dict): Current lane data
        """
        if lane_id not in lane_data:
            return
        
        vehicles_in_lane = lane_data[lane_id]['vehicles']
        emerg_pos = traci.vehicle.getLanePosition(emergency_vehicle_id)
        
        cleared_count = 0
        
        for vehicle_info in vehicles_in_lane:
            veh_id = vehicle_info['id']
            
            if veh_id == emergency_vehicle_id:
                continue
            
            try:
                veh_pos = vehicle_info['position']
                
                # If vehicle is ahead of emergency vehicle
                if veh_pos > emerg_pos:
                    # Try to move vehicle to adjacent lane
                    if self._move_to_adjacent_lane(veh_id, lane_id):
                        cleared_count += 1
                    else:
                        # If can't change lanes, try to speed up
                        self._increase_speed(veh_id)
            
            except Exception as e:
                pass
        
        if cleared_count > 0:
            print(f"  → Cleared {cleared_count} vehicles from {lane_id}")
    
    def _move_to_adjacent_lane(self, vehicle_id, current_lane):
        """
        Attempt to move vehicle to adjacent lane.
        
        Args:
            vehicle_id (str): Vehicle to move
            current_lane (str): Current lane ID
        
        Returns:
            bool: True if successful
        """
        try:
            # Get lane index
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            num_lanes = traci.edge.getLaneNumber(edge_id)
            
            # Try to change to right lane first (if exists)
            if lane_index < num_lanes - 1:
                target_lane = lane_index + 1
                traci.vehicle.changeLane(vehicle_id, target_lane, 5.0)  # 5 second duration
                return True
            # Try left lane
            elif lane_index > 0:
                target_lane = lane_index - 1
                traci.vehicle.changeLane(vehicle_id, target_lane, 5.0)
                return True
        
        except Exception as e:
            pass
        
        return False
    
    def _increase_speed(self, vehicle_id):
        """
        Increase vehicle speed to clear path.
        
        Args:
            vehicle_id (str): Vehicle ID
        """
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
            
            # Set speed to 90% of max
            target_speed = max_speed * 0.9
            traci.vehicle.setSpeed(vehicle_id, max(target_speed, current_speed))
        
        except Exception as e:
            pass
    
    def create_green_corridor(self, emergency_vehicle_id, traffic_lights):
        """
        Create green corridor along emergency vehicle route.
        
        Args:
            emergency_vehicle_id (str): Emergency vehicle ID
            traffic_lights (list): List of traffic light IDs
        """
        try:
            # Get vehicle route
            route = traci.vehicle.getRoute(emergency_vehicle_id)
            current_edge = traci.vehicle.getRoadID(emergency_vehicle_id)
            
            # Find current position in route
            try:
                current_idx = route.index(current_edge)
            except ValueError:
                current_idx = 0
            
            # Get upcoming edges (next 3-5 intersections)
            upcoming_edges = route[current_idx:min(current_idx + 5, len(route))]
            
            print(f"  → Creating green corridor for {emergency_vehicle_id}")
            print(f"  → Route: {' -> '.join(upcoming_edges[:3])}...")
        
        except Exception as e:
            print(f"Error creating green corridor: {e}")
    
    def get_highest_priority_vehicle(self):
        """
        Get the emergency vehicle with highest priority.
        
        Returns:
            str: Vehicle ID or None
        """
        if not self.active_emergencies:
            return None
        
        # Find vehicle with lowest priority number (highest priority)
        highest_priority = min(self.active_emergencies.items(),
                             key=lambda x: x[1]['priority'])
        
        return highest_priority[0]
    
    def normalize_traffic(self, vehicle_id):
        """
        Return traffic to normal after emergency vehicle passes.
        
        Args:
            vehicle_id (str): Emergency vehicle that has passed
        """
        if vehicle_id in self.lane_clearance_initiated:
            self.lane_clearance_initiated.remove(vehicle_id)
            print(f"  → Normalizing traffic after {vehicle_id}")
    
    def get_active_emergencies(self):
        """
        Get list of active emergency vehicles.
        
        Returns:
            dict: Active emergency vehicles with details
        """
        return self.active_emergencies.copy()
