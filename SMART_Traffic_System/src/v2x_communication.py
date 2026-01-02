"""
V2X Communication Module for SMART Traffic System
Simulates Vehicle-to-Everything (V2X) communication
"""

import math
from collections import defaultdict


class V2XCommunication:
    """
    Handles V2X communication between vehicles and infrastructure.
    
    Features:
    - V2V (Vehicle-to-Vehicle) communication
    - V2I (Vehicle-to-Infrastructure) communication
    - Broadcast messaging within range
    - Emergency vehicle notifications
    """
    
    def __init__(self, range_meters=300, frequency=10):
        """
        Initialize V2X communication system.
        
        Args:
            range_meters (int): Communication range in meters
            frequency (int): Broadcast frequency in Hz
        """
        self.communication_range = range_meters
        self.broadcast_frequency = frequency
        self.message_queue = defaultdict(list)
        self.vehicle_positions = {}
        self.last_broadcast_time = 0
        
        print(f"✓ V2X Communication initialized (range: {range_meters}m, freq: {frequency}Hz)")
    
    def update(self, vehicle_ids, current_time):
        """
        Update V2X communication for all vehicles.
        
        Args:
            vehicle_ids (list): List of vehicle IDs in simulation
            current_time (float): Current simulation time
        """
        import traci
        
        # Update vehicle positions
        for veh_id in vehicle_ids:
            try:
                position = traci.vehicle.getPosition(veh_id)
                self.vehicle_positions[veh_id] = position
            except:
                pass
        
        # Broadcast messages at specified frequency
        if current_time - self.last_broadcast_time >= (1.0 / self.broadcast_frequency):
            self._broadcast_messages(vehicle_ids)
            self.last_broadcast_time = current_time
    
    def _broadcast_messages(self, vehicle_ids):
        """
        Broadcast V2X messages between nearby vehicles.
        
        Args:
            vehicle_ids (list): List of vehicle IDs
        """
        import traci
        
        # Clear old messages
        self.message_queue.clear()
        
        for veh_id in vehicle_ids:
            if veh_id not in self.vehicle_positions:
                continue
            
            # Get vehicle info
            try:
                veh_type = traci.vehicle.getTypeID(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                veh_lane = traci.vehicle.getLaneID(veh_id)
                veh_pos = self.vehicle_positions[veh_id]
                
                # Create message
                message = {
                    'sender': veh_id,
                    'type': veh_type,
                    'speed': veh_speed,
                    'lane': veh_lane,
                    'position': veh_pos,
                    'is_emergency': self._is_emergency_vehicle(veh_type)
                }
                
                # Broadcast to nearby vehicles
                for other_veh_id in vehicle_ids:
                    if other_veh_id != veh_id and other_veh_id in self.vehicle_positions:
                        distance = self._calculate_distance(veh_pos, self.vehicle_positions[other_veh_id])
                        
                        if distance <= self.communication_range:
                            self.message_queue[other_veh_id].append(message)
            
            except:
                pass
    
    def _calculate_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1 (tuple): First position (x, y)
            pos2 (tuple): Second position (x, y)
        
        Returns:
            float: Distance in meters
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _is_emergency_vehicle(self, vehicle_type):
        """
        Check if vehicle type is an emergency vehicle.
        
        Args:
            vehicle_type (str): Vehicle type ID
        
        Returns:
            bool: True if emergency vehicle
        """
        emergency_types = ['ambulance', 'fire_truck', 'police']
        return any(emerg_type in vehicle_type.lower() for emerg_type in emergency_types)
    
    def get_messages(self, vehicle_id):
        """
        Get messages received by a specific vehicle.
        
        Args:
            vehicle_id (str): Vehicle ID
        
        Returns:
            list: List of messages
        """
        return self.message_queue.get(vehicle_id, [])
    
    def broadcast_emergency_alert(self, emergency_vehicle_id):
        """
        Broadcast emergency vehicle alert to all nearby vehicles.
        
        Args:
            emergency_vehicle_id (str): ID of emergency vehicle
        """
        if emergency_vehicle_id not in self.vehicle_positions:
            return
        
        emerg_pos = self.vehicle_positions[emergency_vehicle_id]
        
        alert_message = {
            'type': 'EMERGENCY_ALERT',
            'vehicle_id': emergency_vehicle_id,
            'position': emerg_pos,
            'action': 'CLEAR_LANE'
        }
        
        # Send alert to all vehicles in extended range
        for veh_id in self.vehicle_positions:
            if veh_id != emergency_vehicle_id:
                distance = self._calculate_distance(emerg_pos, self.vehicle_positions[veh_id])
                if distance <= self.communication_range * 2:  # Extended range for emergencies
                    self.message_queue[veh_id].append(alert_message)
    
    def get_nearby_vehicles(self, vehicle_id, range_override=None):
        """
        Get list of vehicles within communication range.
        
        Args:
            vehicle_id (str): Reference vehicle ID
            range_override (float): Optional range override
        
        Returns:
            list: List of nearby vehicle IDs
        """
        if vehicle_id not in self.vehicle_positions:
            return []
        
        veh_pos = self.vehicle_positions[vehicle_id]
        comm_range = range_override or self.communication_range
        nearby = []
        
        for other_id, other_pos in self.vehicle_positions.items():
            if other_id != vehicle_id:
                distance = self._calculate_distance(veh_pos, other_pos)
                if distance <= comm_range:
                    nearby.append(other_id)
        
        return nearby
