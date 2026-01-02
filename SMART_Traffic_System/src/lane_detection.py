"""
Lane Detection Module for SMART Traffic System
Monitors and analyzes traffic on each lane
"""

import traci
from collections import defaultdict


class LaneDetection:
    """
    Detects and monitors traffic conditions on individual lanes.
    
    Features:
    - Per-lane vehicle counting
    - Speed monitoring
    - Congestion detection
    - Queue length estimation
    - Lane occupancy calculation
    """
    
    CONGESTION_THRESHOLD_SPEED = 5.0  # m/s (18 km/h)
    CONGESTION_THRESHOLD_DENSITY = 0.15  # vehicles per meter
    
    def __init__(self):
        """Initialize lane detection system."""
        self.lane_data = defaultdict(lambda: {
            'vehicle_count': 0,
            'total_speed': 0.0,
            'avg_speed': 0.0,
            'queue_length': 0,
            'congested': False,
            'occupancy': 0.0,
            'vehicles': []
        })
        
        print("✓ Lane Detection initialized")
    
    def analyze_lanes(self, vehicle_ids):
        """
        Analyze current traffic conditions on all lanes.
        
        Args:
            vehicle_ids (list): List of vehicle IDs in simulation
        
        Returns:
            dict: Lane analysis data
        """
        # Reset lane data
        for lane_id in self.lane_data:
            self.lane_data[lane_id]['vehicle_count'] = 0
            self.lane_data[lane_id]['total_speed'] = 0.0
            self.lane_data[lane_id]['vehicles'] = []
        
        # Collect data for each vehicle
        for veh_id in vehicle_ids:
            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getLanePosition(veh_id)
                
                lane_data = self.lane_data[lane_id]
                lane_data['vehicle_count'] += 1
                lane_data['total_speed'] += speed
                lane_data['vehicles'].append({
                    'id': veh_id,
                    'speed': speed,
                    'position': position
                })
                
            except traci.exceptions.TraCIException:
                pass
        
        # Calculate statistics for each lane
        for lane_id, data in self.lane_data.items():
            if data['vehicle_count'] > 0:
                data['avg_speed'] = data['total_speed'] / data['vehicle_count']
                
                # Detect congestion
                data['congested'] = self._is_congested(data)
                
                # Calculate queue length
                data['queue_length'] = self._calculate_queue_length(data['vehicles'])
                
                # Calculate occupancy
                try:
                    lane_length = traci.lane.getLength(lane_id)
                    data['occupancy'] = (data['vehicle_count'] * 5.0) / lane_length  # Assume 5m per vehicle
                except:
                    data['occupancy'] = 0.0
        
        return dict(self.lane_data)
    
    def _is_congested(self, lane_data):
        """
        Determine if a lane is congested.
        
        Args:
            lane_data (dict): Lane statistics
        
        Returns:
            bool: True if lane is congested
        """
        # Congestion criteria: low average speed or high density
        low_speed = lane_data['avg_speed'] < self.CONGESTION_THRESHOLD_SPEED
        high_density = lane_data['occupancy'] > self.CONGESTION_THRESHOLD_DENSITY
        
        return low_speed and high_density
    
    def _calculate_queue_length(self, vehicles):
        """
        Calculate queue length (number of stopped/slow vehicles).
        
        Args:
            vehicles (list): List of vehicle data on lane
        
        Returns:
            int: Number of vehicles in queue
        """
        queue_count = 0
        for vehicle in vehicles:
            if vehicle['speed'] < 1.0:  # Nearly stopped
                queue_count += 1
        
        return queue_count
    
    def get_lane_congestion_level(self, lane_id):
        """
        Get congestion level for a specific lane.
        
        Args:
            lane_id (str): Lane identifier
        
        Returns:
            str: Congestion level ('free', 'moderate', 'heavy', 'severe')
        """
        if lane_id not in self.lane_data:
            return 'unknown'
        
        data = self.lane_data[lane_id]
        avg_speed = data['avg_speed']
        occupancy = data['occupancy']
        
        if avg_speed > 10.0 and occupancy < 0.1:
            return 'free'
        elif avg_speed > 7.0 and occupancy < 0.15:
            return 'moderate'
        elif avg_speed > 3.0:
            return 'heavy'
        else:
            return 'severe'
    
    def get_summary(self):
        """
        Get summary of all lanes.
        
        Returns:
            dict: Summary of lane conditions
        """
        return dict(self.lane_data)
    
    def get_lanes_by_edge(self, edge_id):
        """
        Get all lanes for a specific edge.
        
        Args:
            edge_id (str): Edge identifier
        
        Returns:
            list: List of lane IDs
        """
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            return [f"{edge_id}_{i}" for i in range(num_lanes)]
        except:
            return []
    
    def get_most_congested_lane(self):
        """
        Find the most congested lane in the network.
        
        Returns:
            tuple: (lane_id, congestion_level)
        """
        max_congestion = -1
        most_congested = None
        
        for lane_id, data in self.lane_data.items():
            if data['vehicle_count'] > 0:
                congestion_score = data['occupancy'] * (1.0 / (data['avg_speed'] + 0.1))
                if congestion_score > max_congestion:
                    max_congestion = congestion_score
                    most_congested = lane_id
        
        return most_congested, max_congestion
    
    def detect_lane_for_vehicle(self, vehicle_id):
        """
        Detect which lane a specific vehicle is in.
        
        Args:
            vehicle_id (str): Vehicle identifier
        
        Returns:
            str: Lane ID or None
        """
        try:
            return traci.vehicle.getLaneID(vehicle_id)
        except:
            return None
    
    def get_vehicles_in_lane(self, lane_id):
        """
        Get all vehicles currently in a specific lane.
        
        Args:
            lane_id (str): Lane identifier
        
        Returns:
            list: List of vehicle IDs
        """
        if lane_id in self.lane_data:
            return [v['id'] for v in self.lane_data[lane_id]['vehicles']]
        return []
