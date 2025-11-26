#!/usr/bin/env python3
"""
Vehicle Controller Module

Handles individual vehicle control, state management, and behavior modeling
for the Smart Traffic Management System.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# Ensure SUMO tools are in path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

try:
    import traci
except ImportError:
    traci = None

logger = logging.getLogger('VehicleController')


class LaneChangeState(Enum):
    """Lane change states for vehicles."""
    NONE = 0
    PREPARING = 1
    EXECUTING = 2
    COMPLETED = 3


@dataclass
class VehicleState:
    """Data class representing vehicle state."""
    vehicle_id: str
    position: Tuple[float, float]
    speed: float
    acceleration: float
    lane_id: str
    lane_index: int
    angle: float
    vehicle_type: str
    route: List[str]
    route_index: int
    waiting_time: float = 0.0
    is_emergency: bool = False
    lane_change_state: LaneChangeState = LaneChangeState.NONE
    neighbors: List[str] = field(default_factory=list)


class VehicleController:
    """
    Controls and manages vehicles in the simulation.
    
    Features:
    - Track vehicle states
    - Control speed and lane changes
    - Handle emergency vehicle behavior
    - Implement car-following models
    """
    
    def __init__(self):
        """Initialize the vehicle controller."""
        self.vehicles: Dict[str, VehicleState] = {}
        self.emergency_vehicle_ids: Set[str] = set()
        
        # Vehicle type classifications
        self.emergency_types = {'ambulance', 'police', 'firetruck', 'emergency'}
        self.large_vehicle_types = {'bus', 'truck', 'firetruck'}
        
        # Control parameters
        self.safe_gap = 2.5  # meters
        self.max_decel = 4.5  # m/s^2
        self.lane_change_duration = 3.0  # seconds
        
    def update_vehicle_state(self, vehicle_id: str) -> Optional[VehicleState]:
        """
        Update and return the state of a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            VehicleState object or None if vehicle not found
        """
        if traci is None:
            return None
            
        try:
            position = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            acceleration = traci.vehicle.getAcceleration(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            angle = traci.vehicle.getAngle(vehicle_id)
            veh_type = traci.vehicle.getTypeID(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
            
            is_emergency = any(em in veh_type.lower() for em in self.emergency_types)
            
            if is_emergency:
                self.emergency_vehicle_ids.add(vehicle_id)
            
            state = VehicleState(
                vehicle_id=vehicle_id,
                position=position,
                speed=speed,
                acceleration=acceleration,
                lane_id=lane_id,
                lane_index=lane_index,
                angle=angle,
                vehicle_type=veh_type,
                route=list(route),
                route_index=route_index,
                waiting_time=waiting_time,
                is_emergency=is_emergency
            )
            
            self.vehicles[vehicle_id] = state
            return state
            
        except Exception as e:
            logger.debug(f"Could not update state for {vehicle_id}: {e}")
            return None
    
    def get_all_vehicles(self) -> Dict[str, VehicleState]:
        """
        Get states of all vehicles in simulation.
        
        Returns:
            Dictionary mapping vehicle IDs to states
        """
        if traci is None:
            return {}
            
        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            self.update_vehicle_state(veh_id)
        
        # Remove vehicles no longer in simulation
        current_ids = set(vehicle_ids)
        to_remove = [vid for vid in self.vehicles if vid not in current_ids]
        for vid in to_remove:
            del self.vehicles[vid]
            self.emergency_vehicle_ids.discard(vid)
        
        return self.vehicles
    
    def set_vehicle_speed(self, vehicle_id: str, speed: float) -> bool:
        """
        Set target speed for a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            speed: Target speed in m/s
            
        Returns:
            True if successful
        """
        if traci is None:
            return False
            
        try:
            traci.vehicle.setSpeed(vehicle_id, max(0, speed))
            return True
        except Exception as e:
            logger.debug(f"Could not set speed for {vehicle_id}: {e}")
            return False
    
    def slow_down_vehicle(self, vehicle_id: str, target_speed: float,
                          duration: float = 3.0) -> bool:
        """
        Gradually slow down a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            target_speed: Target speed in m/s
            duration: Time to reach target speed
            
        Returns:
            True if successful
        """
        if traci is None:
            return False
            
        try:
            traci.vehicle.slowDown(vehicle_id, target_speed, duration)
            return True
        except Exception as e:
            logger.debug(f"Could not slow down {vehicle_id}: {e}")
            return False
    
    def change_lane(self, vehicle_id: str, lane_index: int,
                    duration: float = 3.0) -> bool:
        """
        Request lane change for a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            lane_index: Target lane index
            duration: Duration for lane change
            
        Returns:
            True if lane change initiated
        """
        if traci is None:
            return False
            
        try:
            # Check if lane change is safe
            if self._is_lane_change_safe(vehicle_id, lane_index):
                traci.vehicle.changeLane(vehicle_id, lane_index, duration)
                if vehicle_id in self.vehicles:
                    self.vehicles[vehicle_id].lane_change_state = LaneChangeState.EXECUTING
                return True
            return False
        except Exception as e:
            logger.debug(f"Could not change lane for {vehicle_id}: {e}")
            return False
    
    def _is_lane_change_safe(self, vehicle_id: str, target_lane: int) -> bool:
        """
        Check if lane change is safe.
        
        Args:
            vehicle_id: ID of the vehicle
            target_lane: Target lane index
            
        Returns:
            True if lane change is safe
        """
        if traci is None:
            return False
            
        try:
            # Get current edge
            road_id = traci.vehicle.getRoadID(vehicle_id)
            lane_count = traci.edge.getLaneNumber(road_id)
            
            # Check if target lane exists
            if target_lane < 0 or target_lane >= lane_count:
                return False
            
            # Get vehicle position and speed
            pos = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            
            # Check for vehicles in target lane
            target_lane_id = f"{road_id}_{target_lane}"
            
            # Get vehicles on target lane
            leader = traci.vehicle.getLeader(vehicle_id, 50.0)
            if leader:
                leader_id, gap = leader
                if gap < self.safe_gap + speed * 0.5:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking lane change safety: {e}")
            return False
    
    def get_neighbors(self, vehicle_id: str, radius: float = 100.0) -> List[str]:
        """
        Get neighboring vehicles within radius.
        
        Args:
            vehicle_id: ID of the vehicle
            radius: Search radius in meters
            
        Returns:
            List of neighboring vehicle IDs
        """
        if traci is None or vehicle_id not in self.vehicles:
            return []
            
        neighbors = []
        try:
            veh_pos = self.vehicles[vehicle_id].position
            
            for other_id, other_state in self.vehicles.items():
                if other_id == vehicle_id:
                    continue
                    
                # Calculate distance
                dx = other_state.position[0] - veh_pos[0]
                dy = other_state.position[1] - veh_pos[1]
                distance = (dx**2 + dy**2)**0.5
                
                if distance <= radius:
                    neighbors.append(other_id)
            
            # Update state
            if vehicle_id in self.vehicles:
                self.vehicles[vehicle_id].neighbors = neighbors
                
        except Exception as e:
            logger.debug(f"Error getting neighbors for {vehicle_id}: {e}")
            
        return neighbors
    
    def yield_to_emergency(self, vehicle_id: str, emergency_id: str) -> bool:
        """
        Make a vehicle yield to an emergency vehicle.
        
        Args:
            vehicle_id: ID of vehicle to yield
            emergency_id: ID of emergency vehicle
            
        Returns:
            True if yield action taken
        """
        if traci is None:
            return False
            
        try:
            # Get current lane info
            road_id = traci.vehicle.getRoadID(vehicle_id)
            current_lane = traci.vehicle.getLaneIndex(vehicle_id)
            lane_count = traci.edge.getLaneNumber(road_id)
            
            # Try to move to rightmost lane
            if current_lane > 0:
                return self.change_lane(vehicle_id, current_lane - 1)
            elif lane_count > 1 and current_lane < lane_count - 1:
                return self.change_lane(vehicle_id, current_lane + 1)
            else:
                # Slow down if can't change lane
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                return self.slow_down_vehicle(vehicle_id, current_speed * 0.5)
                
        except Exception as e:
            logger.debug(f"Error yielding for {vehicle_id}: {e}")
            return False
    
    def is_emergency_vehicle(self, vehicle_id: str) -> bool:
        """Check if a vehicle is an emergency vehicle."""
        return vehicle_id in self.emergency_vehicle_ids
    
    def get_emergency_vehicles(self) -> List[str]:
        """Get list of all emergency vehicles in simulation."""
        return list(self.emergency_vehicle_ids)
    
    def calculate_time_to_collision(self, vehicle_id: str) -> Optional[float]:
        """
        Calculate time to collision with leading vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Time to collision in seconds, or None if no collision risk
        """
        if traci is None:
            return None
            
        try:
            leader = traci.vehicle.getLeader(vehicle_id, 100.0)
            if not leader:
                return None
                
            leader_id, gap = leader
            
            my_speed = traci.vehicle.getSpeed(vehicle_id)
            leader_speed = traci.vehicle.getSpeed(leader_id)
            
            relative_speed = my_speed - leader_speed
            
            if relative_speed <= 0:
                return None  # Not approaching
                
            ttc = gap / relative_speed
            return ttc if ttc > 0 else None
            
        except Exception as e:
            logger.debug(f"Error calculating TTC for {vehicle_id}: {e}")
            return None
    
    def get_vehicle_emissions(self, vehicle_id: str) -> Dict[str, float]:
        """
        Get emission data for a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Dictionary with emission values
        """
        if traci is None:
            return {}
            
        try:
            return {
                'CO2': traci.vehicle.getCO2Emission(vehicle_id),
                'CO': traci.vehicle.getCOEmission(vehicle_id),
                'HC': traci.vehicle.getHCEmission(vehicle_id),
                'NOx': traci.vehicle.getNOxEmission(vehicle_id),
                'PMx': traci.vehicle.getPMxEmission(vehicle_id),
                'fuel': traci.vehicle.getFuelConsumption(vehicle_id)
            }
        except Exception as e:
            logger.debug(f"Error getting emissions for {vehicle_id}: {e}")
            return {}
    
    def apply_speed_advisory(self, vehicle_id: str, recommended_speed: float,
                             max_deviation: float = 5.0) -> bool:
        """
        Apply a speed advisory to a vehicle (from V2I).
        
        Args:
            vehicle_id: ID of the vehicle
            recommended_speed: Recommended speed in m/s
            max_deviation: Maximum speed change per application
            
        Returns:
            True if advisory applied
        """
        if traci is None:
            return False
            
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            
            # Calculate new speed (gradual change)
            speed_diff = recommended_speed - current_speed
            speed_change = max(min(speed_diff, max_deviation), -max_deviation)
            new_speed = current_speed + speed_change * 0.3  # 30% adjustment per step
            
            # Apply speed limit
            max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
            new_speed = min(new_speed, max_speed)
            
            traci.vehicle.setSpeed(vehicle_id, max(0, new_speed))
            return True
            
        except Exception as e:
            logger.debug(f"Error applying speed advisory to {vehicle_id}: {e}")
            return False
