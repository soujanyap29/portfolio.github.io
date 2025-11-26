#!/usr/bin/env python3
"""
Emergency Vehicle Handler Module

Handles emergency vehicle operations including:
- Emergency vehicle detection
- Signal preemption requests
- Path clearing coordination
- Priority routing
- Recovery after passage
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math
import time

# Ensure SUMO tools are in path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

try:
    import traci
except ImportError:
    traci = None

logger = logging.getLogger('EmergencyHandler')


@dataclass
class EmergencyVehicleState:
    """State of an emergency vehicle."""
    vehicle_id: str
    vehicle_type: str
    position: Tuple[float, float]
    speed: float
    route: List[str]
    route_index: int
    approaching_signals: List[str] = field(default_factory=list)
    preempted_signals: Set[str] = field(default_factory=set)
    yielding_vehicles: Set[str] = field(default_factory=set)
    active: bool = True


@dataclass
class PreemptionRequest:
    """Signal preemption request."""
    emergency_id: str
    signal_id: str
    request_time: float
    estimated_arrival: float
    priority: int = 1  # Higher = more urgent


class EmergencyHandler:
    """
    Handles emergency vehicle operations.
    
    Features:
    - Emergency vehicle tracking
    - Signal preemption coordination
    - Vehicle yielding management
    - Path clearing
    - Recovery after passage
    """
    
    def __init__(self, detection_range: float = 200.0,
                 preemption_advance_time: float = 30.0):
        """
        Initialize emergency handler.
        
        Args:
            detection_range: Range for detecting approaching signals
            preemption_advance_time: Time before arrival to request preemption
        """
        self.detection_range = detection_range
        self.preemption_advance_time = preemption_advance_time
        
        # Emergency vehicle tracking
        self.emergency_vehicles: Dict[str, EmergencyVehicleState] = {}
        
        # Active preemption requests
        self.preemption_requests: Dict[str, PreemptionRequest] = {}
        
        # Signal positions (cached)
        self.signal_positions: Dict[str, Tuple[float, float]] = {}
        
        # Statistics
        self.total_preemptions = 0
        self.total_yield_requests = 0
        
        # Emergency type priorities
        self.priority_map = {
            'ambulance': 3,
            'firetruck': 3,
            'fire': 3,
            'police': 2,
            'emergency': 1
        }
        
    def handle_emergency_vehicles(self, emergency_ids: List[str],
                                   vehicle_states: Dict,
                                   traffic_light_controller) -> None:
        """
        Handle all emergency vehicles in the simulation.
        
        Args:
            emergency_ids: List of emergency vehicle IDs
            vehicle_states: All vehicle states
            traffic_light_controller: Traffic light controller instance
        """
        current_time = time.time()
        
        # Update emergency vehicle states
        for em_id in emergency_ids:
            if em_id not in self.emergency_vehicles:
                self._register_emergency_vehicle(em_id, vehicle_states.get(em_id, {}))
            else:
                self._update_emergency_vehicle(em_id, vehicle_states.get(em_id, {}))
        
        # Remove vehicles that are no longer emergency or left
        to_remove = []
        for em_id in self.emergency_vehicles:
            if em_id not in emergency_ids:
                to_remove.append(em_id)
        
        for em_id in to_remove:
            self._handle_emergency_departure(em_id, traffic_light_controller)
        
        # Process each emergency vehicle
        for em_id, em_state in self.emergency_vehicles.items():
            if not em_state.active:
                continue
            
            # Detect approaching signals
            self._detect_approaching_signals(em_state)
            
            # Request preemption for approaching signals
            self._request_preemptions(em_state, traffic_light_controller, current_time)
            
            # Request yielding from nearby vehicles
            self._request_vehicle_yielding(em_state, vehicle_states)
    
    def _register_emergency_vehicle(self, vehicle_id: str, state: Dict) -> None:
        """Register a new emergency vehicle."""
        vehicle_type = state.get('type', 'emergency').lower()
        
        self.emergency_vehicles[vehicle_id] = EmergencyVehicleState(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            position=state.get('position', (0, 0)),
            speed=state.get('speed', 0),
            route=state.get('route', []),
            route_index=state.get('route_index', 0)
        )
        
        logger.info(f"Registered emergency vehicle: {vehicle_id} ({vehicle_type})")
    
    def _update_emergency_vehicle(self, vehicle_id: str, state: Dict) -> None:
        """Update emergency vehicle state."""
        if vehicle_id not in self.emergency_vehicles:
            return
        
        em_state = self.emergency_vehicles[vehicle_id]
        em_state.position = state.get('position', em_state.position)
        em_state.speed = state.get('speed', em_state.speed)
        em_state.route_index = state.get('route_index', em_state.route_index)
    
    def _handle_emergency_departure(self, vehicle_id: str,
                                     traffic_light_controller) -> None:
        """Handle when an emergency vehicle leaves or is no longer emergency."""
        if vehicle_id not in self.emergency_vehicles:
            return
        
        em_state = self.emergency_vehicles[vehicle_id]
        
        # Clear all preemptions for this vehicle
        for signal_id in em_state.preempted_signals:
            traffic_light_controller.clear_preemption(signal_id, vehicle_id)
            
            # Remove preemption request
            req_key = f"{vehicle_id}_{signal_id}"
            if req_key in self.preemption_requests:
                del self.preemption_requests[req_key]
        
        del self.emergency_vehicles[vehicle_id]
        logger.info(f"Emergency vehicle departed: {vehicle_id}")
    
    def _detect_approaching_signals(self, em_state: EmergencyVehicleState) -> None:
        """Detect traffic signals the emergency vehicle is approaching."""
        if traci is None:
            return
        
        em_state.approaching_signals = []
        
        try:
            # Get all traffic lights
            tl_ids = traci.trafficlight.getIDList()
            
            for tl_id in tl_ids:
                # Cache signal position
                if tl_id not in self.signal_positions:
                    try:
                        self.signal_positions[tl_id] = traci.junction.getPosition(tl_id)
                    except Exception:
                        continue
                
                signal_pos = self.signal_positions[tl_id]
                
                # Calculate distance
                distance = self._calculate_distance(em_state.position, signal_pos)
                
                if distance <= self.detection_range:
                    # Check if on route (simplified)
                    em_state.approaching_signals.append((tl_id, distance))
            
            # Sort by distance
            em_state.approaching_signals.sort(key=lambda x: x[1])
            
        except Exception as e:
            logger.debug(f"Error detecting signals: {e}")
    
    def _request_preemptions(self, em_state: EmergencyVehicleState,
                              traffic_light_controller,
                              current_time: float) -> None:
        """Request signal preemption for approaching signals."""
        for signal_id, distance in em_state.approaching_signals:
            # Skip if already preempted
            if signal_id in em_state.preempted_signals:
                continue
            
            # Calculate ETA
            speed = max(em_state.speed, 1.0)
            eta = distance / speed
            
            # Request preemption if within advance time
            if eta <= self.preemption_advance_time:
                # Get priority
                priority = self._get_priority(em_state.vehicle_type)
                
                # Create preemption request
                req_key = f"{em_state.vehicle_id}_{signal_id}"
                self.preemption_requests[req_key] = PreemptionRequest(
                    emergency_id=em_state.vehicle_id,
                    signal_id=signal_id,
                    request_time=current_time,
                    estimated_arrival=current_time + eta,
                    priority=priority
                )
                
                # Request preemption from traffic light controller
                success = traffic_light_controller.request_preemption(
                    signal_id, em_state.vehicle_id
                )
                
                if success:
                    em_state.preempted_signals.add(signal_id)
                    self.total_preemptions += 1
                    logger.info(f"Preemption granted at {signal_id} for {em_state.vehicle_id}")
        
        # Clear preemptions for passed signals
        self._clear_passed_signals(em_state, traffic_light_controller)
    
    def _clear_passed_signals(self, em_state: EmergencyVehicleState,
                               traffic_light_controller) -> None:
        """Clear preemptions for signals that have been passed."""
        to_clear = []
        
        for signal_id in em_state.preempted_signals:
            if signal_id not in self.signal_positions:
                continue
            
            signal_pos = self.signal_positions[signal_id]
            distance = self._calculate_distance(em_state.position, signal_pos)
            
            # Consider passed if beyond signal and moving away
            # Simplified: if beyond detection range
            if distance > self.detection_range:
                to_clear.append(signal_id)
        
        for signal_id in to_clear:
            traffic_light_controller.clear_preemption(signal_id, em_state.vehicle_id)
            em_state.preempted_signals.discard(signal_id)
            
            req_key = f"{em_state.vehicle_id}_{signal_id}"
            if req_key in self.preemption_requests:
                del self.preemption_requests[req_key]
    
    def _request_vehicle_yielding(self, em_state: EmergencyVehicleState,
                                   vehicle_states: Dict) -> None:
        """Request nearby vehicles to yield."""
        if traci is None:
            return
        
        yield_range = 100.0  # meters
        new_yielding = set()
        
        for veh_id, state in vehicle_states.items():
            # Skip emergency vehicles
            if veh_id in self.emergency_vehicles:
                continue
            
            veh_pos = state.get('position', (0, 0))
            distance = self._calculate_distance(em_state.position, veh_pos)
            
            if distance <= yield_range:
                # Check if vehicle is ahead
                if self._is_vehicle_ahead(em_state, state):
                    new_yielding.add(veh_id)
                    
                    if veh_id not in em_state.yielding_vehicles:
                        self._request_yield(veh_id, em_state.vehicle_id)
                        self.total_yield_requests += 1
        
        em_state.yielding_vehicles = new_yielding
    
    def _is_vehicle_ahead(self, em_state: EmergencyVehicleState,
                           other_state: Dict) -> bool:
        """Check if another vehicle is ahead of emergency vehicle."""
        # Simplified check based on position
        em_pos = em_state.position
        other_pos = other_state.get('position', (0, 0))
        
        # Check if on same lane/road
        # For simplicity, consider all vehicles within range as needing to yield
        return True
    
    def _request_yield(self, vehicle_id: str, emergency_id: str) -> None:
        """Request a vehicle to yield to emergency vehicle."""
        if traci is None:
            return
        
        try:
            # Get current lane info
            road_id = traci.vehicle.getRoadID(vehicle_id)
            current_lane = traci.vehicle.getLaneIndex(vehicle_id)
            lane_count = traci.edge.getLaneNumber(road_id)
            
            # Try to move to rightmost lane
            if current_lane > 0:
                traci.vehicle.changeLane(vehicle_id, current_lane - 1, 5.0)
            elif lane_count > 1:
                # If already in rightmost, slow down
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                traci.vehicle.setSpeed(vehicle_id, current_speed * 0.5)
                
        except Exception as e:
            logger.debug(f"Could not request yield from {vehicle_id}: {e}")
    
    def _get_priority(self, vehicle_type: str) -> int:
        """Get priority level for emergency vehicle type."""
        vehicle_type = vehicle_type.lower()
        for type_key, priority in self.priority_map.items():
            if type_key in vehicle_type:
                return priority
        return 1
    
    def _calculate_distance(self, pos1: Tuple[float, float],
                            pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_emergency_vehicle(self, vehicle_id: str) -> bool:
        """Check if a vehicle is tracked as emergency."""
        return vehicle_id in self.emergency_vehicles
    
    def get_emergency_vehicles(self) -> List[str]:
        """Get list of all emergency vehicle IDs."""
        return list(self.emergency_vehicles.keys())
    
    def get_preempted_signals(self, vehicle_id: str) -> Set[str]:
        """Get signals preempted for an emergency vehicle."""
        if vehicle_id in self.emergency_vehicles:
            return self.emergency_vehicles[vehicle_id].preempted_signals.copy()
        return set()
    
    def get_statistics(self) -> Dict:
        """Get emergency handling statistics."""
        return {
            'active_emergency_vehicles': len(self.emergency_vehicles),
            'active_preemptions': sum(
                len(em.preempted_signals) for em in self.emergency_vehicles.values()
            ),
            'total_preemptions': self.total_preemptions,
            'total_yield_requests': self.total_yield_requests,
            'active_yielding_vehicles': sum(
                len(em.yielding_vehicles) for em in self.emergency_vehicles.values()
            )
        }
