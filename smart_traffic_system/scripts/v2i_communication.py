#!/usr/bin/env python3
"""
V2I (Vehicle-to-Infrastructure) Communication Module

Handles all vehicle-to-infrastructure communication including:
- Signal Phase and Timing (SPaT) information
- Green wave recommendations
- Queue length information
- Speed advisories
- Emergency vehicle coordination with signals
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

logger = logging.getLogger('V2ICommunication')


@dataclass
class RSUState:
    """Roadside Unit state."""
    rsu_id: str
    position: Tuple[float, float]
    traffic_light_id: str
    communication_range: float
    connected_vehicles: Set[str] = field(default_factory=set)


@dataclass
class SPaTMessage:
    """Signal Phase and Timing message."""
    rsu_id: str
    timestamp: float
    current_phase: str
    current_state: str
    time_to_change: float
    next_phase: str
    cycle_time: float
    queue_length: int
    congestion_level: str  # 'low', 'medium', 'high'


@dataclass
class SpeedAdvisory:
    """Speed advisory message."""
    rsu_id: str
    timestamp: float
    recommended_speed: float
    reason: str  # 'green_wave', 'queue', 'congestion', 'safety'
    distance_to_signal: float
    time_to_green: float


class V2ICommunication:
    """
    Manages vehicle-to-infrastructure communication.
    
    Features:
    - RSU (Roadside Unit) registration
    - SPaT message broadcasting
    - Speed advisory calculation
    - Green wave optimization
    - Queue management
    """
    
    def __init__(self, communication_range: float = 200.0):
        """
        Initialize V2I communication manager.
        
        Args:
            communication_range: Communication range in meters
        """
        self.communication_range = communication_range
        
        # RSU registry
        self.rsus: Dict[str, RSUState] = {}
        
        # Message buffers
        self.vehicle_messages: Dict[str, Dict] = defaultdict(dict)
        
        # Enable/disable communication
        self.enabled = True
        
        # Statistics
        self.total_spat_messages = 0
        self.total_advisories = 0
        
        # Green wave corridors
        self.green_wave_corridors: Dict[str, List[str]] = {}
        self.green_wave_speed = 50 / 3.6  # 50 km/h in m/s
        
    def register_rsu(self, traffic_light_id: str, 
                     position: Tuple[float, float] = None) -> None:
        """
        Register a Roadside Unit at a traffic light.
        
        Args:
            traffic_light_id: Associated traffic light ID
            position: RSU position (auto-detected if None)
        """
        if traci is None:
            return
            
        try:
            if position is None:
                # Get position from junction
                position = traci.junction.getPosition(traffic_light_id)
            
            self.rsus[traffic_light_id] = RSUState(
                rsu_id=traffic_light_id,
                position=position,
                traffic_light_id=traffic_light_id,
                communication_range=self.communication_range
            )
            
            logger.debug(f"Registered RSU at {traffic_light_id}")
            
        except Exception as e:
            logger.warning(f"Could not register RSU at {traffic_light_id}: {e}")
    
    def process_communication(self, vehicle_states: Dict) -> Dict:
        """
        Process V2I communication for all vehicles.
        
        Args:
            vehicle_states: Dictionary of vehicle states
            
        Returns:
            Dictionary mapping vehicle IDs to V2I messages
        """
        if not self.enabled:
            return {}
            
        current_time = time.time()
        messages = {}
        
        for veh_id, state in vehicle_states.items():
            veh_pos = state.get('position', (0, 0))
            
            # Find RSUs in range
            nearby_rsus = self._find_rsus_in_range(veh_pos)
            
            if nearby_rsus:
                messages[veh_id] = self._generate_messages_for_vehicle(
                    veh_id, state, nearby_rsus, current_time
                )
        
        return messages
    
    def _find_rsus_in_range(self, position: Tuple[float, float]) -> List[RSUState]:
        """
        Find RSUs within communication range.
        
        Args:
            position: Vehicle position
            
        Returns:
            List of RSUs in range
        """
        nearby = []
        
        for rsu_id, rsu in self.rsus.items():
            distance = self._calculate_distance(position, rsu.position)
            if distance <= rsu.communication_range:
                nearby.append(rsu)
        
        return nearby
    
    def _generate_messages_for_vehicle(self, vehicle_id: str, state: Dict,
                                        nearby_rsus: List[RSUState],
                                        current_time: float) -> Dict:
        """
        Generate V2I messages for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            state: Vehicle state
            nearby_rsus: List of nearby RSUs
            current_time: Current timestamp
            
        Returns:
            Dictionary with V2I messages
        """
        messages = {
            'spat': [],
            'advisories': [],
            'queue_info': []
        }
        
        veh_pos = state.get('position', (0, 0))
        veh_speed = state.get('speed', 0)
        veh_heading = state.get('angle', 0)
        
        for rsu in nearby_rsus:
            # Add vehicle to RSU's connected list
            rsu.connected_vehicles.add(vehicle_id)
            
            # Generate SPaT message
            spat = self._generate_spat(rsu, current_time)
            if spat:
                messages['spat'].append(spat)
                self.total_spat_messages += 1
            
            # Calculate and send speed advisory
            distance = self._calculate_distance(veh_pos, rsu.position)
            advisory = self._calculate_speed_advisory(
                rsu, veh_speed, distance, current_time
            )
            if advisory:
                messages['advisories'].append(advisory)
                self.total_advisories += 1
            
            # Get queue information
            queue_info = self._get_queue_info(rsu)
            if queue_info:
                messages['queue_info'].append(queue_info)
        
        # Calculate recommended speed based on all information
        if messages['advisories']:
            messages['recommended_speed'] = self._aggregate_speed_advisory(
                messages['advisories']
            )
        
        return messages
    
    def _generate_spat(self, rsu: RSUState, current_time: float) -> Optional[Dict]:
        """
        Generate Signal Phase and Timing message.
        
        Args:
            rsu: RSU state
            current_time: Current timestamp
            
        Returns:
            SPaT message dictionary
        """
        if traci is None:
            return None
            
        try:
            tl_id = rsu.traffic_light_id
            
            current_phase = traci.trafficlight.getPhase(tl_id)
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            
            # Get program logic
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            phase = logic.phases[current_phase]
            
            # Calculate time to change
            # Note: This is simplified; actual implementation would track phase time
            next_switch = traci.trafficlight.getNextSwitch(tl_id)
            sim_time = traci.simulation.getTime()
            time_to_change = max(0, next_switch - sim_time)
            
            # Calculate next phase
            num_phases = len(logic.phases)
            next_phase_idx = (current_phase + 1) % num_phases
            next_phase_state = logic.phases[next_phase_idx].state
            
            # Calculate cycle time
            cycle_time = sum(p.duration for p in logic.phases)
            
            # Estimate congestion
            queue_length = self._estimate_queue_length(tl_id)
            congestion = 'high' if queue_length > 15 else 'medium' if queue_length > 5 else 'low'
            
            return {
                'rsu_id': rsu.rsu_id,
                'timestamp': current_time,
                'current_phase': current_phase,
                'current_state': current_state,
                'time_to_change': time_to_change,
                'next_phase_state': next_phase_state,
                'cycle_time': cycle_time,
                'queue_length': queue_length,
                'congestion_level': congestion
            }
            
        except Exception as e:
            logger.debug(f"Error generating SPaT for {rsu.rsu_id}: {e}")
            return None
    
    def _calculate_speed_advisory(self, rsu: RSUState, current_speed: float,
                                   distance: float, current_time: float) -> Optional[Dict]:
        """
        Calculate speed advisory for approaching vehicle.
        
        Args:
            rsu: RSU state
            current_speed: Vehicle's current speed
            distance: Distance to RSU
            current_time: Current timestamp
            
        Returns:
            Speed advisory dictionary
        """
        if traci is None:
            return None
            
        try:
            tl_id = rsu.traffic_light_id
            
            # Get signal timing
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            next_switch = traci.trafficlight.getNextSwitch(tl_id)
            sim_time = traci.simulation.getTime()
            time_to_change = max(0, next_switch - sim_time)
            
            # Check if signal is green for approaching direction
            # Simplified: check if any green in state
            is_green = 'G' in current_state or 'g' in current_state
            
            recommended_speed = current_speed
            reason = 'maintain'
            
            if is_green:
                # Green phase - calculate if vehicle can pass
                if time_to_change > 0:
                    required_speed = distance / time_to_change
                    
                    if required_speed < current_speed and required_speed >= 5.0:
                        # Can slow down and still make it
                        recommended_speed = required_speed * 1.1  # 10% buffer
                        reason = 'green_wave'
                    elif required_speed > current_speed * 1.5:
                        # Too far to make it, slow down
                        recommended_speed = current_speed * 0.8
                        reason = 'wait_for_next_green'
            else:
                # Red phase - calculate when it will turn green
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                current_phase = traci.trafficlight.getPhase(tl_id)
                
                time_to_green = self._estimate_time_to_green(
                    current_phase, logic, time_to_change
                )
                
                if time_to_green > 0:
                    optimal_speed = distance / time_to_green
                    
                    if 10 <= optimal_speed <= self.green_wave_speed:
                        recommended_speed = optimal_speed
                        reason = 'green_wave'
                    else:
                        # Slow down to save fuel
                        recommended_speed = min(current_speed, 
                                               self.green_wave_speed * 0.7)
                        reason = 'fuel_saving'
            
            # Constrain to safe limits
            recommended_speed = max(5.0, min(recommended_speed, self.green_wave_speed))
            
            return {
                'rsu_id': rsu.rsu_id,
                'timestamp': current_time,
                'recommended_speed': recommended_speed,
                'reason': reason,
                'distance_to_signal': distance,
                'time_to_green': time_to_change if is_green else self._estimate_time_to_green(
                    traci.trafficlight.getPhase(tl_id),
                    traci.trafficlight.getAllProgramLogics(tl_id)[0],
                    time_to_change
                )
            }
            
        except Exception as e:
            logger.debug(f"Error calculating speed advisory: {e}")
            return None
    
    def _estimate_time_to_green(self, current_phase: int, logic, 
                                 time_remaining: float) -> float:
        """Estimate time until green phase."""
        total_time = time_remaining
        num_phases = len(logic.phases)
        
        for i in range(1, num_phases):
            phase_idx = (current_phase + i) % num_phases
            phase = logic.phases[phase_idx]
            
            if 'G' in phase.state or 'g' in phase.state:
                return total_time
            
            total_time += phase.duration
        
        return total_time
    
    def _estimate_queue_length(self, tl_id: str) -> int:
        """Estimate queue length at traffic light."""
        if traci is None:
            return 0
            
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            total_queue = 0
            
            for lane_id in controlled_lanes:
                # Count vehicles with low speed
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in vehicles:
                    if traci.vehicle.getSpeed(veh_id) < 1.0:
                        total_queue += 1
            
            return total_queue
            
        except Exception as e:
            return 0
    
    def _get_queue_info(self, rsu: RSUState) -> Optional[Dict]:
        """Get queue information for an RSU."""
        if traci is None:
            return None
            
        try:
            tl_id = rsu.traffic_light_id
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            queue_info = {}
            for lane_id in controlled_lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                queue_count = sum(1 for v in vehicles 
                                 if traci.vehicle.getSpeed(v) < 1.0)
                queue_info[lane_id] = queue_count
            
            return {
                'rsu_id': rsu.rsu_id,
                'queues': queue_info,
                'total_queue': sum(queue_info.values())
            }
            
        except Exception as e:
            return None
    
    def _aggregate_speed_advisory(self, advisories: List[Dict]) -> float:
        """
        Aggregate multiple speed advisories.
        
        Args:
            advisories: List of speed advisories
            
        Returns:
            Recommended speed
        """
        if not advisories:
            return self.green_wave_speed
        
        # Take minimum recommended speed for safety
        return min(a['recommended_speed'] for a in advisories)
    
    def _calculate_distance(self, pos1: Tuple[float, float],
                            pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def setup_green_wave_corridor(self, corridor_id: str,
                                   traffic_light_ids: List[str]) -> None:
        """
        Set up a green wave corridor.
        
        Args:
            corridor_id: Identifier for the corridor
            traffic_light_ids: Ordered list of traffic light IDs
        """
        self.green_wave_corridors[corridor_id] = traffic_light_ids
        logger.info(f"Set up green wave corridor: {corridor_id}")
    
    def notify_emergency_approach(self, tl_id: str, emergency_id: str,
                                   eta: float) -> None:
        """
        Notify RSU of approaching emergency vehicle.
        
        Args:
            tl_id: Traffic light ID
            emergency_id: Emergency vehicle ID
            eta: Estimated time of arrival
        """
        if tl_id not in self.rsus:
            return
            
        logger.info(f"Emergency vehicle {emergency_id} approaching {tl_id}, ETA: {eta:.1f}s")
        # This would trigger preemption in the traffic light controller
    
    def get_statistics(self) -> Dict:
        """Get communication statistics."""
        return {
            'total_spat_messages': self.total_spat_messages,
            'total_advisories': self.total_advisories,
            'active_rsus': len(self.rsus),
            'total_connected_vehicles': sum(
                len(rsu.connected_vehicles) for rsu in self.rsus.values()
            )
        }
    
    def get_traffic_info(self, position: Tuple[float, float]) -> Dict:
        """
        Get traffic information for a position.
        
        Args:
            position: Query position
            
        Returns:
            Traffic information dictionary
        """
        nearby_rsus = self._find_rsus_in_range(position)
        
        if not nearby_rsus:
            return {'status': 'no_coverage'}
        
        info = {
            'status': 'available',
            'signals': [],
            'queues': []
        }
        
        current_time = time.time()
        for rsu in nearby_rsus:
            spat = self._generate_spat(rsu, current_time)
            if spat:
                info['signals'].append(spat)
            
            queue = self._get_queue_info(rsu)
            if queue:
                info['queues'].append(queue)
        
        return info
