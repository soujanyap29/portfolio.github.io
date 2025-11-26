#!/usr/bin/env python3
"""
Traffic Light Controller Module

Handles traffic light control including:
- Fixed-time signal control
- Actuated (sensor-based) control
- Adaptive signal control
- Emergency vehicle preemption
- Green wave optimization
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Ensure SUMO tools are in path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

try:
    import traci
except ImportError:
    traci = None

logger = logging.getLogger('TrafficLightController')


class SignalPhase(Enum):
    """Signal phase states."""
    RED = 'r'
    YELLOW = 'y'
    GREEN = 'G'
    GREEN_PRIORITY = 'g'
    OFF = 'O'


@dataclass
class TrafficLightState:
    """Data class for traffic light state."""
    tl_id: str
    current_phase: int
    current_state: str
    phase_duration: float
    time_in_phase: float
    program_id: str
    controlled_lanes: List[str] = field(default_factory=list)
    vehicle_count: Dict[str, int] = field(default_factory=dict)
    queue_length: Dict[str, float] = field(default_factory=dict)


@dataclass
class SignalProgram:
    """Signal timing program."""
    program_id: str
    phases: List[Dict]
    cycle_time: float
    offset: float = 0.0


class TrafficLightController:
    """
    Controls traffic lights in the simulation.
    
    Features:
    - Multiple control strategies (fixed, actuated, adaptive)
    - Emergency vehicle preemption
    - Green wave coordination
    - Queue-based optimization
    """
    
    def __init__(self, adaptive: bool = True):
        """
        Initialize the traffic light controller.
        
        Args:
            adaptive: Whether to use adaptive control
        """
        self.traffic_lights: Dict[str, TrafficLightState] = {}
        self.adaptive = adaptive
        
        # Preemption tracking
        self.preempted_lights: Set[str] = set()
        self.preemption_requests: Dict[str, List[str]] = defaultdict(list)
        
        # Control parameters
        self.min_green = 10.0  # Minimum green time (seconds)
        self.max_green = 60.0  # Maximum green time (seconds)
        self.yellow_time = 3.0  # Yellow time (seconds)
        self.all_red_time = 2.0  # All-red clearance time
        
        # Queue thresholds for adaptive control
        self.queue_threshold_extend = 5  # Vehicles to extend green
        self.queue_threshold_switch = 10  # Vehicles to force switch
        
        # Green wave parameters
        self.green_wave_speed = 50 / 3.6  # 50 km/h in m/s
        self.green_wave_corridors: Dict[str, List[str]] = {}
        
    def register_traffic_light(self, tl_id: str) -> None:
        """
        Register a traffic light for control.
        
        Args:
            tl_id: Traffic light ID
        """
        if traci is None:
            return
            
        try:
            current_phase = traci.trafficlight.getPhase(tl_id)
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            program_id = traci.trafficlight.getProgram(tl_id)
            controlled_lanes = list(traci.trafficlight.getControlledLanes(tl_id))
            
            self.traffic_lights[tl_id] = TrafficLightState(
                tl_id=tl_id,
                current_phase=current_phase,
                current_state=current_state,
                phase_duration=0.0,
                time_in_phase=0.0,
                program_id=program_id,
                controlled_lanes=controlled_lanes
            )
            
            logger.debug(f"Registered traffic light: {tl_id}")
            
        except Exception as e:
            logger.warning(f"Could not register traffic light {tl_id}: {e}")
    
    def update_signals(self, vehicle_states: Dict, v2i_messages: Dict) -> None:
        """
        Update all traffic signals based on current conditions.
        
        Args:
            vehicle_states: Current vehicle states
            v2i_messages: V2I communication messages
        """
        if traci is None:
            return
            
        for tl_id in self.traffic_lights:
            try:
                # Update state
                self._update_light_state(tl_id)
                
                # Calculate queue lengths
                self._calculate_queues(tl_id, vehicle_states)
                
                # Check for preemption requests
                if tl_id in self.preemption_requests and self.preemption_requests[tl_id]:
                    self._handle_preemption(tl_id)
                elif self.adaptive:
                    self._adaptive_control(tl_id)
                    
            except Exception as e:
                logger.debug(f"Error updating signal {tl_id}: {e}")
    
    def _update_light_state(self, tl_id: str) -> None:
        """Update the state of a traffic light."""
        if traci is None:
            return
            
        try:
            state = self.traffic_lights[tl_id]
            
            new_phase = traci.trafficlight.getPhase(tl_id)
            new_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            
            if new_phase != state.current_phase:
                state.time_in_phase = 0.0
            else:
                state.time_in_phase += 0.1  # Step length
                
            state.current_phase = new_phase
            state.current_state = new_state
            
        except Exception as e:
            logger.debug(f"Error updating state for {tl_id}: {e}")
    
    def _calculate_queues(self, tl_id: str, vehicle_states: Dict) -> None:
        """
        Calculate queue lengths at a traffic light.
        
        Args:
            tl_id: Traffic light ID
            vehicle_states: Vehicle state dictionary
        """
        if traci is None or tl_id not in self.traffic_lights:
            return
            
        state = self.traffic_lights[tl_id]
        state.vehicle_count = {}
        state.queue_length = {}
        
        for lane_id in state.controlled_lanes:
            count = 0
            total_length = 0.0
            
            for veh_id, veh_state in vehicle_states.items():
                if veh_state.get('lane_id') == lane_id:
                    if veh_state.get('speed', 0) < 1.0:  # Essentially stopped
                        count += 1
                        total_length += 5.0  # Approximate vehicle length
            
            state.vehicle_count[lane_id] = count
            state.queue_length[lane_id] = total_length
    
    def _adaptive_control(self, tl_id: str) -> None:
        """
        Apply adaptive signal control logic.
        
        Args:
            tl_id: Traffic light ID
        """
        if traci is None or tl_id not in self.traffic_lights:
            return
            
        state = self.traffic_lights[tl_id]
        
        try:
            # Get current signal state
            current_state = state.current_state
            time_in_phase = state.time_in_phase
            
            # Calculate demand for each approach
            green_lanes = []
            red_lanes = []
            
            for i, lane_id in enumerate(state.controlled_lanes):
                if i < len(current_state):
                    if current_state[i] in ['G', 'g']:
                        green_lanes.append(lane_id)
                    elif current_state[i] in ['r']:
                        red_lanes.append(lane_id)
            
            # Calculate queue lengths
            green_queue = sum(state.vehicle_count.get(lane, 0) for lane in green_lanes)
            red_queue = sum(state.vehicle_count.get(lane, 0) for lane in red_lanes)
            
            # Adaptive logic
            if time_in_phase >= self.min_green:
                # Consider switching if red queue is high
                if red_queue >= self.queue_threshold_switch and green_queue < self.queue_threshold_extend:
                    self._switch_to_next_phase(tl_id)
                elif time_in_phase >= self.max_green:
                    self._switch_to_next_phase(tl_id)
                elif green_queue >= self.queue_threshold_extend and time_in_phase < self.max_green:
                    # Extend green
                    pass  # Let current phase continue
                    
        except Exception as e:
            logger.debug(f"Error in adaptive control for {tl_id}: {e}")
    
    def _switch_to_next_phase(self, tl_id: str) -> None:
        """Switch traffic light to next phase."""
        if traci is None:
            return
            
        try:
            current_phase = traci.trafficlight.getPhase(tl_id)
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            num_phases = len(logic.phases)
            
            next_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(tl_id, next_phase)
            
        except Exception as e:
            logger.debug(f"Error switching phase for {tl_id}: {e}")
    
    def request_preemption(self, tl_id: str, emergency_id: str) -> bool:
        """
        Request signal preemption for emergency vehicle.
        
        Args:
            tl_id: Traffic light ID
            emergency_id: Emergency vehicle ID
            
        Returns:
            True if preemption request accepted
        """
        if tl_id not in self.traffic_lights:
            return False
            
        if emergency_id not in self.preemption_requests[tl_id]:
            self.preemption_requests[tl_id].append(emergency_id)
            logger.info(f"Preemption requested at {tl_id} for {emergency_id}")
            return True
        return False
    
    def clear_preemption(self, tl_id: str, emergency_id: str) -> None:
        """
        Clear preemption request after emergency vehicle passes.
        
        Args:
            tl_id: Traffic light ID
            emergency_id: Emergency vehicle ID
        """
        if tl_id in self.preemption_requests:
            if emergency_id in self.preemption_requests[tl_id]:
                self.preemption_requests[tl_id].remove(emergency_id)
                logger.info(f"Preemption cleared at {tl_id} for {emergency_id}")
                
        if not self.preemption_requests[tl_id]:
            self.preempted_lights.discard(tl_id)
            self._restore_normal_operation(tl_id)
    
    def _handle_preemption(self, tl_id: str) -> None:
        """
        Handle emergency vehicle preemption.
        
        Args:
            tl_id: Traffic light ID
        """
        if traci is None:
            return
            
        try:
            # Get approach direction of emergency vehicle
            emergency_id = self.preemption_requests[tl_id][0]
            
            # Get emergency vehicle's lane
            try:
                emergency_lane = traci.vehicle.getLaneID(emergency_id)
            except traci.exceptions.TraCIException:
                # Vehicle no longer exists
                self.clear_preemption(tl_id, emergency_id)
                return
            
            state = self.traffic_lights[tl_id]
            
            # Find which phase gives green to emergency vehicle's approach
            target_phase = self._find_green_phase_for_lane(tl_id, emergency_lane)
            
            if target_phase is not None:
                current_phase = traci.trafficlight.getPhase(tl_id)
                if current_phase != target_phase:
                    # Insert yellow phase if needed
                    current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                    if 'G' in current_state or 'g' in current_state:
                        # Need yellow transition
                        yellow_state = current_state.replace('G', 'y').replace('g', 'y')
                        traci.trafficlight.setRedYellowGreenState(tl_id, yellow_state)
                    else:
                        # Direct switch to green for emergency
                        traci.trafficlight.setPhase(tl_id, target_phase)
                        
            self.preempted_lights.add(tl_id)
            
        except Exception as e:
            logger.debug(f"Error handling preemption at {tl_id}: {e}")
    
    def _find_green_phase_for_lane(self, tl_id: str, lane_id: str) -> Optional[int]:
        """
        Find which phase gives green to a specific lane.
        
        Args:
            tl_id: Traffic light ID
            lane_id: Lane ID
            
        Returns:
            Phase index or None
        """
        if traci is None:
            return None
            
        try:
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            controlled_lanes = list(traci.trafficlight.getControlledLanes(tl_id))
            
            # Find lane index
            lane_index = -1
            for i, cl in enumerate(controlled_lanes):
                if lane_id.startswith(cl.rsplit('_', 1)[0]):  # Match edge
                    lane_index = i
                    break
            
            if lane_index < 0:
                return None
            
            # Find phase with green for this lane
            for phase_idx, phase in enumerate(logic.phases):
                if lane_index < len(phase.state):
                    if phase.state[lane_index] in ['G', 'g']:
                        return phase_idx
                        
            return None
            
        except Exception as e:
            logger.debug(f"Error finding green phase: {e}")
            return None
    
    def _restore_normal_operation(self, tl_id: str) -> None:
        """Restore normal signal operation after preemption."""
        if traci is None:
            return
            
        try:
            # Reset to default program
            default_program = self.traffic_lights[tl_id].program_id
            traci.trafficlight.setProgram(tl_id, default_program)
            logger.debug(f"Restored normal operation at {tl_id}")
        except Exception as e:
            logger.debug(f"Error restoring normal operation at {tl_id}: {e}")
    
    def get_signal_timing(self, tl_id: str) -> Dict:
        """
        Get current signal timing information.
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Dictionary with timing information
        """
        if traci is None or tl_id not in self.traffic_lights:
            return {}
            
        try:
            state = self.traffic_lights[tl_id]
            
            # Get remaining time in current phase
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            current_phase = traci.trafficlight.getPhase(tl_id)
            phase_duration = logic.phases[current_phase].duration
            remaining_time = max(0, phase_duration - state.time_in_phase)
            
            # Calculate time to green for each approach
            time_to_green = {}
            current_state = state.current_state
            
            for i, lane_id in enumerate(state.controlled_lanes):
                if i < len(current_state):
                    if current_state[i] in ['G', 'g']:
                        time_to_green[lane_id] = 0.0
                    else:
                        # Estimate time to green
                        time_to_green[lane_id] = self._estimate_time_to_green(
                            tl_id, i, current_phase, logic
                        )
            
            return {
                'tl_id': tl_id,
                'current_phase': current_phase,
                'current_state': current_state,
                'remaining_time': remaining_time,
                'time_to_green': time_to_green,
                'is_preempted': tl_id in self.preempted_lights
            }
            
        except Exception as e:
            logger.debug(f"Error getting signal timing for {tl_id}: {e}")
            return {}
    
    def _estimate_time_to_green(self, tl_id: str, lane_index: int,
                                 current_phase: int, logic) -> float:
        """Estimate time until a lane gets green."""
        total_time = 0.0
        num_phases = len(logic.phases)
        
        for i in range(num_phases):
            phase_idx = (current_phase + i) % num_phases
            phase = logic.phases[phase_idx]
            
            if lane_index < len(phase.state):
                if phase.state[lane_index] in ['G', 'g']:
                    return total_time
                    
            total_time += phase.duration
            
        return total_time
    
    def setup_green_wave(self, corridor_id: str, tl_ids: List[str],
                         speed: float = None) -> None:
        """
        Setup green wave coordination for a corridor.
        
        Args:
            corridor_id: Identifier for the corridor
            tl_ids: List of traffic light IDs in order
            speed: Target speed for green wave (m/s)
        """
        if speed is None:
            speed = self.green_wave_speed
            
        self.green_wave_corridors[corridor_id] = tl_ids
        
        if traci is None:
            return
            
        try:
            # Calculate offsets based on distance and speed
            total_offset = 0.0
            
            for i, tl_id in enumerate(tl_ids):
                if i > 0:
                    # Get position of traffic lights
                    pos1 = traci.junction.getPosition(tl_ids[i-1])
                    pos2 = traci.junction.getPosition(tl_id)
                    
                    # Calculate distance
                    distance = ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
                    
                    # Calculate offset
                    offset = distance / speed
                    total_offset += offset
                    
                # Apply offset to signal program
                # Note: This modifies the signal timing
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                logic.offset = total_offset % logic.phases[0].duration
                
                traci.trafficlight.setProgramLogic(tl_id, logic)
                
            logger.info(f"Green wave setup for corridor {corridor_id}")
            
        except Exception as e:
            logger.warning(f"Error setting up green wave: {e}")
    
    def get_queue_info(self, tl_id: str) -> Dict:
        """
        Get queue information at a traffic light.
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Dictionary with queue information
        """
        if tl_id not in self.traffic_lights:
            return {}
            
        state = self.traffic_lights[tl_id]
        return {
            'vehicle_count': state.vehicle_count.copy(),
            'queue_length': state.queue_length.copy(),
            'total_vehicles': sum(state.vehicle_count.values()),
            'total_queue_length': sum(state.queue_length.values())
        }
