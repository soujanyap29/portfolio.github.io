#!/usr/bin/env python3
"""
V2V (Vehicle-to-Vehicle) Communication Module

Handles all vehicle-to-vehicle communication including:
- Beacon messages (position, speed, heading)
- Alert messages (braking, hazards, lane changes)
- Cooperative awareness
- Collision avoidance
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math
import time

logger = logging.getLogger('V2VCommunication')


@dataclass
class V2VMessage:
    """Data class for V2V message."""
    message_id: str
    sender_id: str
    message_type: str  # 'beacon', 'alert', 'request', 'response'
    timestamp: float
    position: Tuple[float, float]
    speed: float
    heading: float
    data: Dict = field(default_factory=dict)
    ttl: int = 3  # Time to live (hops)


@dataclass
class VehicleCommunicationState:
    """Communication state for a vehicle."""
    vehicle_id: str
    last_beacon_time: float = 0.0
    received_messages: List[V2VMessage] = field(default_factory=list)
    neighbors: Set[str] = field(default_factory=set)
    message_count_sent: int = 0
    message_count_received: int = 0


class V2VCommunication:
    """
    Manages vehicle-to-vehicle communication.
    
    Features:
    - Periodic beacon broadcasting
    - Alert message propagation
    - Neighbor discovery
    - Collision warning
    - Lane change coordination
    """
    
    def __init__(self, communication_range: float = 100.0, 
                 beacon_frequency: float = 10.0):
        """
        Initialize V2V communication manager.
        
        Args:
            communication_range: Communication range in meters
            beacon_frequency: Beacon frequency in Hz
        """
        self.communication_range = communication_range
        self.beacon_interval = 1.0 / beacon_frequency
        
        # Communication state per vehicle
        self.vehicle_states: Dict[str, VehicleCommunicationState] = {}
        
        # Message buffers
        self.message_buffer: Dict[str, List[V2VMessage]] = defaultdict(list)
        self.alert_buffer: List[V2VMessage] = []
        
        # Enable/disable communication
        self.enabled = True
        
        # Statistics
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_alerts = 0
        
        # Message ID counter
        self._message_counter = 0
        
    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        self._message_counter += 1
        return f"msg_{self._message_counter}_{int(time.time() * 1000)}"
    
    def process_communication(self, vehicle_states: Dict) -> Dict:
        """
        Process V2V communication for all vehicles.
        
        Args:
            vehicle_states: Dictionary of vehicle states
            
        Returns:
            Dictionary mapping vehicle IDs to received messages
        """
        if not self.enabled:
            return {}
            
        current_time = time.time()
        received_messages = {}
        
        # Update communication states
        for veh_id, state in vehicle_states.items():
            if veh_id not in self.vehicle_states:
                self.vehicle_states[veh_id] = VehicleCommunicationState(veh_id)
            
            # Generate beacon if interval passed
            veh_comm_state = self.vehicle_states[veh_id]
            if current_time - veh_comm_state.last_beacon_time >= self.beacon_interval:
                beacon = self._create_beacon(veh_id, state, current_time)
                self._broadcast_message(beacon, vehicle_states)
                veh_comm_state.last_beacon_time = current_time
                veh_comm_state.message_count_sent += 1
            
            # Check for alert conditions
            self._check_alert_conditions(veh_id, state, vehicle_states, current_time)
        
        # Process message buffers
        for veh_id in vehicle_states:
            if veh_id in self.message_buffer:
                received_messages[veh_id] = self._process_received_messages(veh_id)
        
        # Clean up vehicles that left
        active_ids = set(vehicle_states.keys())
        to_remove = [vid for vid in self.vehicle_states if vid not in active_ids]
        for vid in to_remove:
            del self.vehicle_states[vid]
            if vid in self.message_buffer:
                del self.message_buffer[vid]
        
        return received_messages
    
    def _create_beacon(self, vehicle_id: str, state: Dict, 
                       timestamp: float) -> V2VMessage:
        """
        Create a beacon message for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            state: Vehicle state dictionary
            timestamp: Current timestamp
            
        Returns:
            V2VMessage beacon
        """
        position = state.get('position', (0, 0))
        speed = state.get('speed', 0)
        angle = state.get('angle', 0)
        acceleration = state.get('acceleration', 0)
        lane_id = state.get('lane_id', '')
        
        return V2VMessage(
            message_id=self._generate_message_id(),
            sender_id=vehicle_id,
            message_type='beacon',
            timestamp=timestamp,
            position=position,
            speed=speed,
            heading=angle,
            data={
                'acceleration': acceleration,
                'lane_id': lane_id,
                'vehicle_type': state.get('type', 'car'),
                'braking': acceleration < -2.0,
                'lane_change_intent': state.get('signals', 0) & 0x3 != 0
            }
        )
    
    def _broadcast_message(self, message: V2VMessage, 
                           vehicle_states: Dict) -> None:
        """
        Broadcast a message to all vehicles in range.
        
        Args:
            message: Message to broadcast
            vehicle_states: Dictionary of vehicle states
        """
        sender_pos = message.position
        
        for veh_id, state in vehicle_states.items():
            if veh_id == message.sender_id:
                continue
                
            receiver_pos = state.get('position', (0, 0))
            distance = self._calculate_distance(sender_pos, receiver_pos)
            
            if distance <= self.communication_range:
                self.message_buffer[veh_id].append(message)
                self.total_messages_sent += 1
                
                # Update neighbor list
                if message.sender_id in self.vehicle_states:
                    self.vehicle_states[message.sender_id].neighbors.add(veh_id)
    
    def _check_alert_conditions(self, vehicle_id: str, state: Dict,
                                 all_states: Dict, current_time: float) -> None:
        """
        Check if vehicle should send alert messages.
        
        Args:
            vehicle_id: Vehicle ID
            state: Vehicle state
            all_states: All vehicle states
            current_time: Current timestamp
        """
        # Check for hard braking
        acceleration = state.get('acceleration', 0)
        if acceleration < -3.0:
            self._send_alert(vehicle_id, state, 'emergency_braking', 
                           all_states, current_time,
                           {'deceleration': acceleration})
        
        # Check for emergency vehicle approaching
        veh_type = state.get('type', '').lower()
        if any(em in veh_type for em in ['ambulance', 'police', 'fire']):
            self._send_alert(vehicle_id, state, 'emergency_vehicle',
                           all_states, current_time,
                           {'vehicle_type': veh_type})
        
        # Check for potential collision
        self._check_collision_risk(vehicle_id, state, all_states, current_time)
    
    def _send_alert(self, vehicle_id: str, state: Dict, alert_type: str,
                    all_states: Dict, timestamp: float, 
                    extra_data: Dict = None) -> None:
        """Send an alert message to nearby vehicles."""
        position = state.get('position', (0, 0))
        speed = state.get('speed', 0)
        angle = state.get('angle', 0)
        
        data = {'alert_type': alert_type}
        if extra_data:
            data.update(extra_data)
        
        alert = V2VMessage(
            message_id=self._generate_message_id(),
            sender_id=vehicle_id,
            message_type='alert',
            timestamp=timestamp,
            position=position,
            speed=speed,
            heading=angle,
            data=data
        )
        
        self._broadcast_message(alert, all_states)
        self.total_alerts += 1
    
    def _check_collision_risk(self, vehicle_id: str, state: Dict,
                               all_states: Dict, current_time: float) -> None:
        """
        Check for collision risk with other vehicles.
        
        Args:
            vehicle_id: Vehicle ID
            state: Vehicle state
            all_states: All vehicle states
            current_time: Current timestamp
        """
        position = state.get('position', (0, 0))
        speed = state.get('speed', 0)
        heading = math.radians(state.get('angle', 0))
        
        # Calculate future position (2 seconds ahead)
        look_ahead = 2.0
        future_x = position[0] + speed * look_ahead * math.sin(heading)
        future_y = position[1] + speed * look_ahead * math.cos(heading)
        
        for other_id, other_state in all_states.items():
            if other_id == vehicle_id:
                continue
                
            other_pos = other_state.get('position', (0, 0))
            other_speed = other_state.get('speed', 0)
            other_heading = math.radians(other_state.get('angle', 0))
            
            # Calculate other's future position
            other_future_x = other_pos[0] + other_speed * look_ahead * math.sin(other_heading)
            other_future_y = other_pos[1] + other_speed * look_ahead * math.cos(other_heading)
            
            # Check if paths intersect
            future_distance = math.sqrt(
                (future_x - other_future_x)**2 + (future_y - other_future_y)**2
            )
            
            if future_distance < 5.0:  # Less than 5 meters
                self._send_alert(vehicle_id, state, 'collision_warning',
                               all_states, current_time,
                               {'target_vehicle': other_id,
                                'estimated_distance': future_distance})
    
    def _process_received_messages(self, vehicle_id: str) -> Dict:
        """
        Process received messages for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Processed message information
        """
        messages = self.message_buffer.get(vehicle_id, [])
        if not messages:
            return {'beacons': [], 'alerts': []}
        
        beacons = []
        alerts = []
        
        for msg in messages:
            self.total_messages_received += 1
            
            if vehicle_id in self.vehicle_states:
                self.vehicle_states[vehicle_id].message_count_received += 1
                self.vehicle_states[vehicle_id].neighbors.add(msg.sender_id)
            
            if msg.message_type == 'beacon':
                beacons.append({
                    'sender': msg.sender_id,
                    'position': msg.position,
                    'speed': msg.speed,
                    'heading': msg.heading,
                    'braking': msg.data.get('braking', False),
                    'lane_change': msg.data.get('lane_change_intent', False)
                })
            elif msg.message_type == 'alert':
                alerts.append({
                    'sender': msg.sender_id,
                    'type': msg.data.get('alert_type', 'unknown'),
                    'position': msg.position,
                    'data': msg.data
                })
        
        # Clear buffer
        self.message_buffer[vehicle_id] = []
        
        return {'beacons': beacons, 'alerts': alerts}
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                            pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, vehicle_id: str) -> Set[str]:
        """
        Get current neighbors of a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            Set of neighbor vehicle IDs
        """
        if vehicle_id in self.vehicle_states:
            return self.vehicle_states[vehicle_id].neighbors.copy()
        return set()
    
    def get_statistics(self) -> Dict:
        """
        Get communication statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_messages_sent': self.total_messages_sent,
            'total_messages_received': self.total_messages_received,
            'total_alerts': self.total_alerts,
            'active_vehicles': len(self.vehicle_states),
            'delivery_rate': (self.total_messages_received / 
                            max(1, self.total_messages_sent)) * 100
        }
    
    def send_lane_change_intent(self, vehicle_id: str, target_lane: int,
                                 vehicle_states: Dict) -> None:
        """
        Broadcast lane change intention.
        
        Args:
            vehicle_id: Vehicle ID
            target_lane: Target lane index
            vehicle_states: All vehicle states
        """
        if vehicle_id not in vehicle_states:
            return
            
        state = vehicle_states[vehicle_id]
        position = state.get('position', (0, 0))
        
        intent_msg = V2VMessage(
            message_id=self._generate_message_id(),
            sender_id=vehicle_id,
            message_type='alert',
            timestamp=time.time(),
            position=position,
            speed=state.get('speed', 0),
            heading=state.get('angle', 0),
            data={
                'alert_type': 'lane_change_intent',
                'target_lane': target_lane,
                'current_lane': state.get('lane_index', 0)
            }
        )
        
        self._broadcast_message(intent_msg, vehicle_states)
    
    def request_gap(self, requester_id: str, target_id: str,
                    vehicle_states: Dict) -> None:
        """
        Request gap from a specific vehicle for merging.
        
        Args:
            requester_id: Requesting vehicle ID
            target_id: Target vehicle ID
            vehicle_states: All vehicle states
        """
        if requester_id not in vehicle_states:
            return
            
        state = vehicle_states[requester_id]
        
        gap_request = V2VMessage(
            message_id=self._generate_message_id(),
            sender_id=requester_id,
            message_type='request',
            timestamp=time.time(),
            position=state.get('position', (0, 0)),
            speed=state.get('speed', 0),
            heading=state.get('angle', 0),
            data={
                'request_type': 'gap_request',
                'target_vehicle': target_id
            }
        )
        
        # Send only to target vehicle
        if target_id in vehicle_states:
            self.message_buffer[target_id].append(gap_request)
            self.total_messages_sent += 1
