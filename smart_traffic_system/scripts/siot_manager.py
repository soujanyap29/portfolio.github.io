#!/usr/bin/env python3
"""
SIoT (Social Internet of Things) Manager Module

Implements Social Internet of Things behavior for vehicles including:
- Social relationship formation
- Trust score management
- Cooperative routing
- Information verification
- Alert sharing and propagation
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import math
import time
import random

logger = logging.getLogger('SIoTManager')


class RelationshipType(Enum):
    """Types of social relationships between vehicles."""
    CO_LOCATION = 'co_location'      # Same area
    CO_MOVEMENT = 'co_movement'      # Same route/direction
    SOCIAL_OBJECT = 'social_object'  # Same owner/type
    SERVICE = 'service'              # Provide service (e.g., RSU)
    PARENTAL = 'parental'            # Manufacturer relationship


@dataclass
class TrustScore:
    """Trust score for a vehicle."""
    vehicle_id: str
    accuracy_score: float = 0.5      # How accurate past info was
    timeliness_score: float = 0.5    # How timely info was shared
    consistency_score: float = 0.5   # How consistent behavior is
    interaction_count: int = 0        # Number of interactions
    last_update: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Calculate overall trust score."""
        # Weighted average
        weights = {'accuracy': 0.4, 'timeliness': 0.3, 'consistency': 0.3}
        return (self.accuracy_score * weights['accuracy'] +
                self.timeliness_score * weights['timeliness'] +
                self.consistency_score * weights['consistency'])
    
    def update(self, accuracy: float = None, timeliness: float = None,
               consistency: float = None) -> None:
        """Update trust scores with new observations."""
        alpha = 0.3  # Learning rate
        
        if accuracy is not None:
            self.accuracy_score = (1 - alpha) * self.accuracy_score + alpha * accuracy
        if timeliness is not None:
            self.timeliness_score = (1 - alpha) * self.timeliness_score + alpha * timeliness
        if consistency is not None:
            self.consistency_score = (1 - alpha) * self.consistency_score + alpha * consistency
        
        self.interaction_count += 1
        self.last_update = time.time()


@dataclass
class SocialRelationship:
    """Social relationship between two entities."""
    entity1_id: str
    entity2_id: str
    relationship_type: RelationshipType
    strength: float = 0.5
    created_at: float = 0.0
    last_interaction: float = 0.0
    interaction_count: int = 0
    
    def update_interaction(self) -> None:
        """Update relationship after interaction."""
        current_time = time.time()
        
        # Strengthen relationship with recent interaction
        time_since_last = current_time - self.last_interaction
        if time_since_last < 60:  # Within last minute
            self.strength = min(1.0, self.strength + 0.05)
        elif time_since_last < 300:  # Within 5 minutes
            self.strength = min(1.0, self.strength + 0.02)
        
        self.last_interaction = current_time
        self.interaction_count += 1
    
    def decay(self) -> None:
        """Apply time-based decay to relationship strength."""
        current_time = time.time()
        time_since_last = current_time - self.last_interaction
        
        # Decay by 10% per hour of no interaction
        decay_rate = 0.1 * (time_since_last / 3600)
        self.strength = max(0.0, self.strength - decay_rate)


@dataclass 
class SharedInformation:
    """Information shared between vehicles."""
    info_id: str
    sender_id: str
    info_type: str  # 'congestion', 'hazard', 'route', 'parking'
    content: Dict
    timestamp: float
    verified: bool = False
    verification_count: int = 0


class SIoTManager:
    """
    Manages Social Internet of Things behavior.
    
    Features:
    - Trust network management
    - Social relationship formation
    - Cooperative decision making
    - Information verification
    - Alert propagation
    """
    
    def __init__(self, trust_threshold: float = 0.6):
        """
        Initialize SIoT manager.
        
        Args:
            trust_threshold: Minimum trust for accepting information
        """
        self.trust_threshold = trust_threshold
        
        # Trust network: vehicle_id -> TrustScore
        self.trust_network: Dict[str, Dict[str, TrustScore]] = defaultdict(dict)
        
        # Social relationships
        self.relationships: Dict[str, SocialRelationship] = {}
        
        # Shared information pool
        self.information_pool: Dict[str, SharedInformation] = {}
        
        # Enable/disable SIoT
        self.enabled = True
        
        # Infrastructure trust (always high)
        self.infrastructure_trust = 0.95
        
        # Statistics
        self.total_relationships_formed = 0
        self.total_info_shared = 0
        self.total_info_verified = 0
        
    def update_relationships(self, vehicle_states: Dict,
                             v2v_messages: Dict) -> None:
        """
        Update social relationships based on interactions.
        
        Args:
            vehicle_states: Current vehicle states
            v2v_messages: V2V communication messages
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Update relationships based on proximity
        self._update_co_location_relationships(vehicle_states, current_time)
        
        # Update relationships based on movement
        self._update_co_movement_relationships(vehicle_states, current_time)
        
        # Update trust based on messages
        self._update_trust_from_messages(v2v_messages, vehicle_states, current_time)
        
        # Decay old relationships
        self._decay_relationships()
        
        # Clean up vehicles that left
        self._cleanup_left_vehicles(set(vehicle_states.keys()))
    
    def _update_co_location_relationships(self, vehicle_states: Dict,
                                           current_time: float) -> None:
        """Update co-location relationships."""
        # Group vehicles by area (simplified grid)
        grid_size = 100  # 100 meter grid
        grid = defaultdict(list)
        
        for veh_id, state in vehicle_states.items():
            pos = state.get('position', (0, 0))
            grid_key = (int(pos[0] // grid_size), int(pos[1] // grid_size))
            grid[grid_key].append(veh_id)
        
        # Form relationships within same grid cell
        for cell_vehicles in grid.values():
            if len(cell_vehicles) > 1:
                for i, veh1 in enumerate(cell_vehicles):
                    for veh2 in cell_vehicles[i+1:]:
                        rel_key = self._get_relationship_key(veh1, veh2)
                        
                        if rel_key not in self.relationships:
                            self.relationships[rel_key] = SocialRelationship(
                                entity1_id=veh1,
                                entity2_id=veh2,
                                relationship_type=RelationshipType.CO_LOCATION,
                                created_at=current_time,
                                last_interaction=current_time
                            )
                            self.total_relationships_formed += 1
                        else:
                            self.relationships[rel_key].update_interaction()
    
    def _update_co_movement_relationships(self, vehicle_states: Dict,
                                           current_time: float) -> None:
        """Update co-movement relationships."""
        # Group vehicles by route/direction
        route_groups = defaultdict(list)
        
        for veh_id, state in vehicle_states.items():
            route = state.get('route', [])
            if route:
                # Use first and last edge as route identifier
                route_key = (route[0], route[-1]) if len(route) > 1 else (route[0], route[0])
                route_groups[route_key].append(veh_id)
        
        # Form relationships for vehicles on same route
        for route_vehicles in route_groups.values():
            if len(route_vehicles) > 1:
                for i, veh1 in enumerate(route_vehicles):
                    for veh2 in route_vehicles[i+1:]:
                        rel_key = self._get_relationship_key(veh1, veh2)
                        
                        if rel_key not in self.relationships:
                            self.relationships[rel_key] = SocialRelationship(
                                entity1_id=veh1,
                                entity2_id=veh2,
                                relationship_type=RelationshipType.CO_MOVEMENT,
                                strength=0.6,  # Start stronger for same route
                                created_at=current_time,
                                last_interaction=current_time
                            )
                            self.total_relationships_formed += 1
                        else:
                            rel = self.relationships[rel_key]
                            if rel.relationship_type == RelationshipType.CO_LOCATION:
                                # Upgrade to co-movement
                                rel.relationship_type = RelationshipType.CO_MOVEMENT
                                rel.strength = min(1.0, rel.strength + 0.1)
                            rel.update_interaction()
    
    def _update_trust_from_messages(self, v2v_messages: Dict,
                                     vehicle_states: Dict,
                                     current_time: float) -> None:
        """Update trust scores based on received messages."""
        for receiver_id, messages in v2v_messages.items():
            beacons = messages.get('beacons', [])
            alerts = messages.get('alerts', [])
            
            for beacon in beacons:
                sender_id = beacon.get('sender')
                if sender_id:
                    self._process_beacon_trust(receiver_id, sender_id, beacon,
                                              vehicle_states, current_time)
            
            for alert in alerts:
                sender_id = alert.get('sender')
                if sender_id:
                    self._process_alert_trust(receiver_id, sender_id, alert,
                                             vehicle_states, current_time)
    
    def _process_beacon_trust(self, receiver_id: str, sender_id: str,
                               beacon: Dict, vehicle_states: Dict,
                               current_time: float) -> None:
        """Process beacon message for trust update."""
        if sender_id not in self.trust_network[receiver_id]:
            self.trust_network[receiver_id][sender_id] = TrustScore(sender_id)
        
        trust = self.trust_network[receiver_id][sender_id]
        
        # Verify beacon accuracy if possible
        if sender_id in vehicle_states:
            actual_state = vehicle_states[sender_id]
            reported_pos = beacon.get('position', (0, 0))
            actual_pos = actual_state.get('position', (0, 0))
            
            # Calculate position error
            error = math.sqrt((reported_pos[0] - actual_pos[0])**2 +
                            (reported_pos[1] - actual_pos[1])**2)
            
            # Convert to accuracy score (1.0 for < 5m error, decreases linearly)
            accuracy = max(0, 1 - error / 50)
            
            # Update trust
            trust.update(accuracy=accuracy, timeliness=0.9)
    
    def _process_alert_trust(self, receiver_id: str, sender_id: str,
                              alert: Dict, vehicle_states: Dict,
                              current_time: float) -> None:
        """Process alert message for trust update."""
        if sender_id not in self.trust_network[receiver_id]:
            self.trust_network[receiver_id][sender_id] = TrustScore(sender_id)
        
        trust = self.trust_network[receiver_id][sender_id]
        
        alert_type = alert.get('type', '')
        
        # Verify alert if possible
        if alert_type == 'emergency_braking':
            # Check if sender is actually braking
            if sender_id in vehicle_states:
                actual_accel = vehicle_states[sender_id].get('acceleration', 0)
                if actual_accel < -2.0:
                    trust.update(accuracy=0.95, consistency=0.9)
                else:
                    trust.update(accuracy=0.3, consistency=0.5)
        else:
            # For unverifiable alerts, give moderate trust update
            trust.update(timeliness=0.8)
    
    def _decay_relationships(self) -> None:
        """Apply decay to all relationships."""
        to_remove = []
        
        for rel_key, relationship in self.relationships.items():
            relationship.decay()
            
            # Remove very weak relationships
            if relationship.strength < 0.1:
                to_remove.append(rel_key)
        
        for key in to_remove:
            del self.relationships[key]
    
    def _cleanup_left_vehicles(self, active_ids: Set[str]) -> None:
        """Remove vehicles that have left the simulation."""
        # Clean trust network
        to_remove_trust = []
        for veh_id in self.trust_network:
            if veh_id not in active_ids:
                to_remove_trust.append(veh_id)
        
        for veh_id in to_remove_trust:
            del self.trust_network[veh_id]
        
        # Clean relationships
        to_remove_rel = []
        for rel_key, rel in self.relationships.items():
            if rel.entity1_id not in active_ids or rel.entity2_id not in active_ids:
                to_remove_rel.append(rel_key)
        
        for key in to_remove_rel:
            del self.relationships[key]
    
    def _get_relationship_key(self, id1: str, id2: str) -> str:
        """Generate consistent relationship key."""
        return f"{min(id1, id2)}_{max(id1, id2)}"
    
    def get_trust_score(self, observer_id: str, target_id: str) -> float:
        """
        Get trust score for a vehicle.
        
        Args:
            observer_id: Observing vehicle ID
            target_id: Target vehicle ID
            
        Returns:
            Trust score (0-1)
        """
        if observer_id in self.trust_network:
            if target_id in self.trust_network[observer_id]:
                return self.trust_network[observer_id][target_id].overall_score
        return 0.5  # Default neutral trust
    
    def get_trust_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Get all trust scores.
        
        Returns:
            Nested dictionary of trust scores
        """
        result = {}
        for observer_id, targets in self.trust_network.items():
            result[observer_id] = {
                target_id: trust.overall_score
                for target_id, trust in targets.items()
            }
        return result
    
    def share_information(self, sender_id: str, info_type: str,
                          content: Dict) -> str:
        """
        Share information to the SIoT network.
        
        Args:
            sender_id: Sending vehicle ID
            info_type: Type of information
            content: Information content
            
        Returns:
            Information ID
        """
        info_id = f"info_{len(self.information_pool)}_{int(time.time() * 1000)}"
        
        self.information_pool[info_id] = SharedInformation(
            info_id=info_id,
            sender_id=sender_id,
            info_type=info_type,
            content=content,
            timestamp=time.time()
        )
        
        self.total_info_shared += 1
        return info_id
    
    def get_verified_information(self, receiver_id: str,
                                  info_type: str = None) -> List[Dict]:
        """
        Get verified information for a vehicle.
        
        Args:
            receiver_id: Receiving vehicle ID
            info_type: Optional filter by type
            
        Returns:
            List of verified information
        """
        result = []
        
        for info_id, info in self.information_pool.items():
            # Filter by type if specified
            if info_type and info.info_type != info_type:
                continue
            
            # Check sender trust
            sender_trust = self.get_trust_score(receiver_id, info.sender_id)
            
            if sender_trust >= self.trust_threshold or info.verified:
                result.append({
                    'info_id': info_id,
                    'type': info.info_type,
                    'content': info.content,
                    'sender': info.sender_id,
                    'trust': sender_trust,
                    'timestamp': info.timestamp
                })
        
        return result
    
    def verify_information(self, info_id: str, verifier_id: str) -> bool:
        """
        Verify shared information.
        
        Args:
            info_id: Information ID
            verifier_id: Verifying vehicle ID
            
        Returns:
            True if verification successful
        """
        if info_id not in self.information_pool:
            return False
        
        info = self.information_pool[info_id]
        info.verification_count += 1
        
        # Mark as verified if multiple verifications
        if info.verification_count >= 3:
            info.verified = True
            self.total_info_verified += 1
        
        return True
    
    def get_cooperative_recommendation(self, vehicle_id: str,
                                        vehicle_states: Dict,
                                        recommendation_type: str) -> Optional[Dict]:
        """
        Get cooperative recommendation based on social network.
        
        Args:
            vehicle_id: Vehicle ID
            vehicle_states: All vehicle states
            recommendation_type: Type of recommendation ('route', 'speed', 'lane')
            
        Returns:
            Recommendation dictionary or None
        """
        if not self.enabled:
            return None
        
        # Get trusted neighbors
        trusted_neighbors = []
        
        for target_id in vehicle_states:
            if target_id == vehicle_id:
                continue
            
            trust = self.get_trust_score(vehicle_id, target_id)
            if trust >= self.trust_threshold:
                trusted_neighbors.append((target_id, trust))
        
        if not trusted_neighbors:
            return None
        
        if recommendation_type == 'route':
            return self._get_route_recommendation(vehicle_id, trusted_neighbors,
                                                  vehicle_states)
        elif recommendation_type == 'speed':
            return self._get_speed_recommendation(vehicle_id, trusted_neighbors,
                                                  vehicle_states)
        elif recommendation_type == 'lane':
            return self._get_lane_recommendation(vehicle_id, trusted_neighbors,
                                                 vehicle_states)
        
        return None
    
    def _get_route_recommendation(self, vehicle_id: str,
                                   trusted_neighbors: List[Tuple[str, float]],
                                   vehicle_states: Dict) -> Optional[Dict]:
        """Get route recommendation from trusted neighbors."""
        # Collect route information from neighbors
        route_info = []
        
        for neighbor_id, trust in trusted_neighbors:
            neighbor_state = vehicle_states.get(neighbor_id, {})
            route = neighbor_state.get('route', [])
            speed = neighbor_state.get('speed', 0)
            
            if route:
                route_info.append({
                    'neighbor_id': neighbor_id,
                    'trust': trust,
                    'route': route,
                    'speed': speed
                })
        
        if route_info:
            # Aggregate: prefer routes of high-trust, fast-moving neighbors
            best_info = max(route_info, key=lambda x: x['trust'] * x['speed'])
            return {
                'type': 'route',
                'suggestion': 'follow_similar_route',
                'reference_vehicle': best_info['neighbor_id'],
                'confidence': best_info['trust']
            }
        
        return None
    
    def _get_speed_recommendation(self, vehicle_id: str,
                                   trusted_neighbors: List[Tuple[str, float]],
                                   vehicle_states: Dict) -> Optional[Dict]:
        """Get speed recommendation from trusted neighbors."""
        weighted_speed = 0
        total_weight = 0
        
        for neighbor_id, trust in trusted_neighbors:
            neighbor_state = vehicle_states.get(neighbor_id, {})
            speed = neighbor_state.get('speed', 0)
            
            if speed > 0:
                weighted_speed += speed * trust
                total_weight += trust
        
        if total_weight > 0:
            recommended_speed = weighted_speed / total_weight
            return {
                'type': 'speed',
                'recommended_speed': recommended_speed,
                'confidence': total_weight / len(trusted_neighbors)
            }
        
        return None
    
    def _get_lane_recommendation(self, vehicle_id: str,
                                  trusted_neighbors: List[Tuple[str, float]],
                                  vehicle_states: Dict) -> Optional[Dict]:
        """Get lane recommendation from trusted neighbors."""
        lane_speeds = defaultdict(list)
        
        for neighbor_id, trust in trusted_neighbors:
            neighbor_state = vehicle_states.get(neighbor_id, {})
            lane = neighbor_state.get('lane_index', 0)
            speed = neighbor_state.get('speed', 0)
            
            lane_speeds[lane].append((speed, trust))
        
        if lane_speeds:
            # Find lane with best speed
            best_lane = -1
            best_weighted_speed = 0
            
            for lane, speed_trust_pairs in lane_speeds.items():
                weighted_speed = sum(s * t for s, t in speed_trust_pairs)
                total_trust = sum(t for _, t in speed_trust_pairs)
                avg_weighted_speed = weighted_speed / max(1, total_trust)
                
                if avg_weighted_speed > best_weighted_speed:
                    best_weighted_speed = avg_weighted_speed
                    best_lane = lane
            
            if best_lane >= 0:
                return {
                    'type': 'lane',
                    'recommended_lane': best_lane,
                    'expected_speed': best_weighted_speed,
                    'confidence': len(lane_speeds[best_lane]) / len(trusted_neighbors)
                }
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get SIoT statistics."""
        relationship_counts = defaultdict(int)
        for rel in self.relationships.values():
            relationship_counts[rel.relationship_type.value] += 1
        
        trust_scores = []
        for targets in self.trust_network.values():
            for trust in targets.values():
                trust_scores.append(trust.overall_score)
        
        avg_trust = sum(trust_scores) / max(1, len(trust_scores))
        
        return {
            'total_relationships': len(self.relationships),
            'relationships_by_type': dict(relationship_counts),
            'total_trust_pairs': len(trust_scores),
            'average_trust': avg_trust,
            'total_info_shared': self.total_info_shared,
            'total_info_verified': self.total_info_verified,
            'active_information': len(self.information_pool)
        }
