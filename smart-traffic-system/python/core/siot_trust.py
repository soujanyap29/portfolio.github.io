"""
Social Internet of Things (SIoT) Trust Management
Part of Smart Traffic Management System

Implements social relationships and trust scoring among vehicles and infrastructure.
Enables cooperative decision-making based on trust levels.

Course Mappings:
- Computer Networks: Trust protocols, distributed systems
- OOPS: Object relationships, design patterns
- DBMS: Graph database concepts, relationship modeling
"""

import traci
import sqlite3
import time
import random
from collections import defaultdict
import json

class SIoTTrustManager:
    """
    Manages social relationships and trust scores in the traffic network.
    
    Types of Relationships:
    - Parental Object Relationship (POR): Same manufacturer/type
    - Co-location Object Relationship (CLOR): Same area frequently
    - Co-work Object Relationship (CWOR): Same route regularly
    - Social Object Relationship (SOR): Direct interaction history
    """
    
    def __init__(self):
        """Initialize SIoT trust management system"""
        # Trust scores: (entity1, entity2) -> trust_score (0.0 to 1.0)
        self.trust_scores = defaultdict(lambda: 0.5)  # Default neutral trust
        
        # Relationship types
        self.relationships = defaultdict(set)  # entity -> set of related entities
        self.relationship_types = {}  # (entity1, entity2) -> relationship_type
        
        # Interaction history
        self.interaction_count = defaultdict(int)
        self.positive_interactions = defaultdict(int)
        self.negative_interactions = defaultdict(int)
        
        # Message authenticity tracking
        self.message_reliability = defaultdict(lambda: 0.5)
        
        self.db_connection = None
        self.init_database()
        
    def init_database(self):
        """Initialize database for trust and relationship data"""
        self.db_connection = sqlite3.connect('../../database/siot_trust.db')
        cursor = self.db_connection.cursor()
        
        # Trust scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                entity1 TEXT,
                entity2 TEXT,
                trust_score REAL,
                relationship_type TEXT,
                interaction_count INTEGER
            )
        ''')
        
        # Trust events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                entity1 TEXT,
                entity2 TEXT,
                event_type TEXT,
                trust_change REAL,
                reason TEXT
            )
        ''')
        
        self.db_connection.commit()
        
    def establish_relationship(self, entity1, entity2, relationship_type):
        """
        Establish a social relationship between two entities.
        
        Args:
            entity1: First entity ID (vehicle or RSU)
            entity2: Second entity ID
            relationship_type: Type of relationship (POR, CLOR, CWOR, SOR)
        """
        # Add bidirectional relationship
        self.relationships[entity1].add(entity2)
        self.relationships[entity2].add(entity1)
        
        # Store relationship type
        key1 = (entity1, entity2)
        key2 = (entity2, entity1)
        self.relationship_types[key1] = relationship_type
        self.relationship_types[key2] = relationship_type
        
        # Initial trust boost based on relationship type
        trust_boost = {
            'POR': 0.2,   # Same type/manufacturer
            'CLOR': 0.15, # Co-location
            'CWOR': 0.15, # Co-work
            'SOR': 0.1    # Social
        }
        
        current_trust = self.trust_scores[key1]
        self.trust_scores[key1] += trust_boost.get(relationship_type, 0.1)
        self.trust_scores[key2] = self.trust_scores[key1]
        
        print(f"[SIOT] Established {relationship_type} relationship between {entity1} and {entity2}")
        
    def discover_relationships(self):
        """
        Automatically discover relationships based on vehicle properties and behavior.
        """
        try:
            all_vehicles = traci.vehicle.getIDList()
            
            # Parental Object Relationship (same type)
            vehicle_types = defaultdict(list)
            for veh_id in all_vehicles:
                veh_type = traci.vehicle.getTypeID(veh_id)
                vehicle_types[veh_type].append(veh_id)
            
            for veh_type, vehicles in vehicle_types.items():
                for i in range(len(vehicles)):
                    for j in range(i+1, min(i+5, len(vehicles))):  # Limit connections
                        self.establish_relationship(vehicles[i], vehicles[j], 'POR')
            
            # Co-location Object Relationship (same edge)
            edge_vehicles = defaultdict(list)
            for veh_id in all_vehicles:
                try:
                    edge = traci.vehicle.getRoadID(veh_id)
                    edge_vehicles[edge].append(veh_id)
                except:
                    pass
            
            for edge, vehicles in edge_vehicles.items():
                if len(vehicles) > 1:
                    for i in range(len(vehicles)):
                        for j in range(i+1, min(i+3, len(vehicles))):
                            key = (vehicles[i], vehicles[j])
                            if key not in self.relationship_types:
                                self.establish_relationship(vehicles[i], vehicles[j], 'CLOR')
                                
        except Exception as e:
            print(f"Error discovering relationships: {e}")
            
    def record_interaction(self, entity1, entity2, is_positive, reason=""):
        """
        Record an interaction between two entities and update trust.
        
        Args:
            entity1: First entity ID
            entity2: Second entity ID
            is_positive: Whether interaction was positive (True) or negative (False)
            reason: Description of interaction
        """
        key = (entity1, entity2)
        reverse_key = (entity2, entity1)
        
        # Update interaction counts
        self.interaction_count[key] += 1
        
        if is_positive:
            self.positive_interactions[key] += 1
            trust_delta = 0.05  # Small trust increase
        else:
            self.negative_interactions[key] += 1
            trust_delta = -0.10  # Larger trust decrease (negative bias)
        
        # Update trust score
        old_trust = self.trust_scores[key]
        new_trust = max(0.0, min(1.0, old_trust + trust_delta))
        self.trust_scores[key] = new_trust
        self.trust_scores[reverse_key] = new_trust
        
        # Log to database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO trust_events 
            (timestamp, entity1, entity2, event_type, trust_change, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.time(), entity1, entity2, 
              'POSITIVE' if is_positive else 'NEGATIVE',
              trust_delta, reason))
        self.db_connection.commit()
        
        event_type = "positive" if is_positive else "negative"
        print(f"[TRUST] {event_type.capitalize()} interaction: {entity1} <-> {entity2}, "
              f"trust: {old_trust:.2f} -> {new_trust:.2f}")
        
    def get_trust_score(self, entity1, entity2):
        """
        Get trust score between two entities.
        
        Returns:
            float: Trust score (0.0 to 1.0)
        """
        return self.trust_scores[(entity1, entity2)]
        
    def get_trusted_neighbors(self, entity, min_trust=0.6):
        """
        Get list of entities that are trusted by the given entity.
        
        Args:
            entity: Entity ID
            min_trust: Minimum trust threshold
            
        Returns:
            list: Trusted entity IDs
        """
        trusted = []
        for neighbor in self.relationships[entity]:
            if self.get_trust_score(entity, neighbor) >= min_trust:
                trusted.append(neighbor)
        return trusted
        
    def validate_message(self, sender, message_type, content):
        """
        Validate a message based on sender's reliability and message content.
        
        Args:
            sender: Sender entity ID
            message_type: Type of message (V2V_BRAKE, V2V_SPEED, etc.)
            content: Message content dictionary
            
        Returns:
            bool: Whether message is valid/trustworthy
        """
        # Check sender's message reliability
        reliability = self.message_reliability[sender]
        
        # Content validation based on message type
        is_valid = True
        
        if message_type == 'V2V_SPEED':
            # Speed should be reasonable
            speed = content.get('speed', 0)
            if speed < 0 or speed > 50:  # 50 m/s = 180 km/h max
                is_valid = False
                
        elif message_type == 'V2V_POSITION':
            # Position should change reasonably between messages
            pos = content.get('position', (0, 0))
            # Could add historical position checking here
            
        elif message_type == 'V2V_INCIDENT':
            # Incident messages should be verifiable
            # For now, trust based on sender reliability
            pass
        
        # Update reliability based on validation
        if is_valid:
            self.message_reliability[sender] = min(1.0, reliability + 0.01)
        else:
            self.message_reliability[sender] = max(0.0, reliability - 0.05)
            print(f"[TRUST] Invalid message from {sender}: {message_type}")
            
        return is_valid and reliability > 0.3  # Threshold for message acceptance
        
    def cooperative_decision(self, vehicle_id, decision_type, options):
        """
        Make a cooperative decision based on trusted neighbors' recommendations.
        
        Args:
            vehicle_id: Vehicle making the decision
            decision_type: Type of decision (route, lane_change, speed)
            options: Available options
            
        Returns:
            Selected option based on trust-weighted voting
        """
        trusted_neighbors = self.get_trusted_neighbors(vehicle_id, min_trust=0.5)
        
        if not trusted_neighbors:
            # No trusted neighbors, make individual decision
            return random.choice(options) if options else None
        
        # Collect recommendations from trusted neighbors
        votes = defaultdict(float)
        
        for neighbor in trusted_neighbors:
            # Simulate neighbor recommendation (in real system, would be from V2V message)
            recommendation = random.choice(options)
            trust_weight = self.get_trust_score(vehicle_id, neighbor)
            votes[recommendation] += trust_weight
        
        # Select option with highest trust-weighted votes
        if votes:
            best_option = max(votes.items(), key=lambda x: x[1])[0]
            return best_option
        
        return random.choice(options) if options else None
        
    def update_trust_scores(self, step):
        """
        Periodic trust score updates based on behavior patterns.
        
        Args:
            step: Current simulation step
        """
        try:
            all_vehicles = traci.vehicle.getIDList()
            
            # Check for cooperative behavior
            for veh_id in all_vehicles:
                neighbors = self.relationships.get(veh_id, set())
                
                for neighbor_id in neighbors:
                    if neighbor_id not in all_vehicles:
                        continue
                    
                    try:
                        # Check for cooperative lane change (maintaining safe distance)
                        veh_speed = traci.vehicle.getSpeed(veh_id)
                        neighbor_speed = traci.vehicle.getSpeed(neighbor_id)
                        
                        # Reward similar speed maintenance (cooperative flow)
                        speed_diff = abs(veh_speed - neighbor_speed)
                        if speed_diff < 2.0:  # Within 2 m/s
                            if random.random() < 0.01:  # Occasional reward
                                self.record_interaction(veh_id, neighbor_id, True,
                                                      "Cooperative speed maintenance")
                                
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error updating trust scores: {e}")
            
    def save_trust_snapshot(self):
        """Save current trust scores to database"""
        cursor = self.db_connection.cursor()
        
        for (entity1, entity2), trust_score in self.trust_scores.items():
            rel_type = self.relationship_types.get((entity1, entity2), 'UNKNOWN')
            interaction_count = self.interaction_count.get((entity1, entity2), 0)
            
            cursor.execute('''
                INSERT INTO trust_scores 
                (timestamp, entity1, entity2, trust_score, relationship_type, interaction_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (time.time(), entity1, entity2, trust_score, rel_type, interaction_count))
        
        self.db_connection.commit()
        
    def generate_report(self):
        """Generate SIoT trust management report"""
        cursor = self.db_connection.cursor()
        
        # Total relationships
        total_relationships = len(self.relationships)
        
        # Average trust score
        if self.trust_scores:
            avg_trust = sum(self.trust_scores.values()) / len(self.trust_scores)
        else:
            avg_trust = 0.5
        
        # Trust distribution
        high_trust = sum(1 for score in self.trust_scores.values() if score > 0.7)
        low_trust = sum(1 for score in self.trust_scores.values() if score < 0.3)
        
        # Interaction statistics
        cursor.execute("SELECT COUNT(*) FROM trust_events")
        total_interactions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trust_events WHERE event_type='POSITIVE'")
        positive_count = cursor.fetchone()[0]
        
        print("\n" + "="*60)
        print("SIOT TRUST MANAGEMENT REPORT")
        print("="*60)
        print(f"Total entities with relationships: {total_relationships}")
        print(f"Total trust relationships: {len(self.trust_scores)}")
        print(f"Average trust score: {avg_trust:.3f}")
        print(f"High trust relationships (>0.7): {high_trust}")
        print(f"Low trust relationships (<0.3): {low_trust}")
        print(f"\nTotal interactions recorded: {total_interactions}")
        print(f"Positive interactions: {positive_count}")
        print(f"Negative interactions: {total_interactions - positive_count}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    manager = SIoTTrustManager()
    
    # Simulate some relationships and interactions
    manager.establish_relationship("vehicle_1", "vehicle_2", "POR")
    manager.establish_relationship("vehicle_1", "vehicle_3", "CLOR")
    
    manager.record_interaction("vehicle_1", "vehicle_2", True, "Cooperative lane change")
    manager.record_interaction("vehicle_1", "vehicle_3", False, "Near collision")
    
    print(f"\nTrust score vehicle_1 -> vehicle_2: {manager.get_trust_score('vehicle_1', 'vehicle_2'):.3f}")
    print(f"Trust score vehicle_1 -> vehicle_3: {manager.get_trust_score('vehicle_1', 'vehicle_3'):.3f}")
    
    manager.generate_report()
    manager.db_connection.close()
