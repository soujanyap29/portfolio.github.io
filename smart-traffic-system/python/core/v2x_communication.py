"""
V2V and V2I Communication Module
Part of Smart Traffic Management System

Implements Vehicle-to-Vehicle and Vehicle-to-Infrastructure communication protocols.
Handles message broadcasting, routing, and processing.

Course Mappings:
- Computer Networks: Message protocols, routing, broadcast domains
- Operating Systems: Message queuing, inter-process communication
- DBMS: Message logging and retrieval
"""

import traci
import sqlite3
import time
import json
from collections import deque, defaultdict
from enum import Enum

class MessageType(Enum):
    """Enumeration of V2X message types"""
    V2V_POSITION = "V2V_POSITION"
    V2V_SPEED = "V2V_SPEED"
    V2V_BRAKE = "V2V_BRAKE"
    V2V_LANE_CHANGE = "V2V_LANE_CHANGE"
    V2V_INCIDENT = "V2V_INCIDENT"
    V2I_SPAT = "V2I_SPAT"  # Signal Phase and Timing
    V2I_MAP = "V2I_MAP"    # Road network map data
    V2I_ROUTING = "V2I_ROUTING"
    V2I_EMERGENCY = "V2I_EMERGENCY"

class V2XCommunication:
    """
    Manages V2V and V2I communication in the traffic network.
    
    Features:
    - Message broadcasting and unicast
    - Communication range modeling
    - Message latency simulation
    - Protocol compliance checking
    - Message logging and analytics
    """
    
    def __init__(self, communication_range=300):
        """
        Initialize V2X communication system.
        
        Args:
            communication_range: Maximum communication range in meters
        """
        self.communication_range = communication_range  # meters
        self.message_queue = deque()  # Message queue for processing
        self.message_history = defaultdict(list)  # entity_id -> messages
        
        # Statistics
        self.stats = {
            'v2v_sent': 0,
            'v2v_received': 0,
            'v2i_sent': 0,
            'v2i_received': 0,
            'messages_dropped': 0,
            'total_latency': 0.0
        }
        
        # RSU (Roadside Unit) positions
        self.rsu_positions = {}
        
        self.db_connection = None
        self.init_database()
        
    def init_database(self):
        """Initialize database for communication logging"""
        self.db_connection = sqlite3.connect('../../database/v2x_communication.db')
        cursor = self.db_connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                sender_id TEXT,
                receiver_id TEXT,
                message_type TEXT,
                content TEXT,
                latency REAL,
                distance REAL,
                success INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS communication_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                stat_type TEXT,
                value REAL
            )
        ''')
        
        self.db_connection.commit()
        
    def register_rsu(self, rsu_id, position):
        """
        Register a Roadside Unit (RSU).
        
        Args:
            rsu_id: RSU identifier
            position: (x, y) position tuple
        """
        self.rsu_positions[rsu_id] = position
        print(f"[V2X] Registered RSU {rsu_id} at position {position}")
        
    def calculate_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1: (x, y) tuple
            pos2: (x, y) tuple
            
        Returns:
            float: Distance in meters
        """
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        
    def simulate_latency(self, distance):
        """
        Simulate communication latency based on distance.
        
        Args:
            distance: Distance in meters
            
        Returns:
            float: Latency in seconds
        """
        # Base latency + distance-based delay
        base_latency = 0.001  # 1ms base
        propagation_delay = distance / 3e8  # Speed of light
        processing_delay = 0.002  # 2ms processing
        
        return base_latency + propagation_delay + processing_delay
        
    def is_in_range(self, sender_pos, receiver_pos):
        """
        Check if receiver is within communication range of sender.
        
        Args:
            sender_pos: Sender position (x, y)
            receiver_pos: Receiver position (x, y)
            
        Returns:
            bool: True if in range
        """
        distance = self.calculate_distance(sender_pos, receiver_pos)
        return distance <= self.communication_range
        
    def broadcast_v2v(self, sender_id, message_type, content):
        """
        Broadcast a V2V message to nearby vehicles.
        
        Args:
            sender_id: Sender vehicle ID
            message_type: MessageType enum value
            content: Message content dictionary
            
        Returns:
            int: Number of vehicles that received the message
        """
        try:
            sender_pos = traci.vehicle.getPosition(sender_id)
            all_vehicles = traci.vehicle.getIDList()
            
            received_count = 0
            
            for receiver_id in all_vehicles:
                if receiver_id == sender_id:
                    continue
                
                try:
                    receiver_pos = traci.vehicle.getPosition(receiver_id)
                    
                    if self.is_in_range(sender_pos, receiver_pos):
                        distance = self.calculate_distance(sender_pos, receiver_pos)
                        latency = self.simulate_latency(distance)
                        
                        # Deliver message
                        self.deliver_message(sender_id, receiver_id, message_type, 
                                           content, latency, distance)
                        received_count += 1
                        
                except:
                    pass
            
            self.stats['v2v_sent'] += 1
            self.stats['v2v_received'] += received_count
            
            return received_count
            
        except Exception as e:
            print(f"Error broadcasting V2V message: {e}")
            return 0
            
    def send_v2i(self, sender_id, rsu_id, message_type, content):
        """
        Send a V2I message from vehicle to RSU.
        
        Args:
            sender_id: Sender vehicle ID
            rsu_id: Target RSU ID
            message_type: MessageType enum value
            content: Message content dictionary
            
        Returns:
            bool: Success status
        """
        try:
            sender_pos = traci.vehicle.getPosition(sender_id)
            rsu_pos = self.rsu_positions.get(rsu_id)
            
            if not rsu_pos:
                print(f"[V2X] RSU {rsu_id} not found")
                return False
            
            if self.is_in_range(sender_pos, rsu_pos):
                distance = self.calculate_distance(sender_pos, rsu_pos)
                latency = self.simulate_latency(distance)
                
                self.deliver_message(sender_id, rsu_id, message_type, 
                                   content, latency, distance)
                
                self.stats['v2i_sent'] += 1
                return True
            else:
                self.stats['messages_dropped'] += 1
                return False
                
        except Exception as e:
            print(f"Error sending V2I message: {e}")
            return False
            
    def send_i2v(self, rsu_id, receiver_id, message_type, content):
        """
        Send an I2V message from RSU to vehicle.
        
        Args:
            rsu_id: Sender RSU ID
            receiver_id: Target vehicle ID
            message_type: MessageType enum value
            content: Message content dictionary
            
        Returns:
            bool: Success status
        """
        try:
            rsu_pos = self.rsu_positions.get(rsu_id)
            if not rsu_pos:
                return False
            
            receiver_pos = traci.vehicle.getPosition(receiver_id)
            
            if self.is_in_range(rsu_pos, receiver_pos):
                distance = self.calculate_distance(rsu_pos, receiver_pos)
                latency = self.simulate_latency(distance)
                
                self.deliver_message(rsu_id, receiver_id, message_type, 
                                   content, latency, distance)
                
                self.stats['v2i_received'] += 1
                return True
            else:
                self.stats['messages_dropped'] += 1
                return False
                
        except Exception as e:
            print(f"Error sending I2V message: {e}")
            return False
            
    def deliver_message(self, sender_id, receiver_id, message_type, 
                       content, latency, distance):
        """
        Deliver a message and log it to database.
        
        Args:
            sender_id: Sender entity ID
            receiver_id: Receiver entity ID
            message_type: MessageType enum value
            content: Message content dictionary
            latency: Simulated latency in seconds
            distance: Communication distance in meters
        """
        # Store in message history
        message = {
            'timestamp': time.time(),
            'sender': sender_id,
            'receiver': receiver_id,
            'type': message_type.value if isinstance(message_type, MessageType) else message_type,
            'content': content,
            'latency': latency,
            'distance': distance
        }
        
        self.message_history[receiver_id].append(message)
        
        # Keep only recent messages (last 100)
        if len(self.message_history[receiver_id]) > 100:
            self.message_history[receiver_id] = self.message_history[receiver_id][-100:]
        
        # Log to database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO messages 
            (timestamp, sender_id, receiver_id, message_type, content, latency, distance, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), sender_id, receiver_id, 
              message['type'], json.dumps(content), latency, distance, 1))
        
        self.db_connection.commit()
        
        self.stats['total_latency'] += latency
        
    def broadcast_spat(self, junction_id, rsu_id):
        """
        Broadcast Signal Phase and Timing (SPaT) information.
        
        Args:
            junction_id: Traffic light junction ID
            rsu_id: RSU broadcasting the SPaT
        """
        try:
            # Get signal state
            state = traci.trafficlight.getRedYellowGreenState(junction_id)
            phase = traci.trafficlight.getPhase(junction_id)
            next_switch = traci.trafficlight.getNextSwitch(junction_id)
            
            spat_content = {
                'junction_id': junction_id,
                'state': state,
                'phase': phase,
                'next_switch': next_switch,
                'timestamp': time.time()
            }
            
            # Broadcast to all vehicles in range
            rsu_pos = self.rsu_positions.get(rsu_id)
            if not rsu_pos:
                return
            
            all_vehicles = traci.vehicle.getIDList()
            broadcast_count = 0
            
            for vehicle_id in all_vehicles:
                try:
                    vehicle_pos = traci.vehicle.getPosition(vehicle_id)
                    if self.is_in_range(rsu_pos, vehicle_pos):
                        distance = self.calculate_distance(rsu_pos, vehicle_pos)
                        latency = self.simulate_latency(distance)
                        self.deliver_message(rsu_id, vehicle_id, MessageType.V2I_SPAT,
                                           spat_content, latency, distance)
                        broadcast_count += 1
                except:
                    pass
            
            if broadcast_count > 0:
                print(f"[V2X] SPaT broadcast from {rsu_id} reached {broadcast_count} vehicles")
                
        except Exception as e:
            print(f"Error broadcasting SPaT: {e}")
            
    def process_messages(self, vehicle_id):
        """
        Process messages received by a vehicle and take appropriate actions.
        
        Args:
            vehicle_id: Vehicle processing messages
        """
        messages = self.message_history.get(vehicle_id, [])
        recent_messages = [msg for msg in messages if time.time() - msg['timestamp'] < 5.0]
        
        for msg in recent_messages:
            msg_type = msg['type']
            content = msg['content']
            
            try:
                if msg_type == 'V2V_BRAKE':
                    # Brake warning received
                    brake_sender = msg['sender']
                    # Could adjust own speed here
                    pass
                    
                elif msg_type == 'V2V_INCIDENT':
                    # Incident alert received
                    incident_location = content.get('location')
                    # Could reroute if necessary
                    pass
                    
                elif msg_type == 'V2I_SPAT':
                    # Signal timing received
                    next_switch = content.get('next_switch', 0)
                    # Could optimize speed to catch green
                    pass
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                
    def generate_report(self):
        """Generate V2X communication statistics report"""
        cursor = self.db_connection.cursor()
        
        # Total messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        # Messages by type
        cursor.execute("""
            SELECT message_type, COUNT(*) 
            FROM messages 
            GROUP BY message_type
        """)
        messages_by_type = cursor.fetchall()
        
        # Average latency
        cursor.execute("SELECT AVG(latency) FROM messages")
        avg_latency = cursor.fetchone()[0] or 0
        
        # Average distance
        cursor.execute("SELECT AVG(distance) FROM messages")
        avg_distance = cursor.fetchone()[0] or 0
        
        print("\n" + "="*60)
        print("V2X COMMUNICATION REPORT")
        print("="*60)
        print(f"Total messages exchanged: {total_messages}")
        print(f"V2V messages sent: {self.stats['v2v_sent']}")
        print(f"V2V messages received: {self.stats['v2v_received']}")
        print(f"V2I messages sent: {self.stats['v2i_sent']}")
        print(f"V2I messages received: {self.stats['v2i_received']}")
        print(f"Messages dropped (out of range): {self.stats['messages_dropped']}")
        print(f"\nAverage latency: {avg_latency*1000:.3f} ms")
        print(f"Average communication distance: {avg_distance:.2f} m")
        print(f"\nMessages by type:")
        for msg_type, count in messages_by_type:
            print(f"  {msg_type}: {count}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    v2x = V2XCommunication(communication_range=300)
    
    # Register RSUs
    v2x.register_rsu("rsu_1", (500, 500))
    v2x.register_rsu("rsu_2", (1500, 500))
    
    print("\nV2X Communication System initialized")
    print(f"Communication range: {v2x.communication_range} meters")
    
    v2x.generate_report()
    v2x.db_connection.close()
