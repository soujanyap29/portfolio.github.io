"""
V2V and V2I Communication Module
Maps to: Computer Networks Course

Implements vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) 
communication protocols with trust-based messaging and state machines.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class MessageType(Enum):
    """Types of messages in V2V/V2I communication"""
    EMERGENCY = "emergency"
    TRAFFIC_UPDATE = "traffic_update"
    COLLISION_WARNING = "collision_warning"
    SPEED_ADVISORY = "speed_advisory"
    LANE_CHANGE = "lane_change"
    SIGNAL_STATE = "signal_state"
    TRUST_QUERY = "trust_query"
    TRUST_RESPONSE = "trust_response"


class MessageState(Enum):
    """FSM states for message processing (Compiler Design mapping)"""
    CREATED = "created"
    VALIDATED = "validated"
    TRANSMITTED = "transmitted"
    RECEIVED = "received"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class Message:
    """
    Message structure for V2V/V2I communication
    Includes security and trust fields
    """
    msg_id: str
    msg_type: MessageType
    sender_id: str
    timestamp: datetime
    payload: Dict
    ttl: int = 5  # Time-to-live in seconds
    priority: int = 1  # 1-5, 5 being highest
    trust_level: float = 0.5  # 0.0-1.0
    state: MessageState = MessageState.CREATED
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Serialize message to dictionary"""
        return {
            'msg_id': self.msg_id,
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp.isoformat(),
            'payload': self.payload,
            'ttl': self.ttl,
            'priority': self.priority,
            'trust_level': self.trust_level,
            'state': self.state.value
        }


@dataclass
class TrustRelationship:
    """
    Trust relationship between two agents
    Maps to: DAA/DSA - Graph algorithms for trust propagation
    """
    agent_a: str
    agent_b: str
    trust_score: float = 0.5  # 0.0 to 1.0
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    
    def update_trust(self, feedback: float) -> None:
        """
        Update trust score based on interaction feedback
        Uses weighted average with decay factor
        """
        alpha = 0.3  # Learning rate
        self.trust_score = (1 - alpha) * self.trust_score + alpha * feedback
        self.trust_score = max(0.0, min(1.0, self.trust_score))
        self.interaction_count += 1
        self.last_interaction = datetime.now()


class TrustGraph:
    """
    Graph structure for managing trust relationships
    Maps to: DAA/DSA - Graph data structure and algorithms
    """
    
    def __init__(self):
        self._edges: Dict[str, Dict[str, TrustRelationship]] = {}
        self._nodes: Set[str] = set()
    
    def add_node(self, agent_id: str) -> None:
        """Add an agent to the trust graph"""
        self._nodes.add(agent_id)
        if agent_id not in self._edges:
            self._edges[agent_id] = {}
    
    def add_edge(self, agent_a: str, agent_b: str, initial_trust: float = 0.5) -> None:
        """Create trust relationship between two agents"""
        self.add_node(agent_a)
        self.add_node(agent_b)
        
        relationship = TrustRelationship(agent_a, agent_b, initial_trust)
        self._edges[agent_a][agent_b] = relationship
        
        # Bidirectional trust
        reverse_relationship = TrustRelationship(agent_b, agent_a, initial_trust)
        self._edges[agent_b][agent_a] = reverse_relationship
    
    def get_trust(self, agent_a: str, agent_b: str) -> float:
        """Get trust score between two agents"""
        if agent_a in self._edges and agent_b in self._edges[agent_a]:
            return self._edges[agent_a][agent_b].trust_score
        return 0.5  # Default neutral trust
    
    def update_trust(self, agent_a: str, agent_b: str, feedback: float) -> None:
        """Update trust relationship based on interaction"""
        if agent_a in self._edges and agent_b in self._edges[agent_a]:
            self._edges[agent_a][agent_b].update_trust(feedback)
    
    def propagate_trust_bfs(self, source: str, max_depth: int = 3) -> Dict[str, float]:
        """
        Breadth-First Search for trust propagation
        Maps to: DAA/DSA - BFS algorithm
        """
        trust_values = {source: 1.0}
        visited = {source}
        queue = [(source, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            if current in self._edges:
                for neighbor in self._edges[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Propagate trust with decay
                        decay_factor = 0.8 ** (depth + 1)
                        trust_values[neighbor] = (
                            trust_values[current] * 
                            self._edges[current][neighbor].trust_score * 
                            decay_factor
                        )
                        queue.append((neighbor, depth + 1))
        
        return trust_values


class MessageValidator:
    """
    Validates messages using FSM
    Maps to: Compiler Design - Finite State Machine
    """
    
    @staticmethod
    def validate(message: Message, trust_threshold: float = 0.3) -> bool:
        """
        Validate message through state transitions
        FSM: CREATED -> VALIDATED -> TRANSMITTED
        """
        if message.state != MessageState.CREATED:
            return False
        
        # Check trust level
        if message.trust_level < trust_threshold:
            message.state = MessageState.FAILED
            return False
        
        # Check TTL
        time_elapsed = (datetime.now() - message.timestamp).total_seconds()
        if time_elapsed > message.ttl:
            message.state = MessageState.FAILED
            return False
        
        # Check payload validity
        if not message.payload or not isinstance(message.payload, dict):
            message.state = MessageState.FAILED
            return False
        
        # Validation successful
        message.state = MessageState.VALIDATED
        return True


class CommunicationChannel:
    """
    Manages message transmission with latency and reliability simulation
    Maps to: Computer Networks - Protocol implementation
    """
    
    def __init__(self, latency_ms: float = 10.0, packet_loss_rate: float = 0.01):
        self.latency_ms = latency_ms
        self.packet_loss_rate = packet_loss_rate
        self._message_queue: List[Message] = []
        self._transmission_log: List[Dict] = []
    
    def send_message(self, message: Message, receiver_ids: List[str]) -> bool:
        """
        Send message to receivers with simulated network conditions
        """
        import random
        
        # Simulate packet loss
        if random.random() < self.packet_loss_rate:
            message.state = MessageState.FAILED
            self._log_transmission(message, receiver_ids, success=False)
            return False
        
        # Validate message
        if not MessageValidator.validate(message):
            self._log_transmission(message, receiver_ids, success=False)
            return False
        
        # Queue message for transmission
        message.state = MessageState.TRANSMITTED
        self._message_queue.append(message)
        self._log_transmission(message, receiver_ids, success=True)
        
        return True
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """
        Receive messages for a specific agent
        Note: In production, this should use async operations instead of blocking sleep
        """
        # Network latency is simulated at transmission time, not reception
        # to avoid blocking in real-time simulation
        
        # Filter messages intended for this agent
        received = []
        remaining = []
        for msg in self._message_queue:
            if msg.state == MessageState.TRANSMITTED:
                # In a full implementation, would check receiver_id
                # For now, mark as received for processing
                msg.state = MessageState.RECEIVED
                received.append(msg)
            else:
                remaining.append(msg)
        
        self._message_queue = remaining
        return received
    
    def _log_transmission(self, message: Message, receivers: List[str], success: bool) -> None:
        """Log message transmission for analytics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message_id': message.msg_id,
            'sender': message.sender_id,
            'receivers': receivers,
            'type': message.msg_type.value,
            'success': success,
            'latency_ms': self.latency_ms
        }
        self._transmission_log.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """Get communication statistics"""
        total = len(self._transmission_log)
        successful = sum(1 for log in self._transmission_log if log['success'])
        
        return {
            'total_messages': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_latency_ms': self.latency_ms
        }


class V2XCommunicationManager:
    """
    Main manager for V2V and V2I communication
    Integrates trust graph and communication channels
    """
    
    def __init__(self):
        self.trust_graph = TrustGraph()
        self.channel = CommunicationChannel()
        self._agent_mailboxes: Dict[str, List[Message]] = {}
    
    def register_agent(self, agent_id: str) -> None:
        """Register an agent in the communication system"""
        self.trust_graph.add_node(agent_id)
        self._agent_mailboxes[agent_id] = []
    
    def send_v2v_message(self, sender_id: str, receiver_id: str, 
                         msg_type: MessageType, payload: Dict) -> bool:
        """
        Send V2V message from one vehicle to another
        """
        trust_score = self.trust_graph.get_trust(sender_id, receiver_id)
        
        message = Message(
            msg_id=f"{sender_id}_{datetime.now().timestamp()}",
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=datetime.now(),
            payload=payload,
            trust_level=trust_score
        )
        
        success = self.channel.send_message(message, [receiver_id])
        
        if success and receiver_id in self._agent_mailboxes:
            self._agent_mailboxes[receiver_id].append(message)
        
        return success
    
    def broadcast_message(self, sender_id: str, msg_type: MessageType, 
                         payload: Dict, radius: float = 100.0) -> int:
        """
        Broadcast message to all agents within radius
        Simulates wireless V2V communication range
        """
        # In real implementation, would use spatial queries
        receivers = list(self._agent_mailboxes.keys())
        receivers.remove(sender_id) if sender_id in receivers else None
        
        count = 0
        for receiver_id in receivers:
            if self.send_v2v_message(sender_id, receiver_id, msg_type, payload):
                count += 1
        
        return count
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get all messages for an agent"""
        if agent_id not in self._agent_mailboxes:
            return []
        
        messages = self._agent_mailboxes[agent_id]
        self._agent_mailboxes[agent_id] = []  # Clear mailbox
        
        return messages
    
    def update_trust_after_interaction(self, agent_a: str, agent_b: str, 
                                      feedback: float) -> None:
        """Update trust based on interaction quality"""
        self.trust_graph.update_trust(agent_a, agent_b, feedback)
