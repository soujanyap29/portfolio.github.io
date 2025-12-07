"""
Database Schema and Event Logging
Maps to: Database Management Systems Course

Implements persistent storage for simulation events, vehicle states,
and analytics queries.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Dict, List, Optional
import json

Base = declarative_base()


class Vehicle(Base):
    """
    Vehicle entity table
    Stores static information about each vehicle
    """
    __tablename__ = 'vehicles'
    
    id = Column(String(50), primary_key=True)
    vehicle_type = Column(String(30), nullable=False, index=True)
    config_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    events = relationship("SimulationEvent", back_populates="vehicle")
    states = relationship("VehicleState", back_populates="vehicle")
    
    def __repr__(self):
        return f"<Vehicle(id='{self.id}', type='{self.vehicle_type}')>"


class SimulationEvent(Base):
    """
    Simulation events table
    Logs all significant events during simulation
    """
    __tablename__ = 'simulation_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(50), nullable=False, index=True)
    vehicle_id = Column(String(50), ForeignKey('vehicles.id'), index=True)
    event_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    position_x = Column(Float)
    position_y = Column(Float)
    speed = Column(Float)
    event_data = Column(JSON)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="events")
    
    # Composite index for common queries
    __table_args__ = (
        Index('idx_sim_vehicle_time', 'simulation_id', 'vehicle_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Event(type='{self.event_type}', vehicle='{self.vehicle_id}', time='{self.timestamp}')>"


class VehicleState(Base):
    """
    Vehicle state snapshots
    Stores periodic state information for each vehicle
    """
    __tablename__ = 'vehicle_states'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(50), nullable=False, index=True)
    vehicle_id = Column(String(50), ForeignKey('vehicles.id'), index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    acceleration = Column(Float)
    lane_id = Column(String(50))
    route_index = Column(Integer)
    state_data = Column(JSON)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="states")
    
    __table_args__ = (
        Index('idx_vehicle_time', 'vehicle_id', 'timestamp'),
    )


class CommunicationLog(Base):
    """
    V2V/V2I communication logs
    Tracks all message exchanges
    """
    __tablename__ = 'communication_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(50), nullable=False, index=True)
    message_id = Column(String(100), nullable=False)
    sender_id = Column(String(50), nullable=False, index=True)
    receiver_id = Column(String(50), index=True)
    message_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    trust_level = Column(Float)
    success = Column(Integer)  # 1 for success, 0 for failure
    latency_ms = Column(Float)
    message_data = Column(JSON)
    
    __table_args__ = (
        Index('idx_comm_time', 'simulation_id', 'timestamp'),
    )


class TrustScore(Base):
    """
    Trust relationships between agents
    Stores trust graph edges
    """
    __tablename__ = 'trust_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(50), nullable=False, index=True)
    agent_a = Column(String(50), nullable=False, index=True)
    agent_b = Column(String(50), nullable=False, index=True)
    trust_score = Column(Float, nullable=False)
    interaction_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_trust_pair', 'agent_a', 'agent_b'),
    )


class Simulation(Base):
    """
    Simulation metadata
    Tracks simulation runs and configurations
    """
    __tablename__ = 'simulations'
    
    id = Column(String(50), primary_key=True)
    scenario_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    config_data = Column(JSON)
    statistics = Column(JSON)
    status = Column(String(20))  # running, completed, failed
    
    def __repr__(self):
        return f"<Simulation(id='{self.id}', scenario='{self.scenario_name}')>"


class DatabaseManager:
    """
    Database connection and session management
    Handles CRUD operations and queries
    """
    
    def __init__(self, connection_string: str = None):
        """
        Initialize database manager with secure connection string
        
        Args:
            connection_string: Database URL. If None, uses secure default location.
        """
        if connection_string is None:
            import os
            # Use a secure default location in the project directory
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'traffic_simulation.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            connection_string = f"sqlite:///{db_path}"
        
        self.engine = create_engine(connection_string, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def log_vehicle_state(self, simulation_id: str, vehicle_id: str, 
                         position: tuple, speed: float, state_data: Dict) -> None:
        """Log vehicle state snapshot"""
        session = self.get_session()
        try:
            state = VehicleState(
                simulation_id=simulation_id,
                vehicle_id=vehicle_id,
                timestamp=datetime.utcnow(),
                position_x=position[0],
                position_y=position[1],
                speed=speed,
                state_data=state_data
            )
            session.add(state)
            session.commit()
        finally:
            session.close()
    
    def log_event(self, simulation_id: str, vehicle_id: str, event_type: str,
                  position: tuple, speed: float, event_data: Dict) -> None:
        """Log simulation event"""
        session = self.get_session()
        try:
            event = SimulationEvent(
                simulation_id=simulation_id,
                vehicle_id=vehicle_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                position_x=position[0],
                position_y=position[1],
                speed=speed,
                event_data=event_data
            )
            session.add(event)
            session.commit()
        finally:
            session.close()
    
    def log_communication(self, simulation_id: str, message_id: str, 
                         sender_id: str, receiver_id: str, message_type: str,
                         trust_level: float, success: bool, latency_ms: float,
                         message_data: Dict) -> None:
        """Log V2V/V2I communication"""
        session = self.get_session()
        try:
            comm_log = CommunicationLog(
                simulation_id=simulation_id,
                message_id=message_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                timestamp=datetime.utcnow(),
                trust_level=trust_level,
                success=1 if success else 0,
                latency_ms=latency_ms,
                message_data=message_data
            )
            session.add(comm_log)
            session.commit()
        finally:
            session.close()
    
    def update_trust_score(self, simulation_id: str, agent_a: str, 
                          agent_b: str, trust_score: float) -> None:
        """Update or create trust score"""
        session = self.get_session()
        try:
            trust = session.query(TrustScore).filter_by(
                simulation_id=simulation_id,
                agent_a=agent_a,
                agent_b=agent_b
            ).first()
            
            if trust:
                trust.trust_score = trust_score
                trust.interaction_count += 1
                trust.last_updated = datetime.utcnow()
            else:
                trust = TrustScore(
                    simulation_id=simulation_id,
                    agent_a=agent_a,
                    agent_b=agent_b,
                    trust_score=trust_score,
                    interaction_count=1
                )
                session.add(trust)
            
            session.commit()
        finally:
            session.close()
    
    def get_vehicle_trajectory(self, vehicle_id: str, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Get vehicle trajectory data
        Analytics query for vehicle movement analysis
        """
        session = self.get_session()
        try:
            query = session.query(VehicleState).filter_by(vehicle_id=vehicle_id)
            
            if start_time:
                query = query.filter(VehicleState.timestamp >= start_time)
            if end_time:
                query = query.filter(VehicleState.timestamp <= end_time)
            
            states = query.order_by(VehicleState.timestamp).all()
            
            trajectory = [
                {
                    'timestamp': state.timestamp.isoformat(),
                    'position': (state.position_x, state.position_y),
                    'speed': state.speed,
                    'acceleration': state.acceleration
                }
                for state in states
            ]
            
            return trajectory
        finally:
            session.close()
    
    def get_events_by_type(self, simulation_id: str, event_type: str) -> List[Dict]:
        """Get all events of a specific type"""
        session = self.get_session()
        try:
            events = session.query(SimulationEvent).filter_by(
                simulation_id=simulation_id,
                event_type=event_type
            ).all()
            
            return [
                {
                    'vehicle_id': event.vehicle_id,
                    'timestamp': event.timestamp.isoformat(),
                    'position': (event.position_x, event.position_y),
                    'speed': event.speed,
                    'data': event.event_data
                }
                for event in events
            ]
        finally:
            session.close()
    
    def get_communication_statistics(self, simulation_id: str) -> Dict:
        """
        Get communication statistics for a simulation
        Analytics aggregation query
        """
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            stats = session.query(
                func.count(CommunicationLog.id).label('total_messages'),
                func.sum(CommunicationLog.success).label('successful_messages'),
                func.avg(CommunicationLog.latency_ms).label('avg_latency'),
                func.avg(CommunicationLog.trust_level).label('avg_trust')
            ).filter_by(simulation_id=simulation_id).first()
            
            return {
                'total_messages': stats.total_messages or 0,
                'successful_messages': stats.successful_messages or 0,
                'average_latency_ms': float(stats.avg_latency or 0.0),
                'average_trust': float(stats.avg_trust or 0.0),
                'success_rate': (stats.successful_messages / stats.total_messages * 100 
                               if stats.total_messages else 0.0)
            }
        finally:
            session.close()
    
    def get_vehicle_type_statistics(self, simulation_id: str) -> Dict[str, Dict]:
        """
        Get statistics grouped by vehicle type
        Complex aggregation query
        """
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            # Get average speed by vehicle type
            speed_stats = session.query(
                Vehicle.vehicle_type,
                func.avg(VehicleState.speed).label('avg_speed'),
                func.count(VehicleState.id).label('state_count')
            ).join(VehicleState).filter(
                VehicleState.simulation_id == simulation_id
            ).group_by(Vehicle.vehicle_type).all()
            
            result = {}
            for stat in speed_stats:
                result[stat.vehicle_type] = {
                    'average_speed': float(stat.avg_speed or 0.0),
                    'data_points': stat.state_count
                }
            
            return result
        finally:
            session.close()


# ETL (Extract, Transform, Load) Functions
class ETLPipeline:
    """
    ETL pipeline for analytics processing
    Maps to: DBMS - Data warehousing and ETL
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def extract_simulation_data(self, simulation_id: str) -> Dict:
        """Extract all data for a simulation"""
        session = self.db.get_session()
        try:
            simulation = session.query(Simulation).filter_by(id=simulation_id).first()
            if not simulation:
                return {}
            
            return {
                'simulation': {
                    'id': simulation.id,
                    'scenario': simulation.scenario_name,
                    'start_time': simulation.start_time.isoformat(),
                    'duration': simulation.duration_seconds,
                    'config': simulation.config_data
                },
                'statistics': self.db.get_communication_statistics(simulation_id),
                'vehicle_stats': self.db.get_vehicle_type_statistics(simulation_id)
            }
        finally:
            session.close()
    
    def transform_for_dashboard(self, raw_data: Dict) -> Dict:
        """Transform data for dashboard visualization"""
        # Simplify and format data for frontend consumption
        return {
            'scenario': raw_data.get('simulation', {}).get('scenario', 'Unknown'),
            'metrics': {
                'total_messages': raw_data.get('statistics', {}).get('total_messages', 0),
                'success_rate': round(raw_data.get('statistics', {}).get('success_rate', 0.0), 2),
                'avg_trust': round(raw_data.get('statistics', {}).get('average_trust', 0.0), 2)
            },
            'vehicle_performance': raw_data.get('vehicle_stats', {})
        }
    
    def load_to_cache(self, transformed_data: Dict, cache_key: str) -> None:
        """
        Load transformed data to cache for quick access
        
        Note: This is a placeholder for production cache integration.
        In a production system, this would integrate with Redis, Memcached,
        or another caching solution for improved performance.
        
        For now, data is simply returned without caching.
        """
        # TODO: Implement actual caching when deploying to production
        # Example: redis_client.set(cache_key, json.dumps(transformed_data))
        pass
