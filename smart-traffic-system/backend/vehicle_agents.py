"""
Vehicle Agent Base Classes - OOP Implementation
Maps to: Object-Oriented Programming Course

This module defines the class hierarchy for all vehicle types in the system.
Each vehicle type inherits from a base Agent class and implements specific behaviors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import uuid


class VehicleType(Enum):
    """Enumeration of all supported vehicle types"""
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    BICYCLE = "bicycle"
    TRAM = "tram"
    PEDESTRIAN = "pedestrian"


@dataclass
class VehicleConfig:
    """Configuration dataclass for vehicle physical properties"""
    length: float  # meters
    width: float   # meters
    height: float  # meters
    color: str     # hex color code
    sumo_shape: str  # SUMO vType shape
    icon_path: str  # Dashboard icon path
    max_speed: float  # m/s
    acceleration: float  # m/s^2
    deceleration: float  # m/s^2


class Agent(ABC):
    """
    Abstract base class for all agents in the traffic system
    Demonstrates OOP principles: Abstraction, Encapsulation
    """
    
    def __init__(self, agent_id: str, vehicle_type: VehicleType, config: VehicleConfig):
        self._id = agent_id if agent_id else str(uuid.uuid4())
        self._vehicle_type = vehicle_type
        self._config = config
        self._position: Tuple[float, float] = (0.0, 0.0)
        self._speed: float = 0.0
        self._route: List[str] = []
        self._state: str = "idle"
        
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def vehicle_type(self) -> VehicleType:
        return self._vehicle_type
    
    @property
    def config(self) -> VehicleConfig:
        return self._config
    
    @abstractmethod
    def update_state(self, time_step: float) -> None:
        """Update agent state for each simulation step"""
        pass
    
    @abstractmethod
    def handle_message(self, message: Dict) -> None:
        """Handle V2V/V2I messages"""
        pass
    
    def get_position(self) -> Tuple[float, float]:
        return self._position
    
    def set_position(self, position: Tuple[float, float]) -> None:
        self._position = position
    
    def get_speed(self) -> float:
        return self._speed
    
    def set_speed(self, speed: float) -> None:
        self._speed = min(speed, self._config.max_speed)


class MotorizedVehicle(Agent):
    """
    Base class for all motorized vehicles
    Demonstrates OOP: Inheritance, Polymorphism
    """
    
    def __init__(self, agent_id: str, vehicle_type: VehicleType, config: VehicleConfig):
        super().__init__(agent_id, vehicle_type, config)
        self._fuel_level: float = 100.0
        self._engine_state: str = "off"
        self._communication_range: float = 100.0  # meters for V2V
    
    def update_state(self, time_step: float) -> None:
        """Update motorized vehicle state"""
        if self._engine_state == "on":
            self._fuel_level -= 0.01 * time_step
    
    def handle_message(self, message: Dict) -> None:
        """Handle V2V/V2I communication messages"""
        if message.get("type") == "emergency":
            self._state = "yielding"
        elif message.get("type") == "traffic_update":
            self._update_route(message.get("recommended_route", []))
    
    def _update_route(self, new_route: List[str]) -> None:
        self._route = new_route
    
    def start_engine(self) -> None:
        self._engine_state = "on"
    
    def stop_engine(self) -> None:
        self._engine_state = "off"


class Car(MotorizedVehicle):
    """
    Car vehicle class
    Specific implementation for standard passenger cars
    """
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=4.5,
            width=1.8,
            height=1.5,
            color="#FF0000",
            sumo_shape="passenger",
            icon_path="/icons/car.svg",
            max_speed=33.3,  # ~120 km/h
            acceleration=2.6,
            deceleration=4.5
        )
        super().__init__(agent_id, VehicleType.CAR, config)
        self._passenger_count: int = 1
    
    def update_state(self, time_step: float) -> None:
        super().update_state(time_step)
        # Car-specific state updates


class Bus(MotorizedVehicle):
    """Bus vehicle class for public transit"""
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=12.0,
            width=2.5,
            height=3.2,
            color="#0000FF",
            sumo_shape="bus",
            icon_path="/icons/bus.svg",
            max_speed=22.2,  # ~80 km/h
            acceleration=1.2,
            deceleration=3.5
        )
        super().__init__(agent_id, VehicleType.BUS, config)
        self._passenger_count: int = 0
        self._max_passengers: int = 60
        self._stops: List[str] = []
    
    def update_state(self, time_step: float) -> None:
        super().update_state(time_step)
        # Bus-specific state updates (stops, passenger boarding)


class Truck(MotorizedVehicle):
    """Truck vehicle class for freight transport"""
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=16.5,
            width=2.6,
            height=4.0,
            color="#FFA500",
            sumo_shape="truck",
            icon_path="/icons/truck.svg",
            max_speed=25.0,  # ~90 km/h
            acceleration=1.0,
            deceleration=3.0
        )
        super().__init__(agent_id, VehicleType.TRUCK, config)
        self._cargo_weight: float = 0.0
        self._max_cargo: float = 20000.0  # kg


class Motorcycle(MotorizedVehicle):
    """Motorcycle vehicle class"""
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=2.2,
            width=0.8,
            height=1.3,
            color="#00FF00",
            sumo_shape="motorcycle",
            icon_path="/icons/motorcycle.svg",
            max_speed=36.1,  # ~130 km/h
            acceleration=3.5,
            deceleration=5.0
        )
        super().__init__(agent_id, VehicleType.MOTORCYCLE, config)


class AutoRickshaw(MotorizedVehicle):
    """Auto-rickshaw (three-wheeler) vehicle class"""
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=2.7,
            width=1.3,
            height=1.7,
            color="#FFFF00",
            sumo_shape="delivery",
            icon_path="/icons/auto_rickshaw.svg",
            max_speed=13.9,  # ~50 km/h
            acceleration=1.8,
            deceleration=3.5
        )
        super().__init__(agent_id, VehicleType.AUTO_RICKSHAW, config)
        self._passenger_count: int = 0
        self._max_passengers: int = 3


class Bicycle(Agent):
    """
    Bicycle class - non-motorized vehicle
    Demonstrates different behavior patterns
    """
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=1.8,
            width=0.6,
            height=1.1,
            color="#00FFFF",
            sumo_shape="bicycle",
            icon_path="/icons/bicycle.svg",
            max_speed=6.9,  # ~25 km/h
            acceleration=1.5,
            deceleration=2.5
        )
        super().__init__(agent_id, VehicleType.BICYCLE, config)
    
    def update_state(self, time_step: float) -> None:
        """Bicycle-specific state updates"""
        pass
    
    def handle_message(self, message: Dict) -> None:
        """Bicycles have limited communication capabilities"""
        pass


class Pedestrian(Agent):
    """Pedestrian agent class"""
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=0.6,
            width=0.4,
            height=1.7,
            color="#FF00FF",
            sumo_shape="pedestrian",
            icon_path="/icons/pedestrian.svg",
            max_speed=1.4,  # ~5 km/h
            acceleration=0.5,
            deceleration=1.0
        )
        super().__init__(agent_id, VehicleType.PEDESTRIAN, config)
        self._crossing: bool = False
    
    def update_state(self, time_step: float) -> None:
        """Pedestrian movement logic"""
        pass
    
    def handle_message(self, message: Dict) -> None:
        """Pedestrians can receive crossing signals"""
        if message.get("type") == "crossing_signal":
            self._crossing = message.get("can_cross", False)


class Tram(MotorizedVehicle):
    """Tram vehicle class for rail-based transit"""
    
    def __init__(self, agent_id: str = None):
        config = VehicleConfig(
            length=30.0,
            width=2.4,
            height=3.5,
            color="#800080",
            sumo_shape="rail",
            icon_path="/icons/tram.svg",
            max_speed=19.4,  # ~70 km/h
            acceleration=1.0,
            deceleration=2.5
        )
        super().__init__(agent_id, VehicleType.TRAM, config)
        self._track_id: str = ""


# Vehicle Configuration Lookup Table
VEHICLE_CONFIGS = {
    VehicleType.CAR: Car,
    VehicleType.BUS: Bus,
    VehicleType.TRUCK: Truck,
    VehicleType.MOTORCYCLE: Motorcycle,
    VehicleType.AUTO_RICKSHAW: AutoRickshaw,
    VehicleType.BICYCLE: Bicycle,
    VehicleType.PEDESTRIAN: Pedestrian,
    VehicleType.TRAM: Tram,
}


def create_vehicle(vehicle_type: VehicleType, agent_id: str = None) -> Agent:
    """
    Factory method for creating vehicle instances
    Demonstrates OOP: Factory Pattern
    """
    vehicle_class = VEHICLE_CONFIGS.get(vehicle_type)
    if vehicle_class is None:
        raise ValueError(f"Unsupported vehicle type: {vehicle_type}")
    return vehicle_class(agent_id)
