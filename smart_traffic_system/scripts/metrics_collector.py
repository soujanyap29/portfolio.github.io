#!/usr/bin/env python3
"""
Metrics Collector Module

Collects and stores simulation metrics including:
- Traffic metrics (travel time, waiting time, speed, etc.)
- Communication metrics (message count, delivery rate, latency)
- SIoT metrics (trust scores, cooperation levels)
- Environmental metrics (emissions)
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import csv
import time

# Ensure SUMO tools are in path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

try:
    import traci
except ImportError:
    traci = None

logger = logging.getLogger('MetricsCollector')


@dataclass
class StepMetrics:
    """Metrics for a single simulation step."""
    timestamp: float
    vehicle_count: int
    average_speed: float
    total_waiting_time: float
    total_travel_time: float
    stopped_vehicles: int
    v2v_messages: int
    v2i_messages: int
    average_trust: float
    emissions: Dict[str, float] = field(default_factory=dict)


@dataclass
class VehicleTripMetrics:
    """Metrics for a single vehicle trip."""
    vehicle_id: str
    vehicle_type: str
    departure_time: float
    arrival_time: float = 0.0
    travel_time: float = 0.0
    waiting_time: float = 0.0
    stops: int = 0
    distance: float = 0.0
    average_speed: float = 0.0
    emissions: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and manages simulation metrics.
    
    Features:
    - Step-by-step metric collection
    - Vehicle trip tracking
    - Aggregate statistics calculation
    - Export to various formats
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize metrics collector.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Step metrics history
        self.step_metrics: List[StepMetrics] = []
        
        # Vehicle trip metrics
        self.trip_metrics: Dict[str, VehicleTripMetrics] = {}
        self.completed_trips: List[VehicleTripMetrics] = []
        
        # Communication metrics
        self.v2v_message_count = 0
        self.v2i_message_count = 0
        self.message_latencies: List[float] = []
        
        # SIoT metrics
        self.trust_history: List[Tuple[float, float]] = []  # (time, avg_trust)
        self.cooperation_events = 0
        
        # Queue metrics
        self.queue_lengths: Dict[str, List[int]] = defaultdict(list)
        
        # Emission totals
        self.total_emissions = defaultdict(float)
        
    def collect_step_metrics(self, sim_time: float, vehicle_states: Dict,
                              v2v_messages: Dict, v2i_messages: Dict,
                              trust_scores: Dict) -> None:
        """
        Collect metrics for a simulation step.
        
        Args:
            sim_time: Current simulation time
            vehicle_states: Dictionary of vehicle states
            v2v_messages: V2V messages for this step
            v2i_messages: V2I messages for this step
            trust_scores: Current trust score network
        """
        # Calculate step metrics
        vehicle_count = len(vehicle_states)
        
        total_speed = 0
        total_waiting = 0
        stopped_count = 0
        
        for veh_id, state in vehicle_states.items():
            speed = state.get('speed', 0)
            waiting = state.get('waiting_time', 0)
            
            total_speed += speed
            total_waiting += waiting
            
            if speed < 0.1:
                stopped_count += 1
            
            # Update trip metrics
            self._update_trip_metrics(veh_id, state, sim_time)
            
            # Collect emissions
            emissions = self._collect_vehicle_emissions(veh_id)
            for em_type, value in emissions.items():
                self.total_emissions[em_type] += value
        
        avg_speed = total_speed / max(1, vehicle_count)
        
        # Count messages
        v2v_count = sum(len(m.get('beacons', [])) + len(m.get('alerts', []))
                       for m in v2v_messages.values())
        v2i_count = sum(len(m.get('spat', [])) + len(m.get('advisories', []))
                       for m in v2i_messages.values())
        
        self.v2v_message_count += v2v_count
        self.v2i_message_count += v2i_count
        
        # Calculate average trust
        all_trust = []
        for targets in trust_scores.values():
            all_trust.extend(targets.values())
        avg_trust = sum(all_trust) / max(1, len(all_trust)) if all_trust else 0.5
        
        self.trust_history.append((sim_time, avg_trust))
        
        # Create step metrics
        step = StepMetrics(
            timestamp=sim_time,
            vehicle_count=vehicle_count,
            average_speed=avg_speed,
            total_waiting_time=total_waiting,
            total_travel_time=0,  # Calculated at trip completion
            stopped_vehicles=stopped_count,
            v2v_messages=v2v_count,
            v2i_messages=v2i_count,
            average_trust=avg_trust,
            emissions=dict(self.total_emissions)
        )
        
        self.step_metrics.append(step)
        
        # Check for completed trips
        self._check_completed_trips(set(vehicle_states.keys()), sim_time)
    
    def _update_trip_metrics(self, vehicle_id: str, state: Dict,
                              sim_time: float) -> None:
        """Update trip metrics for a vehicle."""
        if vehicle_id not in self.trip_metrics:
            self.trip_metrics[vehicle_id] = VehicleTripMetrics(
                vehicle_id=vehicle_id,
                vehicle_type=state.get('type', 'car'),
                departure_time=sim_time
            )
        
        trip = self.trip_metrics[vehicle_id]
        trip.waiting_time = state.get('waiting_time', trip.waiting_time)
        
        # Update stops count
        if state.get('speed', 0) < 0.1:
            # Simplified: increment stops when stopped
            pass
    
    def _check_completed_trips(self, active_vehicles: set,
                                sim_time: float) -> None:
        """Check for vehicles that have completed their trips."""
        completed = []
        
        for veh_id, trip in self.trip_metrics.items():
            if veh_id not in active_vehicles:
                trip.arrival_time = sim_time
                trip.travel_time = sim_time - trip.departure_time
                completed.append(veh_id)
                self.completed_trips.append(trip)
        
        for veh_id in completed:
            del self.trip_metrics[veh_id]
    
    def _collect_vehicle_emissions(self, vehicle_id: str) -> Dict[str, float]:
        """Collect emission data for a vehicle."""
        if traci is None:
            return {}
        
        try:
            return {
                'CO2': traci.vehicle.getCO2Emission(vehicle_id),
                'CO': traci.vehicle.getCOEmission(vehicle_id),
                'HC': traci.vehicle.getHCEmission(vehicle_id),
                'NOx': traci.vehicle.getNOxEmission(vehicle_id),
                'PMx': traci.vehicle.getPMxEmission(vehicle_id),
                'fuel': traci.vehicle.getFuelConsumption(vehicle_id)
            }
        except Exception:
            return {}
    
    def generate_summary(self) -> Dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        summary = {}
        
        # Traffic metrics
        if self.completed_trips:
            travel_times = [t.travel_time for t in self.completed_trips]
            waiting_times = [t.waiting_time for t in self.completed_trips]
            
            summary['traffic'] = {
                'total_trips': len(self.completed_trips),
                'average_travel_time': sum(travel_times) / len(travel_times),
                'max_travel_time': max(travel_times),
                'min_travel_time': min(travel_times),
                'average_waiting_time': sum(waiting_times) / len(waiting_times),
                'total_waiting_time': sum(waiting_times)
            }
        else:
            summary['traffic'] = {
                'total_trips': 0,
                'average_travel_time': 0,
                'average_waiting_time': 0
            }
        
        # Speed metrics
        if self.step_metrics:
            speeds = [s.average_speed for s in self.step_metrics]
            summary['speed'] = {
                'average_speed': sum(speeds) / len(speeds),
                'max_average_speed': max(speeds),
                'min_average_speed': min(speeds)
            }
        
        # Communication metrics
        summary['communication'] = {
            'total_v2v_messages': self.v2v_message_count,
            'total_v2i_messages': self.v2i_message_count,
            'total_messages': self.v2v_message_count + self.v2i_message_count
        }
        
        # SIoT metrics
        if self.trust_history:
            trust_values = [t[1] for t in self.trust_history]
            summary['siot'] = {
                'initial_trust': trust_values[0] if trust_values else 0.5,
                'final_trust': trust_values[-1] if trust_values else 0.5,
                'average_trust': sum(trust_values) / len(trust_values),
                'trust_improvement': trust_values[-1] - trust_values[0] if len(trust_values) > 1 else 0
            }
        
        # Emission metrics
        summary['emissions'] = dict(self.total_emissions)
        
        # Vehicle throughput
        if self.step_metrics:
            duration = self.step_metrics[-1].timestamp - self.step_metrics[0].timestamp
            if duration > 0:
                summary['throughput'] = {
                    'vehicles_per_hour': len(self.completed_trips) * 3600 / duration,
                    'simulation_duration': duration
                }
        
        return summary
    
    def export_step_metrics(self, filename: str = None) -> str:
        """
        Export step-by-step metrics to CSV.
        
        Args:
            filename: Output filename (default: step_metrics.csv)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = "step_metrics.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'vehicle_count', 'average_speed',
                'total_waiting_time', 'stopped_vehicles',
                'v2v_messages', 'v2i_messages', 'average_trust',
                'CO2', 'NOx', 'fuel'
            ])
            
            # Data
            for step in self.step_metrics:
                writer.writerow([
                    step.timestamp,
                    step.vehicle_count,
                    round(step.average_speed, 2),
                    round(step.total_waiting_time, 2),
                    step.stopped_vehicles,
                    step.v2v_messages,
                    step.v2i_messages,
                    round(step.average_trust, 3),
                    round(step.emissions.get('CO2', 0), 2),
                    round(step.emissions.get('NOx', 0), 4),
                    round(step.emissions.get('fuel', 0), 4)
                ])
        
        logger.info(f"Step metrics exported to {filepath}")
        return filepath
    
    def export_trip_metrics(self, filename: str = None) -> str:
        """
        Export trip metrics to CSV.
        
        Args:
            filename: Output filename (default: trip_metrics.csv)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = "trip_metrics.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'vehicle_id', 'vehicle_type', 'departure_time',
                'arrival_time', 'travel_time', 'waiting_time'
            ])
            
            # Data
            for trip in self.completed_trips:
                writer.writerow([
                    trip.vehicle_id,
                    trip.vehicle_type,
                    round(trip.departure_time, 2),
                    round(trip.arrival_time, 2),
                    round(trip.travel_time, 2),
                    round(trip.waiting_time, 2)
                ])
        
        logger.info(f"Trip metrics exported to {filepath}")
        return filepath
    
    def export_summary(self, filename: str = None) -> str:
        """
        Export summary to JSON.
        
        Args:
            filename: Output filename (default: summary.json)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = "summary.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        summary = self.generate_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported to {filepath}")
        return filepath
    
    def export_trust_evolution(self, filename: str = None) -> str:
        """
        Export trust evolution data.
        
        Args:
            filename: Output filename (default: trust_evolution.csv)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = "trust_evolution.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'average_trust'])
            
            for timestamp, trust in self.trust_history:
                writer.writerow([round(timestamp, 2), round(trust, 4)])
        
        logger.info(f"Trust evolution exported to {filepath}")
        return filepath
    
    def export_all(self) -> Dict[str, str]:
        """
        Export all metrics to files.
        
        Returns:
            Dictionary mapping metric type to file path
        """
        return {
            'step_metrics': self.export_step_metrics(),
            'trip_metrics': self.export_trip_metrics(),
            'summary': self.export_summary(),
            'trust_evolution': self.export_trust_evolution()
        }
    
    def get_real_time_metrics(self) -> Dict:
        """
        Get current real-time metrics.
        
        Returns:
            Dictionary with current metrics
        """
        if not self.step_metrics:
            return {}
        
        latest = self.step_metrics[-1]
        
        return {
            'timestamp': latest.timestamp,
            'vehicle_count': latest.vehicle_count,
            'average_speed': latest.average_speed,
            'stopped_vehicles': latest.stopped_vehicles,
            'v2v_messages': self.v2v_message_count,
            'v2i_messages': self.v2i_message_count,
            'average_trust': latest.average_trust
        }
