"""
Performance Metrics Module for SMART Traffic System
Collects, analyzes, and exports simulation performance data
"""

import csv
import os
from datetime import datetime
from collections import defaultdict
import statistics


class PerformanceMetrics:
    """
    Collects and analyzes simulation performance metrics.
    
    Metrics collected:
    - Traffic flow (vehicles/hour)
    - Average speeds
    - Travel times
    - Queue lengths
    - Signal timing performance
    - Emergency vehicle response times
    """
    
    def __init__(self, output_dir='results/metrics'):
        """
        Initialize performance metrics collector.
        
        Args:
            output_dir (str): Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.time_series_data = defaultdict(list)
        self.vehicle_data = defaultdict(dict)
        self.lane_metrics = defaultdict(list)
        self.emergency_events = []
        
        # Statistics
        self.total_vehicles = 0
        self.completed_trips = 0
        
        print(f"✓ Performance Metrics initialized (output: {output_dir})")
    
    def collect_data(self, vehicle_ids, lane_data, current_time):
        """
        Collect performance data for current time step.
        
        Args:
            vehicle_ids (list): Current vehicle IDs
            lane_data (dict): Lane analysis data
            current_time (float): Current simulation time
        """
        import traci
        
        # Record vehicle count
        self.time_series_data['vehicle_count'].append({
            'time': current_time,
            'count': len(vehicle_ids)
        })
        
        # Collect per-vehicle data
        total_speed = 0
        total_waiting_time = 0
        
        for veh_id in vehicle_ids:
            try:
                if veh_id not in self.vehicle_data:
                    self.vehicle_data[veh_id] = {
                        'depart_time': current_time,
                        'arrival_time': None,
                        'total_waiting_time': 0,
                        'travel_distance': 0
                    }
                
                speed = traci.vehicle.getSpeed(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                distance = traci.vehicle.getDistance(veh_id)
                
                total_speed += speed
                total_waiting_time += waiting_time
                
                self.vehicle_data[veh_id]['total_waiting_time'] = waiting_time
                self.vehicle_data[veh_id]['travel_distance'] = distance
            
            except:
                pass
        
        # Calculate averages
        if len(vehicle_ids) > 0:
            avg_speed = total_speed / len(vehicle_ids)
            avg_waiting = total_waiting_time / len(vehicle_ids)
            
            self.time_series_data['avg_speed'].append({
                'time': current_time,
                'speed': avg_speed
            })
            
            self.time_series_data['avg_waiting_time'].append({
                'time': current_time,
                'waiting_time': avg_waiting
            })
        
        # Collect lane metrics
        for lane_id, data in lane_data.items():
            self.lane_metrics[lane_id].append({
                'time': current_time,
                'vehicle_count': data['vehicle_count'],
                'avg_speed': data['avg_speed'],
                'queue_length': data['queue_length'],
                'congested': data['congested']
            })
    
    def record_emergency_event(self, vehicle_id, event_type, details=None):
        """
        Record emergency vehicle event.
        
        Args:
            vehicle_id (str): Emergency vehicle ID
            event_type (str): Event type (detected, priority_activated, cleared)
            details (dict): Additional details
        """
        import traci
        
        event = {
            'time': traci.simulation.getTime(),
            'vehicle_id': vehicle_id,
            'event_type': event_type,
            'details': details or {}
        }
        
        self.emergency_events.append(event)
    
    def calculate_throughput(self):
        """
        Calculate traffic throughput (vehicles/hour).
        
        Returns:
            float: Throughput in vehicles per hour
        """
        if not self.time_series_data['vehicle_count']:
            return 0.0
        
        # Count unique vehicles
        unique_vehicles = len(self.vehicle_data)
        
        # Get simulation duration in hours
        if self.time_series_data['vehicle_count']:
            duration_seconds = self.time_series_data['vehicle_count'][-1]['time']
            duration_hours = duration_seconds / 3600.0
            
            if duration_hours > 0:
                return unique_vehicles / duration_hours
        
        return 0.0
    
    def calculate_average_travel_time(self):
        """
        Calculate average travel time for completed trips.
        
        Returns:
            float: Average travel time in seconds
        """
        completed_times = []
        
        for veh_id, data in self.vehicle_data.items():
            if data['arrival_time'] is not None:
                travel_time = data['arrival_time'] - data['depart_time']
                completed_times.append(travel_time)
        
        if completed_times:
            return statistics.mean(completed_times)
        return 0.0
    
    def calculate_average_delay(self):
        """
        Calculate average delay per vehicle.
        
        Returns:
            float: Average delay in seconds
        """
        delays = []
        
        for veh_id, data in self.vehicle_data.items():
            delays.append(data['total_waiting_time'])
        
        if delays:
            return statistics.mean(delays)
        return 0.0
    
    def export_csv(self, output_dir=None):
        """
        Export all metrics to CSV files.
        
        Args:
            output_dir (str): Optional output directory override
        """
        output_path = output_dir or self.output_dir
        os.makedirs(output_path, exist_ok=True)
        
        # Export time series data
        self._export_time_series(output_path)
        
        # Export lane metrics
        self._export_lane_metrics(output_path)
        
        # Export vehicle data
        self._export_vehicle_data(output_path)
        
        # Export emergency events
        self._export_emergency_events(output_path)
        
        print(f"✓ Metrics exported to {output_path}")
    
    def _export_time_series(self, output_dir):
        """Export time series data to CSV."""
        filename = os.path.join(output_dir, 'time_series.csv')
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Vehicle_Count', 'Avg_Speed', 'Avg_Waiting_Time'])
            
            # Merge data by time
            times = sorted(set([d['time'] for d in self.time_series_data['vehicle_count']]))
            
            for time in times:
                # Find data for this time
                veh_count = next((d['count'] for d in self.time_series_data['vehicle_count'] 
                                 if d['time'] == time), 0)
                
                avg_speed = next((d['speed'] for d in self.time_series_data.get('avg_speed', [])
                                 if d['time'] == time), 0)
                
                avg_wait = next((d['waiting_time'] for d in self.time_series_data.get('avg_waiting_time', [])
                                if d['time'] == time), 0)
                
                writer.writerow([time, veh_count, avg_speed, avg_wait])
    
    def _export_lane_metrics(self, output_dir):
        """Export lane-specific metrics to CSV."""
        filename = os.path.join(output_dir, 'lane_metrics.csv')
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Lane_ID', 'Vehicle_Count', 'Avg_Speed', 
                           'Queue_Length', 'Congested'])
            
            for lane_id, data_points in self.lane_metrics.items():
                for data in data_points:
                    writer.writerow([
                        data['time'],
                        lane_id,
                        data['vehicle_count'],
                        data['avg_speed'],
                        data['queue_length'],
                        1 if data['congested'] else 0
                    ])
    
    def _export_vehicle_data(self, output_dir):
        """Export vehicle trip data to CSV."""
        filename = os.path.join(output_dir, 'vehicle_trips.csv')
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle_ID', 'Depart_Time', 'Arrival_Time', 
                           'Travel_Time', 'Waiting_Time', 'Distance'])
            
            for veh_id, data in self.vehicle_data.items():
                travel_time = 0
                if data['arrival_time']:
                    travel_time = data['arrival_time'] - data['depart_time']
                
                writer.writerow([
                    veh_id,
                    data['depart_time'],
                    data['arrival_time'] or 'N/A',
                    travel_time,
                    data['total_waiting_time'],
                    data['travel_distance']
                ])
    
    def _export_emergency_events(self, output_dir):
        """Export emergency vehicle events to CSV."""
        filename = os.path.join(output_dir, 'emergency_events.csv')
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Vehicle_ID', 'Event_Type', 'Details'])
            
            for event in self.emergency_events:
                writer.writerow([
                    event['time'],
                    event['vehicle_id'],
                    event['event_type'],
                    str(event['details'])
                ])
    
    def generate_summary_report(self, output_dir=None):
        """
        Generate summary report with key statistics.
        
        Args:
            output_dir (str): Output directory
        """
        output_path = output_dir or self.output_dir
        filename = os.path.join(output_path, 'summary_report.txt')
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SMART Traffic System - Performance Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Traffic Flow Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Vehicles: {len(self.vehicle_data)}\n")
            f.write(f"Throughput: {self.calculate_throughput():.2f} veh/hour\n")
            f.write(f"Average Travel Time: {self.calculate_average_travel_time():.2f} seconds\n")
            f.write(f"Average Delay: {self.calculate_average_delay():.2f} seconds\n\n")
            
            f.write("Emergency Vehicle Events:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Emergency Events: {len(self.emergency_events)}\n\n")
            
            if self.lane_metrics:
                f.write("Lane Performance:\n")
                f.write("-" * 40 + "\n")
                for lane_id in list(self.lane_metrics.keys())[:5]:
                    data_points = self.lane_metrics[lane_id]
                    if data_points:
                        avg_count = statistics.mean([d['vehicle_count'] for d in data_points])
                        avg_speed = statistics.mean([d['avg_speed'] for d in data_points])
                        f.write(f"{lane_id}: avg {avg_count:.1f} vehicles, {avg_speed:.1f} m/s\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"✓ Summary report generated: {filename}")
