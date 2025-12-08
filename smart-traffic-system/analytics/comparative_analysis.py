"""
Comparative Analysis Script
Part of Smart Traffic Management System

Compares performance metrics across different scenarios:
1. Baseline (Fixed-time signals)
2. Actuated signals
3. Smart system (SIoT + V2V + V2I)

Course Mappings:
- DBMS: Complex queries, data aggregation, performance analysis
- Operating Systems: Performance metrics, resource utilization
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

class ComparativeAnalyzer:
    """
    Analyzes and compares performance metrics across different traffic management scenarios.
    """
    
    def __init__(self, db_paths):
        """
        Initialize analyzer with database paths for each scenario.
        
        Args:
            db_paths: Dictionary mapping scenario names to database file paths
        """
        self.db_paths = db_paths
        self.results = {}
        self.metrics = [
            'avg_travel_time',
            'avg_waiting_time',
            'avg_speed',
            'total_stops',
            'throughput',
            'avg_co2_emission',
            'avg_fuel_consumption',
            'emergency_response_time'
        ]
        
    def connect_db(self, scenario_name):
        """Connect to database for a specific scenario"""
        db_path = self.db_paths.get(scenario_name)
        if not db_path:
            raise ValueError(f"No database path for scenario: {scenario_name}")
        return sqlite3.connect(db_path)
        
    def calculate_travel_time_metrics(self, scenario_name):
        """Calculate average travel time and related metrics"""
        conn = self.connect_db(scenario_name)
        
        query = """
        SELECT 
            AVG(duration) as avg_travel_time,
            AVG(waiting_time) as avg_waiting_time,
            AVG(time_loss) as avg_time_loss,
            COUNT(*) as total_trips
        FROM trip_info
        WHERE duration > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'avg_travel_time': df['avg_travel_time'].iloc[0],
            'avg_waiting_time': df['avg_waiting_time'].iloc[0],
            'avg_time_loss': df['avg_time_loss'].iloc[0],
            'total_trips': df['total_trips'].iloc[0]
        }
        
    def calculate_speed_metrics(self, scenario_name):
        """Calculate average speed metrics"""
        conn = self.connect_db(scenario_name)
        
        query = """
        SELECT 
            AVG(speed) as avg_speed,
            MIN(speed) as min_speed,
            MAX(speed) as max_speed,
            vehicle_type
        FROM vehicle_logs
        WHERE speed >= 0
        GROUP BY vehicle_type
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'overall_avg_speed': df['avg_speed'].mean(),
            'by_vehicle_type': df.to_dict('records')
        }
        
    def calculate_throughput(self, scenario_name):
        """Calculate network throughput (vehicles/hour)"""
        conn = self.connect_db(scenario_name)
        
        query = """
        SELECT COUNT(DISTINCT vehicle_id) as total_vehicles,
               (MAX(timestamp) - MIN(timestamp)) / 3600.0 as duration_hours
        FROM vehicle_logs
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df['duration_hours'].iloc[0] > 0:
            throughput = df['total_vehicles'].iloc[0] / df['duration_hours'].iloc[0]
        else:
            throughput = 0
            
        return throughput
        
    def calculate_emissions(self, scenario_name):
        """Calculate average emissions"""
        conn = self.connect_db(scenario_name)
        
        query = """
        SELECT 
            AVG(co2_emission) as avg_co2,
            AVG(fuel_consumption) as avg_fuel,
            SUM(co2_emission) as total_co2
        FROM vehicle_logs
        WHERE co2_emission IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'avg_co2_emission': df['avg_co2'].iloc[0],
            'avg_fuel_consumption': df['avg_fuel'].iloc[0],
            'total_co2': df['total_co2'].iloc[0]
        }
        
    def calculate_emergency_response(self, scenario_name):
        """Calculate emergency vehicle response time"""
        conn = self.connect_db(scenario_name)
        
        try:
            query = """
            SELECT 
                AVG(response_time) as avg_response_time,
                MIN(response_time) as min_response_time,
                MAX(response_time) as max_response_time,
                COUNT(*) as emergency_count
            FROM emergency_events
            WHERE event_type = 'COMPLETED'
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return {
                'avg_response_time': df['avg_response_time'].iloc[0] or 0,
                'min_response_time': df['min_response_time'].iloc[0] or 0,
                'max_response_time': df['max_response_time'].iloc[0] or 0,
                'emergency_count': df['emergency_count'].iloc[0] or 0
            }
        except:
            conn.close()
            return {
                'avg_response_time': 0,
                'min_response_time': 0,
                'max_response_time': 0,
                'emergency_count': 0
            }
            
    def calculate_congestion_metrics(self, scenario_name):
        """Calculate congestion-related metrics"""
        conn = self.connect_db(scenario_name)
        
        query = """
        SELECT 
            AVG(occupancy) as avg_occupancy,
            MAX(occupancy) as max_occupancy,
            AVG(vehicle_count) as avg_vehicles_per_lane,
            SUM(CASE WHEN congestion_level = 'HIGH' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as high_congestion_pct
        FROM lane_metrics
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'avg_occupancy': df['avg_occupancy'].iloc[0],
            'max_occupancy': df['max_occupancy'].iloc[0],
            'avg_vehicles_per_lane': df['avg_vehicles_per_lane'].iloc[0],
            'high_congestion_pct': df['high_congestion_pct'].iloc[0]
        }
        
    def calculate_communication_metrics(self, scenario_name):
        """Calculate V2X communication metrics"""
        conn = self.connect_db(scenario_name)
        
        try:
            query = """
            SELECT 
                COUNT(*) as total_messages,
                AVG(latency) as avg_latency,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                COUNT(DISTINCT sender_id) as active_senders
            FROM messages
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return {
                'total_messages': df['total_messages'].iloc[0] or 0,
                'avg_latency_ms': (df['avg_latency'].iloc[0] or 0) * 1000,
                'success_rate': df['success_rate'].iloc[0] or 0,
                'active_senders': df['active_senders'].iloc[0] or 0
            }
        except:
            conn.close()
            return {
                'total_messages': 0,
                'avg_latency_ms': 0,
                'success_rate': 0,
                'active_senders': 0
            }
            
    def analyze_scenario(self, scenario_name):
        """
        Perform complete analysis for a scenario.
        
        Args:
            scenario_name: Name of the scenario to analyze
            
        Returns:
            dict: Complete metrics for the scenario
        """
        print(f"\nAnalyzing scenario: {scenario_name}")
        print("=" * 60)
        
        results = {
            'scenario': scenario_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Travel time metrics
        travel_metrics = self.calculate_travel_time_metrics(scenario_name)
        results.update(travel_metrics)
        
        # Speed metrics
        speed_metrics = self.calculate_speed_metrics(scenario_name)
        results['avg_speed'] = speed_metrics['overall_avg_speed']
        results['speed_by_type'] = speed_metrics['by_vehicle_type']
        
        # Throughput
        results['throughput'] = self.calculate_throughput(scenario_name)
        
        # Emissions
        emission_metrics = self.calculate_emissions(scenario_name)
        results.update(emission_metrics)
        
        # Emergency response
        emergency_metrics = self.calculate_emergency_response(scenario_name)
        results.update(emergency_metrics)
        
        # Congestion
        congestion_metrics = self.calculate_congestion_metrics(scenario_name)
        results.update(congestion_metrics)
        
        # Communication (for smart scenarios)
        if 'smart' in scenario_name.lower():
            comm_metrics = self.calculate_communication_metrics(scenario_name)
            results.update(comm_metrics)
        
        self.results[scenario_name] = results
        
        print(f"Analysis complete for {scenario_name}")
        return results
        
    def compare_all_scenarios(self):
        """Compare all scenarios and generate comparison report"""
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS REPORT")
        print("=" * 80)
        
        for scenario_name in self.db_paths.keys():
            self.analyze_scenario(scenario_name)
        
        # Generate comparison tables
        self.print_comparison_table()
        
        # Calculate improvements
        self.calculate_improvements()
        
        # Export results
        self.export_results()
        
    def print_comparison_table(self):
        """Print formatted comparison table"""
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS COMPARISON")
        print("=" * 80)
        
        scenarios = list(self.results.keys())
        
        # Travel Time
        print("\n1. TRAVEL TIME METRICS")
        print("-" * 80)
        print(f"{'Metric':<30} " + " ".join([f"{s:>15}" for s in scenarios]))
        print("-" * 80)
        
        metrics = ['avg_travel_time', 'avg_waiting_time', 'avg_time_loss']
        for metric in metrics:
            values = [self.results[s].get(metric, 0) for s in scenarios]
            print(f"{metric:<30} " + " ".join([f"{v:>15.2f}" for v in values]))
        
        # Speed
        print("\n2. SPEED METRICS (m/s)")
        print("-" * 80)
        for s in scenarios:
            avg_speed = self.results[s].get('avg_speed', 0)
            print(f"{s:<30} {avg_speed:>15.2f}")
        
        # Throughput
        print("\n3. NETWORK THROUGHPUT (vehicles/hour)")
        print("-" * 80)
        for s in scenarios:
            throughput = self.results[s].get('throughput', 0)
            print(f"{s:<30} {throughput:>15.2f}")
        
        # Emissions
        print("\n4. EMISSIONS")
        print("-" * 80)
        print(f"{'Metric':<30} " + " ".join([f"{s:>15}" for s in scenarios]))
        print("-" * 80)
        metrics = ['avg_co2_emission', 'avg_fuel_consumption']
        for metric in metrics:
            values = [self.results[s].get(metric, 0) for s in scenarios]
            print(f"{metric:<30} " + " ".join([f"{v:>15.4f}" for v in values]))
        
        # Emergency Response
        print("\n5. EMERGENCY RESPONSE TIME (seconds)")
        print("-" * 80)
        for s in scenarios:
            response_time = self.results[s].get('avg_response_time', 0)
            print(f"{s:<30} {response_time:>15.2f}")
        
        # Congestion
        print("\n6. CONGESTION METRICS")
        print("-" * 80)
        print(f"{'Metric':<30} " + " ".join([f"{s:>15}" for s in scenarios]))
        print("-" * 80)
        metrics = ['avg_occupancy', 'high_congestion_pct']
        for metric in metrics:
            values = [self.results[s].get(metric, 0) for s in scenarios]
            print(f"{metric:<30} " + " ".join([f"{v:>15.2f}" for v in values]))
        
    def calculate_improvements(self):
        """Calculate percentage improvements of smart system over baseline"""
        scenarios = list(self.results.keys())
        
        if len(scenarios) < 2:
            return
        
        baseline = scenarios[0]  # Assume first scenario is baseline
        smart = scenarios[-1]    # Assume last scenario is smart system
        
        print("\n" + "=" * 80)
        print(f"IMPROVEMENTS: {smart} vs {baseline}")
        print("=" * 80)
        
        improvements = {}
        
        # Metrics where lower is better
        lower_better = ['avg_travel_time', 'avg_waiting_time', 'avg_time_loss',
                       'avg_co2_emission', 'avg_fuel_consumption', 
                       'avg_response_time', 'avg_occupancy']
        
        # Metrics where higher is better
        higher_better = ['avg_speed', 'throughput']
        
        for metric in lower_better:
            baseline_val = self.results[baseline].get(metric, 0)
            smart_val = self.results[smart].get(metric, 0)
            
            if baseline_val > 0:
                improvement = ((baseline_val - smart_val) / baseline_val) * 100
                improvements[metric] = improvement
                sign = "↓" if improvement > 0 else "↑"
                print(f"{metric:<30} {improvement:>10.2f}% {sign}")
        
        print()
        for metric in higher_better:
            baseline_val = self.results[baseline].get(metric, 0)
            smart_val = self.results[smart].get(metric, 0)
            
            if baseline_val > 0:
                improvement = ((smart_val - baseline_val) / baseline_val) * 100
                improvements[metric] = improvement
                sign = "↑" if improvement > 0 else "↓"
                print(f"{metric:<30} {improvement:>10.2f}% {sign}")
        
        self.improvements = improvements
        
    def export_results(self, output_file='../../logs/comparative_analysis.json'):
        """Export results to JSON file"""
        export_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'scenarios': self.results,
            'improvements': getattr(self, 'improvements', {})
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n\nResults exported to: {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    # Example usage
    db_paths = {
        'Baseline (Fixed-time)': '../../database/baseline.db',
        'Actuated Signals': '../../database/actuated.db',
        'Smart System (SIoT+V2X)': '../../database/smart_system.db'
    }
    
    analyzer = ComparativeAnalyzer(db_paths)
    analyzer.compare_all_scenarios()
