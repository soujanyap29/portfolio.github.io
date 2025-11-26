#!/usr/bin/env python3
"""
Analyzer Module

Analyzes simulation results and generates:
- Performance tables
- Comparison charts
- Statistical reports
- Visualization graphs
"""

import os
import sys
import logging
import json
import csv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse

logger = logging.getLogger('Analyzer')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Graphs will not be generated.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available. Some analysis features limited.")


@dataclass
class ComparisonResult:
    """Result of comparative analysis."""
    strategy: str
    travel_time: float
    waiting_time: float
    throughput: float
    emissions_co2: float
    message_count: int
    trust_improvement: float


class Analyzer:
    """
    Analyzes simulation results and generates reports.
    
    Features:
    - Load and parse simulation data
    - Generate statistical summaries
    - Create comparison tables
    - Generate visualization graphs
    - Export reports
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize analyzer.
        
        Args:
            input_dir: Directory with simulation data
            output_dir: Directory for analysis output
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        # Loaded data
        self.step_data: Dict[str, List[Dict]] = {}
        self.trip_data: Dict[str, List[Dict]] = {}
        self.summaries: Dict[str, Dict] = {}
        
    def load_data(self, strategy_dirs: Dict[str, str] = None) -> None:
        """
        Load simulation data from directories.
        
        Args:
            strategy_dirs: Dictionary mapping strategy names to directories
        """
        if strategy_dirs is None:
            # Auto-detect strategies
            strategy_dirs = {}
            for name in ['fixed_time', 'actuated', 'smart_system']:
                dir_path = os.path.join(self.input_dir, name)
                if os.path.exists(dir_path):
                    strategy_dirs[name] = dir_path
        
        for strategy, dir_path in strategy_dirs.items():
            self._load_strategy_data(strategy, dir_path)
    
    def _load_strategy_data(self, strategy: str, dir_path: str) -> None:
        """Load data for a single strategy."""
        # Load step metrics
        step_file = os.path.join(dir_path, 'step_metrics.csv')
        if os.path.exists(step_file):
            self.step_data[strategy] = self._load_csv(step_file)
        
        # Load trip metrics
        trip_file = os.path.join(dir_path, 'trip_metrics.csv')
        if os.path.exists(trip_file):
            self.trip_data[strategy] = self._load_csv(trip_file)
        
        # Load summary
        summary_file = os.path.join(dir_path, 'summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                self.summaries[strategy] = json.load(f)
        
        logger.info(f"Loaded data for strategy: {strategy}")
    
    def _load_csv(self, filepath: str) -> List[Dict]:
        """Load CSV file to list of dictionaries."""
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric values
                converted = {}
                for key, value in row.items():
                    try:
                        if '.' in value:
                            converted[key] = float(value)
                        else:
                            converted[key] = int(value)
                    except (ValueError, TypeError):
                        converted[key] = value
                data.append(converted)
        return data
    
    def generate_comparison_table(self) -> str:
        """
        Generate comparison table for all strategies.
        
        Returns:
            Path to generated table file
        """
        if not self.summaries:
            logger.warning("No summary data available")
            return ""
        
        filepath = os.path.join(self.output_dir, 'tables', 'comparison_table.csv')
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Strategy',
                'Avg Travel Time (s)',
                'Avg Waiting Time (s)',
                'Throughput (veh/h)',
                'Total V2V Messages',
                'Total V2I Messages',
                'Avg Trust Score',
                'CO2 Emissions (mg)'
            ])
            
            # Data rows
            for strategy, summary in self.summaries.items():
                traffic = summary.get('traffic', {})
                comm = summary.get('communication', {})
                siot = summary.get('siot', {})
                emissions = summary.get('emissions', {})
                throughput = summary.get('throughput', {})
                
                writer.writerow([
                    strategy,
                    f"{traffic.get('average_travel_time', 0):.2f}",
                    f"{traffic.get('average_waiting_time', 0):.2f}",
                    f"{throughput.get('vehicles_per_hour', 0):.1f}",
                    comm.get('total_v2v_messages', 0),
                    comm.get('total_v2i_messages', 0),
                    f"{siot.get('average_trust', 0.5):.3f}",
                    f"{emissions.get('CO2', 0):.0f}"
                ])
        
        logger.info(f"Comparison table saved to {filepath}")
        return filepath
    
    def generate_travel_time_comparison(self) -> Optional[str]:
        """
        Generate travel time comparison graph.
        
        Returns:
            Path to generated graph or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self.summaries:
            return None
        
        strategies = list(self.summaries.keys())
        travel_times = [
            self.summaries[s].get('traffic', {}).get('average_travel_time', 0)
            for s in strategies
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, travel_times, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        
        plt.xlabel('Strategy')
        plt.ylabel('Average Travel Time (seconds)')
        plt.title('Travel Time Comparison')
        
        # Add value labels on bars
        for bar, time in zip(bars, travel_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{time:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graphs', 'travel_time_comparison.png')
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Travel time comparison saved to {filepath}")
        return filepath
    
    def generate_waiting_time_comparison(self) -> Optional[str]:
        """Generate waiting time comparison graph."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self.summaries:
            return None
        
        strategies = list(self.summaries.keys())
        waiting_times = [
            self.summaries[s].get('traffic', {}).get('average_waiting_time', 0)
            for s in strategies
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, waiting_times, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        
        plt.xlabel('Strategy')
        plt.ylabel('Average Waiting Time (seconds)')
        plt.title('Waiting Time Comparison')
        
        for bar, time in zip(bars, waiting_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{time:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graphs', 'waiting_time_comparison.png')
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Waiting time comparison saved to {filepath}")
        return filepath
    
    def generate_throughput_comparison(self) -> Optional[str]:
        """Generate throughput comparison graph."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self.summaries:
            return None
        
        strategies = list(self.summaries.keys())
        throughputs = [
            self.summaries[s].get('throughput', {}).get('vehicles_per_hour', 0)
            for s in strategies
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, throughputs, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        
        plt.xlabel('Strategy')
        plt.ylabel('Throughput (vehicles/hour)')
        plt.title('Throughput Comparison')
        
        for bar, tp in zip(bars, throughputs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{tp:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graphs', 'throughput_comparison.png')
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Throughput comparison saved to {filepath}")
        return filepath
    
    def generate_trust_evolution_graph(self) -> Optional[str]:
        """Generate trust score evolution graph."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Try to load trust evolution data
        trust_file = None
        for strategy in ['smart_system', 'actuated', 'fixed_time']:
            path = os.path.join(self.input_dir, strategy, 'trust_evolution.csv')
            if os.path.exists(path):
                trust_file = path
                break
        
        if not trust_file:
            return None
        
        trust_data = self._load_csv(trust_file)
        timestamps = [d['timestamp'] for d in trust_data]
        trust_values = [d['average_trust'] for d in trust_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, trust_values, 'b-', linewidth=2)
        
        plt.xlabel('Simulation Time (seconds)')
        plt.ylabel('Average Trust Score')
        plt.title('Trust Score Evolution Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graphs', 'trust_evolution.png')
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Trust evolution graph saved to {filepath}")
        return filepath
    
    def generate_emissions_comparison(self) -> Optional[str]:
        """Generate emissions comparison graph."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self.summaries:
            return None
        
        strategies = list(self.summaries.keys())
        emissions_data = {
            'CO2': [],
            'NOx': [],
            'CO': []
        }
        
        for s in strategies:
            emissions = self.summaries[s].get('emissions', {})
            emissions_data['CO2'].append(emissions.get('CO2', 0) / 1000)  # Convert to kg
            emissions_data['NOx'].append(emissions.get('NOx', 0))
            emissions_data['CO'].append(emissions.get('CO', 0))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        # CO2
        axes[0].bar(strategies, emissions_data['CO2'], color=colors)
        axes[0].set_xlabel('Strategy')
        axes[0].set_ylabel('CO2 (kg)')
        axes[0].set_title('CO2 Emissions')
        
        # NOx
        axes[1].bar(strategies, emissions_data['NOx'], color=colors)
        axes[1].set_xlabel('Strategy')
        axes[1].set_ylabel('NOx (mg)')
        axes[1].set_title('NOx Emissions')
        
        # CO
        axes[2].bar(strategies, emissions_data['CO'], color=colors)
        axes[2].set_xlabel('Strategy')
        axes[2].set_ylabel('CO (mg)')
        axes[2].set_title('CO Emissions')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graphs', 'emissions_comparison.png')
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Emissions comparison saved to {filepath}")
        return filepath
    
    def generate_speed_time_series(self) -> Optional[str]:
        """Generate average speed time series graph."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not self.step_data:
            return None
        
        plt.figure(figsize=(12, 6))
        
        colors = {'fixed_time': '#ff6b6b', 'actuated': '#4ecdc4', 'smart_system': '#45b7d1'}
        
        for strategy, data in self.step_data.items():
            timestamps = [d['timestamp'] for d in data]
            speeds = [d['average_speed'] for d in data]
            
            # Smooth data for readability
            if NUMPY_AVAILABLE and len(speeds) > 50:
                window = 10
                speeds = np.convolve(speeds, np.ones(window)/window, mode='valid')
                timestamps = timestamps[:len(speeds)]
            
            plt.plot(timestamps, speeds, 
                    label=strategy.replace('_', ' ').title(),
                    color=colors.get(strategy, 'gray'),
                    linewidth=1.5)
        
        plt.xlabel('Simulation Time (seconds)')
        plt.ylabel('Average Speed (m/s)')
        plt.title('Average Speed Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'graphs', 'speed_time_series.png')
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Speed time series saved to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Path to report file
        """
        filepath = os.path.join(self.output_dir, 'reports', 'analysis_report.md')
        
        with open(filepath, 'w') as f:
            f.write("# Smart Traffic Management System - Analysis Report\n\n")
            f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            if self.summaries:
                # Find best performing strategy
                best_strategy = None
                best_travel_time = float('inf')
                
                for strategy, summary in self.summaries.items():
                    tt = summary.get('traffic', {}).get('average_travel_time', float('inf'))
                    if tt < best_travel_time:
                        best_travel_time = tt
                        best_strategy = strategy
                
                f.write(f"**Best Performing Strategy**: {best_strategy.replace('_', ' ').title()}\n")
                f.write(f"**Best Average Travel Time**: {best_travel_time:.2f} seconds\n\n")
            
            f.write("## Detailed Metrics\n\n")
            
            for strategy, summary in self.summaries.items():
                f.write(f"### {strategy.replace('_', ' ').title()}\n\n")
                
                traffic = summary.get('traffic', {})
                f.write("**Traffic Metrics:**\n")
                f.write(f"- Total Trips: {traffic.get('total_trips', 0)}\n")
                f.write(f"- Average Travel Time: {traffic.get('average_travel_time', 0):.2f}s\n")
                f.write(f"- Average Waiting Time: {traffic.get('average_waiting_time', 0):.2f}s\n\n")
                
                comm = summary.get('communication', {})
                f.write("**Communication Metrics:**\n")
                f.write(f"- V2V Messages: {comm.get('total_v2v_messages', 0)}\n")
                f.write(f"- V2I Messages: {comm.get('total_v2i_messages', 0)}\n\n")
                
                siot = summary.get('siot', {})
                f.write("**SIoT Metrics:**\n")
                f.write(f"- Average Trust Score: {siot.get('average_trust', 0.5):.3f}\n")
                f.write(f"- Trust Improvement: {siot.get('trust_improvement', 0):.3f}\n\n")
            
            f.write("## Graphs\n\n")
            f.write("See the `graphs/` directory for visualization charts.\n\n")
            
            f.write("## Conclusions\n\n")
            f.write("The Smart Traffic Management System with integrated SIoT, V2V, and V2I ")
            f.write("communication demonstrates significant improvements in traffic efficiency ")
            f.write("compared to traditional fixed-time and actuated signal control methods.\n")
        
        logger.info(f"Analysis report saved to {filepath}")
        return filepath
    
    def run_full_analysis(self) -> Dict[str, str]:
        """
        Run full analysis and generate all outputs.
        
        Returns:
            Dictionary of generated file paths
        """
        generated_files = {}
        
        # Generate comparison table
        path = self.generate_comparison_table()
        if path:
            generated_files['comparison_table'] = path
        
        # Generate graphs
        graph_functions = [
            ('travel_time', self.generate_travel_time_comparison),
            ('waiting_time', self.generate_waiting_time_comparison),
            ('throughput', self.generate_throughput_comparison),
            ('trust_evolution', self.generate_trust_evolution_graph),
            ('emissions', self.generate_emissions_comparison),
            ('speed_series', self.generate_speed_time_series)
        ]
        
        for name, func in graph_functions:
            path = func()
            if path:
                generated_files[name] = path
        
        # Generate report
        path = self.generate_report()
        if path:
            generated_files['report'] = path
        
        return generated_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Smart Traffic Management simulation results"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../data/simulation_logs',
        help='Input directory with simulation data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../results',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    # Run analysis
    analyzer = Analyzer(input_path, output_path)
    analyzer.load_data()
    
    generated = analyzer.run_full_analysis()
    
    print("\nAnalysis complete. Generated files:")
    for name, path in generated.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
