# Sample Simulation Results

This directory contains sample output files from the Smart Traffic Management System simulation.

## Directory Structure

```
results/
├── graphs/
│   ├── travel_time_comparison.png
│   ├── waiting_time_comparison.png
│   ├── throughput_comparison.png
│   ├── trust_evolution.png
│   ├── emissions_comparison.png
│   └── speed_time_series.png
├── tables/
│   ├── comparison_table.csv
│   └── metrics_summary.csv
└── reports/
    └── analysis_report.md
```

## Sample Comparative Results

### Strategy Comparison

| Metric | Fixed-Time | Actuated | Smart System (SIoT+V2V+V2I) |
|--------|------------|----------|------------------------------|
| Avg Travel Time (s) | 245.3 | 198.7 | 156.2 |
| Avg Waiting Time (s) | 89.4 | 62.1 | 41.3 |
| Throughput (veh/h) | 820 | 945 | 1120 |
| V2V Messages | 0 | 0 | 245,000 |
| V2I Messages | 0 | 0 | 89,000 |
| Avg Trust Score | N/A | N/A | 0.823 |
| CO2 Emissions (kg) | 1,245 | 1,089 | 892 |

### Key Findings

1. **Travel Time Improvement**: The Smart System reduces average travel time by:
   - 36.3% compared to Fixed-Time signals
   - 21.4% compared to Actuated signals

2. **Waiting Time Reduction**: Vehicles spend significantly less time waiting:
   - 53.8% reduction vs Fixed-Time
   - 33.5% reduction vs Actuated

3. **Throughput Increase**: More vehicles can pass through the network:
   - 36.6% improvement vs Fixed-Time
   - 18.5% improvement vs Actuated

4. **Environmental Benefits**: Reduced emissions due to less idling:
   - 28.4% CO2 reduction vs Fixed-Time
   - 18.1% CO2 reduction vs Actuated

5. **Trust Evolution**: SIoT trust scores increase over time as vehicles:
   - Share accurate information
   - Cooperate in traffic situations
   - Build social relationships

## Expected Graph Descriptions

### Travel Time Comparison
Bar chart showing average travel time for each strategy. Smart System shows the lowest travel time.

### Waiting Time Comparison
Bar chart demonstrating reduced waiting times with intelligent signal control.

### Throughput Comparison
Bar chart illustrating improved network capacity with V2V/V2I communication.

### Trust Evolution
Line graph showing how average trust scores improve over simulation time.

### Emissions Comparison
Multi-panel chart comparing CO2, NOx, and CO emissions across strategies.

### Speed Time Series
Line graph showing average network speed over time for each strategy.

## Running Your Own Analysis

To generate these results with your own simulation:

```bash
# Run simulation with comparison mode
python scripts/main.py --compare --output results/

# Or run individual strategy
python scripts/main.py --config sumo_config/simulation.sumocfg --output results/smart_system/

# Analyze results
python scripts/analyzer.py --input results/ --output results/analysis/
```

## Notes

- Results will vary based on:
  - Network topology
  - Traffic demand
  - Signal timing configurations
  - Random seed used
  - Simulation duration

- For meaningful comparison, always:
  - Use the same random seed across strategies
  - Run for sufficient duration (at least 1 hour recommended)
  - Use identical traffic demand patterns
