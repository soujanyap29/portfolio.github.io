# Simulation Data Directory

This directory stores simulation output data.

## Structure

```
data/
└── simulation_logs/
    ├── fixed_time/
    │   ├── step_metrics.csv
    │   ├── trip_metrics.csv
    │   ├── summary.json
    │   └── trust_evolution.csv
    ├── actuated/
    │   └── ...
    └── smart_system/
        └── ...
```

## Files Generated

- `step_metrics.csv`: Per-timestep metrics (vehicle count, speed, waiting time)
- `trip_metrics.csv`: Per-vehicle trip information
- `summary.json`: Aggregate statistics
- `trust_evolution.csv`: SIoT trust score over time

## Usage

Data in this directory is generated automatically when running simulations:

```bash
python scripts/main.py --output data/simulation_logs/
```

Use the analyzer to process this data:

```bash
python scripts/analyzer.py --input data/simulation_logs/ --output results/
```
