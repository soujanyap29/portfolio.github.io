# SMART Traffic System - Quick Reference

## Directory Structure
```
SMART_Traffic_System/
├── README.md                     ← Start here! Complete project overview
├── LICENSE                       ← MIT License
├── requirements.txt              ← Python dependencies
├── examples.py                   ← Usage examples (run this to learn)
├── .gitignore                    ← Git ignore rules
│
├── config/                       ← Configuration files
│   ├── simulation_config.json    ← Main simulation parameters
│   ├── signal_config.json        ← Traffic signal timing
│   └── vehicle_config.json       ← Vehicle type definitions
│
├── src/                          ← Python source code
│   ├── main.py                   ← Main entry point
│   ├── config_manager.py         ← Configuration handling
│   ├── v2x_communication.py      ← V2X messaging
│   ├── lane_detection.py         ← Traffic monitoring
│   ├── traffic_signal_control.py ← Adaptive signals
│   ├── emergency_vehicle_priority.py ← Emergency handling
│   ├── performance_metrics.py    ← Data collection
│   └── utils.py                  ← Helper functions
│
├── sumo_files/                   ← SUMO configuration
│   ├── 4lane_network.net.xml     ← Road network
│   ├── traffic_routes.rou.xml    ← Vehicle routes
│   ├── simulation.sumocfg        ← SUMO config
│   ├── traffic_lights.tll.xml    ← Signal programs
│   └── additional_files.add.xml  ← Detectors
│
├── docs/                         ← Documentation
│   ├── 01_SUMO_Network_Creation.md    ← How to create SUMO networks
│   ├── 02_Python_Modules.md           ← Python API documentation
│   ├── 03_Emergency_Priority.md       ← Emergency vehicle system
│   ├── 04_Smart_Enhancements.md       ← Future features
│   └── 05_Troubleshooting.md          ← Common issues & fixes
│
└── results/                      ← Simulation outputs (generated)
    ├── metrics/                  ← CSV performance data
    ├── logs/                     ← Log files
    └── visualizations/           ← Charts and graphs
```

## Quick Start (3 Steps)

### 1. Install SUMO
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools

# Set environment variable
export SUMO_HOME="/usr/share/sumo"
```

### 2. Install Python Dependencies
```bash
cd SMART_Traffic_System
pip install -r requirements.txt
```

### 3. Run Simulation
```bash
# Basic simulation
python src/main.py

# With GUI
python src/main.py --gui

# Or run examples
python examples.py
```

## Key Features Summary

### ✅ Implemented
- **V2X Communication** - Vehicle-to-Everything messaging (300m range, 10Hz)
- **Adaptive Signal Control** - Webster's method + queue-based timing
- **Lane Detection** - Per-lane monitoring with congestion detection
- **Emergency Priority** - Auto-detection, lane clearance, signal preemption
- **Performance Metrics** - Real-time data collection and CSV export

### 🔄 Proposed Enhancements
- Multi-modal sensor fusion (acoustic, camera, RF)
- LED display boards for lane guidance
- ML-based route prediction
- Mobile app integration
- Real-time monitoring dashboard

## Usage Examples

### Example 1: Basic Simulation
```python
from src.main import TrafficSimulation

sim = TrafficSimulation(config_file='config/simulation_config.json')
sim.run(duration=3600)  # 1 hour
```

### Example 2: Emergency Vehicle Test
```python
sim = TrafficSimulation(gui=True)
sim.add_emergency_vehicle(time=300, lane=2, vehicle_type='ambulance')
sim.run(duration=900)
```

### Example 3: Custom Configuration
```python
config = {
    'traffic': {'vehicle_density': 'high'},
    'v2x': {'communication_range': 400}
}

# Save and use
import json
with open('config/custom.json', 'w') as f:
    json.dump(config, f)

sim = TrafficSimulation(config_file='config/custom.json', gui=True)
sim.run()
```

## Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  --config FILE         Configuration file (default: config/simulation_config.json)
  --gui                 Run with SUMO GUI
  --duration SECONDS    Simulation duration (overrides config)
  --output DIR          Output directory for results
  --emergency-test      Run emergency vehicle test scenario
  --test                Run system test and exit
```

## Configuration Quick Reference

### simulation_config.json
```json
{
    "simulation": {
        "total_time": 3600,      // Simulation duration (seconds)
        "step_length": 0.1       // Time step (seconds)
    },
    "traffic": {
        "vehicle_density": "medium"  // low/medium/high/congested
    },
    "v2x": {
        "communication_range": 300   // meters
    }
}
```

## Output Files

After simulation, check `results/` directory:

- **time_series.csv** - Vehicle count, speed, waiting time over time
- **lane_metrics.csv** - Per-lane traffic statistics
- **vehicle_trips.csv** - Individual vehicle trip data
- **emergency_events.csv** - Emergency vehicle logs
- **summary_report.txt** - Overall performance summary

## Emergency Vehicle Priority

### How It Works
1. **Detection** → Emergency vehicle enters network
2. **Alert** → V2X broadcast to nearby vehicles
3. **Clearance** → Vehicles in same lane move aside
4. **Preemption** → Traffic signals turn green
5. **Corridor** → Multiple signals coordinated
6. **Normalization** → Return to normal after passage

### Priority Levels
- **Ambulance** → Priority 1 (Highest)
- **Fire Truck** → Priority 2
- **Police** → Priority 3

## Performance Metrics

### Key Metrics Collected
- **Throughput** - Vehicles per hour
- **Travel Time** - Origin to destination
- **Delay** - Time lost due to signals/congestion
- **Queue Length** - Vehicles waiting at intersections
- **Speed** - Average and per-vehicle
- **Lane Occupancy** - Percentage of lane occupied

## Troubleshooting Quick Fixes

### "SUMO_HOME not set"
```bash
export SUMO_HOME="/usr/share/sumo"
```

### "Network file not found"
```bash
# Must run from SMART_Traffic_System directory
cd SMART_Traffic_System
python src/main.py
```

### "TraCI connection failed"
```bash
# Check SUMO is installed
sumo --version

# Validate network
sumo -c sumo_files/simulation.sumocfg --no-step-log
```

## Documentation Files

| File | Purpose |
|------|---------|
| **README.md** | Complete project overview and workflow |
| **01_SUMO_Network_Creation.md** | Step-by-step SUMO network creation |
| **02_Python_Modules.md** | Python API and module documentation |
| **03_Emergency_Priority.md** | Emergency vehicle system details |
| **04_Smart_Enhancements.md** | Proposed future features |
| **05_Troubleshooting.md** | Common issues and solutions |

## Key Python Classes

| Class | Purpose | File |
|-------|---------|------|
| `TrafficSimulation` | Main controller | main.py |
| `ConfigManager` | Configuration handling | config_manager.py |
| `V2XCommunication` | V2X messaging | v2x_communication.py |
| `LaneDetection` | Traffic monitoring | lane_detection.py |
| `TrafficSignalControl` | Adaptive signals | traffic_signal_control.py |
| `EmergencyVehiclePriority` | Emergency handling | emergency_vehicle_priority.py |
| `PerformanceMetrics` | Data collection | performance_metrics.py |

## SUMO Files Explained

| File | Purpose |
|------|---------|
| **4lane_network.net.xml** | Road network topology with 4 lanes per direction |
| **traffic_routes.rou.xml** | Vehicle routes, flows, and types |
| **simulation.sumocfg** | Main SUMO configuration |
| **traffic_lights.tll.xml** | Signal timing programs |
| **additional_files.add.xml** | Detectors and monitoring devices |

## Next Steps

1. **Read README.md** - Understand complete system
2. **Run examples.py** - See system in action
3. **Read docs/** - Deep dive into specific topics
4. **Modify configs/** - Customize for your needs
5. **Extend src/** - Add new features

## Resources

- **SUMO Documentation**: https://sumo.dlr.de/docs/
- **TraCI Tutorial**: https://sumo.dlr.de/docs/TraCI.html
- **V2X Standards**: IEEE 802.11p, DSRC
- **Traffic Engineering**: Webster's Method, SCOOT, SCATS

## License

MIT License - See LICENSE file for details

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Contact:** See README.md for support information
