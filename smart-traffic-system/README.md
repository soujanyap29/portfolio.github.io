# Smart Traffic Management System

**A Comprehensive Backend Simulation Project**

## Overview

This project delivers a comprehensive, realistic Smart Traffic Management System using real-world road networks and modern communication protocols. The solution employs **SUMO** for traffic simulation, **Python/TraCI** for real-time scenario control, **NS3** for high-fidelity networking, and **OpenStreetMap** for authentic map data. SIoT, V2V, and V2I concepts drive cooperative, adaptive agent behaviors.

## Key Features

- **Real-world Road Networks**: OpenStreetMap integration for authentic city layouts
- **Multi-lane Support**: 4-lane and 6-lane road scenarios
- **Adaptive Traffic Signals**: Congestion-aware signal timing
- **Emergency Vehicle Priority**: Smart green wave and traffic halting
- **V2V Communication**: Vehicle-to-Vehicle messaging (braking, speed, incidents)
- **V2I Communication**: Vehicle-to-Infrastructure interaction (SPaT, routing)
- **SIoT Social Intelligence**: Trust-based cooperative decision making
- **Comprehensive Logging**: SQLite/CSV for all events and metrics
- **Analytics Suite**: Comparative analysis and performance metrics

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **SUMO** | Road traffic simulation; manages movement, signals, lane logic |
| **Python/TraCI** | Real-time control of vehicle behavior and adaptive signals |
| **NS3** | Network simulation for V2V/V2I communication |
| **OpenStreetMap** | Source of authentic city road maps |
| **SQLite/CSV** | Event logging and structured data storage |
| **SIoT/V2V/V2I** | Social intelligence and communication protocols |

## Project Structure

```
smart-traffic-system/
├── sumo/                  # SUMO simulation files
│   ├── networks/          # Road network definitions (.net.xml)
│   ├── routes/            # Vehicle routes (.rou.xml)
│   ├── signals/           # Traffic signal configurations
│   └── scenarios/         # Complete simulation scenarios
├── python/                # Python/TraCI control scripts
│   ├── adaptive_signals.py
│   ├── emergency_priority.py
│   ├── lane_management.py
│   ├── siot_trust.py
│   └── vehicle_controller.py
├── ns3/                   # NS3 networking scripts
│   ├── v2v_protocol.cc
│   ├── v2i_protocol.cc
│   └── network_config.h
├── maps/                  # OpenStreetMap data
│   ├── city_network.osm
│   └── conversion_scripts/
├── database/              # SQLite schemas and queries
│   ├── schema.sql
│   └── queries/
├── logs/                  # Simulation output logs
│   ├── events.csv
│   ├── metrics.csv
│   └── communication.csv
├── analytics/             # Analysis and reporting scripts
│   ├── comparative_analysis.py
│   ├── performance_metrics.py
│   └── visualization.py
├── docs/                  # Documentation (100 pages, 4 authors)
│   ├── pages_01-25_soujanya_patil.md
│   ├── pages_26-50_soujanya_poojari.md
│   ├── pages_51-75_anushka.md
│   └── pages_76-100_apoorva.md
├── diagrams/              # System diagrams and flowcharts
└── configs/               # Configuration files
```

## Installation

### Prerequisites

```bash
# Install SUMO
sudo apt-get install sumo sumo-tools sumo-doc

# Install Python dependencies
pip install traci numpy pandas sqlite3 matplotlib

# Install NS3 (optional for networking simulation)
# Follow NS3 installation guide at https://www.nsnam.org/
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd smart-traffic-system

# Verify SUMO installation
sumo --version

# Set environment variables
export SUMO_HOME=/usr/share/sumo
```

## Quick Start

### 1. Convert OSM Map to SUMO Network

```bash
cd maps/conversion_scripts
python convert_osm_to_sumo.py ../city_network.osm
```

### 2. Run Basic Simulation

```bash
cd sumo/scenarios
sumo-gui -c basic_traffic.sumocfg
```

### 3. Run Adaptive Traffic Control

```bash
cd python
python adaptive_signals.py --scenario basic --duration 3600
```

### 4. Run Complete Simulation with All Features

```bash
python run_simulation.py --config configs/full_scenario.json
```

## Simulation Scenarios

### Baseline (Fixed-time Signals)
- Traditional fixed-cycle traffic lights
- No vehicle communication
- No adaptive behavior

### Actuated Signals
- Vehicle detection-based signal timing
- Basic congestion response
- No V2V/V2I communication

### Smart System (SIoT + V2V + V2I)
- Full adaptive signal control
- Vehicle-to-Vehicle communication
- Vehicle-to-Infrastructure messaging
- Social intelligence and trust management
- Emergency vehicle priority

## Key Modules

### 1. Adaptive Signal Control
```python
# Example: Extend green phase for congested lane
if lane_congestion["lane4"] > threshold:
    traci.trafficlight.setPhaseDuration("junction", extended_time)
```

### 2. Emergency Vehicle Priority
```xml
<!-- Emergency vehicle declaration -->
<vehicle id="ambulance_1" type="emergency" route="route_1" priority="high"/>
```

### 3. V2V Communication
- Real-time position and speed sharing
- Incident alerts
- Lane change intentions
- Braking warnings

### 4. SIoT Trust Management
- Vehicle relationship modeling
- Trust score calculation
- Cooperative decision making

## Output and Metrics

### Event Logs (CSV)
```csv
time,vehicle_id,event,lane,signal_state,priority,result
101.2,amb1,priority_cross,4,green,emergency,traffic_halted
121.7,car45,lane_change,6,green,normal,changed_lane
```

### Performance Metrics
- Average travel time
- Average waiting time
- Average speed
- Number of stops
- Lane occupancy
- Throughput
- Emissions
- Communication statistics

### SQL Analytics
```sql
-- Average speed by vehicle type
SELECT AVG(speed) FROM logs WHERE priority='normal';

-- Emergency vehicle crossings
SELECT COUNT(*) FROM logs 
WHERE event='priority_cross' AND priority='emergency';
```

## Documentation Structure

The project includes 100 pages of comprehensive documentation divided among 4 authors:

- **Pages 1-25** (Soujanya Patil): System architecture, SUMO setup, network design
- **Pages 26-50** (Soujanya Poojari): Python/TraCI implementation, adaptive algorithms
- **Pages 51-75** (Anushka): NS3 networking, V2V/V2I protocols, SIoT layer
- **Pages 76-100** (Apoorva): Analytics, comparative studies, CS course mapping

Each page includes:
- Module description
- Tool and code details
- Code snippets and logs
- CS course mapping (Networks, OS, OOPS, Compiler, DBMS)
- To-do checklist

## CS Course Mappings

| Course | Concepts Applied |
|--------|------------------|
| **Computer Networks** | V2V/V2I protocols, message routing, latency analysis |
| **Operating Systems** | Process scheduling (traffic signals), resource management |
| **OOPS** | Agent classes, inheritance, polymorphism |
| **Compiler Design** | Configuration parsing, state machines |
| **DBMS** | SQLite logging, query optimization, data analytics |

## Comparative Results

The system provides side-by-side comparison of:
1. Fixed-time signals (baseline)
2. Actuated signals
3. Smart system (SIoT + V2V + V2I)

Metrics include travel time reduction, congestion mitigation, and emergency response time.

## Contributing

This is an academic research project. For questions or collaboration:
- Soujanya Patil: [contact info]
- Soujanya Poojari: [contact info]
- Anushka: [contact info]
- Apoorva: [contact info]

## License

This project is for academic and research purposes.

## Citation

If you use this project in your research, please cite:
```
Smart Traffic Management System: A Comprehensive Backend Simulation
Soujanya Patil, Soujanya Poojari, Anushka, Apoorva
[Institution], 2024
```

## References

1. SUMO Documentation: https://sumo.dlr.de/docs/
2. TraCI Documentation: https://sumo.dlr.de/docs/TraCI.html
3. NS3 Documentation: https://www.nsnam.org/documentation/
4. OpenStreetMap: https://www.openstreetmap.org/

---

**Note**: This project is entirely backend-focused. All operations, metrics, and analytics are performed by code and backend scripts, with output as tables, logs, and diagrams. No GUI or dashboard is included—everything is engineered for reproducibility, comparative study, and scientific validity.
