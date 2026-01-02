# SMART_ Project: Intelligent Traffic Control and V2X Communication System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Complete Workflow](#complete-workflow)
6. [Installation Guide](#installation-guide)
7. [Emergency Vehicle Priority System](#emergency-vehicle-priority-system)
8. [Smart Enhancements](#smart-enhancements)
9. [Usage Examples](#usage-examples)
10. [Performance Metrics](#performance-metrics)

---

## Project Overview

The **SMART_ Project** (Intelligent Traffic Control and V2X Communication System) is a comprehensive, modular, Python-based system designed to model and manage intelligent urban traffic using advanced communication and control mechanisms. The project integrates SUMO (Simulation of Urban MObility) with intelligent control algorithms to create a realistic traffic simulation environment.

### What Makes This System Smart?

- **Real-time Decision Making**: Adaptive traffic signals that respond to current traffic conditions
- **V2X Communication**: Vehicles communicate with infrastructure and each other
- **Emergency Prioritization**: Automatic detection and prioritization of emergency vehicles
- **Lane-Level Intelligence**: Per-lane traffic monitoring and control
- **Data-Driven Optimization**: Continuous performance monitoring and metrics collection

---

## Key Features

### 1. V2X (Vehicle-to-Everything) Communication
- **V2I (Vehicle-to-Infrastructure)**: Vehicles communicate with traffic lights and road sensors
- **V2V (Vehicle-to-Vehicle)**: Direct communication between vehicles for coordination
- **V2P (Vehicle-to-Pedestrian)**: Safety alerts for pedestrian crossings
- Message broadcasting with configurable frequency and range
- Real-time data exchange for traffic optimization

### 2. Real-Time Lane Detection and Monitoring
- **Per-Lane Traffic Analysis**: Individual monitoring of each lane
- **Congestion Detection**: Automatic identification of traffic buildup
- **Flow Rate Calculation**: Vehicles per hour per lane
- **Density Monitoring**: Vehicle spacing and queue length analysis
- **Occupancy Tracking**: Percentage of lane occupied by vehicles

### 3. SUMO-Based Traffic Simulation
- **Custom 4-Lane Road Networks**: Configurable junction and road layouts
- **Multiple Vehicle Types**: Cars, trucks, buses, motorcycles, emergency vehicles
- **Realistic Driver Behavior**: Lane changing, gap acceptance, speed variation
- **Scalable Simulations**: From single intersections to city-wide networks

### 4. Intelligent Traffic Signal Control
- **Adaptive Signal Timing**: Dynamic green time allocation based on traffic demand
- **Webster's Method**: Optimal cycle time calculation
- **Queue-Based Control**: Signal changes based on queue lengths
- **Coordination**: Multi-junction signal synchronization
- **Time-of-Day Patterns**: Different signal plans for peak and off-peak hours

### 5. Emergency Vehicle Priority Management
- **Automatic Detection**: Using V2X communication and sensor fusion
- **Lane-Level Clearance**: Vehicles instructed to clear specific lanes
- **Green Corridor Creation**: Consecutive signals turn green for emergency route
- **Priority Levels**: Ambulance > Fire Truck > Police
- **Automatic Normalization**: Traffic returns to normal after emergency passes

### 6. Automated Data Collection
- **Performance Metrics**: Delay, queue length, throughput, travel time
- **CSV Export**: Structured data output for analysis
- **Real-Time Logging**: Continuous event tracking
- **Statistical Analysis**: Mean, median, 95th percentile calculations
- **Comparative Reports**: Before/after analysis

### 7. Configuration-Based Management
- **JSON Configuration Files**: Easy parameter modification
- **Simulation Scenarios**: Pre-defined traffic patterns
- **Vehicle Mix Control**: Percentage of each vehicle type
- **Signal Timing Templates**: Reusable timing plans

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SMART Traffic System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   SUMO       │◄──►│   TraCI      │◄──►│   Python     │  │
│  │  Simulation  │    │  Interface   │    │  Controller  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Core Modules                             │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ • V2X Communication                                   │  │
│  │ • Lane Detection & Monitoring                        │  │
│  │ • Traffic Signal Control                             │  │
│  │ • Emergency Vehicle Priority                         │  │
│  │ • Performance Metrics Collection                     │  │
│  │ • Configuration Manager                              │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Results &   │    │  Real-Time   │    │  Reports &   │  │
│  │  Analytics   │    │  Dashboard   │    │  Logs        │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
SMART_Traffic_System/
│
├── README.md                          # This file - Complete project documentation
│
├── sumo_files/                        # SUMO configuration and network files
│   ├── 4lane_network.net.xml         # 4-lane road network definition
│   ├── traffic_routes.rou.xml        # Vehicle routes and flows
│   ├── traffic_lights.tll.xml        # Traffic light programs
│   ├── simulation.sumocfg            # Main SUMO configuration
│   └── additional_files.add.xml      # Additional detectors and outputs
│
├── src/                               # Python source code
│   ├── main.py                       # Main simulation controller
│   ├── v2x_communication.py          # V2X communication module
│   ├── lane_detection.py             # Lane monitoring and detection
│   ├── traffic_signal_control.py     # Adaptive signal control
│   ├── emergency_vehicle_priority.py # Emergency vehicle management
│   ├── performance_metrics.py        # Data collection and analysis
│   ├── config_manager.py             # Configuration file handler
│   └── utils.py                      # Utility functions
│
├── config/                            # Configuration files
│   ├── simulation_config.json        # Simulation parameters
│   ├── signal_config.json            # Traffic signal timing
│   └── vehicle_config.json           # Vehicle type definitions
│
├── docs/                              # Detailed documentation
│   ├── 01_SUMO_Network_Creation.md   # Step-by-step network creation guide
│   ├── 02_Python_Modules.md          # Module documentation
│   ├── 03_Emergency_Priority.md      # Emergency vehicle system details
│   ├── 04_Smart_Enhancements.md      # Future enhancement proposals
│   └── 05_Troubleshooting.md         # Common issues and solutions
│
└── results/                           # Simulation outputs
    ├── metrics/                      # Performance metrics (CSV)
    ├── logs/                         # Simulation logs
    └── visualizations/               # Charts and graphs
```

---

## Complete Workflow

### Phase 1: Creating a 4-Lane Road Network in SUMO

#### Step 1.1: Install SUMO
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc

# macOS
brew install sumo

# Windows: Download from https://sumo.dlr.de/docs/Downloads.php
```

#### Step 1.2: Create Network Nodes
Create a file `nodes.nod.xml`:
```xml
<nodes>
    <!-- Intersection nodes -->
    <node id="J0" x="0.0" y="0.0" type="traffic_light"/>
    <node id="J1" x="500.0" y="0.0" type="traffic_light"/>
    
    <!-- Edge nodes -->
    <node id="N" x="250.0" y="500.0" type="priority"/>
    <node id="S" x="250.0" y="-500.0" type="priority"/>
    <node id="E" x="1000.0" y="0.0" type="priority"/>
    <node id="W" x="-500.0" y="0.0" type="priority"/>
</nodes>
```

#### Step 1.3: Create Network Edges (Roads)
Create a file `edges.edg.xml`:
```xml
<edges>
    <!-- North-South roads (4 lanes each) -->
    <edge id="N_to_J0" from="N" to="J0" numLanes="4" speed="13.89"/>
    <edge id="J0_to_S" from="J0" to="S" numLanes="4" speed="13.89"/>
    
    <!-- East-West roads (4 lanes each) -->
    <edge id="W_to_J0" from="W" to="J0" numLanes="4" speed="13.89"/>
    <edge id="J0_to_J1" from="J0" to="J1" numLanes="4" speed="13.89"/>
    <edge id="J1_to_E" from="J1" to="E" numLanes="4" speed="13.89"/>
    
    <!-- Internal connections -->
    <edge id="J1_to_S" from="J1" to="S" numLanes="4" speed="13.89"/>
</edges>
```

#### Step 1.4: Generate Network
```bash
cd sumo_files
netconvert --node-files=nodes.nod.xml --edge-files=edges.edg.xml --output-file=4lane_network.net.xml
```

#### Step 1.5: Create Traffic Light Program
The system will auto-generate, but you can customize in `traffic_lights.tll.xml`

---

### Phase 2: Verify Required SUMO Files

#### File Checklist:
- ✅ **4lane_network.net.xml** - Road network topology
- ✅ **traffic_routes.rou.xml** - Vehicle routes and departures
- ✅ **simulation.sumocfg** - Main configuration file
- ✅ **traffic_lights.tll.xml** - Signal timing programs
- ✅ **additional_files.add.xml** - Detectors and output definitions

#### Verification Commands:
```bash
# Validate network file
sumo -c simulation.sumocfg --no-step-log --duration-log.disable --no-warnings

# Check for errors
grep -i "error" sumo_output.log
```

---

### Phase 3: Configure Simulation Parameters

#### simulation_config.json
```json
{
    "simulation": {
        "step_length": 0.1,
        "total_time": 3600,
        "gui_enabled": true,
        "real_time_factor": 1.0
    },
    "traffic": {
        "vehicle_density": "medium",
        "peak_hour_multiplier": 1.5,
        "emergency_vehicle_probability": 0.01
    },
    "v2x": {
        "communication_range": 300,
        "message_frequency": 10,
        "broadcast_enabled": true
    }
}
```

---

### Phase 4: Execute the Simulation

#### Basic Execution:
```bash
cd SMART_Traffic_System
python src/main.py --config config/simulation_config.json
```

#### With GUI:
```bash
python src/main.py --config config/simulation_config.json --gui
```

#### Advanced Options:
```bash
python src/main.py \
    --config config/simulation_config.json \
    --duration 7200 \
    --emergency-frequency 0.02 \
    --output results/simulation_001
```

---

### Phase 5: Monitor and Analyze

#### Real-Time Monitoring:
The system provides live console output:
```
[TIME: 00:05:23] Traffic Status:
  Junction J0:
    - Lane 0: 12 vehicles, avg speed: 8.5 m/s
    - Lane 1: 8 vehicles, avg speed: 11.2 m/s
    - Lane 2: 15 vehicles, avg speed: 6.3 m/s (CONGESTED)
    - Lane 3: 5 vehicles, avg speed: 13.1 m/s
  
  Emergency Alert: AMB_001 detected on Lane 2
  Action: Clearing Lane 2, Signal switching to GREEN
```

#### Performance Metrics:
Generated in `results/metrics/`:
- `traffic_flow.csv` - Vehicle counts per lane
- `signal_timing.csv` - Signal phase durations
- `emergency_events.csv` - Emergency vehicle logs
- `performance_summary.csv` - Overall statistics

---

## Emergency Vehicle Priority System

### How It Works

#### 1. Detection Phase
When an emergency vehicle enters the network:
```python
# Automatic detection via V2X
emergency_vehicle = detect_emergency_vehicle()
if emergency_vehicle:
    lane = get_vehicle_lane(emergency_vehicle)
    notify_infrastructure(emergency_vehicle, lane)
```

#### 2. Lane Clearance Phase
Vehicles in the same lane receive clearance instructions:
```
┌─────────────────────────────────────┐
│  BEFORE Emergency Vehicle Approach  │
├─────────────────────────────────────┤
│  Lane 0: [V1] [V2] [V3]            │
│  Lane 1: [V4] [V5] [V6]            │
│  Lane 2: [V7] [V8] [AMB] [V9]      │ ← Emergency lane
│  Lane 3: [V10] [V11]                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  AFTER Lane Clearance Activated     │
├─────────────────────────────────────┤
│  Lane 0: [V1] [V2] [V3] [V7]       │ ← V7 moved
│  Lane 1: [V4] [V5] [V6] [V8]       │ ← V8 moved
│  Lane 2: [AMB] ────────────         │ ← Clear path
│  Lane 3: [V10] [V11] [V9]           │ ← V9 moved
└─────────────────────────────────────┘
```

#### 3. Green Corridor Creation
Consecutive signals along the route turn green:
```
Intersection 1: [GREEN ✓]
      ↓
Intersection 2: [GREEN ✓] (preemptively)
      ↓
Intersection 3: [GREEN ✓] (preemptively)
```

#### 4. Normalization Phase
After the emergency vehicle passes:
- Signals return to adaptive control
- Vehicles resume normal lane positions
- Performance metrics continue collection

### Priority Levels
1. **Ambulance**: Highest priority, immediate response
2. **Fire Truck**: High priority, preemptive clearing
3. **Police**: Medium-high priority, coordinated response

### Implementation Details
See `docs/03_Emergency_Priority.md` for complete technical documentation.

---

## Smart Enhancements

### Implemented Features

#### 1. ✅ V2X Communication
- Real-time vehicle-to-infrastructure messaging
- Broadcast radius: 300 meters
- Update frequency: 10 Hz

#### 2. ✅ Adaptive Signal Control
- Webster's optimal cycle calculation
- Queue-based phase extension
- Time-of-day adjustment

#### 3. ✅ Emergency Vehicle Priority
- Automatic detection and tracking
- Lane-level clearance mechanism
- Multi-junction coordination

### Proposed Enhancements

#### 1. 🔄 Advanced Detection Systems
- **Siren-Based Sensors**: Acoustic detection with direction finding
- **RF Tag Reading**: RFID/DSRC for vehicle identification
- **Camera Recognition**: License plate and visual classification
- **Multi-Modal Fusion**: Combine all detection methods

#### 2. 🔄 Enhanced Lane Clearance
- **Visual Display Boards**: LED signs showing clearance instructions
- **Directional Arrows**: Flashing indicators for lane changes
- **Audio Alerts**: Speaker system for driver guidance
- **Mobile App Integration**: Smartphone notifications

#### 3. 🔄 Predictive Green Corridor
- **Route Prediction**: ML-based trajectory forecasting
- **Preemptive Signal Changes**: Signals ahead of vehicle
- **Dynamic Re-routing**: Real-time path optimization
- **Coordination Algorithm**: City-wide signal synchronization

#### 4. 🔄 Smart Normalization
- **Gradual Return**: Smooth transition back to normal
- **Queue Balancing**: Prioritize backed-up lanes
- **Learning System**: Improve based on past events

#### 5. 🔄 Multi-Emergency Handling
- **Priority Queue**: Manage multiple simultaneous emergencies
- **Conflict Resolution**: Handle opposing directions
- **Resource Allocation**: Optimal signal distribution

#### 6. 🔄 Manual Override System
- **Traffic Control Center**: Remote manual control
- **Emergency Buttons**: On-site activation
- **Authentication**: Secure access control

#### 7. 🔄 Advanced Dashboard
- **3D Visualization**: Live city-wide traffic view
- **Historical Analysis**: Pattern recognition
- **Predictive Analytics**: Forecast congestion
- **Mobile Access**: Remote monitoring capability

---

## Usage Examples

### Example 1: Basic Simulation
```bash
# Run a 1-hour simulation with default settings
python src/main.py --duration 3600
```

### Example 2: Peak Hour Traffic
```bash
# Simulate rush hour with increased vehicle density
python src/main.py \
    --config config/simulation_config.json \
    --density high \
    --duration 7200 \
    --output results/peak_hour_analysis
```

### Example 3: Emergency Vehicle Testing
```bash
# Test emergency vehicle priority system
python src/main.py \
    --emergency-test \
    --emergency-frequency 0.05 \
    --duration 1800
```

### Example 4: Custom Scenario
```python
# Python script for custom scenarios
from src.main import TrafficSimulation

sim = TrafficSimulation(config_file="config/custom_config.json")
sim.add_emergency_vehicle(time=300, lane=2, vehicle_type="ambulance")
sim.add_emergency_vehicle(time=600, lane=1, vehicle_type="fire_truck")
sim.run(duration=1800)
sim.export_results("results/custom_scenario")
```

---

## Performance Metrics

### Collected Metrics

#### Traffic Flow Metrics:
- **Throughput**: Vehicles per hour per lane
- **Average Speed**: Mean speed across all vehicles
- **Travel Time**: Origin to destination duration
- **Delay**: Time lost due to signals and congestion

#### Signal Performance:
- **Cycle Length**: Total time for all phases
- **Green Time Ratio**: Percentage of green per approach
- **Queue Length**: Maximum and average queue sizes
- **Stop Delay**: Time spent stopped at signals

#### Emergency Metrics:
- **Response Time**: Detection to clearance duration
- **Lane Clearance Time**: Time to create clear path
- **Travel Time Reduction**: Compared to normal conditions
- **Success Rate**: Percentage of successful prioritizations

### Sample Output:
```csv
Time,Lane,Vehicle_Count,Avg_Speed,Queue_Length,Delay
00:05:00,0,12,8.5,5,23.4
00:05:00,1,8,11.2,2,12.1
00:05:00,2,15,6.3,8,45.2
00:05:00,3,5,13.1,0,5.3
```

---

## Installation Guide

### Prerequisites:
- Python 3.8 or higher
- SUMO 1.12.0 or higher
- Required Python packages (see requirements.txt)

### Installation Steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/SMART_Traffic_System.git
   cd SMART_Traffic_System
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install SUMO**
   Follow platform-specific instructions at: https://sumo.dlr.de/docs/Installing/

4. **Configure Environment**
   ```bash
   export SUMO_HOME="/usr/share/sumo"  # Linux
   export SUMO_HOME="/Applications/sumo"  # macOS
   ```

5. **Verify Installation**
   ```bash
   python src/main.py --test
   ```

---

## Quick Start

### 3-Minute Quickstart:
```bash
# 1. Navigate to project
cd SMART_Traffic_System

# 2. Run basic simulation
python src/main.py --gui

# 3. View results
ls -l results/metrics/
```

### What You'll See:
- SUMO GUI opens with 4-lane network
- Vehicles start flowing through intersections
- Traffic lights adapt to traffic demand
- Emergency vehicles get priority when they appear
- Real-time console output shows statistics
- Results saved to CSV files

---

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

### Areas for Contribution:
- Additional traffic scenarios
- Machine learning integration
- Mobile app development
- Enhanced visualization
- Documentation improvements

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact & Support

- **Documentation**: See `docs/` folder for detailed guides
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join our community forum
- **Email**: support@smarttraffic.example.com

---

## References

1. SUMO Documentation: https://sumo.dlr.de/docs/
2. V2X Communication Standards: IEEE 802.11p, DSRC
3. Traffic Signal Control: Webster's Method, SCOOT, SCATS
4. Emergency Vehicle Preemption: NEMA TS-2

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Active Development
