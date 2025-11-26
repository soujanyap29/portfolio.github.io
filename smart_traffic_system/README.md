# Smart Traffic Management System

A comprehensive, intelligent traffic management system combining **SIoT (Social Internet of Things)**, **V2V (Vehicle-to-Vehicle)** communication, **V2I (Vehicle-to-Infrastructure)** communication, **SUMO simulation**, and **TraCI-based real-time control**.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage Guide](#usage-guide)
8. [Configuration](#configuration)
9. [Metrics and Analysis](#metrics-and-analysis)
10. [Results](#results)
11. [Contributing](#contributing)

## 🎯 Overview

This Smart Traffic Management System is designed to simulate and analyze intelligent traffic control using real-world map data from OpenStreetMap. The system integrates multiple advanced communication paradigms to achieve optimal traffic flow, reduced congestion, and improved emergency vehicle response times.

### Key Technologies

| Technology | Purpose |
|------------|---------|
| **SUMO** | Simulation of Urban Mobility - traffic simulation |
| **TraCI** | Traffic Control Interface - real-time control |
| **OpenStreetMap** | Real-world map data source |
| **V2V** | Vehicle-to-Vehicle communication |
| **V2I** | Vehicle-to-Infrastructure communication |
| **SIoT** | Social Internet of Things behavior |

## ✨ Features

### 1. Map Processing
- Import real-world road networks from OpenStreetMap
- Convert `.osm` files to SUMO network files using `netconvert`
- Generate all required configuration files

### 2. Traffic Light Control
- Configurable signal phases (red/yellow/green)
- Adaptive signal control using TraCI
- Emergency vehicle priority at intersections
- Green wave coordination

### 3. Multi-Modal Vehicle Support
- Cars, buses, trucks, two-wheelers
- Emergency vehicles (ambulance, police, fire truck)
- Realistic vehicle parameters (acceleration, deceleration, size)

### 4. Dynamic Lane Changing
- Traffic-based lane changes
- Speed-based lane changes
- Route-based lane changes

### 5. Traffic Rules Compliance
- Traffic light obedience
- Speed limit enforcement
- Safe lane-changing gaps
- Right-of-way behavior

### 6. Emergency Vehicle Handling
- Priority routing
- Signal preemption
- Automatic yielding by other vehicles

### 7. SIoT (Social Internet of Things)
- Social relationships between vehicles
- Trust-based information sharing
- Cooperative routing
- Alert propagation

### 8. V2V Communication
- Speed and location sharing
- Braking event alerts
- Lane-change intentions
- Collision avoidance

### 9. V2I Communication
- Signal Phase and Timing (SPaT) information
- Recommended speed advisories
- Delay predictions
- Green wave optimization

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SMART TRAFFIC MANAGEMENT SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │  OpenStreetMap  │───▶│   netconvert    │───▶│   SUMO Network Files    │ │
│  │   (.osm file)   │    │                 │    │  (.net.xml, .rou.xml)   │ │
│  └─────────────────┘    └─────────────────┘    └───────────┬─────────────┘ │
│                                                             │               │
│                                                             ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SUMO SIMULATOR                               │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │   Vehicles    │  │Traffic Lights │  │    Road Network       │   │   │
│  │  │   (Cars,      │  │  (Adaptive    │  │   (Lanes, Edges,      │   │   │
│  │  │   Buses,      │  │   Control)    │  │    Junctions)         │   │   │
│  │  │   Emergency)  │  │               │  │                       │   │   │
│  │  └───────┬───────┘  └───────┬───────┘  └───────────────────────┘   │   │
│  └──────────┼──────────────────┼──────────────────────────────────────┘   │
│             │                  │                                           │
│             ▼                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        TraCI INTERFACE                              │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Python Control Scripts                      │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │ │   │
│  │  │  │ V2V Module  │  │ V2I Module  │  │   SIoT Module       │   │ │   │
│  │  │  │             │  │             │  │                     │   │ │   │
│  │  │  │ • Speed     │  │ • SPaT      │  │ • Trust Scores      │   │ │   │
│  │  │  │ • Location  │  │ • Speed Adv │  │ • Social Relations  │   │ │   │
│  │  │  │ • Braking   │  │ • Delay     │  │ • Coop Routing      │   │ │   │
│  │  │  │ • Alerts    │  │ • Green Wave│  │ • Alert Sharing     │   │ │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      METRICS COLLECTOR                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │   │
│  │  │Traffic Metrics  │  │Comm Metrics     │  │ SIoT Metrics        │ │   │
│  │  │• Travel Time    │  │• Message Count  │  │ • Trust Evolution   │ │   │
│  │  │• Waiting Time   │  │• Delivery Rate  │  │ • Cooperation Level │ │   │
│  │  │• Speed          │  │• Latency        │  │                     │ │   │
│  │  │• Queue Length   │  │                 │  │                     │ │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   ANALYSIS & VISUALIZATION                          │   │
│  │              (Graphs, Tables, Reports, Comparisons)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📦 Prerequisites

### Required Software

1. **SUMO (Simulation of Urban Mobility)** - Version 1.15.0 or higher
   ```bash
   # Ubuntu/Debian
   sudo add-apt-repository ppa:sumo/stable
   sudo apt-get update
   sudo apt-get install sumo sumo-tools sumo-doc
   
   # Windows
   # Download from: https://sumo.dlr.de/docs/Downloads.php
   ```

2. **Python** - Version 3.8 or higher
   ```bash
   python --version
   ```

3. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Environment Setup

Set the SUMO_HOME environment variable:

```bash
# Linux/macOS
export SUMO_HOME="/usr/share/sumo"

# Windows
set SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd smart_traffic_system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify SUMO Installation**
   ```bash
   sumo --version
   ```

4. **Prepare Your Map** (Optional)
   - Download your area from [OpenStreetMap](https://www.openstreetmap.org/export)
   - Place the `.osm` file in the `maps/` directory

## 📁 Project Structure

```
smart_traffic_system/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docs/
│   ├── SETUP.md             # Detailed setup instructions
│   ├── ARCHITECTURE.md      # System architecture details
│   └── API_REFERENCE.md     # API documentation
├── maps/
│   └── belagavi_map.osm     # OpenStreetMap data (user provided)
├── sumo_config/
│   ├── belagavi.net.xml     # SUMO network file
│   ├── vehicles.vtype.xml   # Vehicle type definitions
│   ├── routes.rou.xml       # Route definitions
│   ├── additional.add.xml   # Additional configuration
│   └── simulation.sumocfg   # Main SUMO configuration
├── scripts/
│   ├── main.py              # Main simulation runner
│   ├── map_converter.py     # OSM to SUMO converter
│   ├── vehicle_controller.py# Vehicle control module
│   ├── traffic_light_controller.py  # Traffic light control
│   ├── v2v_communication.py # V2V communication module
│   ├── v2i_communication.py # V2I communication module
│   ├── siot_manager.py      # SIoT behavior module
│   ├── emergency_handler.py # Emergency vehicle handler
│   ├── metrics_collector.py # Metrics collection
│   └── analyzer.py          # Analysis and visualization
├── data/
│   └── simulation_logs/     # Simulation output data
└── results/
    ├── graphs/              # Generated graphs
    ├── tables/              # Generated tables
    └── reports/             # Analysis reports
```

## 📖 Usage Guide

### Step 1: Prepare Your Map

```bash
# Convert OSM to SUMO network
python scripts/map_converter.py --input maps/your_map.osm --output sumo_config/
```

### Step 2: Configure Simulation

Edit `sumo_config/simulation.sumocfg` to adjust:
- Simulation duration
- Step length
- Output files

### Step 3: Run Simulation

```bash
# Run with GUI
python scripts/main.py --gui

# Run without GUI (faster)
python scripts/main.py

# Run comparison mode (all three strategies)
python scripts/main.py --compare
```

### Step 4: Analyze Results

```bash
python scripts/analyzer.py --input data/simulation_logs/ --output results/
```

## ⚙️ Configuration

### Vehicle Types

Configure in `sumo_config/vehicles.vtype.xml`:

| Type | Speed | Acceleration | Length |
|------|-------|--------------|--------|
| Car | 50 km/h | 2.6 m/s² | 4.5m |
| Bus | 40 km/h | 1.2 m/s² | 12m |
| Truck | 35 km/h | 1.0 m/s² | 15m |
| Two-wheeler | 60 km/h | 3.0 m/s² | 2m |
| Ambulance | 70 km/h | 3.5 m/s² | 6m |

### Traffic Light Phases

Configure in `sumo_config/additional.add.xml`:

- Green Phase: 30-60 seconds
- Yellow Phase: 3-5 seconds
- Red Phase: Variable

## 📊 Metrics and Analysis

### Traffic Metrics
- **Travel Time**: Average time to complete routes
- **Waiting Time**: Time spent waiting at signals
- **Average Speed**: Mean vehicle speed
- **Throughput**: Vehicles per hour
- **Queue Length**: Vehicles waiting at intersections
- **Stops**: Number of complete stops
- **Emissions**: CO2, NOx, PM emissions

### Communication Metrics
- **Message Count**: Total V2V/V2I messages
- **Delivery Rate**: Successful message delivery percentage
- **Latency**: Message transmission delay

### SIoT Metrics
- **Trust Score Evolution**: How trust changes over time
- **Cooperation Level**: Degree of cooperative behavior

### Comparative Evaluation

The system compares three strategies:
1. **Fixed-time Signals**: Traditional fixed timing
2. **Actuated Signals**: Sensor-based adaptive
3. **Smart System (SIoT+V2V+V2I)**: Full intelligent system

## 📈 Results

Results are generated in the `results/` directory:

```
results/
├── graphs/
│   ├── travel_time_comparison.png
│   ├── waiting_time_comparison.png
│   ├── throughput_comparison.png
│   ├── trust_evolution.png
│   └── emissions_comparison.png
├── tables/
│   ├── metrics_summary.csv
│   └── comparative_analysis.csv
└── reports/
    ├── simulation_report.md
    └── performance_analysis.pdf
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- SUMO Development Team
- OpenStreetMap Contributors
- TraCI Python Library Developers
