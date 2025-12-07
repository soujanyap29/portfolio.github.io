# Smart Traffic Management System

## Overview
A comprehensive full-stack Smart Traffic Management System featuring realistic multi-agent simulation with SUMO, TraCI, OpenStreetMap integration, V2V/V2I communication, and a live analytics dashboard.

## System Architecture

### Core Components
1. **OSM Integration Module**: Real-world map data preprocessing
2. **Vehicle Agent System**: OOP-based multi-vehicle type modeling
3. **SUMO Simulation Engine**: Traffic simulation with TraCI control
4. **V2V/V2I Communication**: Social IoT trust-based networking
5. **Database Layer**: Event logging and analytics
6. **Live Dashboard**: Real-time visualization and control

## Vehicle Types Supported
- Cars (sedan, SUV, electric)
- Buses (public transit, school, tour)
- Trucks (delivery, freight)
- Motorcycles and E-bikes
- Auto/E-rickshaws
- Bicycles
- Pedestrians
- Animals
- Wheelchairs
- Trams

## CS Subject Mappings
- **OOP**: Modular agent class hierarchy
- **Operating Systems**: Multi-process orchestration, IPC
- **Computer Networks**: V2V/V2I protocols, latency simulation
- **Compiler Design**: Message rule engine, FSM validation
- **DAA/DSA**: Graph routing, trust propagation, event queues
- **DBMS**: Simulation logging, analytics pipelines

## Installation

### Requirements
```
Python 3.8+
SUMO 1.15+
PostgreSQL 13+
Node.js 16+
```

### Setup
```bash
pip install -r requirements.txt
npm install
```

## Usage

### Running Simulation
```bash
python simulation/main.py --scenario city_center --duration 3600
```

### Starting Dashboard
```bash
cd frontend && npm start
```

## Documentation
Complete 100-page documentation available in `/docs` directory, organized by author contributions.

## Authors
- Soujanya Patil (Architecture & System Design)
- Soujanya Poojari (Algorithms & Protocols)
- Anushka (UI/UX & Scenarios)
- Apoorva (Database & Testing)

## License
MIT License
