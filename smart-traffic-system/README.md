# Smart Traffic Management System - NETWORKING Mini Project

## Overview
A **Computer Networks** focused project implementing V2V (Vehicle-to-Vehicle) and V2I (Vehicle-to-Infrastructure) communication protocols in a realistic traffic simulation environment. This mini project demonstrates core networking concepts including protocol design, network performance analysis, security, and wireless communication in vehicular ad-hoc networks (VANETs).

## Project Focus: Computer Networks

This is a **NETWORKING mini project** that covers:

### Primary Networking Topics
1. **V2V/V2I Communication Protocols**: IEEE 802.11p, DSRC, WAVE standards
2. **Vehicular Ad-hoc Networks (VANET)**: Mobile network topology and routing
3. **Network Performance Analysis**: Latency, throughput, packet delivery ratio
4. **Wireless Network Security**: Authentication, trust propagation, attack detection
5. **Protocol Design and Implementation**: Message formats, state machines, QoS
6. **Network Simulation**: SUMO integration for realistic scenarios

### Core Network Components
1. **Protocol Stack Implementation**: 
   - Application Layer: Traffic safety messages
   - Network Layer: Geographic routing protocols
   - MAC Layer: IEEE 802.11p CSMA/CA
   - Physical Layer: 5.9 GHz DSRC
   
2. **V2V Communication System**: 
   - Broadcast and unicast messaging
   - Multi-hop message forwarding
   - Trust-based routing
   - Emergency message dissemination

3. **V2I Infrastructure**: 
   - RSU (Road Side Unit) communication
   - Signal Phase and Timing (SPAT) messages
   - Infrastructure-based routing

4. **Network Performance Monitoring**: 
   - Real-time metrics dashboard
   - Packet delivery ratio tracking
   - Latency measurement
   - Throughput analysis

5. **Security Mechanisms**: 
   - Message authentication
   - Trust score calculation
   - Attack detection (Sybil, DoS)

## Vehicle Types (for Network Simulation)
- Cars, Buses, Trucks (different communication ranges)
- Motorcycles, Bicycles (limited communication)  
- Pedestrians (V2P - Vehicle-to-Pedestrian)
- Emergency vehicles (priority messaging)

## Network Architecture

```
┌─────────────────────────────────────────────────────┐
│           Application Layer                         │
│  (Traffic Safety, Infotainment, Traffic Management)│
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│         Network Layer (IPv6 + Geographic Routing)   │
│  - GPSR (Greedy Perimeter Stateless Routing)       │
│  - Trust-based routing with BFS propagation        │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│         MAC Layer (IEEE 802.11p)                    │
│  - CSMA/CA with priority (EDCA)                    │
│  - Multi-channel operation (CCH + SCH)             │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│         Physical Layer (DSRC 5.9 GHz)               │
│  - Communication range: 300m                        │
│  - Data rates: 3-27 Mbps                           │
└─────────────────────────────────────────────────────┘
```

## Network Performance Metrics

| Metric | Requirement | Achieved |
|--------|-------------|----------|
| Packet Delivery Ratio (Safety) | >95% | 98.4% |
| End-to-End Latency (Safety) | <100ms | 10-15ms |
| Throughput | >1 Mbps | 2.5 Mbps |
| Communication Range | 300m | 300m |
| Network Overhead | <20% | 12% |

## Installation

### Requirements
```
Python 3.8+
SUMO 1.15+ (for traffic simulation)
NumPy, Matplotlib (for analysis)
```

### Setup
```bash
cd smart-traffic-system
pip install -r requirements.txt
```

## Usage

### Running Network Simulation
```bash
# Basic V2V/V2I communication simulation
python main.py

# View network performance dashboard
open frontend/dashboard.html
```

### Network Configuration
Edit `main.py` to configure:
- Vehicle density (affects network load)
- Communication range (default: 300m)
- Message transmission rate
- Trust threshold for routing

## Documentation

### Complete 100-Page Networking Documentation
See `docs/NETWORKING_DOCUMENTATION_100PAGES.md` for:
- **Pages 1-25**: Network architecture, protocols (IEEE 802.11p, WAVE, DSRC)
- **Pages 26-50**: Routing algorithms, trust propagation, protocol implementation
- **Pages 51-75**: Performance analysis, simulation results, case studies
- **Pages 76-100**: Network security, 5G integration, future technologies

Each page includes:
- Detailed content specifications
- Required diagrams and figures
- Code implementations
- Performance analysis
- References to networking standards

## Key Networking Concepts Demonstrated

### 1. Protocol Design
- Message format specification
- Finite State Machine for message validation
- QoS-based priority handling
- Multi-channel coordination

### 2. Wireless Network Performance
- Latency analysis (processing, queuing, transmission, propagation)
- Throughput vs network load
- Collision detection and avoidance
- Channel utilization optimization

### 3. Mobile Network Routing
- Geographic routing (GPSR)
- Trust-based route selection
- Multi-hop forwarding
- Broadcast storm mitigation

### 4. Network Security
- Message authentication
- Trust score propagation
- Sybil attack detection
- Privacy preservation

## Network Simulation Results

Sample results from 225-vehicle simulation:
- **Messages sent**: 699
- **Success rate**: 95.4%
- **Average latency**: 10.9 ms
- **Trust score**: 0.80
- **Active vehicles**: 92

## Authors (Networking Project)
- **Soujanya Patil**: Network architecture and protocol foundations (Pages 1-25)
- **Soujanya Poojari**: Communication protocols and algorithms (Pages 26-50)
- **Anushka**: Network performance and simulation (Pages 51-75)
- **Apoorva**: Network security and advanced topics (Pages 76-100)

## References
- IEEE 802.11p Standard for Wireless Access in Vehicular Environments
- IEEE 1609 WAVE (Wireless Access in Vehicular Environments) Standards
- DSRC (Dedicated Short-Range Communications) Specifications
- VANET (Vehicular Ad-hoc Network) Research Papers

## License
MIT License - Educational/Research Purpose
