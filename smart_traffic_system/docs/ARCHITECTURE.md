# System Architecture

This document provides detailed architectural information about the Smart Traffic Management System.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Descriptions](#component-descriptions)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Communication Protocols](#communication-protocols)
5. [Module Interactions](#module-interactions)

---

## High-Level Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         SMART TRAFFIC MANAGEMENT SYSTEM                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        INPUT LAYER                                       │ ║
║  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │ ║
║  │  │ OpenStreetMap │  │Configuration  │  │  User Parameters          │   │ ║
║  │  │   Map Data    │  │   Files       │  │  (Duration, Scenarios)    │   │ ║
║  │  └───────┬───────┘  └───────┬───────┘  └─────────────┬─────────────┘   │ ║
║  └──────────┼──────────────────┼────────────────────────┼─────────────────┘ ║
║             │                  │                        │                    ║
║             ▼                  ▼                        ▼                    ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     PROCESSING LAYER                                     │ ║
║  │                                                                          │ ║
║  │  ┌──────────────────────────────────────────────────────────────────┐   │ ║
║  │  │                    SUMO SIMULATOR ENGINE                          │   │ ║
║  │  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐   │   │ ║
║  │  │  │   Network    │ │   Vehicle    │ │  Traffic Light         │   │   │ ║
║  │  │  │   Manager    │ │   Manager    │ │  Manager               │   │   │ ║
║  │  │  └──────────────┘ └──────────────┘ └────────────────────────┘   │   │ ║
║  │  └──────────────────────────────────────────────────────────────────┘   │ ║
║  │                              ▲                                           │ ║
║  │                              │ TraCI Interface                           │ ║
║  │                              ▼                                           │ ║
║  │  ┌──────────────────────────────────────────────────────────────────┐   │ ║
║  │  │                   INTELLIGENT CONTROL LAYER                       │   │ ║
║  │  │                                                                   │   │ ║
║  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │   │ ║
║  │  │  │    V2V     │ │    V2I     │ │   SIoT     │ │  Emergency   │  │   │ ║
║  │  │  │  Module    │ │  Module    │ │  Module    │ │   Handler    │  │   │ ║
║  │  │  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │   │ ║
║  │  │                                                                   │   │ ║
║  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────────────────┐   │   │ ║
║  │  │  │  Vehicle   │ │ Traffic    │ │  Lane Change              │   │   │ ║
║  │  │  │ Controller │ │ Light Ctrl │ │  Controller               │   │   │ ║
║  │  │  └────────────┘ └────────────┘ └────────────────────────────┘   │   │ ║
║  │  └──────────────────────────────────────────────────────────────────┘   │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║             │                                                                 ║
║             ▼                                                                 ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                       OUTPUT LAYER                                       │ ║
║  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │ ║
║  │  │   Metrics     │  │  Simulation   │  │     Analysis &            │   │ ║
║  │  │   Collector   │  │    Logs       │  │   Visualization           │   │ ║
║  │  └───────────────┘  └───────────────┘  └───────────────────────────┘   │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Descriptions

### 1. Input Layer

#### OpenStreetMap Data
- **Format**: `.osm` (XML-based)
- **Content**: Road network, intersections, traffic signals, lane information
- **Processing**: Converted to SUMO format using `netconvert`

#### Configuration Files
- **Network (`.net.xml`)**: Road topology, lanes, junctions
- **Routes (`.rou.xml`)**: Vehicle paths and flows
- **Vehicle Types (`.vtype.xml`)**: Vehicle characteristics
- **Additional (`.add.xml`)**: Traffic lights, detectors

### 2. Processing Layer

#### SUMO Simulator Engine

| Component | Responsibility |
|-----------|---------------|
| Network Manager | Handle road network topology |
| Vehicle Manager | Track and update vehicle states |
| Traffic Light Manager | Control signal phases |

#### TraCI Interface
- Real-time bidirectional communication
- Vehicle state queries and modifications
- Traffic light control
- Simulation step control

### 3. Intelligent Control Layer

#### V2V Communication Module
```python
class V2VCommunication:
    """
    Handles vehicle-to-vehicle communication
    
    Features:
    - Broadcast speed and position
    - Emergency brake alerts
    - Lane change intentions
    - Collision warnings
    """
```

#### V2I Communication Module
```python
class V2ICommunication:
    """
    Handles vehicle-to-infrastructure communication
    
    Features:
    - Signal Phase and Timing (SPaT)
    - Green wave recommendations
    - Queue length information
    - Delay predictions
    """
```

#### SIoT Module
```python
class SIoTManager:
    """
    Manages Social Internet of Things behavior
    
    Features:
    - Trust score management
    - Social relationship formation
    - Cooperative decision making
    - Information verification
    """
```

#### Emergency Handler
```python
class EmergencyHandler:
    """
    Handles emergency vehicle operations
    
    Features:
    - Priority routing
    - Signal preemption
    - Path clearing
    - Recovery after passage
    """
```

### 4. Output Layer

#### Metrics Collector
- Traffic metrics (travel time, waiting time, speed)
- Communication metrics (message count, latency)
- SIoT metrics (trust scores, cooperation levels)

#### Analysis Engine
- Statistical analysis
- Comparative evaluation
- Visualization generation

---

## Data Flow Diagrams

### Main Simulation Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIMULATION STEP FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                 │
│  │   START     │──────│ Initialize  │──────│ Load Config │                 │
│  │   STEP      │      │   TraCI     │      │   Files     │                 │
│  └─────────────┘      └─────────────┘      └──────┬──────┘                 │
│                                                    │                        │
│                                                    ▼                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    MAIN SIMULATION LOOP                                │ │
│  │                                                                        │ │
│  │   ┌─────────────┐                                                     │ │
│  │   │  Get All    │                                                     │ │
│  │   │  Vehicles   │◄─────────────────────────────────────────────┐     │ │
│  │   └──────┬──────┘                                               │     │ │
│  │          │                                                       │     │ │
│  │          ▼                                                       │     │ │
│  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │     │ │
│  │   │  Process     │   │   Process    │   │  Process     │       │     │ │
│  │   │  V2V Comms   │──▶│   V2I Comms  │──▶│  SIoT Logic  │       │     │ │
│  │   └──────────────┘   └──────────────┘   └──────┬───────┘       │     │ │
│  │                                                 │               │     │ │
│  │                                                 ▼               │     │ │
│  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │     │ │
│  │   │   Handle     │◀──│   Update     │◀──│   Control    │       │     │ │
│  │   │  Emergency   │   │   Traffic    │   │   Vehicles   │       │     │ │
│  │   │   Vehicles   │   │   Lights     │   │              │       │     │ │
│  │   └──────┬───────┘   └──────────────┘   └──────────────┘       │     │ │
│  │          │                                                       │     │ │
│  │          ▼                                                       │     │ │
│  │   ┌──────────────┐   ┌──────────────┐                           │     │ │
│  │   │   Collect    │──▶│  Simulation  │───────────────────────────┘     │ │
│  │   │   Metrics    │   │    Step      │                                 │ │
│  │   └──────────────┘   └──────────────┘                                 │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                 │
│  │  Generate   │◀─────│   Close     │◀─────│    END      │                 │
│  │  Reports    │      │   TraCI     │      │  SIMULATION │                 │
│  └─────────────┘      └─────────────┘      └─────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### V2V Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      V2V COMMUNICATION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                           ┌─────────────────┐          │
│  │    Vehicle A    │                           │    Vehicle B    │          │
│  │                 │                           │                 │          │
│  │  Position: X1,Y1│                           │  Position: X2,Y2│          │
│  │  Speed: V1      │                           │  Speed: V2      │          │
│  │  Lane: L1       │                           │  Lane: L2       │          │
│  └────────┬────────┘                           └────────┬────────┘          │
│           │                                             │                    │
│           │  ┌────────────────────────────────────┐    │                    │
│           │  │          V2V MESSAGE               │    │                    │
│           │  │  ┌──────────────────────────────┐ │    │                    │
│           │  │  │ sender_id: "vehicle_A"       │ │    │                    │
│           ├─▶│  │ position: (X1, Y1)           │ │◀───┤                    │
│           │  │  │ speed: V1                    │ │    │                    │
│           │  │  │ lane: L1                     │ │    │                    │
│           │  │  │ braking: false               │ │    │                    │
│           │  │  │ lane_change_intent: "none"   │ │    │                    │
│           │  │  │ timestamp: T                 │ │    │                    │
│           │  │  └──────────────────────────────┘ │    │                    │
│           │  └────────────────────────────────────┘    │                    │
│           │                                             │                    │
│           ▼                                             ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    V2V PROCESSING ENGINE                             │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │   │
│  │  │ Collision      │  │ Speed          │  │ Lane Change            │ │   │
│  │  │ Detection      │  │ Harmonization  │  │ Coordination           │ │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### V2I Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      V2I COMMUNICATION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐│
│  │         VEHICLES                 │    │      INFRASTRUCTURE            ││
│  │                                  │    │                                 ││
│  │  ┌─────────┐  ┌─────────┐       │    │  ┌─────────────────────────┐   ││
│  │  │Vehicle 1│  │Vehicle 2│  ...  │    │  │    Traffic Light        │   ││
│  │  └────┬────┘  └────┬────┘       │    │  │    Controller           │   ││
│  │       │            │            │    │  └────────────┬────────────┘   ││
│  │       │            │            │    │               │                 ││
│  │       └──────┬─────┘            │    │  ┌────────────▼────────────┐   ││
│  └──────────────┼──────────────────┘    │  │    Roadside Unit        │   ││
│                 │                        │  │       (RSU)             │   ││
│                 │                        │  └────────────┬────────────┘   ││
│                 │                        └───────────────┼────────────────┘│
│                 │                                        │                  │
│                 │        ┌──────────────────────┐        │                  │
│                 │        │    V2I MESSAGES       │        │                  │
│                 │        │                       │        │                  │
│                 │        │  Vehicle → RSU:       │        │                  │
│                 ├───────▶│  - Position          │◀───────┤                  │
│                 │        │  - Speed             │        │                  │
│                 │        │  - Destination       │        │                  │
│                 │        │                       │        │                  │
│                 │        │  RSU → Vehicle:       │        │                  │
│                 │◀───────│  - Signal Phase      │────────┤                  │
│                 │        │  - Time to Green     │        │                  │
│                 │        │  - Recommended Speed │        │                  │
│                 │        │  - Queue Length      │        │                  │
│                 │        └──────────────────────┘        │                  │
│                 │                                        │                  │
│                 ▼                                        ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     OPTIMIZATION ENGINE                               │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │  │
│  │  │ Green Wave     │  │ Speed Advisory │  │ Queue Management       │  │  │
│  │  │ Optimization   │  │ Calculation    │  │                        │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SIoT Relationship Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIoT RELATIONSHIP MODEL                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌─────────────────────────────┐                          │
│                    │       TRUST NETWORK         │                          │
│                    └─────────────────────────────┘                          │
│                                                                              │
│      ┌─────────┐                                       ┌─────────┐          │
│      │ Vehicle │◄──────── Trust: 0.85 ────────────────▶│ Vehicle │          │
│      │    A    │                                       │    B    │          │
│      └────┬────┘                                       └────┬────┘          │
│           │                                                  │               │
│           │ Trust: 0.72                          Trust: 0.91 │               │
│           │                                                  │               │
│           ▼                                                  ▼               │
│      ┌─────────┐         Trust: 0.65              ┌─────────┐              │
│      │ Vehicle │◄─────────────────────────────────│ Vehicle │              │
│      │    C    │                                  │    D    │              │
│      └────┬────┘                                  └────┬────┘              │
│           │                                            │                    │
│           │              ┌────────────────┐            │                    │
│           └─────────────▶│   Traffic      │◀───────────┘                    │
│                          │   Light (RSU)  │                                 │
│                          │   Trust: 1.0   │                                 │
│                          └────────────────┘                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    TRUST SCORE CALCULATION                            │   │
│  │                                                                       │   │
│  │  Trust(A→B) = α × Accuracy + β × Timeliness + γ × Consistency        │   │
│  │                                                                       │   │
│  │  Where:                                                               │   │
│  │    - Accuracy: How accurate past information was                      │   │
│  │    - Timeliness: How timely information was shared                    │   │
│  │    - Consistency: How consistent the vehicle's behavior is            │   │
│  │    - α, β, γ: Weighting factors (sum = 1)                            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    RELATIONSHIP TYPES                                 │   │
│  │                                                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │   │
│  │  │  Co-Location    │  │  Co-Movement    │  │  Social Object      │   │   │
│  │  │  Relationship   │  │  Relationship   │  │  Relationship       │   │   │
│  │  │  (Same area)    │  │  (Same route)   │  │  (Same owner/type)  │   │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Communication Protocols

### Message Types

| Type | Direction | Content | Frequency |
|------|-----------|---------|-----------|
| Beacon | V2V | Position, Speed, Lane | 10 Hz |
| Alert | V2V | Braking, Hazard | Event-based |
| SPaT | V2I | Signal Phase/Timing | 10 Hz |
| Advisory | V2I | Speed, Route | 1 Hz |
| Trust Update | SIoT | Trust Score | Event-based |

### Message Structure

```python
# V2V Message
v2v_message = {
    "type": "beacon",
    "sender_id": "veh_001",
    "timestamp": 1234567890.123,
    "position": {"x": 100.5, "y": 200.3},
    "speed": 13.5,
    "acceleration": 0.5,
    "lane_id": "edge_1_0",
    "heading": 45.0,
    "braking": False,
    "lane_change_intent": None,
    "trust_score": 0.85
}

# V2I Message (Infrastructure to Vehicle)
v2i_message = {
    "type": "spat",
    "signal_id": "tl_001",
    "timestamp": 1234567890.123,
    "current_phase": "green",
    "time_to_change": 15.5,
    "recommended_speed": 40.0,
    "queue_length": 5,
    "congestion_level": "low"
}

# SIoT Trust Update
siot_message = {
    "type": "trust_update",
    "from_id": "veh_001",
    "to_id": "veh_002",
    "timestamp": 1234567890.123,
    "trust_score": 0.88,
    "reason": "accurate_information"
}
```

---

## Module Interactions

### Sequence Diagram: Emergency Vehicle Handling

```
┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌─────────┐     ┌────────┐
│Emergency│     │   V2I   │     │Traffic Light│     │   V2V   │     │Regular │
│ Vehicle │     │ Module  │     │ Controller  │     │ Module  │     │Vehicles│
└────┬────┘     └────┬────┘     └──────┬──────┘     └────┬────┘     └────┬───┘
     │               │                 │                 │               │
     │  Announce     │                 │                 │               │
     │  Emergency    │                 │                 │               │
     │──────────────▶│                 │                 │               │
     │               │                 │                 │               │
     │               │  Request        │                 │               │
     │               │  Priority       │                 │               │
     │               │────────────────▶│                 │               │
     │               │                 │                 │               │
     │               │                 │  Set Green      │               │
     │               │                 │────────────┐    │               │
     │               │                 │            │    │               │
     │               │                 │◀───────────┘    │               │
     │               │                 │                 │               │
     │               │  Confirm        │                 │               │
     │               │◀────────────────│                 │               │
     │               │                 │                 │               │
     │               │                 │  Broadcast      │               │
     │               │                 │  Alert          │               │
     │               │────────────────────────────────────│──────────────▶│
     │               │                 │                 │               │
     │               │                 │                 │  Receive      │
     │               │                 │                 │  Alert        │
     │               │                 │                 │◀──────────────│
     │               │                 │                 │               │
     │               │                 │                 │  Clear Path   │
     │               │                 │                 │──────────────▶│
     │               │                 │                 │               │
     │  Pass Through │                 │                 │               │
     │──────────────────────────────────────────────────────────────────▶│
     │               │                 │                 │               │
     │               │  Reset          │                 │               │
     │               │  Normal         │                 │               │
     │               │────────────────▶│                 │               │
     │               │                 │                 │               │
└────┴────┘     └────┴────┘     └──────┴──────┘     └────┴────┘     └────┴───┘
```

### Class Diagram (Simplified)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLASS RELATIONSHIPS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐         ┌────────────────────┐                      │
│  │   SimulationMain   │◄────────│  MetricsCollector  │                      │
│  ├────────────────────┤         ├────────────────────┤                      │
│  │ - traci_connection │         │ - traffic_metrics  │                      │
│  │ - vehicles         │         │ - comm_metrics     │                      │
│  │ - traffic_lights   │         │ - siot_metrics     │                      │
│  ├────────────────────┤         ├────────────────────┤                      │
│  │ + initialize()     │         │ + collect()        │                      │
│  │ + run_step()       │         │ + export()         │                      │
│  │ + terminate()      │         │ + visualize()      │                      │
│  └─────────┬──────────┘         └────────────────────┘                      │
│            │                                                                 │
│            │  uses                                                           │
│            ▼                                                                 │
│  ┌────────────────────┐         ┌────────────────────┐                      │
│  │ VehicleController  │◄───────▶│   V2VCommunication │                      │
│  ├────────────────────┤         ├────────────────────┤                      │
│  │ - vehicle_id       │         │ - communication_   │                      │
│  │ - position         │         │   range            │                      │
│  │ - speed            │         │ - message_queue    │                      │
│  ├────────────────────┤         ├────────────────────┤                      │
│  │ + update_state()   │         │ + broadcast()      │                      │
│  │ + apply_action()   │         │ + receive()        │                      │
│  └────────────────────┘         │ + process_alerts() │                      │
│            │                    └────────────────────┘                      │
│            │  uses                        │                                  │
│            ▼                              │                                  │
│  ┌────────────────────┐                   │                                  │
│  │ TrafficLightCtrl   │◄──────────────────┘                                 │
│  ├────────────────────┤         ┌────────────────────┐                      │
│  │ - signal_id        │         │   V2ICommunication │                      │
│  │ - phases           │◄───────▶├────────────────────┤                      │
│  │ - current_phase    │         │ - rsu_location     │                      │
│  ├────────────────────┤         │ - spat_data        │                      │
│  │ + set_phase()      │         ├────────────────────┤                      │
│  │ + get_timing()     │         │ + send_spat()      │                      │
│  │ + handle_emergency │         │ + receive_vehicle_ │                      │
│  └────────────────────┘         │   data()           │                      │
│            │                    └────────────────────┘                      │
│            │  uses                        │                                  │
│            ▼                              │                                  │
│  ┌────────────────────┐                   │                                  │
│  │    SIoTManager     │◄──────────────────┘                                 │
│  ├────────────────────┤         ┌────────────────────┐                      │
│  │ - trust_network    │         │  EmergencyHandler  │                      │
│  │ - relationships    │◄───────▶├────────────────────┤                      │
│  ├────────────────────┤         │ - emergency_       │                      │
│  │ + update_trust()   │         │   vehicles         │                      │
│  │ + get_trust()      │         │ - priority_queue   │                      │
│  │ + form_relation()  │         ├────────────────────┤                      │
│  └────────────────────┘         │ + detect_emergency │                      │
│                                 │ + request_priority │                      │
│                                 │ + clear_path()     │                      │
│                                 └────────────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Considerations

### Scalability

| Vehicles | Recommended Step Length | Memory (Approx) |
|----------|-------------------------|-----------------|
| < 100 | 0.1 s | 500 MB |
| 100-500 | 0.1 s | 1 GB |
| 500-1000 | 0.5 s | 2 GB |
| > 1000 | 1.0 s | 4+ GB |

### Optimization Strategies

1. **Spatial Indexing**: Use quadtrees for efficient neighbor queries
2. **Message Batching**: Group V2V messages for efficiency
3. **Adaptive Communication**: Reduce frequency in low-traffic areas
4. **Lazy Trust Updates**: Only update when significant changes occur
