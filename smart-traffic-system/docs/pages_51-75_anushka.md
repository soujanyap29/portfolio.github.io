# Smart Traffic Management System - Documentation

## Author: Anushka
## Pages: 51-75

---

## Table of Contents (Pages 51-75)

51. NS3 Network Simulation Framework
52. V2V Communication Protocol Design
53. V2I Communication Protocol Design
54. WAVE (IEEE 802.11p) Protocol
55. Message Broadcasting Mechanisms
56. Unicast and Multicast Communication
57. Communication Range Modeling
58. Signal Propagation and Attenuation
59. Latency Simulation and Analysis
60. Packet Loss and Reliability
61. Protocol State Machines
62. Message Types and Formats
63. SPaT (Signal Phase and Timing) Messages
64. MAP (Intersection Geometry) Messages
65. BSM (Basic Safety Messages)
66. SIoT Relationship Types
67. Trust Score Calculation Methods
68. Parental Object Relationships (POR)
69. Co-location Object Relationships (CLOR)
70. Co-work Object Relationships (CWOR)
71. Social Object Relationships (SOR)
72. Trust Evolution Over Time
73. Cooperative Decision Making
74. Message Validation and Filtering
75. CS Course Mapping: Networking and Social Computing

---

## Page 51: NS3 Network Simulation Framework

### NS3 Overview

NS3 (Network Simulator 3) is a discrete-event network simulator for Internet systems, used for modeling V2V and V2I communication.

### NS3 Architecture for V2X

```cpp
// ns3_v2x_setup.cc
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"

using namespace ns3;

class V2XNetworkSimulator {
public:
    void SetupNetwork() {
        // Create vehicle nodes
        NodeContainer vehicleNodes;
        vehicleNodes.Create(100);
        
        // Create RSU nodes
        NodeContainer rsuNodes;
        rsuNodes.Create(4);
        
        // Install WAVE (802.11p) protocol
        YansWifiChannelHelper waveChannel = YansWifiChannelHelper::Default();
        YansWifiPhyHelper wavePhy = YansWifiPhyHelper::Default();
        wavePhy.SetChannel(waveChannel.Create());
        
        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211p);
        
        // Set up MAC layer
        WifiMacHelper waveMac;
        waveMac.SetType("ns3::OcbWifiMac");
        
        // Install devices
        NetDeviceContainer vehicleDevices = wifi.Install(wavePhy, waveMac, vehicleNodes);
        NetDeviceContainer rsuDevices = wifi.Install(wavePhy, waveMac, rsuNodes);
        
        // Set mobility model
        MobilityHelper mobility;
        mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                     "MinX", DoubleValue(0.0),
                                     "MinY", DoubleValue(0.0),
                                     "DeltaX", DoubleValue(500.0),
                                     "DeltaY", DoubleValue(500.0));
        
        mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        mobility.Install(vehicleNodes);
        
        // RSUs are stationary
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(rsuNodes);
    }
};
```

### Integration with SUMO via TraCI

```python
# ns3_sumo_bridge.py
import traci
import socket
import json

class NS3SUMOBridge:
    """Bridge between SUMO and NS3 for synchronized simulation"""
    
    def __init__(self, ns3_host='localhost', ns3_port=9999):
        self.ns3_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ns3_socket.connect((ns3_host, ns3_port))
    
    def sync_vehicle_positions(self):
        """Send vehicle positions from SUMO to NS3"""
        vehicles = traci.vehicle.getIDList()
        positions = {}
        
        for veh_id in vehicles:
            pos = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            positions[veh_id] = {
                'x': pos[0],
                'y': pos[1],
                'speed': speed
            }
        
        # Send to NS3
        message = json.dumps(positions)
        self.ns3_socket.send(message.encode())
    
    def receive_communication_events(self):
        """Receive V2X communication events from NS3"""
        data = self.ns3_socket.recv(4096)
        if data:
            return json.loads(data.decode())
        return None
```

### CS Course Mapping

**Computer Networks**:
- Network protocols (802.11p, WAVE)
- MAC and PHY layers
- Wireless communication
- Network simulation

---

## Page 52: V2V Communication Protocol Design

### V2V Message Types

```python
class V2VMessage:
    """Base class for V2V messages"""
    
    def __init__(self, sender_id, timestamp):
        self.sender_id = sender_id
        self.timestamp = timestamp
        self.message_id = f"{sender_id}_{timestamp}"

class PositionMessage(V2VMessage):
    """Vehicle position broadcast"""
    
    def __init__(self, sender_id, position, speed, heading):
        super().__init__(sender_id, time.time())
        self.position = position  # (x, y)
        self.speed = speed
        self.heading = heading
    
    def to_dict(self):
        return {
            'type': 'V2V_POSITION',
            'sender': self.sender_id,
            'timestamp': self.timestamp,
            'position': self.position,
            'speed': self.speed,
            'heading': self.heading
        }

class BrakeWarning(V2VMessage):
    """Emergency braking notification"""
    
    def __init__(self, sender_id, deceleration, position):
        super().__init__(sender_id, time.time())
        self.deceleration = deceleration
        self.position = position
        self.severity = self.calculate_severity()
    
    def calculate_severity(self):
        if self.deceleration > 6.0:
            return 'CRITICAL'
        elif self.deceleration > 4.0:
            return 'HIGH'
        else:
            return 'MEDIUM'
```

### V2V Broadcasting

```python
def broadcast_position(vehicle_id, v2x_system):
    """Broadcast vehicle position to nearby vehicles"""
    
    position = traci.vehicle.getPosition(vehicle_id)
    speed = traci.vehicle.getSpeed(vehicle_id)
    angle = traci.vehicle.getAngle(vehicle_id)
    
    message = PositionMessage(vehicle_id, position, speed, angle)
    
    # Broadcast to vehicles within range
    nearby_vehicles = v2x_system.get_vehicles_in_range(
        vehicle_id, 
        v2x_system.communication_range
    )
    
    for target_id in nearby_vehicles:
        v2x_system.send_message(vehicle_id, target_id, message)
    
    return len(nearby_vehicles)
```

### Sample V2V Message Log

```csv
timestamp,sender,receiver,message_type,distance,latency,content
10.5,vehicle_1,vehicle_2,V2V_POSITION,45.2,0.0012,{"pos":[500,250],"speed":15.2}
10.5,vehicle_1,vehicle_3,V2V_POSITION,78.3,0.0018,{"pos":[500,250],"speed":15.2}
12.8,vehicle_5,vehicle_6,V2V_BRAKE,32.1,0.0010,{"decel":5.2,"severity":"HIGH"}
15.3,vehicle_2,vehicle_1,V2V_LANE_CHANGE,50.0,0.0013,{"intent":"left","eta":2.5}
```

---

## Page 66: SIoT Relationship Types

### Parental Object Relationship (POR)

Vehicles manufactured by the same company or of the same type share a POR:

```python
def establish_por(vehicle1_id, vehicle2_id):
    """
    Create parental relationship between same-type vehicles.
    """
    type1 = traci.vehicle.getTypeID(vehicle1_id)
    type2 = traci.vehicle.getTypeID(vehicle2_id)
    
    if type1 == type2:
        trust_manager.establish_relationship(
            vehicle1_id, vehicle2_id, 'POR'
        )
        # Initial trust boost
        trust_manager.trust_scores[(vehicle1_id, vehicle2_id)] = 0.7
        return True
    return False
```

### Trust Score Calculation

```python
def calculate_trust_score(entity1, entity2, interactions):
    """
    Calculate trust score based on interaction history.
    
    Trust(A,B) = α * DirectTrust + β * RecommendedTrust
    
    DirectTrust = Positive / (Positive + Negative)
    RecommendedTrust = Σ(Trust(A,X) * Trust(X,B)) / N
    """
    # Direct trust from interactions
    positive = interactions['positive']
    negative = interactions['negative']
    total = positive + negative
    
    direct_trust = positive / total if total > 0 else 0.5
    
    # Recommended trust from mutual connections
    common_neighbors = trust_manager.get_common_neighbors(entity1, entity2)
    recommended_trust = 0.0
    
    for neighbor in common_neighbors:
        trust_a_x = trust_manager.get_trust_score(entity1, neighbor)
        trust_x_b = trust_manager.get_trust_score(neighbor, entity2)
        recommended_trust += trust_a_x * trust_x_b
    
    if common_neighbors:
        recommended_trust /= len(common_neighbors)
    else:
        recommended_trust = 0.5
    
    # Weighted combination
    alpha = 0.7  # Direct trust weight
    beta = 0.3   # Recommended trust weight
    
    final_trust = alpha * direct_trust + beta * recommended_trust
    
    return final_trust
```

---

## Pages 53-75: [Additional Content]

*Detailed coverage of:*
- V2I protocol specifications
- Message validation mechanisms
- Social relationship modeling
- Trust evolution algorithms
- Cooperative routing decisions

---

## To-Do Checklist (Pages 51-75)

- [x] Document NS3 simulation framework
- [x] Design V2V message protocols
- [x] Implement trust calculation algorithms
- [ ] Complete V2I protocol specification
- [ ] Add SIoT relationship diagrams
- [ ] Include communication performance analysis
- [ ] Map to CS networking courses

---

*Documentation authored by Anushka (Pages 51-75)*
