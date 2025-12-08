# Smart Traffic Management System - Documentation

## Author: Soujanya Poojari
## Pages: 26-50

---

## Table of Contents (Pages 26-50)

26. Python/TraCI Real-Time Control
27. Adaptive Signal Control Algorithms
28. Congestion Detection Methods
29. Lane Occupancy Monitoring
30. Dynamic Phase Duration Adjustment
31. Green Wave Coordination
32. Emergency Vehicle Detection
33. Priority Signal Override
34. Traffic Halting for Emergency Passage
35. Green Wave Corridor Creation
36. Vehicle Behavior Modification
37. Speed Control and Optimization
38. Lane Change Intent Communication
39. Cooperative Driving Algorithms
40. Event-Driven Simulation Architecture
41. State Machine Design for Signals
42. Real-Time Metric Collection
43. Performance Optimization Techniques
44. Multi-Threading and Concurrency
45. Error Handling and Fault Tolerance
46. Database Integration with SQLite
47. CSV Logging Infrastructure
48. Metric Aggregation and Analysis
49. Simulation Time Management
50. CS Course Mapping: Python Control Systems

---

## Page 26: Python/TraCI Real-Time Control

### TraCI Architecture

TraCI (Traffic Control Interface) provides a TCP-based client-server architecture for controlling SUMO in real-time.

```python
import traci

# Connection parameters
SUMO_PORT = 8813
SUMO_HOST = "localhost"

# Start SUMO with TraCI server
sumoCmd = ["sumo", "-c", "simulation.sumocfg", "--remote-port", str(SUMO_PORT)]
traci.start(sumoCmd)

# Subscribe to vehicle data for efficient updates
traci.vehicle.subscribe("vehicle_1", [
    traci.constants.VAR_SPEED,
    traci.constants.VAR_POSITION,
    traci.constants.VAR_LANE_ID
])

# Main control loop
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    # Get subscribed data efficiently
    vehicle_data = traci.vehicle.getSubscriptionResults("vehicle_1")
    speed = vehicle_data[traci.constants.VAR_SPEED]
    position = vehicle_data[traci.constants.VAR_POSITION]
    
    # Apply control logic
    if speed < 5.0:
        traci.vehicle.setSpeed("vehicle_1", 10.0)

traci.close()
```

### Key TraCI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `simulationStep()` | Advance simulation | `traci.simulationStep()` |
| `getIDList()` | Get entity IDs | `traci.vehicle.getIDList()` |
| `getSpeed()` | Get vehicle speed | `traci.vehicle.getSpeed("v1")` |
| `setSpeed()` | Set vehicle speed | `traci.vehicle.setSpeed("v1", 10)` |
| `getPosition()` | Get coordinates | `traci.vehicle.getPosition("v1")` |
| `trafficlight.getState()` | Get signal state | `traci.trafficlight.getRedYellowGreenState("tl1")` |

### Real-Time Control Example

```python
def adaptive_control_junction(junction_id, congestion_threshold=0.7):
    """
    Adaptive signal control based on real-time congestion.
    """
    # Get incoming lanes
    controlled_links = traci.trafficlight.getControlledLinks(junction_id)
    
    # Calculate occupancy for each approach
    occupancies = {}
    for link_list in controlled_links:
        for link in link_list:
            lane_id = link[0]
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            occupancy = vehicle_count / (lane_length / 7.5)  # 7.5m per vehicle
            occupancies[lane_id] = occupancy
    
    # Adapt signal timing
    current_phase = traci.trafficlight.getPhase(junction_id)
    current_duration = traci.trafficlight.getPhaseDuration(junction_id)
    
    # Find most congested lane with green
    max_occupancy = max(occupancies.values())
    
    if max_occupancy > congestion_threshold:
        # Extend green phase
        new_duration = min(current_duration + 15, 90)
        traci.trafficlight.setPhaseDuration(junction_id, new_duration)
        print(f"Extended green at {junction_id} to {new_duration}s due to congestion")
```

### CS Course Mapping

**Operating Systems**:
- Client-server architecture (TraCI protocol)
- Inter-process communication (IPC)
- Real-time system constraints
- Process synchronization

**Computer Networks**:
- TCP socket communication
- Request-response protocol
- Network latency handling
- Data serialization

---

## Page 27: Adaptive Signal Control Algorithms

### Occupancy-Based Adaptation

The core algorithm monitors lane occupancy and extends green phases for congested approaches:

```python
class AdaptiveSignalController:
    def __init__(self):
        self.occupancy_threshold = 0.7
        self.min_green = 20  # seconds
        self.max_green = 90  # seconds
        self.extension = 15  # seconds
    
    def calculate_lane_occupancy(self, lane_id):
        """Calculate occupancy as percentage of capacity"""
        vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
        length = traci.lane.getLength(lane_id)
        capacity = length / 7.5  # Average vehicle + gap
        return vehicles / capacity if capacity > 0 else 0
    
    def should_extend_green(self, junction_id, lane_id):
        """Determine if green phase should be extended"""
        occupancy = self.calculate_lane_occupancy(lane_id)
        
        # Check if lane currently has green signal
        state = traci.trafficlight.getRedYellowGreenState(junction_id)
        lane_index = self.get_lane_signal_index(junction_id, lane_id)
        
        has_green = state[lane_index] == 'G' if lane_index >= 0 else False
        is_congested = occupancy > self.occupancy_threshold
        
        # Check time until phase switch
        next_switch = traci.trafficlight.getNextSwitch(junction_id)
        current_time = traci.simulation.getTime()
        time_remaining = next_switch - current_time
        
        return has_green and is_congested and time_remaining < 5
    
    def extend_green_phase(self, junction_id):
        """Extend current green phase duration"""
        current_duration = traci.trafficlight.getPhaseDuration(junction_id)
        new_duration = min(current_duration + self.extension, self.max_green)
        traci.trafficlight.setPhaseDuration(junction_id, new_duration)
        
        return new_duration
```

### Webster's Method (Baseline Comparison)

```python
def websters_method(flows, lost_time=3):
    """
    Calculate optimal cycle time using Webster's method.
    
    Args:
        flows: Dictionary of {phase: critical_flow_ratio}
        lost_time: Lost time per phase (seconds)
    
    Returns:
        Optimal cycle time (seconds)
    """
    Y = sum(flows.values())  # Sum of critical flow ratios
    L = lost_time * len(flows)  # Total lost time
    
    cycle_time = (1.5 * L + 5) / (1 - Y)
    
    return cycle_time

# Example usage
critical_flows = {
    'NS_phase': 0.3,
    'EW_phase': 0.4
}

optimal_cycle = websters_method(critical_flows)
print(f"Optimal cycle time: {optimal_cycle:.1f} seconds")
```

### Sample Output Log

```
[10.5s] Junction junction_1: Occupancy 0.45 - Normal operation
[45.2s] Junction junction_1: Occupancy 0.73 - CONGESTION DETECTED
[45.2s] Extended green at junction_1 from 42s to 57s
[60.8s] Junction junction_1: Occupancy 0.52 - Congestion clearing
[120.3s] Junction junction_2: Occupancy 0.81 - SEVERE CONGESTION
[120.3s] Extended green at junction_2 from 42s to 57s
[135.7s] Junction junction_2: Occupancy 0.69 - Improvement observed
```

---

## Page 28: Congestion Detection Methods

### Multi-Criteria Congestion Detection

```python
class CongestionDetector:
    def __init__(self):
        self.thresholds = {
            'occupancy': 0.7,
            'speed': 5.0,  # m/s
            'waiting_time': 30.0  # seconds
        }
    
    def detect_congestion_level(self, lane_id):
        """
        Detect congestion using multiple criteria.
        
        Returns:
            str: 'LOW', 'MEDIUM', 'HIGH', or 'SEVERE'
        """
        # Criterion 1: Occupancy
        occupancy = self.calculate_occupancy(lane_id)
        
        # Criterion 2: Average speed
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Criterion 3: Waiting time
        waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
        max_waiting = max(waiting_times) if waiting_times else 0
        
        # Scoring
        score = 0
        if occupancy > self.thresholds['occupancy']:
            score += 2
        if avg_speed < self.thresholds['speed']:
            score += 2
        if max_waiting > self.thresholds['waiting_time']:
            score += 1
        
        # Classify
        if score >= 4:
            return 'SEVERE'
        elif score >= 3:
            return 'HIGH'
        elif score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
```

---

## Pages 29-50: [Additional Content]

*Each page continues with detailed implementations of:*
- Lane occupancy monitoring systems
- Dynamic phase adjustment algorithms
- Emergency vehicle priority logic
- V2V communication protocols
- Database logging mechanisms
- Performance optimization techniques

---

## To-Do Checklist (Pages 26-50)

- [x] Document TraCI control architecture
- [x] Implement adaptive signal algorithms
- [x] Create congestion detection system
- [ ] Complete emergency priority documentation
- [ ] Add V2V protocol specifications
- [ ] Include performance benchmarks
- [ ] Map all Python concepts to CS courses
- [ ] Create code testing procedures

---

*Documentation authored by Soujanya Poojari (Pages 26-50)*
