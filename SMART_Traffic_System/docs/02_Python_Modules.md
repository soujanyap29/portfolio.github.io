# Python Modules Documentation

This document provides detailed documentation for all Python modules in the SMART Traffic System.

---

## Module Overview

```
src/
├── main.py                       # Main simulation controller
├── config_manager.py             # Configuration management
├── v2x_communication.py          # V2X communication
├── lane_detection.py             # Lane monitoring
├── traffic_signal_control.py     # Signal control
├── emergency_vehicle_priority.py # Emergency handling
├── performance_metrics.py        # Metrics collection
└── utils.py                      # Utility functions
```

---

## 1. main.py - Main Simulation Controller

### Class: `TrafficSimulation`

**Purpose:** Orchestrates the entire simulation, managing SUMO connection and coordinating all subsystems.

**Key Methods:**

#### `__init__(config_file, gui)`
Initialize the simulation.

```python
sim = TrafficSimulation(
    config_file='config/simulation_config.json',
    gui=True  # Use SUMO GUI
)
```

**Parameters:**
- `config_file` (str): Path to configuration JSON
- `gui` (bool): Whether to use SUMO-GUI

#### `start_sumo()`
Start SUMO simulation with TraCI connection.

```python
sim.start_sumo()
```

**Returns:** None  
**Side Effects:** 
- Launches SUMO process
- Establishes TraCI connection
- Initializes all subsystems

#### `run(duration)`
Execute the main simulation loop.

```python
sim.run(duration=3600)  # Run for 1 hour
```

**Parameters:**
- `duration` (int): Simulation time in seconds (optional)

**Process:**
1. Execute simulation steps
2. Update all subsystems
3. Detect emergency vehicles
4. Collect performance metrics
5. Display status every 60 seconds

#### `add_emergency_vehicle(time, lane, vehicle_type)`
Schedule an emergency vehicle.

```python
sim.add_emergency_vehicle(
    time=300,           # Depart at 300 seconds
    lane=2,             # In lane 2
    vehicle_type='ambulance'
)
```

#### `export_results(output_dir)`
Export collected metrics.

```python
sim.export_results('results/run_001')
```

---

## 2. config_manager.py - Configuration Management

### Class: `ConfigManager`

**Purpose:** Handles loading, validation, and management of configuration files.

**Key Methods:**

#### `__init__(config_file)`
Load configuration from file.

```python
config_mgr = ConfigManager('config/simulation_config.json')
```

#### `get_config()`
Get complete configuration dictionary.

```python
config = config_mgr.get_config()
print(config['simulation']['total_time'])  # 3600
```

#### `get(*keys, default)`
Get nested configuration value.

```python
comm_range = config_mgr.get('v2x', 'communication_range', default=300)
# Returns 300 if not found
```

#### `set(*keys, value)`
Set nested configuration value.

```python
config_mgr.set('traffic', 'vehicle_density', value='high')
```

#### `save_config(output_file)`
Save current configuration.

```python
config_mgr.save_config('config/modified_config.json')
```

**Default Configuration:**
```python
{
    'simulation': {
        'step_length': 0.1,
        'total_time': 3600,
        'gui_enabled': False
    },
    'traffic': {
        'vehicle_density': 'medium',
        'emergency_vehicle_probability': 0.01
    },
    'v2x': {
        'communication_range': 300,
        'message_frequency': 10
    }
}
```

---

## 3. v2x_communication.py - V2X Communication

### Class: `V2XCommunication`

**Purpose:** Simulates Vehicle-to-Everything (V2X) communication.

**Key Methods:**

#### `__init__(range_meters, frequency)`
Initialize V2X system.

```python
v2x = V2XCommunication(
    range_meters=300,  # 300m range
    frequency=10       # 10 Hz
)
```

#### `update(vehicle_ids, current_time)`
Update V2X communications.

```python
v2x.update(traci.vehicle.getIDList(), 150.5)
```

**Process:**
1. Update vehicle positions
2. Broadcast messages at frequency
3. Distribute to nearby vehicles

#### `broadcast_emergency_alert(emergency_vehicle_id)`
Send emergency alert to nearby vehicles.

```python
v2x.broadcast_emergency_alert('ambulance_001')
```

**Alert Message Structure:**
```python
{
    'type': 'EMERGENCY_ALERT',
    'vehicle_id': 'ambulance_001',
    'position': (123.4, 567.8),
    'action': 'CLEAR_LANE'
}
```

#### `get_messages(vehicle_id)`
Get messages for specific vehicle.

```python
messages = v2x.get_messages('car_123')
for msg in messages:
    if msg['is_emergency']:
        print(f"Emergency vehicle nearby: {msg['sender']}")
```

#### `get_nearby_vehicles(vehicle_id, range_override)`
Get vehicles within communication range.

```python
nearby = v2x.get_nearby_vehicles('car_001', range_override=500)
# Returns list of vehicle IDs
```

**V2X Message Format:**
```python
{
    'sender': 'vehicle_id',
    'type': 'car/truck/ambulance',
    'speed': 12.5,  # m/s
    'lane': 'N_to_J0_2',
    'position': (x, y),
    'is_emergency': False
}
```

---

## 4. lane_detection.py - Lane Monitoring

### Class: `LaneDetection`

**Purpose:** Monitors and analyzes traffic on individual lanes.

**Key Methods:**

#### `analyze_lanes(vehicle_ids)`
Analyze current lane conditions.

```python
lane_detector = LaneDetection()
lane_data = lane_detector.analyze_lanes(vehicle_ids)
```

**Returns:** Dictionary with lane statistics:
```python
{
    'N_to_J0_0': {
        'vehicle_count': 12,
        'avg_speed': 8.5,      # m/s
        'queue_length': 5,      # vehicles
        'congested': True,
        'occupancy': 0.18,      # 18%
        'vehicles': [...]       # Vehicle details
    },
    'N_to_J0_1': {...}
}
```

#### `get_lane_congestion_level(lane_id)`
Get congestion level classification.

```python
level = lane_detector.get_lane_congestion_level('N_to_J0_2')
# Returns: 'free', 'moderate', 'heavy', or 'severe'
```

**Thresholds:**
- **Free:** speed > 10 m/s, occupancy < 0.1
- **Moderate:** speed > 7 m/s, occupancy < 0.15
- **Heavy:** speed > 3 m/s
- **Severe:** speed ≤ 3 m/s

#### `get_most_congested_lane()`
Find most congested lane.

```python
lane_id, score = lane_detector.get_most_congested_lane()
print(f"Most congested: {lane_id} (score: {score})")
```

#### `get_vehicles_in_lane(lane_id)`
Get all vehicles in specific lane.

```python
vehicles = lane_detector.get_vehicles_in_lane('N_to_J0_2')
# Returns list of vehicle IDs
```

**Congestion Detection:**
- Uses speed threshold: 5.0 m/s
- Uses density threshold: 0.15 vehicles/meter
- Both conditions must be met

---

## 5. traffic_signal_control.py - Signal Control

### Class: `TrafficSignalControl`

**Purpose:** Manages adaptive and intelligent traffic signal control.

**Key Methods:**

#### `update_adaptive_control(lane_data, current_time)`
Update signals using adaptive algorithm.

```python
signal_control = TrafficSignalControl()
signal_control.update_adaptive_control(lane_data, 150.5)
```

**Algorithm:**
1. Calculate demand for each approach (N-S, E-W)
2. Apply Webster's method for optimal cycle
3. Allocate green time proportionally
4. Constrain within min/max limits

#### `emergency_mode(emergency_vehicle_id)`
Activate emergency signal preemption.

```python
signal_control.emergency_mode('ambulance_001')
```

**Process:**
1. Get vehicle route
2. Find traffic lights on route
3. Switch to emergency program
4. Give green to emergency direction

#### `deactivate_emergency_mode(tl_id)`
Return signal to normal operation.

```python
signal_control.deactivate_emergency_mode('J0')
```

#### `set_time_of_day_plan(hour)`
Set signal plan based on time.

```python
signal_control.set_time_of_day_plan(17)  # 5 PM
# Activates 'peak_hour' program
```

**Time Periods:**
- **Peak:** 7-9 AM, 4-7 PM → 'peak_hour'
- **Night:** 10 PM - 6 AM → 'off_peak'
- **Normal:** Other times → 'adaptive'

#### `get_current_phase(tl_id)`
Get current signal phase.

```python
phase = signal_control.get_current_phase('J0')
# Returns 0, 1, 2, or 3
```

**Webster's Method Implementation:**
```python
L = yellow_time * 2 + all_red_time * 2  # Lost time
Y = total_demand / 3600.0  # Critical flow ratio
optimal_cycle = (1.5 * L + 5) / (1 - Y)
```

---

## 6. emergency_vehicle_priority.py - Emergency Handling

### Class: `EmergencyVehiclePriority`

**Purpose:** Manages detection and priority for emergency vehicles.

**Key Methods:**

#### `detect_emergency_vehicles(vehicle_ids)`
Detect emergency vehicles.

```python
emerg_priority = EmergencyVehiclePriority()
emergencies = emerg_priority.detect_emergency_vehicles(vehicle_ids)
```

**Returns:** List of emergency vehicle IDs

**Detected Types:**
- Ambulance (Priority 1)
- Fire Truck (Priority 2)
- Police (Priority 3)

#### `activate_priority(emergency_vehicle_id, lane_data)`
Activate priority handling.

```python
emerg_priority.activate_priority('ambulance_001', lane_data)
```

**Actions:**
1. Identify emergency vehicle lane
2. Clear vehicles from lane
3. Activate signal preemption
4. Log event

#### `create_green_corridor(emergency_vehicle_id, traffic_lights)`
Create green corridor.

```python
emerg_priority.create_green_corridor('ambulance_001', tl_list)
```

**Process:**
1. Get emergency vehicle route
2. Find junctions on route
3. Preemptively switch signals to green
4. Coordinate timing

#### `get_highest_priority_vehicle()`
Get vehicle with highest priority.

```python
top_priority = emerg_priority.get_highest_priority_vehicle()
# Returns vehicle ID or None
```

**Priority Arbitration:**
```python
EMERGENCY_TYPES = {
    'ambulance': {'priority': 1, 'name': 'Ambulance'},
    'fire_truck': {'priority': 2, 'name': 'Fire Truck'},
    'police': {'priority': 3, 'name': 'Police'}
}
```

#### `normalize_traffic(vehicle_id)`
Return to normal after emergency passes.

```python
emerg_priority.normalize_traffic('ambulance_001')
```

**Lane Clearance Algorithm:**
```python
def _clear_lane(emerg_vehicle_id, lane_id):
    1. Get vehicles ahead of emergency vehicle
    2. For each vehicle:
        a. Try move to adjacent lane
        b. If not possible, increase speed
        c. Log successful clearances
```

---

## 7. performance_metrics.py - Metrics Collection

### Class: `PerformanceMetrics`

**Purpose:** Collects, analyzes, and exports simulation performance data.

**Key Methods:**

#### `__init__(output_dir)`
Initialize metrics collector.

```python
metrics = PerformanceMetrics(output_dir='results/metrics')
```

#### `collect_data(vehicle_ids, lane_data, current_time)`
Collect data for current step.

```python
metrics.collect_data(vehicle_ids, lane_data, 150.5)
```

**Collected Metrics:**
- Vehicle count (time series)
- Average speeds
- Waiting times
- Lane statistics
- Individual vehicle data

#### `record_emergency_event(vehicle_id, event_type, details)`
Log emergency event.

```python
metrics.record_emergency_event(
    'ambulance_001',
    'priority_activated',
    {'lane': 2, 'time': 150.5}
)
```

#### `calculate_throughput()`
Calculate traffic throughput.

```python
throughput = metrics.calculate_throughput()
# Returns vehicles per hour
```

#### `calculate_average_travel_time()`
Calculate average travel time.

```python
avg_time = metrics.calculate_average_travel_time()
# Returns seconds
```

#### `export_csv(output_dir)`
Export metrics to CSV files.

```python
metrics.export_csv('results/simulation_001')
```

**Output Files:**
- `time_series.csv` - Time-based metrics
- `lane_metrics.csv` - Per-lane data
- `vehicle_trips.csv` - Trip data
- `emergency_events.csv` - Emergency logs

#### `generate_summary_report(output_dir)`
Generate text summary report.

```python
metrics.generate_summary_report('results/simulation_001')
```

**Report Contents:**
```
=============================================
SMART Traffic System - Performance Summary
=============================================

Traffic Flow Metrics:
----------------------------------------
Total Vehicles: 1234
Throughput: 823.5 veh/hour
Average Travel Time: 145.2 seconds
Average Delay: 23.4 seconds

Emergency Vehicle Events:
----------------------------------------
Total Emergency Events: 5

Lane Performance:
----------------------------------------
N_to_J0_0: avg 12.3 vehicles, 8.5 m/s
N_to_J0_1: avg 10.1 vehicles, 9.2 m/s
...
```

---

## 8. utils.py - Utility Functions

### Standalone Functions

#### `format_time(seconds)`
Format seconds as HH:MM:SS.

```python
from utils import format_time
time_str = format_time(3665)  # "01:01:05"
```

#### `calculate_distance(pos1, pos2)`
Calculate Euclidean distance.

```python
from utils import calculate_distance
dist = calculate_distance((0, 0), (3, 4))  # 5.0 meters
```

#### `validate_sumo_installation()`
Check SUMO installation.

```python
from utils import validate_sumo_installation
if validate_sumo_installation():
    print("SUMO is ready!")
```

#### `mps_to_kmh(speed_mps)` / `kmh_to_mps(speed_kmh)`
Convert speed units.

```python
from utils import mps_to_kmh, kmh_to_mps

kmh = mps_to_kmh(13.89)  # 50.0 km/h
mps = kmh_to_mps(50.0)   # 13.89 m/s
```

### Class: `ProgressBar`

Display console progress bar.

```python
from utils import ProgressBar

progress = ProgressBar(total=100, prefix='Loading:')
for i in range(100):
    progress.update(i + 1)
    time.sleep(0.01)

# Output:
# Loading: |████████████████████████| 100% (100/100)
```

---

## Module Integration Example

Here's how all modules work together:

```python
from main import TrafficSimulation
from config_manager import ConfigManager
from v2x_communication import V2XCommunication
from lane_detection import LaneDetection
from traffic_signal_control import TrafficSignalControl
from emergency_vehicle_priority import EmergencyVehiclePriority
from performance_metrics import PerformanceMetrics

# 1. Load configuration
config_mgr = ConfigManager('config/simulation_config.json')
config = config_mgr.get_config()

# 2. Initialize subsystems
v2x = V2XCommunication(
    range_meters=config['v2x']['communication_range'],
    frequency=config['v2x']['message_frequency']
)

lane_detector = LaneDetection()
signal_control = TrafficSignalControl()
emerg_priority = EmergencyVehiclePriority()
metrics = PerformanceMetrics()

# 3. Simulation loop (simplified)
for step in range(total_steps):
    # Get vehicles
    vehicle_ids = traci.vehicle.getIDList()
    
    # Update V2X
    v2x.update(vehicle_ids, current_time)
    
    # Analyze lanes
    lane_data = lane_detector.analyze_lanes(vehicle_ids)
    
    # Check emergencies
    emergencies = emerg_priority.detect_emergency_vehicles(vehicle_ids)
    
    if emergencies:
        for emerg_id in emergencies:
            emerg_priority.activate_priority(emerg_id, lane_data)
            signal_control.emergency_mode(emerg_id)
    else:
        signal_control.update_adaptive_control(lane_data, current_time)
    
    # Collect metrics
    metrics.collect_data(vehicle_ids, lane_data, current_time)

# 4. Export results
metrics.export_csv('results')
metrics.generate_summary_report('results')
```

---

## Error Handling

All modules include error handling:

```python
try:
    speed = traci.vehicle.getSpeed(vehicle_id)
except traci.exceptions.TraCIException:
    # Vehicle no longer in simulation
    pass
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
```

---

## Testing Modules

### Unit Test Example

```python
import unittest
from lane_detection import LaneDetection

class TestLaneDetection(unittest.TestCase):
    def setUp(self):
        self.detector = LaneDetection()
    
    def test_congestion_detection(self):
        lane_data = {
            'vehicle_count': 15,
            'avg_speed': 4.0,  # Low speed
            'occupancy': 0.20  # High occupancy
        }
        
        is_congested = self.detector._is_congested(lane_data)
        self.assertTrue(is_congested)

if __name__ == '__main__':
    unittest.main()
```

---

## Performance Considerations

### Optimization Tips:

1. **Batch Operations:**
   ```python
   # Instead of:
   for veh_id in vehicles:
       speed = traci.vehicle.getSpeed(veh_id)
   
   # Use:
   speeds = {veh_id: traci.vehicle.getSpeed(veh_id) 
             for veh_id in vehicles}
   ```

2. **Cache Results:**
   ```python
   # Cache vehicle types
   self.vehicle_types = {}
   
   def get_vehicle_type(self, veh_id):
       if veh_id not in self.vehicle_types:
           self.vehicle_types[veh_id] = traci.vehicle.getTypeID(veh_id)
       return self.vehicle_types[veh_id]
   ```

3. **Update Frequency:**
   ```python
   # Don't update every step if not needed
   if step % 10 == 0:  # Update every 10 steps
       self.update_expensive_calculation()
   ```

---

## Debugging

### Enable Verbose Output:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.debug("Detailed debug information")
```

### TraCI Connection Issues:

```python
# Check connection
if traci.isLoaded():
    print("TraCI connected")
else:
    print("TraCI not connected")

# Get SUMO version
version = traci.getVersion()
print(f"SUMO version: {version}")
```

---

This completes the Python modules documentation for the SMART Traffic System.
