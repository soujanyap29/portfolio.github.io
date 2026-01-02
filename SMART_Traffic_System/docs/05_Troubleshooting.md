# Troubleshooting Guide

Common issues and solutions for the SMART Traffic System.

---

## Installation Issues

### Problem: SUMO_HOME not set

**Error:**
```
Please declare environment variable 'SUMO_HOME'
```

**Solution:**
```bash
# Linux/Mac
export SUMO_HOME="/usr/share/sumo"
# Add to ~/.bashrc or ~/.zshrc for persistence

# Windows
set SUMO_HOME=C:\Program Files\SUMO
# Or add via System Environment Variables
```

**Verify:**
```bash
echo $SUMO_HOME  # Should show SUMO path
ls $SUMO_HOME/tools  # Should show tools directory
```

### Problem: TraCI import fails

**Error:**
```python
ModuleNotFoundError: No module named 'traci'
```

**Solution:**
```bash
# Ensure SUMO tools are in Python path
export PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH}"

# Or in Python:
import sys
sys.path.append('/usr/share/sumo/tools')
import traci
```

### Problem: Python dependencies missing

**Error:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:**
```bash
# Install all requirements
pip install -r requirements.txt

# Or install individually
pip install numpy pandas matplotlib
```

---

## SUMO Configuration Issues

### Problem: Network file not found

**Error:**
```
Error: Could not open network file '4lane_network.net.xml'
```

**Solution:**
```bash
# Check file exists
ls -l sumo_files/4lane_network.net.xml

# Check working directory
pwd  # Should be in SMART_Traffic_System/

# Run from correct directory
cd SMART_Traffic_System
python src/main.py
```

### Problem: Invalid network topology

**Error:**
```
Error: Network contains errors
```

**Solution:**
```bash
# Validate network
netconvert --sumo-net-file=sumo_files/4lane_network.net.xml \
           --output-file=test.net.xml

# Check for specific errors
sumo -c sumo_files/simulation.sumocfg --no-step-log \
     --error-log=errors.txt

# Review errors.txt
cat errors.txt
```

### Problem: Route file issues

**Error:**
```
Error: Route 'route_N_S' not found
```

**Solution:**
1. Verify route definitions in `traffic_routes.rou.xml`
2. Check edge IDs match network file
3. Validate routes:
```bash
sumo -c simulation.sumocfg --route-files traffic_routes.rou.xml \
     --no-step-log --duration-log.disable
```

---

## Simulation Runtime Issues

### Problem: Simulation crashes immediately

**Symptoms:**
- SUMO window opens and closes
- TraCI connection error

**Solution:**
```python
# Add error handling to main.py
try:
    sim = TrafficSimulation(config_file='config/simulation_config.json')
    sim.run(duration=300)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

**Check logs:**
```bash
# Run with verbose output
python src/main.py --config config/simulation_config.json 2>&1 | tee simulation.log

# Check SUMO log
ls -l *.log
cat sumo-log.txt
```

### Problem: High CPU usage

**Symptoms:**
- Slow simulation
- System becomes unresponsive

**Solutions:**
1. Reduce simulation step length:
```json
{
    "simulation": {
        "step_length": 0.5  // Increase from 0.1
    }
}
```

2. Disable GUI:
```python
sim = TrafficSimulation(gui=False)
```

3. Reduce vehicle density:
```json
{
    "traffic": {
        "vehicle_density": "low"
    }
}
```

### Problem: Memory leak

**Symptoms:**
- Memory usage continuously increases
- System runs out of memory

**Solutions:**
1. Limit simulation duration
2. Clear old data periodically:
```python
# In performance_metrics.py
if current_time % 600 == 0:  # Every 10 minutes
    self._cleanup_old_data()
```

3. Use iterators instead of storing all data:
```python
# Instead of:
all_speeds = [get_speed(v) for v in vehicles]

# Use:
for vehicle in vehicles:
    speed = get_speed(vehicle)
    process_speed(speed)  # Process immediately
```

---

## TraCI Communication Issues

### Problem: TraCI connection timeout

**Error:**
```
TraCIException: Connection to SUMO lost
```

**Solutions:**
1. Increase connection timeout:
```python
traci.start(sumo_cmd, numRetries=10)
```

2. Check SUMO is running:
```bash
ps aux | grep sumo
```

3. Check port availability:
```bash
netstat -an | grep 8813  # Default TraCI port
```

### Problem: Vehicle commands fail

**Error:**
```
TraCIException: Vehicle 'car_001' is not known
```

**Solutions:**
1. Check vehicle exists:
```python
if vehicle_id in traci.vehicle.getIDList():
    traci.vehicle.setSpeed(vehicle_id, speed)
```

2. Handle departures/arrivals:
```python
try:
    speed = traci.vehicle.getSpeed(vehicle_id)
except traci.exceptions.TraCIException:
    # Vehicle has left simulation
    pass
```

---

## Emergency Vehicle Issues

### Problem: Emergency vehicles not detected

**Check:**
1. Vehicle type is correct:
```python
veh_type = traci.vehicle.getTypeID(veh_id)
print(f"Vehicle type: {veh_type}")
# Should be 'ambulance', 'fire_truck', or 'police'
```

2. V2X is enabled:
```json
{
    "v2x": {
        "broadcast_enabled": true
    }
}
```

3. Detection range is adequate:
```json
{
    "emergency": {
        "detection_range": 500
    }
}
```

### Problem: Lane clearance not working

**Debug:**
```python
# Add debug output in emergency_vehicle_priority.py
def _clear_lane(self, emerg_vehicle_id, lane_id, lane_data):
    print(f"Clearing lane {lane_id}")
    print(f"Vehicles in lane: {len(lane_data[lane_id]['vehicles'])}")
    
    for vehicle_info in lane_data[lane_id]['vehicles']:
        success = self._move_to_adjacent_lane(vehicle_info['id'], lane_id)
        print(f"  Vehicle {vehicle_info['id']}: {'SUCCESS' if success else 'FAILED'}")
```

---

## Performance Metrics Issues

### Problem: CSV export fails

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'results/metrics/time_series.csv'
```

**Solutions:**
1. Check directory permissions:
```bash
ls -ld results/metrics
chmod 755 results/metrics
```

2. Close any open CSV files
3. Ensure directory exists:
```python
import os
os.makedirs('results/metrics', exist_ok=True)
```

### Problem: Missing data in metrics

**Check:**
1. Data collection is called:
```python
# In main simulation loop
metrics.collect_data(vehicle_ids, lane_data, current_time)
```

2. Simulation runs long enough:
```python
# At least 60 seconds for meaningful data
sim.run(duration=300)  # 5 minutes minimum
```

---

## GUI Issues

### Problem: SUMO-GUI doesn't open

**Solutions:**
1. Check SUMO-GUI binary:
```bash
which sumo-gui
sumo-gui --version
```

2. Use sumo instead:
```python
sim = TrafficSimulation(gui=False)
```

3. Check X11 forwarding (Linux):
```bash
echo $DISPLAY
# Should show :0 or similar
```

### Problem: GUI is very slow

**Solutions:**
1. Reduce visualization detail:
   - Settings → Vehicles → Hide vehicle shapes
   - Settings → Background → Disable background
   
2. Increase delay between steps:
```xml
<!-- In simulation.sumocfg -->
<gui_only>
    <delay value="200"/>  <!-- Increase from 100 -->
</gui_only>
```

3. Disable detailed rendering:
   - View → Show As → Simple shapes

---

## Configuration Issues

### Problem: Configuration not loading

**Error:**
```
FileNotFoundError: config/simulation_config.json
```

**Solutions:**
1. Check file path:
```bash
ls -l config/simulation_config.json
```

2. Use absolute path:
```python
import os
config_path = os.path.join(os.path.dirname(__file__), 
                          'config/simulation_config.json')
sim = TrafficSimulation(config_file=config_path)
```

### Problem: Invalid JSON

**Error:**
```
JSONDecodeError: Expecting ',' delimiter
```

**Solutions:**
1. Validate JSON:
```bash
python -m json.tool config/simulation_config.json
```

2. Use online JSON validator: https://jsonlint.com/
3. Check for:
   - Missing commas
   - Trailing commas
   - Unquoted strings
   - Incorrect brackets

---

## Network Performance Issues

### Problem: Vehicles not spawning

**Check:**
1. Flow definitions in route file:
```xml
<flow id="flow_car_N_S" type="car" route="route_N_S" 
      begin="0" end="3600" vehsPerHour="800"/>
```

2. Routes are valid:
```xml
<route id="route_N_S" edges="N_to_J0 J0_to_S"/>
```

3. Edges exist in network:
```bash
grep "N_to_J0" sumo_files/4lane_network.net.xml
```

### Problem: Massive traffic jams

**Solutions:**
1. Reduce vehicle density:
```json
{
    "traffic": {
        "vehicle_density": "low"
    }
}
```

2. Increase road capacity:
```xml
<!-- In network file -->
<edge id="N_to_J0" numLanes="6" speed="16.67"/>
```

3. Adjust signal timing:
```json
{
    "signal_control": {
        "max_green_time": 120
    }
}
```

---

## Common Python Errors

### Problem: Import errors

**Error:**
```python
ImportError: cannot import name 'TrafficSimulation'
```

**Solutions:**
```python
# Ensure correct path
import sys
sys.path.append('src')
from main import TrafficSimulation

# Or use relative imports
from src.main import TrafficSimulation
```

### Problem: Attribute errors

**Error:**
```python
AttributeError: 'NoneType' object has no attribute 'update'
```

**Solution:**
```python
# Add None checks
if self.v2x is not None:
    self.v2x.update(vehicle_ids, current_time)
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Add Breakpoints

```python
# Use pdb for debugging
import pdb

# Add breakpoint
pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()
```

### Profile Performance

```python
import cProfile
import pstats

# Profile simulation
profiler = cProfile.Profile()
profiler.enable()

sim.run(duration=300)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Getting Help

### Check Logs

1. **Simulation log:**
```bash
cat simulation.log
```

2. **SUMO log:**
```bash
cat sumo-log.txt
```

3. **Python errors:**
```bash
python src/main.py 2>&1 | tee error.log
```

### System Information

```python
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"SUMO_HOME: {os.environ.get('SUMO_HOME', 'NOT SET')}")
```

### Test Components Individually

```python
# Test configuration
from src.config_manager import ConfigManager
config = ConfigManager('config/simulation_config.json')
print(config.get_config())

# Test V2X
from src.v2x_communication import V2XCommunication
v2x = V2XCommunication(300, 10)
print("V2X initialized")

# Test utilities
from src.utils import validate_sumo_installation
validate_sumo_installation()
```

---

## Reporting Issues

When reporting issues, include:

1. **System information:**
   - OS and version
   - Python version
   - SUMO version
   
2. **Error message:**
   - Complete stack trace
   - Error logs
   
3. **Steps to reproduce:**
   - Configuration used
   - Commands run
   - Expected vs actual behavior
   
4. **Files:**
   - Configuration files
   - Network files
   - Logs

**Template:**
```markdown
## Issue Description
[Brief description]

## System Information
- OS: Ubuntu 22.04
- Python: 3.9.7
- SUMO: 1.14.0

## Error Message
```
[Error output]
```

## Steps to Reproduce
1. cd SMART_Traffic_System
2. python src/main.py --config config/simulation_config.json
3. [Error occurs]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]
```

---

## Quick Fixes Checklist

Before asking for help, try:

- [ ] Check SUMO_HOME is set correctly
- [ ] Verify all files exist in correct locations
- [ ] Run from SMART_Traffic_System directory
- [ ] Check Python version (3.8+ required)
- [ ] Install all requirements: `pip install -r requirements.txt`
- [ ] Validate SUMO installation: `sumo --version`
- [ ] Check for typos in configuration files
- [ ] Review simulation.log for errors
- [ ] Try running without GUI: `gui=False`
- [ ] Reduce simulation duration for testing
- [ ] Check file permissions: `ls -l`

---

**Still having issues?** Check the project README.md and documentation in the `docs/` folder.
