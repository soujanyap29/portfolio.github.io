# Setup Guide for Smart Traffic Management System

This document provides detailed step-by-step instructions for setting up and configuring the Smart Traffic Management System.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [SUMO Installation](#sumo-installation)
3. [Python Environment Setup](#python-environment-setup)
4. [Map Preparation](#map-preparation)
5. [Configuration Files](#configuration-files)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Python**: 3.8 or higher

### Recommended Requirements
- **RAM**: 16 GB
- **Storage**: 20 GB free space
- **GPU**: For visualization (optional)

---

## SUMO Installation

### Ubuntu/Debian

```bash
# Add SUMO repository
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update

# Install SUMO and tools
sudo apt-get install sumo sumo-tools sumo-doc

# Set environment variable
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

### Windows

1. Download the Windows installer from [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php)
2. Run the installer and follow the prompts
3. Add SUMO to your system PATH:
   - Open System Properties > Advanced > Environment Variables
   - Add `C:\Program Files (x86)\Eclipse\Sumo\bin` to PATH
   - Add `SUMO_HOME` variable pointing to `C:\Program Files (x86)\Eclipse\Sumo`

### macOS

```bash
# Using Homebrew
brew tap dlr-ts/sumo
brew install sumo

# Set environment variable
echo 'export SUMO_HOME="/usr/local/opt/sumo/share/sumo"' >> ~/.zshrc
source ~/.zshrc
```

### Verify Installation

```bash
sumo --version
# Should output: Eclipse SUMO sumo Version X.X.X
```

---

## Python Environment Setup

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Python Setup

```python
# Test TraCI import
python -c "import traci; print('TraCI imported successfully')"
```

---

## Map Preparation

### Step 1: Download OpenStreetMap Data

#### Option A: Using the OSM Website
1. Go to [OpenStreetMap Export](https://www.openstreetmap.org/export)
2. Navigate to your desired area (e.g., Belagavi)
3. Select the area using the "Manually select a different area" option
4. Click "Export" to download the `.osm` file

#### Option B: Using Overpass API
```bash
# Download using wget (for Belagavi example)
wget -O maps/belagavi_map.osm "https://overpass-api.de/api/map?bbox=74.45,15.82,74.55,15.92"
```

#### Option C: Using JOSM
1. Download [JOSM](https://josm.openstreetmap.de/)
2. Download the area you need
3. Save as `.osm` file

### Step 2: Convert OSM to SUMO Network

```bash
# Basic conversion
python scripts/map_converter.py --input maps/your_map.osm --output sumo_config/

# With custom options
python scripts/map_converter.py \
    --input maps/belagavi_map.osm \
    --output sumo_config/ \
    --network-name belagavi \
    --keep-all-routes \
    --add-traffic-lights
```

### Step 3: Verify Network

```bash
# Open in SUMO GUI to verify
sumo-gui -n sumo_config/belagavi.net.xml
```

---

## Configuration Files

### Network Configuration (`.net.xml`)

Generated automatically by `netconvert`. Contains:
- Edges (roads)
- Lanes
- Junctions
- Traffic lights
- Connections

### Vehicle Types (`vehicles.vtype.xml`)

Define vehicle characteristics:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <vType id="car" accel="2.6" decel="4.5" length="4.5" maxSpeed="50"/>
    <vType id="bus" accel="1.2" decel="3.0" length="12.0" maxSpeed="40"/>
    <!-- More types... -->
</additional>
```

### Routes (`routes.rou.xml`)

Define vehicle flows and routes:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="flow_main" type="car" from="edge1" to="edge2" 
          begin="0" end="3600" probability="0.5"/>
    <!-- More flows... -->
</routes>
```

### Simulation Configuration (`simulation.sumocfg`)

Main configuration file:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="belagavi.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="additional.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>
</configuration>
```

---

## Verification

### 1. Test Network Loading

```bash
sumo-gui -c sumo_config/simulation.sumocfg
```

### 2. Test TraCI Connection

```bash
python -c "
import traci
import sumolib
print('All imports successful')
"
```

### 3. Run Test Simulation

```bash
python scripts/main.py --test-mode --duration 100
```

### 4. Check Output

Verify that output files are created in `data/simulation_logs/`

---

## Troubleshooting

### Common Issues

#### 1. "SUMO_HOME not set"

**Solution:**
```bash
# Linux/macOS
export SUMO_HOME="/usr/share/sumo"

# Windows (Command Prompt)
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

#### 2. "No module named 'traci'"

**Solution:**
```bash
# Add SUMO tools to Python path
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

Or install via pip:
```bash
pip install sumolib traci
```

#### 3. "Network file not found"

**Solution:**
- Verify file paths in `.sumocfg`
- Use absolute paths if relative paths fail
- Check file permissions

#### 4. "Invalid route" errors

**Solution:**
- Ensure edge IDs in routes match network edge IDs
- Use `randomTrips.py` to generate valid routes:
```bash
python $SUMO_HOME/tools/randomTrips.py -n network.net.xml -o trips.xml
```

#### 5. Slow simulation

**Solutions:**
- Reduce simulation step length
- Disable GUI mode
- Reduce vehicle count
- Use `--no-step-log` option

### Getting Help

1. Check [SUMO Documentation](https://sumo.dlr.de/docs/)
2. Visit [SUMO mailing list](https://eclipse.dev/sumo/contact/)
3. Open an issue in this repository

---

## Next Steps

After completing the setup:

1. Review the [Architecture Documentation](ARCHITECTURE.md)
2. Customize vehicle types and routes
3. Configure traffic light programs
4. Run simulations and analyze results
