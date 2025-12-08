# Installation and Setup Guide
# Smart Traffic Management System

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space
- **Python**: 3.8 or higher
- **SUMO**: 1.16.0 or higher

### Required Software

1. **SUMO (Simulation of Urban MObility)**
2. **Python 3.8+**
3. **NS3 (Optional, for network simulation)**
4. **SQLite3**

---

## Installation Steps

### 1. Install SUMO

#### Ubuntu/Linux
```bash
# Add SUMO repository
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update

# Install SUMO and tools
sudo apt-get install sumo sumo-tools sumo-doc

# Set environment variable
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
sumo --version
```

#### Windows
```powershell
# Download installer from https://sumo.dlr.de/docs/Downloads.php
# Run installer and follow instructions

# Add to PATH (adjust version as needed)
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
setx PATH "%PATH%;%SUMO_HOME%\bin"

# Verify installation
sumo --version
```

#### macOS
```bash
# Using Homebrew
brew tap dlr-ts/sumo
brew install sumo

# Set environment variable
echo 'export SUMO_HOME="/usr/local/opt/sumo/share/sumo"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
sumo --version
```

---

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv traffic_env
source traffic_env/bin/activate  # On Windows: traffic_env\Scripts\activate

# Install required packages
pip install traci
pip install numpy
pip install pandas
pip install matplotlib
pip install sqlite3  # Usually included with Python

# Verify installation
python -c "import traci; print('TraCI installed successfully')"
```

---

### 3. Install NS3 (Optional)

NS3 is required for detailed network simulation. Skip if only running SUMO-based simulations.

#### Ubuntu/Linux
```bash
# Install dependencies
sudo apt-get install gcc g++ python3 cmake ninja-build git

# Download NS3
cd /opt
sudo git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev

# Configure and build
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Set environment variable
echo 'export NS3_HOME="/opt/ns-3-dev"' >> ~/.bashrc
source ~/.bashrc
```

---

### 4. Clone Repository

```bash
# Clone the project
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io/smart-traffic-system

# Verify structure
ls -la
```

---

### 5. Initialize Database

```bash
# Navigate to database directory
cd database/schemas

# Initialize databases
sqlite3 ../traffic_events.db < complete_schema.sql
sqlite3 ../v2x_communication.db < complete_schema.sql
sqlite3 ../siot_trust.db < complete_schema.sql
sqlite3 ../emergency_events.db < complete_schema.sql

# Verify creation
ls -lh ../
```

---

## Configuration

### 1. Edit Configuration File

```bash
cd configs
nano scenario_config.json
```

Adjust parameters as needed:
- `duration`: Simulation time in seconds
- `communication_range`: V2X range in meters
- `use_gui`: Enable/disable SUMO GUI
- `rsu_positions`: Roadside unit locations

### 2. Validate SUMO Files

```bash
# Validate network file
netconvert --sumo-net-file sumo/networks/city_network.net.xml --plain-output-prefix test

# Validate route file
duarouter -n sumo/networks/city_network.net.xml -r sumo/routes/vehicles.rou.xml --no-warnings

# Validate complete configuration
sumo -c sumo/scenarios/basic_traffic.sumocfg --no-step-log --duration-log.disable
```

---

## Running Simulations

### Basic Simulation

```bash
# Run without GUI (faster)
python run_simulation.py --config configs/scenario_config.json --duration 3600

# Run with GUI (for visualization)
python run_simulation.py --config configs/scenario_config.json --duration 3600 --gui
```

### Component-Specific Runs

#### Adaptive Signals Only
```bash
cd python/core
python adaptive_signals.py ../../sumo/scenarios/basic_traffic.sumocfg
```

#### Emergency Priority Only
```bash
cd python/core
python emergency_priority.py ../../sumo/scenarios/basic_traffic.sumocfg
```

#### V2X Communication
```bash
cd python/core
python v2x_communication.py
```

---

## Running Analytics

### Generate Reports

```bash
# Comparative analysis
cd analytics
python comparative_analysis.py

# Check output
cat ../logs/comparative_analysis.json
```

### Run SQL Queries

```bash
# Connect to database
sqlite3 database/traffic_events.db

# Run sample queries
.read database/queries/analytics_queries.sql

# Export results
.mode csv
.output results.csv
SELECT * FROM avg_metrics_by_type;
.quit
```

---

## Troubleshooting

### SUMO Not Found

```bash
# Check SUMO_HOME
echo $SUMO_HOME

# If empty, set it
export SUMO_HOME="/usr/share/sumo"  # Adjust path as needed
```

### TraCI Connection Error

```bash
# Check if SUMO port is available
netstat -tuln | grep 8813

# Kill existing SUMO processes
killall sumo
killall sumo-gui
```

### Database Locked Error

```bash
# Close all database connections
# Restart Python interpreter
# Check for orphaned processes
ps aux | grep python
```

---

## Verification

### Test Installation

```bash
# Run test script
python tests/test_installation.py

# Expected output:
# ✓ SUMO installed
# ✓ Python dependencies installed
# ✓ Databases initialized
# ✓ Configuration valid
# All tests passed!
```

---

## Quick Start Example

```bash
# Complete workflow
cd /path/to/smart-traffic-system

# 1. Check prerequisites
python --version  # Should be 3.8+
sumo --version    # Should be 1.16+

# 2. Initialize environment
source traffic_env/bin/activate

# 3. Run basic simulation
python run_simulation.py --duration 300

# 4. View results
cat logs/summary.txt
sqlite3 database/traffic_events.db "SELECT COUNT(*) FROM vehicle_logs;"

# 5. Generate report
cd analytics
python comparative_analysis.py
```

---

## Directory Structure

```
smart-traffic-system/
├── sumo/              # SUMO configuration files
│   ├── networks/      # Road network definitions
│   ├── routes/        # Vehicle routes
│   ├── signals/       # Traffic signal configs
│   └── scenarios/     # Complete scenarios
├── python/            # Python control scripts
│   ├── core/          # Main modules
│   └── utils/         # Utility functions
├── database/          # SQLite databases
│   ├── schemas/       # Database schemas
│   └── queries/       # SQL queries
├── logs/              # Simulation output
├── analytics/         # Analysis scripts
├── docs/              # Documentation
├── configs/           # Configuration files
└── run_simulation.py  # Main entry point
```

---

## Additional Resources

- **SUMO Documentation**: https://sumo.dlr.de/docs/
- **TraCI Documentation**: https://sumo.dlr.de/docs/TraCI.html
- **NS3 Documentation**: https://www.nsnam.org/documentation/
- **Project Wiki**: [Link to project wiki]

---

## Support

For issues and questions:
- Open an issue on GitHub
- Email: [contact information]
- Documentation: See `docs/` directory

---

## License

This project is for academic and research purposes.

---

*Installation Guide v1.0*
*Smart Traffic Management System*
