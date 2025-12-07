# Installation Guide

## Prerequisites

### Required Software
- **Python 3.8 or higher**
- **Git** (for cloning the repository)

### Optional (for full SUMO integration)
- **SUMO 1.15+** - Simulation of Urban MObility
  - Download from: https://www.eclipse.org/sumo/
  - Installation guide: https://sumo.dlr.de/docs/Installing/index.html
- **PostgreSQL 13+** (optional, SQLite is used by default)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io/smart-traffic-system
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
# Install basic dependencies (simulation will work without SUMO)
pip install numpy pandas matplotlib networkx

# For full database support (optional):
pip install sqlalchemy psycopg2-binary

# Note: The requirements.txt includes optional dependencies
# Install only what you need for your use case
```

### 4. Verify Installation

```bash
# Run the simulation without SUMO
python main.py
```

Expected output:
```
============================================================
Initializing Smart Traffic Management System
Simulation ID: sim_YYYYMMDD_HHMMSS
Scenario: city_center
============================================================

✓ Created 100 car vehicles
✓ Created 10 bus vehicles
...
Total vehicles created: 225
```

### 5. View the Dashboard

Open your web browser and navigate to:
```
file:///path/to/smart-traffic-system/frontend/dashboard.html
```

Or use a simple HTTP server:
```bash
cd frontend
python3 -m http.server 8000
# Then open: http://localhost:8000/dashboard.html
```

## Optional: SUMO Integration

### Install SUMO

#### Ubuntu/Debian
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

#### macOS (using Homebrew)
```bash
brew install sumo
```

#### Windows
Download and install from: https://sumo.dlr.de/docs/Downloads.php

### Configure SUMO_HOME

Add SUMO to your environment:

```bash
# Linux/Mac (add to ~/.bashrc or ~/.zshrc)
export SUMO_HOME="/usr/share/sumo"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# Windows
setx SUMO_HOME "C:\Program Files\SUMO"
setx PYTHONPATH "%SUMO_HOME%\tools;%PYTHONPATH%"
```

### Install TraCI

```bash
pip install traci sumolib
```

### Test SUMO Integration

```bash
# This will fail gracefully if SUMO is not installed
python simulation/sumo_controller.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Install the missing package with `pip install xxx`

#### 2. SUMO Not Found
```
TraCI not available. Install SUMO and add to PYTHONPATH.
```
**Solution**: 
- Install SUMO following the steps above
- Verify SUMO_HOME is set correctly
- Restart your terminal/IDE after setting environment variables

#### 3. Database Connection Issues
```
Failed to create database directory
```
**Solution**: Ensure write permissions in the project directory or specify a custom database path

#### 4. Dashboard Not Loading
**Solution**: Use a simple HTTP server instead of opening the file directly:
```bash
cd frontend
python3 -m http.server 8000
```

### Getting Help

- Check the [README.md](../README.md) for overview
- Review the [documentation](docs/DOCUMENTATION_INDEX.md)
- Open an issue on GitHub

## System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4 GB
- Disk: 500 MB free space
- OS: Linux, macOS, or Windows

### Recommended (for city-scale simulation)
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 2 GB free space
- OS: Linux (best performance)

## Next Steps

After installation:

1. **Run Basic Simulation**: `python main.py`
2. **View Dashboard**: Open `frontend/dashboard.html`
3. **Explore Documentation**: Check `docs/DOCUMENTATION_INDEX.md`
4. **Customize Scenarios**: Edit configuration in `main.py`
5. **Add Vehicle Types**: Extend classes in `backend/vehicle_agents.py`

## Development Setup

For contributing to the project:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Format code
black backend/ simulation/ database/

# Lint code
flake8 backend/ simulation/ database/
```

## Docker Setup (Future)

Docker support is planned for easier deployment:

```bash
# Build image
docker build -t smart-traffic-system .

# Run container
docker run -p 8000:8000 smart-traffic-system
```

## Support

For issues, questions, or contributions:
- **Issues**: https://github.com/soujanyap29/portfolio.github.io/issues
- **Documentation**: See `docs/` directory
- **Examples**: Check `examples/` directory (when available)
