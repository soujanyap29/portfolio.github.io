# Portfolio - Soujanya P

## Smart Traffic Management System

This repository contains a comprehensive **Smart Traffic Management System** implementation - a full-stack multi-agent traffic simulation system suitable for peer-reviewed journal publication.

### 🚦 Project Overview

The Smart Traffic Management System is a high-fidelity, real-world vehicle modeling and simulation platform that integrates:

- **OSM (OpenStreetMap)** integration for real-world city networks
- **SUMO (Simulation of Urban MObility)** traffic simulation
- **TraCI** for real-time simulation control
- **V2V (Vehicle-to-Vehicle)** communication
- **V2I (Vehicle-to-Infrastructure)** communication
- **Social IoT (SIoT)** trust-based networking
- **Live Analytics Dashboard**
- **Comprehensive Database System**

### 📊 System Features

#### Supported Vehicle Types
- 🚗 Cars (Passenger vehicles)
- 🚌 Buses (Public transit)
- 🚚 Trucks (Freight)
- 🏍️ Motorcycles
- 🚲 Bicycles
- 🚶 Pedestrians
- 🚊 Trams
- 🛺 Auto Rickshaws

#### Core Technologies
- **Backend**: Python 3.8+
- **Simulation**: SUMO 1.15+ with TraCI
- **Database**: SQLAlchemy with PostgreSQL/SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Communication**: Custom V2V/V2I protocol implementation

### 🏗️ Architecture

The system is built with strong Computer Science foundations mapping to core courses:

| CS Course | Implementation |
|-----------|----------------|
| **Object-Oriented Programming** | Vehicle agent class hierarchy, factory patterns |
| **Operating Systems** | Multi-process orchestration, TraCI IPC, resource scheduling |
| **Computer Networks** | V2V/V2I protocols, latency simulation, trust networking |
| **Compiler Design** | Message validation FSM, protocol rule engine |
| **Data Structures & Algorithms** | Graph-based routing, trust propagation, event queues |
| **Database Management** | Schema design, ETL pipelines, analytics queries |

### 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/soujanyap29/portfolio.github.io.git
cd portfolio.github.io/smart-traffic-system

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py

# Open the dashboard
# Navigate to frontend/dashboard.html in your browser
```

### 📁 Project Structure

```
smart-traffic-system/
├── backend/
│   ├── vehicle_agents.py      # OOP vehicle class hierarchy
│   └── communication.py        # V2V/V2I communication module
├── simulation/
│   └── sumo_controller.py      # SUMO/TraCI integration
├── database/
│   └── schema.py               # Database schema and ETL
├── frontend/
│   └── dashboard.html          # Live analytics dashboard
├── docs/
│   └── DOCUMENTATION_INDEX.md  # 100-page documentation outline
├── requirements.txt            # Python dependencies
├── main.py                     # Main integration script
└── README.md                   # This file
```

### 📸 Screenshots

#### Dashboard - Initial State
![Dashboard Initial](https://github.com/user-attachments/assets/dc66c3e2-f752-4290-9a69-8a1726cdbe13)

#### Dashboard - Running Simulation
![Dashboard Running](https://github.com/user-attachments/assets/5f0563cd-a33b-4813-a02e-6367eb63b329)

### 📖 Documentation

Complete 100-page journal-quality documentation is available in the `docs/` directory, organized by four author contributions:

- **Pages 1-25** (Soujanya Patil): System Architecture & Foundation
- **Pages 26-50** (Soujanya Poojari): Algorithms & Communication Protocols
- **Pages 51-75** (Anushka): UI/UX & Simulation Scenarios
- **Pages 76-100** (Apoorva): Database, Testing & Deployment

See `docs/DOCUMENTATION_INDEX.md` for the complete table of contents.

### 🔬 Research Contributions

This system advances the state-of-the-art in:

1. **High-Fidelity Vehicle Modeling**: Accurate representation of diverse urban vehicles
2. **Trust-Based Communication**: Social IoT integration in V2V/V2I networks
3. **Real-Time Visualization**: Live dashboard with per-vehicle-type analytics
4. **Comprehensive CS Integration**: Explicit mapping to core CS subjects
5. **Scalable Architecture**: City-scale simulation capability

### 🧪 Testing

```bash
# Run unit tests (when available)
pytest tests/

# Run simulation with test scenario
python main.py --scenario test_city
```

### 📊 Sample Output

```
============================================================
Initializing Smart Traffic Management System
Simulation ID: sim_20251207_105530
Scenario: city_center
============================================================

✓ Created 100 car vehicles
✓ Created 10 bus vehicles
✓ Created 15 truck vehicles
✓ Created 30 motorcycle vehicles
✓ Created 20 bicycle vehicles
✓ Created 50 pedestrian vehicles

Total vehicles created: 225

Simulation Summary:
  - Total Messages: 5
  - Success Rate: 100.00%
  - Avg Latency: 10.00 ms
```

### 👥 Authors

- **Soujanya Patil** - Architecture & System Design
- **Soujanya Poojari** - Algorithms & Protocols
- **Anushka** - UI/UX & Scenarios
- **Apoorva** - Database & Testing

### 📄 License

MIT License - See LICENSE file for details

### 🤝 Contributing

This project is part of academic research. For collaboration inquiries, please open an issue.

### 📧 Contact

For questions or collaboration opportunities:
- Repository: [github.com/soujanyap29/portfolio.github.io](https://github.com/soujanyap29/portfolio.github.io)
- Issues: [Open an issue](https://github.com/soujanyap29/portfolio.github.io/issues)

---

**Note**: This project requires SUMO (Simulation of Urban MObility) to be installed for full functionality. The system is designed for research and educational purposes, demonstrating advanced concepts in traffic simulation, multi-agent systems, and distributed communication.
