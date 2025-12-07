"""
SUMO Simulation Controller with TraCI
Maps to: Operating Systems Course

Manages SUMO processes, TraCI communication, and real-time simulation control.
Demonstrates multi-process orchestration, IPC, and process scheduling.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path
import time


class SUMOConfig:
    """Configuration for SUMO simulation"""
    
    def __init__(self, network_file: str, route_file: str, 
                 gui: bool = False, step_length: float = 0.1):
        self.network_file = network_file
        self.route_file = route_file
        self.gui = gui
        self.step_length = step_length
        self.additional_files: List[str] = []
        self.output_files: Dict[str, str] = {}
    
    def to_xml(self, output_path: str) -> str:
        """Generate SUMO configuration XML file"""
        root = ET.Element('configuration')
        
        # Input section
        input_elem = ET.SubElement(root, 'input')
        ET.SubElement(input_elem, 'net-file', value=self.network_file)
        ET.SubElement(input_elem, 'route-files', value=self.route_file)
        
        if self.additional_files:
            ET.SubElement(input_elem, 'additional-files', 
                         value=','.join(self.additional_files))
        
        # Time section
        time_elem = ET.SubElement(root, 'time')
        ET.SubElement(time_elem, 'begin', value='0')
        ET.SubElement(time_elem, 'step-length', value=str(self.step_length))
        
        # Output section
        if self.output_files:
            output_elem = ET.SubElement(root, 'output')
            for key, value in self.output_files.items():
                ET.SubElement(output_elem, key, value=value)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return output_path


class VehicleTypeGenerator:
    """
    Generates SUMO vehicle type definitions
    Maps vehicle agent configurations to SUMO XML
    """
    
    @staticmethod
    def generate_vehicle_types_xml(output_path: str) -> str:
        """Generate SUMO vType definitions for all vehicle types"""
        root = ET.Element('additional')
        
        # Car
        car = ET.SubElement(root, 'vType')
        car.set('id', 'car')
        car.set('vClass', 'passenger')
        car.set('length', '4.5')
        car.set('width', '1.8')
        car.set('height', '1.5')
        car.set('maxSpeed', '33.3')
        car.set('accel', '2.6')
        car.set('decel', '4.5')
        car.set('sigma', '0.5')
        car.set('color', '1,0,0')  # Red
        
        # Bus
        bus = ET.SubElement(root, 'vType')
        bus.set('id', 'bus')
        bus.set('vClass', 'bus')
        bus.set('length', '12.0')
        bus.set('width', '2.5')
        bus.set('height', '3.2')
        bus.set('maxSpeed', '22.2')
        bus.set('accel', '1.2')
        bus.set('decel', '3.5')
        bus.set('color', '0,0,1')  # Blue
        
        # Truck
        truck = ET.SubElement(root, 'vType')
        truck.set('id', 'truck')
        truck.set('vClass', 'truck')
        truck.set('length', '16.5')
        truck.set('width', '2.6')
        truck.set('height', '4.0')
        truck.set('maxSpeed', '25.0')
        truck.set('accel', '1.0')
        truck.set('decel', '3.0')
        truck.set('color', '1,0.65,0')  # Orange
        
        # Motorcycle
        motorcycle = ET.SubElement(root, 'vType')
        motorcycle.set('id', 'motorcycle')
        motorcycle.set('vClass', 'motorcycle')
        motorcycle.set('length', '2.2')
        motorcycle.set('width', '0.8')
        motorcycle.set('height', '1.3')
        motorcycle.set('maxSpeed', '36.1')
        motorcycle.set('accel', '3.5')
        motorcycle.set('decel', '5.0')
        motorcycle.set('color', '0,1,0')  # Green
        
        # Bicycle
        bicycle = ET.SubElement(root, 'vType')
        bicycle.set('id', 'bicycle')
        bicycle.set('vClass', 'bicycle')
        bicycle.set('length', '1.8')
        bicycle.set('width', '0.6')
        bicycle.set('height', '1.1')
        bicycle.set('maxSpeed', '6.9')
        bicycle.set('accel', '1.5')
        bicycle.set('decel', '2.5')
        bicycle.set('color', '0,1,1')  # Cyan
        
        # Pedestrian
        pedestrian = ET.SubElement(root, 'vType')
        pedestrian.set('id', 'pedestrian')
        pedestrian.set('vClass', 'pedestrian')
        pedestrian.set('length', '0.6')
        pedestrian.set('width', '0.4')
        pedestrian.set('height', '1.7')
        pedestrian.set('maxSpeed', '1.4')
        pedestrian.set('color', '1,0,1')  # Magenta
        
        # Tram
        tram = ET.SubElement(root, 'vType')
        tram.set('id', 'tram')
        tram.set('vClass', 'rail')
        tram.set('length', '30.0')
        tram.set('width', '2.4')
        tram.set('height', '3.5')
        tram.set('maxSpeed', '19.4')
        tram.set('accel', '1.0')
        tram.set('decel', '2.5')
        tram.set('color', '0.5,0,0.5')  # Purple
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return output_path


class ProcessManager:
    """
    Manages SUMO process lifecycle
    Maps to: Operating Systems - Process management and IPC
    """
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.status: Dict[str, str] = {}
    
    def start_process(self, process_id: str, command: List[str], 
                     env: Optional[Dict] = None) -> bool:
        """
        Start a new process
        Demonstrates OS process creation and management
        """
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env or os.environ.copy()
            )
            
            self.processes[process_id] = process
            self.status[process_id] = "running"
            
            return True
        except Exception as e:
            print(f"Failed to start process {process_id}: {e}")
            self.status[process_id] = "failed"
            return False
    
    def stop_process(self, process_id: str, timeout: int = 5) -> bool:
        """
        Stop a running process gracefully
        Demonstrates process termination and cleanup
        """
        if process_id not in self.processes:
            return False
        
        process = self.processes[process_id]
        
        try:
            process.terminate()
            process.wait(timeout=timeout)
            self.status[process_id] = "stopped"
            return True
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            process.kill()
            process.wait()
            self.status[process_id] = "killed"
            return True
        except Exception as e:
            print(f"Error stopping process {process_id}: {e}")
            return False
    
    def get_status(self, process_id: str) -> Optional[str]:
        """Get process status"""
        return self.status.get(process_id)
    
    def is_running(self, process_id: str) -> bool:
        """Check if process is running"""
        if process_id not in self.processes:
            return False
        
        process = self.processes[process_id]
        return process.poll() is None
    
    def cleanup(self) -> None:
        """
        Clean up all processes
        Demonstrates resource cleanup and error recovery
        """
        for process_id in list(self.processes.keys()):
            if self.is_running(process_id):
                self.stop_process(process_id)


class TraCIController:
    """
    TraCI (Traffic Control Interface) Controller
    Manages real-time interaction with SUMO simulation
    Maps to: Operating Systems - Inter-Process Communication (IPC)
    """
    
    def __init__(self, port: int = 8813):
        self.port = port
        self.connected = False
        self.traci = None
    
    def connect(self, max_retries: int = 5) -> bool:
        """
        Connect to SUMO via TraCI
        Demonstrates IPC initialization with retry logic
        """
        try:
            import traci
            self.traci = traci
            
            for attempt in range(max_retries):
                try:
                    traci.init(self.port)
                    self.connected = True
                    print(f"Connected to SUMO on port {self.port}")
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Connection attempt {attempt + 1} failed, retrying...")
                        time.sleep(1)
                    else:
                        print(f"Failed to connect after {max_retries} attempts: {e}")
                        return False
        except ImportError:
            print("TraCI not available. Install SUMO and add to PYTHONPATH.")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from SUMO"""
        if self.connected and self.traci:
            try:
                self.traci.close()
                self.connected = False
            except Exception as e:
                print(f"Error disconnecting: {e}")
    
    def simulation_step(self) -> bool:
        """
        Execute one simulation step
        Demonstrates synchronized process communication
        """
        if not self.connected:
            return False
        
        try:
            self.traci.simulationStep()
            return True
        except Exception as e:
            print(f"Error in simulation step: {e}")
            return False
    
    def get_vehicle_ids(self) -> List[str]:
        """Get list of all vehicle IDs in simulation"""
        if not self.connected:
            return []
        
        try:
            return self.traci.vehicle.getIDList()
        except Exception:
            return []
    
    def get_vehicle_position(self, vehicle_id: str) -> Optional[Tuple[float, float]]:
        """Get vehicle position"""
        if not self.connected:
            return None
        
        try:
            return self.traci.vehicle.getPosition(vehicle_id)
        except Exception:
            return None
    
    def get_vehicle_speed(self, vehicle_id: str) -> Optional[float]:
        """Get vehicle speed"""
        if not self.connected:
            return None
        
        try:
            return self.traci.vehicle.getSpeed(vehicle_id)
        except Exception:
            return None
    
    def set_vehicle_speed(self, vehicle_id: str, speed: float) -> bool:
        """
        Set vehicle speed (control action)
        Demonstrates real-time process control via IPC
        """
        if not self.connected:
            return False
        
        try:
            self.traci.vehicle.setSpeed(vehicle_id, speed)
            return True
        except Exception as e:
            print(f"Error setting speed for {vehicle_id}: {e}")
            return False
    
    def change_vehicle_route(self, vehicle_id: str, edge_list: List[str]) -> bool:
        """Change vehicle route dynamically"""
        if not self.connected:
            return False
        
        try:
            self.traci.vehicle.setRoute(vehicle_id, edge_list)
            return True
        except Exception as e:
            print(f"Error changing route for {vehicle_id}: {e}")
            return False


class SimulationController:
    """
    Main simulation controller
    Orchestrates SUMO process, TraCI connection, and simulation execution
    """
    
    def __init__(self, config: SUMOConfig):
        self.config = config
        self.process_manager = ProcessManager()
        self.traci_controller = TraCIController()
        self.simulation_time: float = 0.0
        self.running: bool = False
    
    def start(self) -> bool:
        """
        Start simulation with process orchestration
        Maps to: OS - Multi-process coordination
        """
        # Generate configuration file
        config_path = "simulation.sumocfg"
        self.config.to_xml(config_path)
        
        # Determine SUMO binary
        sumo_binary = "sumo-gui" if self.config.gui else "sumo"
        
        # Build command
        command = [
            sumo_binary,
            "-c", config_path,
            "--remote-port", str(self.traci_controller.port),
            "--step-length", str(self.config.step_length)
        ]
        
        # Start SUMO process
        success = self.process_manager.start_process("sumo", command)
        if not success:
            return False
        
        # Wait for SUMO to initialize
        time.sleep(2)
        
        # Connect via TraCI
        if not self.traci_controller.connect():
            self.process_manager.stop_process("sumo")
            return False
        
        self.running = True
        return True
    
    def step(self) -> bool:
        """Execute one simulation step"""
        if not self.running:
            return False
        
        if self.traci_controller.simulation_step():
            self.simulation_time += self.config.step_length
            return True
        return False
    
    def stop(self) -> None:
        """
        Stop simulation with cleanup
        Demonstrates graceful shutdown and resource cleanup
        """
        self.running = False
        self.traci_controller.disconnect()
        self.process_manager.cleanup()
    
    def get_simulation_time(self) -> float:
        """Get current simulation time"""
        return self.simulation_time
    
    def get_all_vehicles(self) -> List[str]:
        """Get all vehicle IDs"""
        return self.traci_controller.get_vehicle_ids()
    
    def get_vehicle_data(self, vehicle_id: str) -> Optional[Dict]:
        """Get comprehensive vehicle data"""
        position = self.traci_controller.get_vehicle_position(vehicle_id)
        speed = self.traci_controller.get_vehicle_speed(vehicle_id)
        
        if position is None or speed is None:
            return None
        
        return {
            'id': vehicle_id,
            'position': position,
            'speed': speed,
            'time': self.simulation_time
        }
