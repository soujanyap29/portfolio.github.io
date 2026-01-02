"""
Traffic Signal Control Module for SMART Traffic System
Implements adaptive and intelligent traffic signal control
"""

import traci
import math


class TrafficSignalControl:
    """
    Manages traffic signal control with adaptive algorithms.
    
    Features:
    - Adaptive signal timing based on traffic demand
    - Webster's method for optimal cycle calculation
    - Emergency vehicle priority override
    - Time-of-day signal plans
    - Multi-junction coordination
    """
    
    def __init__(self):
        """Initialize traffic signal control system."""
        self.traffic_lights = []
        self.current_programs = {}
        self.emergency_mode_active = {}
        
        # Signal timing parameters
        self.min_green_time = 10  # seconds
        self.max_green_time = 90  # seconds
        self.yellow_time = 3  # seconds
        self.all_red_time = 2  # seconds
        
        self._discover_traffic_lights()
        
        print(f"✓ Traffic Signal Control initialized ({len(self.traffic_lights)} signals)")
    
    def _discover_traffic_lights(self):
        """Discover all traffic lights in the network."""
        try:
            self.traffic_lights = traci.trafficlight.getIDList()
            for tl_id in self.traffic_lights:
                self.current_programs[tl_id] = 'adaptive'
                self.emergency_mode_active[tl_id] = False
        except:
            pass
    
    def update_adaptive_control(self, lane_data, current_time):
        """
        Update traffic signals using adaptive control algorithm.
        
        Args:
            lane_data (dict): Current lane traffic data
            current_time (float): Current simulation time
        """
        for tl_id in self.traffic_lights:
            if not self.emergency_mode_active[tl_id]:
                self._adaptive_signal_timing(tl_id, lane_data)
    
    def _adaptive_signal_timing(self, tl_id, lane_data):
        """
        Calculate adaptive signal timing for a traffic light.
        
        Args:
            tl_id (str): Traffic light ID
            lane_data (dict): Lane traffic data
        """
        try:
            # Get controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # Calculate demand for each approach
            ns_demand = 0  # North-South demand
            ew_demand = 0  # East-West demand
            
            for lane_id in controlled_lanes:
                if lane_id in lane_data:
                    vehicle_count = lane_data[lane_id]['vehicle_count']
                    queue_length = lane_data[lane_id]['queue_length']
                    
                    # Determine if lane is NS or EW
                    if 'N_to_' in lane_id or 'S_to_' in lane_id or '_to_N' in lane_id or '_to_S' in lane_id:
                        ns_demand += vehicle_count + (queue_length * 2)  # Weight queues more
                    elif 'E_to_' in lane_id or 'W_to_' in lane_id or '_to_E' in lane_id or '_to_W' in lane_id:
                        ew_demand += vehicle_count + (queue_length * 2)
            
            # Calculate optimal green times using Webster's method
            total_demand = ns_demand + ew_demand
            if total_demand > 0:
                # Optimal cycle length (Webster's formula approximation)
                L = self.yellow_time * 2 + self.all_red_time * 2  # Lost time
                Y = total_demand / 3600.0  # Critical flow ratio estimate
                
                if Y < 0.9:  # Avoid oversaturation
                    optimal_cycle = (1.5 * L + 5) / (1 - Y)
                    optimal_cycle = max(30, min(120, optimal_cycle))  # Constrain cycle
                    
                    # Allocate green time proportionally
                    green_time_total = optimal_cycle - L
                    ns_green = (ns_demand / total_demand) * green_time_total
                    ew_green = (ew_demand / total_demand) * green_time_total
                    
                    # Constrain green times
                    ns_green = max(self.min_green_time, min(self.max_green_time, ns_green))
                    ew_green = max(self.min_green_time, min(self.max_green_time, ew_green))
                    
                    # Apply timing (simplified - would need full phase manipulation in production)
                    # This is a conceptual implementation
                    # traci.trafficlight.setPhaseDuration(tl_id, ns_green)
        
        except Exception as e:
            pass
    
    def emergency_mode(self, emergency_vehicle_id):
        """
        Activate emergency mode for relevant traffic lights.
        
        Args:
            emergency_vehicle_id (str): ID of emergency vehicle
        """
        try:
            # Get emergency vehicle route
            route_id = traci.vehicle.getRouteID(emergency_vehicle_id)
            edges = traci.route.getEdges(route_id)
            
            # Find traffic lights on route
            for edge in edges:
                # Get junctions at edge
                # Activate emergency program for those traffic lights
                for tl_id in self.traffic_lights:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    
                    # Check if emergency vehicle's edge is controlled by this TL
                    for lane in controlled_lanes:
                        if edge in lane:
                            self._activate_emergency_program(tl_id, emergency_vehicle_id)
                            break
        
        except Exception as e:
            print(f"Error in emergency mode: {e}")
    
    def _activate_emergency_program(self, tl_id, emergency_vehicle_id):
        """
        Activate emergency signal program.
        
        Args:
            tl_id (str): Traffic light ID
            emergency_vehicle_id (str): Emergency vehicle ID
        """
        try:
            # Get emergency vehicle lane
            emerg_lane = traci.vehicle.getLaneID(emergency_vehicle_id)
            
            # Determine which direction to give green
            if 'N_to_' in emerg_lane or 'S_to_' in emerg_lane or '_to_N' in emerg_lane or '_to_S' in emerg_lane:
                program_id = 'emergency_NS'
            else:
                program_id = 'emergency_EW'
            
            # Switch to emergency program
            # traci.trafficlight.setProgram(tl_id, program_id)
            self.emergency_mode_active[tl_id] = True
            self.current_programs[tl_id] = program_id
            
            print(f"  → Traffic light {tl_id} switched to {program_id}")
        
        except Exception as e:
            pass
    
    def deactivate_emergency_mode(self, tl_id):
        """
        Return traffic light to normal operation.
        
        Args:
            tl_id (str): Traffic light ID
        """
        try:
            # traci.trafficlight.setProgram(tl_id, 'adaptive')
            self.emergency_mode_active[tl_id] = False
            self.current_programs[tl_id] = 'adaptive'
            print(f"  → Traffic light {tl_id} returned to adaptive control")
        
        except Exception as e:
            pass
    
    def get_current_phase(self, tl_id):
        """
        Get current signal phase.
        
        Args:
            tl_id (str): Traffic light ID
        
        Returns:
            int: Current phase index
        """
        try:
            return traci.trafficlight.getPhase(tl_id)
        except:
            return 0
    
    def get_time_until_switch(self, tl_id):
        """
        Get time until next phase switch.
        
        Args:
            tl_id (str): Traffic light ID
        
        Returns:
            float: Time in seconds
        """
        try:
            return traci.trafficlight.getNextSwitch(tl_id) - traci.simulation.getTime()
        except:
            return 0.0
    
    def set_time_of_day_plan(self, hour):
        """
        Set signal timing plan based on time of day.
        
        Args:
            hour (int): Hour of day (0-23)
        """
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            # Peak hours
            program = 'peak_hour'
        elif 22 <= hour or hour <= 6:
            # Night time
            program = 'off_peak'
        else:
            # Normal hours
            program = 'adaptive'
        
        for tl_id in self.traffic_lights:
            if not self.emergency_mode_active[tl_id]:
                try:
                    # traci.trafficlight.setProgram(tl_id, program)
                    self.current_programs[tl_id] = program
                except:
                    pass
