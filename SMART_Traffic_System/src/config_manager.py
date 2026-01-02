"""
Configuration Manager for SMART Traffic System
Handles loading and managing configuration files
"""

import json
import os
from pathlib import Path


class ConfigManager:
    """
    Manages configuration files for the simulation.
    
    Supports:
    - JSON configuration files
    - Default values
    - Configuration validation
    - Parameter override
    """
    
    DEFAULT_CONFIG = {
        'simulation': {
            'step_length': 0.1,
            'total_time': 3600,
            'gui_enabled': False,
            'real_time_factor': 1.0
        },
        'traffic': {
            'vehicle_density': 'medium',
            'peak_hour_multiplier': 1.5,
            'emergency_vehicle_probability': 0.01
        },
        'v2x': {
            'communication_range': 300,
            'message_frequency': 10,
            'broadcast_enabled': True
        },
        'signal_control': {
            'adaptive_enabled': True,
            'min_green_time': 10,
            'max_green_time': 90,
            'yellow_time': 3,
            'all_red_time': 2
        },
        'emergency': {
            'detection_range': 500,
            'priority_duration': 120,
            'lane_clearance_time': 15
        }
    }
    
    def __init__(self, config_file=None):
        """
        Initialize configuration manager.
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            print(f"Using default configuration")
    
    def load_config(self, config_file):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str): Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            
            # Merge with defaults
            self._deep_merge(self.config, user_config)
            print(f"✓ Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    
    def _deep_merge(self, base, update):
        """
        Deep merge two dictionaries.
        
        Args:
            base (dict): Base dictionary
            update (dict): Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get_config(self):
        """
        Get current configuration.
        
        Returns:
            dict: Configuration dictionary
        """
        return self.config
    
    def get(self, *keys, default=None):
        """
        Get configuration value by nested keys.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys, value):
        """
        Set configuration value by nested keys.
        
        Args:
            *keys: Nested keys to access
            value: Value to set
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save_config(self, output_file):
        """
        Save current configuration to file.
        
        Args:
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"✓ Configuration saved to {output_file}")
