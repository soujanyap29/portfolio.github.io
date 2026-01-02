"""
Utility Functions for SMART Traffic System
Common helper functions used across modules
"""

import os
import sys
import math
import json
from datetime import datetime


def format_time(seconds):
    """
    Format seconds into HH:MM:SS string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1 (tuple): First position (x, y)
        pos2 (tuple): Second position (x, y)
    
    Returns:
        float: Distance in meters
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def ensure_dir(directory):
    """
    Ensure directory exists, create if not.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_json(filepath):
    """
    Load JSON file.
    
    Args:
        filepath (str): Path to JSON file
    
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data, filepath):
    """
    Save data to JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def get_timestamp():
    """
    Get current timestamp as string.
    
    Returns:
        str: Timestamp in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def print_header(text, width=60):
    """
    Print formatted header.
    
    Args:
        text (str): Header text
        width (int): Header width
    """
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def print_section(text, width=60):
    """
    Print formatted section.
    
    Args:
        text (str): Section text
        width (int): Section width
    """
    print("\n" + "-"*width)
    print(text)
    print("-"*width)


def validate_sumo_installation():
    """
    Validate SUMO installation and environment.
    
    Returns:
        bool: True if SUMO is properly installed
    """
    if 'SUMO_HOME' not in os.environ:
        print("ERROR: SUMO_HOME environment variable not set")
        print("Please install SUMO and set SUMO_HOME")
        return False
    
    sumo_home = os.environ['SUMO_HOME']
    if not os.path.exists(sumo_home):
        print(f"ERROR: SUMO_HOME directory does not exist: {sumo_home}")
        return False
    
    # Check for required executables
    tools_dir = os.path.join(sumo_home, 'tools')
    if not os.path.exists(tools_dir):
        print(f"ERROR: SUMO tools directory not found: {tools_dir}")
        return False
    
    print("✓ SUMO installation validated")
    print(f"  SUMO_HOME: {sumo_home}")
    return True


def get_lane_index_from_id(lane_id):
    """
    Extract lane index from lane ID.
    
    Args:
        lane_id (str): Lane ID (e.g., "edge_0_1")
    
    Returns:
        int: Lane index
    """
    try:
        return int(lane_id.split('_')[-1])
    except:
        return 0


def get_edge_from_lane(lane_id):
    """
    Extract edge ID from lane ID.
    
    Args:
        lane_id (str): Lane ID
    
    Returns:
        str: Edge ID
    """
    parts = lane_id.split('_')
    return '_'.join(parts[:-1]) if len(parts) > 1 else lane_id


def mps_to_kmh(speed_mps):
    """
    Convert meters per second to kilometers per hour.
    
    Args:
        speed_mps (float): Speed in m/s
    
    Returns:
        float: Speed in km/h
    """
    return speed_mps * 3.6


def kmh_to_mps(speed_kmh):
    """
    Convert kilometers per hour to meters per second.
    
    Args:
        speed_kmh (float): Speed in km/h
    
    Returns:
        float: Speed in m/s
    """
    return speed_kmh / 3.6


class ProgressBar:
    """Simple progress bar for console output."""
    
    def __init__(self, total, prefix='Progress:', length=50):
        """
        Initialize progress bar.
        
        Args:
            total (int): Total number of steps
            prefix (str): Prefix text
            length (int): Bar length in characters
        """
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
    
    def update(self, current):
        """
        Update progress bar.
        
        Args:
            current (int): Current step
        """
        self.current = current
        percent = int((current / self.total) * 100)
        filled = int(self.length * current // self.total)
        bar = '█' * filled + '-' * (self.length - filled)
        
        print(f'\r{self.prefix} |{bar}| {percent}% ({current}/{self.total})', end='')
        
        if current >= self.total:
            print()  # New line when complete


def summarize_dict(data, max_items=5):
    """
    Create summary of dictionary for display.
    
    Args:
        data (dict): Dictionary to summarize
        max_items (int): Maximum items to show
    
    Returns:
        str: Formatted summary
    """
    if not data:
        return "Empty"
    
    items = list(data.items())[:max_items]
    summary = "\n".join([f"  {k}: {v}" for k, v in items])
    
    if len(data) > max_items:
        summary += f"\n  ... ({len(data) - max_items} more items)"
    
    return summary
