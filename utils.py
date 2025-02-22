
# utils.py
# --------
# This module provides utility functions that are used throughout the trading bot application. Functions include:
#   - load_tick_sizes and save_tick_sizes: Managing the persistent storage of tick size data in a JSON file
#   - round_to_tick: Rounding numeric price values to the nearest valid tick size for precision in order execution
# These helper functions promote code reuse and ensure consistent handling of common operations across the project.

import os
import json

TICK_SIZE_FILE = 'tick_sizes.json'

def load_tick_sizes():
    """
    Load tick sizes from the persistent JSON file.
    """
    if os.path.exists(TICK_SIZE_FILE):
        with open(TICK_SIZE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_tick_sizes(tick_sizes):
    """
    Save tick sizes to the persistent JSON file.
    """
    with open(TICK_SIZE_FILE, 'w') as f:
        json.dump(tick_sizes, f)

def round_to_tick(value, tick_size):
    """
    Round the given value to the nearest tick size.
    """
    rounded_value = round(value / tick_size) * tick_size
    # Ensure precision up to 8 decimal places, important for low-priced stocks
    return float(f"{rounded_value:.8f}")
