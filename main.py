#!/usr/bin/env python3
"""
Marine Vessel Simulation System
Main entry point for running simulations
"""

import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress matplotlib backend warning
warnings.filterwarnings('ignore', category=UserWarning, message='The PostScript backend does not support transparency')

from src.core.simulation import run_simulation

if __name__ == "__main__":
    run_simulation()