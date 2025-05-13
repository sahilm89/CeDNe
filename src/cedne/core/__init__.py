"""
Core module for CeDNe.

This module contains the fundamental classes and functions for the CeDNe package.
"""

# Organism containers
from .animal import Animal, Worm, Fly

# Neurons and cell primitives
from .neuron import Cell, Neuron, NeuronGroup

# Network graph and system-level logic
from .network import NervousSystem

# Connections and edge logic
from .connection import (
    Connection, ChemicalSynapse, GapJunction, BulkConnection,
    ConnectionGroup, Path
)

# Experimental trial recordings and features
from .recordings import Trial, StimResponse

# Utilities and constants
from .config import F_SAMPLE, RANDOM_SEED, RECURSION_LIMIT
from .io import load_pickle, load_worm, generate_random_string
from .logger import logger
from .mapping import GraphMap
from .behavior import Behavior

__all__ = [
    # Organisms
    "Animal", "Worm", "Fly",
    
    # Neuron model
    "Cell", "Neuron", "NeuronGroup",
    
    # Network model
    "NervousSystem",
    
    # Connections
    "Connection", "ChemicalSynapse", "GapJunction", "BulkConnection",
    "ConnectionGroup", "Path",
    
    # Recordings
    "Trial", "StimResponse",
    
    # Graph mapping
    "GraphMap",

    # Behavior
    "Behavior",
    
    # Utilities
    "F_SAMPLE", "RANDOM_SEED", "RECURSION_LIMIT",
    "load_pickle", "load_worm", "generate_random_string",
    "logger"
]