"""
CeDNe: Connectome-embedded Dynamical Networks
A composable framework for building, analyzing, and visualizing nervous systems.

This top-level module exports the core components (e.g., Worm, Fly, Neuron, NervousSystem)
and utility functions (e.g., graph motifs, loaders, visualization) from its submodules.

Modules:
- `core`: Core neuroscience data structures (neurons, networks, trials)
- `utils`: High-level tools for graph analysis, data loading, and visualization
"""

# Core modeling components
from .core import (
    Animal, Worm, Fly,
    Cell, Neuron, NeuronGroup,
    NervousSystem,
    Connection, ChemicalSynapse, GapJunction, BulkConnection,
    ConnectionGroup, Path,
    Trial, StimResponse,
    F_SAMPLE, RANDOM_SEED, RECURSION_LIMIT,
    load_pickle, load_worm, generate_random_string,
    GraphMap,
    Behavior,
    logger, configure_logging
)

# Utility layer (plotting, analysis, loaders)
from . import utils

# Analysis module (state space, dynamical models)
from . import analysis

__all__ = [
    # Core models
    "Animal", "Worm", "Fly",
    "Cell", "Neuron", "NeuronGroup",
    "NervousSystem",
    "Connection", "ChemicalSynapse", "GapJunction", "BulkConnection",
    "ConnectionGroup", "Path",
    "Trial", "StimResponse", "Behavior", "GraphMap",

    # Config + I/O
    "F_SAMPLE", "RANDOM_SEED", "RECURSION_LIMIT",
    "load_pickle", "load_worm", "generate_random_string",

    # Logging
    "logger", "configure_logging",

    # Utilities (as a namespace)
    "utils",

    # Analysis
    "analysis",
]
