"""
Global configuration constants for CeDNe core.
"""

import sys

# Random seed for reproducibility
RANDOM_SEED = 42

# Sampling rate for physiology calcium data (Hz)
F_SAMPLE = 5

# Maximum recursion depth (used in network folding, motif search)
RECURSION_LIMIT = 10000
sys.setrecursionlimit(RECURSION_LIMIT)