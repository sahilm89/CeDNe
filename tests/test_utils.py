"""
Tests for CeDNe utility functions.

This module contains tests for:
- Data loading and building (loader.py)
- Plotting and visualization (plotting.py)
- Graph manipulation (graphtools.py)
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cedne.utils.loader import makeWorm, getLigands, getLigandsAndReceptors
from cedne.utils.plotting import simpleaxis, plot_spiral
from cedne.utils.graphtools import joinLRNodes, foldByNeuronType, is_left_neuron
from cedne.core.neuron import Neuron
from cedne.core.network import NervousSystem
from cedne.core.animal import Worm

# Test fixtures
@pytest.fixture
def sample_ligand_table():
    """Create a sample ligand table for testing."""
    return pd.DataFrame({
        'Neuron': ['AVAL', 'AVAR', 'AVBL', 'AVBR'],
        'Neurotransmitter 1': ['GABA', 'GABA', 'ACh', 'ACh'],
        'Neurotransmitter 2': ['', '', 'GABA', 'GABA']
    })

@pytest.fixture
def sample_npr_table():
    """Create a sample neurotransmitter receptor table for testing."""
    return pd.DataFrame({
        'gene_name': ['gab-1', 'gab-2', 'ach-1', 'ach-2'],
        'AVAL': [True, False, True, False],
        'AVAR': [True, False, True, False]
    })

@pytest.fixture
def sample_ligmap_table():
    """Create a sample ligand mapping table for testing."""
    return pd.DataFrame({
        'gene': ['gab-1', 'gab-2', 'ach-1', 'ach-2'],
        'ligand': ['GABA', 'GABA', 'ACh', 'ACh']
    })

@pytest.fixture
def sample_neural_network():
    """Create a sample neural network for testing."""
    worm = Worm('test_worm')
    nn = NervousSystem(worm, "test_network")
    
    # Create neurons
    neurons = {}
    for name in ['AVAL', 'AVAR', 'AVBL', 'AVBR']:
        neurons[name] = Neuron(name, nn)
    
    # Create adjacency dictionary
    adjacency = {
        'AVAL': {
            'AVBL': {'weight': 1}
        },
        'AVAR': {
            'AVBR': {'weight': 1}
        },
        'AVBL': {
            'AVAL': {'weight': 1}
        },
        'AVBR': {
            'AVAR': {'weight': 1}
        }
    }
    
    # Add connections
    nn.setup_connections(adjacency, connection_type='chemical')
    
    return nn

class TestLoader:
    """Tests for loader.py utilities."""
    
    def test_makeWorm(self):
        """Test worm creation."""
        worm = makeWorm('test_worm')
        assert worm.name == 'test_worm'
        assert hasattr(worm, 'networks')
        assert 'Neutral' in worm.networks
        assert isinstance(worm.networks['Neutral'], NervousSystem)
        
    def test_getLigands(self):
        """Test getting ligands for a neuron."""
        ligands = getLigands('AVAL', sex='Hermaphrodite')
        assert isinstance(ligands, list)
        assert all(isinstance(ligand, str) for ligand in ligands)
        
    def test_getLigandsAndReceptors(self, sample_npr_table, sample_ligmap_table):
        """Test getting ligands and receptors."""
        receptor_ligand = getLigandsAndReceptors(sample_npr_table, sample_ligmap_table, 'AVAL')
        assert 'gab-1' in receptor_ligand
        assert 'ach-1' in receptor_ligand
        assert receptor_ligand['gab-1'] == 'GABA'
        assert receptor_ligand['ach-1'] == 'ACh'

class TestPlotting:
    """Tests for plotting.py utilities."""
    
    def test_simpleaxis(self):
        """Test axis simplification."""
        fig, ax = plt.subplots()
        simpleaxis(ax)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        assert ax.spines['bottom'].get_visible()
        assert ax.spines['left'].get_visible()
        plt.close()
        
    def test_simpleaxis_every(self):
        """Test axis simplification with every=True."""
        fig, ax = plt.subplots()
        simpleaxis(ax, every=True)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        assert not ax.spines['bottom'].get_visible()
        assert not ax.spines['left'].get_visible()
        plt.close()
        
    def test_plot_spiral(self, sample_neural_network):
        """Test spiral plot generation."""
        pos = plot_spiral(sample_neural_network)
        assert isinstance(pos, dict)
        assert all(node in pos for node in sample_neural_network.neurons.values())
        plt.close()

class TestGraphtools:
    """Tests for graphtools.py utilities."""
    
    def test_joinLRNodes(self, sample_neural_network):
        """Test joining left and right nodes."""
        new_network = joinLRNodes(sample_neural_network)
        assert isinstance(new_network, NervousSystem)
        assert 'AVAL' not in new_network.neurons
        assert 'AVAR' not in new_network.neurons
        assert 'AVA' in new_network.neurons
        
    def test_foldByNeuronType(self, sample_neural_network):
        """Test folding by neuron type."""
        new_network = foldByNeuronType(sample_neural_network)
        assert isinstance(new_network, NervousSystem)
        assert len(new_network.neurons) <= len(sample_neural_network.neurons)
        
    def test_is_left_neuron(self):
        """Test left neuron identification."""
        assert is_left_neuron('AVAL')
        assert not is_left_neuron('AVAR')
        assert not is_left_neuron('AVA')
