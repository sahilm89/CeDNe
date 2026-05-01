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
import cedne.utils.loader as loader_utils
from cedne.utils.loader import makeWorm, getLigands, getLigandsAndReceptors, make_ciona
from cedne.utils.plotting import simpleaxis, plot_spiral, plot_ciona_anatomical
from cedne.utils.graphtools import joinLRNodes, foldByNeuronType, is_left_neuron, getNeuronClass
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
    
    def test_makeWorm(self, monkeypatch):
        """Test worm creation without depending on packaged pickle payloads."""
        captured = {}

        def fake_build_nervous_system(nn, **kwargs):
            captured["network"] = nn
            captured.update(kwargs)

        monkeypatch.setattr(loader_utils, "build_nervous_system", fake_build_nervous_system)

        worm = makeWorm('test_worm')
        assert worm.name == 'test_worm'
        assert hasattr(worm, 'networks')
        assert 'Neutral' in worm.networks
        assert isinstance(worm.networks['Neutral'], NervousSystem)
        assert captured["network"] is worm.networks["Neutral"]
        assert captured["neuron_data"] == loader_utils.cell_list
        assert captured["chem_synapses"] == loader_utils.chemsyns
        assert captured["elec_synapses"] == loader_utils.elecsyns
        assert captured["positions"] == loader_utils.neuronPositions
        assert worm.citations["cook_connectome"] == loader_utils.citations["cook_connectome"]
        
    # def test_getLigands(self):
    #     """Test getting ligands for a neuron."""
    #     ligands = getLigands('AVAL', sex='Hermaphrodite')
    #     assert isinstance(ligands, list)
    #     assert all(isinstance(ligand, str) for ligand in ligands)
        
    def test_getLigandsAndReceptors(self, sample_npr_table, sample_ligmap_table):
        """Test getting ligands and receptors."""
        receptor_ligand = getLigandsAndReceptors(sample_npr_table, sample_ligmap_table, 'AVAL')
        assert 'gab-1' in receptor_ligand
        assert 'ach-1' in receptor_ligand
        assert receptor_ligand['gab-1'] == 'GABA'
        assert receptor_ligand['ach-1'] == 'Acetylcholine'

    def test_make_ciona_does_not_fallback_to_nodes_layout(self, tmp_path, monkeypatch):
        """Test that Ciona positions only come from anatomical coordinates."""
        nodes = pd.DataFrame({
            'index': [0, 1],
            'name': ['NeuronA', 'NeuronB'],
            'color': ['FF0000', '00FF00'],
            '_pos': ['array([0.0, 23.0])', 'array([1.0, 24.0])'],
        })
        edges = pd.DataFrame(columns=['source', 'target', 'depth'])
        fig1 = pd.DataFrame({
            'Cell IDs': ['NeuronA', 'NeuronB'],
            'Cell Type': ['TypeA', 'TypeB'],
            'Annotation': ['Sensory', 'Motor neuron'],
        })
        fig3 = pd.DataFrame({
            'Cell': [0],
            'First': [1],
            'Last': [2],
            'Z': [30.0],
            'X': [10.0],
            'Y': [20.0],
        })

        nodes.to_csv(tmp_path / 'nodes.csv', index=False)
        edges.to_csv(tmp_path / 'edges.csv', index=False)
        fig1.to_excel(tmp_path / 'elife-16962-fig1-data1-v1.xlsx', sheet_name='Sheet1', index=False)
        fig3.to_excel(tmp_path / 'elife-16962-fig3-data1-v1.xlsx', sheet_name='Sheet2', index=False)

        monkeypatch.setattr(loader_utils, 'ciona_connectome', tmp_path)

        animal = make_ciona()
        nn = animal.networks['Neutral']

        np.testing.assert_array_equal(nn.neurons['NeuronA'].position, np.array([10.0, 20.0, 30.0]))
        assert nn.neurons['NeuronB'].position is None

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

    def test_plot_ciona_anatomical_skips_missing_positions(self):
        """Test that Ciona plotting omits neurons without anatomical coordinates."""
        worm = Worm('test_worm')
        nn = NervousSystem(worm, "test_network")
        nn.create_neurons(
            ['A', 'B', 'C'],
            position={
                'A': np.array([1.0, 2.0, 3.0]),
                'B': np.array([4.0, 5.0, 6.0]),
            },
            annotation={
                'A': 'Sensory',
                'B': 'Interneuron',
                'C': 'Motor neuron',
            },
        )

        with pytest.warns(UserWarning, match='Skipping 1 neurons without anatomical coordinates'):
            fig = plot_ciona_anatomical(nn, view='2d')

        ax = fig.axes[0]
        assert len(ax.collections[0].get_offsets()) == 2
        plt.close(fig)

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

    @pytest.mark.parametrize('name,expected', [
        ('ADAL', 'ADA'),     # paired L
        ('AVAR', 'AVA'),     # paired R
        ('AVL', 'AVL'),      # unpaired — too short to strip
        ('RIM', 'RIM'),      # no L/R suffix
        ('M2L', 'M2'),       # 2-char trunk ending in digit accepts L
        ('I1R', 'I1'),
        ('MCL', 'MC'),       # MC explicit exception
        ('DA01', 'DA'),      # body-wall motor neuron with numeric suffix
        ('VA12', 'VA'),
        ('RMDDL', 'RMD'),    # longest matching suffix wins (DL > L)
        ('IL2DR', 'IL2'),    # DR > R
        ('PHAL', 'PHA'),
        ('URYDL', 'URY'),    # DL > L
    ])
    def test_getNeuronClass(self, name, expected):
        assert getNeuronClass(name) == expected
