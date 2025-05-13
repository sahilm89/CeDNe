"""Tests for worm creation and network functionality."""

import pytest
from cedne import utils
from cedne.core.network import NervousSystem

def test_worm_creation_and_network():
    """Test basic worm creation and network functionality."""
    # Create worm and get nervous system
    w = utils.makeWorm()
    nn = w.networks["Neutral"]
    
    # Verify basic structure
    assert isinstance(nn, NervousSystem)
    assert len(nn.neurons) > 0
    
    # Set property on sensory neurons
    sensory_count = 0
    for n in nn.neurons:
        if nn.neurons[n].type == 'sensory' and nn.neurons[n].modality == 'amphid':
            nn.neurons[n].set_property('sensory-stim', True)
            sensory_count += 1
    
    # Verify some sensory neurons were found and modified
    assert sensory_count > 0
    
    # Create subgraph based on property
    g = nn.return_network_where(neurons_have={'sensory-stim': True})
    
    # Verify subgraph properties
    assert len(g.nodes) == sensory_count
    assert len(g.edges) > 0  # Should have some connections
    
    # Verify property was correctly set
    for node in g.nodes:
        assert g.nodes[node].get('sensory-stim', False) is True 