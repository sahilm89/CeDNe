import cedne
import pytest
from cedne import Neuron, NeuronGroup, NervousSystem

# Example test for NervousSystem subNetwork method
def test_nervous_system_subnetwork():
    # Arrange: Create a NervousSystem instance and a list of neuron names
    ns = NervousSystem()
    neuron_list = ['AWCL', 'AWCR', 'AWBL', 'AWBR', 'RIAL', 'RIAR']

    # Act: Create a subnetwork
    sub_net = ns.subNetwork(neuron_list)

    # Assert: Check that the subnetwork is a deep copy and contains the right neurons
    assert sub_net != ns
    assert all(n.name in neuron_list for n in sub_net.neuronDict.values())