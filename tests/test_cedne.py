import networkx as nx
import numpy as np
import pytest
from cedne.cedne import Worm, NervousSystem, Behavior, Neuron, Trial, StimResponse, Connection, Mapping, load_worm, F_SAMPLE

### Worm
class TestWorm:
    @pytest.fixture
    def worm(self):
        return Worm()

    def test_init(self, worm):
        """
        Tests if the Worm object is initialized correctly.
        """
        assert worm.name.startswith('Worm-')
        assert worm.stage == 'Day-1 Adult'
        assert worm.sex == 'Hermaphrodite'
        assert worm.genotype == 'N2'

    def test_save_load(self, tmpdir):
        """
        Tests if a Worm object can be saved and loaded correctly.
        """
        worm = Worm(name='Test Worm', stage='Day-2 Adult', sex='Male', genotype='N2-mutant')
        file_path = tmpdir.join('worm.pkl')
        worm.save(str(file_path))
        loaded_worm = load_worm(str(file_path))
        assert worm.name == loaded_worm.name
        assert worm.stage == loaded_worm.stage
        assert worm.sex == loaded_worm.sex
        assert worm.genotype == loaded_worm.genotype

## Behavior
class TestBehavior:
    @pytest.fixture
    def behavior(self):
        return Behavior()

    def test_init(self, behavior):
        """
        Tests if the Behavior object is initialized correctly.
        """
        assert behavior.worm is not None

    def test_init_with_params(self):
        """
        Tests if the Behavior object is initialized correctly with parameters.
        """
        worm = Worm(name='Test Worm', stage='Day-2 Adult', sex='Male', genotype='N2-mutant')
        behavior = Behavior(worm=worm, condition='Test Condition')
        assert behavior.worm is worm

## NervousSystem

class TestNervousSystem:
    @pytest.fixture
    def nervous_system(self):
        return NervousSystem()

    def test_init(self, nervous_system):
        """
        Tests if the NervousSystem object is initialized correctly.
        """
        assert nervous_system.worm is not None
        assert isinstance(nervous_system, nx.MultiDiGraph)

    def test_build_network(self, tmpdir):
        """
        Tests if the NervousSystem object can be built from a pickle file.
        """
        nervous_system = NervousSystem()
        neurons = tmpdir.join('neuron_data.pkl')
        adj = {}
        label = 'test'
        nervous_system.build_network(str(neurons), adj, label)
        assert len(nervous_system.neurons) > 0
        assert len(nervous_system.connections) == 0

    def test_update_neurons(self, nervous_system):
        """
        Tests if the NervousSystem object updates the neurons dictionary correctly.
        """
        nervous_system.update_neurons()
        assert len(nervous_system.neurons) > 0

    def test_setup_connections(self, nervous_system):
        """
        Tests if the NervousSystem object sets up connections correctly.
        """
        adjacency_matrix = {0: {1: {'weight': 1}}}
        nervous_system.setup_connections(adjacency_matrix, 'test')
        assert len(nervous_system.connections) > 0

    def test_create_neurons(self, nervous_system):
        """
        Tests if the NervousSystem object creates neurons correctly.
        """
        labels = ['test']
        nervous_system.create_neurons(labels)
        assert 'test' in nervous_system.neurons

    def test_setup_chemical_connections(self, nervous_system):
        """
        Tests if the NervousSystem object sets up chemical connections correctly.
        """
        chemical_adjacency = {0: {1: {'weight': 1}}}
        nervous_system.setup_chemical_connections(chemical_adjacency)
        assert len(nervous_system.connections) > 0

    def test_setup_gap_junctions(self, nervous_system):
        """
        Tests if the NervousSystem object sets up gap junctions correctly.
        """
        gap_junction_adjacency = {0: {1: {'weight': 1}}}
        nervous_system.setup_gap_junctions(gap_junction_adjacency)
        assert len(nervous_system.connections) > 0

    def test_subnetwork(self, nervous_system):
        """
        Tests if the NervousSystem object generates a subnetwork correctly.
        """
        neuron_names = ['test']
        subnetwork = nervous_system.subnetwork(neuron_names)
        assert len(subnetwork.neurons) == 1

    def test_fold_network(self, nervous_system):
        """
        Tests if the NervousSystem object folds the network correctly.
        """
        nervous_system.fold_network(('test', 'test'))
        assert len(nervous_system.neurons) == 1

    def test_contract_neurons(self, nervous_system):
        """
        Tests if the NervousSystem object contracts neurons correctly.
        """
        nervous_system.contract_neurons(('test', 'test'))
        assert len(nervous_system.neurons) == 1

    def test_return_network_where(self, nervous_system):
        """
        Tests if the NervousSystem object returns a subnetwork with the specified conditions.
        """
        subnetwork = nervous_system.return_network_where(neurons_have={'test': 'test'})
        assert len(subnetwork.neurons) == len(nervous_system.neurons)

    def test_copy(self, nervous_system):
        """
        Tests if the NervousSystem object can be copied correctly.
        """
        copied_nervous_system = nervous_system.copy()
        assert copied_nervous_system is not nervous_system
### Neuron

class TestNeuron:
    def test_set_presynapse(self):
        """
        Tests if the `set_presynapse` method sets the presynapse correctly.
        """
        network = NervousSystem()
        neuron = Neuron('test', network)
        neuron.set_presynapse(['pre1', 'pre2'])
        assert neuron.presynapse == ['pre1', 'pre2']

    def test_set_postsynapse(self):
        """
        Tests if the `set_postsynapse` method sets the postsynapse correctly.
        """
        network = NervousSystem()
        neuron = Neuron('test', network)
        neuron.set_postsynapse({'receptor1': ['ligand1', 'ligand2'],
                                'receptor2': ['ligand3']})
        assert neuron.postsynapse == {'receptor1': ['ligand1', 'ligand2'],
                                     'receptor2': ['ligand3']}

    def test_add_trial(self):
        """
        Tests if the `add_trial` method adds a new trial to the `trial` dictionary correctly.
        """
        network = NervousSystem()
        neuron = Neuron('test', network)
        trial = neuron.add_trial(trial_num=5)
        assert neuron.trial[5] == trial

    def test_get_all_connections(self):
        """
        Tests if the `get_all_connections` method returns all connections that the neuron is involved in correctly.
        """
        network = NervousSystem()
        neuron1 = Neuron('test1', network)
        neuron2 = Neuron('test2', network)
        network.add_edge(neuron1, neuron2)
        assert neuron1.get_all_connections() == [(neuron1, neuron2)]

    def test_outgoing(self):
        """
        Tests if the `outgoing` method returns all outgoing connections from the current object correctly.
        """
        network = NervousSystem()
        neuron1 = Neuron('test1', network)
        neuron2 = Neuron('test2', network)
        network.add_edge(neuron1, neuron2)
        assert neuron1.outgoing() == []
        assert neuron2.outgoing() == [(neuron1, neuron2)]

    def test_incoming(self):
        """
        Tests if the `incoming` method returns all incoming connections to the current object correctly.
        """
        network = NervousSystem()
        neuron1 = Neuron('test1', network)
        neuron2 = Neuron('test2', network)
        network.add_edge(neuron1, neuron2)
        assert neuron1.incoming() == [(neuron1, neuron2)]
        assert neuron2.incoming() == []

    def test_set_property(self):
        """
        Tests if the `set_property` method sets a new property attribute for the class correctly.
        """
        network = NervousSystem()
        neuron = Neuron('test', network)
        neuron.set_property('test_property', 'test_value')
        assert neuron.get_property('test_property') == 'test_value'

## Trial

class TestTrial:
    def test_recording(self):
        """
        Tests the recording property of the Trial class.
        """
        trial = Trial(None, 0)
        trial.recording = np.array([1, 2, 3])
        assert np.allclose(trial.recording, np.array([1, 2, 3]))

    def test_recording_discard(self):
        """
        Tests the recording property of the Trial class with discard.
        """
        trial = Trial(None, 0)
        trial.recording = np.array([1, 2, 3, 4, 5]), 2
        assert np.allclose(trial.recording, np.array([3, 4, 5]))
        assert trial.discard == 2*F_SAMPLE

    def test_recording_invalid(self):
        """
        Tests the recording property of the Trial class with an invalid length.
        """
        trial = Trial(None, 0)
        with pytest.raises(ValueError):
            trial.recording = np.array([1, 2])

    def test_recording_negative_discard(self):
        """
        Tests the recording property of the Trial class with a negative discard value.
        """
        trial = Trial(None, 0)
        with pytest.raises(ValueError):
            trial.recording = np.array([1, 2, 3]), -1


class TestStimResponse:
    def test_extract_feature_0(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(0) == 3

    def test_extract_feature_1(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(1) == 2

    def test_extract_feature_2(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(2) == 2

    def test_extract_feature_3(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(3) == 1

    def test_extract_feature_4(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(4) == 1

    def test_extract_feature_5(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(5) == 1

    def test_extract_feature_6(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(6) == 0

    def test_extract_feature_7(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(7) == (0, 0)

    def test_extract_feature_8(self):
        stim_response = StimResponse(None, np.array([1, 2, 3]), np.array([1, 2, 3]), 0)
        assert stim_response.extract_feature(8) == 2

## Connection
class TestConnection:
    def test_init(self):
        pre_neuron = Neuron(None, 0, neuron_type='regular')
        post_neuron = Neuron(None, 1, neuron_type='regular')
        c = Connection(pre_neuron, post_neuron)
        assert c.pre == pre_neuron
        assert c.post == post_neuron
        assert c.uid == 0
        assert c.connection_type == 'chemical-synapse'
        assert c.weight == 1

    def test_update_weight(self):
        pre_neuron = Neuron(None, 0, 'regular')
        post_neuron = Neuron(None, 1, 'regular')
        c = Connection(pre_neuron, post_neuron)
        c.update_weight(2)
        assert c.weight == 2
        c.update_weight(3, delta=True)
        assert c.weight == 5

    def test_set_property(self):
        pre_neuron = Neuron(None, 0, 'regular')
        post_neuron = Neuron(None, 1, 'regular')
        c = Connection(pre_neuron, post_neuron)
        c.set_property('color', 'blue')
        assert c.color == 'blue'
        assert nx.get_edge_attributes(c.network.graph, c._id)['color'] == 'blue'

## Mapping
class TestMapping:
    @pytest.fixture
    def mapping(self):
        return Mapping()

    def test_init(self, mapping):
        assert isinstance(mapping, nx.DiGraph)

    def test_add_node(self, mapping):
        mapping.add_node('node1')
        assert 'node1' in mapping.nodes()

    def test_add_edge(self, mapping):
        mapping.add_edge('node1', 'node2')
        assert ('node1', 'node2') in mapping.edges()

    def test_add_edge_with_property(self, mapping):
        mapping.add_edge('node1', 'node2', weight=5)
        assert ('node1', 'node2') in mapping.edges()
        assert mapping.edges['node1', 'node2']['weight'] == 5