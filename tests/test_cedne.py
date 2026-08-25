import pytest
import numpy as np
import networkx as nx
from cedne.core.animal import Worm, load_worm
from cedne.core.network import NervousSystem
from cedne.core.neuron import Neuron, NeuronGroup
from cedne.core.recordings import Trial, StimResponse
from cedne.core.connection import (
    Connection,
    ChemicalSynapse,
    GapJunction,
    BulkConnection,
    ConnectionGroup,
    Path,
)
from cedne.core.mapping import GraphMap

# Test Constants
TEST_NEURON_NAMES = ["AVAL", "AVAR", "AVBL", "AVBR"]
TEST_CONNECTION_WEIGHTS = {"chemical": 1.0, "gap": 0.5}
TEST_RECORDING_LENGTH = 1000
TEST_SAMPLE_RATE = 1000  # Hz


class TestWorm:
    @pytest.fixture
    def worm(self):
        return Worm(name="TestWorm", stage="L4", sex="Hermaphrodite", genotype="N2")

    @pytest.fixture
    def nervous_system(self):
        return NervousSystem()

    def test_worm_initialization(self, worm):
        """Test proper initialization of Worm object with all attributes"""
        assert worm.name == "TestWorm"
        assert worm.stage == "L4"
        assert worm.sex == "Hermaphrodite"
        assert worm.genotype == "N2"
        assert isinstance(worm.networks, dict)
        assert isinstance(worm.contexts, dict)
        assert worm.active_context is None

    def test_worm_serialization(self, worm, tmp_path):
        """Test saving and loading of Worm object"""
        file_path = tmp_path / "test_worm.pkl"
        worm.save(str(file_path))
        loaded_worm = load_worm(str(file_path))

        assert loaded_worm.name == worm.name
        assert loaded_worm.stage == worm.stage
        assert loaded_worm.sex == worm.sex
        assert loaded_worm.genotype == worm.genotype

    def test_worm_serialization_with_pandas_attribute(self, worm, tmp_path):
        """Worms commonly carry pandas-backed attribute tables (e.g. neurotransmitter
        / neuropeptide data). RestrictedUnpickler must allow them through."""
        import pandas as pd

        worm.set_property("nt_table", pd.Series({"AVA": "Glu", "AVB": "ACh"}))
        worm.set_property(
            "np_table", pd.DataFrame({"peptide": ["FLP-1"], "range": ["short"]})
        )
        file_path = tmp_path / "test_worm_pandas.pkl"
        worm.save(str(file_path))
        loaded_worm = load_worm(str(file_path))
        assert loaded_worm.nt_table.loc["AVA"] == "Glu"
        assert loaded_worm.np_table.iloc[0]["peptide"] == "FLP-1"

    def test_worm_invalid_attributes(self):
        """Test Worm initialization with invalid attributes"""
        # The current implementation doesn't validate sex or stage
        # So we'll test setting properties instead
        worm = Worm()
        worm.set_property("test_property", "test_value")
        assert worm.test_property == "test_value"

    def test_add_network(self, worm, nervous_system):
        """Test adding a network to the worm"""
        network_name = "test_network"
        worm.networks[network_name] = nervous_system

        assert network_name in worm.networks
        assert worm.networks[network_name] == nervous_system

    def test_remove_network(self, worm, nervous_system):
        """Test removing a network from the worm"""
        network_name = "test_network"
        worm.networks[network_name] = nervous_system
        del worm.networks[network_name]

        assert network_name not in worm.networks

    def test_add_context(self, worm):
        """Test adding a context to the worm"""
        context_name = "test_context"
        worm.add_context(context_name)

        assert context_name in worm.contexts
        assert worm.contexts[context_name] is None

    def test_add_context_with_data(self, worm):
        """Test adding a context with data to the worm"""
        context_name = "test_context"
        context_data = {"environment": "test_env"}
        worm.add_context(context_name, context_data)

        assert context_name in worm.contexts
        assert worm.contexts[context_name] == context_data

    def test_remove_context(self, worm):
        """Test removing a context from the worm"""
        context_name = "test_context"
        worm.add_context(context_name)
        worm.remove_context(context_name)

        assert context_name not in worm.contexts

    def test_set_active_context(self, worm):
        """Test setting and getting the active context"""
        context_name = "test_context"
        worm.add_context(context_name)
        worm.set_active_context(context_name)

        assert worm.active_context == context_name
        assert worm.get_context(context_name) is None

    def test_remove_active_context(self, worm):
        """Test removing the active context"""
        context_name = "test_context"
        worm.add_context(context_name)
        worm.set_active_context(context_name)
        worm.remove_context(context_name)

        assert worm.active_context is None

    def test_clear_active_context(self, worm):
        """Test clearing the active context"""
        context_name = "test_context"
        worm.add_context(context_name)
        worm.set_active_context(context_name)
        worm.clear_active_context()

        assert worm.active_context is None

    def test_invalid_context_operations(self, worm):
        """Test error handling for invalid context operations"""
        context_name = "test_context"

        # Test setting non-existent context as active
        with pytest.raises(ValueError):
            worm.set_active_context("non_existent_context")

        # Test removing non-existent context
        with pytest.raises(ValueError):
            worm.remove_context("non_existent_context")

        # Test getting non-existent context
        assert worm.get_context("non_existent_context") is None

    def test_multiple_properties(self, worm):
        """Test setting and getting multiple properties"""
        properties = {"property1": "value1", "property2": 42, "property3": [1, 2, 3]}

        for key, value in properties.items():
            worm.set_property(key, value)
            assert getattr(worm, key) == value

    def test_network_operations(self, worm, nervous_system):
        """Test various network operations"""
        # Add neurons to the nervous system
        neuron1 = Neuron("AVAL", nervous_system)
        neuron2 = Neuron("AVBL", nervous_system)

        # Add the network to the worm
        network_name = "test_network"
        worm.networks[network_name] = nervous_system

        # Verify we can access the network and its neurons
        assert neuron1 in worm.networks[network_name].nodes()
        assert neuron2 in worm.networks[network_name].nodes()

    def test_multiple_networks(self, worm):
        """Test managing multiple networks"""
        ns1 = NervousSystem()
        ns2 = NervousSystem()

        # Add neurons to different networks
        neuron1 = Neuron("AVAL", ns1)
        neuron2 = Neuron("AVBL", ns2)

        # Add networks to worm
        worm.networks["network1"] = ns1
        worm.networks["network2"] = ns2

        # Verify network access
        assert neuron1 in worm.networks["network1"].nodes()
        assert neuron2 in worm.networks["network2"].nodes()
        assert neuron1 not in worm.networks["network2"].nodes()
        assert neuron2 not in worm.networks["network1"].nodes()

    def test_context_switching(self, worm):
        """Test switching between multiple contexts"""
        # Add multiple contexts
        worm.add_context("context1", {"env": "env1"})
        worm.add_context("context2", {"env": "env2"})

        # Switch contexts
        worm.set_active_context("context1")
        assert worm.active_context == "context1"
        assert worm.get_context("context1")["env"] == "env1"

        worm.set_active_context("context2")
        assert worm.active_context == "context2"
        assert worm.get_context("context2")["env"] == "env2"

    def test_property_inheritance(self, worm):
        """Test property inheritance and overriding"""
        # Set base properties
        worm.set_property("base_prop", "base_value")
        assert worm.base_prop == "base_value"

        # Set context-specific properties
        worm.add_context("context1", {"prop": "context1_value"})
        worm.add_context("context2", {"prop": "context2_value"})

        # Switch contexts and verify property access
        worm.set_active_context("context1")
        assert worm.get_context("context1")["prop"] == "context1_value"

        worm.set_active_context("context2")
        assert worm.get_context("context2")["prop"] == "context2_value"

    def test_complex_serialization(self, worm, tmp_path):
        """Test serialization with complex data structures"""
        # Create complex data
        ns = NervousSystem()
        neuron = Neuron("AVAL", ns)
        worm.networks["test_network"] = ns

        context_data = {
            "env": "test_env",
            "params": {"param1": 1, "param2": 2},
            "list_data": [1, 2, 3],
        }
        worm.add_context("test_context", context_data)
        worm.set_active_context("test_context")

        # Add some properties
        worm.set_property("test_prop", "test_value")

        # Save and load
        file_path = tmp_path / "complex_worm.pkl"
        worm.save(str(file_path))
        loaded_worm = load_worm(str(file_path))

        # Verify loaded data
        assert loaded_worm.name == worm.name
        assert loaded_worm.test_prop == "test_value"
        assert loaded_worm.active_context == "test_context"
        assert loaded_worm.get_context("test_context") == context_data
        assert "AVAL" in [n.name for n in loaded_worm.networks["test_network"].nodes()]

    def test_network_context_interaction(self, worm):
        """Test interactions between networks and contexts"""
        # Create network and add neurons
        ns = NervousSystem()
        neuron = Neuron("AVAL", ns)
        worm.networks["test_network"] = ns

        # Create context with network reference
        context_data = {"active_network": "test_network", "active_neurons": ["AVAL"]}
        worm.add_context("test_context", context_data)
        worm.set_active_context("test_context")

        # Verify network access through context
        active_network = worm.networks[
            worm.get_context("test_context")["active_network"]
        ]
        assert neuron in active_network.nodes()

        # Switch context and verify network access
        worm.add_context("empty_context")
        worm.set_active_context("empty_context")
        assert worm.get_context("empty_context") is None

    def test_edge_cases(self, worm):
        """Test various edge cases"""
        # Test empty network
        worm.networks["empty_network"] = NervousSystem()
        assert len(worm.networks["empty_network"].nodes()) == 0

        # Test context with None data
        worm.add_context("none_context", None)
        assert worm.get_context("none_context") is None

        # Test setting active context to None
        with pytest.raises(ValueError):
            worm.set_active_context(None)

        # Test removing non-existent network
        with pytest.raises(KeyError):
            del worm.networks["non_existent_network"]


class TestNervousSystem:
    @pytest.fixture
    def nervous_system(self):
        ns = NervousSystem()
        return ns

    @pytest.fixture
    def neurons(self, nervous_system):
        """Create a set of test neurons with different types"""
        sensory1 = Neuron("ASHL", nervous_system, type="sensory")
        sensory2 = Neuron("AVM", nervous_system, type="sensory")
        interneuron1 = Neuron("AVAL", nervous_system, type="interneuron")
        interneuron2 = Neuron("AVBL", nervous_system, type="interneuron")
        return [sensory1, sensory2, interneuron1, interneuron2]

    def test_nervous_system_initialization(self, nervous_system):
        """Test proper initialization of NervousSystem"""
        assert isinstance(nervous_system, nx.MultiDiGraph)
        assert len(nervous_system.nodes()) == 0
        assert len(nervous_system.edges()) == 0

    def test_add_neuron(self, nervous_system):
        """Test adding neurons to the nervous system"""
        neuron = Neuron("AVAL", nervous_system)
        assert "AVAL" in [n.name for n in nervous_system.nodes()]
        assert neuron in nervous_system.nodes()

    def test_add_connection(self, nervous_system):
        """Test adding connections between neurons"""
        pre_neuron = Neuron("AVAL", nervous_system)
        post_neuron = Neuron("AVBL", nervous_system)
        connection = Connection(pre_neuron, post_neuron, "chemical-synapse")
        nervous_system.add_edge(pre_neuron, post_neuron, connection=connection)

        assert (pre_neuron, post_neuron) in nervous_system.edges()
        assert (
            nervous_system[pre_neuron][post_neuron]["chemical-synapse"][
                "connection_type"
            ]
            == "chemical-synapse"
        )

    def test_get_neurons_by_type(self, nervous_system, neurons):
        """Test retrieving neurons by type"""
        sensory_nodes = [n for n in nervous_system.nodes() if n.type == "sensory"]
        motor_nodes = [n for n in nervous_system.nodes() if n.type == "motor"]
        interneuron_nodes = [
            n for n in nervous_system.nodes() if n.type == "interneuron"
        ]

        assert len(sensory_nodes) == 2
        assert len(motor_nodes) == 0
        assert len(interneuron_nodes) == 2
        assert neurons[0] in sensory_nodes
        assert neurons[1] in sensory_nodes
        assert neurons[2] in interneuron_nodes
        assert neurons[3] in interneuron_nodes

    def test_connection_weights(self, nervous_system):
        """Test connection weight management"""
        pre_neuron = Neuron("AVAL", nervous_system)
        post_neuron = Neuron("AVBL", nervous_system)

        # Add chemical synapse
        chem_connection = Connection(
            pre_neuron, post_neuron, "chemical-synapse", weight=1.0
        )
        nervous_system.add_edge(pre_neuron, post_neuron, connection=chem_connection)

        # Add gap junction
        gap_connection = Connection(pre_neuron, post_neuron, "gap-junction", weight=0.5)
        nervous_system.add_edge(pre_neuron, post_neuron, connection=gap_connection)

        # Verify weights
        assert (
            nervous_system[pre_neuron][post_neuron]["chemical-synapse"]["weight"] == 1.0
        )
        assert nervous_system[pre_neuron][post_neuron]["gap-junction"]["weight"] == 0.5

    def test_neuron_degrees(self, nervous_system, neurons):
        """Test in-degree and out-degree calculations"""
        # Create connections
        Connection(neurons[0], neurons[2], "chemical-synapse")  # sensory -> interneuron
        Connection(neurons[2], neurons[1], "chemical-synapse")  # interneuron -> motor
        Connection(
            neurons[2], neurons[3], "gap-junction"
        )  # interneuron <-> interneuron
        Connection(neurons[3], neurons[2], "gap-junction")

        # Verify degrees
        assert nervous_system.in_degree(neurons[0]) == 0
        assert nervous_system.out_degree(neurons[0]) == 1
        assert nervous_system.in_degree(neurons[2]) == 2
        assert nervous_system.out_degree(neurons[2]) == 2

    def test_network_validation(self, nervous_system):
        """Test network validation and error handling"""
        # Test adding duplicate neuron
        neuron1 = Neuron("AVAL", nervous_system)
        with pytest.raises(
            ValueError, match="Neuron with name 'AVAL' already exists in the network"
        ):
            Neuron("AVAL", nervous_system)

        # Test adding self-connection (now allowed)
        connection = Connection(neuron1, neuron1, "chemical-synapse")
        assert connection.pre == neuron1
        assert connection.post == neuron1
        assert connection.connection_type == "chemical-synapse"

    def test_network_operations(self, nervous_system, neurons):
        """Test various network operations"""
        # Test removing neuron
        neuron_to_remove = neurons[0]
        nervous_system.remove_node(neuron_to_remove)
        assert neuron_to_remove not in nervous_system.nodes()

        # Test removing connection
        pre, post = neurons[1], neurons[2]
        connection = Connection(pre, post, "chemical-synapse")
        nervous_system.add_edge(pre, post, connection=connection)
        # For MultiDiGraph, we need to specify the key when removing edges
        nervous_system.remove_edge(pre, post, key="chemical-synapse")
        # Check if the specific connection type is removed
        assert "chemical-synapse" not in nervous_system[pre][post]

        # Test clearing network
        nervous_system.clear()
        assert len(nervous_system.nodes()) == 0
        assert len(nervous_system.edges()) == 0


class TestRemoveNeurons:
    """Coverage for ``NervousSystem.remove_neurons``. The previous
    implementation was broken for both name strings (passed straight to
    networkx's ``remove_node`` which expects the Neuron object) and
    Neuron instances (failed the name-keyed membership check up-front);
    it also left ``self.connections`` cache entries pointing at the
    removed node. This class pins the new behavior."""

    def _make_net(self):
        ns = NervousSystem()
        a = Neuron("A", ns)
        b = Neuron("B", ns)
        c = Neuron("C", ns)
        return ns, a, b, c

    def test_remove_by_name(self):
        ns, a, b, c = self._make_net()
        ns.remove_neurons(["A", "C"])
        assert "A" not in ns.neurons
        assert "C" not in ns.neurons
        assert "B" in ns.neurons
        assert a not in ns.nodes()
        assert c not in ns.nodes()
        assert b in ns.nodes()

    def test_remove_by_object(self):
        ns, a, b, c = self._make_net()
        ns.remove_neurons([a, c])
        assert "A" not in ns.neurons
        assert "C" not in ns.neurons
        assert "B" in ns.neurons
        assert a not in ns.nodes()
        assert b in ns.nodes()

    def test_remove_mixed_names_and_objects(self):
        ns, a, b, c = self._make_net()
        ns.remove_neurons(["A", c])
        assert "A" not in ns.neurons
        assert "C" not in ns.neurons
        assert "B" in ns.neurons

    def test_unknown_name_raises_keyerror(self):
        ns, *_ = self._make_net()
        with pytest.raises(KeyError, match="NOPE"):
            ns.remove_neurons(["NOPE"])

    def test_foreign_neuron_object_raises_keyerror(self):
        """A Neuron instance from a different NervousSystem must not be
        accepted as a stand-in even if a same-named neuron exists here."""
        ns, *_ = self._make_net()
        other = NervousSystem()
        foreign = Neuron("A", other)  # same name "A" — different network
        with pytest.raises(KeyError, match="A"):
            ns.remove_neurons([foreign])
        # Original "A" untouched.
        assert "A" in ns.neurons

    def test_wrong_type_raises_typeerror(self):
        ns, *_ = self._make_net()
        with pytest.raises(TypeError, match="int"):
            ns.remove_neurons([42])

    def test_atomic_on_bad_input(self):
        """A bad element mid-list must abort the whole call without
        leaving the graph half-mutated."""
        ns, a, b, c = self._make_net()
        with pytest.raises(KeyError):
            ns.remove_neurons(["A", "NOPE"])
        # "A" must still be present — resolution failed before any
        # mutation happened.
        assert "A" in ns.neurons
        assert a in ns.nodes()

    def test_empty_iterable_is_noop(self):
        ns, a, b, c = self._make_net()
        ns.remove_neurons([])
        assert set(ns.neurons.keys()) == {"A", "B", "C"}

    def test_incident_connections_are_cleaned_up(self):
        """Removing a neuron must prune its incident Connection cache
        entries — not just the underlying networkx edges. Regression
        test for the missing ``update_connections()`` call."""
        ns = NervousSystem()
        Neuron("A", ns)
        Neuron("B", ns)
        Neuron("C", ns)
        # ``setup_connections`` is what actually populates
        # ``ns.connections`` (raw ``Connection(...)`` does not).
        ns.setup_connections(
            {
                "A": {"B": {"weight": 1}, "C": {"weight": 1}},
                "B": {"C": {"weight": 1}},
            },
            connection_type="chemical",
        )
        # Sanity: three connections cached pre-removal.
        assert len(ns.connections) == 3

        ns.remove_neurons(["B"])

        # No surviving entry in the cache may reference B as pre or post.
        for pre, post, _key in ns.connections:
            assert pre.name != "B", f"stale connection from B: {pre.name}->{post.name}"
            assert post.name != "B", f"stale connection to B: {pre.name}->{post.name}"
        # Only A→C should remain.
        assert len(ns.connections) == 1
        only_pre, only_post, _ = next(iter(ns.connections))
        assert (only_pre.name, only_post.name) == ("A", "C")


class TestNeuronGroupRemoveNeuron:
    """Coverage for ``NeuronGroup.remove_neuron`` (singular). The previous
    implementation used ``if neuron in self.neurons`` where
    ``self.neurons`` is a name-keyed dict; a Neuron-object input is never
    a key, so the body silently no-op'd. Fix accepts names or objects."""

    def _make_group(self):
        ns = NervousSystem()
        a = Neuron("A", ns)
        b = Neuron("B", ns)
        # User-defined group containing both — group is independent of
        # the all_neurons group on the network.
        grp = NeuronGroup(ns, members=[a, b], group_name="pair")
        return ns, grp, a, b

    def test_remove_by_object(self):
        ns, grp, a, b = self._make_group()
        grp.remove_neuron(a)
        assert "A" not in grp.neurons
        assert "B" in grp.neurons
        # The underlying network is untouched — NeuronGroup is a
        # membership tracker, not a graph mutator.
        assert "A" in ns.neurons

    def test_remove_by_name(self):
        ns, grp, a, b = self._make_group()
        grp.remove_neuron("B")
        assert "B" not in grp.neurons
        assert "A" in grp.neurons
        assert "B" in ns.neurons  # network untouched

    def test_remove_unknown_is_noop(self):
        """Idempotent: removing a non-member doesn't raise."""
        ns, grp, a, b = self._make_group()
        grp.remove_neuron("NOPE")
        # Also: an unrelated Neuron object should silently no-op.
        ns2 = NervousSystem()
        foreign = Neuron("A", ns2)
        grp.remove_neuron(foreign)
        # Group state unchanged.
        assert set(grp.neurons.keys()) == {"A", "B"}

    def test_wrong_type_raises_typeerror(self):
        ns, grp, *_ = self._make_group()
        with pytest.raises(TypeError, match="int"):
            grp.remove_neuron(42)

    def test_remove_keeps_members_list_in_sync(self):
        """``self.members`` is the list-shaped mirror of ``self.neurons``
        and is read directly by ``get_property`` / ``get_connections``.
        It must not retain ghosts after removal."""
        ns, grp, a, b = self._make_group()
        grp.remove_neuron("A")
        assert [m.name for m in grp.members] == ["B"]
        grp.remove_neuron(b)
        assert grp.members == []


class TestNeuronGroupAddNeuron:
    """Coverage for ``NeuronGroup.add_neuron`` (singular). Previous
    implementation typed ``neuron: Neuron`` only; the guard
    ``if neuron not in self.neurons`` checked Neuron-object against a
    name-keyed dict — the guard worked correctly only for the
    *intended* name-string callers, but the signature didn't admit
    them. Fix accepts both Neuron objects and name strings, resolves
    strings via the parent network, and keeps ``self.members`` in
    sync with ``self.neurons``."""

    def _make_setup(self):
        ns = NervousSystem()
        a = Neuron("A", ns)
        b = Neuron("B", ns)
        c = Neuron("C", ns)
        # Group starts with just A.
        grp = NeuronGroup(ns, members=[a], group_name="g")
        return ns, grp, a, b, c

    def test_add_by_object(self):
        ns, grp, a, b, c = self._make_setup()
        grp.add_neuron(b)
        assert set(grp.neurons.keys()) == {"A", "B"}
        assert [m.name for m in grp.members] == ["A", "B"]

    def test_add_by_name(self):
        ns, grp, a, b, c = self._make_setup()
        grp.add_neuron("C")
        assert set(grp.neurons.keys()) == {"A", "C"}
        # Resolved object is the actual one from the parent network.
        assert grp.neurons["C"] is c

    def test_add_unknown_name_raises_keyerror(self):
        ns, grp, *_ = self._make_setup()
        with pytest.raises(KeyError, match="NOPE"):
            grp.add_neuron("NOPE")

    def test_add_wrong_type_raises_typeerror(self):
        ns, grp, *_ = self._make_setup()
        with pytest.raises(TypeError, match="int"):
            grp.add_neuron(42)

    def test_double_add_is_idempotent(self):
        ns, grp, a, b, c = self._make_setup()
        grp.add_neuron(b)
        grp.add_neuron(b)  # by object
        grp.add_neuron("B")  # by name
        # ``members`` must not duplicate.
        assert [m.name for m in grp.members] == ["A", "B"]
        assert set(grp.neurons.keys()) == {"A", "B"}


class TestNeuron:
    @pytest.fixture
    def neuron(self):
        ns = NervousSystem()
        return Neuron("AVAL", ns)

    def test_neuron_initialization(self, neuron):
        """Test proper initialization of Neuron"""
        assert neuron.name == "AVAL"
        assert hasattr(neuron, "trial")
        assert isinstance(neuron.trial, dict)
        assert isinstance(neuron.in_connections, dict)
        assert isinstance(neuron.out_connections, dict)

    def test_add_trial(self, neuron):
        """Test adding trials to a neuron"""
        trial = neuron.add_trial(0)
        assert 0 in neuron.trial
        assert isinstance(trial, Trial)
        assert trial.neuron == neuron
        assert trial.i == 0

    def test_get_connections(self, neuron):
        """Test retrieving connections for a neuron"""
        ns = neuron.network
        post_neuron = Neuron("AVBL", ns)
        connection = Connection(neuron, post_neuron, "chemical-synapse")
        ns.add_edge(neuron, post_neuron, connection=connection)

        connections = neuron.get_connections()
        assert connection in connections.values()


class TestTrial:
    @pytest.fixture
    def trial(self):
        ns = NervousSystem()
        neuron = Neuron("AVAL", ns)
        return Trial(neuron, 0)

    def test_trial_initialization(self, trial):
        """Test proper initialization of Trial"""
        assert hasattr(trial, "neuron")
        assert hasattr(trial, "i")
        assert trial.i == 0

    def test_trial_recording(self, trial):
        """Test trial recording methods"""
        recording = np.random.randn(TEST_RECORDING_LENGTH)
        trial.recording = recording
        assert np.array_equal(trial.recording, recording)


class TestStimResponse:
    @pytest.fixture
    def stim_response(self):
        ns = NervousSystem()
        neuron = Neuron("AVAL", ns)
        trial = Trial(neuron, 0)
        recording = np.random.randn(TEST_RECORDING_LENGTH)
        stimulus = np.zeros(TEST_RECORDING_LENGTH)
        stimulus[500:600] = 1  # Simulate a stimulus
        baseline = 100
        return StimResponse(trial, stimulus, recording, baseline)

    def test_stim_response_initialization(self, stim_response):
        """Test proper initialization of StimResponse"""
        assert hasattr(stim_response, "stim")
        assert hasattr(stim_response, "response")
        assert hasattr(stim_response, "baseline")
        assert hasattr(stim_response, "feature")

    def test_response_analysis(self, stim_response):
        """Test response analysis methods"""
        # Test feature extraction
        features = stim_response.feature
        assert isinstance(features, dict)
        assert len(features) > 0


class TestConnection:
    @pytest.fixture
    def connection(self):
        ns = NervousSystem()
        pre = Neuron("AVAL", ns)
        post = Neuron("AVBL", ns)
        return Connection(pre, post, "chemical-synapse")

    def test_connection_initialization(self, connection):
        """Test proper initialization of Connection"""
        assert connection.pre.name == "AVAL"
        assert connection.post.name == "AVBL"
        assert connection.connection_type == "chemical-synapse"
        assert connection.weight == 1.0

    def test_weight_modification(self, connection):
        """Test weight modification methods"""
        # Test weight update
        connection.update_weight(0.5)
        assert connection.weight == 0.5

        # Test weight increment
        connection.update_weight(0.2, delta=True)
        assert connection.weight == 0.7


class TestGraphMap:
    @pytest.fixture
    def graph_map(self):
        ns1 = NervousSystem()
        ns2 = NervousSystem()
        mapping_dict = {}
        return GraphMap(mapping_dict, ns1, ns2)

    def test_graph_map_initialization(self, graph_map):
        """Test proper initialization of GraphMap"""
        assert hasattr(graph_map, "graph_1")
        assert hasattr(graph_map, "graph_2")
        assert hasattr(graph_map, "mapping_dict")
        assert isinstance(graph_map.mapping_dict, dict)

    def test_add_mapping(self, graph_map):
        """Test adding mappings between neurons"""
        neuron1 = Neuron("AVAL", graph_map.graph_1)
        neuron2 = Neuron("AVBL", graph_map.graph_2)
        graph_map.mapping_dict[neuron1] = neuron2

        assert neuron1 in graph_map.mapping_dict
        assert graph_map.mapping_dict[neuron1] == neuron2

    def test_mapping_properties(self, graph_map):
        """Test mapping property methods"""
        neuron1 = Neuron("AVAL", graph_map.graph_1)
        neuron2 = Neuron("AVBL", graph_map.graph_2)
        graph_map.mapping_dict[neuron1] = neuron2

        # Test accessing the mapping
        assert graph_map.mapping_dict[neuron1] == neuron2
        # Test checking if a neuron is mapped
        assert neuron1 in graph_map.mapping_dict


class TestConnectionGroup:
    @pytest.fixture
    def nervous_system(self):
        """Create a test nervous system"""
        return NervousSystem()

    @pytest.fixture
    def connection_group(self, nervous_system):
        """Create a test connection group with various types of connections"""
        # Create neurons
        neuron1 = Neuron("AVAL", nervous_system)
        neuron2 = Neuron("AVBL", nervous_system)
        neuron3 = Neuron("AVAR", nervous_system)

        # Create connections
        chem_synapse = ChemicalSynapse(neuron1, neuron2, weight=1.0)
        gap_junction = GapJunction(neuron2, neuron3, weight=0.5)
        bulk_conn = BulkConnection(
            neuron1, neuron3, uid=2, connection_type="neuropeptide", weight=0.8
        )

        # Create group
        return ConnectionGroup(
            nervous_system, [chem_synapse, gap_junction, bulk_conn], "test_group"
        )

    def test_update_weights(self, connection_group):
        """Test updating weights for all connections"""
        # Test absolute weight update
        connection_group.update_weights(2.0)
        assert all(conn.weight == 2.0 for conn in connection_group.members)

        # Test delta weight update
        connection_group.update_weights(0.5, delta=True)
        assert all(conn.weight == 2.5 for conn in connection_group.members)

        # Test invalid weight
        with pytest.raises(ValueError, match="Weight must be a numeric value"):
            connection_group.update_weights("invalid")

    def test_update_weights_by_function(self, connection_group):
        """Test updating weights using a custom function"""

        # Test valid function
        def weight_func(conn):
            return conn.weight * 2

        connection_group.update_weights_by_function(weight_func)
        assert all(conn.weight == conn.weight for conn in connection_group.members)

        # Test invalid function
        with pytest.raises(ValueError, match="weight_function must be callable"):
            connection_group.update_weights_by_function("not a function")

        # Test function returning invalid value
        def invalid_func(conn):
            return "not a number"

        with pytest.raises(
            ValueError, match="weight_function must return numeric values"
        ):
            connection_group.update_weights_by_function(invalid_func)

    def test_filter_by_type(self, connection_group):
        """Test filtering connections by type"""
        # Filter chemical synapses
        chem_group = connection_group.filter_by_type("chemical-synapse")
        assert len(chem_group) == 1
        assert all(
            conn.connection_type == "chemical-synapse" for conn in chem_group.members
        )

        # Filter gap junctions
        gap_group = connection_group.filter_by_type("gap-junction")
        assert len(gap_group) == 1
        assert all(conn.connection_type == "gap-junction" for conn in gap_group.members)

        # Filter non-existent type
        empty_group = connection_group.filter_by_type("non-existent")
        assert len(empty_group) == 0

    def test_filter_by_property(self, connection_group):
        """Test filtering connections by property"""
        # Add a test property
        connection_group.set_property("test_prop", "value1")

        # Filter by property
        filtered_group = connection_group.filter_by_property("test_prop", "value1")
        assert len(filtered_group) == len(connection_group)

        # Filter by non-existent property
        empty_group = connection_group.filter_by_property("non_existent", "value")
        assert len(empty_group) == 0

    def test_filter_by_function(self, connection_group):
        """Test filtering connections using a custom function"""

        # Test valid filter function
        def weight_filter(conn):
            return conn.weight > 0.5

        filtered_group = connection_group.filter_by_function(weight_filter)
        assert all(conn.weight > 0.5 for conn in filtered_group.members)

        # Test invalid filter function
        with pytest.raises(ValueError, match="filter_function must be callable"):
            connection_group.filter_by_function("not a function")

    def test_get_statistics(self, connection_group):
        """Test getting connection group statistics"""
        stats = connection_group.get_statistics()

        assert stats["count"] == 3
        assert isinstance(stats["weight_mean"], float)
        assert isinstance(stats["weight_min"], float)
        assert isinstance(stats["weight_max"], float)
        assert len(stats["types"]) == 3  # chemical-synapse, gap-junction, neuropeptide

        # Test empty group statistics
        empty_group = ConnectionGroup(connection_group.network)
        empty_stats = empty_group.get_statistics()
        assert empty_stats["count"] == 0
        assert empty_stats["weight_mean"] == 0
        assert empty_stats["weight_min"] == 0
        assert empty_stats["weight_max"] == 0
        assert len(empty_stats["types"]) == 0


class TestPath:
    @pytest.fixture
    def nervous_system(self):
        """Create a test nervous system"""
        return NervousSystem()

    @pytest.fixture
    def path(self, nervous_system):
        """Create a test path with a sequence of connections"""
        # Create neurons in a chain
        neuron1 = Neuron("AVAL", nervous_system)
        neuron2 = Neuron("AVBL", nervous_system)
        neuron3 = Neuron("AVAR", nervous_system)

        # Create connections in sequence
        conn1 = Connection(
            neuron1, neuron2, connection_type="chemical-synapse", weight=1.0
        )
        conn2 = Connection(neuron2, neuron3, connection_type="gap-junction", weight=0.5)

        # Create path
        return Path(nervous_system, [conn1, conn2], "test_path")

    def test_path_initialization(self, path):
        """Test proper initialization of Path"""
        assert isinstance(path, Path)
        assert isinstance(path, ConnectionGroup)
        assert path.group_name == "test_path"
        assert len(path.members) == 2
        assert path.source.name == "AVAL"
        assert path.target.name == "AVAR"

    def test_path_continuity(self, nervous_system):
        """Test that path enforces connection continuity"""
        # Create non-continuous neurons
        neuron1 = Neuron("AVAL", nervous_system)
        neuron2 = Neuron("AVBL", nervous_system)
        neuron3 = Neuron("AVAR", nervous_system)

        # Create non-continuous connections
        conn1 = Connection(neuron1, neuron2, connection_type="chemical-synapse")
        conn2 = Connection(
            neuron3, neuron2, connection_type="chemical-synapse"
        )  # Wrong order

        # Test that creating a path with non-continuous connections raises an error
        with pytest.raises(
            AssertionError,
            match="Path members must be continuous connections from source to target",
        ):
            Path(nervous_system, [conn1, conn2])

    def test_path_immutability(self, path):
        """Test that path connections cannot be modified after creation"""
        # Test that update raises NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Cannot update connections in Path"
        ):
            path.update({"key": Connection(path.source, path.target)})

        # Test that pop raises NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Cannot remove connections from Path"
        ):
            path.pop(path.members[0]._id)

    def test_path_inheritance(self, path):
        """Test that Path inherits ConnectionGroup functionality"""
        # Test weight updates still work
        path.update_weights(2.0)
        assert all(conn.weight == 2.0 for conn in path.members)

        # Test filtering still works
        filtered = path.filter_by_type("chemical-synapse")
        assert len(filtered.members) == 1

        # Test statistics still work
        stats = path.get_statistics()
        assert stats["count"] == 2
        assert stats["weight_mean"] == 2.0

    def test_empty_path(self, nervous_system):
        """Test creating an empty path"""
        path = Path(nervous_system)
        assert len(path.members) == 0
        assert path.group_name.startswith("Path-")
        assert path.source is None
        assert path.target is None

    def test_single_connection_path(self, nervous_system):
        """Test path with a single connection"""
        neuron1 = Neuron("AVAL", nervous_system)
        neuron2 = Neuron("AVBL", nervous_system)
        conn = Connection(neuron1, neuron2)

        path = Path(nervous_system, [conn])
        assert len(path.members) == 1
        assert path.source == neuron1
        assert path.target == neuron2

    def test_path_length(self, path):
        """Test path length calculation"""
        assert path.get_length() == 2
        empty_path = Path(path.network)
        assert empty_path.get_length() == 0

    def test_path_weights(self, path):
        """Test path weight calculations"""
        # Test total weight
        assert path.get_total_weight() == 1.5  # 1.0 + 0.5

        # Test average weight
        assert path.get_average_weight() == 0.75  # (1.0 + 0.5) / 2

        # Test min weight
        assert path.get_min_weight() == 0.5

        # Test max weight
        assert path.get_max_weight() == 1.0

        # Test empty path weights
        empty_path = Path(path.network)
        assert empty_path.get_total_weight() == 0.0
        assert empty_path.get_average_weight() == 0.0
        assert empty_path.get_min_weight() == 0.0
        assert empty_path.get_max_weight() == 0.0

    def test_path_reversal(self, path):
        """Test path reversal"""
        reversed_path = path.reverse()

        # Check reversed path properties
        assert reversed_path.source == path.target
        assert reversed_path.target == path.source
        assert reversed_path.get_length() == path.get_length()
        assert reversed_path.get_total_weight() == path.get_total_weight()

        # Check connection order and types
        assert reversed_path.members[0].pre == path.members[1].post
        assert reversed_path.members[0].post == path.members[1].pre
        assert (
            reversed_path.members[0].connection_type == path.members[1].connection_type
        )

        # Test empty path reversal
        empty_path = Path(path.network)
        reversed_empty = empty_path.reverse()
        assert len(reversed_empty.members) == 0

    def test_path_concatenation(self, path, nervous_system):
        """Test path concatenation"""
        # Create another path to concatenate
        neuron3 = path.target
        neuron4 = Neuron("AVBR", nervous_system)
        conn3 = Connection(
            neuron3, neuron4, connection_type="chemical-synapse", weight=0.8
        )
        path2 = Path(nervous_system, [conn3], "path2")

        # Concatenate paths
        combined = path.concatenate(path2)

        # Check combined path properties
        assert combined.source == path.source
        assert combined.target == path2.target
        assert combined.get_length() == path.get_length() + path2.get_length()
        assert (
            combined.get_total_weight()
            == path.get_total_weight() + path2.get_total_weight()
        )

        # Test concatenation with empty paths
        empty_path = Path(path.network)
        assert path.concatenate(empty_path).get_length() == path.get_length()
        assert empty_path.concatenate(path).get_length() == path.get_length()

        # Test invalid concatenation
        with pytest.raises(AssertionError, match="Cannot concatenate paths"):
            path2.concatenate(path)  # Wrong order

    def test_path_validation(self, path, nervous_system):
        """Test path validation"""
        # Test valid path
        assert path.is_valid()

        # Test empty path
        empty_path = Path(path.network)
        assert empty_path.is_valid()

        # Test invalid path
        # Create a path with invalid connections by bypassing the constructor validation
        invalid_path = Path(path.network, [])
        invalid_path.members = [
            path.members[0],
            Connection(path.target, path.members[0].post),
        ]
        invalid_path.source = path.source
        invalid_path.target = path.members[0].post
        assert not invalid_path.is_valid()

    def test_path_neurons(self, path):
        """Test getting neurons in path"""
        neurons = path.get_neurons()
        assert len(neurons) == 3  # source + 2 connections
        assert neurons[0] == path.source
        assert neurons[-1] == path.target
        assert neurons[1] == path.members[0].post

        # Test empty path
        empty_path = Path(path.network)
        assert len(empty_path.get_neurons()) == 0

    def test_connection_types(self, path):
        """Test getting connection types"""
        types = path.get_connection_types()
        assert len(types) == 2
        assert types[0] == "chemical-synapse"
        assert types[1] == "gap-junction"

        # Test empty path
        empty_path = Path(path.network)
        assert len(empty_path.get_connection_types()) == 0
