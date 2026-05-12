"""
Tests for new NeuronGroup and ConnectionGroup set operations
(union, intersection, difference) and operator overloads.

Extends the existing test_cedne.py suite.
"""

import pytest
from cedne.core.network import NervousSystem
from cedne.core.neuron import Neuron, NeuronGroup
from cedne.core.connection import Connection, ConnectionGroup


# ===========================================================================
# NeuronGroup set operations
# ===========================================================================


class TestNeuronGroupSetOperations:
    @pytest.fixture
    def setup(self):
        """Create a network with 6 neurons and two overlapping NeuronGroups."""
        net = NervousSystem()
        neurons = [Neuron(f"N{i}", net) for i in range(6)]
        # Group A: N0, N1, N2, N3
        ga = NeuronGroup(net, members=neurons[:4], group_name="ga")
        # Group B: N2, N3, N4, N5
        gb = NeuronGroup(net, members=neurons[2:6], group_name="gb")
        return net, neurons, ga, gb

    # --- union ---

    def test_union_members(self, setup):
        net, neurons, ga, gb = setup
        result = ga.union(gb, group_name="ga_u_gb")
        assert set(result.neurons.keys()) == {f"N{i}" for i in range(6)}

    def test_union_operator(self, setup):
        _, _, ga, gb = setup
        result = ga | gb
        assert len(result) == 6

    def test_union_registered_in_network(self, setup):
        net, _, ga, gb = setup
        result = ga.union(gb, group_name="ga_u_gb")
        assert "ga_u_gb" in net.groups

    def test_union_auto_name(self, setup):
        _, _, ga, gb = setup
        result = ga | gb
        assert "ga" in result.group_name and "gb" in result.group_name

    # --- intersection ---

    def test_intersection_members(self, setup):
        _, _, ga, gb = setup
        result = ga.intersection(gb, group_name="ga_i_gb")
        assert set(result.neurons.keys()) == {"N2", "N3"}

    def test_intersection_operator(self, setup):
        _, _, ga, gb = setup
        result = ga & gb
        assert len(result) == 2

    def test_intersection_disjoint(self):
        net = NervousSystem()
        n1 = Neuron("A", net)
        n2 = Neuron("B", net)
        g1 = NeuronGroup(net, [n1], group_name="only_a")
        g2 = NeuronGroup(net, [n2], group_name="only_b")
        result = g1 & g2
        assert len(result) == 0

    # --- difference ---

    def test_difference_members(self, setup):
        _, _, ga, gb = setup
        result = ga.difference(gb, group_name="ga_d_gb")
        assert set(result.neurons.keys()) == {"N0", "N1"}

    def test_difference_operator(self, setup):
        _, _, ga, gb = setup
        result = ga - gb
        assert len(result) == 2

    def test_difference_asymmetric(self, setup):
        _, _, ga, gb = setup
        ab = ga - gb
        ba = gb - ga
        assert set(ab.neurons.keys()) == {"N0", "N1"}
        assert set(ba.neurons.keys()) == {"N4", "N5"}

    def test_difference_self(self, setup):
        net, _, ga, _ = setup
        result = ga.difference(ga, group_name="ga_self_diff")
        assert len(result) == 0

    # --- validation ---

    def test_different_networks_raises(self):
        net1 = NervousSystem()
        net2 = NervousSystem()
        n1 = Neuron("A", net1)
        n2 = Neuron("B", net2)
        g1 = NeuronGroup(net1, [n1], group_name="g1")
        g2 = NeuronGroup(net2, [n2], group_name="g2")
        with pytest.raises(AssertionError, match="same network"):
            g1.union(g2)

    def test_invalid_operand_type(self, setup):
        _, _, ga, _ = setup
        with pytest.raises(AssertionError, match="NeuronGroup"):
            ga.union("not_a_group")


# ===========================================================================
# ConnectionGroup set operations
# ===========================================================================


class TestConnectionGroupSetOperations:
    @pytest.fixture
    def setup(self):
        """Create a network with connections and two overlapping ConnectionGroups."""
        net = NervousSystem()
        neurons = [Neuron(f"N{i}", net) for i in range(5)]
        c01 = Connection(neurons[0], neurons[1], uid="c01")
        c12 = Connection(neurons[1], neurons[2], uid="c12")
        c23 = Connection(neurons[2], neurons[3], uid="c23")
        c34 = Connection(neurons[3], neurons[4], uid="c34")
        # Group A: c01, c12, c23
        cga = ConnectionGroup(net, members=[c01, c12, c23], group_name="cga")
        # Group B: c12, c23, c34
        cgb = ConnectionGroup(net, members=[c12, c23, c34], group_name="cgb")
        return net, cga, cgb

    def test_union(self, setup):
        _, cga, cgb = setup
        result = cga | cgb
        assert len(result) == 4

    def test_intersection(self, setup):
        _, cga, cgb = setup
        result = cga & cgb
        assert len(result) == 2

    def test_difference(self, setup):
        _, cga, cgb = setup
        result = cga - cgb
        assert len(result) == 1

    def test_difference_reverse(self, setup):
        _, cga, cgb = setup
        result = cgb - cga
        assert len(result) == 1

    def test_union_registered_in_network(self, setup):
        net, cga, cgb = setup
        result = cga.union(cgb, group_name="cga_u_cgb")
        assert "cga_u_cgb" in net.groups

    def test_invalid_operand_type(self, setup):
        _, cga, _ = setup
        with pytest.raises(AssertionError, match="ConnectionGroup"):
            cga.union("not_a_group")

    def test_different_networks_raises(self):
        net1 = NervousSystem()
        net2 = NervousSystem()
        n1 = Neuron("A", net1)
        n2 = Neuron("B", net1)
        n3 = Neuron("C", net2)
        n4 = Neuron("D", net2)
        c1 = Connection(n1, n2, uid="x1")
        c2 = Connection(n3, n4, uid="x2")
        cg1 = ConnectionGroup(net1, [c1], group_name="cg_net1")
        cg2 = ConnectionGroup(net2, [c2], group_name="cg_net2")
        with pytest.raises(AssertionError, match="same network"):
            cg1 | cg2
