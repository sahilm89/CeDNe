from __future__ import annotations

import pytest

from cedne import NervousSystem, ChemicalSynapse, GapJunction, BulkConnection


@pytest.fixture
def small_network():
    network = NervousSystem(network="AdjacencyTest")
    network.create_neurons(["A", "B", "C"])

    a = network.neurons["A"]
    b = network.neurons["B"]
    c = network.neurons["C"]

    ChemicalSynapse(a, b, weight=2.0)
    ChemicalSynapse(a, b, weight=3.0)
    GapJunction(b, c, weight=5.0)
    BulkConnection(c, a, uid="bulk-1", connection_type="monoamine", weight=7.0)

    return network


def test_adjacency_default_order_is_deterministic_by_name(small_network):
    adjacency = small_network.adjacency()

    assert adjacency.shape == (3, 3)
    assert adjacency.tolist() == [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]


def test_adjacency_weighted_sums_multiple_edges(small_network):
    adjacency = small_network.adjacency(weighted=True, connection_type="chemical")

    assert adjacency.tolist() == [
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_adjacency_filters_gap_junctions_and_bulk(small_network):
    gap = small_network.adjacency(connection_type="gap-junction", weighted=True)
    bulk = small_network.adjacency(connection_type="bulk", weighted=True)

    assert gap.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
    ]
    assert bulk.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
    ]


def test_adjacency_accepts_multiple_connection_type_selectors(small_network):
    adjacency = small_network.adjacency(
        connection_type=["chemical", "monoamine"],
        weighted=True,
    )

    assert adjacency.tolist() == [
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
    ]


def test_adjacency_respects_explicit_order_by_name(small_network):
    adjacency = small_network.adjacency(order=["C", "A", "B"], weighted=True)

    assert adjacency.tolist() == [
        [0.0, 7.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 0.0],
    ]


def test_adjacency_rejects_unknown_neuron_in_order(small_network):
    with pytest.raises(KeyError, match="Neuron 'Z' not found in network"):
        small_network.adjacency(order=["A", "Z"])


def test_adjacency_rejects_non_string_connection_type_entries(small_network):
    with pytest.raises(TypeError, match="connection_type entries must be strings"):
        small_network.adjacency(connection_type=["chemical", 3])
