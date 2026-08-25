"""Tests for the chained-FFL sampling machinery in cedne.utils.graphtools.

The core guarantee: ``ChainSampler.exhaustive`` reproduces VF2 induced
subgraph-isomorphism counts (``NervousSystem.search_motifs`` semantics)
exactly, for sequential hierarchies and intermediate node-chains of
lengths 1-3, including the autapse rule. The sampled estimator must be
consistent with the exhaustive count within its confidence interval.
"""

import networkx as nx
import numpy as np
import pytest

from cedne.utils import (
    ChainSampler,
    enumerate_induced_ffls,
    make_hypermotifs,
    remove_autapse_nodes,
    return_triads,
)


def _ffl_motif():
    # Relabelled so node 3 is the output apex, as in the motif notebooks.
    return nx.relabel_nodes(return_triads()["030T"], {1: 1, 2: 3, 3: 2})


def _chain_motif(length, join_kind):
    ffl = _ffl_motif()
    if length == 1:
        return ffl
    mapping = (3, 1) if join_kind == "seq" else (2, 1)
    return make_hypermotifs(ffl, length, [mapping])


def _vf2_count(graph, motif):
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(graph, motif)
    return sum(1 for _ in matcher.subgraph_isomorphisms_iter())


def _random_digraph(n_nodes, edge_prob, seed):
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    for a in range(n_nodes):
        for b in range(n_nodes):
            if a != b and rng.random() < edge_prob:
                graph.add_edge(a, b)
    return graph


class TestExhaustiveMatchesVF2:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("join_kind", ["seq", "int"])
    @pytest.mark.parametrize("length", [1, 2, 3])
    def test_random_graphs(self, seed, join_kind, length):
        graph = _random_digraph(n_nodes=30, edge_prob=0.12, seed=seed)
        sampler = ChainSampler(list(graph.edges()), seed=seed)
        expected = _vf2_count(graph, _chain_motif(length, join_kind))
        assert sampler.exhaustive(length, join_kind) == expected

    def test_dense_graph_with_reciprocals(self):
        # Heavier reciprocal-edge density stresses the induced check.
        graph = _random_digraph(n_nodes=20, edge_prob=0.3, seed=7)
        sampler = ChainSampler(list(graph.edges()), seed=7)
        for join_kind in ("seq", "int"):
            for length in (1, 2, 3):
                expected = _vf2_count(graph, _chain_motif(length, join_kind))
                assert sampler.exhaustive(length, join_kind) == expected


class TestAutapseSemantics:
    def test_autapse_node_excluded_like_vf2(self):
        # A self-looped node cannot match any motif position under VF2
        # induced semantics; remove_autapse_nodes must reproduce that.
        graph = _random_digraph(n_nodes=25, edge_prob=0.15, seed=11)
        looped = list(graph.nodes())[:3]
        for node in looped:
            graph.add_edge(node, node)
        vf2 = _vf2_count(graph, _ffl_motif())
        kept, autapse_nodes = remove_autapse_nodes(list(graph.edges()))
        assert autapse_nodes == set(looped)
        sampler = ChainSampler(kept, seed=0)
        assert len(sampler.ffls) == vf2

    def test_sampler_rejects_self_loops(self):
        with pytest.raises(ValueError):
            ChainSampler([(1, 2), (2, 2)], seed=0)


class TestCandidateTotal:
    def test_matches_bruteforce_join_enumeration(self):
        graph = _random_digraph(n_nodes=25, edge_prob=0.15, seed=5)
        sampler = ChainSampler(list(graph.edges()), seed=5)
        for join_kind in ("seq", "int"):
            join_pos = 2 if join_kind == "seq" else 1
            pairs = sum(
                1 for f1 in sampler.ffls for f2 in sampler.ffls if f2[0] == f1[join_pos]
            )
            assert sampler.candidate_total(2, join_kind) == pairs

    def test_ffl_enumeration_no_duplicates(self):
        graph = _random_digraph(n_nodes=30, edge_prob=0.15, seed=9)
        ffls, _ = enumerate_induced_ffls(list(graph.edges()))
        assert len(ffls) == len(set(ffls))


class TestEstimator:
    def test_estimate_consistent_with_exhaustive(self):
        graph = _random_digraph(n_nodes=40, edge_prob=0.12, seed=13)
        sampler = ChainSampler(list(graph.edges()), seed=13)
        for join_kind in ("seq", "int"):
            exact = sampler.exhaustive(2, join_kind)
            estimate, ci95, accepted = sampler.estimate(2, join_kind, 20_000)
            # 1.96-sigma CI; allow 3 sigma for a deterministic-seed test.
            assert abs(estimate - exact) <= max(3 * ci95 / 1.96, 1e-9)
            for chain in accepted[:50]:
                assert len(set(chain)) == 5

    def test_accepted_chains_are_valid_matches(self):
        graph = _random_digraph(n_nodes=40, edge_prob=0.12, seed=17)
        sampler = ChainSampler(list(graph.edges()), seed=17)
        motif = _chain_motif(2, "seq")
        _, _, accepted = sampler.estimate(2, "seq", 5_000)
        motif_nodes = ["1.1", "1.2", "1.3-2.1", "2.2", "2.3"]
        for chain in accepted[:20]:
            mapping = dict(zip(chain, motif_nodes))
            sub = graph.subgraph(chain)
            relabelled = nx.relabel_nodes(sub, mapping)
            assert set(relabelled.edges()) == set(motif.edges())


@pytest.mark.slow
class TestWormGroundTruth:
    """Exact reproduction of the worm-connectome counts used in the paper."""

    def test_worm_counts(self):
        from cedne import utils

        worm = utils.makeWorm(chem_only=True)
        nn = worm.networks["Neutral"]
        raw = list({(pre.name, post.name) for (pre, post, _uid) in nn.connections})
        edges, autapse_nodes = remove_autapse_nodes(raw)
        assert len(autapse_nodes) == 38
        sampler = ChainSampler(edges, seed=0)
        assert len(sampler.ffls) == 1380
        assert sampler.exhaustive(3, "seq") == 9414
        assert sampler.exhaustive(2, "seq") == 7115
        assert sampler.exhaustive(2, "int") == 3813
        assert sampler.exhaustive(3, "int") == 4483
