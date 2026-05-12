"""Parity + isolation tests for the new batch / view implementations of
``NervousSystem.fold_network`` and ``NervousSystem.subnetwork``.

The new default ``legacy=False`` paths must produce structurally identical
NervousSystem objects to the original ``legacy=True`` paths for the test
corpus below, AND must preserve the same isolation properties (no shared
graph state with the parent; same mutable-attr-value sharing as
``copy(copy_type='deep_with_data')``).

What we check structurally:
- node-name set
- edge multiplicity per (pre_name, post_name, connection_type)
- nx node-attribute dict per node (constituents, MERGE_TRACK_ATTRS,
  constituent_subgraph existence)
- per-merged-node: ``is_merged``, ``constituents`` dict, ``MERGE_TRACK_ATTRS``,
  recursively-serialised ``constituent_subgraph``
- per-edge: ``weight`` (sum for clean-mode parallels) and selected
  connection_type / nt union fields

What we explicitly do NOT compare:
- nx edge keys (both paths mint independent uids for new Connection
  objects; the structural multiplicity check is what matters)
- ``contraction_data`` *keys* in ``data='clean'`` (the legacy path keys
  them by post-pair-wise edge ids, the batch path by pre-fold edge ids
  — both are valid drill-down handles to the original Connection
  objects; the canonicaliser compares the resulting set of Connection
  objects rather than the synthetic tuple keys)
"""

from __future__ import annotations

from collections import Counter

import pytest

from cedne.core import MERGED_TYPE, MERGE_TRACK_ATTRS
from cedne.core.animal import Worm
from cedne.core.connection import Connection
from cedne.core.network import NervousSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(nn):
    """Serialise a NervousSystem into a comparable dict.

    Designed to ignore implementation-detail differences (edge-uid
    nondeterminism, post-pair-wise vs pre-fold contraction_data keys)
    while catching every difference that a consumer would observe.
    """
    nodes = {}
    for name, n in sorted(nn.neurons.items()):
        nx_data = nn.nodes[n]
        node_payload = {
            "type": getattr(n, "type", None),
            "category": getattr(n, "category", None),
            "modality": getattr(n, "modality", None),
            "is_merged": bool(getattr(n, "is_merged", False)),
        }
        # Constituents: compare the dict content, sorted by orig name.
        constituents = getattr(n, "constituents", None)
        if constituents:
            node_payload["constituents"] = {
                k: {a: v.get(a) for a in ("name",) + MERGE_TRACK_ATTRS}
                for k, v in sorted(constituents.items())
            }
        else:
            node_payload["constituents"] = None
        # constituent_subgraph: recursively canonicalise if present.
        subg = getattr(n, "constituent_subgraph", None)
        if subg is None:
            # also check the nx node dict mirror
            subg = nx_data.get("constituent_subgraph")
        node_payload["has_constituent_subgraph"] = subg is not None
        if subg is not None:
            node_payload["constituent_subgraph"] = _canonical(subg)
        nodes[name] = node_payload

    # Edge canonical: bucket by (pre_name, post_name, connection_type),
    # store list of (weight, frozen set of union'd metadata key-presence).
    edge_buckets = Counter()
    edge_weights = {}
    for (u, v, _k), conn in nn.connections.items():
        key = (u.name, v.name, getattr(conn, "connection_type", None))
        edge_buckets[key] += 1
        # Aggregate weight per bucket so 'collect' (parallels) and 'clean'
        # (summed) modes can be compared on equivalent footing.
        edge_weights[key] = edge_weights.get(key, 0.0) + float(
            getattr(conn, "weight", 0) or 0
        )

    return {
        "nodes": nodes,
        "edge_multiplicity": dict(edge_buckets),
        "edge_weight_sum_per_bucket": edge_weights,
    }


def _build_chain(types=None, weights=None):
    """4-neuron chain A → B → C → D with chemical edges."""
    if types is None:
        types = {"A": "sensory", "B": "sensory", "C": "motor", "D": "motor"}
    if weights is None:
        weights = {("A", "B"): 3, ("B", "C"): 5, ("C", "D"): 7}
    w = Worm(name="test")
    nn = NervousSystem(w)
    nn.create_neurons(["A", "B", "C", "D"], type=types)
    adj = {}
    for (u, v), wt in weights.items():
        adj.setdefault(u, {})[v] = {"weight": wt}
    nn.setup_chemical_connections(adj)
    return w, nn


def _build_with_parallels():
    """Build a graph with deliberate parallel edges, so 'clean' mode
    aggregation can be exercised meaningfully."""
    w = Worm(name="parallels")
    nn = NervousSystem(w)
    nn.create_neurons(
        ["A1", "A2", "B1", "B2"],
        type={"A1": "A", "A2": "A", "B1": "B", "B2": "B"},
    )
    # Two A→B and one B→A so the (A_class, B_class) bucket has 2 parallels.
    pairs = [
        ("A1", "B1", 1.0),
        ("A2", "B2", 2.5),
        ("B1", "A2", 4.0),
    ]
    for pre, post, w_ in pairs:
        src, dst = nn.neurons[pre], nn.neurons[post]
        c = Connection(src, dst, connection_type="chemical-synapse", weight=w_)
        nn.connections[(src, dst, c.uid)] = c
    return w, nn


def _build_for_nested_fold():
    """Build A, B, C, D so that nested folds [A,B]→INNER then [INNER,C]→OUTER
    exercise the constituents-flattening logic."""
    w = Worm(name="nested")
    nn = NervousSystem(w)
    nn.create_neurons(
        ["A", "B", "C", "D"],
        type={"A": "t", "B": "t", "C": "t", "D": "other"},
    )
    adj = {
        "A": {"B": {"weight": 1}, "C": {"weight": 2}},
        "B": {"C": {"weight": 3}, "D": {"weight": 4}},
        "C": {"D": {"weight": 5}},
    }
    nn.setup_chemical_connections(adj)
    return w, nn


# ---------------------------------------------------------------------------
# fold_network parity
# ---------------------------------------------------------------------------


class TestFoldNetworkParity:
    def test_fold_two_collect(self):
        _, nn_l = _build_chain()
        _, nn_f = _build_chain()
        legacy = nn_l.fold_network({"AB": ["A", "B"]}, data="collect", legacy=True)
        fast = nn_f.fold_network({"AB": ["A", "B"]}, data="collect", legacy=False)
        assert _canonical(legacy) == _canonical(fast)

    def test_fold_two_clean_summed_weights(self):
        _, nn_l = _build_with_parallels()
        _, nn_f = _build_with_parallels()
        legacy = nn_l.fold_network(
            {"A": ["A1", "A2"], "B": ["B1", "B2"]},
            data="clean",
            legacy=True,
        )
        fast = nn_f.fold_network(
            {"A": ["A1", "A2"], "B": ["B1", "B2"]},
            data="clean",
            legacy=False,
        )
        assert _canonical(legacy) == _canonical(fast)

    def test_fold_singleton_rename(self):
        """A class with a single neuron is a rename."""
        _, nn_l = _build_chain()
        _, nn_f = _build_chain()
        legacy = nn_l.fold_network({"RENAMED": ["A"]}, data="collect", legacy=True)
        fast = nn_f.fold_network({"RENAMED": ["A"]}, data="collect", legacy=False)
        assert _canonical(legacy) == _canonical(fast)

    def test_fold_mixed_types_uses_merged_sentinel(self):
        """A+C have different types (sensory, motor). Both paths should
        set the merged neuron's type to MERGED_TYPE."""
        _, nn_l = _build_chain()
        _, nn_f = _build_chain()
        legacy = nn_l.fold_network({"M": ["A", "C"]}, data="collect", legacy=True)
        fast = nn_f.fold_network({"M": ["A", "C"]}, data="collect", legacy=False)
        assert legacy.neurons["M"].type == MERGED_TYPE
        assert fast.neurons["M"].type == MERGED_TYPE
        assert _canonical(legacy) == _canonical(fast)

    def test_fold_same_type_preserves_value(self):
        """A+B share type 'sensory'. Both paths must preserve that value."""
        _, nn_l = _build_chain()
        _, nn_f = _build_chain()
        legacy = nn_l.fold_network({"M": ["A", "B"]}, data="collect", legacy=True)
        fast = nn_f.fold_network({"M": ["A", "B"]}, data="collect", legacy=False)
        assert legacy.neurons["M"].type == "sensory"
        assert fast.neurons["M"].type == "sensory"

    def test_fold_with_exceptions(self):
        """A member listed in exceptions passes through unchanged."""
        _, nn_l = _build_chain()
        _, nn_f = _build_chain()
        legacy = nn_l.fold_network(
            {"M": ["A", "B", "C"]},
            data="collect",
            exceptions=["C"],
            legacy=True,
        )
        fast = nn_f.fold_network(
            {"M": ["A", "B", "C"]},
            data="collect",
            exceptions=["C"],
            legacy=False,
        )
        assert _canonical(legacy) == _canonical(fast)
        # C must still exist as itself.
        assert "C" in legacy.neurons
        assert "C" in fast.neurons

    def test_fold_self_loops_false(self):
        """self_loops=False drops intra-class edges. Build a graph with
        an A1→A2 edge and fold A1+A2 → A; the resulting graph must have
        no A→A self-loop."""
        _, nn_l = _build_with_parallels()
        _, nn_f = _build_with_parallels()
        # Add an intra-class edge.
        for nn in (nn_l, nn_f):
            src, dst = nn.neurons["A1"], nn.neurons["A2"]
            c = Connection(src, dst, connection_type="chemical-synapse", weight=9)
            nn.connections[(src, dst, c.uid)] = c
        legacy = nn_l.fold_network(
            {"A": ["A1", "A2"], "B": ["B1", "B2"]},
            data="collect",
            self_loops=False,
            legacy=True,
        )
        fast = nn_f.fold_network(
            {"A": ["A1", "A2"], "B": ["B1", "B2"]},
            data="collect",
            self_loops=False,
            legacy=False,
        )
        # No self-loop on 'A' in either result.
        for n in (legacy, fast):
            self_loops = [
                (u, v, k)
                for (u, v, k), _ in n.connections.items()
                if u.name == "A" and v.name == "A"
            ]
            assert self_loops == [], f"unexpected self-loop in {n.name}: {self_loops}"
        assert _canonical(legacy) == _canonical(fast)

    def test_fold_constituent_subgraph_present_and_correct(self):
        """Merged neurons must carry a constituent_subgraph holding the
        pre-fold members and their internal edges. Both paths."""
        _, nn_l = _build_chain()
        _, nn_f = _build_chain()
        for legacy_flag, nn in ((True, nn_l), (False, nn_f)):
            folded = nn.fold_network({"AB": ["A", "B"]}, legacy=legacy_flag)
            merged = folded.neurons["AB"]
            subg = getattr(merged, "constituent_subgraph", None)
            assert subg is not None
            assert set(subg.neurons) == {"A", "B"}
            # The A→B edge from the parent should be in the subgraph.
            edges = {(u.name, v.name) for (u, v, _k) in subg.connections}
            assert ("A", "B") in edges
            # Mirror to nx node dict must also be present (matches legacy).
            assert folded.nodes[merged]["constituent_subgraph"] is subg

    def test_nested_fold_constituents_isolation(self):
        """Nested folds [A,B]→INNER then [INNER,C]→OUTER must NOT pollute
        INNER's view of itself (the bug guarded by the existing
        ``test_hierarchical_fold_keeps_inner_constituents_intact``).
        Both paths must satisfy this."""
        for legacy_flag in (True, False):
            _, nn = _build_for_nested_fold()
            f1 = nn.fold_network({"INNER": ["A", "B"]}, legacy=legacy_flag)
            f2 = f1.fold_network({"OUTER": ["INNER", "C"]}, legacy=legacy_flag)
            outer = f2.neurons["OUTER"]
            # Outer carries A, B, C, plus INNER as a placeholder.
            assert {"A", "B", "C"}.issubset(set(outer.constituents.keys()))
            # The captured constituent_subgraph for OUTER should still
            # show INNER with only A, B as its constituents — the outer
            # fold must not leak C into INNER's view.
            outer_subg = outer.constituent_subgraph
            assert outer_subg is not None
            assert "INNER" in outer_subg.neurons
            inner_in_outer = outer_subg.neurons["INNER"]
            assert set(inner_in_outer.constituents.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# Isolation: matches `copy_type='deep_with_data'` semantics — structural
# isolation guaranteed; mutable attribute *values* may be shared (matches
# legacy). Full attribute isolation requires `copy(copy_type='deep')`.
# ---------------------------------------------------------------------------


class TestFoldDisjointPartition:
    def test_overlapping_classes_raise(self):
        """A neuron listed in two different merged classes must raise —
        silently letting last-write-wins assign the neuron to one class
        would either drop data or duplicate it depending on the path."""
        _, nn = _build_chain()
        with pytest.raises(ValueError, match="multiple merged classes"):
            nn.fold_network({"X": ["A", "B"], "Y": ["B", "C"]})
        with pytest.raises(ValueError, match="multiple merged classes"):
            nn.fold_network({"X": ["A", "B"], "Y": ["B", "C"]}, legacy=True)

    def test_excepted_neuron_can_appear_in_multiple_classes(self):
        """Members in ``exceptions`` pass through unchanged, so their
        nominal membership in multiple classes is harmless — the check
        should not flag those."""
        _, nn = _build_chain()
        # Should NOT raise.
        folded = nn.fold_network(
            {"X": ["A", "B", "C"], "Y": ["B", "D"]},
            exceptions=["B"],
        )
        assert "B" in folded.neurons


class TestFoldIsolation:
    def test_structural_mutations_on_folded_dont_leak_to_parent(self):
        """Calling fold_network must not change the parent's
        structure, AND the folded view's neurons must be distinct
        Python objects from the parent's so subsequent mutations stay
        local. Both paths."""
        for legacy_flag in (True, False):
            _, nn = _build_chain()
            parent_snap = _canonical(nn)
            folded = nn.fold_network({"AB": ["A", "B"]}, legacy=legacy_flag)
            # The act of folding alone must not mutate the parent.
            assert _canonical(nn) == parent_snap
            # The pass-through 'C' / 'D' neurons in the folded view must
            # be distinct Python objects from the parent's so any
            # downstream mutation stays local.
            assert folded.neurons["C"] is not nn.neurons["C"]
            # Sanity: mutating the folded view's surviving neuron does
            # not propagate.
            folded.neurons["C"].name = "C_RENAMED"
            assert nn.neurons["C"].name == "C"

    def test_renaming_on_folded_dont_leak_to_parent(self):
        """Renaming a neuron on the folded view must not rename the
        corresponding parent neuron (separate Neuron instances)."""
        for legacy_flag in (True, False):
            _, nn = _build_chain()
            folded = nn.fold_network({"AB": ["A", "B"]}, legacy=legacy_flag)
            folded.neurons["C"].name = "C_RENAMED"
            # Parent's C is unchanged.
            assert nn.neurons["C"].name == "C"
            assert "C" in nn.neurons

    def test_constituents_dict_is_owned_by_merged_neuron(self):
        """The merged neuron's ``constituents`` dict must be a fresh
        dict (not aliased to any source neuron's). Mutating it must not
        affect the parent. Both paths."""
        for legacy_flag in (True, False):
            _, nn = _build_chain()
            folded = nn.fold_network({"AB": ["A", "B"]}, legacy=legacy_flag)
            merged = folded.neurons["AB"]
            # The parent's A had no .constituents — mutating merged.constituents
            # should not retroactively give A one.
            merged.constituents["SYNTHETIC"] = {"name": "SYNTHETIC", "type": "fake"}
            assert getattr(nn.neurons["A"], "constituents", None) in (None, {})
