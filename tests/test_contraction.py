"""Tests for ``NervousSystem.contract_neurons`` and ``contract_connections``.

Covers:
  - the pre-existing edge-redirection / weight-summing semantics,
  - the new ``Neuron.constituents`` / ``is_merged`` / ``constituent_types``
    API and the ``'merged'`` type-merge policy (Issue 10A),
  - the previously-broken ``copy_graph=True`` path,
  - round-tripping the new fields through ``contract_connections``'s
    internal ``create_neurons_from(data=True)`` pathway,
  - ``Neuron.to_dict()`` output for both merged and un-merged nodes.

These functions had **zero** unit-test coverage in CeDNe core prior to
this file, hence the breadth.
"""

from __future__ import annotations

import pytest

from cedne.core import MERGED_TYPE
from cedne.core.animal import Worm
from cedne.core.connection import Connection
from cedne.core.network import NervousSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_chain(types=None):
    """Build a 4-neuron chain A → B → C → D with named chemical edges.

    Returns ``(worm, network)``. Default types: A,B sensory; C,D motor.
    Override per-neuron types via ``types`` dict.
    """
    if types is None:
        types = {"A": "sensory", "B": "sensory", "C": "motor", "D": "motor"}
    w = Worm(name="test")
    nn = NervousSystem(w)
    nn.create_neurons(["A", "B", "C", "D"], type=types)
    nn.setup_chemical_connections(
        {
            "A": {"B": {"weight": 3}},
            "B": {"C": {"weight": 5}},
            "C": {"D": {"weight": 7}},
        }
    )
    return w, nn


def _edge_weights(nn):
    """Return ``{(pre.name, post.name): weight}`` for inspection."""
    out = {}
    for u, v, data in nn.edges(data=True):
        out[(u.name, v.name)] = data.get("weight")
    return out


# ===========================================================================
# Pre-merge baseline
# ===========================================================================


class TestUnmergedNeuronAPI:
    def test_unmerged_neuron_reports_is_merged_false(self):
        """A freshly-created neuron has no constituents and no merge state."""
        _, nn = _build_chain()
        for n in nn.neurons.values():
            assert n.is_merged is False
            assert n.constituent_types == []

    def test_unmerged_neuron_to_dict_omits_merged_fields(self):
        """``to_dict()`` keeps un-merged payloads slim — no merged-only keys."""
        _, nn = _build_chain()
        d = nn.neurons["A"].to_dict()
        assert "is_merged" not in d
        assert "constituents" not in d
        assert "constituent_types" not in d
        assert d["type"] == "sensory"


# ===========================================================================
# contract_neurons: edge-redirection mechanics
# ===========================================================================


class TestContractNeuronsEdgeMechanics:
    def test_target_node_is_removed_and_renamed_source_survives(self):
        """The target neuron disappears; the source is renamed in place."""
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "A_B")
        assert "A" not in nn.neurons
        assert "B" not in nn.neurons
        assert "A_B" in nn.neurons
        # Other neurons untouched.
        assert "C" in nn.neurons and "D" in nn.neurons

    def test_target_inbound_edges_redirect_to_merged_source(self):
        """Edges originally landing on the target now land on the merged
        neuron — so analyses see the same wiring with one fewer node.
        """
        _, nn = _build_chain()
        # Pre-condition: A→B exists.
        assert ("A", "B") in _edge_weights(nn)
        nn.contract_neurons(("C", "B"), "BC")
        weights = _edge_weights(nn)
        # A used to point at B; should now point at BC.
        assert ("A", "BC") in weights
        # The B→C edge collapses into a self-loop A → ... — no, that's
        # the redirection of B's inbound edge from A. The B→C edge had
        # B as source and C as target; after merging B into C (named
        # BC), that edge becomes BC→BC, a self-loop. Verify either kept
        # (default self_loops=True) or dropped.
        assert ("BC", "BC") in weights  # default self_loops=True

    def test_self_loops_false_drops_self_loop_from_redirection(self):
        """``self_loops=False`` is forwarded to networkx so the loop
        produced by redirecting B→C onto C→C is dropped at merge time.
        """
        _, nn = _build_chain()
        nn.contract_neurons(("C", "B"), "BC", self_loops=False)
        weights = _edge_weights(nn)
        assert ("BC", "BC") not in weights
        # But A→BC (formerly A→B) still survives.
        assert ("A", "BC") in weights

    def test_redirected_edge_preserves_weight(self):
        """Weights on edges that survive redirection don't get clobbered."""
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        w = _edge_weights(nn)
        # The B→C edge becomes AB→C with the original weight.
        assert w[("AB", "C")] == 5


# ===========================================================================
# contract_neurons: new merge-provenance API (Issue 10A)
# ===========================================================================


class TestContractNeuronsConstituents:
    def test_first_merge_records_both_originals(self):
        """After contracting A and B, the surviving neuron's constituents
        list includes the original names of *both* — not just the target.
        Otherwise a future ``constituent_types`` call would miss the
        source's own type.
        """
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        merged = nn.neurons["AB"]
        assert set(merged.constituents.keys()) == {"A", "B"}
        assert merged.is_merged is True

    def test_same_type_constituents_preserve_type(self):
        """If all constituents share a type, the merged neuron keeps it.
        ``'merged'`` is reserved for the cross-type case so downstream
        code can rely on it as a signal of disagreement.
        """
        _, nn = _build_chain()  # A, B both sensory
        nn.contract_neurons(("A", "B"), "AB")
        merged = nn.neurons["AB"]
        assert merged.type == "sensory"
        assert merged.constituent_types == ["sensory"]

    def test_different_type_constituents_force_merged_sentinel(self):
        """Cross-type contraction sets type to ``MERGED_TYPE``."""
        _, nn = _build_chain()
        nn.contract_neurons(("A", "C"), "AC")  # sensory + motor
        merged = nn.neurons["AC"]
        assert merged.type == MERGED_TYPE
        assert merged.constituent_types == ["motor", "sensory"]

    def test_merge_policy_applies_to_category_and_modality_too(self):
        """Issue 10 follow-up — the same all-same/mixed → 'merged'
        policy covers every enum-like attribute (type, category,
        modality), not just type. Silent inheritance of category and
        modality from the source neuron was the same gap that motivated
        the type policy in the first place.
        """
        w = Worm(name="enum")
        nn = NervousSystem(w)
        nn.create_neurons(
            ["A", "B"],
            type={"A": "sensory", "B": "sensory"},  # agree
            category={"A": "amphid", "B": "phasmid"},  # disagree
            modality={"A": "chemosensory", "B": "chemosensory"},  # agree
        )
        nn.contract_neurons(("A", "B"), "AB")
        merged = nn.neurons["AB"]
        # Type and modality preserved (constituents agreed on each).
        assert merged.type == "sensory"
        assert merged.modality == "chemosensory"
        # Category was different — must hit the sentinel rather than
        # silently inheriting 'amphid' from the source.
        assert merged.category == MERGED_TYPE
        assert merged.constituent_categories == ["amphid", "phasmid"]
        assert merged.constituent_modalities == ["chemosensory"]

    def test_constituent_values_tolerates_non_string_attrs(self):
        """Real-world pandas-backed loaders sometimes leave attributes
        as ``float('nan')`` when the source row is missing — and the
        resulting mixed set blew up ``sorted({...})`` with a TypeError
        on /graph requests. Regression: drop any non-string value at
        read time so ``constituent_*`` always returns a sortable list.
        """
        import math

        w = Worm(name="nan")
        nn = NervousSystem(w)
        nn.create_neurons(["A", "B"], type={"A": "sensory", "B": "sensory"})
        # Inject NaN modality on one constituent the way upstream
        # loaders do — direct attribute set, no validation.
        nn.neurons["A"].modality = "chemosensory"
        nn.neurons["B"].modality = math.nan
        nn.contract_neurons(("A", "B"), "AB")
        merged = nn.neurons["AB"]
        # Must not raise; must drop the NaN.
        assert merged.constituent_modalities == ["chemosensory"]

    def test_merged_neuron_to_dict_includes_category_and_modality_when_set(self):
        """Serialisation surfaces the new constituent_categories and
        constituent_modalities lists, but only when there's actual
        information to report (un-merged or all-empty constituents stay
        payload-slim)."""
        w = Worm(name="enum")
        nn = NervousSystem(w)
        nn.create_neurons(
            ["A", "B"],
            type={"A": "sensory", "B": "motor"},
            category={"A": "amphid", "B": "phasmid"},
            modality={"A": "chemosensory", "B": ""},
        )
        nn.contract_neurons(("A", "B"), "AB")
        d = nn.neurons["AB"].to_dict()
        assert sorted(d["constituent_categories"]) == ["amphid", "phasmid"]
        assert d["constituent_modalities"] == ["chemosensory"]
        # Each constituent dict carries every tracked attribute.
        for entry in d["constituents"]:
            assert "type" in entry
            assert "category" in entry
            assert "modality" in entry

    def test_transitive_merge_accumulates_all_originals(self):
        """A+B → AB, then AB+C → ABC. ABC.constituents must include
        A, B, AND C (not just the most recent target). Otherwise the
        type-policy decision uses an incomplete set of types.
        """
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        nn.contract_neurons(("AB", "C"), "ABC")
        merged = nn.neurons["ABC"]
        # All original names appear (note: 'AB' may also appear as a
        # placeholder pointing at the intermediate merged-name). We
        # assert the originals are present rather than asserting the
        # set is exactly the originals — the placeholder is harmless.
        assert {"A", "B", "C"}.issubset(merged.constituents.keys())
        assert set(merged.constituent_types) == {"sensory", "motor"}
        assert merged.type == MERGED_TYPE

    def test_contract_neurons_appends_event_to_animal_history(self):
        """A contraction is a structural mutation — it must record an
        Event on the animal's history log so the web app's
        push_network_history dedup doesn't silently skip the snapshot
        (the cedne event log is the dedup key). Regression: prior to
        this fix, contractions were missing from the commit DAG.
        """
        _, nn = _build_chain()
        # Worm.history starts empty; nn.worm.history is the canonical place.
        # Other loaders (e.g. the chain builder) may have populated it; we
        # capture the count before to assert the contraction appends one.
        initial_count = len(getattr(nn.worm, "history", []) or [])
        nn.contract_neurons(("A", "B"), "AB")
        log = getattr(nn.worm, "history", []) or []
        assert len(log) == initial_count + 1
        # The newest event names the operation we just performed.
        assert log[-1].op == "contract_neurons"

    def test_contract_connections_appends_event_to_animal_history(self):
        """Same reasoning as the contract_neurons case above: the cedne
        event log must reflect contract_connections so its event id
        differs from the prior head and the web app persists a fresh
        history snapshot.
        """
        _, nn = _build_chain()
        # Build a trivial single-conn dict (the function accepts arbitrary
        # parallel-edge collapses; one is enough to exercise @record).
        a = nn.neurons["A"]
        b = nn.neurons["B"]
        ab_conn = next(c for c in nn.connections.values() if c.pre is a and c.post is b)
        initial_count = len(getattr(nn.worm, "history", []) or [])
        nn.contract_connections({(a, b, "chemical-synapse"): [ab_conn]})
        log = getattr(nn.worm, "history", []) or []
        assert len(log) == initial_count + 1
        assert log[-1].op == "contract_connections"

    def test_merge_provenance_mirrors_to_nx_node_attrs(self):
        """Mirror to networkx node attrs so paths that reconstruct
        neurons via ``create_neurons_from(data=True)`` (notably
        ``contract_connections`` and ``copy(copy_type='deep_with_data')``)
        propagate the merge state.
        """
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        merged = nn.neurons["AB"]
        nx_attrs = nn.nodes[merged]
        assert "constituents" in nx_attrs
        assert nx_attrs["constituents"] == merged.constituents
        assert nx_attrs.get("type") == "sensory"

    def test_constituents_round_trip_through_create_neurons_from(self):
        """Cloning a contracted graph via ``create_neurons_from(data=True)``
        must preserve the merge state. This is the path
        ``contract_connections`` uses internally — without the nx-attr
        mirror above, every contract_connections call would silently
        drop merge metadata.
        """
        _, nn = _build_chain()
        nn.contract_neurons(("A", "C"), "AC")

        clone = NervousSystem(nn.worm, network="clone")
        clone.create_neurons_from(nn, data=True)
        cloned = clone.neurons["AC"]
        assert cloned.is_merged is True
        assert {"A", "C"}.issubset(cloned.constituents.keys())
        assert cloned.constituent_types == ["motor", "sensory"]
        assert cloned.type == MERGED_TYPE


# ===========================================================================
# contract_neurons: copy_graph=True (previously broken)
# ===========================================================================


class TestContractNeuronsCopyGraph:
    def test_copy_graph_true_returns_new_graph_with_merge(self):
        """Regression: the recursive call previously passed a 3-tuple
        as ``pair``, which destructure-rejected with ValueError. The
        ``copy_graph=True`` path now returns a fresh graph with the
        merge applied.
        """
        _, nn = _build_chain()
        new = nn.contract_neurons(("A", "B"), "AB", copy_graph=True)
        # Returned graph is a different NervousSystem instance.
        assert new is not nn
        # Merge applied on the copy.
        assert "AB" in new.neurons
        assert new.neurons["AB"].is_merged is True

    def test_copy_graph_true_does_not_mutate_original(self):
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB", copy_graph=True)
        # Original unchanged.
        assert "A" in nn.neurons
        assert "B" in nn.neurons
        assert "AB" not in nn.neurons
        assert nn.neurons["A"].is_merged is False


# ===========================================================================
# Neuron.to_dict() with merge state
# ===========================================================================


class TestMergedNeuronSerialization:
    def test_merged_neuron_to_dict_includes_constituents(self):
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        d = nn.neurons["AB"].to_dict()
        assert d["is_merged"] is True
        assert d["constituent_types"] == ["sensory"]
        # constituents serialised as a list of {name, type} dicts.
        assert isinstance(d["constituents"], list)
        names = {c["name"] for c in d["constituents"]}
        assert {"A", "B"} == names
        for c in d["constituents"]:
            assert c["type"] == "sensory"

    def test_merged_neuron_to_dict_uses_merged_sentinel_for_mixed_types(self):
        _, nn = _build_chain()
        nn.contract_neurons(("A", "C"), "AC")
        d = nn.neurons["AC"].to_dict()
        assert d["type"] == MERGED_TYPE
        assert sorted(d["constituent_types"]) == ["motor", "sensory"]


# ===========================================================================
# contract_connections — collapse parallel edges
# ===========================================================================


class TestContractConnectionsBasics:
    def _two_parallel_edges(self):
        """Build A → B with two chemical synapses (parallel edges)."""
        w = Worm(name="ccx")
        nn = NervousSystem(w)
        nn.create_neurons(["A", "B"], type={"A": "sensory", "B": "motor"})
        a, b = nn.neurons["A"], nn.neurons["B"]
        c1 = Connection(
            a, b, connection_type="chemical-synapse", weight=2, ligands=["Glu"]
        )
        c2 = Connection(
            a, b, connection_type="chemical-synapse", weight=5, ligands=["Glu", "GABA"]
        )
        return nn, [c1, c2]

    def test_collapses_parallel_edges_and_sums_weights(self):
        nn, conns = self._two_parallel_edges()
        merged = nn.contract_connections(
            {
                (nn.neurons["A"], nn.neurons["B"], "chemical-synapse"): conns,
            }
        )
        # Single A → B edge in the result.
        edges = list(merged.edges(data=True))
        assert len(edges) == 1
        u, v, data = edges[0]
        assert u.name == "A" and v.name == "B"
        assert data["weight"] == 7  # 2 + 5

    def test_unions_ligands_with_order_preserving_dedupe(self):
        """Ligand metadata from each parallel connection is unioned;
        order-preserving dedupe means the first occurrence wins (matches
        existing ``contract_connections`` behaviour preserved by the
        recent fix).
        """
        nn, conns = self._two_parallel_edges()
        merged = nn.contract_connections(
            {
                (nn.neurons["A"], nn.neurons["B"], "chemical-synapse"): conns,
            }
        )
        merged_conn = list(merged.connections.values())[0]
        # First conn had ['Glu'], second had ['Glu', 'GABA']. Order-
        # preserving union: ['Glu', 'GABA'].
        assert merged_conn.ligands == ["Glu", "GABA"]

    def test_returns_new_graph_without_mutating_original(self):
        nn, conns = self._two_parallel_edges()
        original_edge_count = len(list(nn.edges()))
        new = nn.contract_connections(
            {
                (nn.neurons["A"], nn.neurons["B"], "chemical-synapse"): conns,
            }
        )
        # Original still has two parallel edges.
        assert len(list(nn.edges())) == original_edge_count == 2
        # New graph has one.
        assert len(list(new.edges())) == 1
        assert new is not nn

    def test_preserves_existing_merge_state_through_round_trip(self):
        """The new graph is built via ``create_neurons_from(data=True)``,
        which only sees networkx node attrs. ``contract_neurons``'s
        nx-attr mirror is what makes this round-trip work — assert it
        does so even after a contract_connections pass.
        """
        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        # Now collapse the AB → C edge (just one connection — trivial
        # case but exercises the round-trip).
        ab = nn.neurons["AB"]
        c = nn.neurons["C"]
        ab_c_conn = next(iter(c.in_connections.values()))
        merged = nn.contract_connections(
            {
                (ab, c, "chemical-synapse"): [ab_c_conn],
            }
        )
        ab_in_new = merged.neurons["AB"]
        assert ab_in_new.is_merged is True
        assert {"A", "B"}.issubset(ab_in_new.constituents.keys())
        assert ab_in_new.type == "sensory"

    def test_empty_connection_list_yields_zero_weight_edge(self):
        """Edge case: an empty list in the contraction dict produces a
        weight-0 placeholder. Documented behaviour — tested so future
        refactors don't silently change it.
        """
        nn, _ = self._two_parallel_edges()
        merged = nn.contract_connections(
            {
                (nn.neurons["A"], nn.neurons["B"], "chemical-synapse"): [],
            }
        )
        merged_conn = list(merged.connections.values())[0]
        assert merged_conn.weight == 0


# ===========================================================================
# Cross-function integration
# ===========================================================================


class TestLoaderMergePolicy:
    """Issue 11 — property loaders refuse-by-default on merged graphs.

    The lookups inside ``loadTranscripts`` / ``loadNeurotransmitters``
    / ``loadNeuropeptides`` are keyed by *current* neuron name, so a
    merged neuron silently misses its data (or, worse, crashes with
    ``KeyError`` on the default empty mapping). The fix surfaces a
    structured ``MergedNetworkError`` early so callers can either reload
    + load + merge in that order, or opt into ``aggregate=True``.

    Tests run against in-memory networks (no CSV/XLSX reads), made
    possible by hoisting the merge-policy check to the top of each
    loader.
    """

    def test_load_transcripts_refuses_on_merged_network_by_default(self):
        from cedne.utils import MergedNetworkError, loader

        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        with pytest.raises(MergedNetworkError) as exc_info:
            loader.loadTranscripts(nn)
        err = exc_info.value
        assert err.op_name == "loadTranscripts"
        assert "AB" in err.merged_names

    def test_load_neurotransmitters_refuses_on_merged_network_by_default(self):
        from cedne.utils import MergedNetworkError, loader

        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        with pytest.raises(MergedNetworkError) as exc_info:
            loader.loadNeurotransmitters(nn)
        assert exc_info.value.op_name == "loadNeurotransmitters"

    def test_load_neuropeptides_refuses_on_merged_network_by_default(self):
        from cedne.utils import MergedNetworkError, loader

        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        with pytest.raises(MergedNetworkError) as exc_info:
            loader.loadNeuropeptides(nn)
        assert exc_info.value.op_name == "loadNeuropeptides"

    def test_load_neurotransmitters_aggregate_flag_is_explicitly_unsupported(self):
        """``aggregate=True`` for loadNeurotransmitters is a planned
        follow-up (receptor/ligand union semantics need design). The
        loader should raise ``NotImplementedError`` rather than silently
        falling through to the refuse path or worse, the empty-loader
        path, so the user's intent isn't quietly dropped.
        """
        from cedne.utils import loader

        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        with pytest.raises(NotImplementedError, match="aggregate=True"):
            loader.loadNeurotransmitters(nn, aggregate=True)

    def test_load_neuropeptides_aggregate_flag_is_explicitly_unsupported(self):
        from cedne.utils import loader

        _, nn = _build_chain()
        nn.contract_neurons(("A", "B"), "AB")
        with pytest.raises(NotImplementedError, match="aggregate=True"):
            loader.loadNeuropeptides(nn, aggregate=True)

    def test_merged_network_error_payload_carries_merged_names(self):
        from cedne.utils import MergedNetworkError

        err = MergedNetworkError(["AB", "CD_E"], op_name="loadTranscripts")
        payload = err.to_payload()
        assert payload["error"] == "merged_network"
        assert payload["op_name"] == "loadTranscripts"
        assert set(payload["merged_names"]) == {"AB", "CD_E"}
        assert "AB" in payload["message"]

    def test_unmerged_network_path_unaffected_by_merge_check(self):
        """Smoke check: a freshly-built network (no contractions) must
        not raise MergedNetworkError. The check predicate is correct,
        not over-eager. (The test deliberately avoids the per-loader
        body since that needs CENGEN data; the policy check fires
        before the data read either way.)
        """
        from cedne.utils import MergedNetworkError, loader

        _, nn = _build_chain()
        # Pass aggregate=False (the default) — no merged neurons should
        # mean no error from the policy check itself.
        merged = loader._merged_neuron_names(nn)
        assert merged == []
        # And calling the loader should *not* raise MergedNetworkError
        # on this un-merged network. (It may still raise other errors
        # for missing CENGEN data; we only assert the policy check
        # doesn't fire.)
        try:
            loader.loadTranscripts(nn)
        except MergedNetworkError:
            pytest.fail("MergedNetworkError raised on an un-merged network")
        except Exception:
            pass  # any other failure (missing data, etc.) is fine here


class TestContractNeuronsThenConnections:
    """Mirrors the cedne_web backend's `data='clean'` flow: a sequence
    of ``contract_neurons`` calls followed by ``contract_connections``
    to collapse any parallel edges produced by the redirection.
    """

    def test_clean_mode_pipeline_preserves_merge_metadata(self):
        # A→B and A→C; merge B+C; pre→merged should get a parallel edge.
        w = Worm(name="clean")
        nn = NervousSystem(w)
        nn.create_neurons(
            ["A", "B", "C"], type={"A": "sensory", "B": "motor", "C": "motor"}
        )
        nn.setup_chemical_connections(
            {
                "A": {"B": {"weight": 1}, "C": {"weight": 2}},
            }
        )
        nn.contract_neurons(("B", "C"), "BC")
        # A → BC should now have two parallel chemical synapses.
        nn.reassign_connections()
        a = nn.neurons["A"]
        bc = nn.neurons["BC"]
        a_to_bc = [c for c in nn.connections.values() if c.pre is a and c.post is bc]
        assert len(a_to_bc) >= 2  # the redirect produced parallel edges

        # Now collapse them.
        parsed = {(a, bc, "chemical-synapse"): a_to_bc}
        merged_graph = nn.contract_connections(parsed)
        # Single A → BC edge with summed weight.
        edges = [(u, v, d) for u, v, d in merged_graph.edges(data=True)]
        assert len(edges) == 1
        assert edges[0][2]["weight"] == 3
        # Merge metadata survived the round-trip.
        bc_new = merged_graph.neurons["BC"]
        assert bc_new.is_merged is True
        assert bc_new.constituent_types == ["motor"]
