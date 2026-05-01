"""
Tests for the citation/provenance infrastructure (cedne/core/source.py).

Phase 0 covers:
  - Citation dataclass round-trip
  - serialize_citations helper
  - Citable base class (standalone)
  - Hierarchical resolution across the structural classes:
    Animal/Worm -> NervousSystem -> NeuronGroup/ConnectionGroup -> Neuron/Connection
  - Backward compatibility with the legacy dict-style citations attached by loaders
"""

import pytest

from cedne.core.animal import Worm
from cedne.core.connection import (
    ChemicalSynapse,
    Connection,
    ConnectionGroup,
)
from cedne.core.network import NervousSystem
from cedne.core.neuron import Neuron, NeuronGroup
from cedne.core.source import Citable, Citation, serialize_citations


# ---------------------------------------------------------------------------
# Citation dataclass
# ---------------------------------------------------------------------------

class TestCitation:
    def test_minimal_construction(self):
        c = Citation(key="White1986")
        assert c.key == "White1986"
        assert c.title is None
        assert c.year is None

    def test_full_construction(self):
        c = Citation(
            key="Smith2023",
            title="A paper",
            authors=["Smith", "Jones"],
            year=2023,
            doi="10.1/abc",
        )
        assert c.title == "A paper"
        assert c.authors == ["Smith", "Jones"]
        assert c.doi == "10.1/abc"

    def test_to_dict_omits_none(self):
        c = Citation(key="K", year=2020)
        d = c.to_dict()
        assert d == {"key": "K", "year": 2020}
        assert "title" not in d  # None fields are dropped

    def test_round_trip(self):
        original = Citation(key="K", title="T", year=2024, authors=["A"])
        restored = Citation.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_ignores_unknown_keys(self):
        c = Citation.from_dict({"key": "K", "extra": "ignored", "year": 2024})
        assert c.key == "K"
        assert c.year == 2024


# ---------------------------------------------------------------------------
# serialize_citations helper
# ---------------------------------------------------------------------------

class TestSerializeCitations:
    def test_empty(self):
        assert serialize_citations({}) == {}

    def test_citation_objects_flattened(self):
        d = {"Smith2023": Citation(key="Smith2023", year=2023)}
        out = serialize_citations(d)
        assert out == {"Smith2023": {"key": "Smith2023", "year": 2023}}

    def test_legacy_dicts_passed_through(self):
        legacy = {"cook_connectome": {"authors": ["Cook"], "year": 2019}}
        assert serialize_citations(legacy) == legacy

    def test_mixed(self):
        d = {
            "structured": Citation(key="structured", year=2024),
            "legacy": {"raw": "data"},
        }
        out = serialize_citations(d)
        assert out["structured"] == {"key": "structured", "year": 2024}
        assert out["legacy"] == {"raw": "data"}


# ---------------------------------------------------------------------------
# Citable standalone
# ---------------------------------------------------------------------------

class TestCitableStandalone:
    def test_init_creates_empty_citations(self):
        obj = Citable()
        assert obj.citations == {}

    def test_add_remove_citation(self):
        obj = Citable()
        cite = Citation(key="K1")
        obj.add_citation(cite)
        assert obj.citations["K1"] is cite
        obj.remove_citation("K1")
        assert "K1" not in obj.citations

    def test_remove_missing_is_noop(self):
        obj = Citable()
        obj.remove_citation("never_added")  # must not raise

    def test_default_no_parent(self):
        obj = Citable()
        assert list(obj._parent_citables()) == []

    def test_effective_citations_no_parents(self):
        obj = Citable()
        obj.add_citation(Citation(key="K1"))
        eff = obj.effective_citations()
        assert len(eff) == 1
        _, key, _ = eff[0]
        assert key == "K1"


# ---------------------------------------------------------------------------
# Fixtures: build a small worm with neurons, connections, and groups
# ---------------------------------------------------------------------------

@pytest.fixture
def worm_with_circuit():
    """A minimal Worm -> NervousSystem -> Neurons + Connection setup."""
    w = Worm(name="test_worm")
    ns = NervousSystem(worm=w, network="Default")
    n1 = Neuron("AVAL", ns)
    n2 = Neuron("AVAR", ns)
    n3 = Neuron("AVBL", ns)
    c = ChemicalSynapse(n1, n2, weight=1.5)
    return {"worm": w, "network": ns, "n1": n1, "n2": n2, "n3": n3, "conn": c}


# ---------------------------------------------------------------------------
# Backward compatibility: existing loader pattern must keep working
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_worm_citations_initially_empty(self, worm_with_circuit):
        assert worm_with_circuit["worm"].citations == {}

    def test_legacy_update_pattern(self, worm_with_circuit):
        """loader.py does: w.citations.update({'cook_connectome': {...}})."""
        w = worm_with_circuit["worm"]
        w.citations.update(
            {"cook_connectome": {"authors": ["Cook"], "year": 2019}}
        )
        assert w.citations["cook_connectome"]["year"] == 2019

    def test_all_structural_classes_have_citations(self, worm_with_circuit):
        for obj in worm_with_circuit.values():
            assert hasattr(obj, "citations")
            assert isinstance(obj.citations, dict)


# ---------------------------------------------------------------------------
# Hierarchical resolution: the heart of Phase 0
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_neuron_walks_to_network_and_worm(self, worm_with_circuit):
        w = worm_with_circuit["worm"]
        ns = worm_with_circuit["network"]
        n1 = worm_with_circuit["n1"]

        n1.add_citation(Citation(key="NeuronCite"))
        ns.add_citation(Citation(key="NetworkCite"))
        w.add_citation(Citation(key="WormCite"))

        keys = [k for _, k, _ in n1.effective_citations()]
        assert set(keys) == {"NeuronCite", "NetworkCite", "WormCite"}

    def test_connection_walks_to_network_and_worm(self, worm_with_circuit):
        w = worm_with_circuit["worm"]
        ns = worm_with_circuit["network"]
        c = worm_with_circuit["conn"]

        c.add_citation(Citation(key="ConnCite"))
        ns.add_citation(Citation(key="NetworkCite"))
        w.add_citation(Citation(key="WormCite"))

        keys = [k for _, k, _ in c.effective_citations()]
        assert set(keys) == {"ConnCite", "NetworkCite", "WormCite"}

    def test_neuron_group_citation_reaches_member(self, worm_with_circuit):
        ns = worm_with_circuit["network"]
        n1 = worm_with_circuit["n1"]
        n2 = worm_with_circuit["n2"]
        n3 = worm_with_circuit["n3"]

        g = NeuronGroup(ns, members=[n1, n2], group_name="cmd_interneurons")
        g.add_citation(Citation(key="GroupCite"))

        # n1, n2 are in the group -> see group citation
        keys1 = [k for _, k, _ in n1.effective_citations()]
        keys2 = [k for _, k, _ in n2.effective_citations()]
        assert "GroupCite" in keys1
        assert "GroupCite" in keys2

        # n3 is NOT in the group -> does not see group citation
        keys3 = [k for _, k, _ in n3.effective_citations()]
        assert "GroupCite" not in keys3

    def test_connection_group_citation_reaches_member(self, worm_with_circuit):
        ns = worm_with_circuit["network"]
        c = worm_with_circuit["conn"]

        g = ConnectionGroup(ns, members=[c], group_name="excitatory")
        g.add_citation(Citation(key="ExcGroupCite"))

        keys = [k for _, k, _ in c.effective_citations()]
        assert "ExcGroupCite" in keys

    def test_provenance_labels_identify_level(self, worm_with_circuit):
        w = worm_with_circuit["worm"]
        ns = worm_with_circuit["network"]
        n1 = worm_with_circuit["n1"]

        n1.add_citation(Citation(key="OnNeuron"))
        ns.add_citation(Citation(key="OnNetwork"))
        w.add_citation(Citation(key="OnWorm"))

        labels_by_key = {k: lbl for lbl, k, _ in n1.effective_citations()}
        assert labels_by_key["OnNeuron"].startswith("Neuron")
        assert labels_by_key["OnNetwork"].startswith("NervousSystem")
        assert labels_by_key["OnWorm"].startswith("Worm")

    def test_dedup_keeps_most_specific(self, worm_with_circuit):
        """If the same citation key appears at multiple levels, the most-specific wins."""
        w = worm_with_circuit["worm"]
        n1 = worm_with_circuit["n1"]

        # Same key, different value at neuron vs worm
        neuron_cite = Citation(key="Shared", title="Neuron-level")
        worm_cite = Citation(key="Shared", title="Worm-level")
        n1.add_citation(neuron_cite)
        w.add_citation(worm_cite)

        eff = n1.effective_citations()
        keys = [k for _, k, _ in eff]
        # Only one entry for the shared key
        assert keys.count("Shared") == 1
        # And it's the neuron's version
        label, _, val = next(t for t in eff if t[1] == "Shared")
        assert val.title == "Neuron-level"
        assert label.startswith("Neuron")

    def test_legacy_dict_citations_in_hierarchy(self, worm_with_circuit):
        """Legacy loader-style dict values must traverse the hierarchy intact."""
        w = worm_with_circuit["worm"]
        n1 = worm_with_circuit["n1"]

        w.citations.update(
            {"cook_connectome": {"authors": ["Cook"], "year": 2019}}
        )
        eff = n1.effective_citations()
        keys = [k for _, k, _ in eff]
        assert "cook_connectome" in keys
        # Value is the legacy dict, not converted to a Citation
        val = next(v for _, k, v in eff if k == "cook_connectome")
        assert val == {"authors": ["Cook"], "year": 2019}


# ---------------------------------------------------------------------------
# to_dict integration
# ---------------------------------------------------------------------------

class TestToDictIntegration:
    def test_neuron_to_dict_includes_citations(self, worm_with_circuit):
        n1 = worm_with_circuit["n1"]
        n1.add_citation(Citation(key="K", year=2024))
        d = n1.to_dict()
        assert "citations" in d
        assert d["citations"]["K"] == {"key": "K", "year": 2024}

    def test_neuron_to_dict_omits_citations_when_empty(self, worm_with_circuit):
        n1 = worm_with_circuit["n1"]
        d = n1.to_dict()
        assert "citations" not in d  # key only present when non-empty

    def test_connection_to_dict_includes_citations(self, worm_with_circuit):
        c = worm_with_circuit["conn"]
        c.add_citation(Citation(key="K", title="T"))
        d = c.to_dict()
        assert "citations" in d
        assert d["citations"]["K"] == {"key": "K", "title": "T"}

    def test_connection_to_dict_omits_citations_when_empty(
        self, worm_with_circuit
    ):
        c = worm_with_circuit["conn"]
        d = c.to_dict()
        assert "citations" not in d

    def test_neuron_to_dict_passes_through_legacy_dict_citations(
        self, worm_with_circuit
    ):
        n1 = worm_with_circuit["n1"]
        n1.citations["legacy"] = {"raw": "data"}
        d = n1.to_dict()
        assert d["citations"]["legacy"] == {"raw": "data"}
