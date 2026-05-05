"""Tests for the descriptive event log (cedne/core/history.py).

Covers:
  * ``Event`` content-addressable id stability + tamper detection
  * ``_resolve_animal`` walking from Animal / NervousSystem / tuple
  * ``@record`` decorator end-to-end on a synthetic loader (no fixtures)
  * Constructor-style recording (animal resolved from return value)
  * Parent linkage forms a chain
  * JSON round-trip via ``history_to_json`` / ``history_from_json``
  * Pickle round-trip preserves history (the cedne ``.save`` path)
  * Backward compat: an Animal instantiated from an older pickle without
    ``history`` still accepts new events (lazy-init branch)
"""

import json
import pickle

import pytest

from cedne.core.animal import Animal, Worm
from cedne.core.network import NervousSystem
from cedne.core.history import (
    Event,
    _resolve_animal,
    _target_of,
    history_from_json,
    history_to_json,
    record,
)


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

class TestEvent:
    def _make(self, **overrides):
        defaults = dict(
            op="op",
            args={"k": 1},
            target={"type": "X"},
            citations=[],
            parent_id=None,
            timestamp="2026-05-05T00:00:00Z",
        )
        defaults.update(overrides)
        return Event(**defaults)

    def test_id_is_deterministic(self):
        a = self._make()
        b = self._make()
        assert a.id == b.id

    def test_id_changes_with_args(self):
        a = self._make(args={"k": 1})
        b = self._make(args={"k": 2})
        assert a.id != b.id

    def test_id_changes_with_parent(self):
        a = self._make(parent_id=None)
        b = self._make(parent_id="abc")
        assert a.id != b.id

    def test_id_length(self):
        # We slice the sha256 to 16 hex chars for compactness in URLs/DB keys.
        assert len(self._make().id) == 16


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

class TestResolveAnimal:
    def test_animal_passes_through(self):
        w = Worm()
        assert _resolve_animal(w) is w

    def test_resolves_via_nervous_system(self):
        w = Worm()
        nn = NervousSystem(w)
        assert _resolve_animal(nn) is w

    def test_resolves_inside_tuple(self):
        w = Worm()
        nn = NervousSystem(w)
        # load_nwb returns (NervousSystem, Session) — the decorator scans tuples.
        assert _resolve_animal((nn, "session")) is w

    def test_returns_none_for_unrelated(self):
        assert _resolve_animal("not an animal") is None
        assert _resolve_animal(None) is None


class TestTargetOf:
    def test_named_object(self):
        w = Worm(name="N2")
        desc = _target_of(w)
        assert desc["name"] == "N2"
        assert "Worm" in desc["type"]

    def test_tuple_descends(self):
        nn = NervousSystem(Worm(), network="adult")
        desc = _target_of((nn, "session"))
        assert desc["name"] == "adult"


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

class TestRecordDecorator:
    def test_records_kwargs_against_animal(self):
        w = Worm()
        nn = NervousSystem(w)

        @record("frob")
        def frob(network, *, threshold=4):
            return None

        frob(nn, threshold=7)
        assert len(w.history) == 1
        ev = w.history[0]
        assert ev.op == "frob"
        assert ev.args == {"threshold": 7}
        assert ev.parent_id is None
        assert ev.target.get("name") == nn.name

    def test_positional_args_not_recorded(self):
        # Documented limitation — only kwargs make it into the log.
        w = Worm()
        nn = NervousSystem(w)

        @record("frob")
        def frob(network, threshold=4):
            return None

        frob(nn, 7)
        assert w.history[0].args == {}

    def test_chains_parent_id(self):
        w = Worm()
        nn = NervousSystem(w)

        @record("op")
        def op(network):
            return None

        op(nn)
        op(nn)
        op(nn)
        assert len(w.history) == 3
        assert w.history[0].parent_id is None
        assert w.history[1].parent_id == w.history[0].id
        assert w.history[2].parent_id == w.history[1].id

    def test_constructor_resolves_via_result(self):
        @record("make_thing")
        def make_thing(name=""):
            w = Worm(name=name)
            return w

        result = make_thing(name="X")
        assert isinstance(result, Worm)
        assert len(result.history) == 1
        assert result.history[0].op == "make_thing"
        assert result.history[0].args == {"name": "X"}

    def test_no_op_when_no_animal(self):
        @record("orphan")
        def orphan(x):
            return x

        # Must not raise — recording is best-effort.
        assert orphan(42) == 42

    def test_reserved_kwargs_consumed(self):
        w = Worm()
        nn = NervousSystem(w)

        @record("op")
        def op(network):
            return None

        # _citations and _actor are consumed by the decorator and must not
        # be forwarded (op() would TypeError if they were).
        op(nn, _citations=["RefA"], _actor="user42")
        ev = w.history[0]
        assert ev.citations == ["RefA"]
        assert ev.actor == "user42"

    def test_lazy_init_for_legacy_animals(self):
        # Simulate an Animal pickled before .history existed.
        w = Worm()
        del w.history
        assert not hasattr(w, "history")

        @record("op")
        def op(animal):
            return None

        op(w)
        assert len(w.history) == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_json_round_trip(self):
        w = Worm()
        nn = NervousSystem(w)

        @record("op")
        def op(network, *, k=1):
            return None

        op(nn, k=1)
        op(nn, k=2)

        rows = history_to_json(w.history)
        text = json.dumps(rows)
        rebuilt = history_from_json(json.loads(text))
        assert [e.id for e in rebuilt] == [e.id for e in w.history]
        assert rebuilt[1].parent_id == rebuilt[0].id

    def test_history_from_json_detects_tampering(self):
        w = Worm()
        nn = NervousSystem(w)

        @record("op")
        def op(network):
            return None

        op(nn)
        rows = history_to_json(w.history)
        rows[0]["args"] = {"k": "tampered"}  # body changed but id left as-is
        with pytest.raises(ValueError):
            history_from_json(rows)


class TestPickleRoundTrip:
    def test_save_load_preserves_history(self, tmp_path):
        w = Worm(name="testworm")
        nn = NervousSystem(w)

        @record("op")
        def op(network, *, k=1):
            return None

        op(nn, k=1)
        op(nn, k=2)
        ids_before = [e.id for e in w.history]

        path = tmp_path / "w.cedne"
        w.save(str(path))

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert [e.id for e in loaded.history] == ids_before
        assert loaded.history[1].parent_id == loaded.history[0].id


# ---------------------------------------------------------------------------
# Integration with real loader-style mutations
# ---------------------------------------------------------------------------

class TestBuiltinDecoration:
    def test_fold_network_logs(self):
        # fold_network produces a new graph; the event still belongs to the
        # source's worm because copy_type='deep_with_data' shares the worm.
        w = Worm()
        nn = NervousSystem(w, network="src")
        nn.create_neurons(["A", "B"])

        nn.fold_network({"AB": ["A", "B"]}, name="folded")
        ops = [e.op for e in w.history]
        assert "fold_network" in ops
