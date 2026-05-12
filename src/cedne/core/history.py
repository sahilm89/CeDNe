"""Descriptive event log for CeDNe.

The log lives on ``Animal.history`` and grows as mutating operations run. Each
``Event`` is content-addressed (sha256 of its JSON-serialized body) and carries
its parent's id, so the log forms a linear chain that branches naturally if a
caller restores from an earlier checkpoint and continues.

The log is descriptive, not re-executable: it records *what happened*, with
enough metadata (operation name, kwargs, target, citations, optional source
fingerprints) for a reader to reconstruct the steps by hand. Restoring state
from a checkpoint is the caller's job (e.g. CeDNe_web stores pickled blobs
keyed by event id); cedne itself only owns the log.

Design notes:
  * Decorate at the *loader* boundary, not the parser. When `utils/loader.py`
    splits into loader + parser, the decorator stays on the side that mutates
    a network.
  * Loaders take their target (Network or Animal) as the first positional arg
    and the rest as keyword-only, so the recorded ``args`` dict is always
    self-describing.
  * No checkpoint state lives here. Whoever wants undo (e.g. the web app)
    pickles the Animal themselves and indexes by the head event id.
"""

from __future__ import annotations

__author__ = "Sahil Moza"
__date__ = "2026-05-05"
__license__ = "MIT"

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _cedne_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("cedne")
    except Exception:
        return None


def _jsonify(value: Any) -> Any:
    """Best-effort conversion to a JSON-friendly value.

    Loaders pass user data of varying shapes; we want the recorded ``args``
    dict to round-trip through json.dumps without surprises. Unknown objects
    fall back to their string representation, which keeps the log lossy-but-
    readable for things like file handles or numpy scalars.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in value]
    # numpy scalars
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonify(item())
        except Exception:
            pass
    # Cedne objects: prefer name, then class name
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return f"<{type(value).__name__}>"


@dataclass(frozen=True)
class Event:
    """A single descriptive entry in an Animal's history log.

    The ``id`` is the sha256 of the body (everything except ``id`` itself);
    chaining via ``parent_id`` produces a content-addressable DAG.
    """

    op: str
    args: Dict[str, Any]
    target: Dict[str, Any]
    citations: List[str]
    parent_id: Optional[str]
    timestamp: str
    actor: Optional[str] = None
    code_version: Optional[str] = None
    id: str = field(init=False)

    def __post_init__(self) -> None:
        # Build the body manually — asdict() would try to read self.id, which
        # we haven't set yet. Iterate the field list directly instead.
        body = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "id"}
        encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        object.__setattr__(self, "id", hashlib.sha256(encoded).hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_animal(obj: Any) -> Optional[Any]:
    """Walk from a mutation target (or freshly constructed result) to its Animal.

    Accepts Animals directly, NervousSystems (via ``.worm``), or anything else
    exposing a ``worm`` / ``animal`` attribute. Tuples (e.g. ``load_nwb``
    returns ``(network, session)``) are scanned for the first resolvable
    member. Returns None if no animal can be reached, in which case the
    decorator silently no-ops — we never want history recording to break a
    user's call.
    """
    # Late import to avoid a circular dependency: animal.py imports history.
    from .animal import Animal

    if obj is None:
        return None
    if isinstance(obj, Animal):
        return obj
    for attr in ("worm", "animal"):
        owner = getattr(obj, attr, None)
        if isinstance(owner, Animal):
            return owner
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _resolve_animal(item)
            if found is not None:
                return found
    return None


def _target_of(obj: Any) -> Dict[str, Any]:
    """Describe which object the event acted on, for human readers of the log.

    For tuples (e.g. ``(NervousSystem, Session)`` returned by ``load_nwb``)
    the first member with a usable type/name is described.
    """
    if isinstance(obj, (list, tuple)):
        for item in obj:
            inner = _target_of(item)
            if inner.get("name") or inner.get("type") not in (None, "NoneType"):
                return inner
        return {"type": type(obj).__name__}
    desc: Dict[str, Any] = {"type": type(obj).__name__}
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        desc["name"] = name
    return desc


def record(op: str) -> Callable:
    """Decorate a mutating function so it appends an ``Event`` to the animal log.

    Resolution order for "which animal does this event belong to":
      1. The decorated function's first positional argument (the usual loader
         convention: ``loadX(network_or_animal, ...)``).
      2. Failing that, the function's return value (constructor convention:
         ``makeWorm(...) -> Worm``, ``load_nwb(path) -> (NervousSystem, ...)``).

    Everything keyword-passed is captured in the recorded ``args`` dict, so
    callers using positional arguments leave those out of the log. Loaders
    that want full fidelity should accept their parameters by keyword.

    Reserved kwargs consumed by the decorator (not forwarded to the wrapped
    function): ``_citations=[...]`` for citations introduced by this op,
    ``_actor=...`` for the user/agent that triggered it,
    ``_silent=True`` to invoke the recorded function WITHOUT appending an
    Event to the animal log — used by internal callers that compose
    recorded operations (e.g. ``contract_neurons`` builds a pre-merge
    subgraph via ``subnetwork``; that internal copy is an implementation
    detail of the contraction, not a user-initiated subnetwork op, so
    surfacing it on the log would mislead consumers about what the
    user actually did).
    """

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            citations = kwargs.pop("_citations", None) or []
            actor = kwargs.pop("_actor", None)
            silent = kwargs.pop("_silent", False)
            result = fn(*args, **kwargs)
            if silent:
                return result
            target_obj = args[0] if args else None
            animal = _resolve_animal(target_obj)
            if animal is None:
                # Constructor-style loaders: target is the result.
                animal = _resolve_animal(result)
                target_obj = result if animal is not None else target_obj
            if animal is not None:
                # Lazy-init: older pickles predate this attribute.
                if not hasattr(animal, "history") or animal.history is None:
                    animal.history = []
                parent_id = animal.history[-1].id if animal.history else None
                event = Event(
                    op=op,
                    args=_jsonify(kwargs),
                    target=_target_of(target_obj),
                    citations=list(citations),
                    parent_id=parent_id,
                    timestamp=datetime.datetime.now(datetime.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    actor=actor,
                    code_version=_cedne_version(),
                )
                animal.history.append(event)
            return result

        return wrapper

    return deco


def fingerprint_file(path: Any) -> Dict[str, Any]:
    """Stable identifier for a local data file: basename + sha256 + size.

    Loaders are encouraged to pass ``source_fingerprint=fingerprint_file(p)``
    in their kwargs so the recorded event captures exactly which dataset was
    consumed, not just the (ephemeral) path.
    """
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"basename": p.name, "sha256": h.hexdigest(), "size": size}


def history_to_json(events: List[Event]) -> List[Dict[str, Any]]:
    """Serialize an event list to JSON-friendly dicts (e.g. for sidecar export)."""
    return [e.to_dict() for e in events]


def history_from_json(rows: List[Dict[str, Any]]) -> List[Event]:
    """Inverse of ``history_to_json``. Recomputes ids; raises on corruption."""
    rebuilt: List[Event] = []
    for row in rows:
        original_id = row.get("id")
        kwargs = {k: v for k, v in row.items() if k != "id"}
        ev = Event(**kwargs)
        if original_id is not None and ev.id != original_id:
            raise ValueError(
                f"history row {original_id!r} failed hash check "
                f"(recomputed {ev.id!r})"
            )
        rebuilt.append(ev)
    return rebuilt
