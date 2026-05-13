"""Fold policies: how to aggregate constituent attributes during a fold.

Folding (``Network.fold_network`` / ``Network.contract_neurons`` /
``Network.contract_connections``) collapses N entities into one supernode
or supernedge. Each entity carries attributes (type, recording, ligands,
receptor dicts, etc.); the question is *how the supernode should
represent that attribute*. Historically this was handled ad-hoc inside
``contract_connections`` for a small set of known fields (set-union for
ligands, sum for weight, etc.) — fine for two papers' worth of fields,
brittle as the dataset menu grows.

This module formalizes the choice as a per-attribute ``FoldPolicy``:

    FoldPolicy(name="recording", kind="timeseries", aggregator="timeseries_mean")

A ``FoldPolicySet`` is a mapping of attribute-name to policy that gets
applied during a single fold and **stored on the resulting network** so
the choice is reproducible. The default aggregator for an unregistered
attribute is ``drop`` — silent semantic drift on un-thought-about fields
is worse than missing data.

Aggregator menu (see ``apply_policy`` for semantics):

    Scalar (int/float):     mean | median | max | min | mode
    Time series:            timeseries_mean | timeseries_median | timeseries_max
    List:                   mode_count | keep_all | set_union
    Dict:                   dict_union | dict_intersection | dict_collated
    Categorical:            mode | keep_all
    Any kind:               drop  (return None — attribute not copied)

Time-series aggregators truncate to the shortest constituent's length
(common defensive choice for recording data with motion-rejected frames
or differing trial lengths).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np


__author__ = "Sahil Moza"
__date__ = "2026-05-12"
__license__ = "MIT"


# Aggregator names per kind. Keeping the kinds open as plain strings lets
# downstream code register custom kinds without editing this module.
ScalarAgg = Literal["sum", "mean", "median", "max", "min", "mode"]
TimeseriesAgg = Literal["timeseries_mean", "timeseries_median", "timeseries_max"]
ListAgg = Literal["mode_count", "keep_all", "set_union"]
DictAgg = Literal["dict_union", "dict_intersection", "dict_collated"]
CategoricalAgg = Literal["mode", "keep_all"]

# Sentinel meaning "don't copy this attribute to the supernode". Works on
# every kind; equivalent to the historical default of "not registered =
# not folded".
DROP: str = "drop"

ALL_AGGREGATORS = frozenset(
    [
        "sum",
        "mean",
        "median",
        "max",
        "min",
        "mode",
        "timeseries_mean",
        "timeseries_median",
        "timeseries_max",
        "mode_count",
        "keep_all",
        "set_union",
        "dict_union",
        "dict_intersection",
        "dict_collated",
        DROP,
    ]
)

# Canonical menu order (UI-facing): drives both the picker layout and the
# ``valid_aggregators(kind)`` public helper. _VALID_AGGREGATORS_BY_KIND is
# the membership-check companion (frozenset for O(1) lookup in
# ``__post_init__``) and is derived from this tuple-of-tuples so the two
# can never drift.
_AGGREGATOR_MENUS: Dict[str, tuple] = {
    "scalar": ("sum", "mean", "median", "max", "min", "mode", DROP),
    "timeseries": (
        "timeseries_mean",
        "timeseries_median",
        "timeseries_max",
        DROP,
    ),
    "list": ("mode_count", "keep_all", "set_union", DROP),
    "dict": ("dict_union", "dict_intersection", "dict_collated", DROP),
    "categorical": ("mode", "keep_all", "same_or_merged", DROP),
}

_VALID_AGGREGATORS_BY_KIND: Dict[str, frozenset] = {
    kind: frozenset(menu) for kind, menu in _AGGREGATOR_MENUS.items()
}


def valid_aggregators(kind: str) -> list:
    """Aggregator names valid for ``kind``, in canonical UI menu order.

    Public discovery helper for callers (e.g. backend endpoints, picker
    UIs) that need the per-kind menu without depending on the private
    ``_VALID_AGGREGATORS_BY_KIND`` table. Raises KeyError on unknown kind.
    """
    return list(_AGGREGATOR_MENUS[kind])


def valid_kinds() -> list:
    """All kinds the policy system understands, in declaration order."""
    return list(_AGGREGATOR_MENUS.keys())


# Sentinel returned by the ``same_or_merged`` categorical aggregator when
# constituents disagree on a categorical attribute. Mirrored to
# cedne/core/neuron.py's MERGED_TYPE (same string, defined twice to
# avoid a fold_policy → neuron import cycle).
SAME_OR_MERGED_SENTINEL: str = "merged"


@dataclass(frozen=True)
class FoldPolicy:
    """How to aggregate one named attribute across constituents.

    Attributes:
        name:       The attribute key (e.g. ``"recording"`` for time
                    series, ``"type"`` for categorical, ``"ligands"`` for
                    list-valued connection metadata). Used by callers as
                    the lookup key against the live attribute on each
                    constituent.
        kind:       The shape of the attribute. Determines which
                    aggregators are valid; ``apply_policy`` raises if a
                    kind/aggregator combination doesn't make sense.
        aggregator: How to combine N constituent values into one.
                    Default ``"drop"`` means the attribute isn't copied
                    to the supernode (safe default — no silent merge).
    """

    name: str
    kind: Literal["scalar", "timeseries", "list", "dict", "categorical"]
    aggregator: str = DROP

    def __post_init__(self):
        valid = _VALID_AGGREGATORS_BY_KIND.get(self.kind)
        if valid is None:
            raise ValueError(
                f"Unknown FoldPolicy kind: {self.kind!r}. "
                f"Expected one of {sorted(_VALID_AGGREGATORS_BY_KIND)}."
            )
        if self.aggregator not in valid:
            raise ValueError(
                f"Aggregator {self.aggregator!r} is not valid for kind "
                f"{self.kind!r}. Valid: {sorted(valid)}."
            )

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind,
            "aggregator": self.aggregator,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "FoldPolicy":
        return cls(name=d["name"], kind=d["kind"], aggregator=d.get("aggregator", DROP))


@dataclass
class FoldPolicySet:
    """A bundle of per-attribute policies + a default for unregistered attrs.

    Attached to the resulting network after a fold so the choices that
    produced it are part of the provenance record (alongside the
    constituent subgraphs already captured by ``fold_network``).
    """

    policies: Dict[str, FoldPolicy] = field(default_factory=dict)
    # Fallback for any attribute the caller didn't explicitly register.
    # ``drop`` is the safe default — see module docstring.
    default_aggregator: str = DROP

    def add(self, policy: FoldPolicy) -> "FoldPolicySet":
        self.policies[policy.name] = policy
        return self

    def get(self, name: str, kind: str) -> FoldPolicy:
        """Return the policy for ``name`` or a drop policy of the given kind."""
        if name in self.policies:
            return self.policies[name]
        return FoldPolicy(name=name, kind=kind, aggregator=self.default_aggregator)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policies": {n: p.to_dict() for n, p in self.policies.items()},
            "default_aggregator": self.default_aggregator,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FoldPolicySet":
        return cls(
            policies={
                n: FoldPolicy.from_dict(p) for n, p in (d.get("policies") or {}).items()
            },
            default_aggregator=d.get("default_aggregator", DROP),
        )


# ---------------------------------------------------------------------------
# Aggregator implementations
# ---------------------------------------------------------------------------


def _coerce_numeric(values: Iterable[Any]) -> List[float]:
    """Pull numeric-looking values out of a heterogeneous list."""
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(f) or np.isinf(f):
            continue
        out.append(f)
    return out


def _agg_scalar(values: Iterable[Any], op: str) -> Optional[float]:
    arr = _coerce_numeric(values)
    if not arr:
        # ``sum`` of nothing is conventionally 0; the historical
        # contract_connections behavior initialized weight=0 even for
        # empty constituent lists, so we match that here.
        return 0.0 if op == "sum" else None
    if op == "sum":
        return float(np.sum(arr))
    if op == "mean":
        return float(np.mean(arr))
    if op == "median":
        return float(np.median(arr))
    if op == "max":
        return float(np.max(arr))
    if op == "min":
        return float(np.min(arr))
    if op == "mode":
        try:
            return float(statistics.mode(arr))
        except statistics.StatisticsError:
            # Ties — pick the first multimode value, deterministic.
            return float(statistics.multimode(arr)[0])
    raise ValueError(f"Unknown scalar aggregator: {op!r}")


def _agg_timeseries(values: Iterable[Any], op: str):
    """Aggregate constituent time-series, truncating to the shortest length."""
    arrays = []
    for v in values:
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.ndim == 0 or arr.size == 0:
            continue
        arrays.append(arr)
    if not arrays:
        return None
    min_len = int(min(a.shape[0] for a in arrays))
    stacked = np.stack([a[:min_len] for a in arrays])  # (N, min_len)
    if op == "timeseries_mean":
        return stacked.mean(axis=0)
    if op == "timeseries_median":
        return np.median(stacked, axis=0)
    if op == "timeseries_max":
        return stacked.max(axis=0)
    raise ValueError(f"Unknown timeseries aggregator: {op!r}")


def _agg_categorical(values: Iterable[Any], op: str):
    """Combine categorical values (e.g. neuron type). Always preserves identity."""
    arr = [v for v in values if v is not None and v != ""]
    if not arr:
        return None
    if op == "mode":
        try:
            return statistics.mode(arr)
        except statistics.StatisticsError:
            return statistics.multimode(arr)[0]
    if op == "keep_all":
        # preserve first-occurrence order
        seen = set()
        out = []
        for v in arr:
            key = v if isinstance(v, (str, int, float, bool)) else repr(v)
            if key not in seen:
                seen.add(key)
                out.append(v)
        return out
    if op == "same_or_merged":
        # The historical neuron type-merge policy: if every constituent
        # agrees on the value, preserve it; if they disagree, return the
        # SAME_OR_MERGED_SENTINEL ("merged") so analyses that switch on
        # type can detect the merged case explicitly. Empty input
        # returns None — the caller is expected to skip setting the
        # attribute, matching contract_neurons's prior behavior of
        # leaving the attribute unchanged when no constituent had a
        # usable value.
        distinct = set(arr)
        if len(distinct) == 1:
            return next(iter(distinct))
        return SAME_OR_MERGED_SENTINEL
    raise ValueError(f"Unknown categorical aggregator: {op!r}")


def _set_key(x: Any) -> Any:
    """Hashable key for set-union dedup. Matches the legacy contract:
    lists are coerced to tuples (so ``[a, b]`` and ``(a, b)`` count as
    the same element); primitives pass through; anything else is
    repr-stringified so non-hashable containers still dedupe."""
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


def _agg_list(values: Iterable[Any], op: str):
    """Combine list-valued attributes (ligands, receptors, etc.)."""
    flat: List[Any] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            flat.extend(v)
        else:
            flat.append(v)
    if not flat:
        return [] if op != "mode_count" else None
    if op == "keep_all":
        return list(flat)
    if op == "set_union":
        seen = set()
        out: List[Any] = []
        for x in flat:
            key = _set_key(x)
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out
    if op == "mode_count":
        return Counter(_hashable_only(flat)).most_common(1)[0][0]
    raise ValueError(f"Unknown list aggregator: {op!r}")


def _hashable_only(seq: Iterable[Any]) -> List[Any]:
    """Counter requires hashable values; cast complex objects via repr."""
    out = []
    for x in seq:
        try:
            hash(x)
            out.append(x)
        except TypeError:
            out.append(repr(x))
    return out


def _agg_dict(values: Iterable[Any], op: str):
    """Combine dict-valued attributes.

    union           — every key from every constituent. When the same
                      key appears in multiple constituents, the *first
                      observed* value wins. Output is ``{k: v}``, i.e.
                      same shape as a single-constituent dict. This is
                      the deterministic, non-arbitrary replacement for
                      what contract_connections historically did via a
                      manual ``if k not in merged_receptors`` loop.
    intersection    — only keys present in *every* constituent dict.
                      Value is the first observed value per key. Same
                      shape as ``union``.
    collated        — every key from every constituent. Value is a
                      list of all observed values (richer form for
                      callers that want to see every observation,
                      possibly to compute summary statistics off-line).
                      Output is ``{k: [v0, v1, ...]}``.
    """
    dicts = [v for v in values if isinstance(v, dict)]
    if not dicts:
        return None
    if op == "dict_union":
        out: Dict[Any, Any] = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = v
        return out
    if op == "dict_intersection":
        common = set(dicts[0].keys())
        for d in dicts[1:]:
            common &= set(d.keys())
        out_int: Dict[Any, Any] = {}
        for k in common:
            for d in dicts:
                if k in d:
                    out_int[k] = d[k]
                    break
        return out_int
    if op == "dict_collated":
        out_col: Dict[Any, List[Any]] = {}
        for d in dicts:
            for k, v in d.items():
                out_col.setdefault(k, []).append(v)
        return out_col
    raise ValueError(f"Unknown dict aggregator: {op!r}")


# ---------------------------------------------------------------------------
# Default policy sets for known dataset shapes
# ---------------------------------------------------------------------------
#
# DEFAULT_CONNECTION_FOLD_POLICY captures the behavior that
# ``Network.contract_connections`` had before Phase 2 — weights sum,
# transmitter / ligand / receptor list-valued metadata set-union, and
# receptor dicts merge with first-observed-value semantics. Phase 2
# refactors contract_connections to drive its merge logic through this
# policy set instead of inlining the rules. Callers that pass a custom
# policy set override this; callers that pass nothing get current
# behavior.
#
# Receptors note: pre-Phase-2 contract_connections did "first key wins"
# via ``if rk not in merged_receptors``. ``dict_union`` here means the
# same thing — first observed value per key — chosen to keep parity.

DEFAULT_CONNECTION_FOLD_POLICY = FoldPolicySet(
    policies={
        "weight": FoldPolicy("weight", "scalar", "sum"),
        "ligands": FoldPolicy("ligands", "list", "set_union"),
        "neurotransmitters": FoldPolicy("neurotransmitters", "list", "set_union"),
        "putative_neurotrasmitter_receptors": FoldPolicy(
            "putative_neurotrasmitter_receptors", "list", "set_union"
        ),
        "receptors": FoldPolicy("receptors", "dict", "dict_union"),
    },
    default_aggregator=DROP,
)


# DEFAULT_NEURON_FOLD_POLICY captures the historical contract_neurons /
# fold_network behavior for the three "bounded categorical label"
# attributes that have always been merged via the all-same-or-MERGED-
# sentinel rule (the MERGE_TRACK_ATTRS list in cedne/core/neuron.py).
# Phase 2.2 refactors contract_neurons + the batch path's neuron merge
# to drive that decision through this policy set; parity tests pin the
# behavior on identical inputs.
#
# Numeric properties (degree, length, transcript counts) and time-series
# properties (recordings) are NOT registered here — the historical code
# didn't aggregate them onto the supernode at all; their values stay on
# the constituent neurons (reachable via constituent_subgraph). Callers
# that want a per-fold aggregate (e.g. timeseries_mean on recording)
# should pass a custom FoldPolicySet that extends this one.
DEFAULT_NEURON_FOLD_POLICY = FoldPolicySet(
    policies={
        "type": FoldPolicy("type", "categorical", "same_or_merged"),
        "category": FoldPolicy("category", "categorical", "same_or_merged"),
        "modality": FoldPolicy("modality", "categorical", "same_or_merged"),
    },
    default_aggregator=DROP,
)


def apply_policy(policy: FoldPolicy, values: List[Any]) -> Any:
    """Apply a policy to a list of constituent values.

    Returns the merged value, or ``None`` for the drop sentinel / empty
    input. Never raises on empty/missing data — that's the caller's
    cue to skip copying the attribute to the supernode.
    """
    if policy.aggregator == DROP:
        return None
    if policy.kind == "scalar":
        return _agg_scalar(values, policy.aggregator)
    if policy.kind == "timeseries":
        return _agg_timeseries(values, policy.aggregator)
    if policy.kind == "list":
        return _agg_list(values, policy.aggregator)
    if policy.kind == "dict":
        return _agg_dict(values, policy.aggregator)
    if policy.kind == "categorical":
        return _agg_categorical(values, policy.aggregator)
    raise ValueError(f"Unknown FoldPolicy kind: {policy.kind!r}")
