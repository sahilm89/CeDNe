"""Statistical enrichment/depletion tests for CeDNe group attributes.

The helpers here are intentionally independent of the web backend. They accept
CeDNe ``NeuronGroup`` or ``ConnectionGroup`` objects and return JSON-friendly
dicts that can be used from notebooks, scripts, and API endpoints.
"""

from __future__ import annotations

import math
from collections import Counter
from numbers import Number
from typing import Any, Iterable

import numpy as np
from scipy import stats

from cedne.core.connection import Connection, ConnectionGroup
from cedne.core.neuron import Neuron, NeuronGroup


NETWORK_NULLS = {"network", "full_network", "full"}
COMPLEMENT_NULLS = {"complement", "rest"}
EMPIRICAL_NULLS = {"size_matched", "shuffled", "permutation", "sample"}
MODES = {"auto", "numeric", "categorical", "set_membership", "binary"}
MISSING_POLICIES = {
    "exclude_and_report",
    "empty_is_absent",
    "missing_is_absent",
    "missing_is_unknown",
}


def group_attribute_enrichment(
    group: NeuronGroup | ConnectionGroup,
    attribute: str,
    *,
    mode: str = "auto",
    value: Any | None = None,
    missing_policy: str = "exclude_and_report",
    eligible_filter: dict[str, Any] | None = None,
    reference: NeuronGroup | ConnectionGroup | str | None = None,
    null_model: str = "network",
    attribute_type: str = "auto",
    n_resamples: int = 1000,
    random_state: int | None = None,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Test whether a group is enriched or depleted for an attribute.

    ``mode`` and ``missing_policy`` make attribute semantics explicit. For
    numeric measurements and categorical annotations, missing values are usually
    excluded from the primary test and reported separately. For set-valued
    features such as neurotransmitters or neuropeptides, use
    ``mode="set_membership"`` with ``value=...`` and
    ``missing_policy="empty_is_absent"`` when an empty list means true absence.

    Args:
        group: ``NeuronGroup`` or ``ConnectionGroup`` to test.
        attribute: Attribute name. Dotted paths are supported for dict-like
            attributes, e.g. ``"loadings.PC1"``.
        mode: ``"auto"``, ``"numeric"``, ``"categorical"``,
            ``"set_membership"``, or ``"binary"``.
        value: Required target value for ``set_membership``. Optional target
            value for ``binary``; defaults to ``True``.
        missing_policy: ``"exclude_and_report"``, ``"empty_is_absent"``,
            ``"missing_is_absent"``, or ``"missing_is_unknown"``.
        eligible_filter: Optional dict of attribute filters defining the
            eligible observed/reference universe before testing.
        reference: Optional comparison group, group name, ``"network"``, or
            ``"complement"``. When supplied with a group, this is a direct
            group-vs-group comparison.
        null_model: ``"network"``, ``"complement"``, ``"size_matched"``,
            ``"shuffled"``, ``"permutation"``, ``"sample"``, or ``"group"``.
        attribute_type: ``"auto"``, ``"categorical"``, or ``"numeric"``.
        n_resamples: Number of empirical samples for numeric network nulls.
        random_state: Optional seed for deterministic empirical nulls.
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns:
        JSON-serializable result dict with observed/reference summaries and
        per-value or numeric test results.
    """
    _validate_group(group)
    _validate_options(
        attribute,
        attribute_type,
        null_model,
        alternative,
        n_resamples,
        mode,
        value,
        missing_policy,
    )

    null_key = null_model.lower()
    observed_members_all = _members(group)
    reference_members, reference_name, direct_group_comparison = _resolve_reference(
        group, reference, null_key
    )
    observed_members = _eligible_members(observed_members_all, eligible_filter)
    reference_members = _eligible_members(reference_members, eligible_filter)

    resolved_mode = _resolve_mode(
        observed_members + reference_members,
        attribute,
        mode,
        attribute_type,
        missing_policy,
    )
    observed_values, observed_missing = _records_for_mode(
        observed_members,
        attribute,
        resolved_mode,
        value,
        missing_policy,
    )
    reference_values, reference_missing = _records_for_mode(
        reference_members,
        attribute,
        resolved_mode,
        value,
        missing_policy,
    )
    _validate_resolved_policy(resolved_mode, missing_policy, observed_values + reference_values)
    missingness = _missingness_result(
        len(observed_members),
        observed_missing,
        len(reference_members),
        reference_missing,
    )
    result: dict[str, Any] = {
        "group": getattr(group, "group_name", None),
        "element": "node" if isinstance(group, NeuronGroup) else "edge",
        "attribute": attribute,
        "mode": resolved_mode,
        "attribute_type": resolved_mode if resolved_mode in {"numeric", "categorical"} else attribute_type,
        "value": value,
        "missing_policy": missing_policy,
        "eligible_filter": eligible_filter or {},
        "null_model": null_model,
        "reference": {
            "name": reference_name,
            "size": len(reference_members),
            "valid_size": len(reference_values),
            "missing": reference_missing,
        },
        "observed": {
            "size": len(observed_members_all),
            "eligible_size": len(observed_members),
            "valid_size": len(observed_values),
            "missing": observed_missing,
        },
        "missingness": missingness,
    }
    if not observed_members:
        raise ValueError("No observed group members are eligible for this test")
    if not reference_members:
        raise ValueError("No reference members are eligible for this test")
    if not observed_values or not reference_values:
        result["results"] = []
        result["warning"] = (
            f"No valid observed or reference values for attribute '{attribute}'. "
            "Inspect the missingness block before interpreting depletion."
        )
        return result

    if resolved_mode == "numeric":
        result["results"] = [
            _numeric_result(
                observed_values,
                reference_values,
                direct_group_comparison=direct_group_comparison,
                n_resamples=n_resamples,
                random_state=random_state,
                alternative=alternative,
            )
        ]
    elif resolved_mode in {"set_membership", "binary"}:
        tested_value = True if resolved_mode == "binary" and value is None else value
        result["results"] = [
            _binary_result(
                observed_values,
                reference_values,
                tested_value,
                use_hypergeom=(reference_name == "network" and not direct_group_comparison),
                alternative=alternative,
            )
        ]
    else:
        result["results"] = _categorical_results(
            observed_values,
            reference_values,
            use_hypergeom=(reference_name == "network" and not direct_group_comparison),
            alternative=alternative,
        )
    return result


def test_group_attribute_enrichment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for ``group_attribute_enrichment``.

    This name reads naturally in scripts while preserving the shorter canonical
    function name.
    """
    return group_attribute_enrichment(*args, **kwargs)


def _validate_group(group: Any) -> None:
    if not isinstance(group, (NeuronGroup, ConnectionGroup)):
        raise TypeError("group must be a NeuronGroup or ConnectionGroup")


def _validate_options(
    attribute: str,
    attribute_type: str,
    null_model: str,
    alternative: str,
    n_resamples: int,
    mode: str,
    value: Any | None,
    missing_policy: str,
) -> None:
    if not attribute:
        raise ValueError("attribute must be a non-empty string")
    if attribute_type not in {"auto", "categorical", "numeric"}:
        raise ValueError("attribute_type must be 'auto', 'categorical', or 'numeric'")
    if mode not in MODES:
        raise ValueError("mode must be 'auto', 'numeric', 'categorical', 'set_membership', or 'binary'")
    if mode == "set_membership" and value is None:
        raise ValueError("value is required when mode='set_membership'")
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(
            "missing_policy must be 'exclude_and_report', 'empty_is_absent', "
            "'missing_is_absent', or 'missing_is_unknown'"
        )
    if null_model.lower() not in NETWORK_NULLS | COMPLEMENT_NULLS | EMPIRICAL_NULLS | {"group"}:
        raise ValueError(
            "null_model must be 'network', 'complement', 'size_matched', "
            "'shuffled', 'permutation', 'sample', or 'group'"
        )
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")


def _validate_resolved_policy(
    mode: str,
    missing_policy: str,
    values: list[Any],
) -> None:
    if mode == "numeric":
        if missing_policy in {"empty_is_absent", "missing_is_absent", "missing_is_unknown"}:
            raise ValueError(
                "Numeric enrichment requires missing_policy='exclude_and_report'"
            )
        if any(not _is_numeric_scalar(value) for value in values):
            raise ValueError("Numeric enrichment requires scalar numeric values")


def _members(group: NeuronGroup | ConnectionGroup) -> list[Neuron | Connection]:
    return list(group.values())


def _network_members(group: NeuronGroup | ConnectionGroup) -> list[Neuron | Connection]:
    if isinstance(group, NeuronGroup):
        return list(group.network.neurons.values())
    if group.network.connections:
        return list(group.network.connections.values())
    connections = {}
    for neuron in group.network.neurons.values():
        connections.update(getattr(neuron, "out_connections", {}))
        connections.update(getattr(neuron, "in_connections", {}))
    return list(connections.values())


def _resolve_reference(
    group: NeuronGroup | ConnectionGroup,
    reference: NeuronGroup | ConnectionGroup | str | None,
    null_key: str,
) -> tuple[list[Neuron | Connection], str, bool]:
    direct_group_comparison = null_key == "group"
    if isinstance(reference, str):
        ref_key = reference.lower()
        if ref_key in NETWORK_NULLS:
            reference = None
        elif ref_key in COMPLEMENT_NULLS:
            null_key = "complement"
            reference = None
        elif reference in group.network.groups:
            reference = group.network.groups[reference]
        else:
            raise ValueError(f"Reference group '{reference}' not found")

    if reference is not None:
        _validate_group(reference)
        if isinstance(group, NeuronGroup) != isinstance(reference, NeuronGroup):
            raise TypeError("group and reference must contain the same element type")
        if group.network is not reference.network:
            raise ValueError("group and reference must belong to the same network")
        return _members(reference), getattr(reference, "group_name", "reference"), True

    universe = _network_members(group)
    if null_key in COMPLEMENT_NULLS:
        group_ids = {_member_identity(member) for member in _members(group)}
        return [
            member for member in universe
            if _member_identity(member) not in group_ids
        ], "complement", True
    return universe, "network", direct_group_comparison


def _member_identity(member: Neuron | Connection) -> str | tuple[str, str, Any]:
    if isinstance(member, Neuron):
        return member.name
    return (member.pre.name, member.post.name, member.uid)


def _records_with_attribute(members: Iterable[Neuron | Connection], attribute: str) -> list[Any]:
    values = []
    for member in members:
        value = _get_attribute(member, attribute)
        if not _is_missing(value):
            values.append(value)
    return values


def _get_attribute(member: Neuron | Connection, attribute: str) -> Any:
    if isinstance(member, Neuron) and attribute == "degree":
        return len(member.in_connections) + len(member.out_connections)
    if isinstance(member, Connection) and attribute in {"type", "connection_type"}:
        return member.connection_type

    current: Any = member
    for part in attribute.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _is_empty_container(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset, dict)) and len(value) == 0


def _eligible_members(
    members: Iterable[Neuron | Connection],
    eligible_filter: dict[str, Any] | None,
) -> list[Neuron | Connection]:
    if not eligible_filter:
        return list(members)
    return [
        member for member in members
        if all(_matches_filter(_get_attribute(member, attr), expected)
               for attr, expected in eligible_filter.items())
    ]


def _matches_filter(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(_matches_filter(actual, item) for item in expected)
    if isinstance(actual, (list, tuple, set, frozenset)):
        return expected in actual
    return actual == expected


def _resolve_mode(
    members: list[Neuron | Connection],
    attribute: str,
    mode: str,
    attribute_type: str,
    missing_policy: str,
) -> str:
    if mode != "auto":
        return mode
    if attribute_type != "auto":
        return attribute_type
    values = [
        _get_attribute(member, attribute)
        for member in members
        if not _is_missing(_get_attribute(member, attribute))
    ]
    if not values:
        return "categorical"
    if all(_is_numeric_scalar(value) for value in values):
        return "numeric"
    if missing_policy == "empty_is_absent" and any(
        isinstance(value, (list, tuple, set, frozenset)) for value in values
    ):
        return "categorical"
    return "categorical"


def _records_for_mode(
    members: Iterable[Neuron | Connection],
    attribute: str,
    mode: str,
    value: Any | None,
    missing_policy: str,
) -> tuple[list[Any], int]:
    if mode == "set_membership":
        return _set_membership_records(members, attribute, value, missing_policy)
    if mode == "binary":
        return _binary_records(members, attribute, value, missing_policy)
    return _attribute_records(members, attribute, missing_policy)


def _attribute_records(
    members: Iterable[Neuron | Connection],
    attribute: str,
    missing_policy: str,
) -> tuple[list[Any], int]:
    values = []
    missing = 0
    for member in members:
        raw = _get_attribute(member, attribute)
        if _is_missing(raw) or _is_empty_container(raw):
            if missing_policy == "missing_is_unknown":
                values.append("Unknown")
            elif missing_policy in {"empty_is_absent", "missing_is_absent"}:
                values.append([])
            else:
                missing += 1
            continue
        values.append(raw)
    return values, missing


def _set_membership_records(
    members: Iterable[Neuron | Connection],
    attribute: str,
    value: Any,
    missing_policy: str,
) -> tuple[list[bool], int]:
    values = []
    missing = 0
    for member in members:
        raw = _get_attribute(member, attribute)
        if _is_missing(raw) or _is_empty_container(raw):
            if missing_policy in {"empty_is_absent", "missing_is_absent"}:
                values.append(False)
            else:
                missing += 1
            continue
        values.append(_contains_value(raw, value))
    return values, missing


def _binary_records(
    members: Iterable[Neuron | Connection],
    attribute: str,
    value: Any | None,
    missing_policy: str,
) -> tuple[list[bool], int]:
    target = True if value is None else value
    values = []
    missing = 0
    for member in members:
        raw = _get_attribute(member, attribute)
        if _is_missing(raw) or _is_empty_container(raw):
            if missing_policy in {"empty_is_absent", "missing_is_absent"}:
                values.append(False)
            else:
                missing += 1
            continue
        values.append(raw == target)
    return values, missing


def _contains_value(raw: Any, value: Any) -> bool:
    if isinstance(raw, dict):
        return value in raw and bool(raw[value])
    if isinstance(raw, (list, tuple, set, frozenset)):
        return value in raw
    return raw == value


def _missingness_result(
    observed_n: int,
    observed_missing: int,
    reference_n: int,
    reference_missing: int,
) -> dict[str, Any]:
    observed_valid = observed_n - observed_missing
    reference_valid = reference_n - reference_missing
    table = [[observed_missing, observed_valid], [reference_missing, reference_valid]]
    fisher = stats.fisher_exact(table, alternative="two-sided") if observed_n and reference_n else None
    observed_fraction = observed_missing / observed_n if observed_n else 0.0
    reference_fraction = reference_missing / reference_n if reference_n else 0.0
    return {
        "observed_missing": int(observed_missing),
        "observed_missing_fraction": float(observed_fraction),
        "reference_missing": int(reference_missing),
        "reference_missing_fraction": float(reference_fraction),
        "p_value": None if fisher is None else float(fisher.pvalue),
        "direction": _missingness_direction(observed_fraction, reference_fraction),
    }


def _missingness_direction(observed_fraction: float, reference_fraction: float) -> str:
    if observed_fraction > reference_fraction:
        return "missing_enriched"
    if observed_fraction < reference_fraction:
        return "missing_depleted"
    return "unchanged"


def _resolve_attribute_type(records: list[Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if all(_is_numeric_scalar(value) for value in records):
        return "numeric"
    return "categorical"


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (Number, np.number)) and not isinstance(value, bool)


def _numeric_result(
    observed_values: list[Any],
    reference_values: list[Any],
    *,
    direct_group_comparison: bool,
    n_resamples: int,
    random_state: int | None,
    alternative: str,
) -> dict[str, Any]:
    observed = np.asarray([float(value) for value in observed_values], dtype=float)
    reference = np.asarray([float(value) for value in reference_values], dtype=float)
    observed_mean = float(np.mean(observed))
    reference_mean = float(np.mean(reference))
    out: dict[str, Any] = {
        "statistic": "mean",
        "observed": observed_mean,
        "reference_mean": reference_mean,
        "reference_std": float(np.std(reference, ddof=1)) if len(reference) > 1 else 0.0,
        "delta": observed_mean - reference_mean,
        "fold_change": _safe_ratio(observed_mean, reference_mean),
        "observed_n": int(len(observed)),
        "reference_n": int(len(reference)),
    }

    if direct_group_comparison:
        out.update(_mann_whitney_summary(observed, reference, alternative))
        return out

    if len(observed) > len(reference):
        raise ValueError("Cannot draw size-matched samples larger than the reference")

    rng = np.random.default_rng(random_state)
    null_means = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample = rng.choice(reference, size=len(observed), replace=False)
        null_means[idx] = np.mean(sample)

    p_greater = (np.count_nonzero(null_means >= observed_mean) + 1) / (n_resamples + 1)
    p_less = (np.count_nonzero(null_means <= observed_mean) + 1) / (n_resamples + 1)
    p_two = min(1.0, 2.0 * min(p_greater, p_less))
    out.update({
        "p_value": _select_p_value(p_two, p_greater, p_less, alternative),
        "p_enrichment": float(p_greater),
        "p_depletion": float(p_less),
        "null_mean": float(np.mean(null_means)),
        "null_ci95": [
            float(np.percentile(null_means, 2.5)),
            float(np.percentile(null_means, 97.5)),
        ],
    })
    return out


def _mann_whitney_summary(
    observed: np.ndarray,
    reference: np.ndarray,
    alternative: str,
) -> dict[str, Any]:
    p_greater = stats.mannwhitneyu(observed, reference, alternative="greater").pvalue
    p_less = stats.mannwhitneyu(observed, reference, alternative="less").pvalue
    p_two = stats.mannwhitneyu(observed, reference, alternative="two-sided").pvalue
    return {
        "p_value": _select_p_value(p_two, p_greater, p_less, alternative),
        "p_enrichment": float(p_greater),
        "p_depletion": float(p_less),
        "null_mean": float(np.mean(reference)),
        "null_ci95": [None, None],
    }


def _binary_result(
    observed_values: list[bool],
    reference_values: list[bool],
    value: Any,
    *,
    use_hypergeom: bool,
    alternative: str,
) -> dict[str, Any]:
    observed_n = len(observed_values)
    reference_n = len(reference_values)
    k = sum(bool(item) for item in observed_values)
    ref_k = sum(bool(item) for item in reference_values)
    observed_fraction = k / observed_n
    reference_fraction = ref_k / reference_n
    if use_hypergeom:
        p_enrichment = stats.hypergeom.sf(k - 1, reference_n, ref_k, observed_n)
        p_depletion = stats.hypergeom.cdf(k, reference_n, ref_k, observed_n)
        p_two = min(1.0, 2.0 * min(p_enrichment, p_depletion))
        odds_ratio = _safe_odds_ratio(
            k,
            observed_n - k,
            ref_k - k,
            reference_n - ref_k - observed_n + k,
        )
    else:
        table = [[k, observed_n - k], [ref_k, reference_n - ref_k]]
        p_enrichment = stats.fisher_exact(table, alternative="greater").pvalue
        p_depletion = stats.fisher_exact(table, alternative="less").pvalue
        fisher = stats.fisher_exact(table, alternative="two-sided")
        p_two = fisher.pvalue
        odds_ratio = fisher.statistic

    return {
        "value": value,
        "observed_count": int(k),
        "observed_fraction": float(observed_fraction),
        "reference_count": int(ref_k),
        "reference_fraction": float(reference_fraction),
        "fold_enrichment": _safe_ratio(observed_fraction, reference_fraction),
        "log2_fold_enrichment": _safe_log2_ratio(observed_fraction, reference_fraction),
        "odds_ratio": _finite_or_none(odds_ratio),
        "p_value": _select_p_value(p_two, p_enrichment, p_depletion, alternative),
        "p_enrichment": float(p_enrichment),
        "p_depletion": float(p_depletion),
        "direction": _direction(observed_fraction, reference_fraction),
        "basis": "binary_membership",
    }


def _categorical_results(
    observed_values: list[Any],
    reference_values: list[Any],
    *,
    use_hypergeom: bool,
    alternative: str,
) -> list[dict[str, Any]]:
    observed_sets = [_categorical_values(value) for value in observed_values]
    reference_sets = [_categorical_values(value) for value in reference_values]
    observed_counts = Counter(value for values in observed_sets for value in values)
    reference_counts = Counter(value for values in reference_sets for value in values)
    observed_n = len(observed_sets)
    reference_n = len(reference_sets)
    rows = []
    p_values = []

    all_values = set(reference_counts) | set(observed_counts)
    for value in sorted(all_values, key=lambda item: str(item)):
        k = observed_counts.get(value, 0)
        ref_k = reference_counts.get(value, 0)
        observed_fraction = k / observed_n
        reference_fraction = ref_k / reference_n
        if use_hypergeom:
            p_enrichment = stats.hypergeom.sf(k - 1, reference_n, ref_k, observed_n)
            p_depletion = stats.hypergeom.cdf(k, reference_n, ref_k, observed_n)
            p_two = min(1.0, 2.0 * min(p_enrichment, p_depletion))
            odds_ratio = _safe_odds_ratio(
                k,
                observed_n - k,
                ref_k - k,
                reference_n - ref_k - observed_n + k,
            )
        else:
            table = [[k, observed_n - k], [ref_k, reference_n - ref_k]]
            p_enrichment = stats.fisher_exact(table, alternative="greater").pvalue
            p_depletion = stats.fisher_exact(table, alternative="less").pvalue
            fisher = stats.fisher_exact(table, alternative="two-sided")
            p_two = fisher.pvalue
            odds_ratio = fisher.statistic

        p_value = _select_p_value(p_two, p_enrichment, p_depletion, alternative)
        p_values.append(p_value)
        rows.append({
            "value": value,
            "observed_count": int(k),
            "observed_fraction": float(observed_fraction),
            "reference_count": int(ref_k),
            "reference_fraction": float(reference_fraction),
            "fold_enrichment": _safe_ratio(observed_fraction, reference_fraction),
            "log2_fold_enrichment": _safe_log2_ratio(observed_fraction, reference_fraction),
            "odds_ratio": _finite_or_none(odds_ratio),
            "p_value": float(p_value),
            "p_enrichment": float(p_enrichment),
            "p_depletion": float(p_depletion),
            "direction": _direction(observed_fraction, reference_fraction),
        })

    for row, q_value in zip(rows, _benjamini_hochberg(p_values)):
        row["q_value"] = q_value
    return rows


def _categorical_values(value: Any) -> set[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return {item for item in value if not _is_missing(item)}
    return {value}


def _select_p_value(p_two: float, p_greater: float, p_less: float, alternative: str) -> float:
    if alternative == "greater":
        return float(p_greater)
    if alternative == "less":
        return float(p_less)
    return float(p_two)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _safe_log2_ratio(numerator: float, denominator: float) -> float | None:
    ratio = _safe_ratio(numerator, denominator)
    if ratio is None or ratio <= 0:
        return None
    return float(math.log2(ratio))


def _safe_odds_ratio(a: int, b: int, c: int, d: int) -> float | None:
    if min(a, b, c, d) < 0:
        return None
    if b * c == 0:
        return None
    return float((a * d) / (b * c))


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _direction(observed_fraction: float, reference_fraction: float) -> str:
    if observed_fraction > reference_fraction:
        return "enriched"
    if observed_fraction < reference_fraction:
        return "depleted"
    return "unchanged"


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [0.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[idx] * total / rank)
        adjusted[idx] = float(min(1.0, running))
    return adjusted
