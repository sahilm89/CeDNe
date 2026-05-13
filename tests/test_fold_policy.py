"""Tests for cedne.core.fold_policy.

Covers each aggregator on each kind, plus the policy-set / serialization
round-trip used by the web UI for fold-policy provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from cedne.core.fold_policy import (
    DROP,
    FoldPolicy,
    FoldPolicySet,
    apply_policy,
)


class TestFoldPolicyValidation:
    def test_drop_is_default(self):
        p = FoldPolicy(name="anything", kind="scalar")
        assert p.aggregator == DROP

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown FoldPolicy kind"):
            FoldPolicy(name="x", kind="not-a-kind")  # type: ignore[arg-type]

    def test_mismatched_aggregator_raises(self):
        # mean is scalar; not valid for list-kind attrs.
        with pytest.raises(ValueError, match="not valid for kind"):
            FoldPolicy(name="ligands", kind="list", aggregator="mean")

    def test_drop_is_always_valid(self):
        for kind in ("scalar", "timeseries", "list", "dict", "categorical"):
            FoldPolicy(name="x", kind=kind, aggregator=DROP)  # type: ignore[arg-type]


class TestScalarAggregators:
    def test_mean_median_max_min(self):
        p = FoldPolicy(name="v", kind="scalar", aggregator="mean")
        assert apply_policy(p, [1, 2, 3, 4]) == 2.5
        assert apply_policy(FoldPolicy("v", "scalar", "median"), [1, 2, 3, 4]) == 2.5
        assert apply_policy(FoldPolicy("v", "scalar", "max"), [1, 2, 3]) == 3.0
        assert apply_policy(FoldPolicy("v", "scalar", "min"), [1, 2, 3]) == 1.0

    def test_sum(self):
        # The default policy for connection weights — historically
        # contract_connections accumulated weights via ``weight += ...``.
        # ``sum`` of an empty constituent list returns 0.0 (also matches
        # the pre-Phase-2 initial ``weight = 0`` behavior).
        assert apply_policy(FoldPolicy("w", "scalar", "sum"), [1, 2, 3]) == 6.0
        assert apply_policy(FoldPolicy("w", "scalar", "sum"), []) == 0.0
        assert apply_policy(FoldPolicy("w", "scalar", "sum"), [None, None]) == 0.0

    def test_mode_picks_most_common(self):
        result = apply_policy(FoldPolicy("v", "scalar", "mode"), [1, 2, 2, 3, 3, 3])
        assert result == 3.0

    def test_mode_handles_ties_deterministically(self):
        # All distinct → multimode returns every element; first wins.
        result = apply_policy(FoldPolicy("v", "scalar", "mode"), [1, 2, 3])
        assert result == 1.0

    def test_skips_non_numeric_and_nan(self):
        result = apply_policy(
            FoldPolicy("v", "scalar", "mean"),
            [1, "not a number", None, float("nan"), 3],
        )
        assert result == 2.0  # 1 and 3 survived

    def test_empty_returns_none(self):
        assert apply_policy(FoldPolicy("v", "scalar", "mean"), []) is None
        assert apply_policy(FoldPolicy("v", "scalar", "mean"), [None, None]) is None


class TestTimeseriesAggregators:
    def test_mean_median_max(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_equal(
            apply_policy(FoldPolicy("rec", "timeseries", "timeseries_mean"), [a, b]),
            np.array([2.0, 3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            apply_policy(FoldPolicy("rec", "timeseries", "timeseries_median"), [a, b]),
            np.array([2.0, 3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            apply_policy(FoldPolicy("rec", "timeseries", "timeseries_max"), [a, b]),
            np.array([3.0, 4.0, 5.0]),
        )

    def test_truncates_to_shortest(self):
        # Per-design: differing trial lengths truncate to shortest rather
        # than pad with NaN. Matches existing recordings-data behavior.
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 20.0, 30.0])
        out = apply_policy(FoldPolicy("rec", "timeseries", "timeseries_mean"), [a, b])
        assert out.shape == (3,)
        np.testing.assert_array_equal(out, np.array([5.5, 11.0, 16.5]))

    def test_skips_empty_arrays(self):
        a = np.array([1.0, 2.0])
        out = apply_policy(
            FoldPolicy("rec", "timeseries", "timeseries_mean"),
            [a, None, np.array([])],
        )
        np.testing.assert_array_equal(out, a)


class TestListAggregators:
    def test_keep_all_preserves_duplicates_and_order(self):
        out = apply_policy(
            FoldPolicy("ligands", "list", "keep_all"),
            [["ACh", "Glu"], ["GABA", "ACh"]],
        )
        assert out == ["ACh", "Glu", "GABA", "ACh"]

    def test_set_union_dedupes_preserving_first_seen_order(self):
        out = apply_policy(
            FoldPolicy("ligands", "list", "set_union"),
            [["ACh", "Glu"], ["GABA", "ACh"]],
        )
        assert out == ["ACh", "Glu", "GABA"]

    def test_mode_count_picks_most_frequent(self):
        out = apply_policy(
            FoldPolicy("type", "list", "mode_count"),
            [["sensory"], ["sensory"], ["motor"], ["sensory"]],
        )
        assert out == "sensory"

    def test_list_accepts_non_iterable_singletons(self):
        # A constituent may carry a bare value when expected to be list.
        out = apply_policy(FoldPolicy("tag", "list", "set_union"), ["a", ["b"], "a"])
        assert out == ["a", "b"]


class TestDictAggregators:
    def test_union_first_observed_value_wins(self):
        # union shape = same as a single dict: {k: v}, not {k: [v,...]}.
        # First observation of a key sticks (deterministic; matches the
        # pre-Phase-2 contract_connections behavior we preserve).
        out = apply_policy(
            FoldPolicy("receptors", "dict", "dict_union"),
            [{"glr-1": 0.8}, {"glr-1": 1.2, "nmr-1": 0.4}],
        )
        assert out == {"glr-1": 0.8, "nmr-1": 0.4}

    def test_union_with_no_collisions(self):
        # No key collision → each key's value passes through unchanged.
        out = apply_policy(
            FoldPolicy("receptors", "dict", "dict_union"),
            [{"a": 1}, {"b": 2}, {"c": 3}],
        )
        assert out == {"a": 1, "b": 2, "c": 3}

    def test_collated_lists_all_observations(self):
        # Collated is the explicit "I want every observation" form.
        # Shape changes: {k: [v0, v1, ...]}.
        out = apply_policy(
            FoldPolicy("receptors", "dict", "dict_collated"),
            [{"a": 1}, {"a": 2, "b": 3}],
        )
        assert out == {"a": [1, 2], "b": [3]}

    def test_intersection_keeps_only_common_keys_first_value(self):
        # Only keys present in EVERY constituent; first observed wins.
        out = apply_policy(
            FoldPolicy("receptors", "dict", "dict_intersection"),
            [{"a": 1, "b": 2}, {"a": 10, "c": 30}, {"a": 100, "b": 200}],
        )
        assert out == {"a": 1}

    def test_empty_returns_none(self):
        assert apply_policy(FoldPolicy("r", "dict", "dict_union"), [None, None]) is None


class TestCategoricalAggregators:
    def test_mode_picks_most_common(self):
        out = apply_policy(
            FoldPolicy("type", "categorical", "mode"),
            ["sensory", "sensory", "motor"],
        )
        assert out == "sensory"

    def test_keep_all_dedupes_preserving_order(self):
        out = apply_policy(
            FoldPolicy("type", "categorical", "keep_all"),
            ["sensory", "motor", "sensory"],
        )
        assert out == ["sensory", "motor"]


class TestDropAndPolicySet:
    def test_drop_returns_none_regardless_of_input(self):
        for kind in ("scalar", "timeseries", "list", "dict", "categorical"):
            p = FoldPolicy(name="x", kind=kind, aggregator=DROP)  # type: ignore[arg-type]
            assert apply_policy(p, [1, 2, 3]) is None

    def test_set_get_falls_back_to_default(self):
        s = FoldPolicySet()
        s.add(FoldPolicy("recording", "timeseries", "timeseries_mean"))
        # Registered:
        assert s.get("recording", "timeseries").aggregator == "timeseries_mean"
        # Unregistered: drop sentinel of the kind the caller asked for.
        fallback = s.get("unknown_attr", "scalar")
        assert fallback.aggregator == DROP
        assert fallback.kind == "scalar"

    def test_set_serialization_roundtrip(self):
        s = FoldPolicySet(
            policies={
                "recording": FoldPolicy("recording", "timeseries", "timeseries_mean"),
                "type": FoldPolicy("type", "categorical", "mode"),
            },
            default_aggregator=DROP,
        )
        restored = FoldPolicySet.from_dict(s.to_dict())
        assert restored.policies == s.policies
        assert restored.default_aggregator == s.default_aggregator

    def test_default_aggregator_is_drop(self):
        s = FoldPolicySet()
        assert s.default_aggregator == DROP
