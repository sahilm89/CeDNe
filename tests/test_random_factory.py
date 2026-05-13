"""Tests for ``cedne.random``: the SeedSequence + spawn RNG factory.

The contract:
  * ``get_rng(seed=None)`` returns Generators that are deterministic from
    the package root SeedSequence AND statistically independent across
    successive calls.
  * ``get_rng(seed=int)`` is fully decoupled from the root and reproducible
    from the explicit seed alone.
  * ``set_root_seed(seed)`` resets the root so a fresh sequence of
    ``get_rng()`` calls behaves the same way again.
  * ``get_seed`` mirrors ``get_rng`` for libraries that want an ``int``
    instead of a Generator (NetworkX, sklearn, Optuna).
"""

import numpy as np
import pytest

from cedne import random as cedne_random


@pytest.fixture(autouse=True)
def reset_root():
    """Reset the package root before each test so spawn order is reproducible."""
    cedne_random.set_root_seed(42)
    yield
    cedne_random.set_root_seed(42)


class TestGetRng:
    def test_explicit_seed_is_deterministic(self):
        a = cedne_random.get_rng(123).standard_normal(10)
        b = cedne_random.get_rng(123).standard_normal(10)
        assert np.array_equal(a, b), "Same explicit seed must give same stream"

    def test_explicit_seed_is_decoupled_from_root(self):
        # Drawing with the same explicit seed must give the same values
        # regardless of the root's current state.
        a = cedne_random.get_rng(7).standard_normal(5)
        cedne_random.set_root_seed(99999)  # Move the root somewhere else.
        b = cedne_random.get_rng(7).standard_normal(5)
        assert np.array_equal(a, b)

    def test_none_seed_is_deterministic_from_root(self):
        # After resetting to the same root, the same sequence of get_rng()
        # calls must produce the same children.
        cedne_random.set_root_seed(42)
        first_run = [cedne_random.get_rng().standard_normal(4) for _ in range(3)]
        cedne_random.set_root_seed(42)
        second_run = [cedne_random.get_rng().standard_normal(4) for _ in range(3)]
        for a, b in zip(first_run, second_run):
            assert np.array_equal(a, b)

    def test_spawned_streams_are_independent(self):
        # Two successive get_rng() calls must produce different streams
        # (this is the whole point — spawn yields uncorrelated children).
        a = cedne_random.get_rng().standard_normal(50)
        b = cedne_random.get_rng().standard_normal(50)
        assert not np.array_equal(a, b)
        # Sanity check: correlation should be near zero, not 1.0 (which
        # would indicate the two Generators were entangled).
        corr = float(np.corrcoef(a, b)[0, 1])
        assert (
            abs(corr) < 0.5
        ), f"Spawned streams should be ~independent, got corr={corr}"

    def test_does_not_mutate_global_np_random(self):
        """The factory must never touch np.random global state."""
        np.random.seed(12345)
        before = np.random.get_state()
        _ = cedne_random.get_rng().standard_normal(100)
        _ = cedne_random.get_rng(99).standard_normal(100)
        after = np.random.get_state()
        # Both states must match — equality is per-element on the inner array.
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]


class TestSetRootSeed:
    def test_changing_root_changes_spawn_children(self):
        cedne_random.set_root_seed(1)
        a = cedne_random.get_rng().standard_normal(10)
        cedne_random.set_root_seed(2)
        b = cedne_random.get_rng().standard_normal(10)
        assert not np.array_equal(a, b)

    def test_none_root_pulls_os_entropy(self):
        # Setting None should pull from OS entropy → two consecutive None
        # roots produce different children (extremely high probability).
        cedne_random.set_root_seed(None)
        a = cedne_random.get_rng().standard_normal(10)
        cedne_random.set_root_seed(None)
        b = cedne_random.get_rng().standard_normal(10)
        # Astronomically improbable to collide; if this ever flakes the
        # entropy pool is broken or numpy SeedSequence semantics changed.
        assert not np.array_equal(a, b)


class TestGetSeed:
    def test_returns_int(self):
        seed = cedne_random.get_seed(7)
        assert isinstance(seed, int)

    def test_int_input_is_idempotent(self):
        # SeedSequence(n).entropy == n for non-negative integers, so the
        # caller can do TPESampler(seed=get_seed(7)) and recover the value.
        assert cedne_random.get_seed(7) == 7
        assert cedne_random.get_seed(0) == 0
        assert cedne_random.get_seed(2**31 - 1) == 2**31 - 1

    def test_none_seed_is_deterministic_from_root(self):
        cedne_random.set_root_seed(42)
        a = cedne_random.get_seed()
        cedne_random.set_root_seed(42)
        b = cedne_random.get_seed()
        assert a == b
