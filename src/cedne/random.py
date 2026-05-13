"""Package-wide RNG factory with statistically-independent streams.

CeDNe historically mixed ``np.random.uniform``, ``np.random.seed`` (which
mutates global numpy state), and ad-hoc ``random_state=42`` defaults. Two
problems for scientists:

  1. The optimizer picked a different initial guess every run — the same
     model fit to the same data produced different parameters.
  2. ``np.random.seed`` in one helper silently affected every subsequent
     ``np.random.*`` call in the same Python process.

This module replaces both with the NumPy-recommended **SeedSequence +
spawn** pattern. Every ``get_rng()`` call returns a *local* Generator
spawned from a module-level root SeedSequence:

* Same code + same input → same result every run (root is seeded from
  ``RANDOM_SEED`` in :mod:`cedne.core.config`).
* Two unrelated callers get **statistically independent** streams (spawn
  uses a counter-based hash; this is what NumPy designed it for).
* The root can be overridden once via :func:`set_root_seed` (e.g. to pull
  from an environment variable, or to make a run non-reproducible by
  passing ``None``).

Reference: https://numpy.org/doc/stable/reference/random/parallel.html

CI gate: ``tests/test_no_unseeded_rng.py`` forbids bare ``np.random.*``
calls inside ``src/cedne/`` outside this module.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


_root: Optional[np.random.SeedSequence] = None
"""Module-level root SeedSequence. Lazy-init from RANDOM_SEED on first use."""


def _get_root() -> np.random.SeedSequence:
    global _root
    if _root is None:
        from cedne.core.config import RANDOM_SEED

        _root = np.random.SeedSequence(RANDOM_SEED)
    return _root


def set_root_seed(seed: Optional[Union[int, np.random.SeedSequence]]) -> None:
    """Reset the package-wide root SeedSequence.

    All subsequent ``get_rng()`` / ``get_seed()`` calls with ``seed=None``
    will spawn children from the new root. Existing Generators are
    unaffected.

    Args:
        seed: An integer (reproducible) or ``None`` (OS entropy — every
            run gets a different root). Accepts an existing
            ``SeedSequence`` too, for advanced users threading a root
            through subprocesses.
    """
    global _root
    if isinstance(seed, np.random.SeedSequence):
        _root = seed
    else:
        _root = np.random.SeedSequence(seed)


def get_rng(
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> np.random.Generator:
    """Return a local ``numpy.random.Generator``.

    Args:
        seed:
            * ``None`` (default) → spawn a fresh independent child from
              the package root. Deterministic from ``RANDOM_SEED``;
              statistically independent of every other ``get_rng()``
              call in the same process.
            * ``int`` → deterministic Generator from that seed alone
              (decoupled from the root). Use when the caller wants a
              named, repeatable stream that doesn't depend on call
              order.
            * ``SeedSequence`` → use directly.

    Returns:
        A local Generator. **Never** mutates the global ``np.random``
        state.

    Footgun:
        With ``seed=None``, the *order* of ``get_rng`` calls matters —
        spawn N times in a different order across runs and the children
        differ. If that bites a test, pass an explicit ``seed=`` to that
        site to decouple it from spawn order.
    """
    if seed is None:
        seq = _get_root().spawn(1)[0]
    elif isinstance(seed, np.random.SeedSequence):
        seq = seed
    else:
        seq = np.random.SeedSequence(seed)
    return np.random.default_rng(seq)


def get_seed(seed: Optional[Union[int, np.random.SeedSequence]] = None) -> int:
    """Return an integer seed with the same semantics as :func:`get_rng`.

    Use this for downstream libraries that want an ``int`` seed rather
    than a ``Generator``: NetworkX (``nx.directed_edge_swap(seed=...)``),
    scikit-learn (``NMF(random_state=...)``), Optuna
    (``TPESampler(seed=...)``), etc.

    Args:
        seed: Same semantics as :func:`get_rng`.

    Returns:
        An integer derived from the resolved SeedSequence's entropy. For
        ``seed=int``, returns that same int (``SeedSequence(n).entropy ==
        n`` for non-negative integers). For ``seed=None``, returns a
        deterministic int spawned from the root.
    """
    if seed is None:
        seq = _get_root().spawn(1)[0]
    elif isinstance(seed, np.random.SeedSequence):
        seq = seed
    else:
        seq = np.random.SeedSequence(seed)
    return int(seq.entropy)


__all__ = ["get_rng", "get_seed", "set_root_seed"]
