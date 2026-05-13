"""CI guard: no bare ``np.random.*`` or ``random.*`` outside the RNG factory.

Reproducibility hinges on every randomness call going through
``cedne.random.get_rng`` or ``cedne.random.get_seed``. The audit caught the
old pattern (unseeded ``np.random.uniform`` in the optimizer, global
``np.random.seed`` in graphtools) and we replaced it. This test makes the
fix permanent: a new bare ``np.random.uniform`` slipping into the
codebase will fail CI here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Forbidden patterns. Each is a regex; a match anywhere in a non-allow-listed
# source file flags the violation.
_FORBIDDEN = [
    # ``np.random.seed(...)`` mutates global state. Always wrong.
    re.compile(r"\bnp\.random\.seed\b"),
    # ``np.random.<dist>(...)`` uses the unseeded global RNG. Use
    # ``get_rng(seed).<dist>(...)`` instead.
    re.compile(
        r"\bnp\.random\.(uniform|randn|randint|rand|choice|normal|standard_normal|shuffle|permutation)\b"
    ),
    # ``random.<fn>(...)`` from the stdlib has the same problem for
    # scientific code (different RNG, also unseeded globally).
    re.compile(r"(?<![A-Za-z_])random\.(seed|uniform|randint|choice|shuffle)\b"),
]

# Files allowed to use the bare names:
#   * cedne/random.py itself wraps np.random.default_rng / SeedSequence.
#   * cedne/core/io.py uses random.choice for short ID generation (not
#     scientifically meaningful — UIDs only).
_ALLOWED = {
    Path("src/cedne/random.py"),
    Path("src/cedne/core/io.py"),
}

# Skip tree-archive copies and the JAX-compiled module (uses jax.random,
# not np.random — pattern doesn't match anyway, but archives are noise).
_SKIP_DIRS = {"archive", "__pycache__", ".pytest_cache"}


def _iter_source_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src" / "cedne"
    out: list[Path] = []
    for p in src.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.relative_to(repo_root).parts):
            continue
        out.append(p)
    return out


def test_no_bare_np_random_calls_outside_factory():
    """Fail if any forbidden RNG pattern appears outside the allowlist.

    To add a new randomness call site: import from cedne.random and use
    ``get_rng(seed).method(...)`` (Generator API) or ``get_seed(seed)`` to
    obtain an int for libs that want one. If you genuinely need a bare
    np.random.* call, add the file to ``_ALLOWED`` above with a short
    justification and document the call.
    """
    repo_root = Path(__file__).resolve().parent.parent
    violations: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(repo_root)
        if rel in _ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comment-only lines so the comments in this very file
            # don't flag themselves when copy-pasted as examples.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in _FORBIDDEN:
                if pat.search(line):
                    violations.append(f"{rel}:{lineno}: {line.strip()}")
                    break

    if violations:
        msg = (
            "Forbidden RNG calls found. Route randomness through "
            "cedne.random.get_rng / get_seed (see src/cedne/random.py):\n  "
            + "\n  ".join(violations)
        )
        pytest.fail(msg)
