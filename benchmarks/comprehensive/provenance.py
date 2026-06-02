"""
Shared provenance capture for benchmark results (Phase I.2).

Both ``capture_baseline.py`` (bundle-level ``environment.json``) and
``results.py::BenchmarkResult`` (per-run ``system.provenance``) read from
``capture_run_provenance()``. Factoring out this surface keeps the two
call sites in lockstep when a new field is added.

The ``run_id`` mechanism: every ``run_parallel.py`` invocation sets
``SCX_BENCH_RUN_ID`` in the submitted job environment (ULID-equivalent —
timestamp-ordered, collision-resistant). Results written during that run
all share the same ID so the dashboard can group them without
cross-referencing submitit log paths.
"""

from __future__ import annotations

import logging
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Crockford base32 alphabet for ULID-like IDs (no I/L/O/U — operator-readable).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _crockford_encode(value: int, width: int) -> str:
    """Encode ``value`` in Crockford base32, padded to ``width`` chars."""
    chars = []
    for _ in range(width):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_run_id() -> str:
    """Generate a new ULID-style run identifier.

    26-char string: 10 chars of timestamp (ms) + 16 chars of randomness.
    Timestamp-ordered so `ls -1` on result directories sorts runs
    chronologically; random tail prevents collisions between parallel
    submitters. Matches the ULID spec for shape but we don't pull a
    library dep for it — stdlib is enough.
    """
    ts_ms = int(time.time() * 1000)
    rand = secrets.token_bytes(10)
    rand_int = int.from_bytes(rand, "big")
    return _crockford_encode(ts_ms, 10) + _crockford_encode(rand_int, 16)


def current_run_id() -> str:
    """Return the ambient run ID from ``SCX_BENCH_RUN_ID`` or mint a new one.

    ``run_parallel.py`` sets the env var once per invocation and exports it
    into every submitit job, so all per-triple results share the same ID.
    Benchmarks invoked outside that launcher get a fresh ID per result —
    acceptable since those are typically one-shot debugging sessions.
    """
    existing = os.environ.get("SCX_BENCH_RUN_ID", "").strip()
    if existing:
        return existing
    minted = new_run_id()
    os.environ["SCX_BENCH_RUN_ID"] = minted
    return minted


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *cmd], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def capture_run_provenance() -> dict[str, Any]:
    """Minimal per-run provenance. Shape is stable — add, don't mutate.

    Embedded in every raw JSON result via ``BenchmarkResult.__post_init__``
    so downstream consumers (gate, dashboard, ad-hoc analysis) always have
    the git SHA + thread pinning that produced the number, without having
    to walk back to the bundle-level environment.json.
    """
    repo = _repo_root()
    git_status = _git(["status", "--porcelain"], repo)
    return {
        "run_id": current_run_id(),
        "git_sha": _git(["rev-parse", "HEAD"], repo),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "git_dirty": git_status not in ("", "unknown"),
        "git_describe": _git(["describe", "--tags", "--always", "--dirty"], repo),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", ""),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "pyscx_features": os.environ.get("PYSCX_FEATURES", ""),
        # GPU DE v3 is the unconditional default since ACC-RUST-OPT-V2 §5 Phase
        # V1b; the `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` gates were removed. The
        # recorded `gpu_dispatch_route` (e.g. gpu_csc_v3) is now the route signal.
        # `SCX_GPU_DE_V3_TRACE` is a debug-only stderr trace, still captured.
        "scx_gpu_de_v3_trace": os.environ.get("SCX_GPU_DE_V3_TRACE", ""),
    }
