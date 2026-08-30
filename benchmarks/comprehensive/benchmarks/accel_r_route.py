"""rscx accelerator route-metadata gate (ORG-10.16-3b).

The R binding stamps the same ``scx-accel`` planner record pyscx writes to
``adata.uns["scx_accel"]`` (`rscx::scx_pca` et al. return it as the
``scx_accel`` list element / attribute, or write ``object@misc$scx_accel`` on
a Seurat object). This benchmark is the ``*_route_*_correct``-style gate for
that surface — the R analogue of ``hvg_route_gpu_correct`` and friends — so a
regression in rscx's dispatch or stamping fails ``gate_candidate.py`` rather
than passing silently (rscx has no CI lane).

It shells ``scripts/r_route_probe.R`` into the **rscx conda env** (the
``bpcells_runner.py`` / doublet ``_tool_env.py`` subprocess shape: the
orchestrator's env routing is by format key, and no Python env here carries
R). The probe runs on small synthetic matrices — the gated facts are routing
facts, dataset-independent — so the single scheduled dataset is just a
scheduling slot, not an input.

Gated metrics (all deterministic, floored at 1.0):

* ``r_pca_route_cpu_correct`` — ``scx_pca`` stamps ``cpu_csr`` /
  ``user_forced_cpu`` and the record carries the exact 17 wire keys in order.
* ``r_pca_method_arms_correct`` — the shared covariance-vs-randomized auto
  rule flips at ``COVARIANCE_PCA_THRESHOLD`` (rscx's one real dispatch
  branch, observable via the returned ``method``).
* ``r_wilcoxon_route_cpu_correct`` — ``scx_rank_genes_groups`` honestly
  stamps ``cpu_dense`` (the kernel densifies).
* ``r_pseudobulk_dex_route_correct`` — ``cpu_nb_glm`` with fallback
  ``"none"`` (pyscx parity: a first-class native CPU route).
* ``r_harmony_route_cpu_correct`` — ``scx_harmony_integrate`` stamps
  ``cpu_dense`` / ``user_forced_cpu`` (pyscx stamps ``harmony_integrate``;
  round-1 review found rscx did not).

Env absent → a typed missing result (``no_rscx_env``), mirroring the doublet
runners: an operator without the R env gets a skip they can see, not a red
gate.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result

logger = logging.getLogger(__name__)

# One variant; the probe is CPU-only R and needs no converted file.
FORMAT_KEY = "accel_r_route__rscx_cpu"

# Routing facts are dataset-independent (the probe builds synthetic matrices),
# so schedule exactly one cheap slot rather than one per tier.
FORMAT_DATASET_SCOPE: dict[str, frozenset[str]] = {
    FORMAT_KEY: frozenset({"pbmc3k"}),
}

# The 17-key wire contract, mirrored from `AccelExecutionInfo::fields()`
# (pinned in scx-accel's route_tests.rs and rscx's testthat; re-pinned here so
# the gate also fails on key drift).
WIRE_KEYS = [
    "route", "fallback_reason", "chunk_size", "graph_replay", "csc_available",
    "shards_decoded", "shards_uploaded", "math_mode", "spmm_policy",
    "rapids_version", "cuml_version", "cupy_version", "transfer_mode",
    "device_id", "bytes_uploaded", "n_shards_shufdelta_gpu", "resident_csr",
]

_PROBE = Path(__file__).resolve().parents[1] / "scripts" / "r_route_probe.R"


def accel_r_route_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="rscx accelerators (CPU, route metadata)", key=FORMAT_KEY,
            category="accel", runner="accel_runner",
        ),
    ]


def _rscx_env_prefix() -> Path:
    """The conda env that owns rscx (same resolution as the doublet R runner)."""
    override = os.environ.get("SCX_RSCX_ENV_PREFIX", "")
    if override:
        return Path(override)
    exe = os.environ.get("CONDA_EXE", "")
    root = Path(exe).parent.parent if exe else Path.home() / "miniforge3"
    return root / "envs" / "rscx"


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    if format_variant.key != FORMAT_KEY:
        return None

    prefix = _rscx_env_prefix()
    rscript = prefix / "bin" / "Rscript"
    if not rscript.exists():
        write_missing_result(
            benchmark="accel_r_route", format_key=format_variant.key,
            dataset=dataset.name, missing_reason="no_rscx_env",
            notes=f"Rscript not found at {rscript}; install the rscx conda env "
                  "(R CMD INSTALL rscx/) or set SCX_RSCX_ENV_PREFIX",
        )
        return None

    # Mirror `bpcells_runner._call_r`: the orchestrator schedules this in
    # whatever env `_env_for_format` picked (scx-bench), so prepend the rscx
    # env's bin/lib to keep the right libR/libstdc++ bound to its Rscript, and
    # run --vanilla so a user .Rprofile cannot pollute stdout or the session.
    env = os.environ.copy()
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(rscript), "--vanilla", str(_PROBE)],
        capture_output=True, text=True, timeout=600, env=env,
    )
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        # A broken/uninstalled *rscx* (library(rscx) fails) is a missing env,
        # not a red gate. Match rscx by name only: any other package's load
        # failure — or a broken rscx.so — must raise, not skip, or the floors
        # silently vacate (review round 1, Cursor Agent).
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        stderr = proc.stderr or ""
        if (
            "there is no package called ‘rscx’" in stderr
            or "there is no package called 'rscx'" in stderr
        ):
            write_missing_result(
                benchmark="accel_r_route", format_key=format_variant.key,
                dataset=dataset.name, missing_reason="no_rscx_env",
                notes="library(rscx) failed in the rscx env: " + " | ".join(tail),
            )
            return None
        raise RuntimeError(
            f"r_route_probe.R failed (rc={proc.returncode}): " + " | ".join(tail)
        )

    # The probe prints one JSON object as its last stdout line; parse the
    # final {...} so R warnings above it cannot corrupt the read (same shape
    # as bpcells_runner).
    stdout = proc.stdout.strip()
    brace_start = stdout.rfind("{")
    if brace_start == -1:
        raise RuntimeError(
            f"r_route_probe.R produced no JSON output.\n--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    probe = json.loads(stdout[brace_start:])

    pca_ok = (
        probe.get("pca_route") == "cpu_csr"
        and probe.get("pca_fallback") == "user_forced_cpu"
        and probe.get("pca_record_keys") == WIRE_KEYS
    )
    arms_ok = (
        probe.get("pca_method_small") == "covariance"
        and probe.get("pca_method_big") == "randomized"
    )
    wilcoxon_ok = (
        probe.get("wilcoxon_route") == "cpu_dense"
        and probe.get("wilcoxon_fallback") == "user_forced_cpu"
    )
    dex_ok = (
        probe.get("dex_route") == "cpu_nb_glm"
        and probe.get("dex_fallback") == "none"
    )
    harmony_ok = (
        probe.get("harmony_route") == "cpu_dense"
        and probe.get("harmony_fallback") == "user_forced_cpu"
    )

    result = BenchmarkResult(
        benchmark="accel_r_route",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "rscx_env_prefix": str(prefix),
            "probe": probe,
            "synthetic_probe": True,
        },
    )
    result.add_run(
        wall_s=wall_s,
        r_pca_route_cpu_correct=1.0 if pca_ok else 0.0,
        r_pca_method_arms_correct=1.0 if arms_ok else 0.0,
        r_wilcoxon_route_cpu_correct=1.0 if wilcoxon_ok else 0.0,
        r_pseudobulk_dex_route_correct=1.0 if dex_ok else 0.0,
        r_harmony_route_cpu_correct=1.0 if harmony_ok else 0.0,
    )
    return result
