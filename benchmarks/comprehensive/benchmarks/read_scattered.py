"""
Scattered-read latency benchmark (F5 row-group block-index path).

Unlike ``read_selective`` — which SORTS its indices for locality-friendly
access and never exercises the row-group path — this benchmark drives a
``pyscx.IndexPlanDataset`` with **unsorted, uniformly-random** row plans over a
**row-group-framed** SCX file (the ``scx_compact_trial_g<G>`` variants). Each
gather touches only a tiny, scattered fraction of each shard, so a framed file
decodes just the touched row-groups via the codec-agnostic block index instead
of decoding whole shards.

The dataset is opened with ``scatter_sidecar=False`` + ``scatter_block_index=True``
so the block-index path is isolated from the bit-level Scx1 sidecar path: on a
framed file every scattered group is served by the block index
(``block_index_groups > 0``); on an unframed file it falls back to a full-shard
decode (``full_shard_groups``). This is a direct end-to-end test of the F5
loader-adoption work.

SCX-framed-only: ``run`` returns ``None`` for any format outside
``SUPPORTED_FORMATS`` so the orchestrator silently skips it (the
``index_plan`` / ``fragment_ops`` gating pattern).

Metrics emitted into ``RunRecord.extra`` (median across runs is what the gate
reads):
    ``gather_latency_ms_p50`` / ``gather_latency_ms_p99`` — steady-state
        per-batch scattered-gather latency (first batch dropped: it includes
        prefetch fill + lazy tokio-runtime spin-up).
    ``block_index_groups`` — cumulative count of shard request-groups served
        by the row-group block index. ``> 0`` proves the framed file adopted
        the path (the adoption floor).
    ``full_shard_groups`` / ``sidecar_groups`` — the other two dispatch paths
        (``sidecar_groups`` is ~0 here since the sidecar path is disabled).
    ``block_index_adoption_rate`` — ``block_index / (block_index +
        full_shard)``; the primary success signal (1.0 = every scattered group
        took the block index).
"""

from __future__ import annotations

import gc
import logging
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


# Framed compact-trial variants only. `scx_auto` is monolithic/unframed
# (block_index_groups would always be 0), so it is deliberately excluded — the
# adoption floor requires a framed fixture. Read by `run_parallel.py`'s cohort
# builder and re-checked at the top of `run()` (defense-in-depth).
SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {
        "scx_compact_trial_g128",
        "scx_compact_trial_g256",
        "scx_compact_trial_g512",
        "scx_compact_trial_g1024",
    }
)

# Scattered plans: small per-batch pair count so each shard is touched sparsely
# (the block-index win regime — a handful of rows out of a ~16k-row shard).
_DEFAULT_PAIRS_PER_BATCH = 256
# Batch/run counts are scaled DOWN for large datasets: with the sidecar off +
# prefetch skipped, every batch re-decodes the touched row-groups (no shard cache
# reuse), and on a broadly-scattered plan each of the ~512 rows lands in a
# distinct row-group, so a single batch can decode ~512 groups of G rows —
# measured at 2–3.5 s/batch on smartseq2/tabula (a deliberate worst-case; real
# training uses locality + caching). A fixed 200×5 batches×runs blows the
# tier-small 240 s job budget. The knobs below keep the per-cell wall well under
# it while still proving adoption (`block_index_groups`) and giving a rough p50.
_MAX_N_BATCHES = 50
_MIN_N_BATCHES = 12
# Datasets at/above this get the reduced (2) timed-run count.
_LARGE_N_OBS = 30_000
_RANDOM_SEED = 42


def _n_batches_for(n_obs: int) -> int:
    """Scale batch count inversely with dataset size (bounded decode work)."""
    scaled = 800_000 // max(1, n_obs)
    return int(min(_MAX_N_BATCHES, max(_MIN_N_BATCHES, scaled)))


def _n_runs_for(n_obs: int, harness_n_runs: int) -> int:
    """Cap timed runs so the per-cell wall fits the tier-small 240 s budget:
    2 passes on large (≥30k-cell, multi-shard, slow-per-batch) datasets, else
    up to 3. This is a latency + adoption benchmark; a stable median doesn't
    need the harness's default 5."""
    cap = 2 if n_obs >= _LARGE_N_OBS else 3
    return max(1, min(harness_n_runs, cap))


def _have_pyscx() -> bool:
    try:
        import pyscx  # noqa: F401

        return True
    except ImportError:
        return False


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _random_plans(
    n_obs: int, pairs_per_batch: int, n_batches: int, seed: int = _RANDOM_SEED
):
    """Yield ``n_batches`` plans of ``pairs_per_batch`` (pert, ctrl) row pairs,
    each drawn uniformly at random over ``[0, n_obs)`` — deliberately unsorted
    and non-local so the gather is scattered across shards."""
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        pert = rng.integers(0, n_obs, size=pairs_per_batch).astype(np.int64)
        ctrl = rng.integers(0, n_obs, size=pairs_per_batch).astype(np.int64)
        yield list(zip(map(int, pert), map(int, ctrl)))


def _resolve_framed_path(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    converted_path: Path | None,
) -> Path:
    """Return the on-disk framed SCX file for this (dataset, format).

    Prefers an orchestrator-supplied ``converted_path``; otherwise uses the
    persistent per-format path and converts (once, cached) via the format's
    runner when it is missing — so the local ``run_all.py`` smoke works without
    a separate conversion phase, while the SLURM gate's pre-converted file is
    reused as-is.
    """
    if converted_path is not None and Path(converted_path).exists():
        return Path(converted_path)
    persistent = dataset.path_for_format(format_variant.key)
    if not persistent.exists():
        if not dataset.h5ad_path.exists():
            raise FileNotFoundError(
                f"Source h5ad not found for {dataset.name}: {dataset.h5ad_path}"
            )
        logger.info(
            "read_scattered: converting %s → %s (%s)",
            dataset.name,
            persistent.name,
            format_variant.key,
        )
        persistent.parent.mkdir(parents=True, exist_ok=True)
        make_runner(format_variant).convert_from_h5ad(dataset.h5ad_path, persistent)
    return persistent


def _one_run(scx_path: str, n_obs: int, pairs_per_batch: int, n_batches: int) -> dict[str, Any]:
    """Drive one scattered-gather pass; return latency + adoption metrics."""
    import pyscx

    gc.collect()

    # scatter_sidecar=False + scatter_block_index=True isolates the block-index
    # path: framed shards route through the block index, unframed shards fall
    # back to full-shard decode. Proves the F5 loader-adoption path directly.
    ds = pyscx.IndexPlanDataset(
        scx_path,
        normalize=False,
        cache_shards=128,
        sort_by_shard=True,
        lookahead=4,
        max_plan_size=max(2 * pairs_per_batch, 16384),
        max_memory_mb=8192,
        scatter_sidecar=False,
        scatter_block_index=True,
    )

    per_batch_s: list[float] = []
    t0 = time.perf_counter()
    prev = t0
    seen = 0
    for _batch in ds.iter_with_plans(
        _random_plans(n_obs, pairs_per_batch, n_batches), lookahead=4
    ):
        now = time.perf_counter()
        per_batch_s.append(now - prev)
        prev = now
        seen += 1
    wall = time.perf_counter() - t0

    try:
        cm = ds.cache_metrics()
        sidecar_groups = int(cm.get("sidecar_groups", 0))
        full_shard_groups = int(cm.get("full_shard_groups", 0))
        block_index_groups = int(cm.get("block_index_groups", 0))
    except Exception:
        sidecar_groups = full_shard_groups = block_index_groups = None

    # Steady-state latency: drop the first interval (prefetch fill + lazy
    # tokio-runtime spin-up, not steady-state gather).
    steady = per_batch_s[1:] if len(per_batch_s) > 1 else per_batch_s
    if steady:
        ms = sorted(v * 1000.0 for v in steady)
        p50_ms = ms[len(ms) // 2]
        p99_ms = ms[min(len(ms) - 1, int(len(ms) * 0.99))]
    else:
        p50_ms = p99_ms = None

    if (
        block_index_groups is None
        or full_shard_groups is None
        or (block_index_groups + full_shard_groups) == 0
    ):
        adoption_rate = None
    else:
        adoption_rate = round(
            block_index_groups / (block_index_groups + full_shard_groups), 4
        )

    return {
        "wall_s": wall,
        "n_batches": seen,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "gather_latency_ms_p50": round(p50_ms, 3) if p50_ms is not None else None,
        "gather_latency_ms_p99": round(p99_ms, 3) if p99_ms is not None else None,
        "block_index_groups": block_index_groups,
        "full_shard_groups": full_shard_groups,
        "sidecar_groups": sidecar_groups,
        "block_index_adoption_rate": adoption_rate,
    }


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Scattered-read latency + block-index adoption over a framed SCX file.

    Returns ``None`` for any non-framed format (gating pattern).
    """
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    if not _have_pyscx():
        logger.warning("Skipping read_scattered: pyscx not importable")
        return None

    framed_path = _resolve_framed_path(dataset, format_variant, converted_path)
    scx_path = str(framed_path)

    n_obs = dataset.n_obs
    pairs_per_batch = min(_DEFAULT_PAIRS_PER_BATCH, max(1, n_obs // 4))
    n_batches = _n_batches_for(n_obs)
    n_runs = _n_runs_for(n_obs, n_runs)

    result = BenchmarkResult(
        benchmark="read_scattered",
        format=format_variant.key,
        dataset=dataset.name,
        file_size_bytes=Path(scx_path).stat().st_size,
        metadata={
            "pairs_per_batch": pairs_per_batch,
            "n_batches_target": n_batches,
            "row_group_rows": format_variant.params.get("row_group_rows"),
            "scatter_sidecar": False,
            "scatter_block_index": True,
            "n_runs": n_runs,
        },
    )

    # One short untimed warm-up, then `n_runs` recorded passes. Each pass rebuilds
    # the dataset so counters + cache start clean and per-run medians are
    # comparable. Warm-up is capped small — it only needs to spin up the tokio
    # runtime + page-cache the file, not produce measurements.
    _one_run(scx_path, n_obs, pairs_per_batch, min(n_batches, 3))
    for i in range(max(1, n_runs)):
        try:
            m = _one_run(scx_path, n_obs, pairs_per_batch, n_batches)
        except Exception as e:
            logger.error("read_scattered run %d/%d failed: %s", i + 1, n_runs, e)
            continue
        result.add_run(
            wall_s=m["wall_s"],
            peak_rss_mb=m["peak_rss_mb"],
            gather_latency_ms_p50=m["gather_latency_ms_p50"],
            gather_latency_ms_p99=m["gather_latency_ms_p99"],
            block_index_groups=m["block_index_groups"],
            full_shard_groups=m["full_shard_groups"],
            sidecar_groups=m["sidecar_groups"],
            block_index_adoption_rate=m["block_index_adoption_rate"],
            n_batches=m["n_batches"],
        )

    return result
