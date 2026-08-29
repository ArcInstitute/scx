#!/usr/bin/env python3
"""
SCX Comprehensive Benchmark Suite — Parallel SLURM Launcher via submitit.

Two-phase execution:
  Phase A: Convert each (dataset, format) pair to a persistent path.
  Phase B: Run benchmarks in parallel, reading from pre-converted files.

Each (benchmark, dataset, format) triple is submitted as an independent SLURM
job, enabling massive parallelism across the cluster.

Phase A and Phase B overlap: as soon as a conversion job is submitted,
its dependent benchmark jobs are submitted with a SLURM
``--dependency=afterok:<jid>`` directive. SLURM holds them in PENDING until
the conversion lands, then releases them — so a fast conversion (pbmc3k)
unblocks its benchmarks long before a slow conversion (census_5m, h5ad_lzf)
finishes. Benchmarks whose target file already exists, and ones in
``_NO_CONVERSION``, submit with no dependency at all.

Usage:
    # Full run on small datasets
    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k

    # Large datasets on high-mem partition
    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --datasets census_500k census_1m census_5m \\
        --partition cpu_preemptible --mem-gb 500 --timeout 480

    # Specific benchmarks and formats
    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --benchmarks read_full read_selective \\
        --formats scx_auto zarr_zstd h5ad_gzip \\
        --datasets census_1m

    # Dry run
    python benchmarks/comprehensive/scripts/run_parallel.py --dry-run

    # Skip conversion (use existing pre-converted files)
    python benchmarks/comprehensive/scripts/run_parallel.py --skip-convert

    # Re-convert everything
    python benchmarks/comprehensive/scripts/run_parallel.py --overwrite
"""

from __future__ import annotations

import argparse
import functools
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    ALL_FORMATS,
    DATASETS,
    PRIMARY_FORMATS,
    FormatVariant,
    estimate_memory_gb,
    n_runs_for_dataset,
    partition_for_memory,
)
from benchmarks.comprehensive.convert import convert_dataset_format  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult, write_result  # noqa: E402
from benchmarks.comprehensive.runners import make_runner  # noqa: E402

logger = logging.getLogger(__name__)

from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS  # noqa: E402

# Canonical benchmark list lives in benchmarks/__init__.py::ALL_BENCHMARKS —
# add new benchmarks there, not here. This alias preserves the historical
# name so existing callers / subagents don't need to change.
BENCHMARK_NAMES = list(ALL_BENCHMARKS)

# Benchmarks that work from h5ad source and don't need pre-converted files.
# cell_eval_parity_perf and accel_de_nb_glm generate synthetic data in-process
# (no source .h5ad on disk for pert_synth_* / nb_glm_synth), so Phase A
# conversion would fail — skip it for them.
_NO_CONVERSION = {
    "write", "parallel_write_scaling", "cell_eval_parity_perf", "accel_de_nb_glm",
    "accel_eval_metrics",
    # grouped_sort makes its own plain .scx in-process (and synthetic datasets
    # self-materialize via _pert_synth) — it must not depend on a Phase-A
    # conversion, which would fail for synthetic datasets that have no on-disk
    # h5ad to convert.
    "grouped_sort",
    # grouped_read builds its own grouped .scx / .shad files in-process from
    # the source h5ad (same as grouped_sort) — no Phase-A dependency.
    "grouped_read",
    # shardad_fidelity self-materializes the source + writes a temp .shad
    # in-process (synthetic datasets have no on-disk h5ad) — no Phase-A dep.
    "shardad_fidelity",
    # shuffle_layout consumes the persistent per-codec `.scx` fixtures directly
    # and writes its own shuffled copies in-process; it never needs a Phase-A
    # h5ad conversion.
    "shuffle_layout",
    # conversion_streaming IS the conversion: both its arms read
    # `dataset.h5ad_path` and write their own SCX output to a temp dir, and it
    # ignores `converted_path` entirely.
    #
    # Its inverse, `export_streaming`, is deliberately NOT here — it consumes a
    # pre-converted SCX file, and its own fallback (`dataset.scx_path`, i.e.
    # `<name>.scx`) does not exist on disk for the census fixtures, which are
    # built as `<name>_auto.scx`. Without Phase A it would re-ingest the source
    # h5ad inline on every run.
    "conversion_streaming",
}

# Benchmarks that operate on multimodal h5mu sources. They expect
# `dataset.multimodal=True` and a multimodal-aware format runner; pairing
# them with single-modality datasets / formats is wasted scheduling.
_MULTIMODAL_BENCHMARKS = {
    "multimodal_compression",
    "multimodal_training",
    "multimodal_read_streaming_vs_inmemory",
}

# Format keys that consume `.h5mu` (or write the multimodal SCX layout).
# Non-multimodal benchmarks can't read these; multimodal benchmarks can't
# read non-multimodal formats. Defined inline (not imported from config)
# so this list stays a single source of truth for the orchestrator.
_MULTIMODAL_FORMAT_PREFIXES = ("h5mu_", "zarr_mudata_", "scx_multimodal_")


def _is_multimodal_format(format_key: str) -> bool:
    return any(format_key.startswith(p) for p in _MULTIMODAL_FORMAT_PREFIXES)


def _count_active_user_jobs() -> int:
    """Return total PD+R jobs currently queued for the invoking user.

    Used by ``_wait_under_pending_cap`` to throttle submissions when
    Chimera's ``QOSMaxSubmitJobPerUserLimit`` (~500 per user) is
    approaching. Falls back to 0 when ``squeue`` is unreachable
    (running outside SLURM, or a misconfigured PATH) — disabling the
    throttle is safer than blocking the orchestrator.
    """
    import os
    import subprocess
    user = os.environ.get("USER", "")
    if not user:
        return 0
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-t", "PD,R", "--format=%i"],
            text=True, timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


# Cache for sacct state lookups in the wait loop. Populated on first
# fallback to sacct; subsequent reads of the same job_id are free.
_SACCT_STATE_CACHE: dict[str, str] = {}


def _sacct_head_state(job_id: str) -> str:
    """Return the normalised head-state token for a SLURM job_id via sacct.

    Used as a fallback when ``submitit.Job.state`` returns empty —
    that happens after a job has been reaped from squeue but before
    ``.result()`` has been called. Without this fallback, the wait
    loop falls through to ``.result()`` on a job whose result pickle
    was never written (e.g. a TIMEOUT before the worker flushed
    output), and submitit blocks indefinitely.

    Caches the first non-empty result so a re-poll of the same id
    is free. Empty return = no sacct row or unreachable slurm.
    """
    cached = _SACCT_STATE_CACHE.get(job_id)
    if cached is not None:
        return cached
    import subprocess
    try:
        out = subprocess.check_output(
            ["sacct", "-X", "-j", job_id, "-P", "--noheader", "--format=State"],
            text=True, timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""
    line = next((l.strip() for l in out.splitlines() if l.strip()), "")
    head = line.replace("+", " ").split()[0].upper() if line else ""
    if head:
        _SACCT_STATE_CACHE[job_id] = head
    return head


def _job_terminal_head(job: Any) -> str:
    """Return the head-state token for a submitit job, with sacct fallback.

    ``job.state`` returns the live SLURM state from squeue when the
    job is still tracked; once reaped (a few minutes after a terminal
    transition), it returns empty. Calling ``.result()`` on a
    reaped-but-incomplete job hangs because submitit cannot determine
    whether the missing result pickle means "not yet" or "never".
    Falling back to ``sacct`` resolves the historical state.

    Returns the first whitespace-/+-delimited token uppercased, e.g.
    "TIMEOUT", "FAILED", "COMPLETED", or "" when slurm has no record.
    """
    try:
        raw_state = job.state or ""
    except Exception:
        raw_state = ""
    head = raw_state.replace("+", " ").split()[0].upper() if raw_state else ""
    if not head:
        job_id = getattr(job, "job_id", None)
        if job_id:
            head = _sacct_head_state(str(job_id))
    return head


def _cancel_dep_never_satisfied() -> int:
    """Scancel jobs stuck in ``DependencyNeverSatisfied`` and return
    the count cancelled.

    Required for the throttle to make progress when an upstream
    convert fails AFTER its dependent bench was already submitted —
    those bench jobs sit in the queue forever (counting toward the
    QOS cap) and would otherwise pin the throttle indefinitely.
    Best-effort: silently ignores ``squeue`` / ``scancel`` errors so
    a transient slurm hiccup doesn't crash the orchestrator.
    """
    import os
    import subprocess
    user = os.environ.get("USER", "")
    if not user:
        return 0
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-t", "PD", "--format=%i %r"],
            text=True, timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return 0
    ids = [
        line.split()[0]
        for line in out.splitlines()
        if "DependencyNeverSatisfied" in line
    ]
    if not ids:
        return 0
    try:
        subprocess.run(
            ["scancel", *ids],
            timeout=30, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return 0
    return len(ids)


# Module-level state for `_wait_under_pending_cap`: tracks whether we
# logged an active throttling episode so the per-submit calls don't
# each emit their own "Throttling…/Resumed…" pair. The submit loops
# fire ``_wait_under_pending_cap`` once per job — without dedup, every
# near-cap step repeats the message, flooding the log when 1000+ jobs
# cycle through the cap. A short cooldown after "Resumed" suppresses
# spurious re-throttle emission when the queue oscillates around the
# cap (sub-second; see _THROTTLE_RESUME_COOLDOWN_S below).
_THROTTLE_STATE: dict[str, float] = {
    "in_throttle": 0.0,        # 0.0 = not in throttle, else timestamp of throttle entry
    "last_resumed_at": 0.0,    # monotonic timestamp of most recent "Resumed" log
}
_THROTTLE_RESUME_COOLDOWN_S = 60.0  # don't re-log "Throttling" within 60s of resume


def _wait_under_pending_cap(cap: int, *, kind: str, headroom: int = 0) -> None:
    """Block until ``squeue`` has room for ``headroom`` more PD+R jobs.

    Blocks until the active (PD+R) job count drops below ``cap - headroom``,
    i.e. until there is room to submit ``headroom`` more jobs and still stay
    under ``cap``. With the default ``headroom=0`` this is the original
    "block until ``active < cap``" behaviour. ``headroom`` is used by the
    cohort-array submitter: a ``map_array`` call registers N tasks that each
    count individually against ``QOSMaxSubmitJobPerUserLimit``, so the
    throttle must reserve room for the whole cohort before submitting.

    The effective threshold is clamped to ``>= 1`` so a cohort larger than
    ``cap`` waits for the queue to fully drain and then submits anyway
    (unavoidable overshoot, absorbed by the QOS retry loop and
    ``slurm_array_parallelism``) rather than spinning forever.

    No-op when ``cap <= 0`` (throttle disabled). Polls every 30s.
    Auto-cancels jobs stuck in ``DependencyNeverSatisfied`` (zombies
    from upstream-convert failures that landed AFTER the bench
    submitted) so the queue can drain — without this, a single failed
    slaf / parquet / bpcells convert can pin hundreds of dependents
    and stall the orchestrator for hours. ``kind`` is a label
    ("convert" or "bench") embedded in the wait message.

    Logging contract: at most one "Throttling…" line per *episode*
    (consecutive calls returning before submitting a new job do not
    re-log), and one "Resumed…" line when the episode actually ends.
    Steady-state cycles (queue oscillates around cap as jobs land)
    therefore produce a single throttle entry per logical pause, not
    one per submit.
    """
    import time as _time
    if cap <= 0:
        return

    # Reserve room for the whole cohort: block until active < cap - headroom.
    # Clamp to >= 1 so an oversized cohort waits for a full drain instead of
    # looping forever on an unsatisfiable threshold.
    threshold = max(cap - headroom, 1)

    now = _time.monotonic()

    # Fast path: queue is below threshold. Emit "Resumed" once when we exit
    # a logged throttling episode; subsequent calls stay silent. The
    # ``in_throttle`` slot is a timestamp (0.0 when not throttling) so
    # a future log-line could surface elapsed-throttle duration; the
    # boolean check is value-truthiness on the timestamp.
    initial_active = _count_active_user_jobs()
    if initial_active < threshold:
        if _THROTTLE_STATE["in_throttle"]:
            logger.info(
                "  Resumed %s submission (active=%d < threshold=%d, cap=%d)",
                kind, initial_active, threshold, cap,
            )
            _THROTTLE_STATE["in_throttle"] = 0.0
            _THROTTLE_STATE["last_resumed_at"] = now
        return

    # At-cap. Decide whether this is a fresh throttle episode worth
    # announcing or a near-immediate re-hit after a resume (queue
    # oscillating around the cap during a submission burst). Re-hits
    # within `_THROTTLE_RESUME_COOLDOWN_S` of the last resume are
    # rolled into the same logical episode and stay silent.
    is_fresh_episode = (
        not _THROTTLE_STATE["in_throttle"]
        and (now - _THROTTLE_STATE["last_resumed_at"]) > _THROTTLE_RESUME_COOLDOWN_S
    )
    if is_fresh_episode:
        logger.info(
            "  Throttling %s submission: active=%d >= threshold=%d "
            "(cap=%d, headroom=%d); polling squeue every 30s until "
            "queue drops below threshold.",
            kind, initial_active, threshold, cap, headroom,
        )
        _THROTTLE_STATE["in_throttle"] = now
    elif not _THROTTLE_STATE["in_throttle"]:
        # Recent resume — suppress the log but still mark in-throttle
        # so the matching "Resumed" line emits when we drain.
        _THROTTLE_STATE["in_throttle"] = now

    while True:
        active = _count_active_user_jobs()
        if active < threshold:
            # Don't log "Resumed" here — the next call into
            # `_wait_under_pending_cap` will emit it via the fast
            # path above. This keeps the resume message attached to
            # the log line just before the next submission resumes,
            # rather than appearing 30 s before any visible activity.
            return
        # At-cap: try to drain zombie deps before sleeping.
        cancelled = _cancel_dep_never_satisfied()
        if cancelled:
            logger.info(
                "  Cancelled %d DependencyNeverSatisfied job(s) to free "
                "throttle slots; recheck immediately.",
                cancelled,
            )
            continue
        _time.sleep(30)


def _triple_compatible(bench_name: str, ds_name: str, format_key: str) -> bool:
    """True iff the (benchmark, dataset, format) triple is meaningful.

    Filters at the launcher level so submitit doesn't spawn Phase A
    convert jobs for incompatible pairings (e.g. accel_pca on a
    multimodal dataset, or compression on the multimodal SCX layout).
    Mirrors the inline accel / bench_csc rules in the Phase B loop;
    factored out here because both Phase A (convert) and Phase B
    (benchmark) need the same compatibility view.
    """
    ds = DATASETS.get(ds_name)
    ds_is_multimodal = bool(ds and ds.multimodal)
    fmt_is_multimodal = _is_multimodal_format(format_key)
    bench_is_multimodal = bench_name in _MULTIMODAL_BENCHMARKS

    # Multimodal datasets only pair with multimodal benchmarks +
    # multimodal formats. Non-multimodal datasets never see multimodal
    # benchmarks or formats.
    if ds_is_multimodal != bench_is_multimodal:
        return False
    if ds_is_multimodal != fmt_is_multimodal:
        return False
    if bench_is_multimodal != fmt_is_multimodal:
        return False

    # A format that declares a dataset scope only pairs inside it. Filtering
    # here rather than stubbing inside `run()` is what stops the job being
    # submitted at all.
    scope = _bench_format_dataset_scope(bench_name).get(format_key)
    if scope is not None and ds_name not in scope:
        return False
    return True


@functools.lru_cache(maxsize=None)
def _bench_format_dataset_scope(bench_name: str) -> dict[str, frozenset[str]]:
    """Per-format dataset allow-lists a bench module declares, or `{}`.

    Reads the module-level ``FORMAT_DATASET_SCOPE`` mapping, in the same shape
    as :func:`_bench_supported_formats` — the constraint is owned by the bench,
    not duplicated in the orchestrator.

    This exists because an **arm** variant is not meaningful on every tier.
    `accel_pca__pyscx_gpu_streaming` forces the slow power loop
    (`SCX_GPU_PCA_RESIDENT=0`, 3.2x the resident arm on tabula) and is floored
    only on pbmc3k + tabula_sapiens_100k; without a scope the orchestrator
    scheduled it on census_1m too — a different, ungated, much slower
    experiment. The shufdelta arms had the same gap in the other direction:
    they stubbed out-of-scope datasets *inside* `run()`, which is a typed
    result but only after Chimera has started a preemptible GPU task, activated
    conda and imported pyscx — eight wasted GPU jobs per full accel capture.
    Both found by **Cursor Agent** on PR #474.
    """
    try:
        mod = importlib.import_module(
            f"benchmarks.comprehensive.benchmarks.{bench_name}"
        )
    except ImportError:
        return {}
    val = getattr(mod, "FORMAT_DATASET_SCOPE", None)
    if not val:
        return {}
    return {k: frozenset(v) for k, v in val.items()}


@functools.lru_cache(maxsize=None)
def _bench_supported_formats(bench_name: str) -> frozenset[str] | None:
    """Return the bench module's declared format allow-list, or None for "all".

    Reads the module-level ``SUPPORTED_FORMATS`` attribute that each
    static-guarded bench exposes (e.g. ``cloud_push``, ``correctness``,
    ``roundtrip``). Benches without the attribute return ``None`` —
    "no per-bench restriction beyond the accel/CSC pairing rules". Cached
    so the import cost is paid once per bench name across the whole
    cohort-grouping pass.
    """
    try:
        mod = importlib.import_module(
            f"benchmarks.comprehensive.benchmarks.{bench_name}"
        )
    except ImportError:
        return None
    val = getattr(mod, "SUPPORTED_FORMATS", None)
    return val if val is None else frozenset(val)


@functools.lru_cache(maxsize=None)
def _bench_required_capabilities(bench_name: str) -> frozenset[str]:
    """Return the bench module's declared ``REQUIRED_CAPABILITIES``, or empty.

    Mirrors :func:`_bench_supported_formats` — bench modules expose
    runner-capability requirements next to their runtime guards so the
    constraint is owned by the bench, not duplicated in the orchestrator.
    Cached for the same reason.
    """
    try:
        mod = importlib.import_module(
            f"benchmarks.comprehensive.benchmarks.{bench_name}"
        )
    except ImportError:
        return frozenset()
    val = getattr(mod, "REQUIRED_CAPABILITIES", None)
    return frozenset(val) if val else frozenset()


@functools.lru_cache(maxsize=None)
def _runner_capabilities(format_key: str) -> frozenset[str]:
    """Resolve the runner's capability set for ``format_key``.

    Reads ``runner.capabilities`` after a cheap, side-effect-free
    ``make_runner(fmt)`` instantiation — no file I/O, no cloud calls.
    Cached so each format key pays one instantiation across the whole
    cohort-grouping pass. Returns an empty frozenset if the format is
    unknown or the runner constructor raises (defensive — old YAML
    configs in CI may name retired formats).
    """
    fmt = next((f for f in ALL_FORMATS if f.key == format_key), None)
    if fmt is None:
        return frozenset()
    try:
        return frozenset(make_runner(fmt).capabilities)
    except Exception:
        return frozenset()


def _bench_format_compatible(bench_name: str, format_key: str) -> bool:
    """Check benchmark–format pairing beyond multimodal compatibility.

    Three layered checks:
    1. Accel benchmarks (accel_*) only pair with their own format variants
       (e.g. accel_pca pairs with accel_pca__scx_auto). Similarly,
       bench_csc_dispatch only pairs with bench_csc__* formats. Non-accel
       benchmarks skip accel_* and bench_csc__* format keys.
    2. Static-guarded benches (e.g. ``cloud_push``, ``correctness``)
       expose a module-level ``SUPPORTED_FORMATS`` frozenset and only
       accept format keys in that set.
    3. Capability-guarded benches (e.g. ``cloud_read``, ``ml_loader``)
       expose ``REQUIRED_CAPABILITIES`` and only accept formats whose
       runner declares all of those capabilities.

    Filtering at cohort-build time prevents the silent ``return None``
    path in each bench from producing phantom ``missing_result`` entries
    in ``watch.py``.
    """
    is_accel = bench_name.startswith("accel_") or bench_name == "bench_csc_dispatch"
    fmt_is_accel = format_key.startswith("accel_") or format_key.startswith("bench_csc__")

    if is_accel:
        if bench_name == "bench_csc_dispatch":
            return format_key.startswith("bench_csc__")
        return format_key.startswith(f"{bench_name}__")
    elif fmt_is_accel:
        return False

    allowed = _bench_supported_formats(bench_name)
    if allowed is not None and format_key not in allowed:
        return False

    required = _bench_required_capabilities(bench_name)
    if required and not required.issubset(_runner_capabilities(format_key)):
        return False
    return True


def _build_cohort_key(
    args, ds_name: str, fmt_key: str, bench_name: str
) -> tuple[str, str, str, str, bool]:
    """Build a resource-cohort key for array grouping.

    Returns (dataset, format_key, partition, gres, needs_conversion).
    All benchmarks sharing the same key can safely coexist in one SLURM array.

    Note: conda_env and slurm_setup are pure functions of format_key
    (via _env_for_format / _slurm_setup_cmds), so they are fully
    determined by the key without being explicitly included.
    """
    params = _per_job_slurm_params(
        args, ds_name, fmt_key, bench_name, is_conversion=False
    )
    needs_conv = bench_name not in _NO_CONVERSION
    return (
        ds_name,
        fmt_key,
        params["slurm_partition"],
        params.get("slurm_gres", ""),
        needs_conv,
    )


def _determine_cohort_resources(
    args, cohort_key: tuple, benchmarks: list[str]
) -> dict:
    """Compute the SLURM resource envelope for a cohort's array.

    Since all benchmarks in the cohort already share the same partition,
    gres, and conda env (by construction), we only need max(mem, time).
    Returns a dict suitable for passing to executor.update_parameters().
    """
    ds_name, fmt_key, partition, gres, needs_conv = cohort_key
    max_mem = 0
    max_time = 0

    for bench in benchmarks:
        params = _per_job_slurm_params(
            args, ds_name, fmt_key, bench, is_conversion=False
        )
        max_mem = max(max_mem, params["mem_gb"])
        max_time = max(max_time, params["timeout_min"])

    return {
        "mem_gb": max_mem,
        "timeout_min": max_time,
        "slurm_partition": partition,
        "cpus_per_task": args.cpus,
        # Explicitly set gres to prevent carry-over from prior
        # executor.update_parameters() calls (see §5 note on merge semantics).
        "slurm_gres": gres if gres else "",
        # slurm_setup is a pure function of format_key via _env_for_format
        "slurm_setup": _slurm_setup_cmds(_env_for_format(fmt_key), fmt_key),
    }


_TERMINAL_FAILURE_STATES = {
    "CANCELLED", "TIMEOUT", "NODE_FAIL", "FAILED", "OUT_OF_MEMORY", "PREEMPTED",
}


LOGS_DIR = PROJECT_ROOT / "benchmarks" / "comprehensive" / "logs" / "submitit"


# ---------------------------------------------------------------------------
# Submitit job callables (must be picklable — defined at module level)
# ---------------------------------------------------------------------------


def _run_conversion(
    dataset_name: str,
    format_key: str,
    format_runner: str,
    format_params: dict,
    overwrite: bool,
) -> str:
    """Convert a single (dataset, format) pair. Returns the output path."""
    from benchmarks.comprehensive.config import DATASETS, FormatVariant
    from benchmarks.comprehensive.convert import convert_dataset_format

    dataset = DATASETS[dataset_name]
    fmt = FormatVariant(
        name=format_key, key=format_key,
        category="primary", runner=format_runner, params=format_params,
    )
    path = convert_dataset_format(dataset, fmt, overwrite=overwrite)
    return str(path)


def _run_benchmark(
    bench_name: str,
    dataset_name: str,
    format_key: str,
    format_runner: str,
    format_params: dict,
    n_runs: int,
    cold_cache: bool,
    converted_path_str: str | None,
) -> dict:
    """Run a single (benchmark, dataset, format) triple. Returns result dict."""
    import importlib
    from pathlib import Path

    from benchmarks.comprehensive.config import DATASETS, FormatVariant
    from benchmarks.comprehensive.results import write_result

    dataset = DATASETS[dataset_name]
    fmt = FormatVariant(
        name=format_key, key=format_key,
        category="primary", runner=format_runner, params=format_params,
    )
    converted_path = Path(converted_path_str) if converted_path_str else None

    mod = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{bench_name}")

    # write.py and parallel_write_scaling.py don't accept converted_path
    if bench_name in _NO_CONVERSION:
        result = mod.run(dataset=dataset, format_variant=fmt,
                         n_runs=n_runs, cold_cache=cold_cache)
    else:
        result = mod.run(dataset=dataset, format_variant=fmt,
                         n_runs=n_runs, cold_cache=cold_cache,
                         converted_path=converted_path)

    # Benchmarks may return None when the (bench, format) combo is a
    # deliberate skip (e.g. fragment_ops / cloud_push / cloud_pull only
    # apply to SCX; capability-gated cross-format modules return None for
    # runners that don't declare the capability). That's a successful
    # no-op, not a failure — don't try to persist it.
    if result is None:
        return {"skipped": True, "benchmark": bench_name,
                "format": fmt.key, "dataset": dataset.name}

    write_result(result)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _env_for_format(format_key: str | None) -> str | None:
    """Map a format key to the conda env that has its deps.

    The bench's env split is documented in
    ``benchmarks/README.md § "Which environment to use"``: GPU
    benchmarks live in ``scx-bench-gpu`` (RAPIDS: cuVS, cuGraph),
    SLAF in ``scx-bench-slaf`` (slafdb / lance — pins conflict with
    the main env per the slaf yaml), BPCells in ``scx-bench-r`` (R
    toolchain), and the rest in ``scx-bench``.

    Returns the env name; callers pass it to
    :func:`_slurm_setup_cmds` so each worker activates the env
    whose Python interpreter can actually import its deps. The fix
    eliminates the "cugraph not available" / "cuVS library not
    found — falling back to CPU HNSW" / "slafdb is not installed"
    failures that show up when every worker activates the
    orchestrator's env.

    Returns ``None`` for unknown / missing format keys; the caller
    then falls back to the orchestrator's ``CONDA_PREFIX``.
    """
    if not format_key:
        return None
    if "_gpu" in format_key:
        return "scx-bench-gpu"
    if "slaf" in format_key:
        return "scx-bench-slaf"
    if "bpcells" in format_key:
        return "scx-bench-r"
    return "scx-bench"


# Runtime knobs a *variant* selects, exported into the worker's shell.
#
# Two of these arms are chosen by an environment variable, and one of those
# variables is read **once per process**: `SCX_GPU_PCA_RESIDENT`
# (`gpu_pca_resident.rs` caches it in a OnceLock on purpose, and
# `pyscx/tests/test_gpu_pca_resident.py` runs its two arms as subprocesses
# because of it). An in-process context manager inside the benchmark module is
# therefore not sufficient on its own — by the time it runs, an earlier GPU PCA
# call in the same process may already have latched the value. Exporting here
# sets it before the worker starts, which is the only placement that is right
# for a cached knob.
#
# `SCX_SHUFDELTA_NVCOMP` is read per call and does not need this; it is
# exported anyway so both arms are selected the same way, and so a run driven
# through some other entry point still lands on the arm its name claims.
# The PCA arm's knob is a literal because `accel_pca` selects it through
# `dispatch_env`, not through a data table the way the decode arms do.
_PCA_ARM_ENV: dict[str, dict[str, str]] = {
    "accel_pca__pyscx_gpu_streaming": {"SCX_GPU_PCA_RESIDENT": "0"},
}


@functools.lru_cache(maxsize=1)
def _format_arm_env() -> dict[str, dict[str, str]]:
    """Runtime knobs a *variant* selects, exported into the worker's shell.

    The decode arms' half is read from `accel_to_gpu_anndata.ARMS` rather than
    copied: duplicating it meant a later edit could update one side and silently
    re-open the nvcomp-fallback hole these arms exist to close.

    **Lazy, and it raises.** An eager module-level version of this — the shape
    the first fix shipped — imported `accel_to_gpu_anndata` at import of the
    orchestrator, which evaluates `_HAS_PYSCX` / `_HAS_PYSCX_GPU` and probes
    CUDA. Every `run_parallel` invocation paid for that, including
    `--benchmarks compression` on a CPU-only login node. Worse, it swallowed
    `ImportError` into an empty map, which would drop `SCX_SHUFDELTA_NVCOMP=1`
    from the nvcomp worker *in silence* — the exact failure the arm exists to
    make impossible. A hard-coded dict could not do that, so the lazy form must
    not either: an import failure here is a real error, not a default.
    Regression found by **Cursor Agent** on PR #474 round 2.
    """
    mod = importlib.import_module(
        "benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata"
    )
    return {**_PCA_ARM_ENV, **{k: dict(a.env) for k, a in mod.ARMS.items() if a.env}}


def _arm_env_exports(format_key: str | None) -> list[str]:
    """`export VAR=...` lines selecting the decode/compute arm this format names.

    Empty for every format that is not an arm — which is all but two.
    """
    if not format_key:
        return []
    # Cheap gate first: only an arm variant can carry a knob, and resolving the
    # map imports the decode benchmark, which probes CUDA. Without this, a
    # `--benchmarks compression` capture on a CPU-only host still paid for it
    # (codex, round 3).
    if not (
        format_key in _PCA_ARM_ENV or format_key.startswith("accel_to_gpu_anndata__")
    ):
        return []
    return [
        f"export {var}='{val}'"
        for var, val in sorted(_format_arm_env().get(format_key, {}).items())
    ]


def _slurm_setup_cmds(env_name: str | None = None, format_key: str | None = None) -> list[str]:
    """Per-job shell setup: activate the right Python env and clear inherited SLURM vars.

    `env_name` overrides the orchestrator's ``CONDA_PREFIX`` so each
    worker can land in the env that has its deps (see
    :func:`_env_for_format`). When ``None``, falls back to the
    orchestrator's env — preserves the pre-routing behaviour for
    callers that don't care.
    """
    import os
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    env_cleanup = "unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true"
    # Disable srun's MPI bootstrap. Chimera's slurm.conf has MpiDefault=pmix,
    # but the pmix plugin isn't runtime-loadable — submitit's `srun` would
    # otherwise fail with "Cannot create context for mpi/pmix". We don't
    # use MPI; tell srun so.
    mpi_none = "export SLURM_MPI_TYPE=none"

    # Forward GCP credentials to the worker. ``cloud_*`` benchmarks
    # (cloud_push / cloud_pull / cloud_read / cloud_metadata /
    # cloud_filtered / cloud_reader_vs_pull / cost_model /
    # cloud_large_atlas) need ``GOOGLE_APPLICATION_CREDENTIALS`` to
    # auth against GCS — without it ~64 cloud cells per tier-small
    # run fail with "anonymous gcsfs" / 401 errors. The orchestrator
    # already loads it from `.env` (via bench_env.py + python-dotenv),
    # but submitit doesn't propagate process-environment vars to
    # SLURM workers, so we inject an explicit export here. Also
    # tilde-expand the path because gcsfs / google-auth don't.
    arm_setup = _arm_env_exports(format_key)
    cloud_setup: list[str] = []
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if gac:
        gac = os.path.expanduser(gac)
        # Single-quote so spaces / special chars in the path don't
        # explode the shell. Skip when the file isn't readable on
        # the orchestrator side — the worker will inherit the same
        # Weka filesystem, so unreadable here means unreadable there.
        if os.path.isfile(gac):
            cloud_setup.append(
                f"export GOOGLE_APPLICATION_CREDENTIALS='{gac}'"
            )
            # pyscx.push / pull / open_cloud go through object_store's
            # GoogleCloudStorageBuilder::from_env(), which recognises
            # GOOGLE_SERVICE_ACCOUNT_PATH but does not auto-fall-back to
            # GOOGLE_APPLICATION_CREDENTIALS. Off-GCE hosts otherwise pay a
            # ~14 s IMDS retry timeout per push call before failing. Mirror
            # GAC into GOOGLE_SERVICE_ACCOUNT_PATH unless the operator set
            # one explicitly.
            if not os.environ.get("GOOGLE_SERVICE_ACCOUNT_PATH", "").strip():
                cloud_setup.append(
                    f"export GOOGLE_SERVICE_ACCOUNT_PATH='{gac}'"
                )
        else:
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS=%s does not exist; "
                "cloud_* benchmark jobs will run unauthenticated.",
                gac,
            )
    # Forward GCS bucket / project / region knobs from cloud_fixtures /
    # config so the worker doesn't have to re-read `.env`.
    # `SCX_DISABLE_CUDA_GRAPHS` is a GPU-runtime knob that needs to reach the
    # bench worker; otherwise the OnceLock-cached check defaults graphs on.
    # (The `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` gates were removed once GPU DE v3
    # became the unconditional default.)
    # `SCX_CPU_PROFILE` (Phase-2 task 2.0) turns on the CPU per-stage profiler
    # (io/decode/reduction/marshalling → `runs[].extra["cpu_profile_*"]`); it is
    # read once at profiler init, so it must be exported before the worker
    # imports pyscx — hence forward it here like the other runtime knobs.
    for var in ("GCS_TEST_BUCKET", "GCP_PROJECT", "GCP_BUCKET_REGION",
                "SCX_DATA_DIR", "SCX_WORK_DIR",
                "SCX_DISABLE_CUDA_GRAPHS",
                "SCX_CPU_PROFILE",
                "SCX_BENCH_WITH_CSC"):
        val = os.environ.get(var, "").strip()
        if val:
            val = os.path.expanduser(val) if "DIR" in var else val
            cloud_setup.append(f"export {var}='{val}'")

    # Prefer the explicitly-routed env from `_env_for_format` over
    # the orchestrator's own CONDA_PREFIX. Falls back to CONDA_PREFIX
    # when no route is provided (preserves the legacy "all workers
    # use the orchestrator's env" behaviour for callers that don't
    # pass `env_name`).
    if env_name is None:
        if "scx-bench" in conda_prefix:
            env_name = os.path.basename(conda_prefix)
        else:
            env_name = None  # fall through to the .venv path below

    if env_name is not None:
        conda_base = os.environ.get("CONDA_EXE", "").replace("/bin/conda", "")
        if not conda_base:
            conda_base = str(Path.home() / "miniforge3")
        return [
            env_cleanup,
            mpi_none,
            *cloud_setup,
            *arm_setup,
            f'eval "$({conda_base}/bin/conda shell.bash hook)"',
            f"conda activate {env_name}",
        ]
    return [
        env_cleanup,
        mpi_none,
        *cloud_setup,
        *arm_setup,
        f"export PATH={PROJECT_ROOT}/.venv/bin:$PATH",
    ]


def _slurm_params(args, is_conversion: bool = False) -> dict:
    """Build default submitit executor parameters from CLI args.

    Used as the fallback / cap when per-job sizing isn't applicable
    (e.g. conversion jobs that don't yet know which benchmark will run).
    """
    timeout_min = args.timeout
    if is_conversion and timeout_min < 240:
        timeout_min = 240  # conversions need more time for large datasets

    return {
        "slurm_partition": args.partition,
        "cpus_per_task": args.cpus,
        "mem_gb": args.mem_gb,
        "timeout_min": timeout_min,
        # `_slurm_params` is the fallback for callers that don't know
        # the format yet; without a format we can't route per-env, so
        # we use the orchestrator's CONDA_PREFIX. The per-job path
        # below (`_per_job_slurm_params`) is the one that does the
        # actual routing.
        "slurm_setup": _slurm_setup_cmds(),
    }


def _per_job_slurm_params(
    args,
    dataset_name: str,
    format_key: str,
    benchmark: str | None,
    is_conversion: bool = False,
) -> dict:
    """Per-(benchmark, dataset, format) submitit parameters.

    Sizes ``mem_gb`` from ``estimate_memory_gb`` (or the conversion peak when
    benchmark is None) and auto-routes to ``cpu_high_mem`` when the estimate
    exceeds the preemptible cap. The ``--mem-gb`` CLI arg becomes a floor:
    we never request less than the user explicitly asked for.
    """
    from benchmarks.comprehensive.config import (  # deferred to avoid circ
        MEM_CEILING_GB, estimate_time_minutes,
    )

    cfg = DATASETS[dataset_name]
    scale = max(getattr(args, "scale_factor", 1.0), 0.1)

    if is_conversion or benchmark is None:
        # Conversion needs to load the source h5ad and write the target —
        # peak across read_full + write is a safe upper bound.
        mem = max(
            estimate_memory_gb(cfg, format_key, "read_full"),
            estimate_memory_gb(cfg, format_key, "write"),
        )
        time_bench = "write"
    else:
        mem = estimate_memory_gb(cfg, format_key, benchmark)
        time_bench = benchmark

    # Apply the scale factor (memory + time), then clamp.
    mem = min(int(round(mem * scale)), MEM_CEILING_GB)
    mem = max(mem, args.mem_gb)  # CLI floor

    est_time = estimate_time_minutes(cfg, format_key, time_bench)
    est_time = int(round(est_time * scale))
    timeout_min = max(est_time, args.timeout if is_conversion else 0, 5)
    if is_conversion and timeout_min < 240:
        timeout_min = 240

    partition = partition_for_memory(mem, default=args.partition)

    # Route accelerator benchmarks
    # to the GPU partition **only for variants that actually use the GPU**.
    # Variant naming convention: format_key ending in `_gpu` or containing
    # `_gpu_` indicates GPU dispatch (e.g. `accel_pca__pyscx_gpu_rand_hh`,
    # `accel_pca__rapids_singlecell_gpu`). CPU-only variants (`accel_*__scanpy_cpu`,
    # `accel_*__pyscx_cpu*`, `accel_leiden__leidenalg_cpu`) flow through
    # partition_for_memory which auto-promotes to cpu_high_mem when sizing
    # exceeds the preemptible-GPU node's RAM ceiling — unblocking
    # accel_preprocess on census_1m (needs >64 GB).
    #
    # ml_loader on SCX formats also routes to GPU so the SCX-only
    # `gpu_train` scenario in benchmarks/ml_loader.py:949 can fire (it
    # silently no-ops on hosts without CUDA). Non-SCX ml_loader variants
    # have no GPU scenario and stay on CPU.
    extra_slurm: dict = {}
    needs_gpu = (
        benchmark is not None
        and (
            (
                benchmark.startswith("accel_")
                and ("_gpu_" in format_key or format_key.endswith("_gpu"))
            )
            or (benchmark == "ml_loader" and format_key.startswith("scx_"))
        )
    )
    if needs_gpu:
        partition = "preemptible"  # GPU partition on chimera
        extra_slurm["slurm_gres"] = "gpu:1"
        # Chimera's preemptible GPU QOS caps per-job memory at ~128 GB
        # (matching SLURM_DEFAULTS.gpu.mem_gb). Requests above that fail
        # with `QOSMaxGRESPerJob`. Clamp so census-scale preprocess cells
        # that actually use GPU compute don't get rejected at submit time.
        # (CPU variants have already been routed away via the `needs_gpu`
        # check and will pick up cpu_high_mem via partition_for_memory.)
        _GPU_MEM_CEILING_GB = 128
        if mem > _GPU_MEM_CEILING_GB:
            logger.warning(
                "Clamping %s__%s__%s mem from %dG to %dG (GPU QOS cap)",
                benchmark, format_key, dataset_name,
                mem, _GPU_MEM_CEILING_GB,
            )
            mem = _GPU_MEM_CEILING_GB

    logger.info(
        "Sized %s__%s__%s: mem=%dG time=%dm partition=%s%s (scale=%.2f)",
        benchmark or "convert", format_key, dataset_name,
        mem, timeout_min, partition,
        " gres=gpu:1" if extra_slurm.get("slurm_gres") else "",
        scale,
    )

    params: dict = {
        "slurm_partition": partition,
        "cpus_per_task": args.cpus,
        "mem_gb": mem,
        "timeout_min": timeout_min,
        # Per-job env routing: activate the env that actually has the
        # format's deps. GPU formats land in scx-bench-gpu (cugraph,
        # cuvs), slaf in scx-bench-slaf (slafdb), bpcells in
        # scx-bench-r, everything else in scx-bench. See
        # `_env_for_format` for the mapping rationale.
        "slurm_setup": _slurm_setup_cmds(_env_for_format(format_key), format_key),
        # Submitit's ``executor.update_parameters`` mutates the
        # executor's persistent state; values from a previous submit
        # carry over to the next one unless explicitly cleared. This
        # is a real bug in the wild — an SCX/ml_loader cell sets
        # ``slurm_gres="gpu:1"`` and the very next CPU-only cell
        # (e.g. ``ml_loader/h5ad_none/census_500k`` on
        # ``cpu_high_mem``) inherits the gres request, fails sbatch
        # with ``QOSMaxGRESPerJob`` (cpu_high_mem has no GPUs).
        # Always emit an explicit ``slurm_gres`` (empty string when
        # not needed) so each call fully overrides the executor's
        # cached state.
        "slurm_gres": "",
    }
    params.update(extra_slurm)
    return params


def _assert_release_pyscx() -> None:
    """Refuse to benchmark against a debug pyscx build.

    A debug `.so` (`maturin develop` without `--release`) runs ~4-10x slower
    uniformly and silently poisons every scx timing.
    The shared editable `.so` is what every SLURM worker imports, so checking it
    here in the orchestrator catches the footgun before a whole campaign is
    submitted. Bypass (not recommended) with SCX_BENCH_ALLOW_DEBUG=1.
    """
    if os.environ.get("SCX_BENCH_ALLOW_DEBUG", "").strip() in ("1", "true", "TRUE"):
        return
    try:
        import pyscx
    except Exception as exc:  # pyscx not importable here — leave it to the workers
        logging.warning("pyscx not importable in orchestrator env (%s); "
                        "skipping build-profile guard", exc)
        return
    profile = getattr(pyscx, "__build_profile__", "unknown")
    if profile == "release":
        return
    if profile == "unknown":
        logging.warning(
            "pyscx has no __build_profile__ (pre-guard build); cannot verify release. "
            "Rebuild: cd pyscx && maturin develop --release --features hdf5,gpu"
        )
        return
    raise SystemExit(
        f"ERROR: pyscx is a {profile!r} build — benchmarks require a release build "
        "(a debug .so is ~4-10x slower and poisons all timings). Rebuild with:\n"
        "  cd pyscx && ../.venv/bin/maturin develop --release --features hdf5,gpu\n"
        "Bypass (not recommended) with SCX_BENCH_ALLOW_DEBUG=1."
    )


def main() -> None:
    import submitit

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.dry_run:
        _assert_release_pyscx()

    # Resolve datasets
    datasets = args.datasets
    if not datasets:
        datasets = [name for name, cfg in DATASETS.items() if cfg.h5ad_path.exists()]

    # Resolve formats
    # `accel_formats()` is lazy-loaded — pulls in the `FormatVariant` entries
    # for each `accel_*.py` module's implementation variants (PCA: scanpy_cpu,
    # pyscx_cpu_auto, pyscx_gpu_rand_hh, pyscx_gpu_rand_chol, rapids_singlecell_gpu;
    # kNN / UMAP / Leiden / preprocess / HVG similarly). Included whenever
    # --include-accel is set, when --formats explicitly names an accel key,
    # or when --benchmarks names any accel_* benchmark.
    from benchmarks.comprehensive.config import accel_formats
    need_accel = (
        args.include_accel
        or any((fk or "").startswith("accel_") for fk in (args.formats or []))
        or any((b or "").startswith("accel_") for b in (args.benchmarks or []))
        # `bench_csc_dispatch` exposes its variants through
        # `accel_formats()` even though the benchmark name doesn't carry
        # the `accel_` prefix. Treat its presence the same way.
        or any(b == "bench_csc_dispatch" for b in (args.benchmarks or []))
        or any((fk or "").startswith("bench_csc__") for fk in (args.formats or []))
    )
    # Multimodal benchmarks (Phase K) live outside PRIMARY_FORMATS —
    # auto-include the multimodal format set whenever a multimodal
    # benchmark is requested, mirroring `need_accel`. Without this,
    # `--benchmarks multimodal_*` from `capture_baseline.py` (which
    # never passes `--formats`) yields zero submissions because no
    # h5mu / zarr_mudata / scx_multimodal_* format is in the default
    # format pool.
    from benchmarks.comprehensive.config import MULTIMODAL_FORMATS
    need_multimodal = (
        any(b in _MULTIMODAL_BENCHMARKS for b in (args.benchmarks or []))
        or any(_is_multimodal_format(fk or "") for fk in (args.formats or []))
    )
    pool = list(ALL_FORMATS)
    if need_accel:
        pool.extend(accel_formats())
    if args.formats:
        all_by_key = {f.key: f for f in pool}
        formats = [all_by_key[k] for k in args.formats if k in all_by_key]
    elif args.include_additional and need_accel:
        formats = pool
    elif args.include_additional:
        formats = list(ALL_FORMATS)
    elif need_accel and need_multimodal:
        formats = list(PRIMARY_FORMATS) + accel_formats() + list(MULTIMODAL_FORMATS)
    elif need_accel:
        formats = list(PRIMARY_FORMATS) + accel_formats()
    elif need_multimodal:
        formats = list(PRIMARY_FORMATS) + list(MULTIMODAL_FORMATS)
    else:
        formats = list(PRIMARY_FORMATS)

    # --no-gpu: drop accel GPU variants. By convention accel format keys are
    # `<bench>__<impl>_<device>[_<extra>]`, e.g. `accel_pca__pyscx_gpu_rand_hh`.
    # The substring "_gpu" is unique to GPU variants — CPU keys use "_cpu" and
    # format-benchmark keys (zarr_zstd, scx_auto, …) don't include "_gpu".
    if args.no_gpu:
        before = len(formats)
        formats = [f for f in formats if "_gpu" not in f.key]
        dropped = before - len(formats)
        if dropped:
            logger.info("--no-gpu: dropped %d GPU accel format variants", dropped)

    # Resolve benchmarks
    benchmarks = args.benchmarks or BENCHMARK_NAMES

    # Mint a run ID that every submitted job inherits via its env, so the
    # per-result provenance blocks share a single group key for the
    # dashboard's "runs" view (Phase I.2).
    from benchmarks.comprehensive.provenance import current_run_id
    run_id = current_run_id()
    logger.info("Run ID:     %s", run_id)

    # Pre-submit runner contract check (Phase I.8). Catches capability
    # manifest violations before a 400-job fleet hits the scheduler.
    # Gated off with --skip-smoke for exploratory runs. Skipped automatically
    # when no non-accel benchmarks are scheduled (accel benchmarks never
    # touch the `runners/*_runner.py` contract surface — they go through
    # `accel_*.py` modules in `comprehensive/benchmarks/`, which the smoke
    # gate doesn't validate).
    # `bench_csc_dispatch` is structurally an accel benchmark (self-
    # contained, doesn't go through the runner contract surface) even
    # though its name doesn't carry the `accel_` prefix. Treat it the
    # same way for smoke-test gating.
    non_accel_benchmarks = [
        b for b in benchmarks
        if not b.startswith("accel_") and b != "bench_csc_dispatch"
    ]
    needs_smoke = (
        not getattr(args, "skip_smoke", False)
        and len(non_accel_benchmarks) > 0
    )
    if needs_smoke:
        logger.info("Pre-submit smoke (use --skip-smoke to bypass) …")
        import subprocess as _sp
        rc = _sp.call([
            sys.executable, "-m",
            "benchmarks.comprehensive.scripts.smoke_test_runners",
        ])
        if rc != 0:
            logger.error(
                "Runner contract check failed (rc=%d). Submit blocked. "
                "Fix the runner or re-run with --skip-smoke.", rc,
            )
            sys.exit(1)
    elif not getattr(args, "skip_smoke", False):
        logger.info(
            "Pre-submit smoke skipped: no non-accel benchmarks scheduled. "
            "Pass --skip-smoke to suppress this notice."
        )
    logger.info("Datasets:   %s", datasets)
    logger.info("Formats:    %s", [f.key for f in formats])
    logger.info("Benchmarks: %s", benchmarks)

    # Determine which benchmarks need pre-converted files
    needs_conversion = [b for b in benchmarks if b not in _NO_CONVERSION]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase A: Conversion ---
    # `conversion_paths` records files already on disk; `conv_jobs` records
    # conversions submitted this run (whose Phase-B dependents will gate on
    # `afterok:<jid>`). A `(ds, fmt)` key appears in at most one of the two.
    # `dry_run_conv_keys` mirrors `conv_jobs` for --dry-run so Phase B can
    # report would-be afterok edges without actually submitting anything.
    conversion_paths: dict[tuple[str, str], str] = {}
    conv_jobs: dict[tuple[str, str], submitit.Job] = {}
    dry_run_conv_keys: set[tuple[str, str]] = set()

    if needs_conversion and not args.skip_convert:
        logger.info("=" * 60)
        logger.info("Phase A: Converting datasets to target formats")
        logger.info("=" * 60)

        # `slurm_python="python"` overrides submitit's default of baking
        # the orchestrator's absolute `/path/to/scx-bench/bin/python`
        # into the srun command. With per-job env routing
        # (`_env_for_format`) the worker `conda activate`s the right
        # env in `slurm_setup`; plain `python` then resolves to that
        # env's interpreter via PATH. Without this override, every
        # worker uses the orchestrator's python regardless of which
        # env was activated — defeating the routing fix.
        executor = submitit.AutoExecutor(
            folder=str(LOGS_DIR / "convert"),
            slurm_python="python",
        )

        for ds_name in datasets:
            cfg = DATASETS[ds_name]
            for fmt in formats:
                # Skip incompatible pairings (multimodal dataset ↔
                # single-modality format and vice versa). Sample any
                # benchmark we plan to schedule on this dataset to
                # decide compatibility — the convert phase only cares
                # whether *some* benchmark in the run will use this
                # (dataset, format) pair, which is the per-axis OR.
                if cfg.multimodal != _is_multimodal_format(fmt.key):
                    continue
                key = (ds_name, fmt.key)
                output_path = cfg.path_for_format(fmt.key)

                if output_path.exists() and not args.overwrite:
                    conversion_paths[key] = str(output_path)
                    logger.info("  Exists: %s / %s -> %s", ds_name, fmt.key, output_path)
                    continue

                if args.dry_run:
                    params = _per_job_slurm_params(
                        args, ds_name, fmt.key, benchmark=None, is_conversion=True,
                    )
                    logger.info(
                        "  [DRY RUN] Would convert: %s / %s [%dG, %s]",
                        ds_name, fmt.key, params["mem_gb"], params["slurm_partition"],
                    )
                    dry_run_conv_keys.add(key)
                    continue

                params = _per_job_slurm_params(
                    args, ds_name, fmt.key, benchmark=None, is_conversion=True,
                )
                # Throttle: don't submit if the user's queue is at /
                # near Chimera's QOSMaxSubmitJobPerUserLimit. Polls
                # squeue and blocks until the queue drops below the
                # cap. No-op when --max-pending-jobs <= 0.
                _wait_under_pending_cap(args.max_pending_jobs, kind="convert")
                executor.update_parameters(**params)
                job = executor.submit(
                    _run_conversion,
                    ds_name, fmt.key, fmt.runner, fmt.params, args.overwrite,
                )
                conv_jobs[key] = job
                logger.info(
                    "  Submitted: %s / %s [%dG, %s] -> job %s",
                    ds_name, fmt.key, params["mem_gb"], params["slurm_partition"], job.job_id,
                )

        if conv_jobs and not args.dry_run:
            logger.info(
                "Submitted %d conversion job(s); benchmarks will release as each lands.",
                len(conv_jobs),
            )
    elif needs_conversion:
        # --skip-convert: assume files exist at persistent paths
        for ds_name in datasets:
            cfg = DATASETS[ds_name]
            for fmt in formats:
                if cfg.multimodal != _is_multimodal_format(fmt.key):
                    continue
                path = cfg.path_for_format(fmt.key)
                if path.exists():
                    conversion_paths[(ds_name, fmt.key)] = str(path)

    # --- Phase B: Benchmark jobs (Resource-Cohort Arrays) ---
    #
    # Design: instead of one ``executor.submit(...)`` per (benchmark, dataset,
    # format) triple, we group triples into *resource cohorts* keyed by
    # ``(dataset, format_key, partition, gres, needs_conversion)`` and submit
    # each cohort as a single SLURM Job Array via
    # ``executor.map_array(...)``. SLURM requires every task in an array to
    # share identical resource parameters, so the 5-tuple guarantees:
    #   - GPU and CPU benchmarks auto-separate (different ``partition`` and
    #     ``gres``) — e.g. ``ml_loader`` on ``scx_auto`` goes to the GPU
    #     partition while ``read_full`` on the same key stays on CPU.
    #   - High-memory benchmarks promoted to ``cpu_high_mem`` by
    #     ``partition_for_memory`` get their own cohort.
    #   - ``_NO_CONVERSION`` benchmarks (write, parallel_write_scaling,
    #     cell_eval_parity_perf) skip the ``afterok`` dependency by sitting
    #     in cohorts with ``needs_conversion=False``.
    #   - Accel/CSC pairing rules (an ``accel_pca`` benchmark only with
    #     ``accel_pca__*`` formats; ``bench_csc_dispatch`` only with
    #     ``bench_csc__*`` formats) are enforced by
    #     ``_bench_format_compatible`` before grouping.
    #
    # QOS handling: SLURM counts each array task individually against
    # ``QOSMaxSubmitJobPerUserLimit``. The documented Chimera cap is ~500 on
    # ``cpu_preemptible``, but the May 2026 tier-xl run observed rejections
    # at queue depth 80–110, so we throttle conservatively. Native cap is
    # via ``slurm_array_parallelism`` per cohort, plus a pre-submit
    # ``_wait_under_pending_cap(args.max_pending_jobs)`` block (default 75)
    # and a 3× / 60s retry around ``map_array`` to absorb races between the
    # squeue poll and SLURM's internal counter.
    #
    # The two constraints driving this design: (a) SLURM requires every
    # task in a job array to share identical resource parameters, so the
    # 5-tuple cohort key (dataset, format_key, partition, gres,
    # needs_conversion) is the finest grouping that preserves uniformity;
    # (b) ``QOSMaxSubmitJobPerUserLimit`` counts each array task
    # individually, so the per-task throttle still applies.
    logger.info("=" * 60)
    logger.info("Phase B: Grouping benchmarks into resource cohorts")
    logger.info("=" * 60)

    # Build a format lookup so cohort submission can resolve runner/params
    # from format_key without re-iterating the formats list.
    formats_by_key: dict[str, FormatVariant] = {f.key: f for f in formats}

    # ── Phase B.1: Group benchmarks into resource cohorts ──────────────
    # Each cohort key is (dataset, format_key, partition, gres, needs_conversion).
    # All benchmarks sharing the same key can safely coexist in one SLURM array.
    cohorts: dict[tuple, list[str]] = {}
    for bench_name in benchmarks:
        for ds_name in datasets:
            for fmt in formats:
                # Accel/CSC pairing rules
                if not _bench_format_compatible(bench_name, fmt.key):
                    continue
                # Multimodal compatibility
                if not _triple_compatible(bench_name, ds_name, fmt.key):
                    continue
                key = _build_cohort_key(args, ds_name, fmt.key, bench_name)
                cohorts.setdefault(key, []).append(bench_name)

    logger.info(
        "Grouped %d benchmarks into %d resource cohorts",
        sum(len(v) for v in cohorts.values()), len(cohorts),
    )

    # ── Phase B.2: Submit each cohort as a SLURM Job Array ────────────

    # See companion comment on the convert executor above re: why
    # `slurm_python="python"` is necessary for per-job env routing.
    executor = submitit.AutoExecutor(
        folder=str(LOGS_DIR / "bench"),
        slurm_python="python",
    )

    # Each entry: (label, bench_job, conv_key | None). conv_key points back
    # at conv_jobs so the post-hoc wait loop can attribute a benchmark
    # failure to a cancelled-by-dependency state vs a real benchmark crash.
    bench_jobs: list[tuple[str, submitit.Job, tuple[str, str] | None]] = []
    dep_jobid_for_label: dict[str, str | None] = {}

    def _submit_cohort(cohort_key: tuple, group_benches: list[str]) -> None:
        """Submit a single resource cohort as a SLURM Job Array.

        Mutates the enclosing ``bench_jobs`` / ``dep_jobid_for_label`` in
        place. ``return`` short-circuits the current cohort (the former
        loop-level ``continue``). May raise on QOS-limit exhaustion — the
        caller's ``finally`` persists the manifest before it propagates.
        """
        ds_name, fmt_key, partition, gres, needs_conv = cohort_key

        # --- Resource envelope for this cohort ---
        array_params = _determine_cohort_resources(args, cohort_key, group_benches)

        # --- Build executor kwargs ---
        update_kwargs: dict[str, Any] = dict(array_params)

        # --- Conversion dependency (only for cohorts that need it) ---
        dep_jobid: str | None = None
        if needs_conv and (ds_name, fmt_key) in conv_jobs:
            conv_job = conv_jobs[(ds_name, fmt_key)]
            # Short-circuit if upstream conversion already entered a terminal
            # failure state (matches the historical check)
            head_state = _job_terminal_head(conv_job)
            if head_state in _TERMINAL_FAILURE_STATES:
                logger.warning(
                    "Skipping cohort %s/%s — upstream conversion %s is %s",
                    ds_name, fmt_key, conv_job.job_id, head_state,
                )
                return
            dep_jobid = conv_job.job_id
            update_kwargs["slurm_additional_parameters"] = {
                "dependency": f"afterok:{dep_jobid}"
            }
        elif needs_conv and args.dry_run and (ds_name, fmt_key) in dry_run_conv_keys:
            dep_jobid = "<conv-pending>"
            update_kwargs["slurm_additional_parameters"] = {}
        else:
            # Explicitly clear to prevent merge carry-over from previous cohort
            update_kwargs["slurm_additional_parameters"] = {}

        # --- Array parallelism (native SLURM throttle) ---
        if args.max_pending_jobs > 0:
            # Distribute the budget across cohorts, with a floor of 3
            update_kwargs["slurm_array_parallelism"] = max(
                3, args.max_pending_jobs // max(len(cohorts), 1)
            )

        # --- Construct parallel argument lists for map_array ---
        # These must match _run_benchmark's 8-parameter signature exactly:
        #   (bench_name, dataset_name, format_key, format_runner,
        #    format_params, n_runs, cold_cache, converted_path_str)
        bench_names: list[str] = []
        dataset_names: list[str] = []
        format_keys: list[str] = []
        format_runners: list[str] = []
        format_params_list: list[dict] = []
        n_runs_list: list[int] = []
        cold_caches: list[bool] = []
        converted_path_strs: list[str | None] = []

        fmt_obj = formats_by_key[fmt_key]
        for bench_name in group_benches:
            bench_names.append(bench_name)
            dataset_names.append(ds_name)
            format_keys.append(fmt_key)
            format_runners.append(fmt_obj.runner)
            format_params_list.append(fmt_obj.params)
            n_runs_list.append(n_runs_for_dataset(ds_name))
            cold_caches.append(args.cold_cache)

            # Resolve converted file path
            conv_key = (ds_name, fmt_key)
            if not needs_conv:
                conv_path: str | None = None
            elif conv_key in conversion_paths:
                conv_path = conversion_paths[conv_key]
            elif conv_key in conv_jobs or (args.dry_run and conv_key in dry_run_conv_keys):
                conv_path = str(DATASETS[ds_name].path_for_format(fmt_key))
            else:
                conv_path = None
            converted_path_strs.append(conv_path)

        # Skip cohorts where non-_NO_CONVERSION benchmarks have no source.
        # If ``not dep_jobid`` is true here, the cohort has neither a real
        # conversion jobid nor the ``<conv-pending>`` dry-run placeholder.
        if needs_conv and not dep_jobid:
            # No conversion job and not in conversion_paths — check if we
            # can actually run these benchmarks
            if all(p is None for p in converted_path_strs):
                logger.warning(
                    "  SKIP cohort %s/%s: no converted file available",
                    ds_name, fmt_key,
                )
                return

        # --- Dry-run: print plan without submitting ---
        if args.dry_run:
            logger.info(
                "DRY-RUN cohort %s/%s/%s gres=%s conv=%s: %d tasks, "
                "mem=%dGB time=%dmin",
                ds_name, fmt_key, partition, gres or "none",
                "dep" if dep_jobid else ("no-conv" if not needs_conv else "exists"),
                len(group_benches), array_params["mem_gb"],
                array_params["timeout_min"],
            )
            for bn in group_benches:
                logger.info("  task: %s/%s/%s", bn, ds_name, fmt_key)
            return

        # --- QOS-aware throttle: reserve headroom for the whole cohort ---
        # Always throttle, even when --max-pending-jobs is explicitly
        # zeroed; the May 2026 tier-xl run observed
        # QOSMaxSubmitJobPerUserLimit rejections at queue depth 80–110
        # on cpu_preemptible (the documented ~500 cap does not reflect
        # what the scheduler actually enforces). Each map_array call
        # counts as N individual array tasks against the QOS limit, so
        # we pass the upcoming cohort's size as ``headroom`` — the
        # throttle then blocks until there is room for every task in the
        # cohort, not just one more job.
        tasks_in_cohort = len(group_benches)
        effective_cap = args.max_pending_jobs if args.max_pending_jobs > 0 else 75
        _wait_under_pending_cap(
            effective_cap, kind="bench", headroom=tasks_in_cohort
        )

        executor.update_parameters(**update_kwargs)

        # --- Submit the cohort as a single SLURM Job Array ---
        # Retry once on QOSMaxSubmitJobPerUserLimit — the squeue poll in
        # _wait_under_pending_cap can race with SLURM's internal counter
        # (array tasks still being registered).
        _QOS_RETRY_MAX = 3
        for _attempt in range(_QOS_RETRY_MAX):
            try:
                task_jobs = executor.map_array(
                    _run_benchmark,
                    bench_names,
                    dataset_names,
                    format_keys,
                    format_runners,
                    format_params_list,
                    n_runs_list,
                    cold_caches,
                    converted_path_strs,
                )
                break  # success
            except Exception as exc:
                if "QOSMaxSubmitJobPerUserLimit" in str(exc):
                    if _attempt < _QOS_RETRY_MAX - 1:
                        logger.warning(
                            "QOS limit hit submitting cohort %s/%s "
                            "(attempt %d/%d); waiting 60s for drain...",
                            ds_name, fmt_key, _attempt + 1, _QOS_RETRY_MAX,
                        )
                        time.sleep(60)
                        _cancel_dep_never_satisfied()
                        _wait_under_pending_cap(
                            effective_cap, kind="bench", headroom=tasks_in_cohort
                        )
                        continue
                raise  # re-raise non-QOS errors or final attempt

        # Record jobs for the wait loop and manifest
        for bench_name, job in zip(group_benches, task_jobs):
            label = f"{bench_name}/{ds_name}/{fmt_key}"
            conv_key_val = (ds_name, fmt_key) if dep_jobid else None
            bench_jobs.append((label, job, conv_key_val))
            dep_jobid_for_label[label] = dep_jobid

        logger.info(
            "  Submitted cohort %s/%s/%s: %d tasks, mem=%dGB, "
            "time=%dmin%s -> array %s",
            ds_name, fmt_key, partition, tasks_in_cohort,
            array_params["mem_gb"], array_params["timeout_min"],
            f" deps=afterok:{dep_jobid}" if dep_jobid else "",
            task_jobs[0].job_id.rsplit("_", 1)[0] if task_jobs else "?",
        )

    # ── Write manifest (single JSON dump, matching existing pattern) ──
    def _persist_bench_manifest() -> None:
        if not args.dry_run and bench_jobs:
            manifest_path = LOGS_DIR / "run_manifest.json"
            manifest = {
                "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "submitted": [
                    {
                        "label": label,
                        "job_id": job.job_id,
                        "submitit_folder": str(job.paths.folder),
                        "dependency": dep_jobid_for_label.get(label),
                    }
                    for label, job, _conv_key in bench_jobs
                ],
            }
            import json as _json
            manifest_path.write_text(_json.dumps(manifest, indent=2, default=str))
            logger.info("Run manifest: %s", manifest_path)

    # Submit every cohort, persisting the manifest in a ``finally`` so that
    # cohorts already submitted are recorded even if a later cohort raises
    # (e.g. QOSMaxSubmitJobPerUserLimit exhaustion in the retry loop) —
    # otherwise watch.py would lose track of the in-flight arrays.
    try:
        for cohort_key, group_benches in cohorts.items():
            _submit_cohort(cohort_key, group_benches)
    finally:
        _persist_bench_manifest()

    if args.dry_run:
        n_conv = sum(
            1 for ds in datasets for fmt in formats
            if not DATASETS[ds].path_for_format(fmt.key).exists() or args.overwrite
        )
        n_bench = sum(len(v) for v in cohorts.values())
        n_dep = sum(
            len(v) for k, v in cohorts.items()
            if k[4]  # needs_conversion
            and (
                not DATASETS[k[0]].path_for_format(k[1]).exists() or args.overwrite
            )
        )
        logger.info(
            "DRY RUN: would submit %d conversion + %d benchmark "
            "(%d benchmark→conversion afterok edges) across %d cohort arrays "
            "= %d total SLURM jobs (arrays + individual conversions)",
            n_conv, n_bench, n_dep, len(cohorts), n_conv + len(cohorts),
        )
        return

    # Wait for benchmark jobs. With afterok dependencies in play, the wait
    # loop sees three outcomes per job: success, failure (the benchmark
    # itself crashed), and dep_failed (SLURM cancelled the job because its
    # upstream conversion never succeeded — DependencyNeverSatisfied).
    # Distinguish the latter so dependency-cascade noise doesn't masquerade
    # as benchmark regressions in the operator's eye.
    if bench_jobs:
        logger.info("Waiting for %d benchmark jobs...", len(bench_jobs))
        t0 = time.perf_counter()
        done, failed, dep_failed = 0, 0, 0
        # Periodically scancel jobs stuck in
        # `DependencyNeverSatisfied` while we wait. Without this, a
        # convert that fails AFTER its dependents were submitted
        # leaves them queued forever — `job.result()` blocks
        # indefinitely on a still-PD slurm job, so the orchestrator
        # would hang here for hours. Cleaning periodically lets
        # `.result()` advance (cancelled → FailedJobError → counted
        # as dep_failed). We sweep every N completed jobs to amortise
        # the squeue cost; ``check_freq`` mirrors the throttle's 30s
        # cadence at typical 1-job-per-second drain rates.
        check_freq = max(1, min(len(bench_jobs) // 20, 25))
        for idx, (label, job, conv_key) in enumerate(bench_jobs):
            if args.max_pending_jobs > 0 and idx % check_freq == 0:
                cancelled = _cancel_dep_never_satisfied()
                if cancelled:
                    logger.info(
                        "  [result-wait] cancelled %d zombie "
                        "DependencyNeverSatisfied job(s); .result() "
                        "calls on those will surface as dep_failed.",
                        cancelled,
                    )
            # Fast-path: submitit's `.result()` on a CANCELLED job
            # spins for ~15s per call (it polls slurm's pickle output
            # with backoff before declaring failure). Across thousands
            # of cancelled jobs that's hours of wall-time. Short-
            # circuit by checking ``job.state`` first — when in a
            # terminal failure state, raise immediately so the existing
            # try/except classifies it and we move on.
            #
            # SLURM (via sacct) returns rich state strings, not bare
            # codes:
            #   "CANCELLED by 10024"        — user-cancelled
            #   "CANCELLED+0:0"             — cancelled with exit
            #   "FAILED"                    — runtime error
            #   "TIMEOUT"                   — wallclock exceeded
            #   "NODE_FAIL"                 — node died
            #   "OUT_OF_MEMORY"             — OOM
            #   "PREEMPTED"                 — preempted (cpu_preemptible)
            # Match on the FIRST whitespace-/+-delimited token so all
            # variants land correctly (the previous exact-set match
            # silently fell through on "CANCELLED by 10024", causing
            # ~15 s/job slow drains across thousands of cancelled
            # jobs in earlier rounds).
            #
            # ``_job_terminal_head`` falls back to ``sacct`` when
            # ``job.state`` returns empty — that arises when the job
            # has been reaped from squeue and a bare ``.result()``
            # would hang because submitit can't find a result pickle
            # (typical after TIMEOUT, where the worker is killed
            # mid-flush). Sacct yields the recorded historical state
            # so the fast-fail predicate still fires.
            head = _job_terminal_head(job)
            if head in {"CANCELLED", "TIMEOUT", "NODE_FAIL", "FAILED", "OUT_OF_MEMORY", "PREEMPTED"}:
                logger.warning("  FAST_FAIL: %s -> state=%s", label, head)
                if conv_key is not None and conv_key in conv_jobs:
                    conv_head = _job_terminal_head(conv_jobs[conv_key])
                    if conv_head in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}:
                        dep_failed += 1
                        continue
                failed += 1
                continue
            try:
                job.result()
                done += 1
            except Exception as e:
                upstream_failed = False
                if conv_key is not None and conv_key in conv_jobs:
                    try:
                        conv_jobs[conv_key].result()
                    except Exception:
                        upstream_failed = True
                if upstream_failed:
                    logger.warning(
                        "  DEP_FAILED: %s -> upstream conversion %s (job %s) failed",
                        label, conv_key, conv_jobs[conv_key].job_id,
                    )
                    dep_failed += 1
                else:
                    logger.error("  FAILED: %s -> %s", label, e)
                    failed += 1

        # Surface conversion failures top-level. Without this, a conv crash
        # is only visible indirectly through `dep_failed` counts on its
        # dependents — operators have no top-level signal of *which*
        # conversion failed. By the time we get here every bench job is
        # terminal, so its upstream conv is terminal too; .result() is
        # cached and won't block.
        conv_done, conv_failed = 0, 0
        for conv_key, conv_job in conv_jobs.items():
            try:
                conv_job.result()
                conv_done += 1
            except Exception as e:
                logger.error(
                    "  CONV_FAILED: %s / %s (job %s) -> %s",
                    conv_key[0], conv_key[1], conv_job.job_id, e,
                )
                conv_failed += 1

        elapsed = time.perf_counter() - t0
        summary = f"{done} succeeded, {failed} failed"
        if dep_failed:
            summary += f", {dep_failed} dep_failed"
        if conv_failed:
            summary += f" (+{conv_failed} conversion failures)"
        elif conv_jobs:
            summary += f" (+{conv_done} conversions OK)"
        logger.info("All done: %s in %.1fs (wall)", summary, elapsed)

        # Auto-run the regression gate when a canonical baseline exists
        # (Phase I.6). Non-fatal — reports the gate outcome and returns
        # normally; use the gate's own exit code elsewhere if blocking is
        # desired (the `gate_candidate.py` wrapper does this).
        if getattr(args, "auto_diff", True):
            _maybe_auto_diff(run_id)


def _maybe_auto_diff(run_id: str) -> None:
    """Invoke compare_against_baseline.py at the end of a run_parallel run.

    Best-effort — a missing canonical baseline is treated as a "not wired
    yet" signal rather than an error, since first-run fleets won't have a
    baseline to compare against.
    """
    import subprocess
    baselines_dir = LOGS_DIR.parent / "results" / "baselines"
    latest = baselines_dir / "LATEST"
    if not (latest.is_symlink() or latest.is_file()):
        logger.info(
            "Auto-diff skipped: no canonical baseline at %s (promote one "
            "with scripts/promote_baseline.py to enable).",
            latest,
        )
        return
    # The current snapshot for this run isn't a capture_baseline.py tree —
    # auto-diff works on the *last* captured snapshot. Operators who want a
    # scored PR should use scripts/gate_candidate.py instead.
    logger.info(
        "Auto-diff: run_id=%s — use scripts/gate_candidate.py to compare "
        "against baselines/LATEST; raw results landed in results/raw/.",
        run_id,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCX Benchmark Suite — Parallel SLURM Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--benchmarks", nargs="+", metavar="NAME",
                        help=f"Benchmarks to run. Default: all. Available: {BENCHMARK_NAMES}")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="Datasets to benchmark. Default: all available on disk.")
    parser.add_argument("--formats", nargs="+", metavar="KEY",
                        help="Format keys. Default: all primary formats.")
    parser.add_argument("--include-additional", action="store_true",
                        help="Include additional formats (BPCells, Parquet).")
    parser.add_argument("--include-accel", action="store_true",
                        help="Include accelerator-variant formats (accel_pca__*, "
                             "accel_knn__*, etc.) registered via "
                             "config.accel_formats(). Auto-enabled when "
                             "--benchmarks names any accel_* benchmark or "
                             "--formats names any accel_* key.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Drop accel formats whose key matches *_gpu* "
                             "(e.g. accel_pca__pyscx_gpu_rand_hh, "
                             "accel_pca__rapids_singlecell_gpu). Use on CPU-only hosts or "
                             "when validating CPU-only changes on a GPU box. "
                             "CPU accel variants and format benchmarks are "
                             "unaffected.")
    parser.add_argument("--cold-cache", action="store_true",
                        help="Drop OS caches between runs.")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip conversion phase (assume files exist).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert even if output files exist.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be submitted without running.")

    # SLURM parameters
    parser.add_argument("--partition", default="cpu_preemptible",
                        help="SLURM partition (default: cpu_preemptible)")
    parser.add_argument("--cpus", type=int, default=16,
                        help="CPUs per task (default: 16)")
    parser.add_argument("--mem-gb", type=int, default=8,
                        help="Memory floor in GB per job (default: 8). "
                             "Each job is sized via estimate_memory_gb() based on "
                             "(benchmark, dataset, format); --mem-gb is the lower bound.")
    parser.add_argument("--timeout", type=int, default=240,
                        help="Timeout in minutes per job (default: 240 — "
                             "used as a floor; per-job timeout is sized by "
                             "estimate_time_minutes() × --scale-factor).")
    parser.add_argument(
        "--scale-factor", type=float, default=1.0,
        help="Multiplier applied to per-job memory AND time estimates "
             "(Phase I.5). Use <1.0 (e.g. 0.9) on well-characterized CI, "
             ">1.0 (e.g. 1.3) for conservative operator runs on noisy "
             "clusters. Clamped by MEM_CEILING_GB for memory.",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip the pre-submit runner contract check (Phase I.8).",
    )
    parser.add_argument(
        "--max-pending-jobs", type=int, default=75,
        help="Throttle SLURM submission to keep total queued (PD+R) "
             "jobs at or below this cap. Required on Chimera, where "
             "QOSMaxSubmitJobPerUserLimit aborts sbatch — observed "
             "rejections at queue depth ~80–110 in May 2026 runs (the "
             "documented ~500 limit doesn't reflect what the scheduler "
             "actually enforces). The launcher polls `squeue -u $USER` "
             "between submissions and sleeps when the cap is reached, "
             "resuming as jobs land or fail; before each cohort array it "
             "reserves headroom for the whole cohort, and the cohort retry "
             "loop absorbs any race-window overshoot. The benchmark phase "
             "is ALWAYS throttled: values <= 0 are floored to 75 (an "
             "unthrottled bench phase trips QOSMaxSubmitJobPerUserLimit on "
             "Chimera). 0 disables throttling only for the lighter "
             "conversion phase. Default: 75.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
