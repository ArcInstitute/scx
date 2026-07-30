"""
N-concurrent-rank harness for the data-load Phase-0 premise gate (P-1(c)).

The question this answers: **does per-rank loader cost and page-cache pressure
scale**, i.e. is the loader a *shared* bottleneck once several DDP ranks read the
same files on one node? A single-process throughput number cannot say — contention
on the OS page cache, the shared filesystem, and the node's memory bandwidth only
appears with concurrent readers. See ``docs/performance.md`` § "Out-of-core loader
— cold-cache measurements and the P-1 premise gate" for the results and for the
stated limit (one node is not multi-node DDP).

This module runs the same per-rank workload in ``n_ranks`` child processes and
reports each rank's outcome, so a benchmark can compute

    rank_scaling_efficiency = median(per-rank rate at N) / (rate at N=1)

Efficiency ≈ 1 means the loader is not the shared bottleneck (each rank is as
fast as it would be alone); efficiency ≈ 1/N means the ranks are serialising on
a shared resource.

Two design points that are load-bearing, not stylistic:

* **``spawn``, not ``fork``.** pyscx's loader datasets carry a PID guard: a
  parent-constructed dataset used post-fork raises (see
  ``pyscx/tests/test_fork_safety.py``). Under ``spawn`` each child imports fresh
  and constructs its own dataset, which is the documented-safe pattern. The
  worker callable must therefore be a **module-level, picklable** function —
  a closure or lambda will fail to pickle.
* **A barrier before the timed region.** Without it the ranks stagger by their
  (large, variable) interpreter-startup cost and the "concurrent" window is
  partly serial, which flatters the efficiency ratio. Every child blocks on
  ``Barrier(n_ranks)`` and starts its clock only once all ranks are ready, so
  the timed regions genuinely overlap. Interpreter startup and imports sit
  *outside* the measurement, which is also why the ``n_ranks=1`` reference goes
  through the same child path: the two arms then differ only in concurrency.

Usage::

    # must be module-level so `spawn` can pickle it by reference
    def _gather_rank(rank: int, n_ranks: int, path: str) -> dict:
        import pyscx
        ds = pyscx.SparseCellSetDataset([path])   # constructed IN the child
        ...
        return {"n_sets": n_sets, "sets_per_sec": sps}

    one = run_ranks(1, _gather_rank, (path,))
    many = run_ranks(4, _gather_rank, (path,))
    eff = rank_efficiency(one, many, "sets_per_sec")
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import statistics
import time
import traceback
from typing import Any, Callable, Sequence

from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_n_ranks",
    "run_ranks",
    "rank_efficiency",
    "summarize_ranks",
    "N_RANKS_ENV",
    "DEFAULT_N_RANKS",
    "RECOMMENDED_N_RANKS",
]

N_RANKS_ENV = "SCX_BENCH_N_RANKS"
DEFAULT_N_RANKS = 1
"""**Off by default.** The rank arm costs ``(1 + N)`` extra workloads per run, so
turning it on unconditionally would silently multiply the wall time of every
existing ``cellset_gather`` / ``obs_open`` run — including ad-hoc ones that don't
want it. Callers skip the arm at N=1, so the pre-existing behaviour is unchanged
until someone asks for the measurement."""

RECOMMENDED_N_RANKS = 4
"""What the Phase-0 capture exports: 4 matches the common single-node DDP shape
(4 GPUs/node) while keeping the arm affordable."""

# A rank that hasn't reported within this many seconds past the slowest reporter
# is treated as hung rather than blocking the whole benchmark forever.
_JOIN_GRACE_S = 300.0


def resolve_n_ranks(default: int = DEFAULT_N_RANKS) -> int:
    """Rank count from ``SCX_BENCH_N_RANKS``, clamped to >= 1.

    Returns 1 when the knob is unset-and-``default``-is-1, or when it parses to
    something < 1. Callers skip the rank arm when this resolves to 1 — a
    "4 ranks vs 1 rank" ratio is meaningless at N=1.
    """
    raw = os.environ.get(N_RANKS_ENV, "").strip()
    if not raw:
        return max(1, default)
    try:
        return max(1, int(float(raw)))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", N_RANKS_ENV, raw, default)
        return max(1, default)


def _child_entry(
    queue: Any,
    barrier: Any,
    rank: int,
    n_ranks: int,
    fn: Callable[..., dict[str, Any]],
    args: tuple,
    kwargs: dict[str, Any],
) -> None:
    """Child-process wrapper: sync at the barrier, time ``fn``, report back.

    Module-level (not nested) so ``spawn`` can pickle it. Never raises into the
    child's exit status — a failure is reported as an ``error`` field so the
    parent can distinguish "this rank failed" from "this rank hung".
    """
    result: dict[str, Any] = {"rank": rank, "n_ranks": n_ranks}
    try:
        # Everything expensive (interpreter start, imports, and any setup inside
        # `fn` before it starts its own work) must not skew the overlap window.
        barrier.wait()
        with PeakRssSampler() as sampler:
            t0 = time.perf_counter()
            payload = fn(rank, n_ranks, *args, **kwargs)
            wall_s = time.perf_counter() - t0
        result["wall_s"] = wall_s
        result["peak_rss_mb"] = sampler.peak_mb
        if isinstance(payload, dict):
            result.update(payload)
    except Exception as e:  # noqa: BLE001 — must reach the parent, not the exit code
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=5)
    finally:
        try:
            queue.put(result)
        except Exception:  # noqa: BLE001 — parent may already be gone
            pass


def run_ranks(
    n_ranks: int,
    fn: Callable[..., dict[str, Any]],
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run ``fn(rank, n_ranks, *args, **kwargs)`` in ``n_ranks`` spawned children.

    ``fn`` must be importable by qualified name (module-level function) and
    return a dict of JSON-scalar metrics. Each returned dict is augmented with
    ``rank``, ``n_ranks``, ``wall_s`` and ``peak_rss_mb``; a rank that raised
    carries ``error`` (and ``traceback``) instead of timings.

    Results are returned sorted by rank. A rank that never reports is simply
    absent from the list — callers should check ``len(results) == n_ranks``
    before trusting an aggregate.
    """
    kwargs = dict(kwargs or {})
    n_ranks = max(1, int(n_ranks))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    # `timeout` on the Barrier itself guards the case where a child dies during
    # import: the survivors would otherwise block on wait() forever.
    barrier = ctx.Barrier(n_ranks, timeout=_JOIN_GRACE_S)

    procs = [
        ctx.Process(
            target=_child_entry,
            args=(queue, barrier, rank, n_ranks, fn, tuple(args), kwargs),
            daemon=False,
        )
        for rank in range(n_ranks)
    ]
    for p in procs:
        p.start()

    # Drain the queue BEFORE joining. A child that put()s a large payload blocks
    # in the feeder thread until the parent reads it, so join-then-drain
    # deadlocks on exactly the payloads we care about.
    results: list[dict[str, Any]] = []
    deadline = time.perf_counter() + _JOIN_GRACE_S
    while len(results) < n_ranks and time.perf_counter() < deadline:
        remaining = max(1.0, deadline - time.perf_counter())
        try:
            results.append(queue.get(timeout=remaining))
        except Exception:  # noqa: BLE001 — queue.Empty and friends
            break

    for p in procs:
        p.join(timeout=_JOIN_GRACE_S)
        if p.is_alive():
            logger.error("multirank: rank process %s did not exit; terminating", p.pid)
            p.terminate()
            p.join(timeout=10.0)

    if len(results) != n_ranks:
        logger.error(
            "multirank: only %d/%d ranks reported (n_ranks=%d)",
            len(results),
            n_ranks,
            n_ranks,
        )
    for r in results:
        if "error" in r:
            logger.error("multirank: rank %s failed: %s", r.get("rank"), r["error"])

    return sorted(results, key=lambda r: r.get("rank", 0))


def summarize_ranks(
    results: Sequence[dict[str, Any]], rate_key: str
) -> dict[str, Any] | None:
    """Aggregate + per-rank median of ``rate_key`` across successful ranks.

    Returns ``None`` when no rank reported a numeric ``rate_key`` — the caller
    then omits the arm rather than recording a zero that reads as a measurement.
    """
    rates = [
        float(r[rate_key])
        for r in results
        if "error" not in r and isinstance(r.get(rate_key), (int, float))
    ]
    if not rates:
        return None
    walls = [float(r["wall_s"]) for r in results if "error" not in r and "wall_s" in r]
    rss = [
        float(r["peak_rss_mb"])
        for r in results
        if "error" not in r and "peak_rss_mb" in r
    ]
    return {
        "n_ranks_reported": len(rates),
        # Sum, not mean: this is the node's aggregate throughput, the quantity a
        # DDP job actually gets.
        "aggregate": round(sum(rates), 4),
        "per_rank_median": round(statistics.median(rates), 4),
        "per_rank_min": round(min(rates), 4),
        "per_rank_max": round(max(rates), 4),
        # Slowest rank sets the step time in a synchronous-DDP job, so the max
        # wall is the operationally meaningful one.
        "max_wall_s": round(max(walls), 4) if walls else None,
        "total_peak_rss_mb": round(sum(rss), 1) if rss else None,
    }


def rank_efficiency(
    single: Sequence[dict[str, Any]],
    many: Sequence[dict[str, Any]],
    rate_key: str,
) -> float | None:
    """Per-rank rate at N ranks divided by the rate at 1 rank.

    ``1.0`` — a rank is as fast with N-1 siblings as alone: the loader is **not**
    a shared bottleneck. ``~1/N`` — the ranks serialise on a shared resource
    (page cache, filesystem, memory bandwidth), which *is* a positive P-1(c)
    finding. Returns ``None`` if either arm produced no usable rate.
    """
    one = summarize_ranks(single, rate_key)
    n = summarize_ranks(many, rate_key)
    if one is None or n is None or one["per_rank_median"] <= 0:
        return None
    return round(n["per_rank_median"] / one["per_rank_median"], 4)
