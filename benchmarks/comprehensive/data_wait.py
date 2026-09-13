"""Consumer-observed data-wait accounting, shared by the loader benchmarks.

The phase-0 gate for the ML-loader work asks one question per access regime:
what fraction of a training step is spent waiting on data? Tier-3 loader items
are funded only where that fraction is material, because a decoder speedup `s`
applied to an exposed fraction `p` is worth `1 / [(1 - p) + p/s]` and no more —
1.006x at the one `p` on record (STATE3, 0.006).

Its own module rather than a helper inside `benchmarks/ml_loader.py`, because
`index_plan` needs the same functions and importing `ml_loader` for them would
drag every competitor-loader availability probe (soma_ml, scdataloader,
annbatch, cellstream, ...) into `index_plan`'s import path.

`steady_state_wait` is the one to quote: see its docstring for the measured
case where the all-steps fraction and the same run's p95 said opposite things.
"""

from __future__ import annotations


def data_wait_fraction(waits_s: list[float], wall_s: float) -> float | None:
    """Exposed data wait as a fraction of the timed region, or ``None``.

    ``waits_s`` is one ``next(iterator)`` duration per step; ``wall_s`` is the
    wall clock of the region those steps ran in.

    ``None`` — never ``0.0`` — when there were no steps or no timed wall. The
    distinction is the whole point of the metric. ``compare_against_baseline``'s
    ``_load_current_raw_metric`` *skips* ``None`` and medians what is left, so a
    ``0.0`` placeholder is indistinguishable from a real measurement of a loader
    that never made its consumer wait — which is the conclusion this metric
    exists to test, not one it may assume.

    Clamped to ``[0, 1]``. On the real path ``sum(waits) <= wall_s`` holds by
    construction (the wall is measured with the device synchronised *inside*
    the timer), but a value published into a table read as a percentage must
    not be able to exceed 1 because a clock hiccuped.
    """
    if not waits_s or wall_s <= 0:
        return None
    return min(1.0, max(0.0, sum(waits_s) / wall_s))


def wait_percentiles(waits_s: list[float]) -> dict[str, float | None]:
    """``{p50_ms, p95_ms, p99_ms, max_ms}`` of per-step wait, or all ``None``.

    Reported beside the fraction because a middle statistic alone cannot
    distinguish a loader that is uniformly a little behind from one that is
    fine except for a shard-boundary stall — and only the second is worth
    optimising.

    **The tail is why p99 and the max are here rather than p50/p95 alone.** On
    census_1m the measured p50 is 13 microseconds and p95 799 microseconds,
    while the steady-state fraction is 0.76 — roughly 10 s of wait spread over
    ~976 steps. Those reconcile only if a few tens of steps cost ~200 ms each.
    Stopping at p95 would report "sub-millisecond batches" underneath a number
    saying three quarters of the step budget is data wait, and the two would
    look like a contradiction rather than a description of a heavy tail.
    """
    keys = ("p50_ms", "p95_ms", "p99_ms", "max_ms")
    if not waits_s:
        return dict.fromkeys(keys, None)
    ms = sorted(w * 1000.0 for w in waits_s)

    def _p(q: float) -> float:
        # Nearest-rank: interpolation would invent a latency no step had, and
        # the point of these is to name a step that actually happened.
        idx = min(len(ms) - 1, max(0, int(round(q * (len(ms) - 1)))))
        return round(ms[idx], 3)

    return {
        "p50_ms": _p(0.50),
        "p95_ms": _p(0.95),
        "p99_ms": _p(0.99),
        "max_ms": round(ms[-1], 3),
    }


def steady_state_wait(waits_s: list[float], wall_s: float) -> dict[str, float | int | None]:
    """Split time-to-first-batch out of the wait, and report the rest.

    Returns ``ttfb_s`` (the first ``next()``), ``data_wait_fraction_steady``
    (every later step's wait over the region after the first batch arrived),
    and ``n_steady_steps``.

    **This, not the all-steps fraction, is the number that answers "is the
    loader on the critical path".** The first ``next()`` pays tokio spin-up
    and the first shard decode once per epoch; every later one is served from
    a running pipeline. Folding the two together makes the answer a function
    of how long the benchmark's epoch happens to be.

    That is not a hypothetical. On a 98-step `gpu_train` epoch over
    tabula_sapiens_100k the all-steps fraction read **0.85** while the same
    run's per-batch p50 and p95 were **13 and 15 microseconds** — 2.04 s of
    "wait" in a 2.4 s region, essentially all of it in step 1. Published
    unqualified it would have said the loader is on the critical path 85 % of
    the time, contradicting both its own p95 and the D0 profile's 4 microsecond
    send-wait, and would have argued for throughput work that the measurement
    actually rules out.

    ``None`` for the fraction when there is no steady state to describe: fewer
    than two steps, or a startup that consumed the whole timed region.
    """
    if not waits_s:
        return {"ttfb_s": None, "data_wait_fraction_steady": None, "n_steady_steps": 0}
    ttfb = waits_s[0]
    rest = waits_s[1:]
    if not rest:
        return {"ttfb_s": ttfb, "data_wait_fraction_steady": None, "n_steady_steps": 0}
    steady_wall = wall_s - ttfb
    frac = data_wait_fraction(rest, steady_wall)
    return {
        "ttfb_s": ttfb,
        "data_wait_fraction_steady": frac,
        "n_steady_steps": len(rest),
    }
