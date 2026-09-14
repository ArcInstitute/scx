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
    """Split time-to-first-batch out, and describe **only** the steady state.

    Returns ``ttfb_s`` (the first ``next()``), ``data_wait_fraction_steady``,
    ``n_steady_steps``, and the steady-state wait distribution
    ``p50_ms`` / ``p95_ms`` / ``p99_ms`` / ``max_ms``.

    **The percentiles are computed here, on the post-startup slice, rather than
    left to the caller.** Both emitters previously called
    ``wait_percentiles(waits_s)`` on the *full* list beside this function, so
    the published `batch_wait_ms_max` was simply time-to-first-batch restated in
    milliseconds — measured on census_1m: `ttfb_s` 1.933 against `max_ms`
    1932.967, and 1.9831 against 1983.115. `p99` collapsed onto it too whenever
    ``n <= ~101`` (nearest rank puts 0.99 at the last index), which is exactly
    the short-epoch R3 case. That defeated the purpose of the tail metrics while
    looking like a tail measurement.

    Returning the percentiles from the same call that removes the startup is
    what makes the two impossible to get out of step; `wait_percentiles` stays
    public for callers that genuinely want the all-steps view, and nothing in
    this repo publishes that.

    ``None`` for the fraction and the percentiles when there is no steady state
    to describe: fewer than two steps, or a startup that consumed the whole
    timed region.
    """
    empty = {
        "ttfb_s": None, "data_wait_fraction_steady": None, "n_steady_steps": 0,
        "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None,
    }
    if not waits_s:
        return empty
    ttfb = waits_s[0]
    rest = waits_s[1:]
    if not rest:
        return {**empty, "ttfb_s": ttfb}
    return {
        "ttfb_s": ttfb,
        "data_wait_fraction_steady": data_wait_fraction(rest, wall_s - ttfb),
        "n_steady_steps": len(rest),
        **wait_percentiles(rest),
    }
