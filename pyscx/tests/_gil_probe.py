"""Measurement helpers for pyscx's GIL-release behaviour.

Two things are measured here, and they are different questions:

1. **Does the op release the GIL?** — `largest_gap_during`. A monitor thread
   stamps `perf_counter()` in a tight loop and the answer is the largest gap
   between consecutive stamps. This is a *property* measurement rather than a
   throughput ratio: a "4 threads finish in under 4x the single-threaded time"
   test gets flaky the moment the machine is loaded, which is exactly the
   condition under which CI runs. A loaded host delays the monitor under both
   behaviours, so the contrast (~1.0 of the op vs ~0.01) survives it.

2. **Does the op read a snapshot, or the caller's live buffer?** —
   `run_with_mutator`. A second thread rewrites the input while the op runs.
   That is a race, so a naive "mutate and assert the answer is right" test can
   pass on broken code purely by timing luck. `Probe` therefore reports enough
   for the caller to tell a real pass from a missed reproduction: how long the
   op took, how many mutations landed strictly inside the op window, and
   whether the GIL was released at all. A test that cannot confirm all three
   must skip, not pass.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


def largest_gap_during(op) -> tuple[float, float]:
    """Run `op`, returning `(op_duration, largest monitor-thread stall)`."""
    stamps: list[float] = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    # Let the monitor establish a baseline cadence before the op starts.
    time.sleep(0.02)
    start = time.perf_counter()
    op()
    duration = time.perf_counter() - start
    stop.set()
    t.join(timeout=5)

    return duration, _largest_gap(stamps, start)


def _largest_gap(stamps: list[float], start: float) -> float:
    during = [s for s in stamps if s >= start]
    if len(during) < 2:
        # The monitor produced no samples inside the window at all — that is
        # itself the failure the measurement is looking for.
        return time.perf_counter() - start
    gaps = [b - a for a, b in zip(during, during[1:])]
    # The stall that matters is measured from the last pre-op stamp, so a
    # monitor frozen for the entire op is caught rather than showing zero gaps.
    before = [s for s in stamps if s < start]
    leading = during[0] - (before[-1] if before else start)
    return max(max(gaps), leading)


@dataclass
class Probe:
    """What one mutate-during-op run actually managed to measure."""

    result: object
    duration: float
    #: Mutations that completed strictly inside the op's `[start, end]` window.
    writes_during: int
    #: Largest monitor stall; ~the whole duration means the GIL was never released.
    largest_gap: float


def run_with_mutator(op, mutate) -> Probe:
    """Run `op` while another thread calls `mutate()` in a loop.

    `mutate` must be cheap and idempotent — it is called repeatedly for the
    whole duration of `op`, which is what makes the reproduction reliable: the
    buffer is corrupt for essentially all of the op rather than at one guessed
    instant. It must only ever write **in-range** values; writing an
    out-of-range column index into a live buffer is undefined behaviour on a
    build that still reads it, and would take down the test process instead of
    reporting a failure.
    """
    stamps: list[float] = []
    write_times: list[float] = []
    stop = threading.Event()
    started = threading.Event()

    def worker():
        started.wait()
        while not stop.is_set():
            mutate()
            write_times.append(time.perf_counter())

    def monitor():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.001)

    m = threading.Thread(target=monitor, daemon=True)
    w = threading.Thread(target=worker, daemon=True)
    m.start()
    w.start()
    time.sleep(0.02)

    start = time.perf_counter()
    started.set()
    result = op()
    end = time.perf_counter()
    stop.set()
    w.join(timeout=5)
    m.join(timeout=5)

    return Probe(
        result=result,
        duration=end - start,
        writes_during=sum(1 for t in write_times if start <= t <= end),
        largest_gap=_largest_gap(stamps, start),
    )
