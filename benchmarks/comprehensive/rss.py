"""
Single source of truth for resident-set-size sampling in the comprehensive
benchmark suite.

Every benchmark and runner that needs current RSS imports
``current_rss_mb()`` from here — no inlined ``/proc/self/statm`` reads
elsewhere. The ``parallel_write_scaling`` worker subprocess imports it
the same way it imports the rest of ``benchmarks.comprehensive``.

Centralising here also keeps a future "true peak via sampler thread"
upgrade to ``runners/base.py::timed_run`` confined to one file rather
than spread across the call sites.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time

__all__ = ["current_rss_mb", "PeakRssSampler"]


def current_rss_mb() -> float:
    """Return the *current* resident set size in megabytes.

    Linux: read ``/proc/self/statm`` field 1 (resident pages) and convert
    via ``SC_PAGE_SIZE``. Unlike ``getrusage().ru_maxrss``, this is the
    instantaneous RSS — not the process-lifetime high-water mark — so
    per-operation measurements are not inflated by earlier work in the
    same process.

    Non-Linux / read-failure: fall back to
    ``getrusage(RUSAGE_SELF).ru_maxrss``. The high-water mark is a
    coarser quantity than instantaneous RSS, but it's better than
    silently returning ``0.0``. ``ru_maxrss`` units differ by platform:
    Linux reports kilobytes, macOS (Darwin) reports bytes.
    """
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return ru_maxrss / (1024 * 1024)
        return ru_maxrss / 1024.0


class PeakRssSampler:
    """Background high-water-mark RSS sampler for a timed region.

    ``current_rss_mb()`` is instantaneous, so a single call around an op that
    allocates *and frees* a large transient buffer (e.g. a full-matrix
    materialization) misses the spike. This polls RSS on a daemon thread so the
    true in-region peak is captured — needed for the out-of-core boundary bench,
    where a competing format's ``read_full`` is a single call with no loop to
    sample inside.

    Usage::

        with PeakRssSampler() as s:
            do_work()
        peak = s.peak_mb    # max RSS observed while the block ran
    """

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        # Seed with the entry RSS so peak is never below the starting point.
        self.peak_mb = max(self.peak_mb, current_rss_mb())
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, current_rss_mb())
            self._stop.wait(self.interval_s)
        # Final reading after stop so a spike right before exit isn't missed.
        self.peak_mb = max(self.peak_mb, current_rss_mb())

    def __enter__(self) -> "PeakRssSampler":
        self.peak_mb = current_rss_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
