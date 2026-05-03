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

__all__ = ["current_rss_mb"]


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
