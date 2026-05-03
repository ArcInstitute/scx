"""
Single source of truth for resident-set-size sampling in the comprehensive
benchmark suite (review §2.4).

Pre-consolidation, seven different sites reimplemented essentially the same
``/proc/self/statm``-based reader (with subtle drift — some fell back to
``ru_maxrss``, others returned ``0.0`` on failure, the
``parallel_write_scaling`` worker script even inlined a stringified copy
inside its ``textwrap.dedent`` heredoc). This module is the one place that
implementation lives. All six benchmark / runner sites now delegate to
``current_rss_mb()``; the worker subprocess imports it the same way it
imports the rest of ``benchmarks.comprehensive``.

Consolidating here also lets the §2.1 follow-up (true peak via a 100 ms
sampler thread, currently a TODO in ``runners/base.py::timed_run``) land
in one place rather than seven.
"""

from __future__ import annotations

import os
import resource

__all__ = ["current_rss_mb"]


def current_rss_mb() -> float:
    """Return the *current* resident set size in megabytes.

    Linux: read ``/proc/self/statm`` field 1 (resident pages) and convert
    via ``SC_PAGE_SIZE``. Unlike ``getrusage().ru_maxrss``, this is the
    instantaneous RSS — not the process-lifetime high-water mark — so
    per-operation measurements are not inflated by earlier work in the
    same process.

    Non-Linux / read-failure: fall back to
    ``getrusage(RUSAGE_SELF).ru_maxrss / 1024.0``. The high-water mark
    is a coarser quantity than instantaneous RSS, but it's better than
    silently returning ``0.0`` (one of the pre-consolidation drift
    points called out in §2.4).
    """
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
