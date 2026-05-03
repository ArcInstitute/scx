"""
RSS-reader consolidation.

Pre-consolidation, seven sites independently re-implemented essentially
the same ``/proc/self/statm`` reader. This test pins the consolidation
in place: every benchmark / runner site that needs current RSS must
route through ``benchmarks.comprehensive.rss.current_rss_mb`` so any
future "true peak via sampler thread" upgrade lands in exactly one
file.

The tests cover:

  1. ``current_rss_mb()`` returns a positive float on Linux. (Sanity —
     guards against the consolidation accidentally returning 0.0 like
     two of the pre-consolidation copies did on read failure.)
  2. The five named module-level / staticmethod RSS readers are all
     ``current_rss_mb`` (or trivially delegate to it). If a future PR
     reintroduces a per-module copy, this test fails fast.
  3. The ``parallel_write_scaling`` worker script no longer carries an
     inline copy of the reader — it imports from ``rss`` instead.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.comprehensive import rss
from benchmarks.comprehensive.benchmarks import (
    accel_pca,
    fragment_ops,
    memory,
    ml_loader,
)
from benchmarks.comprehensive.benchmarks import parallel_write_scaling
from benchmarks.comprehensive.runners.accel_runner import AcceleratorRunner
from benchmarks.comprehensive.runners.base import FormatRunner


def test_current_rss_mb_returns_positive_float() -> None:
    val = rss.current_rss_mb()
    assert isinstance(val, float)
    # The pytest process itself has non-trivial RSS; a 0.0 here would
    # mean we're hitting the silent-failure branch.
    assert val > 0.0


def test_format_runner_get_rss_delegates() -> None:
    """``FormatRunner._get_rss_mb`` must agree with the canonical helper."""
    a = FormatRunner._get_rss_mb()
    b = rss.current_rss_mb()
    # Two adjacent calls — RSS may drift by a few KB. Allow 5 MB slack.
    assert abs(a - b) < 5.0


def test_accelerator_runner_get_rss_delegates() -> None:
    a = AcceleratorRunner.get_rss_mb()
    b = rss.current_rss_mb()
    assert abs(a - b) < 5.0


def test_benchmark_modules_reuse_canonical_helper() -> None:
    """Each benchmark module's local ``_current_rss_mb`` / ``_get_rss_mb``
    name must be the canonical helper (not a re-implementation).

    Pinning identity (``is``) catches a future PR re-introducing a local
    copy: a hand-rolled function that happens to return the same number
    would have a distinct identity from ``rss.current_rss_mb`` and
    fail this assertion.
    """
    canonical = rss.current_rss_mb
    assert memory._current_rss_mb is canonical
    assert ml_loader._current_rss_mb is canonical
    assert fragment_ops._current_rss_mb is canonical
    assert accel_pca._get_rss_mb is canonical


def test_parallel_write_scaling_worker_script_imports_rss_helper() -> None:
    """The ``_WORKER_SCRIPT`` heredoc must import the canonical helper
    rather than inlining its own ``/proc/self/statm`` body.

    Before consolidation, the heredoc carried a stringified copy of the
    reader that no static analysis or refactor could keep in sync with
    the runner classes' implementations. Asserting on the script body
    directly pins this regression closed.
    """
    script = parallel_write_scaling._WORKER_SCRIPT
    assert "from benchmarks.comprehensive.rss import current_rss_mb" in script
    # And the inline /proc/self/statm read is gone — no copy survives in
    # the heredoc.
    assert "/proc/self/statm" not in script
