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
    """A module's local ``_current_rss_mb`` / ``_get_rss_mb`` name, **if it has
    one**, must be the canonical helper rather than a re-implementation.

    Pinning identity (``is``) catches a future PR re-introducing a local copy: a
    hand-rolled function that happens to return the same number would have a
    distinct identity from ``rss.current_rss_mb`` and fail this assertion.

    The presence of the name is deliberately *not* asserted. `fragment_ops` and
    `grouped_sort` dropped it entirely when they moved to `PeakRssSampler` —
    they no longer take an instantaneous reading at all, because the ops they
    time (a whole-file rewrite, a grouped convert) allocate and free a transient
    that is gone by the time a post-op sample lands. A module that stopped
    reading RSS satisfies "nobody re-implements the reader" more completely than
    one that routes through it, and the test that pinned the attribute's
    existence would have blocked exactly that improvement.

    `test_no_benchmark_module_reimplements_the_reader` below is the half that
    still has to hold universally.
    """
    canonical = rss.current_rss_mb
    for mod, attr in (
        (memory, "_current_rss_mb"),
        (ml_loader, "_current_rss_mb"),
        (accel_pca, "_get_rss_mb"),
        (fragment_ops, "_current_rss_mb"),
    ):
        got = getattr(mod, attr, None)
        if got is None:
            continue
        assert got is canonical, (
            f"{mod.__name__}.{attr} is not rss.current_rss_mb — a local "
            f"re-implementation is exactly what the consolidation removed"
        )

    # Guard the guard: if every module dropped the name, the loop above would
    # pass while checking nothing.
    assert memory._current_rss_mb is canonical
    assert accel_pca._get_rss_mb is canonical


def test_no_benchmark_module_reimplements_the_reader() -> None:
    """No benchmark module may read ``/proc/self/statm`` itself.

    This is the invariant the consolidation actually bought, and unlike the
    identity check above it holds for every module including the ones that no
    longer take an instantaneous reading at all. Seven sites had their own copy
    before consolidation, and two of them silently returned 0.0 on read failure.
    """
    bench_dir = (
        Path(__file__).resolve().parents[1] / "benchmarks"
    )
    assert bench_dir.is_dir(), f"{bench_dir} moved — retarget this test"

    import ast

    offenders = []
    for path in sorted(bench_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))

        # Docstrings are `ast.Constant` too, and `doublet_interop` now *explains*
        # that `current_rss_mb()` reads /proc/self/statm of the parent — a plain
        # text search flags that sentence. Same trap `test_floor_reachability`
        # records under `test_no_benchmark_passes_extra_as_a_dict`: a grep cannot
        # tell a call from a sentence. Exclude the docstring slot explicitly.
        docstrings = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value == "/proc/self/statm"
                    and id(node) not in docstrings):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"these benchmark modules read /proc/self/statm directly instead of "
        f"going through benchmarks.comprehensive.rss: {offenders}"
    )


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
