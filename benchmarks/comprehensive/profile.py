"""
Opt-in hot-path profiling hooks (Phase I.9).

Honors the ``BENCHMARK_PROFILE`` env var:

  * ``off`` (default) — the context manager is a no-op.
  * ``flamegraph`` — wrap the block in ``py-spy record --subprocess``
    and emit an interactive SVG flamegraph alongside the result JSON.
  * ``perf`` — wrap the block in ``perf record --call-graph dwarf``
    and emit ``perf.data`` next to the result.

Output paths land under
``benchmarks/comprehensive/results/profiles/<run_id>/<label>.{svg,perf}``
so the dashboard's trend view can cross-reference a profile with the
`(benchmark, format, dataset)` triple that produced it.

Usage from a benchmark module:

    from benchmarks.comprehensive.profile import profile_region

    with profile_region(label=f"{benchmark}__{format_key}__{dataset.name}"):
        timing = runner.read_cloud(url)

If profiling isn't configured, the ``with`` block adds one attribute
lookup of overhead — safe to leave always-on in benchmark source.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.comprehensive.config import COMPREHENSIVE_DIR
from benchmarks.comprehensive.provenance import current_run_id

logger = logging.getLogger(__name__)

_PROFILES_DIR = COMPREHENSIVE_DIR / "results" / "profiles"


def _mode() -> str:
    return os.environ.get("BENCHMARK_PROFILE", "off").strip().lower()


@contextmanager
def profile_region(label: str):
    """Context manager: wrap a timed region in the configured profiler.

    The profiler is launched as a subprocess attached to THIS process's
    PID, so the inner Python work is recorded without needing to fork.
    On exit, the profiler is signalled to stop and flush its output.
    """
    mode = _mode()
    if mode in ("", "off", "none"):
        yield
        return

    out_root = _PROFILES_DIR / current_run_id()
    out_root.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "__").replace(" ", "_")
    pid = os.getpid()

    if mode == "flamegraph":
        output = out_root / f"{safe_label}.svg"
        proc = _start_py_spy(pid, output)
    elif mode == "perf":
        output = out_root / f"{safe_label}.perf"
        proc = _start_perf(pid, output)
    else:
        logger.warning(
            "Unknown BENCHMARK_PROFILE=%r; expected off|flamegraph|perf. "
            "Running without profile.", mode,
        )
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        _stop_profiler(proc, output, mode, wall)


def _start_py_spy(pid: int, output: Path) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(
            ["py-spy", "record", "--pid", str(pid),
             "--subprocesses", "--format", "flamegraph",
             "--output", str(output)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("py-spy not on PATH; skipping flamegraph profile.")
        return None


def _start_perf(pid: int, output: Path) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(
            ["perf", "record", "--call-graph", "dwarf",
             "--pid", str(pid), "--output", str(output)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("perf not on PATH; skipping perf profile.")
        return None


def _stop_profiler(
    proc: subprocess.Popen | None, output: Path, mode: str, wall_s: float,
) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("profiler shutdown issue (%s): %s", mode, exc)
        return
    if output.exists() and output.stat().st_size > 0:
        logger.info(
            "Profile (%s, %.2fs wall) written to %s", mode, wall_s, output,
        )
    else:
        logger.warning(
            "Profile output %s empty — likely a privilege (perf_event_paranoid) "
            "or dependency issue.", output,
        )
