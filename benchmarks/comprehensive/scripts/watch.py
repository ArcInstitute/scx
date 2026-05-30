#!/usr/bin/env python3
"""
Terminal dashboard for a running benchmark fleet (Phase I.6).

Tails ``benchmarks/comprehensive/logs/submitit/`` + reads
``run_manifest.json`` to show, in one screen:

  * Submitted triples (from manifest) vs result JSONs landed under
    ``results/raw/`` — catches missing-result failures where submitit
    exits cleanly but no JSON is written.
  * Per-SLURM-state counts (pending / running / completed / failed)
    parsed from ``.sh`` / ``.out`` log filenames submitit generates.
  * Most recent error per failed triple (last ~200 chars of the
    submitit err log).

``rich`` is preferred when available for live-refresh; falls back to a
one-shot snapshot printout when ``rich`` is absent. The snapshot mode is
enough for quick manifest checks; richness is a UX nicety.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    COMPREHENSIVE_DIR,
    RAW_RESULTS_DIR,
)

logger = logging.getLogger(__name__)

_LOGS_DIR = COMPREHENSIVE_DIR / "logs" / "submitit"
_BENCH_LOGS_DIR = _LOGS_DIR / "bench"


@dataclass
class TripleStatus:
    label: str
    job_id: str
    submitit_folder: str
    state: str = "unknown"   # pending / running / completed / failed / missing_result
    error: str = ""


@dataclass
class WatchSnapshot:
    run_id: str
    timestamp: str
    statuses: list[TripleStatus] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.statuses:
            out[s.state] = out.get(s.state, 0) + 1
        return out


def _read_manifest() -> dict | None:
    path = _LOGS_DIR / "run_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _result_exists(label: str) -> bool:
    """Heuristic: label is ``benchmark/dataset/format`` — result JSON is
    ``{benchmark}__{format}__{dataset}.json``. Match by label parts, not
    submitit's own filename."""
    try:
        bench, dataset, fmt = label.split("/", 2)
    except ValueError:
        return False
    return (RAW_RESULTS_DIR / f"{bench}__{fmt}__{dataset}.json").exists()


def _state_from_submitit(folder: Path, job_id: str) -> tuple[str, str]:
    """Inspect submitit's per-job folder for completion markers.

    Submitit names per-task files ``<slurm_job_id>_<task_idx>_*`` where
    ``task_idx`` is the index inside the submitit "job" (always ``0`` for
    single-process benchmarks). This holds for **both** individual jobs
    (``2306028_0_result.pkl``) and SLURM array tasks
    (``2374101_0_0_result.pkl`` for array ``2374101`` task ``0``) — the
    array-task ID is just part of the SLURM job ID. We always append
    ``_0``.
    """
    if not folder.is_dir():
        return "missing_folder", ""

    result_pkl = folder / f"{job_id}_0_result.pkl"
    log_err = folder / f"{job_id}_0_log.err"
    log_out = folder / f"{job_id}_0_log.out"
    if result_pkl.exists():
        return "completed", ""
    if log_err.exists() and log_err.stat().st_size > 0:
        try:
            tail = log_err.read_text(errors="replace")[-400:]
        except OSError:
            tail = ""
        return "failed", tail.strip().splitlines()[-1] if tail.strip() else ""
    if log_out.exists():
        return "running", ""
    return "pending", ""


def build_snapshot() -> WatchSnapshot | None:
    manifest = _read_manifest()
    if manifest is None:
        return None
    statuses: list[TripleStatus] = []
    for entry in manifest.get("submitted", []):
        label = entry["label"]
        job_id = str(entry["job_id"])
        folder = Path(entry["submitit_folder"])
        state, err = _state_from_submitit(folder, job_id)
        if state == "completed" and not _result_exists(label):
            # Submitit thinks it's done but no JSON landed — flag it.
            state = "missing_result"
            err = "submitit completed but no result JSON in results/raw/"
        statuses.append(TripleStatus(
            label=label, job_id=job_id,
            submitit_folder=str(folder),
            state=state, error=err,
        ))
    return WatchSnapshot(
        run_id=manifest.get("run_id", "unknown"),
        timestamp=manifest.get("timestamp", ""),
        statuses=statuses,
    )


def _print_snapshot(snap: WatchSnapshot) -> None:
    counts = snap.counts()
    print(f"=== run_id={snap.run_id} submitted={snap.timestamp} ===")
    print("Status counts:")
    for state in ("pending", "running", "completed", "missing_result",
                  "failed", "missing_folder", "unknown"):
        n = counts.get(state, 0)
        if n:
            print(f"  {state:>18}: {n}")
    # Last error per failed triple.
    failures = [s for s in snap.statuses if s.state in ("failed", "missing_result")]
    if failures:
        print("\nRecent failures:")
        for s in failures[-20:]:
            print(f"  [{s.state}] {s.label}")
            if s.error:
                print(f"      {s.error[:200]}")


def _watch_rich(refresh_s: float) -> None:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    console = Console()

    def render() -> Table:
        snap = build_snapshot()
        if snap is None:
            return Table(title="(no run_manifest.json — run run_parallel.py first)")
        table = Table(
            title=f"SCX bench run {snap.run_id} — submitted {snap.timestamp}",
            show_lines=False,
        )
        table.add_column("State")
        table.add_column("Count", justify="right")
        for state, count in sorted(snap.counts().items(), key=lambda kv: -kv[1]):
            table.add_row(state, str(count))
        return table

    with Live(render(), refresh_per_second=1 / refresh_s, console=console) as live:
        try:
            while True:
                time.sleep(refresh_s)
                live.update(render())
        except KeyboardInterrupt:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", type=float, default=5.0,
        help="Live-refresh interval in seconds (rich mode only). Default 5.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print one snapshot and exit (no live refresh).",
    )
    args = parser.parse_args(argv)

    snap = build_snapshot()
    if snap is None:
        print(
            f"No run_manifest.json at {_LOGS_DIR / 'run_manifest.json'}. "
            f"Start a run with run_parallel.py first.",
            file=sys.stderr,
        )
        return 2

    if args.once:
        _print_snapshot(snap)
        return 0

    try:
        _watch_rich(args.refresh)
    except ImportError:
        logger.info("rich not installed — falling back to one-shot snapshot")
        _print_snapshot(snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
