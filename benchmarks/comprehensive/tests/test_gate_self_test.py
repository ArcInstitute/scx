"""
Phase G.5 — regression-gate self-test.

Hermetic: writes synthetic baseline + current ``summary.json`` trees and
runs ``compare_against_baseline.py --gate`` in a subprocess. Verifies:

  1. A 15% timing regression causes a non-zero exit under ``--gate``.
  2. Adding a matching justification markdown flips the exit code to 0.
  3. An expired justification does NOT suppress the regression.
  4. A disappearing benchmark (present in baseline, missing in current)
     is flagged as a regression under ``--gate``.
  5. An absolute-floor violation from ``--thresholds`` fails the gate
     even when the relative diff is clean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GATE_SCRIPT = (
    PROJECT_ROOT / "benchmarks" / "comprehensive" / "scripts"
    / "compare_against_baseline.py"
)


def _write_summary(dirpath: Path, rows: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "summary.json").write_text(json.dumps({
        "snapshot_name": dirpath.name,
        "tier": "small",
        "rows": rows,
    }))


def _row(wall: float, rss: float = 100.0, size: int = 1000) -> dict:
    return {
        "median_wall_s": wall,
        "peak_rss_mb_median": rss,
        "file_size_bytes": size,
    }


def _run_gate(*extra: str, baseline: Path, current: Path) -> subprocess.CompletedProcess:
    """Run the gate hermetically.

    Auto-defaults for ``--thresholds`` and ``--justifications`` resolve to
    the committed repo paths in the normal on-demand workflow; tests pin
    both to empty sources inside the test tmp_path unless a case overrides
    them, so no committed thresholds bleed into hermetic behavior.
    """
    empty_just = baseline.parent / "_test_empty_justifications"
    empty_thresh = baseline.parent / "_test_empty_thresholds.yaml"
    empty_just.mkdir(exist_ok=True)
    if not empty_thresh.exists():
        empty_thresh.write_text("absolute_floors: []\n")
    flags = list(extra)
    if "--justifications" not in flags:
        flags += ["--justifications", str(empty_just)]
    if "--thresholds" not in flags:
        flags += ["--thresholds", str(empty_thresh)]
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--baseline", str(baseline),
        "--current", str(current),
        "--gate",
        *flags,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_slowdown_fails_gate(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.0)})
    _write_summary(cur, {"read_full__scx_auto__pbmc3k": _row(1.15)})

    result = _run_gate(baseline=base, current=cur)
    assert result.returncode != 0, (
        f"gate must fail on 15% slowdown; stdout={result.stdout!r}"
    )
    assert "read_full" in result.stdout
    assert "scx_auto" in result.stdout


def test_justification_flips_gate_to_pass(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    just = tmp_path / "just"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.0)})
    _write_summary(cur, {"read_full__scx_auto__pbmc3k": _row(1.15)})
    just.mkdir()
    (just / "issue_1234.md").write_text(
        "---\n"
        "triples:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "reason: Accepted regression for upstream HTTP/2 bump.\n"
        "---\n"
        "Prose explanation here.\n"
    )

    result = _run_gate(
        "--justifications", str(just),
        baseline=base, current=cur,
    )
    assert result.returncode == 0, (
        f"gate must pass with matching justification; stdout={result.stdout!r}"
    )


def test_expired_justification_does_not_suppress(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    just = tmp_path / "just"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.0)})
    _write_summary(cur, {"read_full__scx_auto__pbmc3k": _row(1.15)})
    just.mkdir()
    (just / "expired.md").write_text(
        "---\n"
        "triples:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "reason: Stale — should not suppress anymore.\n"
        "expires: 2000-01-01\n"
        "---\n"
    )

    result = _run_gate(
        "--justifications", str(just),
        baseline=base, current=cur,
    )
    assert result.returncode != 0


def test_disappearing_benchmark_flagged(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    _write_summary(base, {
        "read_full__scx_auto__pbmc3k": _row(1.0),
        "read_full__scx_auto__census_1m": _row(10.0),
    })
    # census_1m row disappears.
    _write_summary(cur, {"read_full__scx_auto__pbmc3k": _row(1.0)})

    result = _run_gate(baseline=base, current=cur)
    assert result.returncode != 0, (
        f"disappeared benchmark must fail gate; stdout={result.stdout!r}"
    )
    assert "DISAPPEARED" in result.stdout or "census_1m" in result.stdout


def test_absolute_floor_violation_fails_gate(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    thresh = tmp_path / "thresholds.yaml"
    # Identical summary rows — relative diff is clean.
    rows = {"cloud_push__scx_auto__pbmc3k": _row(1.0)}
    _write_summary(base, rows)
    _write_summary(cur, rows)
    # Raw JSON for the current run declares throughput below the floor.
    raw_dir = cur / "raw"
    raw_dir.mkdir()
    (raw_dir / "cloud_push__scx_auto__pbmc3k.json").write_text(json.dumps({
        "benchmark": "cloud_push",
        "format": "scx_auto",
        "dataset": "pbmc3k",
        "runs": [
            {"wall_s": 1.0, "extra": {"throughput_mbps": 12.3}},
            {"wall_s": 1.0, "extra": {"throughput_mbps": 11.8}},
        ],
    }))
    thresh.write_text(
        "absolute_floors:\n"
        "  - benchmark: cloud_push\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    metric: throughput_mbps\n"
        "    min: 50.0\n"
    )

    result = _run_gate(
        "--thresholds", str(thresh),
        baseline=base, current=cur,
    )
    assert result.returncode != 0, (
        f"absolute-floor violation must fail gate; stdout={result.stdout!r}"
    )
    assert "Absolute-floor" in result.stdout or "floor" in result.stdout.lower()
