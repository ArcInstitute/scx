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
  6. A flakiness override raises the per-row tolerance, flipping a
     small-but-over-default regression to PASS while keeping bigger
     regressions on the same row failing.
  7. A flakiness override whose tolerance is at or below the global
     default emits a warning so the dead entry doesn't silently rot in
     the ledger.
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


def _row(
    wall: float,
    rss: float = 100.0,
    size: int = 1000,
    wall_iqr: float | None = None,
    n_runs: int | None = None,
) -> dict:
    out: dict = {
        "median_wall_s": wall,
        "peak_rss_mb_median": rss,
        "file_size_bytes": size,
    }
    # Omit wall_s_iqr / n_runs entirely when not set so the existing tests
    # exercise the gate's fallback-to-fixed-tolerance path on legacy
    # baselines that predate the variance-aware tolerance schema bump.
    if wall_iqr is not None:
        out["wall_s_iqr"] = wall_iqr
    if n_runs is not None:
        out["n_runs"] = n_runs
    return out


def _run_gate(*extra: str, baseline: Path, current: Path) -> subprocess.CompletedProcess:
    """Run the gate hermetically.

    Auto-defaults for ``--thresholds`` and ``--justifications`` resolve to
    the committed repo paths in the normal on-demand workflow; tests pin
    both to empty sources inside the test tmp_path unless a case overrides
    them, so no committed thresholds bleed into hermetic behavior.
    """
    empty_just = baseline.parent / "_test_empty_justifications"
    empty_thresh = baseline.parent / "_test_empty_thresholds.yaml"
    empty_flaky = baseline.parent / "_test_empty_flakiness"
    empty_just.mkdir(exist_ok=True)
    empty_flaky.mkdir(exist_ok=True)
    if not empty_thresh.exists():
        empty_thresh.write_text("absolute_floors: []\n")
    flags = list(extra)
    if "--justifications" not in flags:
        flags += ["--justifications", str(empty_just)]
    if "--thresholds" not in flags:
        flags += ["--thresholds", str(empty_thresh)]
    if "--flakiness" not in flags:
        flags += ["--flakiness", str(empty_flaky)]
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


# ---------------------------------------------------------------------------
# Loader / median / direction unit tests (hermetic, no subprocess).
# ---------------------------------------------------------------------------


def _import_gate_module():
    """Import the gate script as a module for unit-level tests.

    The module must be registered in ``sys.modules`` before execution
    because dataclass forward-reference resolution looks the module up
    via ``sys.modules[cls.__module__]``.
    """
    import importlib.util

    name = "compare_against_baseline"
    spec = importlib.util.spec_from_file_location(name, str(GATE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_load_current_raw_metric_true_median(tmp_path: Path) -> None:
    mod = _import_gate_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "cloud_push__scx_auto__ds.json").write_text(json.dumps({
        "runs": [
            {"extra": {"throughput_mbps": 1.0}},
            {"extra": {"throughput_mbps": 2.0}},
            {"extra": {"throughput_mbps": 3.0}},
            {"extra": {"throughput_mbps": 4.0}},
        ],
    }))
    value = mod._load_current_raw_metric(
        tmp_path, "cloud_push", "scx_auto", "ds", "throughput_mbps",
    )
    # Four values: the true median is 2.5, not values[len//2] == 3.0.
    assert value == pytest.approx(2.5)


def test_thresholds_yaml_rejects_missing_required_fields(tmp_path: Path) -> None:
    mod = _import_gate_module()
    thresh = tmp_path / "thresholds.yaml"
    thresh.write_text(
        "absolute_floors:\n"
        "  - benchmark: cloud_push\n"
        "    format: scx_auto\n"
        "    # missing dataset and metric\n"
        "    min: 50.0\n"
    )
    with pytest.raises(ValueError, match="missing fields"):
        mod._load_thresholds_yaml(thresh)


def test_thresholds_yaml_rejects_unknown_direction(tmp_path: Path) -> None:
    mod = _import_gate_module()
    thresh = tmp_path / "thresholds.yaml"
    thresh.write_text(
        "absolute_floors:\n"
        "  - benchmark: cloud_push\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    metric: throughput_mbps\n"
        "    min: 50.0\n"
        "    direction: sideways\n"
    )
    with pytest.raises(ValueError, match="direction="):
        mod._load_thresholds_yaml(thresh)


def test_thresholds_yaml_requires_min_or_max(tmp_path: Path) -> None:
    mod = _import_gate_module()
    thresh = tmp_path / "thresholds.yaml"
    thresh.write_text(
        "absolute_floors:\n"
        "  - benchmark: cloud_push\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    metric: throughput_mbps\n"
    )
    with pytest.raises(ValueError, match="must declare either 'min' or 'max'"):
        mod._load_thresholds_yaml(thresh)


# ---------------------------------------------------------------------------
# Flakiness ledger.
# ---------------------------------------------------------------------------


def _write_raw_runs(current: Path, key: str, walls: list[float]) -> None:
    """Drop a raw JSON next to ``current/summary.json`` so the gate can
    compute wall-time CV. Mirrors the shape ``capture_baseline`` archives.
    """
    raw = current / "raw"
    raw.mkdir(exist_ok=True)
    benchmark, fmt, dataset = key.split("__", 2)
    (raw / f"{key}.json").write_text(json.dumps({
        "benchmark": benchmark, "format": fmt, "dataset": dataset,
        "runs": [{"wall_s": w} for w in walls],
    }))


def test_flakiness_override_relaxes_tolerance(tmp_path: Path) -> None:
    """A 6% timing regression fails the default 3% gate but passes when
    a flakiness override raises the per-row tolerance to 10%.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    flaky = tmp_path / "flaky"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.00)})
    _write_summary(cur,  {"read_full__scx_auto__pbmc3k": _row(1.06)})

    # Default gate: 6% > 3% tolerance ⇒ fail.
    result = _run_gate(baseline=base, current=cur)
    assert result.returncode != 0, (
        f"baseline expectation: 6% regression must fail default gate; "
        f"stdout={result.stdout!r}"
    )

    # Adding a flakiness override raising tolerance to 10% ⇒ pass.
    flaky.mkdir()
    (flaky / "noisy_pbmc3k.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.10\n"
        "reason: \"Shared SLURM node — observed 7% wall-time RSD.\"\n"
        "---\n"
        "Until --exclusive is wired into the queue this row is noisy.\n"
    )
    result = _run_gate(
        "--flakiness", str(flaky),
        baseline=base, current=cur,
    )
    assert result.returncode == 0, (
        f"flakiness override must flip 6% gate to pass; stdout={result.stdout!r}"
    )


def test_flakiness_override_does_not_mask_real_regression(tmp_path: Path) -> None:
    """A 12% regression on a row whose override is 10% still fails — the
    override relaxes, it doesn't suppress.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    flaky = tmp_path / "flaky"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.00)})
    _write_summary(cur,  {"read_full__scx_auto__pbmc3k": _row(1.12)})
    flaky.mkdir()
    (flaky / "noisy_pbmc3k.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.10\n"
        "reason: \"Shared SLURM node — 7% RSD observed.\"\n"
        "---\n"
    )
    result = _run_gate(
        "--flakiness", str(flaky),
        baseline=base, current=cur,
    )
    assert result.returncode != 0, (
        f"override must not mask a regression past the relaxed bound; "
        f"stdout={result.stdout!r}"
    )
    assert "regressed (over relaxed)" in result.stdout, (
        f"status must label the row as overrunning the relaxed bound; "
        f"stdout={result.stdout!r}"
    )


def test_flakiness_expired_does_not_relax(tmp_path: Path) -> None:
    """Expired flakiness file does not raise the per-row tolerance — the
    gate still fails on a 6% regression with the default 3% bound.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    flaky = tmp_path / "flaky"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.00)})
    _write_summary(cur,  {"read_full__scx_auto__pbmc3k": _row(1.06)})
    flaky.mkdir()
    (flaky / "expired.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.10\n"
        "reason: Stale — should not relax anymore.\n"
        "expires: 2000-01-01\n"
        "---\n"
    )
    result = _run_gate(
        "--flakiness", str(flaky),
        baseline=base, current=cur,
    )
    assert result.returncode != 0, (
        f"expired flakiness file must not relax the gate; stdout={result.stdout!r}"
    )


def test_flakiness_metric_scoping(tmp_path: Path) -> None:
    """An override for ``median_wall_s`` does not relax the
    ``peak_rss_mb_median`` row for the same triple.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    flaky = tmp_path / "flaky"
    # Timing flat, RSS up 25% (default rss tolerance is 10%).
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.0, rss=100.0)})
    _write_summary(cur,  {"read_full__scx_auto__pbmc3k": _row(1.0, rss=125.0)})
    flaky.mkdir()
    (flaky / "wall_only.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    metric: median_wall_s\n"
        "    tolerance: 0.50\n"
        "reason: Wall-time noise only; RSS contract unchanged.\n"
        "---\n"
    )
    result = _run_gate(
        "--flakiness", str(flaky),
        baseline=base, current=cur,
    )
    assert result.returncode != 0, (
        f"wall-only override must not relax the RSS gate; stdout={result.stdout!r}"
    )


def test_flakiness_report_surfaces_cv_and_tolerance(tmp_path: Path) -> None:
    """Smoke-test the report shape: CV column populated for the timing
    row when raw runs are present, and the override is reported in the
    JSON payload's flakiness_overrides_applied list.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    flaky = tmp_path / "flaky"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.00)})
    _write_summary(cur,  {"read_full__scx_auto__pbmc3k": _row(1.06)})
    # 6% delta with a high-CV sample set.
    _write_raw_runs(cur, "read_full__scx_auto__pbmc3k", [0.95, 1.06, 1.20])
    flaky.mkdir()
    (flaky / "noisy.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.10\n"
        "reason: noisy.\n"
        "---\n"
    )
    payload_path = tmp_path / "payload.json"
    result = _run_gate(
        "--flakiness", str(flaky),
        "--report-json", str(payload_path),
        baseline=base, current=cur,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(payload_path.read_text())
    assert payload["flakiness_overrides_loaded"] == 1
    applied = payload["flakiness_overrides_applied"]
    assert len(applied) == 1
    assert applied[0] == {
        "benchmark": "read_full", "format": "scx_auto",
        "dataset": "pbmc3k", "metric": "median_wall_s",
        "tolerance": 0.10,
    }
    timing_delta = next(
        d for d in payload["deltas"]
        if d["benchmark"] == "read_full" and d["metric"] == "median_wall_s"
    )
    assert timing_delta["effective_tolerance"] == pytest.approx(0.10)
    assert timing_delta["tolerance_relaxed"] is True
    assert timing_delta["wall_cv"] is not None
    assert timing_delta["wall_cv"] > 0


def test_flakiness_loader_rejects_negative_tolerance(tmp_path: Path) -> None:
    """``_flakiness.parse_flakiness_file`` must refuse a tightening (negative)
    tolerance — overrides relax, they don't tighten.
    """
    sys.path.insert(0, str(GATE_SCRIPT.parent))
    try:
        import importlib

        flakiness = importlib.import_module("_flakiness")
    finally:
        # Clean up sys.path so other tests don't see the added entry.
        sys.path[:] = [p for p in sys.path if p != str(GATE_SCRIPT.parent)]
    bad = tmp_path / "bad.md"
    bad.write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: -0.01\n"
        "reason: Tightening is not allowed.\n"
        "---\n"
    )
    with pytest.raises(ValueError, match="non-negative"):
        flakiness.parse_flakiness_file(bad)


def test_flakiness_override_at_or_below_default_warns(tmp_path: Path) -> None:
    """An override whose tolerance is at or below the global default has
    no effect — the gate must surface a warning so the entry doesn't
    silently rot in the ledger.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    flaky = tmp_path / "flaky"
    _write_summary(base, {"read_full__scx_auto__pbmc3k": _row(1.00)})
    _write_summary(cur,  {"read_full__scx_auto__pbmc3k": _row(1.01)})
    flaky.mkdir()
    # Default --timing-tolerance is 0.03; an override at 0.02 cannot
    # relax (overrides only relax, never tighten).
    (flaky / "ineffective.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.02\n"
        "reason: Tighter than the default — has no effect.\n"
        "---\n"
    )
    result = _run_gate(
        "--flakiness", str(flaky),
        baseline=base, current=cur,
    )
    assert result.returncode == 0, (
        f"1% delta must pass default gate even with an ineffective override; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "override has no effect" in result.stderr, (
        f"ineffective override must emit a stderr warning so the entry "
        f"doesn't silently rot in the ledger; stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Variance-aware (IQR-widened) timing tolerance
# ---------------------------------------------------------------------------


def test_iqr_widens_marginal_regression_passes(tmp_path: Path) -> None:
    """A 7% wall-time regression on a baseline whose IQR/median is 10%
    must PASS once the IQR-based noise widening is applied. Default
    --iqr-k=1.5 ⇒ effective tol = max(0.03, 1.5 * 0.10/1.0) = 0.15;
    7% < 15% ⇒ no regression.

    Without the widening, the same 7% would trip the bare 3% gate (the
    false-positive case the variance-aware tolerance was added to fix).
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    _write_summary(base, {
        "read_full__scx_auto__pbmc3k": _row(1.00, wall_iqr=0.10, n_runs=5),
    })
    _write_summary(cur, {
        "read_full__scx_auto__pbmc3k": _row(1.07),
    })

    result = _run_gate(baseline=base, current=cur)
    assert result.returncode == 0, (
        f"7% regression on a baseline with IQR=10% must pass after "
        f"IQR-widening to 15%; stdout={result.stdout!r}"
    )
    # Header should surface the widened-row count so operators can see the
    # mechanism is active.
    assert "Timing rows widened by IQR" in result.stdout, (
        f"report must surface IQR-widened row count; stdout={result.stdout!r}"
    )


def test_iqr_does_not_mask_real_regression(tmp_path: Path) -> None:
    """A 30% wall-time regression on a baseline with IQR=10% must still
    FAIL — IQR widening relaxes the bound to 15% but does not suppress a
    regression that exceeds the relaxed bound. Locks in the rule from the
    review: noise-widening compensates for measurement noise, not for real
    regressions.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    _write_summary(base, {
        "read_full__scx_auto__pbmc3k": _row(1.00, wall_iqr=0.10, n_runs=5),
    })
    _write_summary(cur, {
        "read_full__scx_auto__pbmc3k": _row(1.30),
    })

    result = _run_gate(baseline=base, current=cur)
    assert result.returncode != 0, (
        f"30% regression must still fail even after IQR-widening to 15%; "
        f"stdout={result.stdout!r}"
    )


def test_missing_wall_iqr_falls_back_to_fixed_tolerance(tmp_path: Path) -> None:
    """When the baseline summary.json has no ``wall_s_iqr`` (legacy
    capture predating the variance-aware tolerance), the gate falls
    back to the fixed --timing-tolerance and emits a single WARN.
    A 5% regression must still fail at the default 3% gate.
    """
    base = tmp_path / "baseline"
    cur = tmp_path / "current"
    # Two rows, both lacking wall_s_iqr — verifies WARN fires once total,
    # not once per row.
    _write_summary(base, {
        "read_full__scx_auto__pbmc3k": _row(1.00),
        "read_full__h5ad_none__pbmc3k": _row(2.00),
    })
    _write_summary(cur, {
        "read_full__scx_auto__pbmc3k": _row(1.05),
        "read_full__h5ad_none__pbmc3k": _row(2.10),
    })

    result = _run_gate(baseline=base, current=cur)
    assert result.returncode != 0, (
        f"5% regression must still fail when IQR is unavailable "
        f"(fallback to 3%); stdout={result.stdout!r}"
    )
    assert "no wall_s_iqr" in result.stderr, (
        f"missing wall_s_iqr must surface a WARN so operators know to "
        f"re-capture the baseline; stderr={result.stderr!r}"
    )
    # WARN must fire exactly once across both rows.
    assert result.stderr.count("no wall_s_iqr") == 1, (
        f"missing-IQR WARN must fire once per gate run, not per row; "
        f"stderr={result.stderr!r}"
    )


def test_iqr_widening_respects_flakiness_override_precedence(tmp_path: Path) -> None:
    """Override-vs-IQR-widening precedence: a flakiness override that
    sits BELOW the IQR-widened tolerance is treated as ineffective (not
    silently ignored). The "ineffective override" warning must compare
    against the row's effective tolerance, which here is the
    IQR-widened bound — otherwise an override at 0.04 on a row whose
    baseline IQR already widens to 0.15 would silently pose as relaxing
    when in fact the IQR widening is doing the work.

    Two sub-cases:
      A. override=0.20 > widened=0.15 ⇒ override wins, row tolerance 0.20.
      B. override=0.04 < widened=0.15 ⇒ override is ineffective, WARN fires
         citing the row's effective (widened) tolerance.
    """
    base = tmp_path / "baseline"
    flaky_a = tmp_path / "flaky_a"
    flaky_b = tmp_path / "flaky_b"

    # Baseline: 1.0 s wall, IQR=0.10 ⇒ widened tol = 0.15.
    _write_summary(base, {
        "read_full__scx_auto__pbmc3k": _row(1.00, wall_iqr=0.10, n_runs=5),
    })

    # Sub-case A: 17% regression > 15% widened bound, but override=0.20
    # raises the row's tolerance to 0.20 ⇒ pass.
    cur_a = tmp_path / "current_a"
    _write_summary(cur_a, {
        "read_full__scx_auto__pbmc3k": _row(1.17),
    })
    flaky_a.mkdir()
    (flaky_a / "noisy.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.20\n"
        "reason: \"Override above the IQR-widened bound — should win.\"\n"
        "---\n"
    )
    result_a = _run_gate(
        "--flakiness", str(flaky_a),
        baseline=base, current=cur_a,
    )
    assert result_a.returncode == 0, (
        f"override above IQR-widened bound must apply; stdout={result_a.stdout!r}"
    )

    # Sub-case B: 5% regression < 15% widened bound ⇒ pass even without
    # the override; an override at 0.04 is below the widened bound, so
    # the warning fires citing the row's effective tolerance.
    cur_b = tmp_path / "current_b"
    _write_summary(cur_b, {
        "read_full__scx_auto__pbmc3k": _row(1.05),
    })
    flaky_b.mkdir()
    (flaky_b / "ineffective.md").write_text(
        "---\n"
        "overrides:\n"
        "  - benchmark: read_full\n"
        "    format: scx_auto\n"
        "    dataset: pbmc3k\n"
        "    tolerance: 0.04\n"
        "reason: \"Override below IQR-widened bound — should warn as ineffective.\"\n"
        "---\n"
    )
    result_b = _run_gate(
        "--flakiness", str(flaky_b),
        baseline=base, current=cur_b,
    )
    assert result_b.returncode == 0, (
        f"5% regression must pass under IQR-widened 15% gate even with "
        f"ineffective sub-widened override; stdout={result_b.stdout!r}"
    )
    # The warning must reference the row's effective tolerance (0.15),
    # not the global default (0.03), so operators see why the override
    # is dead weight against this particular row.
    assert "row's effective tolerance" in result_b.stderr, (
        f"ineffective-override WARN must compare against the row's "
        f"effective (IQR-widened) tolerance, not the global default; "
        f"stderr={result_b.stderr!r}"
    )


def test_absolute_floor_max_direction_flags_overrun(tmp_path: Path) -> None:
    """direction=max ⇒ observed above the ceiling is a violation."""
    mod = _import_gate_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Observed 12.0 > ceiling 10.0 — must violate.
    (raw_dir / "cloud_pull__scx_auto__pbmc3k.json").write_text(json.dumps({
        "runs": [{"extra": {"wall_s": 12.0}}, {"extra": {"wall_s": 12.0}}],
    }))
    floors = [{
        "benchmark": "cloud_pull",
        "format": "scx_auto",
        "dataset": "pbmc3k",
        "metric": "wall_s",
        "threshold": 10.0,
        "direction": "max",
    }]
    violations = mod.check_absolute_floors(tmp_path, floors)
    assert len(violations) == 1
    v = violations[0]
    assert v.direction == "max"
    assert v.observed == pytest.approx(12.0)
    # Below the ceiling ⇒ no violation.
    (raw_dir / "cloud_pull__scx_auto__pbmc3k.json").write_text(json.dumps({
        "runs": [{"extra": {"wall_s": 3.0}}, {"extra": {"wall_s": 4.0}}],
    }))
    assert mod.check_absolute_floors(tmp_path, floors) == []


# ---------------------------------------------------------------------------
# schema_version coverage in summary.json
# ---------------------------------------------------------------------------


def _write_summary_with_version(
    dirpath: Path, schema_version: int | None
) -> None:
    """Write a minimal summary.json honoring an explicit schema_version.

    ``None`` omits the field entirely (legacy baseline shape from before
    capture_baseline.py started stamping the field).
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "snapshot_name": dirpath.name,
        "tier": "small",
        "rows": {},
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    (dirpath / "summary.json").write_text(json.dumps(payload))


def test_load_summary_accepts_current_schema(tmp_path: Path) -> None:
    """A summary.json stamped with the current SCHEMA_VERSION loads."""
    from benchmarks.comprehensive.results import SCHEMA_VERSION

    mod = _import_gate_module()
    _write_summary_with_version(tmp_path, SCHEMA_VERSION)
    data = mod._load_summary(tmp_path)
    assert data["schema_version"] == SCHEMA_VERSION


def test_load_summary_rejects_future_schema(tmp_path: Path) -> None:
    """A summary.json stamped with a future SCHEMA_VERSION must refuse.

    capture_baseline.py stamps summary.json so the gate's future-version
    refusal branch is reachable on the canonical artefact (previously
    it was unreachable because summary.json never carried the field).
    """
    from benchmarks.comprehensive.results import SCHEMA_VERSION

    mod = _import_gate_module()
    _write_summary_with_version(tmp_path, SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version"):
        mod._load_summary(tmp_path)


def test_load_summary_back_compat_missing_schema(tmp_path: Path) -> None:
    """Legacy promoted baselines omit schema_version — must still load.

    Promoted baselines captured before capture_baseline.py started
    stamping the field do not carry it. The gate must keep diffing
    them to avoid orphaning historical baselines committed to the
    tree.
    """
    mod = _import_gate_module()
    _write_summary_with_version(tmp_path, None)
    data = mod._load_summary(tmp_path)
    assert "schema_version" not in data
