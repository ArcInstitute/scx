#!/usr/bin/env python3
"""Collect the phase-0 data-wait fractions out of two capture snapshots.

The docs table is transcribed from **one artifact**, not from a job log: a
number read off a log and retyped into `docs/performance.md` has no manifest
row behind it, and `docs/benchmark_manifest.md` requires every triple-shaped
claim to have one.

Reads `runs[].extra` the way the gate does — median over the runs that carry
the key, skipping `None` — so the published median is computed by the same rule
`compare_against_baseline._load_current_raw_metric` would apply if the metric
were ever floored (it is not; see thresholds.yaml's deferred item 21).

    python benchmarks/scripts/_phase0_data_wait_summary.py \\
        benchmarks/comprehensive/results/candidate_phase0_p_r1_abc1234 \\
        benchmarks/comprehensive/results/candidate_phase0_p_r3_nullmodel_abc1234 \\
        --out p_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# The scenarios that can define a data-wait fraction, and the regime each one
# stands in for. A scenario absent from a snapshot is reported as absent, never
# as zero.
_REGIMES = {
    "gpu_train": "R1 i.i.d. minibatches (TrainingDataset)",
    "pyscx_index_plan_dataset_workers2": "R3 paired batches (IndexPlanDataset)",
}

# `data_wait_fraction_steady` is the headline: the all-steps figure folds in a
# once-per-epoch time-to-first-batch and on a short epoch reads ~50x higher.
_METRICS = (
    "data_wait_fraction_steady",
    "ttfb_s",
    "n_steady_steps",
    "data_wait_fraction",
    "batch_wait_ms_p50",
    "batch_wait_ms_p95",
    "batch_wait_ms_p99",
    "batch_wait_ms_max",
    "batches_per_sec",
    "null_model_ms",
)


def _median_of(runs: list[dict], key: str) -> float | None:
    """Median over the runs carrying *key*, skipping nulls. `None` if none do.

    The null-skipping is not defensive tidying: `data_wait_fraction__*` is
    emitted on every scenario and is `None` wherever the consumer had no model
    step, so treating a null as 0.0 here would publish "the loader never
    stalled" for scenarios that measured nothing at all.
    """
    vals = []
    for r in runs:
        v = (r.get("extra") or {}).get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return statistics.median(vals) if vals else None


def _max_of(runs: list[dict], key: str) -> float | None:
    """Max over the runs carrying *key*, skipping nulls. `None` if none do."""
    vals = []
    for r in runs:
        v = (r.get("extra") or {}).get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return max(vals) if vals else None


def _rows(snapshot: Path) -> list[dict[str, Any]]:
    raw = snapshot / "raw"
    if not raw.is_dir():
        raise SystemExit(f"{snapshot}: no raw/ directory — capture did not land")
    out: list[dict[str, Any]] = []
    for path in sorted(raw.glob("*.json")):
        data = json.loads(path.read_text())
        runs = data.get("runs") or []
        for scenario, regime in _REGIMES.items():
            scoped = [r for r in runs if (r.get("extra") or {}).get("scenario") == scenario]
            if not scoped:
                continue
            row: dict[str, Any] = {
                "regime": regime,
                "benchmark": data["benchmark"],
                "format": data["format"],
                "dataset": data["dataset"],
                "scenario": scenario,
                "n_runs": len(scoped),
                "n_steps": [(r.get("extra") or {}).get("n_batches") for r in scoped],
                "source_file": str(path),
                "git_sha": data.get("system", {}).get("provenance", {}).get("git_sha"),
                "git_dirty": data.get("system", {}).get("provenance", {}).get("git_dirty"),
            }
            for m in _METRICS:
                # `batch_wait_ms_max` is aggregated with max, not median. The
                # tail IS the subject (ml_loader's emitter says the same of its
                # cross-run max), and a median over three maxima hides the worst
                # stall — which is the one number the metric exists to show.
                agg = _max_of if m == "batch_wait_ms_max" else _median_of
                row[m] = agg(scoped, f"{m}__{scenario}")
            # `batches_per_sec` is emitted unsuffixed on gpu_train too; fall
            # back so the table can show throughput beside the fraction.
            if row["batches_per_sec"] is None:
                row["batches_per_sec"] = _median_of(scoped, "batches_per_sec")
            row["model"] = next(
                (
                    (r.get("extra") or {}).get("model")
                    for r in scoped
                    if (r.get("extra") or {}).get("model")
                ),
                None,
            )
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for s in args.snapshots:
        rows.extend(_rows(s))

    if not rows:
        raise SystemExit(
            "no scenario in any snapshot emitted a data-wait row. Either the "
            "captures did not run the gpu_train / workers2 scenarios, or the "
            "instrumentation did not reach them — do NOT publish a `p` table."
        )

    measured = [r for r in rows if r["data_wait_fraction_steady"] is not None]
    print(f"{len(rows)} scenario row(s), {len(measured)} with a defined fraction\n")
    hdr = (
        f"{'regime':46s} {'dataset':22s} {'format':10s} "
        f"{'p_steady':>10s} {'p_all':>8s} {'ttfb s':>8s} "
        f"{'p50 ms':>9s} {'p99 ms':>9s} {'max ms':>9s} {'steps':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    def _fmt(v: float | None, spec: str) -> str:
        return "-" if v is None else format(v, spec)

    for r in rows:
        steps = r["n_steps"][0] if r["n_steps"] else None
        print(
            f"{r['regime'][:46]:46s} {r['dataset'][:22]:22s} {r['format'][:10]:10s} "
            f"{_fmt(r['data_wait_fraction_steady'], '.6f'):>10s} "
            f"{_fmt(r['data_wait_fraction'], '.4f'):>8s} "
            f"{_fmt(r['ttfb_s'], '.3f'):>8s} "
            f"{_fmt(r['batch_wait_ms_p50'], '.2f'):>9s} "
            f"{_fmt(r['batch_wait_ms_p99'], '.2f'):>9s} "
            f"{_fmt(r['batch_wait_ms_max'], '.1f'):>9s} "
            f"{'-' if steps is None else str(steps):>7s}"
        )

    if args.out:
        args.out.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
