#!/usr/bin/env python3
"""Issue-10 harmony scaling A/B report.

Reads the issue10_${state}_${dataset}_d30_K100_cpu.json results emitted by
the sbatch wrapper and prints a side-by-side scaling table for the
metrics the optimisation targets: peak_rss_mb, harmony_delta_rss_mb,
wall_s, n_iterations, final-objective parity.
"""
import json
import sys
from pathlib import Path

RUNS = Path("/home/nickyoungblut/dev/rust/scx/benchmarks/results/harmony/runs")

DATASETS = [
    ("tabula_sapiens_100k", "tabula100k", 100_000),
    ("census_500k", "census_500k", 500_000),
    ("census_1m", "census_1m", 1_000_000),
]


def load(state: str, slug: str) -> dict | None:
    p = RUNS / f"issue10_{state}_{slug}_d30_K100_cpu.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt_delta(before: float, after: float, unit: str = "") -> str:
    if before == 0:
        return "—"
    delta = after - before
    pct = (delta / before) * 100
    return f"{delta:+.1f}{unit} ({pct:+.1f}%)"


def main() -> int:
    print(f"\n{'dataset':<22} | {'metric':<24} | {'before':>10} | {'after':>10} | {'delta':>22}")
    print("-" * 100)
    seen_any = False
    for ds_name, slug, n in DATASETS:
        b = load("before", slug)
        a = load("after", slug)
        if not b or not a:
            print(f"{ds_name:<22} | (skipped — files missing)")
            continue
        if not (b.get("ok") and a.get("ok")):
            print(f"{ds_name:<22} | (skipped — error: before.ok={b.get('ok')} after.ok={a.get('ok')})")
            continue
        seen_any = True
        label = f"{ds_name} (N={n:,})"
        for metric, unit in [
            ("wall_s", "s"),
            ("peak_rss_mb", " MB"),
            ("baseline_rss_mb", " MB"),
            ("harmony_delta_rss_mb", " MB"),
            ("n_iterations", ""),
        ]:
            bv, av = b.get(metric), a.get(metric)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                if metric == "n_iterations":
                    delta = f"{av - bv:+d}"
                else:
                    delta = fmt_delta(bv, av, unit)
                print(f"{label:<22} | {metric:<24} | {bv:>10.1f} | {av:>10.1f} | {delta:>22}")
        # Convergence + objective parity
        print(f"{label:<22} | {'converged':<24} | {str(b.get('converged')):>10} | {str(a.get('converged')):>10} |")
        bo = (b.get("objective") or [None])[-1]
        ao = (a.get("objective") or [None])[-1]
        if bo is not None and ao is not None:
            rel = (ao - bo) / bo if bo != 0 else 0.0
            print(f"{label:<22} | {'final objective':<24} | {bo:>10.4f} | {ao:>10.4f} | {rel*100:>+10.4f}% rel")
        # Host info
        print(f"{label:<22} | {'host (before/after)':<24} | "
              f"{(b.get('host') or '?'):>10} | {(a.get('host') or '?'):>10} |")
        print("-" * 100)
    return 0 if seen_any else 1


if __name__ == "__main__":
    sys.exit(main())
