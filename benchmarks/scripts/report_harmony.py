#!/usr/bin/env python3
"""Aggregate Harmony + LISI benchmark JSONs into a markdown report.

Reads every JSON under `benchmarks/results/harmony/runs/`, groups by
(benchmark, impl, device, dataset), and writes:
    benchmarks/results/harmony/REPORT.md
    benchmarks/results/harmony/wall_s_vs_N.png
    benchmarks/results/harmony/peak_rss_vs_N.png

Report contents:
  * Summary table — median `wall_s`, `peak_rss_mb`, `n_iterations`,
    per-PC Pearson r (min across PCs) per (impl x device x dataset).
  * Scaling table — wall + RSS per impl across datasets, with empirical
    exponents fit from log(y) = a * log(N) + b.
  * Secondary sweeps — wall_s vs d and wall_s vs K on D4.
  * LISI comparison — scx-accel vs R lisi: timing, mean-LISI delta.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Census datasets ordered by N for scaling plots.
DATASETS_ORDER = [
    "pbmc3k",
    "pbmc10k",
    "smartseq2",
    "tabula_sapiens_100k",
    "census_500k",
    "census_1m",
    "census_5m",
]
DATASET_N = {
    "pbmc3k": 2_700,
    "pbmc10k": 11_769,
    "smartseq2": 50_000,
    "tabula_sapiens_100k": 100_000,
    "census_500k": 500_000,
    "census_1m": 1_000_000,
    "census_5m": 5_000_000,
}


def _load_runs(runs_dir: Path) -> list[dict]:
    results = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return results


def _fit_exponent(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit log(y) = a * log(x) + b; return (a, b). `a` is the scaling exponent."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    lx = np.log(np.asarray(xs, dtype=np.float64))
    ly = np.log(np.asarray(ys, dtype=np.float64))
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(b)


def _maybe_plot(
    path: Path,
    title: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, pts in series.items():
        if not pts:
            continue
        pts = sorted(pts)
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, marker="o", label=label)
    ax.set_xlabel("N (cells)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _fmt(x: Any, fmt: str = "{:.2f}") -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "—"
    try:
        return fmt.format(x)
    except (TypeError, ValueError):
        return str(x)


def _min_pearson(r_list: list[float] | None) -> float | None:
    if not r_list:
        return None
    finite = [v for v in r_list if v is not None and math.isfinite(v)]
    return min(finite) if finite else None


def build_report(runs_dir: Path, out_dir: Path) -> Path:
    runs = _load_runs(runs_dir)
    harmony_runs = [r for r in runs if r.get("benchmark") == "harmony_integrate"]
    lisi_runs = [r for r in runs if r.get("benchmark") == "lisi"]

    # Bucket harmony runs by (impl, device, dataset).
    table: dict[tuple[str, str, str], dict] = {}
    for r in harmony_runs:
        key = (r["impl"], r.get("device", "cpu"), r["dataset"])
        # If multiple runs for same key, keep the last (replayed).
        table[key] = r

    out_dir.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append("# Harmony2 + LISI — Benchmark Report\n")
    md.append(f"Source: `{runs_dir}` — {len(runs)} JSON files "
              f"({len(harmony_runs)} harmony, {len(lisi_runs)} lisi).\n")

    # ── Summary table ─────────────────────────────────────────────
    md.append("## Summary\n")
    md.append(
        "| impl | device | dataset | N | d | K | n_iters | wall (s) | "
        "peak RSS (MB) | min per-PC r | ok |"
    )
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    # Sort by (N ascending, impl, device) so the scaling story reads top-to-bottom.
    def _row_key(item):
        (impl, device, dataset), r = item
        return (
            r["dataset_meta"].get("n_obs", 0),
            DATASETS_ORDER.index(dataset) if dataset in DATASETS_ORDER else 99,
            impl,
            device,
        )
    for (impl, device, dataset), r in sorted(table.items(), key=_row_key):
        run = r.get("run", {})
        params = r.get("params", {})
        md.append(
            "| {impl} | {dev} | {ds} | {N} | {d} | {K} | {ni} | "
            "{wall} | {rss} | {rpc} | {ok} |".format(
                impl=impl,
                dev=device,
                ds=dataset,
                N=r["dataset_meta"].get("n_obs", "—"),
                d=params.get("n_pcs", "—"),
                K=params.get("n_clusters", "—"),
                ni=_fmt(run.get("n_iterations"), "{:d}")
                if isinstance(run.get("n_iterations"), int)
                else "—",
                wall=_fmt(run.get("wall_s")),
                rss=_fmt(run.get("peak_rss_mb")),
                rpc=_fmt(_min_pearson(run.get("per_pc_pearson_r")), "{:.3f}"),
                ok="yes" if run.get("ok") else r.get("run", {}).get("error", "no"),
            )
        )

    # ── Scaling table ─────────────────────────────────────────────
    md.append("\n## Scaling (D1–D7, d=30, K=100, theta=2, max_iter=10)\n")
    # Per impl: (N, wall) and (N, rss) pairs.
    wall_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rss_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (impl, device, dataset), r in table.items():
        if r.get("params", {}).get("n_pcs") != 30:
            continue
        if r.get("params", {}).get("n_clusters") != 100:
            continue
        n = r["dataset_meta"].get("n_obs")
        run = r.get("run", {})
        if not n or not run.get("ok"):
            continue
        label = f"{impl} ({device})"
        if run.get("wall_s") is not None:
            wall_series[label].append((n, run["wall_s"]))
        if run.get("peak_rss_mb") is not None:
            rss_series[label].append((n, run["peak_rss_mb"]))

    md.append("| impl/device | wall_s exponent α | peak_rss_mb exponent β | pts |")
    md.append("|---|---:|---:|---:|")
    for label in sorted(set(wall_series) | set(rss_series)):
        w_pts = sorted(wall_series.get(label, []))
        r_pts = sorted(rss_series.get(label, []))
        a_w, _ = _fit_exponent([p[0] for p in w_pts], [p[1] for p in w_pts])
        a_r, _ = _fit_exponent([p[0] for p in r_pts], [p[1] for p in r_pts])
        md.append(f"| {label} | {_fmt(a_w, '{:.2f}')} | "
                  f"{_fmt(a_r, '{:.2f}')} | {len(w_pts)} |")

    wall_png = out_dir / "wall_s_vs_N.png"
    rss_png = out_dir / "peak_rss_vs_N.png"
    plotted_w = _maybe_plot(
        wall_png, "Harmony wall time vs N", "wall (s)", wall_series
    )
    plotted_r = _maybe_plot(
        rss_png, "Harmony peak RSS vs N", "peak RSS (MB)", rss_series
    )
    if plotted_w:
        md.append(f"\n![wall vs N]({wall_png.name})\n")
    if plotted_r:
        md.append(f"\n![rss vs N]({rss_png.name})\n")

    # ── Secondary sweeps (D4) ─────────────────────────────────────
    md.append("\n## Secondary sweeps — tabula_sapiens_100k (D4)\n")
    md.append("### wall_s vs d (CPU, K=100)\n")
    md.append("| d | wall_s | peak_rss_mb |")
    md.append("|---:|---:|---:|")
    for r in harmony_runs:
        if r.get("dataset") != "tabula_sapiens_100k":
            continue
        if r.get("impl") != "scx_accel_cpu":
            continue
        if r.get("params", {}).get("n_clusters") != 100:
            continue
        run = r.get("run", {})
        md.append(
            f"| {r['params']['n_pcs']} | {_fmt(run.get('wall_s'))} | "
            f"{_fmt(run.get('peak_rss_mb'))} |"
        )
    md.append("\n### wall_s vs K (CPU, d=30)\n")
    md.append("| K | wall_s | peak_rss_mb |")
    md.append("|---:|---:|---:|")
    for r in harmony_runs:
        if r.get("dataset") != "tabula_sapiens_100k":
            continue
        if r.get("impl") != "scx_accel_cpu":
            continue
        if r.get("params", {}).get("n_pcs") != 30:
            continue
        run = r.get("run", {})
        md.append(
            f"| {r['params']['n_clusters']} | {_fmt(run.get('wall_s'))} | "
            f"{_fmt(run.get('peak_rss_mb'))} |"
        )

    # ── LISI comparison ───────────────────────────────────────────
    md.append("\n## LISI — scx-accel vs R `lisi`\n")
    md.append("| dataset | impl | wall_s | peak_rss_mb | mean_lisi | median |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in sorted(lisi_runs, key=lambda x: (x["dataset"], x["impl"])):
        run = r.get("run", {})
        md.append(
            f"| {r['dataset']} | {r['impl']} | "
            f"{_fmt(run.get('wall_s'))} | {_fmt(r.get('peak_rss_mb'))} | "
            f"{_fmt(run.get('mean_lisi'), '{:.3f}')} | "
            f"{_fmt(run.get('median_lisi'), '{:.3f}')} |"
        )

    # Mean-LISI delta (scx-accel vs r_lisi) where both ran on same dataset.
    by_ds: dict[str, dict[str, float]] = defaultdict(dict)
    for r in lisi_runs:
        run = r.get("run")
        if not run:
            continue
        by_ds[r["dataset"]][r["impl"]] = run.get("mean_lisi")
    md.append("\n### Mean-LISI agreement (scx-accel vs R lisi)\n")
    md.append("| dataset | scx_accel | r_lisi | |Δ| / r_lisi |")
    md.append("|---|---:|---:|---:|")
    for ds, m in sorted(by_ds.items()):
        scx_v = m.get("scx_accel")
        r_v = m.get("r_lisi")
        delta = None
        if scx_v is not None and r_v is not None and r_v != 0:
            delta = abs(scx_v - r_v) / r_v
        md.append(
            f"| {ds} | {_fmt(scx_v, '{:.3f}')} | {_fmt(r_v, '{:.3f}')} | "
            f"{_fmt(delta, '{:.2%}')} |"
        )

    # ── Write the report ──────────────────────────────────────────
    out_md = out_dir / "REPORT.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"[report] wrote {out_md}")
    return out_md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir",
                    default="benchmarks/results/harmony/runs")
    ap.add_argument("--out-dir",
                    default="benchmarks/results/harmony")
    args = ap.parse_args()
    build_report(Path(args.runs_dir), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
