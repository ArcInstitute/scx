#!/usr/bin/env python3
"""Benchmark read performance: SCX vs gzip/lzf-compressed h5ad.

Most researchers use scanpy.write() which defaults to gzip compression.
This benchmark measures how SCX compares against the realistic h5ad baseline,
not just the optimal uncompressed case.

Usage:
    .venv/bin/python benchmarks/scripts/benchmark_compressed_h5ad.py

Environment:
    SCX_WORK_DIR  — directory containing .h5ad and .scx files (set via .env)
"""

import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

from bench_env import WORK_DIR
N_WARMUP = 1
N_REPEATS = 3

BENCHMARK_DATASETS = [
    "pbmc3k", "smartseq2", "tabula_sapiens_100k", "census_1m",
]

COMPRESSIONS = ["none", "gzip", "lzf"]


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _time_read(read_fn, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """Time a read function, returning (median_seconds, result_from_last_call)."""
    for _ in range(n_warmup):
        result = read_fn()
        del result
        gc.collect()

    times = []
    result = None
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = read_fn()
        times.append(time.perf_counter() - t0)
        if _ < n_repeats - 1:
            del result
            result = None
            gc.collect()

    return _median(times), result


def _file_size_mb(path):
    """Return file size in MB, or None if missing."""
    if path.exists():
        return path.stat().st_size / 1e6
    return None


def create_compressed_h5ad(source_h5ad, output_path, compression):
    """Create a compressed h5ad file from an uncompressed source.

    Args:
        source_h5ad: Path to the source h5ad file.
        output_path: Path to write the compressed h5ad.
        compression: "gzip" or "lzf".
    """
    import anndata

    if output_path.exists():
        print(f"    Using existing {output_path.name}")
        return

    print(f"    Creating {output_path.name} (compression={compression})...")
    adata = anndata.read_h5ad(str(source_h5ad))
    adata.write_h5ad(str(output_path), compression=compression)
    del adata
    gc.collect()
    size_mb = output_path.stat().st_size / 1e6
    print(f"    Created: {size_mb:.1f} MB")


def benchmark_dataset(dataset_name):
    """Benchmark read times for all compression variants of a dataset.

    Returns a dict with timing and size data, or None if dataset not found.
    """
    import anndata
    import pyscx

    h5ad_path = WORK_DIR / f"{dataset_name}.h5ad"
    scx_path = WORK_DIR / f"{dataset_name}.scx"

    if not h5ad_path.exists():
        return None

    # Create SCX file if missing
    if not scx_path.exists():
        print(f"  Converting {dataset_name} to SCX...")
        adata = anndata.read_h5ad(str(h5ad_path))
        pyscx.from_anndata(adata, str(scx_path))
        del adata
        gc.collect()

    # Create compressed variants
    compressed_paths = {}
    for comp in COMPRESSIONS:
        if comp == "none":
            compressed_paths[comp] = h5ad_path
        else:
            comp_path = WORK_DIR / f"{dataset_name}_{comp}.h5ad"
            create_compressed_h5ad(h5ad_path, comp_path, comp)
            compressed_paths[comp] = comp_path

    # Gather dataset info
    adata = anndata.read_h5ad(str(h5ad_path))
    result = {
        "dataset": dataset_name,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
    }
    del adata
    gc.collect()

    # File sizes
    result["sizes"] = {}
    for comp in COMPRESSIONS:
        size = _file_size_mb(compressed_paths[comp])
        result["sizes"][f"h5ad_{comp}"] = size
    result["sizes"]["scx"] = _file_size_mb(scx_path)

    # Time h5ad reads for each compression
    result["times"] = {}
    for comp in COMPRESSIONS:
        path = compressed_paths[comp]
        print(f"  Timing h5ad ({comp}): {path.name}...")
        t, _ = _time_read(lambda p=str(path): anndata.read_h5ad(p))
        result["times"][f"h5ad_{comp}"] = round(t, 4)
        gc.collect()
        print(f"    {t:.3f}s")

    # Time SCX read
    print(f"  Timing SCX: {scx_path.name}...")
    t, _ = _time_read(lambda: pyscx.open(str(scx_path)).to_anndata())
    result["times"]["scx"] = round(t, 4)
    gc.collect()
    print(f"    {t:.3f}s")

    # Compute speedups (SCX time / h5ad time — >1 means SCX is faster)
    result["speedups"] = {}
    scx_time = result["times"]["scx"]
    for comp in COMPRESSIONS:
        h5ad_time = result["times"][f"h5ad_{comp}"]
        if scx_time > 0 and h5ad_time > 0:
            result["speedups"][comp] = round(h5ad_time / scx_time, 2)

    return result


def generate_report(results):
    """Generate markdown report from benchmark results."""
    lines = []
    lines.append("# Compressed h5ad Benchmark Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Data directory**: `{WORK_DIR}`")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append("Most researchers write h5ad files via `scanpy.write()`, which defaults to **gzip**")
    lines.append("compression. The existing SCX benchmarks compare against uncompressed h5ad (the")
    lines.append("best-case h5ad read path). This benchmark measures the realistic baseline.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Read timing table
    lines.append("## Read Performance (full file → AnnData)")
    lines.append("")
    lines.append("Median of 3 runs, warm cache.")
    lines.append("")
    lines.append("| Dataset | h5ad (none) | h5ad (gzip) | h5ad (lzf) | SCX | SCX vs none | SCX vs gzip | SCX vs lzf |")
    lines.append("|---------|:-----------:|:-----------:|:----------:|:---:|:-----------:|:-----------:|:----------:|")

    for r in results:
        t_none = r["times"].get("h5ad_none", 0)
        t_gzip = r["times"].get("h5ad_gzip", 0)
        t_lzf = r["times"].get("h5ad_lzf", 0)
        t_scx = r["times"].get("scx", 0)
        sp_none = r["speedups"].get("none", 0)
        sp_gzip = r["speedups"].get("gzip", 0)
        sp_lzf = r["speedups"].get("lzf", 0)

        def _fmt_sp(sp):
            if sp >= 1.0:
                return f"**{sp:.1f}x** ✓"
            return f"{sp:.1f}x"

        lines.append(
            f"| {r['dataset']} "
            f"| {t_none:.3f}s "
            f"| {t_gzip:.3f}s "
            f"| {t_lzf:.3f}s "
            f"| {t_scx:.3f}s "
            f"| {_fmt_sp(sp_none)} "
            f"| {_fmt_sp(sp_gzip)} "
            f"| {_fmt_sp(sp_lzf)} |"
        )

    lines.append("")

    # Gzip slowdown analysis
    lines.append("### Gzip Slowdown Analysis")
    lines.append("")
    lines.append("| Dataset | gzip / none | lzf / none | Gzip decompression overhead |")
    lines.append("|---------|:-----------:|:----------:|:--------------------------:|")
    for r in results:
        t_none = r["times"].get("h5ad_none", 0)
        t_gzip = r["times"].get("h5ad_gzip", 0)
        t_lzf = r["times"].get("h5ad_lzf", 0)
        gzip_ratio = t_gzip / t_none if t_none > 0 else 0
        lzf_ratio = t_lzf / t_none if t_none > 0 else 0
        overhead_pct = (gzip_ratio - 1) * 100 if gzip_ratio > 0 else 0
        lines.append(
            f"| {r['dataset']} "
            f"| {gzip_ratio:.2f}x slower "
            f"| {lzf_ratio:.2f}x slower "
            f"| +{overhead_pct:.0f}% |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # File sizes table
    lines.append("## File Sizes")
    lines.append("")
    lines.append("| Dataset | h5ad (none) | h5ad (gzip) | h5ad (lzf) | SCX (auto) | SCX / gzip |")
    lines.append("|---------|:-----------:|:-----------:|:----------:|:----------:|:----------:|")

    for r in results:
        s = r["sizes"]

        def _fmt_size(mb):
            if mb is None:
                return "—"
            if mb >= 1000:
                return f"{mb / 1000:.2f} GB"
            return f"{mb:.1f} MB"

        s_none = s.get("h5ad_none")
        s_gzip = s.get("h5ad_gzip")
        s_scx = s.get("scx")
        scx_vs_gzip = f"{s_scx / s_gzip:.2f}" if s_scx and s_gzip and s_gzip > 0 else "—"

        lines.append(
            f"| {r['dataset']} "
            f"| {_fmt_size(s_none)} "
            f"| {_fmt_size(s_gzip)} "
            f"| {_fmt_size(s.get('h5ad_lzf'))} "
            f"| {_fmt_size(s_scx)} "
            f"| {scx_vs_gzip} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Check if SCX beats gzip-h5ad on any dataset
    gzip_wins = [r for r in results if r["speedups"].get("gzip", 0) >= 1.0]
    gzip_close = [r for r in results if 0.8 <= r["speedups"].get("gzip", 0) < 1.0]

    if len(gzip_wins) == len(results):
        lines.append("**SCX meets or exceeds gzip-h5ad read speed on ALL datasets.** The performance")
        lines.append("gap reported in previous benchmarks (0.4-0.6× h5ad) was measured against")
        lines.append("uncompressed h5ad, which is not the typical real-world format. Against the")
        lines.append("gzip-compressed h5ad that most researchers use, SCX is already competitive.")
    elif gzip_wins:
        wins = ", ".join(r["dataset"] for r in gzip_wins)
        lines.append(f"**SCX meets or exceeds gzip-h5ad on {len(gzip_wins)}/{len(results)} datasets** ({wins}).")
        if gzip_close:
            close = ", ".join(r["dataset"] for r in gzip_close)
            lines.append(f"Near-parity on: {close}.")
        lines.append("")
        lines.append("The codec optimizations in IMPROVE-FULL-READ.md items 1-4 would further close")
        lines.append("the gap on remaining datasets.")
    else:
        lines.append("SCX is slower than gzip-h5ad on all tested datasets. The codec optimizations")
        lines.append("in IMPROVE-FULL-READ.md items 1-4 are critical for achieving parity.")

    lines.append("")

    return "\n".join(lines)


def run_all():
    """Run compressed h5ad benchmarks on all datasets."""
    results = []
    for name in BENCHMARK_DATASETS:
        print(f"\nBenchmarking: {name}")
        print(f"{'=' * 50}")
        try:
            result = benchmark_dataset(name)
            if result is None:
                print(f"  SKIP: {name}.h5ad not found in {WORK_DIR}")
                continue
            results.append(result)

            # Summary line
            sp_none = result["speedups"].get("none", 0)
            sp_gzip = result["speedups"].get("gzip", 0)
            sp_lzf = result["speedups"].get("lzf", 0)
            print(f"\n  Summary: SCX vs none={sp_none:.1f}x, vs gzip={sp_gzip:.1f}x, vs lzf={sp_lzf:.1f}x")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nNo datasets found. Set SCX_WORK_DIR to the directory containing .h5ad files.")
        return

    # Generate report
    report = generate_report(results)
    report_path = PROJECT_ROOT / "benchmarks" / "results" / "compressed_h5ad_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport written to: {report_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: SCX vs Compressed h5ad")
    print("=" * 80)
    print(f"{'Dataset':<25} {'vs none':>10} {'vs gzip':>10} {'vs lzf':>10}")
    print("-" * 55)
    for r in results:
        sp_none = r["speedups"].get("none", 0)
        sp_gzip = r["speedups"].get("gzip", 0)
        sp_lzf = r["speedups"].get("lzf", 0)
        check_gzip = "✓" if sp_gzip >= 1.0 else ""
        print(f"{r['dataset']:<25} {sp_none:>9.1f}x {sp_gzip:>9.1f}x {check_gzip} {sp_lzf:>8.1f}x")


if __name__ == "__main__":
    ensure_release_build()
    run_all()
