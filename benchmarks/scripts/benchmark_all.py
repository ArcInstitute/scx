#!/usr/bin/env python3
"""Run all benchmarks and output comprehensive markdown report.

Benchmarks:
  A. Compression ratios (SCX auto/none/scx1/zstd vs h5ad vs Zarr+Zstd)
  B. Write performance (h5ad → SCX per codec)
  C. Read performance (SCX vs h5ad, parallel scaling, memory)
  D. Query engine (Rust tests on real data)
  E. File operations (Rust tests on real data)
  F. ML data loader (SCX vs SOTA baselines)

Usage:
    python benchmark_all.py              # Run A, B, C only (fast)
    python benchmark_all.py --all        # Run A-F
    python benchmark_all.py --loader     # Run F only
"""

import argparse
import datetime
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build


def format_size(bytes_val):
    """Format byte count as human-readable string."""
    if bytes_val is None:
        return "N/A"
    if bytes_val < 1e6:
        return f"{bytes_val / 1e3:.1f} KB"
    if bytes_val < 1e9:
        return f"{bytes_val / 1e6:.1f} MB"
    return f"{bytes_val / 1e9:.2f} GB"


def system_info():
    """Collect system info for report header."""
    info = {"hostname": os.uname().nodename, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    gb = int(line.split()[1]) / 1e6
                    info["ram_gb"] = f"{gb:.0f}"
                    break
    except Exception:
        pass
    return info


def generate_comprehensive_markdown(compression, read, write, sysinfo):
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# SCX Comprehensive Benchmark Report\n")
    lines.append(f"**Generated**: {sysinfo.get('date', 'unknown')}")
    lines.append(f"**Host**: {sysinfo.get('hostname', 'unknown')}")
    lines.append(f"**CPU**: {sysinfo.get('cpu', 'unknown')}")
    lines.append(f"**RAM**: {sysinfo.get('ram_gb', '?')} GB")
    lines.append("")

    # --- A. Compression ---
    lines.append("## A. Compression Ratios\n")
    lines.append("| Dataset | Cells | Genes | h5ad | SCX (auto) | Ratio | SCX (none) | SCX (scx1) | SCX (zstd) | Zarr+Zstd |")
    lines.append("|---------|-------|-------|------|-----------|-------|-----------|-----------|-----------|-----------|")
    for r in compression:
        def _sz(key):
            v = r.get(key)
            return format_size(v) if v else "N/A"
        def _rat(key):
            v = r.get(key)
            return f"{v:.3f}" if v else "N/A"
        lines.append(
            f"| {r['dataset']} | {r.get('n_obs', '?'):,} | {r.get('n_vars', '?'):,} "
            f"| {format_size(r['h5ad_size'])} | {_sz('scx_auto_size')} | {_rat('scx_auto_ratio')} "
            f"| {_sz('scx_none_size')} | {_sz('scx_scx1_size')} | {_sz('scx_zstd_size')} "
            f"| {_sz('zarr_zstd_size')} |"
        )
    lines.append("")

    # --- B. Write ---
    lines.append("## B. Write Performance (h5ad → SCX, median of 3)\n")
    lines.append("| Dataset | h5ad Size | auto (s) | auto (MB/s) | none (s) | scx1 (s) | zstd (s) |")
    lines.append("|---------|-----------|----------|-------------|----------|----------|----------|")
    for r in write:
        def _t(codec):
            v = r.get(f"write_{codec}_s")
            return f"{v:.3f}" if v else "N/A"
        def _m(codec):
            v = r.get(f"write_{codec}_mb_s")
            return f"{v:.1f}" if v else "N/A"
        lines.append(
            f"| {r['dataset']} | {r['h5ad_size_mb']:.1f} MB "
            f"| {_t('auto')} | {_m('auto')} | {_t('none')} | {_t('scx1')} | {_t('zstd')} |"
        )
    lines.append("")

    # --- C. Read ---
    lines.append("## C. Read Performance (warm cache, median of 3)\n")
    lines.append("| Dataset | h5ad (s) | SCX (s) | Speedup | SCX Open (ms) | SCX Peak MB | h5ad Peak MB |")
    lines.append("|---------|----------|---------|---------|---------------|-------------|--------------|")
    for r in read:
        lines.append(
            f"| {r['dataset']} | {r['h5ad_warm_s']:.3f} | {r['scx_warm_s']:.3f} "
            f"| {r['speedup']:.1f}x | {r.get('scx_open_s', 0) * 1000:.1f} "
            f"| {r.get('scx_peak_mb', 0):.1f} | {r.get('h5ad_peak_mb', 0):.1f} |"
        )
    lines.append("")

    # Parallel scaling sub-table
    has_parallel = any("parallel_scaling" in r for r in read)
    if has_parallel:
        lines.append("### Parallel Read Scaling\n")
        lines.append("| Dataset | 1 thread (s) | 2 threads | 4 threads | 8 threads |")
        lines.append("|---------|-------------|-----------|-----------|-----------|")
        for r in read:
            ps = r.get("parallel_scaling", {})
            psu = r.get("parallel_speedups", {})
            def _cell(t):
                time_val = ps.get(t)
                speedup = psu.get(t)
                if time_val is None:
                    return "N/A"
                if speedup and t > 1:
                    return f"{time_val:.3f}s ({speedup:.1f}x)"
                return f"{time_val:.3f}s"
            lines.append(
                f"| {r['dataset']} | {_cell(1)} | {_cell(2)} | {_cell(4)} | {_cell(8)} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SCX Comprehensive Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks (A-F)")
    parser.add_argument("--loader", action="store_true", help="Run loader benchmark only (F)")
    parser.add_argument("--ops", action="store_true", help="Run ops+query benchmarks (D-E)")
    args = parser.parse_args()

    print("=" * 60)
    print("SCX Comprehensive Benchmark Suite")
    print("=" * 60)
    print()

    ensure_release_build()
    print()

    sysinfo = system_info()
    compression_results = []
    read_results = []
    write_results = []

    if not args.loader and not args.ops:
        import benchmark_compression
        import benchmark_read
        import benchmark_write

        print("--- A. Compression Benchmarks ---")
        compression_results = benchmark_compression.run_all()
        print()

        print("--- B. Write Benchmarks ---")
        write_results = benchmark_write.run_all()
        print()

        print("--- C. Read Benchmarks ---")
        read_results = benchmark_read.run_all()
        print()

    if args.all or args.ops:
        import benchmark_ops
        import benchmark_query

        print("--- D. Query Engine Benchmarks ---")
        benchmark_query.run_all()
        print()

        print("--- E. File Operations Benchmarks ---")
        benchmark_ops.run_all()
        print()

    if args.all or args.loader:
        import benchmark_loader

        print("--- F. ML Data Loader Benchmarks ---")
        # Run loader benchmarks for each dataset
        for ds_name in ["pbmc3k", "smartseq2", "tabula_sapiens_100k"]:
            print(f"\n  Dataset: {ds_name}")
            try:
                benchmark_loader.run_dataset_benchmarks(ds_name)
            except Exception as e:
                print(f"    ERROR: {e}")
        print()

    if compression_results or read_results or write_results:
        md = generate_comprehensive_markdown(compression_results, read_results, write_results, sysinfo)
        print("=" * 60)
        print(md)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "comprehensive_benchmark.md"
        out_path.write_text(md)
        print(f"\nResults written to: {out_path}")
    else:
        print("No benchmark results generated.")


if __name__ == "__main__":
    main()
