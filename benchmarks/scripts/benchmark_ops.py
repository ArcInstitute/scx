#!/usr/bin/env python3
"""
SCX File Operations Benchmark (Phase 2 Step 3)

Runs Rust integration benchmarks via `cargo test` and collects timing results.
Since pyscx ops bindings don't exist yet (Step 6-7), this script invokes
the Rust benchmark test and parses its output.

Measures:
1. Append overhead: time to append 10K cells, file size vs fresh write
2. Compact efficiency: after 3 appends, compact vs fresh write size
3. Deletion vector read overhead: mark 10% deleted, read time with/without DVs
4. Merge throughput: merge 3 copies, measure MB/s

Usage:
    python benchmarks/scripts/benchmark_ops.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def run_rust_benchmark():
    """Run the Rust ops benchmark test and capture output."""
    result = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "scx-ops",
            "bench_ops_",
            "--release",
            "--",
            "--nocapture",
            "--ignored",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_benchmark_output(output: str) -> dict:
    """Parse JSON benchmark results from test output."""
    results = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "benchmark" in data:
                    results[data["benchmark"]] = data
            except json.JSONDecodeError:
                pass
    return results


def get_system_info():
    """Collect system information."""
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    mem = "unknown"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    mem = f"{kb // (1024 * 1024)} GB"
                    break
    except Exception:
        pass

    uname = "unknown"
    try:
        uname = subprocess.check_output(["uname", "-r"], text=True).strip()
    except Exception:
        pass

    return {"cpu": cpu, "ram": mem, "os": f"Linux {uname}"}


def generate_report(results: dict, sysinfo: dict) -> str:
    """Generate markdown benchmark report."""
    lines = [
        "# SCX File Operations Benchmark Report",
        "",
        "## Test Environment",
        f"- **CPU**: {sysinfo['cpu']}",
        f"- **RAM**: {sysinfo['ram']}",
        f"- **OS**: {sysinfo['os']}",
        "- **SCX version**: 0.1.0 (Phase 2 Step 3)",
        "- **Build**: `--release` profile",
        "- **Datasets**: Synthetic (10K cells × 20K genes, 2 nnz/row for deterministic sizing)",
        "",
    ]

    # 1. Append overhead
    append = results.get("append_10k")
    lines.extend([
        "## 1. Append Overhead",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if append:
        time_s = append.get("time_ms", 0) / 1000
        pass_time = "Yes" if time_s < 2.0 else "No"
        bloat = append.get("bloat_pct", 0)
        pass_bloat = "Yes" if bloat <= 5.0 else "No"
        lines.extend([
            f"| Append 10K cells | {time_s:.3f}s | < 2s | {pass_time} |",
            f"| File size bloat | {bloat:.1f}% | <= 5% | {pass_bloat} |",
            f"| Fresh write size | {append.get('fresh_size_bytes', 0):,} bytes | — | — |",
            f"| After append size | {append.get('append_size_bytes', 0):,} bytes | — | — |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — | — | — |")
    lines.append("")

    # 2. Compact efficiency
    compact = results.get("compact_after_3_appends")
    lines.extend([
        "## 2. Compact Efficiency",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if compact:
        ratio = compact.get("compact_vs_fresh", 0)
        pass_compact = "Yes" if ratio <= 1.05 else "No"
        lines.extend([
            f"| After 3 appends | {compact.get('after_appends_bytes', 0):,} bytes | — | — |",
            f"| After compact | {compact.get('after_compact_bytes', 0):,} bytes | — | — |",
            f"| Fresh equivalent | {compact.get('fresh_size_bytes', 0):,} bytes | — | — |",
            f"| Compact / Fresh | {ratio:.3f} | <= 1.05 | {pass_compact} |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — | — | — |")
    lines.append("")

    # 3. Deletion vector read overhead
    dv = results.get("dv_read_overhead")
    lines.extend([
        "## 3. Deletion Vector Read Overhead",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if dv:
        overhead = dv.get("overhead_pct", 0)
        pass_dv = "Yes" if overhead <= 20.0 else "No"
        lines.extend([
            f"| Read time (no DV) | {dv.get('read_no_dv_ms', 0):.2f} ms | — | — |",
            f"| Read time (10% DV) | {dv.get('read_dv_ms', 0):.2f} ms | — | — |",
            f"| Overhead | {overhead:.1f}% | <= 20% | {pass_dv} |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — | — | — |")
    lines.append("")

    # 4. Merge throughput
    merge = results.get("merge_throughput")
    lines.extend([
        "## 4. Merge Throughput",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if merge:
        throughput = merge.get("throughput_mb_s", 0)
        pass_merge = "Yes" if throughput >= 50 else "No"
        lines.extend([
            f"| Total input size | {merge.get('total_input_mb', 0):.1f} MB | — | — |",
            f"| Time | {merge.get('time_ms', 0) / 1000:.3f}s | — | — |",
            f"| Throughput | {throughput:.1f} MB/s | >= 50 MB/s | {pass_merge} |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — | — | — |")
    lines.append("")

    # Summary
    all_pass = True
    checks = []
    if append:
        checks.append(("Append time < 2s", append.get("time_ms", 0) / 1000 < 2.0))
        checks.append(("Append bloat <= 5%", append.get("bloat_pct", 0) <= 5.0))
    if compact:
        checks.append(("Compact/fresh <= 1.05", compact.get("compact_vs_fresh", 0) <= 1.05))
    if dv:
        checks.append(("DV overhead <= 20%", dv.get("overhead_pct", 0) <= 20.0))
    if merge:
        checks.append(("Merge >= 50 MB/s", merge.get("throughput_mb_s", 0) >= 50))

    lines.extend([
        "## Summary",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ])
    for name, passed in checks:
        all_pass = all_pass and passed
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "The benchmarks use synthetic data (10K cells × 20K genes, 2 nnz/row) to provide",
        "deterministic, reproducible results. Real-world performance will vary based on",
        "data sparsity, value distribution, and disk I/O characteristics.",
        "",
        "- **Append overhead**: The append operation writes new shards at EOF and commits",
        "  atomically via header pwrite. Overhead comes from re-writing the obs metadata",
        "  (Arrow IPC concatenation) and the full catalog. File size bloat is from",
        "  duplicate obs/catalog sections that compact removes.",
        "- **Compact efficiency**: Compaction rewrites the file from scratch, eliminating",
        "  stale sections. The compacted file should be within a few percent of a fresh",
        "  write with the same data.",
        "- **DV read overhead**: Deletion vectors add a bitmap check per row during",
        "  filtered reads. The overhead is minimal for small deletion fractions.",
        "- **Merge throughput**: Merge decodes all input shards and re-encodes them into",
        "  the output file. Throughput depends on codec complexity and I/O speed.",
        "",
    ])

    return "\n".join(lines)


def main():
    print("Running SCX file operations benchmarks...")
    print("=" * 60)

    output = run_rust_benchmark()
    results = parse_benchmark_output(output)

    if not results:
        print("ERROR: No benchmark results found in output.", file=sys.stderr)
        print("Make sure the bench_ops_ tests exist and are marked #[ignore].", file=sys.stderr)
        sys.exit(1)

    sysinfo = get_system_info()
    report = generate_report(results, sysinfo)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "ops_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")

    # Also save raw results
    raw_path = RESULTS_DIR / "ops_benchmark.json"
    raw_path.write_text(json.dumps(results, indent=2))
    print(f"Raw results written to {raw_path}")


if __name__ == "__main__":
    main()
