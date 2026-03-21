#!/usr/bin/env python3
"""
SCX Query Engine Benchmark (Phase 2 Step 4, Phase H)

Runs Rust benchmark tests via `cargo test` and collects timing results.
Generates a markdown report with shard skip rates, query latency,
gene projection speedup, and fused ops performance.

Usage:
    python benchmarks/scripts/benchmark_query.py
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
    """Run the Rust query engine benchmark tests and capture output."""
    result = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "scx-engine",
            "bench_query_",
            "--release",
            "--",
            "--nocapture",
            "--ignored",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=900,
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
        "# SCX Query Engine Benchmark Report",
        "",
        "## Test Environment",
        f"- **CPU**: {sysinfo['cpu']}",
        f"- **RAM**: {sysinfo['ram']}",
        f"- **OS**: {sysinfo['os']}",
        "- **SCX version**: 0.1.0 (Phase 2 Step 4)",
        "- **Build**: `--release` profile",
        "- **Dataset**: Synthetic (100K cells × 30K genes, 7 shards, 5 nnz/row)",
        "",
    ]

    # 1. Shard Skip Rate (H2)
    skip = results.get("shard_skip_rate")
    lines.extend([
        "## 1. Shard Skip Rate (H2)",
        "",
        "Measures predicate pushdown efficiency by cell type. Target: ≥50% average",
        "skip rate for selective queries (ROADMAP.md Go/No-Go criterion).",
        "",
        "| Cell Type | Total Shards | Skipped | Skip Rate | Returned Cells | Expected |",
        "|-----------|-------------|---------|-----------|----------------|----------|",
    ])
    if skip:
        for entry in skip.get("per_cell_type", []):
            ct = entry["cell_type"]
            total = entry["total_shards"]
            skipped = entry["skipped_shards"]
            rate = entry["skip_rate_pct"]
            returned = entry["returned_cells"]
            expected = entry["expected_cells"]
            lines.append(
                f"| {ct} | {total} | {skipped} | {rate:.1f}% | {returned:,} | {expected:,} |"
            )
        avg = skip.get("avg_skip_rate_pct", 0)
        passed = skip.get("pass", False)
        lines.extend([
            "",
            f"**Average skip rate: {avg:.1f}%** — {'PASS ✅' if passed else 'FAIL ❌'} (target ≥50%)",
            "",
        ])
    else:
        lines.append("| *(benchmark not run)* | — | — | — | — | — |")
        lines.append("")

    # 2. Query Latency (H3)
    latency = results.get("query_latency")
    lines.extend([
        "## 2. Query Latency (H3)",
        "",
        "End-to-end `filter_obs().collect()` time over 10 runs.",
        "",
        "| Query Type | Predicate | Median | P95 | P99 | Min | Max | Target | Pass? |",
        "|-----------|-----------|--------|-----|-----|-----|-----|--------|-------|",
    ])
    if latency:
        for qtype in ["selective_query", "broad_query"]:
            q = latency.get(qtype, {})
            label = "Selective" if qtype == "selective_query" else "Broad"
            pred = q.get("predicate", "")
            med = q.get("median_s", 0)
            p95 = q.get("p95_s", 0)
            p99 = q.get("p99_s", 0)
            mn = q.get("min_s", 0)
            mx = q.get("max_s", 0)
            target = q.get("target_s", 0)
            passed = q.get("pass", False)
            lines.append(
                f"| {label} | `{pred}` | {med:.3f}s | {p95:.3f}s | {p99:.3f}s "
                f"| {mn:.3f}s | {mx:.3f}s | <{target}s | {'PASS ✅' if passed else 'FAIL ❌'} |"
            )
        lines.append("")
    else:
        lines.append("| *(benchmark not run)* | — | — | — | — | — | — | — | — |")
        lines.append("")

    # 3. Query vs Full Read (H4)
    vs = results.get("query_vs_full_read")
    lines.extend([
        "## 3. Query vs Full Read (H4)",
        "",
        "Compare SCX query with pushdown vs reading all data then filtering",
        "(simulates the AnnData subsetting pattern `adata = read_h5ad(); adata[mask]`).",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    if vs:
        lines.extend([
            f"| Predicate | `{vs.get('predicate', '')}` |",
            f"| Query (pushdown) median | {vs.get('query_median_s', 0):.3f}s |",
            f"| Full read + filter median | {vs.get('full_read_median_s', 0):.3f}s |",
            f"| Speedup | {vs.get('speedup', 0):.2f}× |",
            f"| Query wins? | {'Yes ✅' if vs.get('query_wins', False) else 'No ❌'} |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — |")
    lines.append("")

    # 4. Gene Projection (H5)
    proj = results.get("gene_projection")
    lines.extend([
        "## 4. Gene Projection Speedup (H5)",
        "",
        "Measures `select_genes(2000 HVG)` on a 30K-gene dataset.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    if proj:
        lines.extend([
            f"| Total genes | {proj.get('n_genes_total', 0):,} |",
            f"| Projected genes | {proj.get('n_genes_projected', 0):,} |",
            f"| Data reduction ratio | {proj.get('data_reduction_ratio', 0):.1f}× |",
            f"| All genes median | {proj.get('all_genes_median_s', 0):.3f}s |",
            f"| Projected median | {proj.get('projected_median_s', 0):.3f}s |",
            f"| Speedup | {proj.get('speedup', 0):.2f}× |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — |")
    lines.append("")

    # 5. Fused Ops (H6)
    fused = results.get("fused_ops")
    lines.extend([
        "## 5. Fused vs Sequential normalize+log1p (H6)",
        "",
        "Compare fused single-pass normalize+log1p against separate passes.",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if fused:
        speedup = fused.get("speedup", 0)
        target = fused.get("target_speedup", 1.5)
        passed = fused.get("pass", False)
        lines.extend([
            f"| Fused median | {fused.get('fused_median_s', 0):.3f}s | — | — |",
            f"| Sequential median | {fused.get('sequential_median_s', 0):.3f}s | — | — |",
            f"| Speedup | {speedup:.2f}× | ≥{target}× | {'PASS ✅' if passed else 'FAIL ❌'} |",
        ])
    else:
        lines.append("| *(benchmark not run)* | — | — | — |")
    lines.append("")

    # Summary
    lines.extend([
        "## Summary",
        "",
        "### Go/No-Go Criterion Validation",
        "",
        "| Criterion | Result |",
        "|-----------|--------|",
    ])

    checks = []
    if skip:
        avg = skip.get("avg_skip_rate_pct", 0)
        checks.append(("Predicate pushdown ≥50% shard skip rate", avg >= 50.0))
    if latency:
        sel = latency.get("selective_query", {})
        broad = latency.get("broad_query", {})
        checks.append(("Selective query <5s", sel.get("pass", False)))
        checks.append(("Broad query <15s", broad.get("pass", False)))

    for name, passed in checks:
        lines.append(f"| {name} | {'PASS ✅' if passed else 'FAIL ❌'} |")
    lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "The benchmarks use synthetic data (100K cells × 30K genes, 7 shards) to provide",
        "deterministic, reproducible results. Real-world performance will vary based on",
        "data distribution, sparsity, codec selection, and disk I/O characteristics.",
        "",
        "- **Shard skip rate** depends on how non-uniformly cell types are distributed",
        "  across shards. The synthetic fixture is designed with intentionally non-uniform",
        "  distribution to demonstrate pushdown. Real datasets with random row ordering",
        "  (e.g., from CELLxGENE Census) may have lower skip rates until the writer",
        "  implements cell-type-aware shard assignment.",
        "- **Gene projection** reduces the column count from 30K to 2K (15× reduction).",
        "  The speedup comes from skipping decode of non-HVG column entries during the",
        "  post-decode projection step.",
        "- **Fused ops** combine normalize and log1p into a single CSR row scan. The",
        "  speedup is modest because I/O dominates over compute for in-memory operations.",
        "  The real benefit is cache efficiency on very large datasets.",
        "",
    ])

    return "\n".join(lines)


def main():
    print("Running SCX query engine benchmarks...")
    print("=" * 60)

    output = run_rust_benchmark()
    results = parse_benchmark_output(output)

    if not results:
        print("ERROR: No benchmark results found in output.", file=sys.stderr)
        print("Make sure the bench_query_ tests exist and are marked #[ignore].", file=sys.stderr)
        sys.exit(1)

    sysinfo = get_system_info()
    report = generate_report(results, sysinfo)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "query_engine_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")

    # Also save raw results
    raw_path = RESULTS_DIR / "query_engine_benchmark.json"
    raw_path.write_text(json.dumps(results, indent=2))
    print(f"Raw results written to {raw_path}")


if __name__ == "__main__":
    main()
