#!/usr/bin/env python3
"""Benchmark auto-codec selection vs explicit codecs (Phase 2, Step 1).

Compares file sizes for codec=auto, none, scx1, zstd across all datasets.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

from bench_env import WORK_DIR


def get_h5ad_files():
    data_path = Path(WORK_DIR)
    if not data_path.exists():
        print(f"WARNING: Data directory not found: {WORK_DIR}")
        return []
    return sorted(data_path.glob("*.h5ad"))


def benchmark_codec(adata, codec, tmpdir):
    """Write SCX with given codec and return (size_bytes, write_time_s)."""
    import pyscx

    path = os.path.join(tmpdir, f"test_{codec}.scx")
    t0 = time.perf_counter()
    pyscx.from_anndata(adata, path, codec=codec)
    elapsed = time.perf_counter() - t0
    size = os.path.getsize(path)
    return size, elapsed


def run_all():
    import anndata

    h5ad_files = get_h5ad_files()
    if not h5ad_files:
        print("No h5ad files found. Run download_datasets.sh first.")
        return []

    codecs = ["auto", "none", "scx1", "zstd"]
    results = []

    for h5ad_path in h5ad_files:
        print(f"\n{'='*60}")
        print(f"Dataset: {h5ad_path.name}")
        h5ad_size = h5ad_path.stat().st_size

        adata = anndata.read_h5ad(str(h5ad_path))
        print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"  h5ad size: {h5ad_size / 1e6:.1f} MB")

        row = {
            "dataset": h5ad_path.stem,
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "h5ad_size": h5ad_size,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for codec in codecs:
                try:
                    size, elapsed = benchmark_codec(adata, codec, tmpdir)
                    ratio = size / h5ad_size
                    row[f"{codec}_size"] = size
                    row[f"{codec}_ratio"] = ratio
                    row[f"{codec}_time"] = elapsed
                    print(
                        f"  {codec:>5}: {size / 1e6:>8.1f} MB "
                        f"(ratio={ratio:.3f}, time={elapsed:.2f}s)"
                    )
                except Exception as e:
                    print(f"  {codec:>5}: ERROR - {e}")
                    row[f"{codec}_size"] = None
                    row[f"{codec}_ratio"] = None
                    row[f"{codec}_time"] = None

        results.append(row)

    return results


def generate_report(results):
    """Generate markdown report."""
    lines = []
    lines.append("# Auto-Codec Benchmark Results\n")
    lines.append(
        "Comparison of auto-codec selection vs explicit codecs. "
        "Auto-codec uses `select_codec()` to choose between Scx1 and Zstd "
        "based on value distribution (median <= 8 → Scx1, else → Zstd).\n"
    )

    # Main comparison table
    lines.append("## File Size by Codec\n")
    lines.append(
        "| Dataset | h5ad | Auto | None | Scx1 | Zstd | Best Manual | Auto Match? |"
    )
    lines.append(
        "|---------|------|------|------|------|------|-------------|-------------|"
    )

    for r in results:
        h5ad = r["h5ad_size"]

        def fmt(key):
            s = r.get(key)
            if s is None:
                return "N/A"
            return f"{s / 1e6:.1f} MB ({s / h5ad:.3f})"

        # Find best manual codec
        manual = {}
        for c in ["none", "scx1", "zstd"]:
            if r.get(f"{c}_ratio") is not None:
                manual[c] = r[f"{c}_ratio"]

        best_name = min(manual, key=manual.get) if manual else "N/A"
        best_ratio = manual.get(best_name, 0)

        auto_ratio = r.get("auto_ratio")
        if auto_ratio and best_ratio:
            match = "Yes" if abs(auto_ratio - best_ratio) < 0.01 else "No"
            if auto_ratio <= best_ratio + 0.001:
                match = "Yes"
        else:
            match = "N/A"

        lines.append(
            f"| {r['dataset']} | {h5ad / 1e6:.1f} MB | {fmt('auto_size')} "
            f"| {fmt('none_size')} | {fmt('scx1_size')} | {fmt('zstd_size')} "
            f"| {best_name} ({best_ratio:.3f}) | {match} |"
        )

    lines.append("")

    # Write throughput table
    lines.append("## Write Throughput by Codec\n")
    lines.append("| Dataset | Auto (s) | None (s) | Scx1 (s) | Zstd (s) |")
    lines.append("|---------|----------|----------|----------|----------|")
    for r in results:
        def tfmt(key):
            t = r.get(key)
            return f"{t:.2f}" if t else "N/A"

        lines.append(
            f"| {r['dataset']} | {tfmt('auto_time')} | {tfmt('none_time')} "
            f"| {tfmt('scx1_time')} | {tfmt('zstd_time')} |"
        )

    lines.append("")

    # Comparison with Phase 1 baselines
    lines.append("## Comparison with Phase 1 Baselines\n")
    lines.append(
        "Phase 1 defaulted to `codec=None`. Phase 2 defaults to `codec=auto`.\n"
    )

    phase1 = {
        "pbmc3k": {"none": 0.477, "scx1": 0.207, "zstd": 0.231},
        "smartseq2": {"none": 0.746, "scx1": 0.852, "zstd": 0.345},
        "tabula_sapiens_100k": {"none": 0.501, "scx1": 0.270, "zstd": 0.203},
    }

    lines.append(
        "| Dataset | Phase 1 Default (None) | Phase 2 Default (Auto) | Improvement | Best Manual |"
    )
    lines.append(
        "|---------|----------------------|----------------------|-------------|-------------|"
    )
    for r in results:
        ds = r["dataset"]
        p1 = phase1.get(ds, {})
        p1_none = p1.get("none", 0)
        auto_ratio = r.get("auto_ratio", 0)
        if p1_none and auto_ratio:
            improvement = f"{(1 - auto_ratio / p1_none) * 100:.0f}% smaller"
        else:
            improvement = "N/A"

        best_manual = min(p1.values()) if p1 else 0
        lines.append(
            f"| {ds} | {p1_none:.3f} | {auto_ratio:.3f} | {improvement} | {best_manual:.3f} |"
        )

    lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    lines.append(
        "### Auto-Codec Selection Algorithm\n"
        "\n"
        "The auto-codec algorithm samples up to 10K raw value bytes, decodes them to u32,\n"
        "and computes the floor median:\n"
        "- **Float/Float16 values** → always Zstd (Rice only handles integers)\n"
        "- **Integer values, median ≤ 8** → Scx1 (Rice coding optimal for small UMI counts)\n"
        "- **Integer values, median > 8** → Zstd (LZ77 dictionary wins for larger values)\n"
    )
    lines.append(
        "### Key Findings\n"
    )

    return "\n".join(lines)


def main():
    results = run_all()
    if not results:
        return

    report = generate_report(results)

    # Print report
    print(f"\n{'='*60}")
    print(report)

    # Save report
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "auto_codec_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport written to: {report_path}")

    # Save raw JSON for future reference
    json_path = results_dir / "auto_codec_benchmark.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Raw data written to: {json_path}")


if __name__ == "__main__":
    ensure_release_build()
    main()
