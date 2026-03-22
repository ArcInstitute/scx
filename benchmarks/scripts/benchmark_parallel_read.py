#!/usr/bin/env python3
"""Benchmark parallel shard decode: measure read time across thread counts.

For each dataset, converts to SCX with auto-codec, then reads with
RAYON_NUM_THREADS=1,2,4,8 via subprocess (to ensure fresh thread pool).
Compares against h5ad baseline. Outputs markdown + JSON to benchmarks/results/.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

DATA_DIR = os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx")
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
THREAD_COUNTS = [1, 2, 4, 8]
N_WARMUP = 1
N_REPEATS = 3


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# Inline script run as subprocess to ensure fresh rayon thread pool
_READ_SCRIPT = r"""
import os, sys, time, json, tracemalloc
sys.path.insert(0, os.environ["PYSCX_PATH"])
import pyscx

scx_path = sys.argv[1]
n_warmup = int(sys.argv[2])
n_repeats = int(sys.argv[3])

# Warmup
for _ in range(n_warmup):
    pyscx.open(scx_path).to_anndata()

# Timed reads
times = []
for _ in range(n_repeats):
    t0 = time.perf_counter()
    pyscx.open(scx_path).to_anndata()
    times.append(time.perf_counter() - t0)

# Peak memory
tracemalloc.start()
pyscx.open(scx_path).to_anndata()
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

result = {"times": times, "median_s": sorted(times)[len(times) // 2], "peak_mb": peak / 1e6}
print(json.dumps(result))
"""


def convert_to_scx(h5ad_path, scx_path):
    """Convert h5ad to SCX with auto-codec."""
    import anndata
    import pyscx

    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, scx_path, codec="auto")
    del adata


def benchmark_h5ad(h5ad_path):
    """Get h5ad read baseline."""
    import anndata

    for _ in range(N_WARMUP):
        anndata.read_h5ad(str(h5ad_path))

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        anndata.read_h5ad(str(h5ad_path))
        times.append(time.perf_counter() - t0)
    return _median(times)


def benchmark_scx_threaded(scx_path, n_threads):
    """Run SCX read in subprocess with specific RAYON_NUM_THREADS."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(n_threads)
    env["PYSCX_PATH"] = str(PROJECT_ROOT / "pyscx")

    python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    proc = subprocess.run(
        [python, "-c", _READ_SCRIPT, str(scx_path), str(N_WARMUP), str(N_REPEATS)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {proc.stderr}")

    # Parse last line of stdout (skip any import warnings)
    for line in reversed(proc.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON output found: {proc.stdout}")


def get_shard_count(scx_path):
    """Get number of shards in an SCX file."""
    import pyscx

    reader = pyscx.open(scx_path)
    return reader.shard_count


def run_all():
    """Run parallel read benchmarks on all datasets."""
    data_path = Path(DATA_DIR)
    h5ad_files = sorted(data_path.glob("*.h5ad")) if data_path.exists() else []
    if not h5ad_files:
        print(f"No h5ad files found in {DATA_DIR}. Run download_datasets.sh first.")
        return

    all_results = []

    for h5ad_path in h5ad_files:
        dataset = h5ad_path.stem
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, f"{dataset}.scx")

            # Convert
            print("  Converting to SCX (auto-codec)...")
            convert_to_scx(h5ad_path, scx_path)

            n_shards = get_shard_count(scx_path)
            print(f"  Shards: {n_shards}")

            # h5ad baseline
            print("  Benchmarking h5ad baseline...")
            h5ad_time = benchmark_h5ad(h5ad_path)
            print(f"  h5ad: {h5ad_time:.3f}s")

            # SCX with different thread counts
            dataset_result = {
                "dataset": dataset,
                "n_shards": n_shards,
                "h5ad_s": round(h5ad_time, 3),
                "threads": {},
            }

            for n_threads in THREAD_COUNTS:
                print(f"  Benchmarking SCX (threads={n_threads})...")
                try:
                    result = benchmark_scx_threaded(scx_path, n_threads)
                    t = result["median_s"]
                    speedup_vs_1t = dataset_result["threads"].get("1", {}).get("median_s", t) / t if t > 0 else 0
                    speedup_vs_h5ad = h5ad_time / t if t > 0 else 0
                    dataset_result["threads"][str(n_threads)] = {
                        "median_s": round(t, 3),
                        "peak_mb": round(result["peak_mb"], 1),
                        "speedup_vs_1t": round(speedup_vs_1t, 2),
                        "speedup_vs_h5ad": round(speedup_vs_h5ad, 2),
                    }
                    print(f"    {t:.3f}s (vs 1-thread: {speedup_vs_1t:.2f}x, vs h5ad: {speedup_vs_h5ad:.2f}x)")
                except Exception as e:
                    print(f"    ERROR: {e}")

            all_results.append(dataset_result)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "parallel_read_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON results: {json_path}")

    # Generate markdown report
    md_path = RESULTS_DIR / "parallel_read_benchmark.md"
    with open(md_path, "w") as f:
        f.write("# Parallel Shard Decode Benchmark\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Warmup: {N_WARMUP}, Repeats: {N_REPEATS} (median)\n")
        f.write(f"- Thread counts: {THREAD_COUNTS}\n")
        f.write(f"- Codec: auto\n\n")

        f.write("## Results\n\n")
        f.write("### Read Time by Thread Count\n\n")
        f.write("| Dataset | Shards | h5ad (s) | 1 thread | 2 threads | 4 threads | 8 threads |\n")
        f.write("|---------|--------|----------|----------|-----------|-----------|----------|\n")
        for r in all_results:
            row = f"| {r['dataset']} | {r['n_shards']} | {r['h5ad_s']:.3f} "
            for tc in THREAD_COUNTS:
                t = r["threads"].get(str(tc), {})
                if t:
                    row += f"| {t['median_s']:.3f} "
                else:
                    row += "| - "
            f.write(row + "|\n")

        f.write("\n### Speedup vs Single Thread\n\n")
        f.write("| Dataset | Shards | 1 thread | 2 threads | 4 threads | 8 threads |\n")
        f.write("|---------|--------|----------|-----------|-----------|----------|\n")
        for r in all_results:
            row = f"| {r['dataset']} | {r['n_shards']} "
            for tc in THREAD_COUNTS:
                t = r["threads"].get(str(tc), {})
                if t:
                    row += f"| {t['speedup_vs_1t']:.2f}x "
                else:
                    row += "| - "
            f.write(row + "|\n")

        f.write("\n### Peak Memory (MB)\n\n")
        f.write("| Dataset | 1 thread | 2 threads | 4 threads | 8 threads |\n")
        f.write("|---------|----------|-----------|-----------|----------|\n")
        for r in all_results:
            row = f"| {r['dataset']} "
            for tc in THREAD_COUNTS:
                t = r["threads"].get(str(tc), {})
                if t:
                    row += f"| {t['peak_mb']:.1f} "
                else:
                    row += "| - "
            f.write(row + "|\n")

        # Criteria check
        f.write("\n## Phase 2 Criteria Check\n\n")
        for r in all_results:
            f.write(f"### {r['dataset']} ({r['n_shards']} shards)\n\n")
            t1 = r["threads"].get("1", {})
            t8 = r["threads"].get("8", {})
            if t1 and t8:
                speedup = t1["median_s"] / t8["median_s"] if t8["median_s"] > 0 else 0
                if r["n_shards"] > 1:
                    met = "MET" if speedup >= 2.0 else "NOT MET"
                    f.write(f"- Multi-shard >=2x speedup: {speedup:.2f}x — **{met}**\n")
                else:
                    regression = abs(t8["median_s"] - t1["median_s"]) / t1["median_s"] * 100 if t1["median_s"] > 0 else 0
                    met = "MET" if regression <= 10 else "NOT MET"
                    f.write(f"- Single-shard no regression: {regression:.1f}% — **{met}**\n")
                mem_ratio = t8["peak_mb"] / t1["peak_mb"] if t1["peak_mb"] > 0 else 0
                met = "MET" if mem_ratio <= 2.0 else "NOT MET"
                f.write(f"- Peak memory within 2x: {mem_ratio:.2f}x — **{met}**\n")
            f.write("\n")

    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    ensure_release_build()
    run_all()
