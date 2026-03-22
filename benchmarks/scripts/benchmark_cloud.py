#!/usr/bin/env python3
"""
SCX Cloud Benchmark (Phase 2-CLOUD, Phase J)

End-to-end benchmarks for cloud operations, measuring:
1. Push throughput — scx push to GCS
2. Pull throughput — scx pull full dataset from GCS
3. Filtered pull — scx pull --filter latency and shard skip rate
4. Cloud-optimize overhead — time and size overhead
5. Explode + pack round-trip — local explode/pack cycle
6. Streaming pull vs naive — scx pull vs gsutil cp + pack
7. Cloud reader metadata latency — scx info on cloud-ready file

Requires:
- GOOGLE_APPLICATION_CREDENTIALS pointing to a GCS service account key
- tabula_sapiens_100k.scx in $SCX_DATA_DIR (default: /scratch/ctc/nickyoungblut/scx)
- Release-built scx-cli with cloud feature

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/c-tc-429521-6f6f5b8ccd93.json \\
        python benchmarks/scripts/benchmark_cloud.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = Path(os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx"))
GCS_PREFIX = os.environ.get("GCS_TEST_BUCKET", "gs://arc-ctc-nextflow/scx-test")
DATASET = "tabula_sapiens_100k"
N_RUNS = 3
FILTER_EXPR = "cell_type == 'macrophage'"  # ~20% of tabula_sapiens_100k


# ---------------------------------------------------------------------------
# Helpers (matching benchmark_cli.py patterns)
# ---------------------------------------------------------------------------

def find_scx_binary():
    """Find or build the release scx-cli binary with cloud feature."""
    candidates = [
        REPO_ROOT / "target" / "release" / "scx-cli",
        REPO_ROOT / "target" / "release" / "scx",
    ]
    # Check if existing binary has cloud feature
    for path in candidates:
        if path.exists():
            result = subprocess.run(
                [str(path), "pull", "--help"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return str(path)

    # Build with cloud feature
    print("Building scx-cli with cloud feature in release mode...")
    result = subprocess.run(
        ["cargo", "build", "-p", "scx-cli", "--features", "cloud", "--release"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"ERROR: cargo build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    for path in candidates:
        if path.exists():
            return str(path)
    print("ERROR: could not find scx-cli binary after build", file=sys.stderr)
    sys.exit(1)


def run_scx(binary, args, check=True, timeout=600):
    """Run the scx CLI binary and return (stdout, stderr, elapsed_s)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [binary] + args,
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if check and result.returncode != 0:
        raise RuntimeError(
            f"scx {' '.join(args)} failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
    return result.stdout, result.stderr, elapsed


def run_cmd(args, check=True, timeout=600):
    """Run an arbitrary command and return (stdout, stderr, elapsed_s)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if check and result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
    return result.stdout, result.stderr, elapsed


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


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


def gcs_path_exists(gcs_url):
    """Check if a GCS path exists via gsutil ls."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", gcs_url],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def dir_size_bytes(path):
    """Total size of all files in a directory."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_gcs_data(binary, scx_path):
    """Ensure tabula_sapiens_100k.scxd/ exists on GCS."""
    gcs_dest = f"{GCS_PREFIX}/{DATASET}.scxd/"
    if gcs_path_exists(gcs_dest):
        print(f"  GCS data already exists: {gcs_dest}")
        return gcs_dest

    print(f"  Uploading {scx_path} to {gcs_dest} ...")
    stdout, stderr, elapsed = run_scx(binary, [
        "push", str(scx_path), gcs_dest,
    ], timeout=600)
    print(f"  Upload done in {elapsed:.1f}s: {stdout.strip()}")
    return gcs_dest


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_push(binary, scx_path, file_size):
    """Benchmark: scx push throughput to GCS."""
    print("\n=== 1. Push Throughput ===")
    times = []
    throughputs = []

    for i in range(N_RUNS):
        gcs_dest = f"{GCS_PREFIX}/bench_push_{i}.scxd/"
        stdout, stderr, elapsed = run_scx(binary, [
            "push", str(scx_path), gcs_dest,
        ], timeout=600)
        throughput = (file_size / 1e6) / elapsed if elapsed > 0 else 0
        times.append(elapsed)
        throughputs.append(throughput)
        print(f"  Run {i+1}: {elapsed:.2f}s ({throughput:.1f} MB/s) — {stdout.strip()}")

    return {
        "benchmark": "push",
        "median_s": round(_median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "median_throughput_mbps": round(_median(throughputs), 1),
        "file_size_mb": round(file_size / 1e6, 1),
        "pass": _median(throughputs) > 50,
    }


def bench_pull(binary, gcs_source, dataset_info):
    """Benchmark: scx pull full dataset from GCS."""
    print("\n=== 2. Pull Throughput ===")
    times = []
    throughputs = []
    file_size = dataset_info["file_size"]

    for i in range(N_RUNS):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "pulled.scx")
            stdout, stderr, elapsed = run_scx(binary, [
                "pull", gcs_source, output,
            ], timeout=600)
            throughput = (file_size / 1e6) / elapsed if elapsed > 0 else 0
            times.append(elapsed)
            throughputs.append(throughput)
            print(f"  Run {i+1}: {elapsed:.2f}s ({throughput:.1f} MB/s) — {stdout.strip()}")

            # Validate on first run
            if i == 0:
                info_out, _, _ = run_scx(binary, ["info", output, "--json"])
                info = json.loads(info_out)
                assert info["n_obs"] == dataset_info["n_obs"], \
                    f"n_obs mismatch: {info['n_obs']} vs {dataset_info['n_obs']}"
                assert info["n_vars"] == dataset_info["n_vars"], \
                    f"n_vars mismatch: {info['n_vars']} vs {dataset_info['n_vars']}"
                print(f"  Validated: {info['n_obs']:,} obs, {info['n_vars']:,} vars")

    return {
        "benchmark": "pull",
        "median_s": round(_median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "median_throughput_mbps": round(_median(throughputs), 1),
        "file_size_mb": round(file_size / 1e6, 1),
        "pass": _median(throughputs) > 50,
    }


def bench_filtered_pull(binary, gcs_source, dataset_info):
    """Benchmark: scx pull --filter latency and shard skip rate."""
    print("\n=== 3. Filtered Pull ===")
    times = []
    results_data = []

    for i in range(N_RUNS):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "filtered.scx")
            stdout, stderr, elapsed = run_scx(binary, [
                "pull", gcs_source, output,
                "--filter", FILTER_EXPR,
            ], timeout=600)
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s — {stdout.strip()}")

            # Parse CLI output: "Selective pull ... (D/T shards, C cells, X MB downloaded, Y MB saved)"
            m = re.search(
                r"\((\d+)/(\d+) shards, (\d+) cells, ([\d.]+) MB downloaded, ([\d.]+) MB saved\)",
                stdout,
            )
            if m:
                downloaded_shards = int(m.group(1))
                total_shards = int(m.group(2))
                matching_cells = int(m.group(3))
                mb_downloaded = float(m.group(4))
                mb_saved = float(m.group(5))
                results_data.append({
                    "downloaded_shards": downloaded_shards,
                    "total_shards": total_shards,
                    "matching_cells": matching_cells,
                    "mb_downloaded": mb_downloaded,
                    "mb_saved": mb_saved,
                })

    r = results_data[0] if results_data else {}
    skip_rate = 0
    if r:
        skipped = r["total_shards"] - r["downloaded_shards"]
        skip_rate = skipped / r["total_shards"] if r["total_shards"] > 0 else 0

    return {
        "benchmark": "filtered_pull",
        "median_s": round(_median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "filter_expr": FILTER_EXPR,
        "matching_cells": r.get("matching_cells", 0),
        "downloaded_shards": r.get("downloaded_shards", 0),
        "total_shards": r.get("total_shards", 0),
        "skip_rate": round(skip_rate, 3),
        "mb_downloaded": r.get("mb_downloaded", 0),
        "mb_saved": r.get("mb_saved", 0),
        "pass": skip_rate > 0,
    }


def bench_cloud_optimize(binary, scx_path, file_size):
    """Benchmark: scx cloud-optimize overhead."""
    print("\n=== 4. Cloud Optimize ===")
    times = []
    size_overheads = []

    for i in range(N_RUNS):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy input
            work = os.path.join(tmpdir, "input.scx")
            shutil.copy2(str(scx_path), work)
            output = os.path.join(tmpdir, "optimized.scx")

            stdout, stderr, elapsed = run_scx(binary, [
                "cloud-optimize", work, "--output", output,
            ])
            times.append(elapsed)

            out_size = os.path.getsize(output)
            overhead_pct = ((out_size - file_size) / file_size) * 100
            size_overheads.append(overhead_pct)
            print(f"  Run {i+1}: {elapsed:.2f}s, overhead {overhead_pct:.2f}%")

    return {
        "benchmark": "cloud_optimize",
        "median_s": round(_median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "input_size_mb": round(file_size / 1e6, 1),
        "median_overhead_pct": round(_median(size_overheads), 2),
        "pass": _median(size_overheads) < 5.0 and _median(times) < 5.0,
    }


def bench_explode_pack(binary, scx_path, dataset_info, file_size):
    """Benchmark: scx explode + scx pack round-trip."""
    print("\n=== 5. Explode + Pack Round-trip ===")
    explode_times = []
    pack_times = []
    total_times = []

    for i in range(N_RUNS):
        with tempfile.TemporaryDirectory() as tmpdir:
            exploded = os.path.join(tmpdir, "exploded.scxd")
            repacked = os.path.join(tmpdir, "repacked.scx")

            # Explode
            _, _, t_explode = run_scx(binary, [
                "explode", str(scx_path), exploded,
            ])
            explode_times.append(t_explode)

            # Pack
            _, _, t_pack = run_scx(binary, [
                "pack", exploded, repacked,
            ])
            pack_times.append(t_pack)
            total_times.append(t_explode + t_pack)

            print(f"  Run {i+1}: explode {t_explode:.2f}s + pack {t_pack:.2f}s"
                  f" = {t_explode + t_pack:.2f}s")

            # Validate on first run
            if i == 0:
                info_out, _, _ = run_scx(binary, ["info", repacked, "--json"])
                info = json.loads(info_out)
                assert info["n_obs"] == dataset_info["n_obs"], \
                    f"n_obs mismatch: {info['n_obs']} vs {dataset_info['n_obs']}"
                assert info["n_vars"] == dataset_info["n_vars"], \
                    f"n_vars mismatch"
                assert info["nnz"] == dataset_info["nnz"], \
                    f"nnz mismatch: {info['nnz']} vs {dataset_info['nnz']}"
                print(f"  Validated: {info['n_obs']:,} obs, {info['n_vars']:,} vars,"
                      f" {info['nnz']:,} nnz")

    return {
        "benchmark": "explode_pack",
        "explode_median_s": round(_median(explode_times), 3),
        "pack_median_s": round(_median(pack_times), 3),
        "total_median_s": round(_median(total_times), 3),
        "total_min_s": round(min(total_times), 3),
        "total_max_s": round(max(total_times), 3),
        "file_size_mb": round(file_size / 1e6, 1),
        "pass": _median(total_times) < 30.0,
    }


def bench_streaming_vs_naive(binary, gcs_source, pull_result):
    """Benchmark: streaming pull vs naive (gsutil cp -r + pack)."""
    print("\n=== 6. Streaming Pull vs Naive ===")

    # Streaming result is reused from bench_pull
    streaming_median = pull_result["median_s"]

    # Naive: gsutil cp -r + pack
    naive_times = []
    for i in range(N_RUNS):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_scxd = os.path.join(tmpdir, "naive.scxd")
            packed = os.path.join(tmpdir, "naive.scx")

            # gsutil cp -r
            t0 = time.perf_counter()
            _, _, t_gsutil = run_cmd([
                "gsutil", "-m", "cp", "-r", gcs_source, local_scxd,
            ], timeout=600)

            # pack
            _, _, t_pack = run_scx(binary, ["pack", local_scxd, packed], timeout=600)

            total = time.perf_counter() - t0
            naive_times.append(total)
            print(f"  Naive run {i+1}: gsutil {t_gsutil:.2f}s + pack {t_pack:.2f}s"
                  f" = {total:.2f}s")

    naive_median = _median(naive_times)
    speedup = naive_median / streaming_median if streaming_median > 0 else 0

    print(f"  Streaming median: {streaming_median:.2f}s")
    print(f"  Naive median:     {naive_median:.2f}s")
    print(f"  Speedup:          {speedup:.2f}x")

    return {
        "benchmark": "streaming_vs_naive",
        "streaming_median_s": round(streaming_median, 3),
        "naive_median_s": round(naive_median, 3),
        "naive_gsutil_times": [round(t, 3) for t in naive_times],
        "speedup": round(speedup, 2),
    }


def bench_metadata_latency(binary, scx_path, file_size):
    """Benchmark: metadata query latency on cloud-ready file."""
    print("\n=== 7. Metadata Latency (cloud-ready file) ===")

    # First create a cloud-optimized file
    with tempfile.TemporaryDirectory() as tmpdir:
        optimized = os.path.join(tmpdir, "cloud_ready.scx")
        run_scx(binary, ["cloud-optimize", str(scx_path), "--output", optimized])

        # Benchmark scx info on cloud-ready vs original
        cr_times = []
        orig_times = []
        for i in range(N_RUNS):
            _, _, t_cr = run_scx(binary, ["info", optimized, "--json"])
            cr_times.append(t_cr)

            _, _, t_orig = run_scx(binary, ["info", str(scx_path), "--json"])
            orig_times.append(t_orig)

        print(f"  Cloud-ready info median: {_median(cr_times)*1000:.1f} ms")
        print(f"  Original info median:    {_median(orig_times)*1000:.1f} ms")

    return {
        "benchmark": "metadata_latency",
        "cloud_ready_median_ms": round(_median(cr_times) * 1000, 1),
        "original_median_ms": round(_median(orig_times) * 1000, 1),
        "cloud_ready_file_size_mb": round(os.path.getsize(optimized) / 1e6, 1)
        if os.path.exists(optimized) else round(file_size / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results, dataset_info, sysinfo):
    """Generate cloud_benchmark.md markdown report."""
    lines = [
        "# SCX Cloud Benchmark Report",
        "",
        "## Test Environment",
        f"- **CPU**: {sysinfo['cpu']}",
        f"- **RAM**: {sysinfo['ram']}",
        f"- **OS**: {sysinfo['os']}",
        "- **SCX version**: 0.1.0 (Phase 2-CLOUD)",
        "- **Build**: `--release --features cloud`",
        f"- **Dataset**: {DATASET} ({dataset_info['n_obs']:,} cells "
        f"x {dataset_info['n_vars']:,} genes, "
        f"{dataset_info['file_size']/1e6:.1f} MB)",
        f"- **GCS bucket**: `{GCS_PREFIX}`",
        f"- **Runs**: {N_RUNS} per benchmark",
        "",
    ]

    by_name = {r["benchmark"]: r for r in all_results}

    # 1. Push
    r = by_name.get("push", {})
    lines.extend([
        "## 1. Push Throughput",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Push {r['file_size_mb']} MB (median) "
            f"| {r['median_s']:.2f}s | — | — |",
            f"| Throughput (median) "
            f"| {r['median_throughput_mbps']:.1f} MB/s | >50 MB/s "
            f"| {'PASS' if r['pass'] else 'FAIL'} |",
            f"| Min / Max | {r['min_s']:.2f}s / {r['max_s']:.2f}s | — | — |",
        ])
    else:
        lines.append("| *(not run)* | — | — | — |")
    lines.append("")

    # 2. Pull
    r = by_name.get("pull", {})
    lines.extend([
        "## 2. Pull Throughput",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Pull {r['file_size_mb']} MB (median) "
            f"| {r['median_s']:.2f}s | — | — |",
            f"| Throughput (median) "
            f"| {r['median_throughput_mbps']:.1f} MB/s | >50 MB/s "
            f"| {'PASS' if r['pass'] else 'FAIL'} |",
            f"| Min / Max | {r['min_s']:.2f}s / {r['max_s']:.2f}s | — | — |",
        ])
    else:
        lines.append("| *(not run)* | — | — | — |")
    lines.append("")

    # 3. Filtered pull
    r = by_name.get("filtered_pull", {})
    lines.extend([
        "## 3. Filtered Pull",
        "",
        f"Filter: `{FILTER_EXPR}`",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Latency (median) | {r['median_s']:.2f}s | — | — |",
            f"| Matching cells | {r['matching_cells']:,} | — | — |",
            f"| Shards downloaded | {r['downloaded_shards']}/{r['total_shards']} | — | — |",
            f"| Shard skip rate | {r['skip_rate']*100:.1f}% | >0% "
            f"| {'PASS' if r['pass'] else 'FAIL'} |",
            f"| Data downloaded | {r['mb_downloaded']:.1f} MB | — | — |",
            f"| Data saved | {r['mb_saved']:.1f} MB | — | — |",
            f"| Min / Max | {r['min_s']:.2f}s / {r['max_s']:.2f}s | — | — |",
        ])

        # Compare with full pull
        full_r = by_name.get("pull", {})
        if full_r:
            speedup = full_r["median_s"] / r["median_s"] if r["median_s"] > 0 else 0
            lines.append(
                f"| vs full pull | {speedup:.1f}x faster | >1x | "
                f"{'PASS' if speedup > 1 else 'FAIL'} |"
            )
    else:
        lines.append("| *(not run)* | — | — | — |")
    lines.append("")

    # 4. Cloud optimize
    r = by_name.get("cloud_optimize", {})
    lines.extend([
        "## 4. Cloud Optimize Overhead",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Time (median) | {r['median_s']:.2f}s | <5s "
            f"| {'PASS' if r['median_s'] < 5 else 'FAIL'} |",
            f"| Size overhead | {r['median_overhead_pct']:.2f}% | <5% "
            f"| {'PASS' if r['median_overhead_pct'] < 5 else 'FAIL'} |",
            f"| Input size | {r['input_size_mb']:.1f} MB | — | — |",
        ])
    else:
        lines.append("| *(not run)* | — | — | — |")
    lines.append("")

    # 5. Explode + pack
    r = by_name.get("explode_pack", {})
    lines.extend([
        "## 5. Explode + Pack Round-trip",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Explode (median) | {r['explode_median_s']:.2f}s | — | — |",
            f"| Pack (median) | {r['pack_median_s']:.2f}s | — | — |",
            f"| Total (median) | {r['total_median_s']:.2f}s | <30s "
            f"| {'PASS' if r['pass'] else 'FAIL'} |",
            f"| Min / Max total | {r['total_min_s']:.2f}s / {r['total_max_s']:.2f}s | — | — |",
        ])
    else:
        lines.append("| *(not run)* | — | — | — |")
    lines.append("")

    # 6. Streaming vs naive
    r = by_name.get("streaming_vs_naive", {})
    lines.extend([
        "## 6. Streaming Pull vs Naive (gsutil cp + pack)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Streaming pull (median) | {r['streaming_median_s']:.2f}s |",
            f"| Naive: gsutil + pack (median) | {r['naive_median_s']:.2f}s |",
            f"| Speedup | **{r['speedup']:.2f}x** |",
        ])
    else:
        lines.append("| *(not run)* | — |")
    lines.append("")

    # 7. Metadata latency
    r = by_name.get("metadata_latency", {})
    lines.extend([
        "## 7. Metadata Latency",
        "",
        "Local `scx info` on cloud-ready vs original file.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    if r:
        lines.extend([
            f"| Cloud-ready file (median) | {r['cloud_ready_median_ms']:.1f} ms |",
            f"| Original file (median) | {r['original_median_ms']:.1f} ms |",
        ])
    else:
        lines.append("| *(not run)* | — |")
    lines.append("")

    # Summary
    lines.extend([
        "## Summary",
        "",
        "| Benchmark | Result | Pass? |",
        "|-----------|--------|-------|",
    ])

    checks = [
        ("push", "Push throughput",
         lambda r: f"{r['median_throughput_mbps']:.1f} MB/s"),
        ("pull", "Pull throughput",
         lambda r: f"{r['median_throughput_mbps']:.1f} MB/s"),
        ("filtered_pull", "Filtered pull skip rate",
         lambda r: f"{r['skip_rate']*100:.1f}%"),
        ("cloud_optimize", "Cloud optimize overhead",
         lambda r: f"{r['median_overhead_pct']:.2f}%"),
        ("explode_pack", "Explode+pack round-trip",
         lambda r: f"{r['total_median_s']:.2f}s"),
    ]
    for key, label, fmt in checks:
        r = by_name.get(key, {})
        if r:
            passed = r.get("pass", True)
            lines.append(f"| {label} | {fmt(r)} | {'PASS' if passed else 'FAIL'} |")
        else:
            lines.append(f"| {label} | *(not run)* | — |")

    r_sv = by_name.get("streaming_vs_naive", {})
    if r_sv:
        lines.append(f"| Streaming vs naive | {r_sv['speedup']:.2f}x | — |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SCX Cloud Benchmark")
    print("=" * 60)

    # Verify GCS credentials
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds or not os.path.exists(creds):
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set or file not found.",
              file=sys.stderr)
        print("Set it to your GCS service account key, e.g.:", file=sys.stderr)
        print("  export GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/key.json",
              file=sys.stderr)
        sys.exit(1)
    print(f"GCS credentials: {creds}")

    binary = find_scx_binary()
    print(f"Binary: {binary}")

    scx_path = DATA_DIR / f"{DATASET}.scx"
    if not scx_path.exists():
        print(f"ERROR: {scx_path} not found.", file=sys.stderr)
        print(f"Set $SCX_DATA_DIR to the directory containing {DATASET}.scx",
              file=sys.stderr)
        sys.exit(1)

    file_size = scx_path.stat().st_size
    print(f"Dataset: {scx_path} ({file_size/1e6:.1f} MB)")

    # Get dataset info
    info_out, _, _ = run_scx(binary, ["info", str(scx_path), "--json"])
    dataset_info = json.loads(info_out)
    dataset_info["file_size"] = file_size
    print(f"  {dataset_info['n_obs']:,} obs x {dataset_info['n_vars']:,} vars,"
          f" {dataset_info['nnz']:,} nnz")

    sysinfo = get_system_info()
    print(f"System: {sysinfo['cpu']}, {sysinfo['ram']}")

    # Setup: ensure data on GCS
    print("\n--- Setup: uploading test data to GCS ---")
    gcs_source = setup_gcs_data(binary, scx_path)
    print(f"GCS source: {gcs_source}")

    # Run benchmarks
    all_results = []

    r_push = bench_push(binary, scx_path, file_size)
    all_results.append(r_push)

    r_pull = bench_pull(binary, gcs_source, dataset_info)
    all_results.append(r_pull)

    r_filter = bench_filtered_pull(binary, gcs_source, dataset_info)
    all_results.append(r_filter)

    r_opt = bench_cloud_optimize(binary, scx_path, file_size)
    all_results.append(r_opt)

    r_ep = bench_explode_pack(binary, scx_path, dataset_info, file_size)
    all_results.append(r_ep)

    r_sv = bench_streaming_vs_naive(binary, gcs_source, r_pull)
    all_results.append(r_sv)

    r_meta = bench_metadata_latency(binary, scx_path, file_size)
    all_results.append(r_meta)

    # Generate report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report(all_results, dataset_info, sysinfo)
    report_path = RESULTS_DIR / "cloud_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport: {report_path}")

    json_path = RESULTS_DIR / "cloud_benchmark.json"
    json_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"JSON:   {json_path}")

    print("\n" + "=" * 60)
    print("Cloud benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
