#!/usr/bin/env python3
"""
SCX GPU Preprocessing Benchmark

Benchmarks GPU fused normalize+log1p vs CPU preprocessing paths.
The GPU preprocessing kernel (normalize_log1p.cu) is used internally by the
GPU decode/loader pipeline. This benchmark measures:
  1. CPU SCX streaming preprocess: pyscx.preprocess(src, dst, ops)
  2. Scanpy baseline: sc.pp.normalize_total + sc.pp.log1p (in-memory)
  3. Validation: output equivalence between SCX CPU and scanpy (rtol=1e-6)

GPU kernel correctness is validated in Rust unit tests
(scx-gpu::gpu_preprocess::test_gpu_normalize_log1p_matches_cpu).

Usage:
    python benchmarks/scripts/benchmark_gpu_preprocess.py --mode validate
    python benchmarks/scripts/benchmark_gpu_preprocess.py --mode bench
    python benchmarks/scripts/benchmark_gpu_preprocess.py --mode all
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
from bench_env import WORK_DIR

DATASETS = {
    "pbmc3k": {"h5ad": WORK_DIR / "pbmc3k.h5ad", "cells": 2_700},
    "tabula_sapiens_100k": {"h5ad": WORK_DIR / "tabula_sapiens_100k.h5ad", "cells": 100_000},
    "census_1m": {"h5ad": WORK_DIR / "census_1m.h5ad", "cells": 1_000_000},
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_scx_file(dataset_name: str) -> Path | None:
    """Ensure an SCX file exists, converting from h5ad if needed."""
    import pyscx

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    scx_path = h5ad_path.with_suffix(".scx")

    if scx_path.exists():
        return scx_path
    if not h5ad_path.exists():
        return None

    print(f"  Converting {dataset_name} h5ad -> SCX...")
    import anndata
    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, str(scx_path))
    del adata
    gc.collect()
    return scx_path


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Validation — SCX preprocess vs scanpy
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocess_validation(dataset_name: str = "pbmc3k") -> dict:
    """Validate SCX CPU preprocessing output matches scanpy."""
    import anndata
    import pyscx
    import scanpy as sc
    import scipy.sparse as sp

    print(f"\n{'='*60}")
    print(f"Preprocessing Validation — {dataset_name}")
    print(f"{'='*60}")

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        return {"error": f"Dataset {dataset_name} not available"}

    scx_path = ensure_scx_file(dataset_name)
    if scx_path is None:
        return {"error": f"Cannot create SCX for {dataset_name}"}

    # Scanpy reference
    print(f"  Running scanpy normalize_total + log1p...")
    adata_scanpy = anndata.read_h5ad(str(h5ad_path))
    t0 = time.perf_counter()
    sc.pp.normalize_total(adata_scanpy, target_sum=1e4)
    sc.pp.log1p(adata_scanpy)
    t_scanpy = time.perf_counter() - t0
    print(f"    Done in {t_scanpy:.3f}s")

    # SCX streaming preprocess
    print(f"  Running pyscx.preprocess (streaming CPU)...")
    dst_path = str(WORK_DIR / f"{dataset_name}_preproc_bench.scx")
    t0 = time.perf_counter()
    pyscx.preprocess(str(scx_path), dst_path, ["normalize_total", "log1p"])
    t_scx = time.perf_counter() - t0
    print(f"    Done in {t_scx:.3f}s")

    # Compare
    print(f"  Comparing outputs...")
    adata_scx = pyscx.open(dst_path).to_anndata()
    X_scanpy = adata_scanpy.X
    X_scx = adata_scx.X
    if sp.issparse(X_scanpy):
        X_scanpy = X_scanpy.toarray()
    if sp.issparse(X_scx):
        X_scx = X_scx.toarray()

    max_abs_diff = float(np.max(np.abs(X_scanpy - X_scx)))
    # Compute relative error only where scanpy values are non-zero
    nonzero_mask = np.abs(X_scanpy) > 1e-10
    if nonzero_mask.any():
        rel_err = np.abs(X_scanpy[nonzero_mask] - X_scx[nonzero_mask]) / np.abs(X_scanpy[nonzero_mask])
        max_rel_err = float(np.max(rel_err))
        mean_rel_err = float(np.mean(rel_err))
    else:
        max_rel_err = 0.0
        mean_rel_err = 0.0

    allclose = bool(np.allclose(X_scanpy, X_scx, rtol=1e-6, atol=1e-7))

    print(f"    Max absolute diff: {max_abs_diff:.2e}")
    print(f"    Max relative error: {max_rel_err:.2e}")
    print(f"    Mean relative error: {mean_rel_err:.2e}")
    print(f"    np.allclose(rtol=1e-6): {allclose}")

    result = {
        "benchmark": "gpu_preprocess_validation",
        "dataset": dataset_name,
        "n_obs": adata_scanpy.n_obs,
        "n_vars": adata_scanpy.n_vars,
        "t_scanpy_s": round(t_scanpy, 3),
        "t_scx_cpu_s": round(t_scx, 3),
        "speedup_vs_scanpy": round(t_scanpy / t_scx, 1) if t_scx > 0 else 0,
        "max_abs_diff": float(f"{max_abs_diff:.2e}"),
        "max_rel_err": float(f"{max_rel_err:.2e}"),
        "mean_rel_err": float(f"{mean_rel_err:.2e}"),
        "allclose_rtol_1e6": allclose,
        "pass": allclose,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Clean up temp file
    try:
        os.remove(dst_path)
    except OSError:
        pass

    verdict = result["pass"]
    print(f"\n  Verdict: {'PASS' if verdict else 'FAIL'}")

    del adata_scanpy, adata_scx
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Timing benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_timing_benchmark(datasets: list[str] | None = None,
                         n_runs: int = 3) -> list[dict]:
    """Benchmark SCX preprocessing vs scanpy across dataset sizes."""
    import anndata
    import pyscx
    import scanpy as sc

    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]

    print(f"\n{'='*60}")
    print(f"Preprocessing Timing Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        info = DATASETS[ds_name]
        h5ad_path = info["h5ad"]
        if not h5ad_path.exists():
            print(f"  SKIP: {ds_name} not found")
            continue

        scx_path = ensure_scx_file(ds_name)
        if scx_path is None:
            continue

        n_obs = info["cells"]
        print(f"\n  --- {ds_name} ({n_obs:,} cells) ---")

        # Scanpy timing
        scanpy_times = []
        for run in range(n_runs):
            adata = anndata.read_h5ad(str(h5ad_path))
            t0 = time.perf_counter()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            scanpy_times.append(time.perf_counter() - t0)
            del adata
            gc.collect()
        scanpy_median = sorted(scanpy_times)[len(scanpy_times) // 2]
        print(f"    Scanpy: {scanpy_median:.3f}s")

        # SCX CPU streaming preprocess timing
        scx_times = []
        for run in range(n_runs):
            dst = str(WORK_DIR / f"{ds_name}_preproc_bench_{run}.scx")
            t0 = time.perf_counter()
            pyscx.preprocess(str(scx_path), dst, ["normalize_total", "log1p"])
            scx_times.append(time.perf_counter() - t0)
            try:
                os.remove(dst)
            except OSError:
                pass
        scx_median = sorted(scx_times)[len(scx_times) // 2]
        speedup = scanpy_median / scx_median if scx_median > 0 else 0
        print(f"    SCX CPU: {scx_median:.3f}s, speedup vs scanpy: {speedup:.1f}x")

        result = {
            "benchmark": "gpu_preprocess_timing",
            "dataset": ds_name,
            "n_obs": n_obs,
            "scanpy_times_s": [round(t, 3) for t in scanpy_times],
            "scanpy_median_s": round(scanpy_median, 3),
            "scx_cpu_times_s": [round(t, 3) for t in scx_times],
            "scx_cpu_median_s": round(scx_median, 3),
            "speedup_vs_scanpy": round(speedup, 1),
            "note": "GPU kernel validated in Rust unit tests (test_gpu_normalize_log1p_matches_cpu)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: Device-dispatch timing (Phase 7.3)
#
# Times `pyscx.accel.normalize_total`, `log1p`, fused `normalize+log1p`, and
# `highly_variable_genes` with `device="cpu"` vs `device="gpu"` on backed SCX
# data. Each run reopens the SCX file so the GPU eager-materialization path
# starts from a fresh backed source.
# ──────────────────────────────────────────────────────────────────────────────

def _reopen_backed(scx_path: Path):
    """Return a fresh `ScxBackedSparseDataset`-wrapped AnnData."""
    import pyscx
    return pyscx.open(str(scx_path)).to_anndata()


def _time_accel_op(op: str, scx_path: Path, device: str, target_sum: float = 1e4) -> float:
    """Time one pyscx.accel op on a fresh backed AnnData, return wall-seconds.

    `op` ∈ {"normalize_total", "log1p", "fused", "highly_variable_genes"}.
    `"fused"` calls `normalize_total` then `log1p` sequentially (which, on GPU,
    triggers the fusion-marker path — see Phase 5.4).
    """
    import pyscx

    adata = _reopen_backed(scx_path)
    t0 = time.perf_counter()
    if op == "normalize_total":
        pyscx.accel.normalize_total(adata, target_sum=target_sum, device=device)
    elif op == "log1p":
        pyscx.accel.log1p(adata, device=device)
    elif op == "fused":
        pyscx.accel.normalize_total(adata, target_sum=target_sum, device=device)
        pyscx.accel.log1p(adata, device=device)
    elif op == "highly_variable_genes":
        # Requires log-normalized input — inline the prereq so this op isolates
        # the HVG kernel cost.
        pyscx.accel.normalize_total(adata, target_sum=target_sum, device=device)
        pyscx.accel.log1p(adata, device=device)
        t0 = time.perf_counter()  # reset: measure only the HVG call
        n_top = min(2000, adata.n_vars)
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3", device=device,
        )
    else:
        raise ValueError(f"unknown op: {op}")
    elapsed = time.perf_counter() - t0
    del adata
    gc.collect()
    return elapsed


def run_device_benchmark(datasets: list[str] | None = None,
                        n_runs: int = 3) -> list[dict]:
    """Benchmark CPU vs GPU device-dispatch for each preprocessing op."""
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]

    print(f"\n{'='*60}")
    print(f"Device Dispatch Benchmark (CPU vs GPU, median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []
    for ds_name in datasets:
        info = DATASETS.get(ds_name)
        if info is None:
            continue
        scx_path = ensure_scx_file(ds_name)
        if scx_path is None:
            print(f"  SKIP: {ds_name} not found")
            continue

        print(f"\n  --- {ds_name} ({info['cells']:,} cells) ---")
        per_op: dict = {}
        for op in ["normalize_total", "log1p", "fused", "highly_variable_genes"]:
            per_op[op] = {}
            for device in ["cpu", "gpu"]:
                times = []
                for _ in range(n_runs):
                    try:
                        times.append(_time_accel_op(op, scx_path, device))
                    except Exception as e:
                        per_op[op][f"{device}_error"] = str(e)
                        break
                if times:
                    med = sorted(times)[len(times) // 2]
                    per_op[op][device] = {
                        "times_s": [round(t, 3) for t in times],
                        "median_s": round(med, 3),
                    }
            cpu = per_op[op].get("cpu", {}).get("median_s")
            gpu = per_op[op].get("gpu", {}).get("median_s")
            if cpu and gpu and gpu > 0:
                per_op[op]["speedup"] = round(cpu / gpu, 1)
                print(f"    {op}: CPU={cpu:.3f}s  GPU={gpu:.3f}s  speedup={cpu/gpu:.1f}x")
            elif cpu:
                print(f"    {op}: CPU={cpu:.3f}s  GPU=n/a")

        results.append({
            "benchmark": "gpu_preprocess_device_dispatch",
            "dataset": ds_name,
            "n_obs": info["cells"],
            "ops": per_op,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(validation_results: list[dict] | None,
                    timing_results: list[dict] | None,
                    device_results: list[dict] | None = None) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU Preprocessing Benchmark Report",
        "",
        f"**Generated**: {ts}",
        "",
        "> GPU fused normalize+log1p kernel correctness is validated in Rust unit tests",
        "> (`scx-gpu::gpu_preprocess::test_gpu_normalize_log1p_matches_cpu`).",
        "> This benchmark compares SCX streaming CPU preprocessing vs scanpy.",
        "",
    ]

    if validation_results:
        lines += [
            "## 1. Correctness Validation (SCX CPU vs Scanpy)",
            "",
            "| Dataset | Cells | Max Abs Diff | Max Rel Err | allclose(rtol=1e-6) | Pass? |",
            "|---------|-------|-------------|-------------|---------------------|-------|",
        ]
        for vr in validation_results:
            if "error" in vr:
                lines.append(f"| {vr.get('dataset', '?')} | — | — | — | — | SKIP |")
                continue
            verdict = "PASS" if vr["pass"] else "FAIL"
            lines.append(
                f"| {vr['dataset']} | {vr['n_obs']:,} | "
                f"{vr['max_abs_diff']:.2e} | {vr['max_rel_err']:.2e} | "
                f"{vr['allclose_rtol_1e6']} | {verdict} |"
            )
        lines.append("")

    if timing_results:
        lines += [
            "## 2. Preprocessing Timing",
            "",
            "| Dataset | Cells | Scanpy (s) | SCX CPU (s) | Speedup |",
            "|---------|-------|-----------|------------|---------|",
        ]
        for r in timing_results:
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | "
                f"{r['scanpy_median_s']:.3f} | {r['scx_cpu_median_s']:.3f} | "
                f"{r['speedup_vs_scanpy']:.1f}x |"
            )
        lines.append("")

    if device_results:
        lines += [
            "## 3. CPU vs GPU Device Dispatch",
            "",
            "| Dataset | Op | CPU (s) | GPU (s) | Speedup |",
            "|---------|----|---------|---------|---------|",
        ]
        for r in device_results:
            ds = r["dataset"]
            for op, rec in r["ops"].items():
                cpu = rec.get("cpu", {}).get("median_s", "—")
                gpu = rec.get("gpu", {}).get("median_s", "—")
                sp = rec.get("speedup", "—")
                cpu_str = f"{cpu:.3f}" if isinstance(cpu, (int, float)) else str(cpu)
                gpu_str = f"{gpu:.3f}" if isinstance(gpu, (int, float)) else str(gpu)
                sp_str = f"{sp:.1f}x" if isinstance(sp, (int, float)) else str(sp)
                lines.append(f"| {ds} | {op} | {cpu_str} | {gpu_str} | {sp_str} |")
        lines.append("")

    lines += ["## Summary", ""]
    all_pass = True
    if validation_results:
        for vr in validation_results:
            if "error" not in vr:
                vp = vr.get("pass", False)
                lines.append(f"- {vr['dataset']}: **{'PASS' if vp else 'FAIL'}**")
                all_pass = all_pass and vp
    lines += ["", f"**Overall: {'PASS' if all_pass else 'FAIL'}**", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX GPU Preprocessing Benchmark"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["validate", "bench", "device", "all"],
        help="validate: output equivalence; bench: scanpy vs SCX CPU timing; "
             "device: CPU vs GPU device-dispatch per op (Phase 7.3); "
             "all: run validate + bench + device.",
    )
    parser.add_argument(
        "--dataset", default=None, choices=list(DATASETS.keys()),
    )
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.mode in ("validate", "all"):
        ds = args.dataset or "pbmc3k"
        validation_results = [run_preprocess_validation(ds)]
        all_results["validation"] = validation_results

        json_path = RESULTS_DIR / "gpu_preprocess_validation.json"
        json_path.write_text(json.dumps(validation_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    if args.mode in ("bench", "all"):
        bench_datasets = (
            [args.dataset] if args.dataset
            else ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
        )
        timing_results = run_timing_benchmark(bench_datasets, args.n_runs)
        all_results["timing"] = timing_results

        json_path = RESULTS_DIR / "gpu_preprocess_timing.json"
        json_path.write_text(json.dumps(timing_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    if args.mode in ("device", "all"):
        device_datasets = (
            [args.dataset] if args.dataset
            else ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
        )
        device_results = run_device_benchmark(device_datasets, args.n_runs)
        all_results["device"] = device_results

        json_path = RESULTS_DIR / "gpu_preprocess_device.json"
        json_path.write_text(json.dumps(device_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    report = generate_report(
        all_results.get("validation"),
        all_results.get("timing"),
        all_results.get("device"),
    )
    md_path = RESULTS_DIR / "gpu_preprocess_benchmark.md"
    md_path.write_text(report)
    print(f"\n  Report saved: {md_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        ensure_release_build()
    main()
