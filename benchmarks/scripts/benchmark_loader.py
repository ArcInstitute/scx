#!/usr/bin/env python3
"""
SCX Training Loader Benchmark (Phase 2 Step 5, H2)

Measures SCX loader throughput, latency, memory, and compares against
SOTA baselines (TileDB-SOMA-ML, AnnData in-memory, scDataLoader).

Can be run standalone or submitted via submit_benchmarks.py to SLURM.

Usage:
    python benchmarks/scripts/benchmark_loader.py --dataset pbmc3k --smoke
    python benchmarks/scripts/benchmark_loader.py --dataset tabula_sapiens_100k
    python benchmarks/scripts/benchmark_loader.py --dataset census_10m_blood --all
"""

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = Path(os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx"))
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

sys.path.insert(0, str(REPO_ROOT / "pyscx"))

DATASETS = {
    "pbmc3k": {
        "h5ad": DATA_DIR / "pbmc3k.h5ad",
        "scx": DATA_DIR / "pbmc3k.scx",
        "cells": 2700,
        "genes": 32738,
        "desc": "PBMC 3K — smoke test",
    },
    "smartseq2": {
        "h5ad": DATA_DIR / "smartseq2.h5ad",
        "scx": DATA_DIR / "smartseq2.scx",
        "cells": 70000,
        "genes": 60000,
        "desc": "Smart-seq2 — full-gene panel",
    },
    "tabula_sapiens_100k": {
        "h5ad": DATA_DIR / "tabula_sapiens_100k.h5ad",
        "scx": DATA_DIR / "tabula_sapiens_100k.scx",
        "cells": 100000,
        "genes": 60000,
        "desc": "Tabula Sapiens 100K — primary throughput",
    },
    "census_1m": {
        "h5ad": DATA_DIR / "census_1m.h5ad",
        "scx": DATA_DIR / "census_1m.scx",
        "cells": 1000000,
        "genes": 61497,
        "desc": "CELLxGENE Census 1M blood — scale validation",
    },
    "census_10m_blood": {
        "h5ad": DATA_DIR / "census_10m_blood.h5ad",
        "scx": DATA_DIR / "census_10m_blood.scx",
        "cells": 10000000,
        "genes": 60000,
        "desc": "CELLxGENE Census 10M — scale & Go/No-Go",
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def get_system_info() -> dict:
    """Collect system hardware/software info."""
    info = {"hostname": os.uname().nodename}

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        info["cpu"] = "unknown"

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / (1024 * 1024), 1)
                    break
    except Exception:
        info["ram_gb"] = 0

    # GPU
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
        info["gpus"] = [line.strip() for line in out.splitlines()]
    except Exception:
        info["gpus"] = []

    # Python packages
    try:
        import importlib.util
        info["pyscx"] = "available" if importlib.util.find_spec("pyscx") else "missing"
    except Exception:
        info["pyscx"] = "missing"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch"] = "missing"
    try:
        import tiledbsoma
        info["tiledbsoma"] = tiledbsoma.__version__
    except ImportError:
        info["tiledbsoma"] = "missing"

    return info


@contextmanager
def peak_rss_tracker():
    """Context manager that tracks peak RSS (in MB) during execution."""
    gc.collect()
    result = {"peak_rss_mb": 0.0}
    yield result
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mb"] = round(peak_kb / 1024, 1)


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self.start = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def percentile(values, pct):
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * pct / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def ensure_scx(dataset_name: str) -> Path:
    """Ensure .scx version of dataset exists, converting from h5ad if needed."""
    ds = DATASETS[dataset_name]
    scx_path = ds["scx"]
    h5ad_path = ds["h5ad"]

    if scx_path.exists():
        return scx_path

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Dataset h5ad not found: {h5ad_path}")

    print(f"  Converting {h5ad_path.name} → {scx_path.name}...")
    import pyscx
    import anndata

    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, str(scx_path))
    print(f"  Done: {scx_path} ({scx_path.stat().st_size / 1e6:.1f} MB)")
    return scx_path


# ---------------------------------------------------------------------------
# SCX Benchmarks
# ---------------------------------------------------------------------------
def bench_throughput(dataset_name: str, batch_size: int = 1024,
                     n_hvg: Optional[int] = 2000, normalize: bool = True,
                     n_warmup_epochs: int = 0, n_epochs: int = 1) -> dict:
    """Measure sustained batch throughput (batches/sec)."""
    import pyscx

    scx_path = ensure_scx(dataset_name)

    # Build HVG indices (first n_hvg genes)
    hvg_indices = list(range(n_hvg)) if n_hvg else None

    ds = pyscx.TrainingDataset(
        str(scx_path),
        batch_size=batch_size,
        hvg_indices=hvg_indices,
        normalize=normalize,
        log1p=normalize,
        seed=42,
    )

    # Warmup
    for _ in range(n_warmup_epochs):
        for _ in ds:
            pass

    # Timed run
    total_batches = 0
    total_cells = 0
    epoch_times = []

    for epoch in range(n_epochs):
        with Timer() as t:
            for batch in ds:
                total_batches += 1
                total_cells += batch["X"].shape[0]
        epoch_times.append(t.elapsed)

    total_time = sum(epoch_times)
    return {
        "benchmark": "throughput",
        "dataset": dataset_name,
        "batch_size": batch_size,
        "n_hvg": n_hvg,
        "normalize": normalize,
        "n_epochs": n_epochs,
        "total_batches": total_batches,
        "total_cells": total_cells,
        "total_time_s": round(total_time, 3),
        "batches_per_sec": round(total_batches / total_time, 1) if total_time > 0 else 0,
        "cells_per_sec": round(total_cells / total_time, 0) if total_time > 0 else 0,
        "epoch_times_s": [round(t, 3) for t in epoch_times],
    }


def bench_time_to_first_batch(dataset_name: str, n_runs: int = 5) -> dict:
    """Measure latency from constructor → first batch."""
    import pyscx

    scx_path = ensure_scx(dataset_name)
    latencies = []

    for _ in range(n_runs):
        gc.collect()
        with Timer() as t:
            ds = pyscx.TrainingDataset(
                str(scx_path), batch_size=1024, normalize=False, log1p=False, seed=42,
            )
            for batch in ds:
                break  # Just get first batch
        latencies.append(t.elapsed)
        del ds

    latencies.sort()
    return {
        "benchmark": "time_to_first_batch",
        "dataset": dataset_name,
        "n_runs": n_runs,
        "median_s": round(percentile(latencies, 50), 4),
        "min_s": round(min(latencies), 4),
        "max_s": round(max(latencies), 4),
        "p95_s": round(percentile(latencies, 95), 4),
        "all_s": [round(x, 4) for x in latencies],
        "target_s": 2.0,
        "pass": percentile(latencies, 50) < 2.0,
    }


def bench_memory(dataset_name: str, max_memory_mb: int = 512) -> dict:
    """Monitor peak RSS during full-epoch iteration."""
    import pyscx

    scx_path = ensure_scx(dataset_name)

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    with peak_rss_tracker() as rss:
        ds = pyscx.TrainingDataset(
            str(scx_path), batch_size=1024, max_memory_mb=max_memory_mb,
            normalize=True, log1p=True, seed=42,
        )
        n_batches = 0
        for batch in ds:
            n_batches += 1
        del ds

    return {
        "benchmark": "memory",
        "dataset": dataset_name,
        "max_memory_mb": max_memory_mb,
        "peak_rss_mb": rss["peak_rss_mb"],
        "rss_before_mb": round(rss_before, 1),
        "n_batches": n_batches,
        "within_budget": rss["peak_rss_mb"] <= max_memory_mb * 1.5,  # 50% tolerance for Python overhead
    }


def bench_hvg_impact(dataset_name: str) -> dict:
    """Compare throughput with and without HVG projection."""
    no_hvg = bench_throughput(dataset_name, n_hvg=None, normalize=False)
    with_hvg = bench_throughput(dataset_name, n_hvg=2000, normalize=False)

    speedup = (with_hvg["batches_per_sec"] / no_hvg["batches_per_sec"]
               if no_hvg["batches_per_sec"] > 0 else 0)

    return {
        "benchmark": "hvg_impact",
        "dataset": dataset_name,
        "all_genes_bps": no_hvg["batches_per_sec"],
        "hvg_2k_bps": with_hvg["batches_per_sec"],
        "speedup": round(speedup, 2),
        "target_speedup": 5.0,
        "pass": speedup >= 5.0,
    }


# ---------------------------------------------------------------------------
# Baseline benchmarks
# ---------------------------------------------------------------------------
def bench_baseline_anndata(dataset_name: str, batch_size: int = 1024,
                           n_hvg: int = 2000) -> dict:
    """AnnData in-memory + manual batching baseline."""
    try:
        import anndata
        import numpy as np
    except ImportError:
        return {"benchmark": "baseline_anndata", "error": "anndata not installed"}

    ds = DATASETS[dataset_name]
    h5ad_path = ds["h5ad"]
    if not h5ad_path.exists():
        return {"benchmark": "baseline_anndata", "error": f"h5ad not found: {h5ad_path}"}

    # Time includes load (part of total pipeline cost)
    gc.collect()
    with Timer() as t_load:
        adata = anndata.read_h5ad(str(h5ad_path))

    with Timer() as t_iter:
        n_batches = 0
        n_cells = adata.n_obs
        indices = np.random.permutation(n_cells)
        for start in range(0, n_cells, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = adata.X[batch_idx].toarray() if hasattr(adata.X, "toarray") else adata.X[batch_idx]
            # Simulate HVG subset
            if n_hvg and X_batch.shape[1] > n_hvg:
                X_batch = X_batch[:, :n_hvg]
            # normalize + log1p
            row_sums = X_batch.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_batch = X_batch / row_sums * 1e4
            X_batch = np.log1p(X_batch).astype(np.float32)
            n_batches += 1

    total_time = t_load.elapsed + t_iter.elapsed
    return {
        "benchmark": "baseline_anndata",
        "dataset": dataset_name,
        "load_time_s": round(t_load.elapsed, 3),
        "iter_time_s": round(t_iter.elapsed, 3),
        "total_time_s": round(total_time, 3),
        "n_batches": n_batches,
        "batches_per_sec": round(n_batches / total_time, 1) if total_time > 0 else 0,
    }


def bench_baseline_soma(dataset_name: str, batch_size: int = 1024) -> dict:
    """TileDB-SOMA-ML baseline."""
    try:
        import tiledbsoma
        from tiledbsoma import io as soma_io
        from tiledbsoma_ml import ExperimentDataset
        import torch
    except ImportError as e:
        return {"benchmark": "baseline_soma", "error": f"import failed: {e}"}

    # We need the h5ad converted to SOMA format first
    ds = DATASETS[dataset_name]
    soma_uri = str(DATA_DIR / f"{dataset_name}.soma")

    if not Path(soma_uri).exists():
        # Convert h5ad → SOMA experiment
        print(f"  Converting {dataset_name} to SOMA format...")
        try:
            import anndata
            adata = anndata.read_h5ad(str(ds["h5ad"]))
            soma_io.from_anndata(
                soma_uri, adata,
                measurement_name="RNA",
            )
            del adata
            gc.collect()
            print(f"  SOMA experiment created at {soma_uri}")
        except Exception as e:
            return {"benchmark": "baseline_soma", "error": f"SOMA conversion failed: {e}"}

    query = None
    exp = None
    try:
        gc.collect()
        with Timer() as t:
            exp = tiledbsoma.Experiment.open(soma_uri)
            query = exp.axis_query("RNA")
            dataset = ExperimentDataset(
                query=query,
                layer_name="data",
                batch_size=batch_size,
                shuffle=True,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=0,
            )
            n_batches = 0
            for batch in loader:
                n_batches += 1

        return {
            "benchmark": "baseline_soma",
            "dataset": dataset_name,
            "total_time_s": round(t.elapsed, 3),
            "n_batches": n_batches,
            "batches_per_sec": round(n_batches / t.elapsed, 1) if t.elapsed > 0 else 0,
        }
    except Exception as e:
        return {"benchmark": "baseline_soma", "error": str(e)}
    finally:
        if query is not None:
            query.close()
        if exp is not None:
            exp.close()


def bench_baseline_scdataloader(dataset_name: str, batch_size: int = 1024) -> dict:
    """scDataLoader baseline."""
    try:
        from scdataloader import SimpleAnnDataset
        from torch.utils.data import DataLoader
    except ImportError:
        return {"benchmark": "baseline_scdataloader", "error": "scdataloader or torch not installed"}

    ds = DATASETS[dataset_name]
    h5ad_path = ds["h5ad"]
    if not h5ad_path.exists():
        return {"benchmark": "baseline_scdataloader", "error": f"h5ad not found: {h5ad_path}"}

    try:
        import anndata

        gc.collect()
        with Timer() as t_load:
            adata = anndata.read_h5ad(str(h5ad_path))

        with Timer() as t_iter:
            dataset = SimpleAnnDataset(adata)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            n_batches = 0
            for batch in loader:
                n_batches += 1

        total_time = t_load.elapsed + t_iter.elapsed
        return {
            "benchmark": "baseline_scdataloader",
            "dataset": dataset_name,
            "load_time_s": round(t_load.elapsed, 3),
            "iter_time_s": round(t_iter.elapsed, 3),
            "total_time_s": round(total_time, 3),
            "n_batches": n_batches,
            "batches_per_sec": round(n_batches / total_time, 1) if total_time > 0 else 0,
        }
    except Exception as e:
        return {"benchmark": "baseline_scdataloader", "error": str(e)}


def bench_baseline_bpcells(dataset_name: str, batch_size: int = 1024) -> dict:
    """BPCells baseline via R subprocess."""
    r_script = REPO_ROOT / "benchmarks" / "scripts" / "benchmark_bpcells.R"
    if not r_script.exists():
        return {"benchmark": "baseline_bpcells", "error": "R script not found"}

    ds = DATASETS[dataset_name]
    h5ad_path = ds["h5ad"]

    try:
        result = subprocess.run(
            ["Rscript", str(r_script), str(h5ad_path), str(batch_size)],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            return {"benchmark": "baseline_bpcells", "error": result.stderr[:500]}
        data = json.loads(result.stdout)
        data["benchmark"] = "baseline_bpcells"
        data["dataset"] = dataset_name
        return data
    except FileNotFoundError:
        return {"benchmark": "baseline_bpcells", "error": "Rscript not found"}
    except Exception as e:
        return {"benchmark": "baseline_bpcells", "error": str(e)}


# ---------------------------------------------------------------------------
# GPU benchmarks
# ---------------------------------------------------------------------------
def bench_gpu_utilization(dataset_name: str) -> dict:
    """Measure GPU utilization during a training-like loop."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"benchmark": "gpu_utilization", "error": "CUDA not available"}
    except ImportError:
        return {"benchmark": "gpu_utilization", "error": "torch not installed"}

    import pyscx

    scx_path = ensure_scx(dataset_name)

    # Start nvidia-smi monitoring in background
    smi_log = Path("/tmp/nvidia_smi_log.csv")
    smi_proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-f", str(smi_log)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        ds = pyscx.TrainingDataset(
            str(scx_path),
            batch_size=1024,
            hvg_indices=list(range(2000)),
            normalize=True, log1p=True, seed=42,
        )

        device = torch.device("cuda:0")

        # Simple forward pass simulation (matrix multiply as proxy for model work)
        # This tests whether the loader can keep the GPU busy
        dummy_weight = torch.randn(2000, 256, device=device)

        n_batches = 0
        with Timer() as t:
            for batch in ds:
                X = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
                # Simulate a forward pass
                _ = X @ dummy_weight
                torch.cuda.synchronize()
                n_batches += 1

        del dummy_weight
        torch.cuda.empty_cache()
    finally:
        smi_proc.terminate()
        smi_proc.wait()

    # Parse nvidia-smi log for GPU utilization
    gpu_utils = []
    if smi_log.exists():
        for line in smi_log.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    gpu_utils.append(int(parts[1]))
                except ValueError:
                    pass
        smi_log.unlink(missing_ok=True)

    avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    return {
        "benchmark": "gpu_utilization",
        "dataset": dataset_name,
        "n_batches": n_batches,
        "total_time_s": round(t.elapsed, 3),
        "batches_per_sec": round(n_batches / t.elapsed, 1) if t.elapsed > 0 else 0,
        "avg_gpu_util_pct": round(avg_util, 1),
        "gpu_util_samples": len(gpu_utils),
        "target_util_pct": 85,
        "pass": avg_util >= 85,
    }


def bench_multi_gpu_scaling(dataset_name: str, n_gpus: int = 1) -> dict:
    """Measure throughput with N GPUs (using round-robin batch dispatch)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"benchmark": "multi_gpu_scaling", "error": "CUDA not available"}
        available_gpus = torch.cuda.device_count()
        if available_gpus < n_gpus:
            return {"benchmark": "multi_gpu_scaling",
                    "error": f"Requested {n_gpus} GPUs but only {available_gpus} available"}
    except ImportError:
        return {"benchmark": "multi_gpu_scaling", "error": "torch not installed"}

    import pyscx

    scx_path = ensure_scx(dataset_name)
    ds = pyscx.TrainingDataset(
        str(scx_path), batch_size=1024,
        hvg_indices=list(range(2000)),
        normalize=True, log1p=True, seed=42,
    )

    devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
    weights = [torch.randn(2000, 256, device=d) for d in devices]

    n_batches = 0
    with Timer() as t:
        for batch in ds:
            gpu_idx = n_batches % n_gpus
            X = torch.from_numpy(batch["X"]).to(devices[gpu_idx], non_blocking=True)
            _ = X @ weights[gpu_idx]
            n_batches += 1
        # Final sync
        for d in devices:
            torch.cuda.synchronize(d)

    for w in weights:
        del w
    torch.cuda.empty_cache()

    return {
        "benchmark": "multi_gpu_scaling",
        "dataset": dataset_name,
        "n_gpus": n_gpus,
        "n_batches": n_batches,
        "total_time_s": round(t.elapsed, 3),
        "batches_per_sec": round(n_batches / t.elapsed, 1) if t.elapsed > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_all_benchmarks(dataset_name: str, n_gpus: int = 0,
                       smoke: bool = False) -> dict:
    """Run all benchmarks for a single dataset configuration."""
    results = {
        "dataset": dataset_name,
        "sysinfo": get_system_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset_name}")
    print(f"{'='*60}")

    # 1. SCX throughput
    print("\n--- SCX Throughput (batch_size=1024, n_hvg=2000) ---")
    try:
        r = bench_throughput(dataset_name, n_epochs=1 if smoke else 3)
        results["benchmarks"]["throughput"] = r
        print(f"  {r['batches_per_sec']} batches/sec ({r['total_batches']} batches)")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["benchmarks"]["throughput"] = {"error": str(e)}

    # 2. Time to first batch
    print("\n--- Time to First Batch ---")
    try:
        r = bench_time_to_first_batch(dataset_name, n_runs=3 if smoke else 5)
        results["benchmarks"]["time_to_first_batch"] = r
        print(f"  Median: {r['median_s']:.4f}s (target <2s: {'PASS' if r['pass'] else 'FAIL'})")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["benchmarks"]["time_to_first_batch"] = {"error": str(e)}

    # 3. Memory
    print("\n--- Memory Budget ---")
    try:
        r = bench_memory(dataset_name)
        results["benchmarks"]["memory"] = r
        print(f"  Peak RSS: {r['peak_rss_mb']} MB (budget {r['max_memory_mb']} MB)")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["benchmarks"]["memory"] = {"error": str(e)}

    # 4. HVG impact
    if not smoke:
        print("\n--- HVG Projection Impact ---")
        try:
            r = bench_hvg_impact(dataset_name)
            results["benchmarks"]["hvg_impact"] = r
            print(f"  Speedup: {r['speedup']:.2f}x (target ≥5x: {'PASS' if r['pass'] else 'FAIL'})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["benchmarks"]["hvg_impact"] = {"error": str(e)}

    # 5. Baselines
    print("\n--- Baseline: AnnData in-memory ---")
    try:
        r = bench_baseline_anndata(dataset_name)
        results["benchmarks"]["baseline_anndata"] = r
        if "error" not in r:
            print(f"  {r['batches_per_sec']} batches/sec (load: {r['load_time_s']:.1f}s)")
        else:
            print(f"  SKIPPED: {r['error']}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["benchmarks"]["baseline_anndata"] = {"error": str(e)}

    if not smoke:
        print("\n--- Baseline: TileDB-SOMA-ML ---")
        try:
            r = bench_baseline_soma(dataset_name)
            results["benchmarks"]["baseline_soma"] = r
            if "error" not in r:
                print(f"  {r['batches_per_sec']} batches/sec")
            else:
                print(f"  SKIPPED: {r['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["benchmarks"]["baseline_soma"] = {"error": str(e)}

        print("\n--- Baseline: scDataLoader ---")
        try:
            r = bench_baseline_scdataloader(dataset_name)
            results["benchmarks"]["baseline_scdataloader"] = r
            if "error" not in r:
                print(f"  {r['batches_per_sec']} batches/sec")
            else:
                print(f"  SKIPPED: {r['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["benchmarks"]["baseline_scdataloader"] = {"error": str(e)}

        print("\n--- Baseline: BPCells ---")
        try:
            r = bench_baseline_bpcells(dataset_name)
            results["benchmarks"]["baseline_bpcells"] = r
            if "error" not in r:
                print(f"  {r['batches_per_sec']} batches/sec")
            else:
                print(f"  SKIPPED: {r['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["benchmarks"]["baseline_bpcells"] = {"error": str(e)}

    # 6. GPU benchmarks (only if GPUs available)
    if n_gpus > 0:
        print(f"\n--- GPU Utilization ({n_gpus} GPU(s)) ---")
        try:
            r = bench_gpu_utilization(dataset_name)
            results["benchmarks"]["gpu_utilization"] = r
            if "error" not in r:
                print(f"  Avg GPU util: {r['avg_gpu_util_pct']:.1f}% "
                      f"(target ≥85%: {'PASS' if r['pass'] else 'FAIL'})")
            else:
                print(f"  SKIPPED: {r['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["benchmarks"]["gpu_utilization"] = {"error": str(e)}

        if n_gpus > 1:
            for ng in [1, 2, min(4, n_gpus)]:
                print(f"\n--- Multi-GPU Scaling ({ng} GPUs) ---")
                try:
                    r = bench_multi_gpu_scaling(dataset_name, n_gpus=ng)
                    results["benchmarks"][f"multi_gpu_{ng}"] = r
                    if "error" not in r:
                        print(f"  {r['batches_per_sec']} batches/sec with {ng} GPU(s)")
                    else:
                        print(f"  SKIPPED: {r['error']}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results["benchmarks"][f"multi_gpu_{ng}"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: list[dict]) -> str:
    """Generate markdown benchmark report from collected results."""
    lines = [
        "# SCX Training Loader Benchmark Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # System info (from first result)
    if all_results:
        sysinfo = all_results[0].get("sysinfo", {})
        lines.extend([
            "## Test Environment",
            "",
            f"- **Host**: {sysinfo.get('hostname', 'unknown')}",
            f"- **CPU**: {sysinfo.get('cpu', 'unknown')}",
            f"- **RAM**: {sysinfo.get('ram_gb', '?')} GB",
            f"- **GPUs**: {', '.join(sysinfo.get('gpus', ['none']))}",
            f"- **PyTorch**: {sysinfo.get('torch', 'N/A')}",
            f"- **TileDB-SOMA**: {sysinfo.get('tiledbsoma', 'N/A')}",
            "",
        ])

    # Dataset summary
    lines.extend([
        "## Datasets",
        "",
        "| Dataset | Cells | Genes | h5ad Size | SCX Size |",
        "|---------|-------|-------|-----------|----------|",
    ])
    for r in all_results:
        ds_name = r["dataset"]
        ds = DATASETS.get(ds_name, {})
        h5ad_size = "—"
        scx_size = "—"
        h5ad_path = ds.get("h5ad")
        scx_path = ds.get("scx")
        if h5ad_path and h5ad_path.exists():
            h5ad_size = f"{h5ad_path.stat().st_size / 1e6:.0f} MB"
        if scx_path and scx_path.exists():
            scx_size = f"{scx_path.stat().st_size / 1e6:.0f} MB"
        lines.append(
            f"| {ds_name} | {ds.get('cells', '?'):,} | {ds.get('genes', '?'):,} "
            f"| {h5ad_size} | {scx_size} |"
        )
    lines.append("")

    # Throughput
    lines.extend([
        "## 1. SCX Throughput",
        "",
        "| Dataset | Batches/sec | Cells/sec | Total Time | Batches |",
        "|---------|-------------|-----------|------------|---------|",
    ])
    for r in all_results:
        tp = r["benchmarks"].get("throughput", {})
        if "error" in tp:
            lines.append(f"| {r['dataset']} | ERROR | — | — | — |")
        else:
            lines.append(
                f"| {r['dataset']} | {tp.get('batches_per_sec', 0):,.1f} "
                f"| {tp.get('cells_per_sec', 0):,.0f} "
                f"| {tp.get('total_time_s', 0):.1f}s "
                f"| {tp.get('total_batches', 0):,} |"
            )
    lines.append("")

    # Time to first batch
    lines.extend([
        "## 2. Time to First Batch",
        "",
        "| Dataset | Median | Min | Max | Target | Pass? |",
        "|---------|--------|-----|-----|--------|-------|",
    ])
    for r in all_results:
        ttfb = r["benchmarks"].get("time_to_first_batch", {})
        if "error" in ttfb:
            lines.append(f"| {r['dataset']} | ERROR | — | — | — | — |")
        else:
            lines.append(
                f"| {r['dataset']} | {ttfb.get('median_s', 0):.3f}s "
                f"| {ttfb.get('min_s', 0):.3f}s "
                f"| {ttfb.get('max_s', 0):.3f}s "
                f"| <{ttfb.get('target_s', 2)}s "
                f"| {'PASS ✅' if ttfb.get('pass') else 'FAIL ❌'} |"
            )
    lines.append("")

    # Memory
    lines.extend([
        "## 3. Memory Budget",
        "",
        "| Dataset | Peak RSS | Budget | Within Budget? |",
        "|---------|----------|--------|----------------|",
    ])
    for r in all_results:
        mem = r["benchmarks"].get("memory", {})
        if "error" in mem:
            lines.append(f"| {r['dataset']} | ERROR | — | — |")
        else:
            lines.append(
                f"| {r['dataset']} | {mem.get('peak_rss_mb', 0):.0f} MB "
                f"| {mem.get('max_memory_mb', 0)} MB "
                f"| {'Yes ✅' if mem.get('within_budget') else 'No ❌'} |"
            )
    lines.append("")

    # HVG impact
    hvg_results = [r for r in all_results if "hvg_impact" in r["benchmarks"]
                   and "error" not in r["benchmarks"]["hvg_impact"]]
    if hvg_results:
        lines.extend([
            "## 4. HVG Projection Impact",
            "",
            "| Dataset | All Genes (b/s) | HVG 2K (b/s) | Speedup | Target | Pass? |",
            "|---------|-----------------|--------------|---------|--------|-------|",
        ])
        for r in hvg_results:
            h = r["benchmarks"]["hvg_impact"]
            lines.append(
                f"| {r['dataset']} | {h['all_genes_bps']:,.1f} "
                f"| {h['hvg_2k_bps']:,.1f} | {h['speedup']:.2f}× "
                f"| ≥5× | {'PASS ✅' if h['pass'] else 'FAIL ❌'} |"
            )
        lines.append("")

    # Baselines comparison
    lines.extend([
        "## 5. SOTA Comparison (batches/sec)",
        "",
        "| Dataset | SCX | AnnData | TileDB-SOMA | scDataLoader | BPCells |",
        "|---------|-----|---------|-------------|--------------|---------|",
    ])
    for r in all_results:
        scx_bps = r["benchmarks"].get("throughput", {}).get("batches_per_sec", "—")
        ann_bps = r["benchmarks"].get("baseline_anndata", {}).get("batches_per_sec", "—")
        soma_bps = r["benchmarks"].get("baseline_soma", {}).get("batches_per_sec", "—")
        scdl_bps = r["benchmarks"].get("baseline_scdataloader", {}).get("batches_per_sec", "—")
        bpc_bps = r["benchmarks"].get("baseline_bpcells", {}).get("batches_per_sec", "—")
        # Handle error cases
        for name in ["baseline_anndata", "baseline_soma", "baseline_scdataloader", "baseline_bpcells"]:
            b = r["benchmarks"].get(name, {})
            if "error" in b:
                if name == "baseline_anndata":
                    ann_bps = "N/A"
                elif name == "baseline_soma":
                    soma_bps = "N/A"
                elif name == "baseline_scdataloader":
                    scdl_bps = "N/A"
                elif name == "baseline_bpcells":
                    bpc_bps = "N/A"
        lines.append(f"| {r['dataset']} | {scx_bps} | {ann_bps} | {soma_bps} | {scdl_bps} | {bpc_bps} |")

    # SCX vs SOMA speedup
    for r in all_results:
        scx_bps = r["benchmarks"].get("throughput", {}).get("batches_per_sec", 0)
        soma_bps = r["benchmarks"].get("baseline_soma", {}).get("batches_per_sec", 0)
        if scx_bps and soma_bps and isinstance(soma_bps, (int, float)) and soma_bps > 0:
            speedup = scx_bps / soma_bps
            lines.append(f"\n**SCX vs TileDB-SOMA-ML on {r['dataset']}**: {speedup:.1f}× "
                         f"(target ≥2×: {'PASS ✅' if speedup >= 2 else 'FAIL ❌'})")
    lines.append("")

    # GPU benchmarks
    gpu_results = [r for r in all_results if "gpu_utilization" in r["benchmarks"]
                   and "error" not in r["benchmarks"].get("gpu_utilization", {})]
    if gpu_results:
        lines.extend([
            "## 6. GPU Utilization",
            "",
            "| Dataset | Avg GPU Util | Batches/sec | Target | Pass? |",
            "|---------|-------------|-------------|--------|-------|",
        ])
        for r in gpu_results:
            g = r["benchmarks"]["gpu_utilization"]
            lines.append(
                f"| {r['dataset']} | {g['avg_gpu_util_pct']:.1f}% "
                f"| {g['batches_per_sec']:,.1f} "
                f"| ≥{g['target_util_pct']}% "
                f"| {'PASS ✅' if g['pass'] else 'FAIL ❌'} |"
            )
        lines.append("")

    # Multi-GPU scaling
    multi_gpu_results = [r for r in all_results
                        if any(k.startswith("multi_gpu_") for k in r["benchmarks"])]
    if multi_gpu_results:
        lines.extend([
            "## 7. Multi-GPU Scaling",
            "",
            "| Dataset | GPUs | Batches/sec | Scaling Efficiency |",
            "|---------|------|-------------|-------------------|",
        ])
        for r in multi_gpu_results:
            base_bps = None
            for ng in [1, 2, 4]:
                mg = r["benchmarks"].get(f"multi_gpu_{ng}", {})
                if "error" in mg:
                    continue
                bps = mg.get("batches_per_sec", 0)
                if ng == 1:
                    base_bps = bps
                efficiency = f"{bps / base_bps * 100 / ng:.0f}%" if base_bps else "—"
                lines.append(f"| {r['dataset']} | {ng} | {bps:,.1f} | {efficiency} |")
        lines.append("")

    # Go/No-Go summary
    lines.extend([
        "## Summary — Go/No-Go Criteria",
        "",
        "| Criterion | Result |",
        "|-----------|--------|",
    ])

    for r in all_results:
        ds = r["dataset"]
        tp = r["benchmarks"].get("throughput", {})
        ttfb = r["benchmarks"].get("time_to_first_batch", {})
        mem = r["benchmarks"].get("memory", {})
        soma = r["benchmarks"].get("baseline_soma", {})
        gpu = r["benchmarks"].get("gpu_utilization", {})

        if tp and "error" not in tp:
            bps = tp.get("batches_per_sec", 0)
            target = 500 if "100k" in ds else (200 if "10m" in ds else 0)
            if target:
                lines.append(
                    f"| {ds}: ≥{target} batches/sec | "
                    f"{'PASS ✅' if bps >= target else 'FAIL ❌'} ({bps:.0f}) |"
                )

        if ttfb and "error" not in ttfb:
            lines.append(
                f"| {ds}: time-to-first-batch <2s | "
                f"{'PASS ✅' if ttfb.get('pass') else 'FAIL ❌'} ({ttfb.get('median_s', 0):.3f}s) |"
            )

        if soma and "error" not in soma:
            scx_bps = tp.get("batches_per_sec", 0)
            soma_bps = soma.get("batches_per_sec", 0)
            if soma_bps > 0:
                speedup = scx_bps / soma_bps
                lines.append(
                    f"| {ds}: SCX ≥2× SOMA throughput | "
                    f"{'PASS ✅' if speedup >= 2 else 'FAIL ❌'} ({speedup:.1f}×) |"
                )

        if gpu and "error" not in gpu:
            util = gpu.get("avg_gpu_util_pct", 0)
            lines.append(
                f"| {ds}: GPU util ≥85% | "
                f"{'PASS ✅' if util >= 85 else 'FAIL ❌'} ({util:.0f}%) |"
            )

    lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "The SCX training loader uses a triple-buffered pipeline architecture:",
        "tokio I/O → rayon decode → Python consumer. Key performance factors:",
        "",
        "- **HVG projection** at decode time avoids materializing full-width dense rows,",
        "  providing significant speedup proportional to gene count reduction.",
        "- **Fused normalize+log1p** eliminates a second pass over the dense matrix.",
        "- **Memory budget** is enforced by auto-tuning shard_group_size and prefetch_batches.",
        "  Peak RSS should be roughly constant regardless of dataset size.",
        "- **GPU utilization** depends on the balance between loader throughput and model compute.",
        "  With a lightweight simulated forward pass, GPU util may be limited by data transfer;",
        "  real models with heavier compute will see higher utilization.",
        "- **TileDB-SOMA-ML** comparison is the primary Go/No-Go criterion. SCX's advantage",
        "  comes from the compressed format, decode-time projection, and fused operations.",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SCX Training Loader Benchmark")
    parser.add_argument("--dataset", default="pbmc3k",
                        choices=list(DATASETS.keys()),
                        help="Dataset to benchmark")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run on all available datasets")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (fewer epochs, skip baselines)")
    parser.add_argument("--n-gpus", type=int, default=0,
                        help="Number of GPUs (0=CPU-only)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    all_results = []

    if args.all_datasets:
        datasets = [name for name, ds in DATASETS.items() if ds["h5ad"].exists()]
    else:
        datasets = [args.dataset]

    for ds_name in datasets:
        if not DATASETS[ds_name]["h5ad"].exists():
            print(f"SKIP: {ds_name} (h5ad not found)")
            continue
        results = run_all_benchmarks(ds_name, n_gpus=args.n_gpus, smoke=args.smoke)
        all_results.append(results)

    if not all_results:
        print("No results collected!")
        return

    # Generate report
    report = generate_report(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "training_loader_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")

    # Save raw JSON
    json_path = args.output or str(RESULTS_DIR / "training_loader_benchmark.json")
    Path(json_path).write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Raw results: {json_path}")


if __name__ == "__main__":
    ensure_release_build()
    main()
