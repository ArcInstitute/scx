#!/usr/bin/env python3
"""Phase 7.4 GPU benchmark: pyscx_index_plan_dataset_workers2 on Lambda HPC.

Self-contained — generates a synthetic 100 000-cell × 2 000-gene fixture
(matches the `tabula_sapiens_100k` benchmark scale) and runs the new
`IndexPlanDataset` workers2 scenario from DEADLOCK-ISSUE.md §7.4 against
it. Captures per-scenario median throughput, peak RSS, and cells-per-sec
into a JSON file.

Mirrors `lambda_phase5_workers2.py` (which targets the TrainingDataset
workers2 scenarios) but for the IndexPlanDataset path.

Usage::

    sbatch benchmarks/scripts/lambda_phase7_index_plan_workers2.sbatch
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Repo root so the benchmarks package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


N_OBS = 100_000
N_VARS = 2_000
NNZ_PER_CELL_AVG = 600
PAIRS_PER_BATCH = 64
N_BATCHES = 200
N_WARMUP = 1
N_RUNS = 3


def _build_fixture(path: Path, seed: int = 0) -> None:
    print(f"[fixture] writing {path} ({N_OBS} cells × {N_VARS} genes)", flush=True)
    import pyscx

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for i in range(N_OBS):
        n = max(1, int(rng.poisson(NNZ_PER_CELL_AVG)))
        cidx = rng.choice(N_VARS, size=min(n, N_VARS), replace=False)
        v = rng.integers(1, 64, size=cidx.size).astype(np.float32)
        rows.append(np.full(cidx.size, i, dtype=np.int64))
        cols.append(cidx.astype(np.int64))
        vals.append(v)
    X = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N_OBS, N_VARS),
        dtype=np.float32,
    )
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )
    pyscx.from_anndata(adata, str(path))
    print(f"[fixture] wrote {path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _run_workers0(scx_path: str) -> tuple[int, int]:
    """Reference: IndexPlanDataset driven directly (no DataLoader workers)."""
    import pyscx
    from benchmarks.comprehensive.benchmarks.index_plan import _random_plans

    ds = pyscx.IndexPlanDataset(
        scx_path,
        normalize=False,
        log1p=False,
        cache_shards=128,
        sort_by_shard=True,
        lookahead=4,
        max_plan_size=16384,
        max_memory_mb=8192,
    )
    n_batches = 0
    n_cells = 0
    plans = _random_plans(N_OBS, PAIRS_PER_BATCH, N_BATCHES, seed=0)
    for batch in ds.iter_with_plans(plans, lookahead=4):
        n_batches += 1
        n_cells += 2 * batch["X"].shape[0]
    return n_batches, n_cells


def _run_workers2(scx_path: str) -> tuple[int, int]:
    """Phase 7.4 workers2 path — the actual fix-acceptance scenario."""
    from benchmarks.comprehensive.benchmarks.index_plan import (
        _run_index_plan_workers2,
    )

    out = _run_index_plan_workers2(
        scx_path,
        n_obs=N_OBS,
        pairs_per_batch=PAIRS_PER_BATCH,
        n_batches=N_BATCHES,
        hvg_indices=None,
        normalize=False,
        sort_by_shard=True,
        lookahead=4,
        cache_shards=128,
        max_plan_size=16384,
    )
    return out.n_batches, out.n_cells


def _bench(label: str, fn, *args) -> dict:
    print(f"[bench] {label}: warmup", flush=True)
    for _ in range(N_WARMUP):
        fn(*args)
        gc.collect()

    bps_runs: list[float] = []
    cps_runs: list[float] = []
    rss_runs: list[float] = []
    wall_runs: list[float] = []
    for i in range(N_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        n_batches, n_cells = fn(*args)
        wall = time.perf_counter() - t0
        rss = _peak_rss_mb()
        bps = n_batches / wall if wall > 0 else 0.0
        cps = n_cells / wall if wall > 0 else 0.0
        bps_runs.append(bps)
        cps_runs.append(cps)
        rss_runs.append(rss)
        wall_runs.append(wall)
        print(
            f"[bench] {label} run {i+1}/{N_RUNS}: wall={wall:.2f}s "
            f"bps={bps:.1f} cps={cps:.0f} rss={rss:.1f}MB nb={n_batches}",
            flush=True,
        )
    return {
        "label": label,
        "n_runs": N_RUNS,
        "median_bps": round(statistics.median(bps_runs), 2),
        "median_cps": round(statistics.median(cps_runs), 0),
        "median_peak_rss_mb": round(statistics.median(rss_runs), 1),
        "median_wall_s": round(statistics.median(wall_runs), 3),
        "all_bps": [round(b, 2) for b in bps_runs],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--scx-path", default=None)
    args = p.parse_args()

    print(f"[info] node={os.uname().nodename} job={os.environ.get('SLURM_JOB_ID')}", flush=True)
    import torch

    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    cleanup = None
    if args.scx_path:
        scx = Path(args.scx_path)
        if not scx.exists():
            sys.exit(f"--scx-path missing: {scx}")
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="scx_phase7_w2_")
        scx = Path(cleanup.name) / "fixture_100k.scx"
        _build_fixture(scx)

    res = {
        "node": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_obs": N_OBS,
        "n_vars": N_VARS,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "n_batches": N_BATCHES,
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
        "scenarios": {
            "pyscx_index_plan_dataset_workers0": _bench(
                "workers0_reference", _run_workers0, str(scx)
            ),
            "pyscx_index_plan_dataset_workers2": _bench(
                "workers2", _run_workers2, str(scx)
            ),
        },
    }

    # Diff workers2 vs workers0 reference.
    w0 = res["scenarios"]["pyscx_index_plan_dataset_workers0"]["median_bps"]
    w2 = res["scenarios"]["pyscx_index_plan_dataset_workers2"]["median_bps"]
    res["scenarios"]["pyscx_index_plan_dataset_workers2"]["bps_ratio_vs_workers0"] = (
        round(w2 / w0, 3) if w0 > 0 else None
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n[done] -> {out}", flush=True)
    print(json.dumps(res["scenarios"], indent=2), flush=True)

    if cleanup is not None:
        cleanup.cleanup()


if __name__ == "__main__":
    main()
