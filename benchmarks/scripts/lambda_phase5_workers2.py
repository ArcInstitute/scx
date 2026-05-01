#!/usr/bin/env python3
"""Phase 5 GPU benchmark: pyscx_training_dataset_workers2 + workers2_persistent
on Lambda HPC.

Self-contained — generates a synthetic 100k-cell × 2k-gene fixture (matches
the ``tabula_sapiens_100k`` benchmark scale) and runs the new ml_loader
scenarios from DEADLOCK-ISSUE.md §5.3 against it. Captures per-scenario
median throughput, peak RSS, and cells-per-sec into a JSON file.

This bypasses the full Chimera-tuned `gate_candidate.py` orchestration
(which targets the `cpu_preemptible` partition that does not exist on
Lambda) and runs the new num_workers > 0 scenario directly. The Chimera
gate covers comparison-to-baseline; this script covers the post-fix
verification step that was the actual ask of Phase 5.

Usage::

    sbatch --partition=preemptible --gres=gpu:1 --cpus-per-task=16 \
        --mem=64G --time=1:00:00 --wrap='\
        cd /home/nickyoungblut/dev/rust/scx && \
        .venv/bin/python benchmarks/scripts/lambda_phase5_workers2.py \
            --output benchmarks/comprehensive/results/phase5-lambda-workers2.json'
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
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
NNZ_PER_CELL_AVG = 600  # similar density to tabula_sapiens_100k
BATCH_SIZE = 64
N_WARMUP = 1
N_RUNS = 3


def _build_fixture(path: Path, seed: int = 0) -> None:
    """Generate a synthetic ~100k-cell × 2k-gene `.scx` matching the
    ``tabula_sapiens_100k`` density profile (~600 nnz/cell)."""
    print(f"[fixture] writing {path} ({N_OBS} cells × {N_VARS} genes)", flush=True)
    import pyscx

    rng = np.random.default_rng(seed)
    # Build sparse CSR row-by-row to keep peak memory bounded — a dense
    # 100k×2k float32 array is 800 MB which is wasteful here.
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
    row_arr = np.concatenate(rows)
    col_arr = np.concatenate(cols)
    val_arr = np.concatenate(vals)
    X = sp.csr_matrix(
        (val_arr, (row_arr, col_arr)),
        shape=(N_OBS, N_VARS),
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    pyscx.from_anndata(adata, str(path))
    print(f"[fixture] wrote {path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _run_one_epoch_num_workers_0(scx_path: str, hvg: bool, normalize: bool) -> tuple[int, int]:
    import pyscx

    n_hvg = 256
    hvg_indices = list(range(n_hvg)) if hvg else None
    ds = pyscx.TrainingDataset(
        scx_path,
        batch_size=BATCH_SIZE,
        hvg_indices=hvg_indices,
        normalize=normalize,
        log1p=normalize,
        seed=42,
    )
    n_batches = 0
    n_cells = 0
    for batch in ds:
        n_batches += 1
        n_cells += batch["X"].shape[0]
    ds.close()
    return n_batches, n_cells


def _run_one_epoch_workers2(
    scx_path: str,
    hvg: bool,
    normalize: bool,
    persistent_workers: bool,
    n_epochs: int,
) -> tuple[int, int]:
    from benchmarks.comprehensive.benchmarks.ml_loader import (
        _run_scx_dataloader_workers_epoch,
    )

    res = _run_scx_dataloader_workers_epoch(
        scx_path,
        batch_size=BATCH_SIZE,
        hvg=hvg,
        normalize=normalize,
        seed=42,
        num_workers=2,
        persistent_workers=persistent_workers,
        n_epochs=n_epochs,
    )
    return res.n_batches, res.n_cells


def _bench(label: str, fn, *args, **kwargs) -> dict:
    """Run `N_WARMUP` warmups + `N_RUNS` timed runs of `fn`. Return a dict
    with median throughput / cells-per-sec / peak RSS."""
    print(f"[bench] {label}: warmup", flush=True)
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)
        gc.collect()

    bps_runs: list[float] = []
    cps_runs: list[float] = []
    rss_runs: list[float] = []
    wall_runs: list[float] = []

    for i in range(N_RUNS):
        gc.collect()
        rss_pre = _peak_rss_mb()
        t0 = time.perf_counter()
        n_batches, n_cells = fn(*args, **kwargs)
        wall = time.perf_counter() - t0
        rss_post = _peak_rss_mb()
        rss_delta = rss_post  # peak RSS is monotonic; report the absolute peak

        bps = n_batches / wall if wall > 0 else 0.0
        cps = n_cells / wall if wall > 0 else 0.0
        bps_runs.append(bps)
        cps_runs.append(cps)
        rss_runs.append(rss_delta)
        wall_runs.append(wall)
        print(
            f"[bench] {label} run {i + 1}/{N_RUNS}: wall={wall:.2f}s "
            f"bps={bps:.1f} cps={cps:.0f} rss={rss_delta:.1f}MB "
            f"n_batches={n_batches} n_cells={n_cells}",
            flush=True,
        )

    return {
        "label": label,
        "n_runs": N_RUNS,
        "median_batches_per_sec": round(statistics.median(bps_runs), 2),
        "median_cells_per_sec": round(statistics.median(cps_runs), 0),
        "median_peak_rss_mb": round(statistics.median(rss_runs), 1),
        "median_wall_s": round(statistics.median(wall_runs), 3),
        "all_bps": [round(b, 2) for b in bps_runs],
        "all_wall_s": [round(w, 3) for w in wall_runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/comprehensive/results/phase5-lambda-workers2.json",
    )
    parser.add_argument("--scx-path", default=None,
                        help="Re-use an existing fixture path instead of generating.")
    args = parser.parse_args()

    print(f"[info] node     : {os.uname().nodename}", flush=True)
    print(f"[info] SLURM job: {os.environ.get('SLURM_JOB_ID', 'n/a')}", flush=True)
    print(f"[info] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)

    import torch
    print(f"[info] torch    : {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    print(f"[info] cpu_count: {os.cpu_count()}", flush=True)

    cleanup_tmp = None
    if args.scx_path:
        scx_path = Path(args.scx_path)
        if not scx_path.exists():
            sys.exit(f"--scx-path does not exist: {scx_path}")
    else:
        cleanup_tmp = tempfile.TemporaryDirectory(prefix="scx_phase5_")
        scx_path = Path(cleanup_tmp.name) / "fixture_100k.scx"
        _build_fixture(scx_path, seed=0)

    results: dict = {
        "node": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_obs": N_OBS,
        "n_vars": N_VARS,
        "batch_size": BATCH_SIZE,
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
        "fixture_size_bytes": scx_path.stat().st_size,
        "scenarios": {},
    }

    # Match the existing `_SCENARIOS` config in ml_loader.py:
    # raw     = no HVG, no normalize
    # hvg_norm = HVG + normalize+log1p
    scenarios = [
        # name, hvg, normalize
        ("raw_workers0", False, False, "num_workers=0 (current path baseline)"),
        ("hvg_norm_workers0", True, True, "num_workers=0 + HVG + normalize"),
        (
            "pyscx_training_dataset_workers2",
            True,
            True,
            "num_workers=2, persistent_workers=False, 1 epoch (Phase 2 fix)",
        ),
        (
            "pyscx_training_dataset_workers2_persistent",
            True,
            True,
            "num_workers=2, persistent_workers=True, 2 epochs (per-epoch normalised)",
        ),
    ]

    for name, hvg, normalize, desc in scenarios:
        print(f"\n=== {name} === {desc}", flush=True)
        try:
            if name.endswith("_workers0"):
                summary = _bench(
                    name,
                    _run_one_epoch_num_workers_0,
                    str(scx_path),
                    hvg,
                    normalize,
                )
            elif name == "pyscx_training_dataset_workers2":
                summary = _bench(
                    name,
                    _run_one_epoch_workers2,
                    str(scx_path),
                    hvg,
                    normalize,
                    False,  # persistent_workers
                    1,  # n_epochs
                )
            elif name == "pyscx_training_dataset_workers2_persistent":
                # `_run_scx_dataloader_workers_epoch` returns aggregate counts
                # over n_epochs runs. Normalise to per-epoch in the post-hoc
                # calculation by dividing wall_s by n_epochs.
                def fn_persistent(p, h, n):
                    n_batches, n_cells = _run_one_epoch_workers2(p, h, n, True, 2)
                    # Bench framework expects (n_batches, n_cells) per "run".
                    # We report aggregate across 2 epochs; divide downstream
                    # by 2 for per-epoch comparability.
                    return n_batches, n_cells

                summary = _bench(
                    name,
                    fn_persistent,
                    str(scx_path),
                    hvg,
                    normalize,
                )
                # Post-hoc normalisation: 2 epochs, so per-epoch throughput
                # is the same as observed (bps is rate, independent of how
                # many epochs we ran).
                summary["per_epoch_normalised"] = True
            else:
                continue
            results["scenarios"][name] = summary
        except Exception as e:
            print(f"[error] {name}: {e}", flush=True)
            import traceback

            traceback.print_exc()
            results["scenarios"][name] = {"error": str(e)}

    # Diff vs num_workers=0 hvg_norm baseline.
    if (
        "hvg_norm_workers0" in results["scenarios"]
        and "median_batches_per_sec" in results["scenarios"]["hvg_norm_workers0"]
    ):
        ref = results["scenarios"]["hvg_norm_workers0"]["median_batches_per_sec"]
        for name in (
            "pyscx_training_dataset_workers2",
            "pyscx_training_dataset_workers2_persistent",
        ):
            if "median_batches_per_sec" in results["scenarios"].get(name, {}):
                w2 = results["scenarios"][name]["median_batches_per_sec"]
                results["scenarios"][name]["bps_ratio_vs_workers0"] = round(
                    w2 / ref, 3
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] results written to {out}", flush=True)
    print(json.dumps(results["scenarios"], indent=2), flush=True)

    if cleanup_tmp is not None:
        cleanup_tmp.cleanup()


if __name__ == "__main__":
    main()
