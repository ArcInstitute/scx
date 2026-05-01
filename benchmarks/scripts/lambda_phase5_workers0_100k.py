"""Minimal workers0-only 100k-cell benchmark — works with both main and
deadlock-fix pyscx (no Phase 5.3 dependency)."""

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

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


N_OBS = 100_000
N_VARS = 2_000
NNZ_PER_CELL_AVG = 600
BATCH_SIZE = 64
N_WARMUP = 1
N_RUNS = 3


def _build_fixture(path: Path, seed: int = 0) -> None:
    print(f"[fixture] writing {path}", flush=True)
    import pyscx
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(N_OBS):
        n = max(1, int(rng.poisson(NNZ_PER_CELL_AVG)))
        cidx = rng.choice(N_VARS, size=min(n, N_VARS), replace=False)
        v = rng.integers(1, 64, size=cidx.size).astype(np.float32)
        rows.append(np.full(cidx.size, i, dtype=np.int64))
        cols.append(cidx.astype(np.int64))
        vals.append(v)
    X = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N_OBS, N_VARS), dtype=np.float32,
    )
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )
    pyscx.from_anndata(adata, str(path))


def _run_one_epoch(scx_path: str, hvg: bool, normalize: bool) -> tuple[int, int]:
    import pyscx
    n_hvg = 256
    hvg_indices = list(range(n_hvg)) if hvg else None
    ds = pyscx.TrainingDataset(
        scx_path, batch_size=BATCH_SIZE,
        hvg_indices=hvg_indices, normalize=normalize, log1p=normalize, seed=42,
    )
    n_batches = n_cells = 0
    for batch in ds:
        n_batches += 1
        n_cells += batch["X"].shape[0]
    if hasattr(ds, "close"):
        ds.close()
    return n_batches, n_cells


def _bench(label, fn, *args):
    print(f"[bench] {label}: warmup", flush=True)
    for _ in range(N_WARMUP):
        fn(*args); gc.collect()
    bps_runs, cps_runs, wall_runs = [], [], []
    for i in range(N_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        nb, nc = fn(*args)
        wall = time.perf_counter() - t0
        bps = nb / wall if wall > 0 else 0
        cps = nc / wall if wall > 0 else 0
        bps_runs.append(bps); cps_runs.append(cps); wall_runs.append(wall)
        print(f"[bench] {label} run {i+1}/{N_RUNS}: wall={wall:.2f}s bps={bps:.1f} cps={cps:.0f} nb={nb}", flush=True)
    return {
        "median_bps": round(statistics.median(bps_runs), 2),
        "median_cps": round(statistics.median(cps_runs), 0),
        "median_wall_s": round(statistics.median(wall_runs), 3),
        "all_bps": [round(b, 2) for b in bps_runs],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--scx-path", default=None)
    args = p.parse_args()

    print(f"[info] node={os.uname().nodename} job={os.environ.get('SLURM_JOB_ID')}", flush=True)
    import pyscx
    print(f"[info] pyscx loaded from: {pyscx.__file__}", flush=True)

    cleanup = None
    if args.scx_path:
        scx = Path(args.scx_path)
        if not scx.exists():
            sys.exit(f"missing scx-path: {scx}")
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="scx_phase5_w0_")
        scx = Path(cleanup.name) / "fixture.scx"
        _build_fixture(scx)

    res = {
        "node": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_obs": N_OBS, "n_vars": N_VARS, "batch_size": BATCH_SIZE,
        "pyscx_file": pyscx.__file__,
        "scenarios": {
            "raw_workers0":      _bench("raw_workers0",      _run_one_epoch, str(scx), False, False),
            "hvg_norm_workers0": _bench("hvg_norm_workers0", _run_one_epoch, str(scx), True, True),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[done] -> {out}", flush=True)
    print(json.dumps(res["scenarios"], indent=2), flush=True)


if __name__ == "__main__":
    main()
