#!/usr/bin/env python3
"""Top-level driver for Harmony scaling + comparison benchmarks.

Given a dataset name + implementation + device:
  1. Load / compute the PCA embedding (cache under SCX_WORK_DIR/benchmarks/pca).
  2. Extract batch labels from obs.
  3. Dispatch to the correct Python interpreter (pyscx requires `.venv` on
     CPU or `scx-gpu` conda env on GPU; harmonypy + R harmony runners can
     use the same CPU interpreter).
  4. Run `harmony_bench_worker.py` as a subprocess so RSS tracking is
     accurate.
  5. Write the JSON produced by the worker to
     `benchmarks/results/harmony/runs/<impl>_<dataset>_d<PCS>_K<K>_<job>.json`,
     adding dataset metadata and system info.

Expected SLURM flow (see slurm_harmony_bench.sh): one sbatch per
(impl, dataset, device) triple invoking this script with the right args.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "scripts"))
from bench_env import DATA_DIR, WORK_DIR  # noqa: E402


# Batch-column choice per dataset (discovered during Phase 6 planning).
BATCH_COLS: dict[str, list[str]] = {
    "pbmc3k": [],                 # no obs cols; synthesise
    "pbmc10k": [],                # no obs cols; synthesise
    "smartseq2": ["dataset_id"],  # 48 levels
    "tabula_sapiens_100k": ["donor_id"],  # 119 levels
    "census_500k": ["dataset_id"],
    "census_1m": ["dataset_id"],
    "census_5m": ["dataset_id"],
}


# Interpreter map (impl/device → python to invoke the worker).
def _interp(impl: str, device: str) -> str:
    if impl == "scx_accel_gpu" or device == "gpu":
        return "/home/nickyoungblut/miniforge3/envs/scx-gpu/bin/python"
    if impl == "scx_accel_cpu":
        return str(REPO_ROOT / ".venv/bin/python")
    if impl == "harmonypy":
        return "/home/nickyoungblut/miniforge3/envs/scx-bench/bin/python"
    if impl == "r_harmony":
        # Worker uses the .venv python but shells out to rscx env's Rscript.
        return str(REPO_ROOT / ".venv/bin/python")
    raise ValueError(f"unknown impl: {impl}")


def _cache_paths(dataset: str, n_pcs: int, n_cells_cap: int | None) -> tuple[Path, Path, Path]:
    base = WORK_DIR / "benchmarks" / "pca_cache"
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"_d{n_pcs}" + (f"_n{n_cells_cap}" if n_cells_cap else "")
    return (
        base / f"{dataset}{suffix}.pca.npy",
        base / f"{dataset}{suffix}.batch.npy",
        base / f"{dataset}{suffix}.meta.json",
    )


def _build_pca_cache(dataset: str, n_pcs: int, n_cells_cap: int | None) -> tuple[Path, Path, Path]:
    pca_p, batch_p, meta_p = _cache_paths(dataset, n_pcs, n_cells_cap)
    if pca_p.exists() and batch_p.exists() and meta_p.exists():
        return pca_p, batch_p, meta_p

    import anndata as ad
    import scanpy as sc  # noqa: F401  (only needed in env with scanpy)

    h5 = DATA_DIR / f"{dataset}.h5ad"
    if not h5.exists():
        raise FileNotFoundError(h5)

    print(f"[cache] building PCA for {dataset} (d={n_pcs})", flush=True)
    adata = ad.read_h5ad(h5)
    if n_cells_cap and adata.n_obs > n_cells_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(adata.n_obs, size=n_cells_cap, replace=False)
        idx.sort()
        adata = adata[idx].copy()

    # Normalise if counts look un-normalised. Many census h5ads already
    # contain normalised values; detect via max.
    x_max = float(adata.X.max())
    if x_max > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    # HVG + PCA WITHOUT sc.pp.scale: scaling densifies the matrix, which
    # OOMs on D4+ at realistic SLURM memory budgets. sc.tl.pca(zero_center=True)
    # handles centering without materialising the dense scaled matrix.
    if adata.n_vars > 2000:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
            adata = adata[:, adata.var["highly_variable"]].copy()
        except Exception:
            # Fall back to the full var set if HVG selection fails (e.g. if
            # the dataset already carries normalised log-counts).
            pass
    sc.tl.pca(adata, n_comps=n_pcs, random_state=0, zero_center=True)

    # Batch column.
    cols = BATCH_COLS.get(dataset, [])
    if not cols:
        # Synthesise 3 balanced batches.
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 3, size=adata.n_obs, dtype=np.int32)
    else:
        col = adata.obs[cols[0]].astype(str)
        categories, codes = np.unique(col.to_numpy(), return_inverse=True)
        batch = codes.astype(np.int32)

    np.save(pca_p, adata.obsm["X_pca"].astype(np.float32))
    np.save(batch_p, batch)
    with open(meta_p, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "n_obs": int(adata.n_obs),
                "n_pcs": n_pcs,
                "n_cells_cap": n_cells_cap,
                "batch_col": cols[0] if cols else "synthetic",
                "n_batches": int(batch.max()) + 1,
            },
            f,
            indent=2,
        )
    return pca_p, batch_p, meta_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BATCH_COLS))
    ap.add_argument("--impl", required=True,
                    choices=["scx_accel_cpu", "scx_accel_gpu", "harmonypy", "r_harmony"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-pcs", type=int, default=30)
    ap.add_argument("--n-clusters", type=int, default=100)
    ap.add_argument("--theta", type=float, default=2.0)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cells-cap", type=int, default=None,
                    help="Optional cap on N (useful for dev/smoke tests)")
    ap.add_argument("--reference", default=None,
                    help="Optional R-harmony .npz reference for per-PC correlation")
    ap.add_argument("--output-dir",
                    default=str(REPO_ROOT / "benchmarks/results/harmony/runs"))
    ap.add_argument("--tag", default=None,
                    help="Optional suffix for the output filename")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pca_p, batch_p, meta_p = _build_pca_cache(
        args.dataset, args.n_pcs, args.n_cells_cap
    )
    with open(meta_p) as f:
        ds_meta = json.load(f)

    worker = Path(__file__).resolve().parent / "harmony_bench_worker.py"
    interpreter = _interp(args.impl, args.device)

    cmd = [
        interpreter, str(worker),
        "--impl", args.impl,
        "--pca", str(pca_p),
        "--batch", str(batch_p),
        "--n-clusters", str(args.n_clusters),
        "--theta", str(args.theta),
        "--max-iter", str(args.max_iter),
        "--seed", str(args.seed),
    ]
    if args.reference:
        cmd += ["--reference", args.reference]

    # Inherit parent env + set the few vars the workers need.
    # * scx-gpu uses cupy, which calls os.path.join(CONDA_PREFIX, ...) when
    #   probing for CUDA — blank CONDA_PREFIX crashes cupy at import time.
    # * rscx's R scripts call reticulate which needs RETICULATE_PYTHON
    #   pointed at a Python with numpy installed.
    env = os.environ.copy()
    if args.impl == "scx_accel_gpu" or args.device == "gpu":
        env.setdefault("CONDA_PREFIX", "/home/nickyoungblut/miniforge3/envs/scx-gpu")
    if args.impl == "r_harmony":
        env.setdefault(
            "RETICULATE_PYTHON",
            "/home/nickyoungblut/miniforge3/envs/scx-bench/bin/python",
        )

    print(f"[driver] {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=8 * 3600,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        print("[driver] TIMEOUT", flush=True)
        worker_json = {"ok": False, "error": f"TimeoutExpired: {e}"}
    else:
        # The worker prints exactly one JSON line on stdout.
        last_line = ""
        for line in proc.stdout.strip().splitlines():
            if line.startswith("{"):
                last_line = line
        if not last_line:
            worker_json = {
                "ok": False,
                "error": f"no JSON from worker (exit={proc.returncode}); "
                         f"stderr={proc.stderr[-400:]}",
            }
        else:
            try:
                worker_json = json.loads(last_line)
            except json.JSONDecodeError as e:
                worker_json = {
                    "ok": False,
                    "error": f"JSON decode: {e}; raw={last_line[:200]}",
                }

    result = {
        "benchmark": "harmony_integrate",
        "impl": args.impl,
        "device": args.device,
        "dataset": args.dataset,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "dataset_meta": ds_meta,
        "params": {
            "n_pcs": args.n_pcs,
            "n_clusters": args.n_clusters,
            "theta": args.theta,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
        "run": worker_json,
    }

    tag = f"_{args.tag}" if args.tag else ""
    out_name = (
        f"harmony_{args.impl}_{args.dataset}_d{args.n_pcs}_K{args.n_clusters}"
        f"_{args.device}{tag}.json"
    )
    out_path = Path(args.output_dir) / out_name
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[driver] wrote {out_path}", flush=True)

    return 0 if worker_json.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
