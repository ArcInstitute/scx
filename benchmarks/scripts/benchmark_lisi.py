#!/usr/bin/env python3
"""LISI benchmark — scx-accel vs R `lisi` package.

Caches the same PCA + batch arrays as benchmark_harmony.py. Emits one JSON
per (impl, dataset) run: wall_s, peak_rss_mb, mean_lisi, median_lisi.

Usage:
    python benchmark_lisi.py --dataset census_500k --impl scx_accel
    python benchmark_lisi.py --dataset census_500k --impl r_lisi
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "scripts"))
from bench_env import WORK_DIR  # noqa: E402

from benchmark_harmony import _build_pca_cache, BATCH_COLS  # noqa: E402


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _run_scx(pca: np.ndarray, batch: np.ndarray, perplexity: float) -> dict:
    import pyscx
    import anndata as ad
    import pandas as pd

    labels_str = [f"b{v}" for v in batch.tolist()]
    adata = ad.AnnData(
        X=np.zeros((pca.shape[0], 1), dtype=np.float32),
        obs=pd.DataFrame({"batch": pd.Categorical(labels_str)}),
    )
    adata.obsm["X_pca"] = pca.astype(np.float32)
    t0 = time.perf_counter()
    arr = pyscx.accel.compute_lisi(adata, "batch", perplexity=perplexity)
    wall_s = time.perf_counter() - t0
    return {
        "wall_s": round(wall_s, 3),
        "mean_lisi": float(np.mean(arr)),
        "median_lisi": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _run_r(pca: np.ndarray, batch: np.ndarray, perplexity: float) -> dict:
    """Invoke R `lisi::compute_lisi` via Rscript."""
    tmp = Path("/tmp") / f"lisi_{int(time.time())}"
    tmp.mkdir(parents=True, exist_ok=True)
    pca_p = tmp / "pca.npy"
    batch_p = tmp / "batch.npy"
    out_p = tmp / "lisi.npy"
    np.save(pca_p, pca.astype(np.float32))
    np.save(batch_p, batch)
    r_body = f"""
      library(reticulate)
      library(lisi)
      np <- import("numpy", convert=TRUE)
      pca <- np$load("{pca_p}")
      batch <- np$load("{batch_p}")
      meta <- data.frame(batch=as.character(batch), stringsAsFactors=FALSE)
      t0 <- Sys.time()
      out <- compute_lisi(as.matrix(pca), meta, "batch", perplexity={perplexity})
      elapsed <- as.numeric(difftime(Sys.time(), t0, units="secs"))
      vec <- as.numeric(out$batch)
      np$save("{out_p}", np$asarray(vec, dtype="float64"))
      cat(sprintf("ELAPSED=%f\\n", elapsed))
    """
    rscript = "/home/nickyoungblut/miniforge3/envs/rscx/bin/Rscript"
    proc = subprocess.run(
        [rscript, "-e", r_body], check=True, capture_output=True, text=True
    )
    elapsed = 0.0
    for line in proc.stdout.splitlines():
        if line.startswith("ELAPSED="):
            elapsed = float(line.split("=", 1)[1])
    arr = np.load(out_p)
    return {
        "wall_s": round(elapsed, 3),
        "mean_lisi": float(np.mean(arr)),
        "median_lisi": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BATCH_COLS))
    ap.add_argument("--impl", required=True, choices=["scx_accel", "r_lisi"])
    ap.add_argument("--n-pcs", type=int, default=30)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-cells-cap", type=int, default=None)
    ap.add_argument("--output-dir",
                    default=str(REPO_ROOT / "benchmarks/results/harmony/runs"))
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pca_p, batch_p, meta_p = _build_pca_cache(
        args.dataset, args.n_pcs, args.n_cells_cap
    )
    with open(meta_p) as f:
        ds_meta = json.load(f)
    pca = np.load(pca_p)
    batch = np.load(batch_p)

    gc.collect()
    t0 = time.perf_counter()
    err = None
    out = None
    try:
        if args.impl == "scx_accel":
            out = _run_scx(pca, batch, args.perplexity)
        else:
            out = _run_r(pca, batch, args.perplexity)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    total_wall_s = time.perf_counter() - t0
    peak_rss = _peak_rss_mb()

    result: dict = {
        "benchmark": "lisi",
        "impl": args.impl,
        "dataset": args.dataset,
        "dataset_meta": ds_meta,
        "perplexity": args.perplexity,
        "peak_rss_mb": round(peak_rss, 1),
        "total_wall_s": round(total_wall_s, 3),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "ok": err is None,
    }
    if err:
        result["error"] = err
    if out:
        result["run"] = out

    name = f"lisi_{args.impl}_{args.dataset}_d{args.n_pcs}.json"
    out_path = Path(args.output_dir) / name
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[lisi] wrote {out_path}", flush=True)
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
