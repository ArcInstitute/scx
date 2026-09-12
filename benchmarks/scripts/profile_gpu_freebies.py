#!/usr/bin/env python
"""PR-12 (OPT-GPU-1/-4/-10/-13) A/B profiler — the arms this PR can actually move.

Three measurements, one process each, so a `ru_maxrss` high-water belongs to
the thing it is reported against rather than to whatever ran before it in the
same interpreter:

``to_gpu_anndata``
    OPT-GPU-10. The multi-shard CSR assemble loop used to recover each shard's
    indptr with a ``dev.dtoh_copy``; on pageable host memory that is a
    host-synchronous copy, so it drained the stream once per shard. The
    decoders now hand the host vector back. **This only moves on a multi-shard
    file**, so the run asserts the shard count rather than hoping for it.

``pca_backed``
    OPT-GPU-13. GPU PCA's column-means pass moved onto
    ``col_means_and_sum_sq_prefetched``. No benchmark in the comprehensive
    harness reaches it: ``accel_pca``'s runner builds an **in-memory** adata, so
    every arm — ``__pyscx_gpu_streaming`` included — goes through pyscx's
    ``BorrowedCsrSource``, whose ``n_shards()`` is 1, and the prefetch pipeline
    takes its sequential fallback at ≤ 1 shard. A **backed** X is the only
    multi-shard GPU-PCA path, which is why this script exists instead of a new
    benchmark arm.

``pca_inmem``
    The falsification. The filed item asks for ``accel_pca / cosine_sim_mean``
    unchanged to the digit; this records the embedding itself, which is
    strictly stronger, and records a same-arm repeat alongside it so the
    cross-arm agreement has a noise floor to be read against. Expected: the two
    arms agree at least as well as one arm agrees with itself.

Run under ``sbatch`` on a GPU node in the ``scx-bench-gpu`` conda env; the
driver is ``benchmarks/scripts/_run_gpu_freebies_ab.sh``, which runs it once
per build.

Env overrides: ``GPU_FB_DATASETS`` (default
``tabula_sapiens_100k,census_500k`` — 7 and 31 CSR shards, and the removed
drain is per shard, so the pair is also a scaling check),
``GPU_FB_SMALL`` (default ``pbmc3k``), ``GPU_FB_RUNS``, ``GPU_FB_OUT``
(JSON path), ``GPU_FB_EMB_DIR`` (where the PCA embeddings are written).
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.bench_env import DATA_DIR  # noqa: E402

DATASETS = tuple(
    d.strip()
    for d in os.environ.get(
        "GPU_FB_DATASETS", "tabula_sapiens_100k,census_500k"
    ).split(",")
    if d.strip()
)
SMALL = os.environ.get("GPU_FB_SMALL", "pbmc3k")
N_RUNS = int(os.environ.get("GPU_FB_RUNS", 3))
OUT_PATH = os.environ.get("GPU_FB_OUT", "")
EMB_DIR = Path(os.environ.get("GPU_FB_EMB_DIR", "."))
N_COMPS = 50
SEED = 0


def _peak_rss_mb() -> float:
    """Process high-water RSS. `RUSAGE_SELF`, so it is this subprocess only."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------
# The three measurements. Each runs in its own process (see `_spawn`).
# --------------------------------------------------------------------------


def _measure_to_gpu_anndata(dataset: str, variant: str) -> dict:
    """`to_gpu_anndata` on a sidecar-carrying file of the named codec."""
    import tempfile

    import pyscx
    from benchmarks.comprehensive.benchmarks.accel_to_gpu_anndata import (
        ARMS,
        _prepare_sidecar_scx,
    )
    from benchmarks.comprehensive.config import DATASETS as DATASET_CONFIGS

    arm = ARMS[variant]
    cfg = DATASET_CONFIGS[dataset]
    with tempfile.TemporaryDirectory() as tmpdir:
        scx_path, n_obs, n_shards = _prepare_sidecar_scx(cfg, tmpdir, arm)
        # The premise. A single-shard file has no per-shard drain to remove, so
        # a flat result on one would say nothing about the change.
        if n_shards < 2:
            raise SystemExit(
                f"premise failed: {dataset}/{variant} prepared {n_shards} shard(s); "
                "the assemble-loop drain this measures is per shard"
            )
        walls: list[float] = []
        info: dict = {}
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            adata = pyscx.open(str(scx_path)).to_gpu_anndata(device="gpu")
            walls.append(time.perf_counter() - t0)
            info = dict(adata.uns.get("scx_accel", {}).get("to_gpu_anndata", {}))
            del adata
    return {
        "n_obs": n_obs,
        "n_shards": n_shards,
        "walls_s": walls,
        "wall_s": min(walls),
        "peak_rss_mb": _peak_rss_mb(),
        "transfer_mode": info.get("transfer_mode"),
        "bytes_uploaded": info.get("bytes_uploaded"),
    }


def _measure_pca_backed(dataset: str) -> dict:
    """GPU PCA over a **backed** X — the only multi-shard GPU-PCA path."""
    import numpy as np
    import pyscx

    scx_path = DATA_DIR / f"{dataset}_auto.scx"
    n_shards = int(pyscx.open(str(scx_path)).shard_count)
    # The premise, same as `to_gpu_anndata`'s: at one shard the prefetch
    # pipeline takes its sequential fallback and this measures nothing.
    if n_shards < 2:
        raise SystemExit(
            f"premise failed: {dataset}_auto.scx has {n_shards} shard(s); "
            "the decode-prefetch this measures needs more than one"
        )
    walls: list[float] = []
    emb = None
    info: dict = {}
    for _ in range(N_RUNS):
        adata = pyscx.open(str(scx_path)).to_anndata(backed=True)
        t0 = time.perf_counter()
        pyscx.accel.pca(adata, n_comps=N_COMPS, device="gpu", random_state=SEED)
        walls.append(time.perf_counter() - t0)
        emb = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        info = dict(adata.uns.get("scx_accel", {}).get("pca", {}))
        del adata
    if emb is not None:
        np.save(EMB_DIR / f"pca_backed__{dataset}.npy", emb)
    return {
        "n_shards": n_shards,
        "walls_s": walls,
        "wall_s": min(walls),
        "peak_rss_mb": _peak_rss_mb(),
        "route": info.get("route"),
        "emb_shape": list(emb.shape) if emb is not None else None,
    }


def _measure_pca_inmem(dataset: str) -> dict:
    """GPU PCA over an in-memory X — the `accel_pca` shape, expected flat.

    Two embeddings from the same build, back to back: the second is this arm's
    own run-to-run noise floor, which is what the cross-arm comparison has to
    be read against. Without it "the arms differ by 1e-6" has no scale.
    """
    import anndata
    import numpy as np
    import pyscx
    import scanpy as sc

    adata = anndata.read_h5ad(str(DATA_DIR / f"{dataset}.h5ad"))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = min(2000, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3", subset=True, span=0.3
        )
    except Exception:  # noqa: BLE001
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)

    embs = []
    walls: list[float] = []
    info: dict = {}
    for i in range(2):
        t0 = time.perf_counter()
        pyscx.accel.pca(adata, n_comps=N_COMPS, device="gpu", random_state=SEED)
        walls.append(time.perf_counter() - t0)
        embs.append(np.asarray(adata.obsm["X_pca"], dtype=np.float64))
        info = dict(adata.uns.get("scx_accel", {}).get("pca", {}))
    np.save(EMB_DIR / f"pca_inmem__{dataset}.npy", embs[0])
    np.save(EMB_DIR / f"pca_inmem_repeat__{dataset}.npy", embs[1])
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "walls_s": walls,
        "wall_s": min(walls),
        "peak_rss_mb": _peak_rss_mb(),
        "route": info.get("route"),
        "self_cosine_min": _cosine_min(embs[0], embs[1]),
    }


def _cosine_min(a, b) -> float:
    """Smallest |cosine| over matched PCA components — 1.0 means identical."""
    import numpy as np

    if a.shape != b.shape:
        return float("nan")
    out = []
    for k in range(a.shape[1]):
        u, v = a[:, k], b[:, k]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        out.append(1.0 if nu == 0 and nv == 0 else abs(float(u @ v) / (nu * nv + 1e-300)))
    return float(min(out)) if out else float("nan")


_MEASUREMENTS = {
    # argv: --one <op> <dataset> [variant]
    "to_gpu_anndata": lambda: _measure_to_gpu_anndata(sys.argv[3], sys.argv[4]),
    "pca_backed": lambda: _measure_pca_backed(sys.argv[3]),
    "pca_inmem": lambda: _measure_pca_inmem(sys.argv[3]),
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _spawn(op: str, *extra: str) -> dict:
    """Run one measurement in a fresh interpreter and parse its one JSON line.

    A subprocess per measurement, not a loop in one process, for two reasons:
    `ru_maxrss` is a high-water mark that never comes back down, and
    `to_gpu_anndata` leaves a CUDA context and a device-resident matrix behind
    that would distort whatever ran next.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", op, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "op": op,
            "error": (proc.stderr or proc.stdout).strip()[-2000:],
        }
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not line:
        return {"op": op, "error": f"no record on stdout: {proc.stdout[-2000:]}"}
    return json.loads(line[-1])


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        op = sys.argv[2]
        rec = _MEASUREMENTS[op]()
        rec["op"] = op
        rec["dataset"] = sys.argv[3]
        if len(sys.argv) >= 5:
            rec["variant"] = sys.argv[4]
        print(json.dumps(rec))
        return 0

    print("=== PR-12 GPU freebies A/B profile ===", flush=True)
    print(f"datasets: {', '.join(DATASETS)}   small: {SMALL}   runs: {N_RUNS}", flush=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in DATASETS:
        for variant in (
            "accel_to_gpu_anndata__scx1_gpu",
            "accel_to_gpu_anndata__shufdelta_gpu",
        ):
            rows.append(_spawn("to_gpu_anndata", ds, variant))
        rows.append(_spawn("pca_backed", ds))
    rows.append(_spawn("pca_inmem", SMALL))
    for r in rows:
        print(json.dumps(r, indent=2), flush=True)
    if OUT_PATH:
        Path(OUT_PATH).write_text(json.dumps(rows, indent=2))
        print(f"wrote {OUT_PATH}", flush=True)
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
