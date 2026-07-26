#!/usr/bin/env python
"""Micro-bench for the flat NumPy result marshalling (Phase-2 task 2.4, Phase-4 4.3).

Captures the ``marshalling`` CPU-profile bucket alongside total ``pca()`` wall
and peak RSS on a synthetic sparse matrix. The 2.4 change (flat ``Vec<f32>`` +
``PyArray1::from_vec``/``reshape`` instead of ``Vec<Vec<f32>>`` +
``PyArray2::from_vec2``) is correctness-neutral (byte-identical output); this
bench documents that the marshalling bucket is a negligible fraction of the PCA
wall — consistent with the 2.0 profiling oracle — so the win is allocation
count, not wall time.

Run with the profiler enabled::

    SCX_CPU_PROFILE=1 python benchmarks/scripts/bench_marshalling.py

Phase 4.3 extended the same treatment to the 1-D writers, so this bench grew two
more ops: ``neighbors`` (six CSR arrays per call -- the largest remaining
marshalling cost, finding 9.5) and ``de`` (``rank_genes_groups``'s structured
arrays, one transfer per group per field, finding 9.6). Both are output-neutral;
the bench exists to size the bucket, not to claim a speedup.

Optional env overrides: MARSHAL_N (rows), MARSHAL_G (genes), MARSHAL_NC
(components), MARSHAL_DENSITY, MARSHAL_K (kNN neighbors), MARSHAL_OPS
(comma-separated subset of pca,neighbors,de).
"""

import os
import resource
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp

import pyscx

N = int(os.environ.get("MARSHAL_N", 300_000))
G = int(os.environ.get("MARSHAL_G", 2_000))
NC = int(os.environ.get("MARSHAL_NC", 50))
DENSITY = float(os.environ.get("MARSHAL_DENSITY", 0.05))
K = int(os.environ.get("MARSHAL_K", 15))
OPS = tuple(
    o.strip() for o in os.environ.get("MARSHAL_OPS", "pca,neighbors,de").split(",") if o.strip()
)


def _report(label: str, wall: float, snap: dict) -> None:
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    marshal = snap.get("marshalling", {})
    ms = float(marshal.get("ms", 0.0))
    share = (ms / (wall * 1000.0) * 100.0) if wall > 0 else 0.0
    print(f"  {label:<10} wall {wall * 1000:9.1f} ms   peak RSS {peak_mb:7.0f} MB")
    print(f"  {'':<10} marshalling {ms:8.1f} ms ({share:.2f}% of wall)  {marshal}")


def main() -> None:
    print(f"building synthetic {N}x{G} CSR (density={DENSITY}) ...", flush=True)
    X = sp.random(N, G, density=DENSITY, format="csr", dtype=np.float32, random_state=0)
    X.data = np.abs(X.data) * 10.0  # positive, log1p-ish
    adata = ad.AnnData(X=X)
    # Two groups so rank_genes_groups has something to contrast.
    adata.obs["grp"] = ["a" if (i // 2) % 2 == 0 else "b" for i in range(N)]
    adata.var_names = [f"g{i}" for i in range(G)]

    snap = pyscx.accel.cpu_profile_snapshot()
    if not snap.get("enabled"):
        print("NOTE: set SCX_CPU_PROFILE=1 to populate the marshalling bucket")
    print(f"n_obs={N} n_vars={G} n_comps={NC} k={K}")

    if "pca" in OPS:
        pyscx.accel.cpu_profile_reset()
        t0 = time.perf_counter()
        pyscx.accel.pca(adata, n_comps=NC, device="cpu")
        wall = time.perf_counter() - t0
        _report("pca", wall, pyscx.accel.cpu_profile_snapshot())
        print(f"  {'':<10} X_pca {adata.obsm['X_pca'].shape} {adata.obsm['X_pca'].dtype}"
              f"  PCs {adata.varm['PCs'].shape}")

    if "neighbors" in OPS:
        if "X_pca" not in adata.obsm:
            pyscx.accel.pca(adata, n_comps=NC, device="cpu")
        pyscx.accel.cpu_profile_reset()
        t0 = time.perf_counter()
        pyscx.accel.neighbors(adata, n_neighbors=K, device="cpu")
        wall = time.perf_counter() - t0
        _report("neighbors", wall, pyscx.accel.cpu_profile_snapshot())
        d, c = adata.obsp["distances"], adata.obsp["connectivities"]
        print(f"  {'':<10} distances nnz={d.nnz} {d.dtype}/{d.indices.dtype}"
              f"  connectivities nnz={c.nnz}")

    if "de" in OPS:
        pyscx.accel.cpu_profile_reset()
        t0 = time.perf_counter()
        pyscx.accel.rank_genes_groups(adata, "grp", device="cpu")
        wall = time.perf_counter() - t0
        _report("de", wall, pyscx.accel.cpu_profile_snapshot())
        rgg = adata.uns["rank_genes_groups"]
        print(f"  {'':<10} groups={len(rgg['names'].dtype.names)} genes={len(rgg['names'])}")


if __name__ == "__main__":
    main()
