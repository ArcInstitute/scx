#!/usr/bin/env python
"""Micro-bench for Phase-2 task 2.4 (flat NumPy result marshalling).

Captures the ``marshalling`` CPU-profile bucket alongside total ``pca()`` wall
and peak RSS on a synthetic sparse matrix. The 2.4 change (flat ``Vec<f32>`` +
``PyArray1::from_vec``/``reshape`` instead of ``Vec<Vec<f32>>`` +
``PyArray2::from_vec2``) is correctness-neutral (byte-identical output); this
bench documents that the marshalling bucket is a negligible fraction of the PCA
wall — consistent with the 2.0 profiling oracle — so the win is allocation
count, not wall time.

Run with the profiler enabled::

    SCX_CPU_PROFILE=1 python benchmarks/scripts/bench_marshalling.py

Optional env overrides: MARSHAL_N (rows), MARSHAL_G (genes), MARSHAL_NC
(components), MARSHAL_DENSITY.
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


def main() -> None:
    print(f"building synthetic {N}x{G} CSR (density={DENSITY}) ...", flush=True)
    X = sp.random(N, G, density=DENSITY, format="csr", dtype=np.float32, random_state=0)
    X.data = np.abs(X.data) * 10.0  # positive, log1p-ish
    adata = ad.AnnData(X=X)

    pyscx.accel.cpu_profile_reset()
    t0 = time.perf_counter()
    pyscx.accel.pca(adata, n_comps=NC, device="cpu")
    wall = time.perf_counter() - t0
    snap = pyscx.accel.cpu_profile_snapshot()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(f"n_obs={N} n_vars={G} n_comps={NC}")
    print(f"pca() total wall: {wall * 1000:.1f} ms")
    print(f"peak RSS: {peak_mb:.0f} MB")
    print(f"X_pca: {adata.obsm['X_pca'].shape} {adata.obsm['X_pca'].dtype}")
    print(f"PCs: {adata.varm['PCs'].shape}")
    marshal = snap.get("marshalling", {})
    if not snap.get("enabled"):
        print("NOTE: set SCX_CPU_PROFILE=1 to populate the marshalling bucket")
    print(f"marshalling bucket: {marshal}")


if __name__ == "__main__":
    main()
