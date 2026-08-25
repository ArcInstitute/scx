#!/bin/bash
# Phase 7e review fix: re-measure `accel_harmony`'s correctness floors after
# aligning the three arms.
#
# The floors in `thresholds.yaml` were calibrated from gate job 2840853, which
# ran pyscx at `random_state=42` against harmonypy at its own default of 0, with
# `epsilon_harmony` / `epsilon_kmeans` also unmatched — while the module and the
# threshold comment both said the two sides shared a seed. Found in review by
# codex and Cursor Agent. Those numbers describe a comparison that no longer
# exists, so they are re-measured rather than carried forward.
#
# Runs all three arms on both gate datasets through the benchmark module's own
# `_run_*` functions, so what is measured is what the gate will measure.

set -uo pipefail
export SCX_GPU_REQUIRE_NVCC=1
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

SCX_DIR=/home/nickyoungblut/dev/rust/scx
cd "$SCX_DIR" || exit 1
CONDA_SH=$( { conda info --base 2>/dev/null || echo /home/nickyoungblut/miniforge3; } )/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate scx-bench-gpu

echo "=== node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
touch scx-gpu/build.rs
( cd pyscx && maturin develop --release --features hdf5,gpu ) || exit 1

python - <<'PY'
import numpy as np, scanpy as sc, sys, time
sys.path.insert(0, "/home/nickyoungblut/dev/rust/scx")
import benchmarks.comprehensive.benchmarks.accel_harmony as H

DS = {
    "pbmc3k":    "/large_storage/arcinfra/projects/scx/benchmarks/datasets/pbmc3k.h5ad",
    "census_1m": "/large_storage/arcinfra/projects/scx/benchmarks/datasets/census_1m.h5ad",
}
SEED = 42
for name, path in DS.items():
    try:
        a = sc.read_h5ad(path)
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        sc.pp.pca(a, n_comps=50, random_state=SEED)
        bk = H._resolve_batch_key(a)
        print(f"\n=== {name}: n_obs={a.n_obs} batch_key={bk} "
              f"n_batches={a.obs[bk].astype(str).nunique()} K={H._n_clusters(a)}", flush=True)
        ref = a.copy(); t=time.time(); H._run_harmonypy(ref, bk, SEED)
        print(f"    harmonypy wall={time.time()-t:.1f}s", flush=True)
        ref_z = np.asarray(ref.obsm["X_pca_harmony"], np.float64)
        print(f"    harmonypy Z_corr shape={ref_z.shape}  (must be (n_obs, n_pcs))", flush=True)
        del ref
        for arm, fn in (("harmonypy_cpu", H._run_harmonypy),
                        ("pyscx_cpu", H._run_pyscx_cpu),
                        ("pyscx_gpu", H._run_pyscx_gpu)):
            try:
                t2 = a.copy(); t=time.time(); fn(t2, bk, SEED); w=time.time()-t
                got = np.asarray(t2.obsm["X_pca_harmony"], np.float64)
                route = t2.uns.get("scx_accel", {}).get("harmony_integrate", {}).get("route")
                print(f"FLOOR {name:10s} {arm:14s} r={H._mean_per_pc_r(ref_z, got)} "
                      f"wall={w:.1f}s route={route}", flush=True)
                del t2
            except Exception as e:
                print(f"FLOOR {name:10s} {arm:14s} RAISED {e!r}", flush=True)
        del a
    except Exception as e:
        print(f"=== {name}: FAILED {e!r}", flush=True)
PY
