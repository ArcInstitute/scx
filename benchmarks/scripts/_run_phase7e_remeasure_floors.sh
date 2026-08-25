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
# `_load_preprocessed` + `_ensure_pca` + `_run_*`, so the embedding the arms see
# is the embedding the gate builds.
#
# The first version claimed it measured "what the gate measures" because it
# called `H._run_*`. It did not: those take an AnnData that already has `X_pca`,
# and this script built that PCA on ALL genes while the gate's
# `AcceleratorRunner.load_preprocessed` subsets to 2000 seurat_v3 HVGs first.
# A 50-PC embedding of all genes is not the 50-PC embedding of 2k HVGs, so the
# numbers did not reproduce `mean_per_pc_r_vs_harmonypy`. Found in review by
# Cursor Agent, round 2.
#
# It also exited 0 when a dataset or an arm failed -- every exception was caught
# and printed, and the Python program never set a non-zero status. Since this
# job is the evidence a floor is written from, a silent partial run is the one
# outcome it must not produce. Found in review by codex, round 2.

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

python - <<'PYBODY'
import numpy as np, sys, time
sys.path.insert(0, "/home/nickyoungblut/dev/rust/scx")
import benchmarks.comprehensive.benchmarks.accel_harmony as H
from benchmarks.comprehensive.benchmarks.accel_knn import _ensure_pca
from benchmarks.comprehensive.config import DATASETS

SEED = 42
N_COMPS = 50
WANT = [(ds, arm) for ds in ("pbmc3k", "census_1m")
        for arm in ("harmonypy_cpu", "pyscx_cpu", "pyscx_gpu")]
seen, failures = set(), []
# `config.DATASETS` is a dict keyed by name, not a list of DatasetConfig.
# The first version iterated it and read `.name` off the keys, which is
# how this job failed with `'str' object has no attribute 'name'` — the
# fail-loud exit added this round is what surfaced it as rc=1 rather than
# as a green job with no floors.
by_name = dict(DATASETS)

for ds in ("pbmc3k", "census_1m"):
    try:
        cfg = by_name.get(ds)
        if cfg is None:
            failures.append(f"{ds}: not in config.DATASETS"); continue
        # The gate's own path: normalize -> log1p -> 2k seurat_v3 HVG subset,
        # then PCA. Anything else measures a different embedding.
        fx = H._load_preprocessed(cfg, n_comps=N_COMPS)
        a = fx.adata.copy()
        _ensure_pca(a, N_COMPS)
        bk = H._resolve_batch_key(a)
        print(f"\n=== {ds}: n_obs={a.n_obs} n_vars={a.n_vars} batch_key={bk} "
              f"n_batches={a.obs[bk].astype(str).nunique()} K={H._n_clusters(a)}", flush=True)
        ref = a.copy(); t = time.time(); H._run_harmonypy(ref, bk, SEED)
        ref_z = np.asarray(ref.obsm["X_pca_harmony"], np.float64)
        print(f"    harmonypy wall={time.time()-t:.1f}s shape={ref_z.shape}", flush=True)
        if ref_z.shape != (a.n_obs, N_COMPS):
            failures.append(f"{ds}: harmonypy reference shape {ref_z.shape}")
        del ref
        for arm, fn in (("harmonypy_cpu", H._run_harmonypy),
                        ("pyscx_cpu", H._run_pyscx_cpu),
                        ("pyscx_gpu", H._run_pyscx_gpu)):
            try:
                w = a.copy(); t = time.time(); fn(w, bk, SEED); wall = time.time()-t
                got = np.asarray(w.obsm["X_pca_harmony"], np.float64)
                r = H._mean_per_pc_r(ref_z, got)
                route = w.uns.get("scx_accel", {}).get("harmony_integrate", {}).get("route")
                if r is None or not np.isfinite(r):
                    failures.append(f"{ds}/{arm}: non-finite r ({r})")
                else:
                    seen.add((ds, arm))
                print(f"FLOOR {ds:10s} {arm:14s} r={r} wall={wall:.1f}s route={route}", flush=True)
                del w
            except Exception as e:
                failures.append(f"{ds}/{arm} raised: {e!r}")
                print(f"FLOOR {ds:10s} {arm:14s} RAISED {e!r}", flush=True)
        del a
    except Exception as e:
        failures.append(f"{ds} raised: {e!r}")
        print(f"=== {ds}: FAILED {e!r}", flush=True)

missing = [f"{d}/{a}" for d, a in WANT if (d, a) not in seen]
if missing:
    failures.append("no usable r for: " + ", ".join(missing))
if failures:
    print("\nREMEASURE FAILED:", flush=True)
    for f in failures:
        print(f"  - {f}", flush=True)
    sys.exit(1)
print(f"\nREMEASURE OK -- {len(seen)} of {len(WANT)} arms produced a finite r", flush=True)
PYBODY
rc=$?
echo "=== remeasure rc=${rc}"
exit "${rc}"
