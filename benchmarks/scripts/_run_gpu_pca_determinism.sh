#!/bin/bash
# Is GPU PCA run-to-run deterministic on a backed source?
#
# PR-12's A/B measured the two arms' PCA embeddings agreeing to min|cos| =
# 0.99999999996 (tabula) / 0.9999999996 (census_500k) but **not** bit-for-bit.
# The column-means pass this PR changed is bit-identical by construction and by
# an existing test, so the residual has to come from somewhere else in
# `randomized_pca_core` — unless it is simply that the GPU path does not
# reproduce itself. Those two readings are opposite: one is a fidelity
# regression to chase, the other is nothing at all.
#
# This settles it by running the same build against itself, which is the scale
# the cross-arm number should have been read against in the first place.
#
# It also **repairs the shared editable install**: job 2938363's `maturin
# develop` repointed `scx-bench-gpu`'s `pyscx.pth` at a worktree it then
# deleted, leaving `import pyscx` succeeding as an empty namespace package.
# Building from ${SCX_DIR} here restores it to the repo.
#
#SBATCH --job-name=scx-pr12-det
#SBATCH --partition=gpu_high_mem,ctc_gpu_priority,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr12/determinism_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr12/determinism_%j.out

set -uo pipefail

# Exit status tracks the MEASUREMENT, not the last echo. Every one of these
# scripts used to end on a successful `echo`, so a crashed profiler or a
# half-finished capture produced a SLURM job reporting COMPLETED 0:0 — job
# 2938439 "COMPLETED" in 15 s having measured nothing. GitHub CI runs none of
# the GPU tests, so a green SLURM job is the only signal here, and a green one
# that measured nothing is worse than a red one.
STATUS=0
fail() { echo "!! $*" >&2; STATUS=1; }
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-pr12
OUT="${WORK}/determinism_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2; exit 1
fi

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo ""
echo "=== rebuilding pyscx from ${SCX_DIR} into ${ENV} (this is also the repair) ==="
echo "  before: $(cat "${ENV}"/lib/python*/site-packages/pyscx.pth 2>/dev/null)"
( cd "${SCX_DIR}/pyscx" \
  && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-pr12-det \
     "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -3
echo "  after : $(cat "${ENV}"/lib/python*/site-packages/pyscx.pth 2>/dev/null)"
python -c "import pyscx; print('  import:', pyscx.__file__)" \
    || { echo "FATAL: pyscx still broken after rebuild"; exit 1; }

cd "${SCX_DIR}" || exit 1
set -a; . ./.env; set +a

python - "${OUT}" <<'PY'
import sys, time
from pathlib import Path

import numpy as np
import pyscx

sys.path.insert(0, "/home/nickyoungblut/dev/rust/scx")
from benchmarks.comprehensive.bench_env import DATA_DIR

out = Path(sys.argv[1])
N_COMPS, SEED = 50, 0


def cos_min(a, b):
    return min(
        abs(float(a[:, k] @ b[:, k]) / (np.linalg.norm(a[:, k]) * np.linalg.norm(b[:, k]) + 1e-300))
        for k in range(a.shape[1])
    )


for dataset in ("tabula_sapiens_100k", "census_500k"):
    p = DATA_DIR / f"{dataset}_auto.scx"
    n_shards = int(pyscx.open(str(p)).shard_count)
    embs, route = [], None
    for i in range(3):
        adata = pyscx.open(str(p)).to_anndata(backed=True)
        t0 = time.perf_counter()
        pyscx.accel.pca(adata, n_comps=N_COMPS, device="gpu", random_state=SEED)
        w = time.perf_counter() - t0
        embs.append(np.asarray(adata.obsm["X_pca"], dtype=np.float64))
        route = adata.uns.get("scx_accel", {}).get("pca", {}).get("route")
        print(f"  {dataset} run {i}: {w:.3f}s route={route}", flush=True)
        del adata
    ident01 = np.array_equal(embs[0], embs[1])
    ident02 = np.array_equal(embs[0], embs[2])
    print(f"{dataset}: shards={n_shards} route={route}")
    print(f"  run0 vs run1: bit-identical={ident01} min|cos|={cos_min(embs[0], embs[1]):.12f}")
    print(f"  run0 vs run2: bit-identical={ident02} min|cos|={cos_min(embs[0], embs[2]):.12f}")
    np.save(out / f"det__{dataset}__run0.npy", embs[0])
    print(flush=True)
PY
[ $? -eq 0 ] || fail "the determinism comparison did not complete"

echo "=== done; raw under ${OUT} ==="

if [ "${STATUS}" -ne 0 ]; then
    echo "=== FAILED: at least one step above did not complete; see !! lines ==="
fi
exit "${STATUS}"
