#!/bin/bash
# Phase-4 task 4.2 — GPU staging host-decode capture, on its own.
#
# `_run_4_2_gpu_verify.sh` runs this as its phase 3, after the cargo suites and
# the failure-set A/B. Those two are already discharged (193/0, 19/0, and 11
# failures on each arm with an empty branch-only list), so re-running them to
# get the profile would waste an hour of H100 time.
#
# **Run alone.** See the header of `_run_4_2_gpu_verify.sh`: the in-tree `.so` is
# global mutable state, and a concurrent job that rebuilds it — even one that
# only *looks* like it builds elsewhere — silently swaps the binary this job
# imports. That is exactly how the first attempt lost its depth-4 arm to a
# no-GPU build.
#
# Scope is trimmed from the first attempt, which took 3 h and still only
# finished one depth. GPU DE at census_1m is ~35 min *per run*; at 3 runs × 2
# depths that single cell is 3.5 h on its own. DE therefore runs at
# tabula_sapiens_100k and census_500k, which already showed 81.7 % and 82.9 %
# host-decode at depth 1 — the signal does not need the third scale. HVG is
# cheap (7–39 s) and keeps all three.
#SBATCH --job-name=scx-4.2-gpu-staging
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.2/gpu_staging_%j.out

set -uo pipefail

SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
OUT="/home/nickyoungblut/scx-bench-4.2/gpu_staging_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV

# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu

export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8

# The in-tree `.so` cannot be assumed GPU-enabled — the previous run of this
# capture died precisely because another job had left a `--features hdf5` build
# there. Rebuild, then *prove* the feature is present before spending GPU hours.
#
# The target dir must be **removed wholesale**, not reused and not merely
# cleared of `maturin/`. A previous job leaves a 0-byte `libpyscx.so` hardlink;
# cargo then reports "Finished in 0.9s", re-links that artifact from `release/`,
# and maturin dies with "Malformed entity: Object is too small" — which is
# exactly how the first attempt at this script failed. A full rebuild costs
# ~10 min and is the only reliable option.
TARGET=/home/nickyoungblut/.cargo-target-42-staging
echo ""
echo "=== rebuilding with hdf5,gpu (clean target dir) ==="
rm -rf "${TARGET}"
( cd "${SCX_DIR}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${TARGET}" \
    "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -3
ls -la "${TARGET}/maturin/libpyscx.so" || { echo "FATAL: no maturin artifact"; exit 1; }

cd "${SCX_DIR}"
python - <<'PY' || { echo "FATAL: refusing to burn GPU hours on a non-GPU build"; exit 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
print("pyscx:", pyscx.__file__)
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
    print("preflight: gpu feature present (op raised for another reason:", e, ")")
else:
    print("preflight: gpu feature present and a GPU op ran")
PY

set -a; . ./.env; set +a
for depth in 1 4; do
    echo ""
    echo "########## SCX_ACCEL_PREFETCH_DEPTH=${depth} ##########"
    echo "--- HVG, all three scales"
    SCX_GPU_PROFILE=1 SCX_ACCEL_PREFETCH_DEPTH="${depth}" \
        GPU_STAGE_OPS=hvg GPU_STAGE_RUNS=3 \
        GPU_STAGE_DATASETS=tabula_sapiens_100k,census_500k,census_1m \
        python benchmarks/scripts/profile_gpu_staging.py \
        2>&1 | tee "${OUT}/hvg_depth${depth}.txt" | grep -vE "warn|^\s*$"
    echo "--- DE, tabula + census_500k (census_1m DE is ~35 min/run)"
    SCX_GPU_PROFILE=1 SCX_ACCEL_PREFETCH_DEPTH="${depth}" \
        GPU_STAGE_OPS=de GPU_STAGE_RUNS=3 \
        GPU_STAGE_DATASETS=tabula_sapiens_100k,census_500k \
        python benchmarks/scripts/profile_gpu_staging.py \
        2>&1 | tee "${OUT}/de_depth${depth}.txt" | grep -vE "warn|^\s*$"
done

echo ""
echo "=== done; artifacts under ${OUT} ==="
