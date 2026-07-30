#!/bin/bash
# Phase-4 task 4.2 — the GPU staging number that actually matters: main vs branch.
#
# The first GPU capture compared `SCX_ACCEL_PREFETCH_DEPTH=1` against `4` on one
# branch build and the write-up read the depth-1 arm as "main". It is not.
# At depth 1 the shared pipeline declines to engage and staging decodes with
# **zero** decode threads, whereas `main` unconditionally spawned a one-ahead
# `std::thread` + `sync_channel(1)` for multi-shard input. The depth-1 arm is
# therefore *slower than main*, and depth1->depth4 overstates the real gain.
#
# This measures the real thing: two builds, both at their own default settings.
#
# **Run alone** — see `_run_4_2_gpu_verify.sh`'s header. Both arms rebuild
# `pyscx` into the conda env, so nothing else may be importing it.
#SBATCH --job-name=scx-4.2-gpu-mainab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.2/gpu_mainab_%j.out

set -uo pipefail
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-4.2
OUT="${WORK}/gpu_mainab_${SLURM_JOB_ID:-manual}"
WT="${WORK}/wt-gpu-mainab"
BASE=2055f74f
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

run_arm() {
    local name="$1" dir="$2"
    local target="/home/nickyoungblut/.cargo-target-mainab-${name}"
    echo ""
    echo "########## ARM ${name} (${dir}) ##########"
    # Wholesale, not just $TARGET/maturin: cargo re-links a 0-byte artifact
    # from release/ and maturin dies on "Object is too small".
    rm -rf "${target}"
    ( cd "${dir}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${target}" \
        "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -2
    cd "${dir}" || return 1
    python -c "import pyscx; print('so:', pyscx.__file__)"
    # Never spend GPU hours on a binary that cannot do the measurement.
    python - <<'PY' || { echo "FATAL: gpu feature missing on arm"; return 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
PY
    set -a; . ./.env; set +a
    # Default settings on both arms -- no depth override. That is the point:
    # main's staging ignores the knob, the branch's uses its default of 4.
    SCX_GPU_PROFILE=1 GPU_STAGE_OPS=hvg GPU_STAGE_RUNS=3 \
        GPU_STAGE_DATASETS=tabula_sapiens_100k,census_500k,census_1m \
        python benchmarks/scripts/profile_gpu_staging.py \
        2>&1 | tee "${OUT}/hvg_${name}.txt" | grep -vE "warn|^\s*$"
    SCX_GPU_PROFILE=1 GPU_STAGE_OPS=de GPU_STAGE_RUNS=3 \
        GPU_STAGE_DATASETS=tabula_sapiens_100k,census_500k \
        python benchmarks/scripts/profile_gpu_staging.py \
        2>&1 | tee "${OUT}/de_${name}.txt" | grep -vE "warn|^\s*$"
}

rm -rf "${WT}"; git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT}" "${BASE}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add failed"; exit 1; }
# main predates profile_gpu_staging.py; pin the harness so only the library differs.
cp "${SCX_DIR}/benchmarks/scripts/profile_gpu_staging.py" "${WT}/benchmarks/scripts/"
cp "${SCX_DIR}/.env" "${WT}/.env"

run_arm main   "${WT}"
run_arm branch "${SCX_DIR}"

git -C "${SCX_DIR}" worktree remove --force "${WT}" 2>/dev/null; rm -rf "${WT}"
echo ""
echo "=== done; artifacts under ${OUT} ==="
