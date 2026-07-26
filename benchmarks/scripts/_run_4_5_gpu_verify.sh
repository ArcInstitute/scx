#!/bin/bash
# Phase-4 task 4.5 — GPU verification on an H100 (§9.11 + §9.13).
#
# Two things, one job because both need the same build:
#
#   1. `cargo test -p scx-gpu` and `-p scx-accel --features gpu`. 4.5 changes
#      four CUDA kernels and the staging driver, so the device-side unit tests
#      are the first line — in particular `test_csr_shard_to_dense_chunk_parity`
#      and the new `test_csr_gene_major_and_pseudobulk_window_parity`, which pin
#      the windowed kernels against a host reference written the pre-windowing
#      way.
#   2. A **failure-set A/B** of the pyscx GPU suites, main vs branch. A single
#      run cannot answer "are these N failures mine"; diffing the sets can.
#
#   sbatch benchmarks/scripts/_run_4_5_gpu_verify.sh
#
# **Run this alone. No other scx job may be queued or running.** Chain anything
# else behind it:
#
#   v=$(sbatch --parsable benchmarks/scripts/_run_4_5_gpu_verify.sh)
#   sbatch --dependency=afterany:$v benchmarks/scripts/_run_4_5_gpu_main_vs_branch.sh
#
# `$SCX_DIR/pyscx/python/pyscx/*.so` is **global mutable state shared by every
# job on the cluster** — every venv and conda env resolves `pyscx` to a single
# editable install pointing at it. The rule is not "don't run two builds at
# once", it is:
#
#     any job that runs `maturin develop` in $SCX_DIR invalidates every other
#     job that will *import* pyscx, whether or not that job builds anything.
#
# Cost of getting that wrong, in 4.2: ~3 GPU-hours. Also recorded in
# CLAUDE.local.md and benchmarks/README.md.
#
# Two details from #372, both load-bearing:
#
#   * The arms need **separate** `CARGO_TARGET_DIR`s, and each must be removed
#     **wholesale** — clearing only `$TARGET/maturin` lets cargo re-link a
#     0-byte `libpyscx.so` from `release/` and maturin dies on "Object is too
#     small", while the tests silently run against the previous build.
#   * The conda env must be **activated**, not invoked by absolute path, or
#     cupy's CUDA-path probe crashes on `CONDA_PREFIX=None`.
#SBATCH --job-name=scx-4.5-gpu-verify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.5/gpu_verify_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.5/gpu_verify_%j.out

set -uo pipefail

SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-4.5
OUT="${WORK}/gpu_verify_${SLURM_JOB_ID:-manual}"
WT="${WORK}/wt-gpu-main"
BASE_SHA=2d1fe16b

mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV

# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu

# The full GPU suite in one process cascades via
# CUDA_ERROR_STREAM_CAPTURE_IMPLICIT; run per-file with graphs disabled.
export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8

# Present on both arms. 4.5 changes the CSR DE kernels *and* the staging driver
# every GPU streaming op shares, so the sweep stays wide rather than DE-only.
SHARED="test_accel_pca_gpu test_accel_fused test_accel_pipeline_gpu test_accel_route_metadata \
test_eval_metrics_gpu_parity test_csc_dispatch test_csc_dispatch_lazy test_csc_lifecycle \
test_pdex_ref_gpu_parity test_pdex_ref_gpu_csc_parity test_rank_genes_groups_gpu_parity \
test_rank_genes_groups_gpu_csc_parity test_pca_axis_view test_accel_axis_view \
test_to_gpu_anndata_e2e test_hvg_gpu_batch_key test_accel_gpu_device \
test_prefetch_equivalence"
# Branch-only files (absent at the base commit).
BRANCH_ONLY="test_gpu_de_resident"

run_arm() {
    local name="$1" dir="$2" target="$3" files="$4"
    echo
    echo "############ ARM: ${name} (${dir}) ############"
    # Wholesale, not just $TARGET/maturin — see the header.
    rm -rf "${target}"
    export CARGO_TARGET_DIR="${target}"
    cd "${dir}/pyscx" || return 1
    VIRTUAL_ENV="${ENV}" "${ENV}/bin/maturin" develop --release --features hdf5,gpu 2>&1 | tail -3
    cd "${dir}" || return 1
    python -c "import pyscx; print('so:', pyscx.__file__)"

    # Never spend GPU hours on a binary that cannot do the measurement.
    python - <<'PY' || { echo "FATAL: gpu feature missing on arm ${name}"; return 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
PY

    : > "${OUT}/ab_${name}_failures.txt"
    for f in ${files}; do
        [ -f "pyscx/tests/${f}.py" ] || { echo "  (skip ${f}: absent on this arm)"; continue; }
        echo "--- ${f}"
        python -m pytest "pyscx/tests/${f}.py" -q -p no:randomly 2>&1 \
            | tee "${OUT}/log_${name}_${f}.txt" | tail -3
        # `^ERROR` as well as `^FAILED`: a collection or fixture error never
        # produces a FAILED line, so a diff that greps only FAILED is blind to
        # a whole class of regression — the branch could error out every test
        # in a file and the diff would report nothing.
        grep -E "^(FAILED|ERROR)" "${OUT}/log_${name}_${f}.txt" | sed 's/ - .*//' \
            >> "${OUT}/ab_${name}_failures.txt"
    done
    sort -o "${OUT}/ab_${name}_failures.txt" "${OUT}/ab_${name}_failures.txt"
}

echo
echo "############ 1. cargo GPU suites (branch) ############"
cd "${SCX_DIR}" || exit 1
CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-45-branch \
    cargo test -p scx-gpu --release -- --test-threads=1 2>&1 | tail -25
CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-45-branch \
    cargo test -p scx-accel --features gpu --release -- --test-threads=1 2>&1 | tail -25

echo
echo "############ 2. pytest failure-set A/B ############"
rm -rf "${WT}"
git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT}" "${BASE_SHA}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add failed"; exit 1; }
cp "${SCX_DIR}/.env" "${WT}/.env" 2>/dev/null || true

run_arm main   "${WT}"       /home/nickyoungblut/.cargo-target-45-main   "${SHARED}"
run_arm branch "${SCX_DIR}"  /home/nickyoungblut/.cargo-target-45-branch "${SHARED} ${BRANCH_ONLY}"

echo
echo "############ DIFF ############"
echo "--- failures on main only (pre-existing, fixed on branch):"
comm -23 "${OUT}/ab_main_failures.txt" "${OUT}/ab_branch_failures.txt"
echo "--- failures on BOTH (pre-existing, unchanged):"
comm -12 "${OUT}/ab_main_failures.txt" "${OUT}/ab_branch_failures.txt"
echo "--- failures on branch only (>>> REGRESSIONS <<<):"
comm -13 "${OUT}/ab_main_failures.txt" "${OUT}/ab_branch_failures.txt"
echo
echo "main:   $(wc -l < "${OUT}/ab_main_failures.txt") failures"
echo "branch: $(wc -l < "${OUT}/ab_branch_failures.txt") failures"

git -C "${SCX_DIR}" worktree remove --force "${WT}" 2>/dev/null
rm -rf "${WT}"

echo
echo "=== done; artifacts under ${OUT} ==="
