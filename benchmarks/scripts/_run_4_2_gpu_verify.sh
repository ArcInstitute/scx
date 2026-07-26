#!/bin/bash
# Phase-4 tasks 4.2 + 4.3 — GPU verification on an H100.
#
# Three things, in one job because each needs the same build:
#
#   1. `cargo test -p scx-gpu` and `-p scx-accel --features gpu`.
#   2. A **failure-set A/B** of the pyscx GPU suites, main vs branch. A single
#      run cannot answer "are these N failures mine"; diffing the sets can. This
#      is the harness that turned "14 GPU failures" into "14, identical on both
#      arms, zero regressions" during #372.
#   3. The `gpu_profile` host-decode / HTOD buckets with the decode-prefetch
#      pipeline off and on, which is 4.2's GPU acceptance signal.
#
#   sbatch benchmarks/scripts/_run_4_2_gpu_verify.sh
#
# Two hard-won details, both from #372 and both load-bearing:
#
#   * The two arms need **separate** `CARGO_TARGET_DIR`s. Sharing one leaves a
#     0-byte `libpyscx.so` hardlink that makes the next `maturin develop` die
#     with "Object is too small" *while the tests silently run against the
#     previous build* — a green result for the wrong binary.
#   * The conda env must be **activated**, not just invoked by absolute path, or
#     cupy's CUDA-path probe crashes on `CONDA_PREFIX=None`.
#SBATCH --job-name=scx-4.2-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.2/gpu_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.2/gpu_%j.out

set -uo pipefail

SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-4.2
OUT="${WORK}/gpu_${SLURM_JOB_ID:-manual}"
WT="${WORK}/wt-gpu-main"
BASE_SHA=2055f74f

mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV

# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu

# Full GPU suite in one process cascades via CUDA_ERROR_STREAM_CAPTURE_IMPLICIT;
# run per-file with graphs disabled.
export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8

# Present on both arms. 4.2 touches the staging path every GPU streaming op
# shares, so the sweep is deliberately wide rather than DE-only.
SHARED="test_accel_pca_gpu test_accel_fused test_accel_pipeline_gpu test_accel_route_metadata \
test_eval_metrics_gpu_parity test_csc_dispatch test_csc_dispatch_lazy test_csc_lifecycle \
test_pdex_ref_gpu_parity test_pdex_ref_gpu_csc_parity test_rank_genes_groups_gpu_parity \
test_rank_genes_groups_gpu_csc_parity test_pca_axis_view test_accel_axis_view"
# Branch-only files (they do not exist at the base commit).
BRANCH_ONLY="test_prefetch_equivalence test_marshalling_dtypes test_col_aggs_gil"

run_arm() {
    local name="$1" dir="$2" target="$3" files="$4"
    echo
    echo "############ ARM: ${name} (${dir}) ############"
    rm -rf "${target}/maturin"
    export CARGO_TARGET_DIR="${target}"
    cd "${dir}/pyscx" || return 1
    VIRTUAL_ENV="${ENV}" "${ENV}/bin/maturin" develop --release --features hdf5,gpu 2>&1 | tail -3
    ls -la "${target}/maturin/libpyscx.so" || echo "WARNING: no maturin artifact"
    python -c "import pyscx; print('so:', pyscx.__file__)"

    cd "${dir}" || return 1
    : > "${OUT}/ab_${name}_failures.txt"
    for f in ${files}; do
        [ -f "pyscx/tests/${f}.py" ] || { echo "  (skip ${f}: absent on this arm)"; continue; }
        echo "--- ${f}"
        python -m pytest "pyscx/tests/${f}.py" -q -p no:randomly 2>&1 \
            | tee "${OUT}/log_${name}_${f}.txt" | tail -3
        grep -E "^FAILED" "${OUT}/log_${name}_${f}.txt" | sed 's/ - .*//' \
            >> "${OUT}/ab_${name}_failures.txt"
    done
    sort -o "${OUT}/ab_${name}_failures.txt" "${OUT}/ab_${name}_failures.txt"
}

echo
echo "############ 1. cargo GPU suites (branch) ############"
cd "${SCX_DIR}"
CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-42-branch \
    cargo test -p scx-gpu --release -- --test-threads=1 2>&1 | tail -20
CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-42-branch \
    cargo test -p scx-accel --features gpu --release -- --test-threads=1 2>&1 | tail -20

echo
echo "############ 2. pytest failure-set A/B ############"
rm -rf "${WT}"
git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT}" "${BASE_SHA}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add failed"; exit 1; }

run_arm main   "${WT}"       /home/nickyoungblut/.cargo-target-42-main   "${SHARED}"
run_arm branch "${SCX_DIR}"  /home/nickyoungblut/.cargo-target-42-branch "${SHARED} ${BRANCH_ONLY}"

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
echo "############ 3. gpu_profile staging buckets (branch) ############"
cd "${SCX_DIR}"
set -a; . ./.env; set +a
for depth in 1 4; do
    echo
    echo "--- SCX_ACCEL_PREFETCH_DEPTH=${depth}"
    SCX_GPU_PROFILE=1 SCX_ACCEL_PREFETCH_DEPTH="${depth}" \
        python benchmarks/scripts/profile_gpu_staging.py \
        2>&1 | tee "${OUT}/gpu_staging_depth${depth}.txt"
done

echo
echo "=== done; artifacts under ${OUT} ==="
