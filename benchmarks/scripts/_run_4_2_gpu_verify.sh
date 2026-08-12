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
# **Run this alone. No other scx job may be queued or running.** Chain anything
# else behind it:
#
#   gpu=$(sbatch --parsable benchmarks/scripts/_run_4_2_gpu_verify.sh)
#   sbatch --dependency=afterany:$gpu <the next one>
#
# `$SCX_DIR/pyscx/python/pyscx/*.so` is **global mutable state shared by every
# job on the cluster**, because every venv and conda env resolves `pyscx` to a
# single editable install pointing at it. The rule is not "don't run two builds
# at once" — it is:
#
#     any job that runs `maturin develop` in $SCX_DIR invalidates every other
#     job that will *import* pyscx, whether or not that job builds anything.
#
# Learned three times, the last one expensively: a marshalling A/B was submitted
# alongside this job on the reasoning that it "targets $VENV and writes into the
# tree it builds". Both clauses were true; the conclusion was not, because the
# tree it built *was* $SCX_DIR. It wrote a `--features hdf5` (no-GPU) `.so`, and
# this job's second staging arm — which had not started yet — failed every op
# with "pyscx was built without the 'gpu' feature", losing ~3 GPU-hours.
#
# Slurm will also co-schedule jobs on one node (two landed together on GPU71BA),
# where cargo builds steal CPU from a *timing* capture unevenly across its arms,
# biasing the result rather than merely adding noise.
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

# GPU tests are `#[ignore]`d (see docs/testing.md § GPU Test Skip Behavior), so
# `--include-ignored` is required or this script exits 0 having run nothing.
# `--lib --tests` keeps the flag away from rustdoc, where ```ignore fences mean
# the same thing. SCX_REQUIRE_GPU=1 makes a missing device a failure, not a skip.
export SCX_REQUIRE_GPU=1
VERIFY_STATUS=0

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
# The pipe to `tail` discards cargo's status and there is no `set -e`, so
# capture PIPESTATUS: after the #[ignore] sweep these two invocations select
# and run all 204 GPU tests, and a silently-discarded verdict here is the same
# bug this PR removes, at the job level.
CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-42-branch \
    cargo test -p scx-gpu --release --lib --tests -- --include-ignored --test-threads=1 2>&1 | tail -20
CARGO_GPU_RC=${PIPESTATUS[0]}
CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-42-branch \
    cargo test -p scx-accel --features gpu --release --lib --tests -- --include-ignored --test-threads=1 2>&1 | tail -20
CARGO_ACCEL_RC=${PIPESTATUS[0]}
if [[ ${CARGO_GPU_RC} -ne 0 || ${CARGO_ACCEL_RC} -ne 0 ]]; then
    echo "!! cargo GPU suites FAILED (scx-gpu=${CARGO_GPU_RC} scx-accel=${CARGO_ACCEL_RC})"
    VERIFY_STATUS=1
fi

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
echo "=== done (status ${VERIFY_STATUS}); artifacts under ${OUT} ==="
exit ${VERIFY_STATUS}
