#!/bin/bash
# PR-12 — watched red. Every test that exercises this PR is
# `#[ignore = "requires a CUDA GPU"]`, so "I watched it fail first" costs a
# SLURM job rather than a `cargo test`, and this is that job.
#
# Three mutations, each the specific wrong thing the change could have been,
# each run against the one test that is supposed to catch it. For every
# mutation the job asserts **both** halves: the unmutated test PASSES and the
# mutated one FAILS. A mutation that does not apply (a `sed` that matched
# nothing after a refactor) otherwise reads exactly like a passing baseline,
# which is how a watched-red check quietly stops checking.
#
# Runs in a detached worktree at HEAD so the live tree is never mutated, and
# uses its own CARGO_TARGET_DIR. Cargo only — no `maturin`, so this job does
# not repoint the shared in-tree `.so`. Chain it anyway: a co-scheduled build
# biases whatever else is measuring.
#
#SBATCH --job-name=scx-pr12-red
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr12/watched_red_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr12/watched_red_%j.out

set -uo pipefail
SCX_DIR=/home/nickyoungblut/dev/rust/scx
WORK=/home/nickyoungblut/scx-bench-pr12
WT="${WORK}/wt-pr12-red"
TARGET=/home/nickyoungblut/.cargo-target-pr12-red
HEAD_SHA=$(git -C "${SCX_DIR}" rev-parse HEAD)
mkdir -p "${WORK}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
echo "head: ${HEAD_SHA}"

if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2
    exit 1
fi
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export SCX_REQUIRE_GPU=1
export SCX_GPU_REQUIRE_NVCC=1
export SCX_DISABLE_CUDA_GRAPHS=1

rm -rf "${WT}"; git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT}" "${HEAD_SHA}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add failed"; exit 1; }
# Wholesale: a reused target dir can re-link a stale artifact.
rm -rf "${TARGET}"

STATUS=0
cd "${WT}" || exit 1

# $1 label  $2 crate-and-feature args (as one string)  $3 test filter
run_test() {
    local label="$1" cargoargs="$2" filter="$3"
    # shellcheck disable=SC2086
    CARGO_TARGET_DIR="${TARGET}" cargo test ${cargoargs} --release "${filter}" \
        -- --include-ignored --nocapture --test-threads=1 \
        >"${WORK}/red_${label}.log" 2>&1
    local rc=$?
    # A filter that selects nothing exits 0 and looks exactly like a pass.
    if grep -qE 'running 0 tests' "${WORK}/red_${label}.log"; then
        echo "  !! ${label}: the filter '${filter}' selected NO tests"
        return 99
    fi
    return ${rc}
}

# $1 label  $2 file  $3 sed-expression  $4 cargo args  $5 test filter
mutate_and_expect_red() {
    local label="$1" file="$2" expr="$3" cargoargs="$4" filter="$5"

    echo ""
    echo "########## ${label} ##########"
    echo "--- baseline (must PASS) ---"
    if ! run_test "${label}_base" "${cargoargs}" "${filter}"; then
        echo "  !! ${label}: the UNMUTATED test did not pass — the mutation below"
        echo "     would prove nothing. See ${WORK}/red_${label}_base.log"
        STATUS=1
        return
    fi
    echo "  ok  baseline green"

    cp "${file}" "${file}.orig"
    sed -i "${expr}" "${file}"
    if cmp -s "${file}" "${file}.orig"; then
        echo "  !! ${label}: the mutation MATCHED NOTHING — ${file} was not changed."
        echo "     A no-op mutation reads exactly like a passing test. Fix the"
        echo "     sed expression; do not treat this as a pass."
        mv "${file}.orig" "${file}"
        STATUS=1
        return
    fi
    echo "--- mutated (must FAIL) ---"
    diff -u "${file}.orig" "${file}" | sed -n '1,20p'
    run_test "${label}_mut" "${cargoargs}" "${filter}"
    local rc=$?
    mv "${file}.orig" "${file}"
    if [ ${rc} -eq 0 ]; then
        echo "  !! ${label}: the mutated build PASSED. The test does not catch this."
        STATUS=1
    elif [ ${rc} -eq 99 ]; then
        echo "  !! ${label}: filter selected nothing under mutation."
        STATUS=1
    else
        echo "  ok  ${label}: red under mutation (exit ${rc})"
    fi
}

# 1. OPT-GPU-4. Trim the WRONG window — keep the last `total` instead of the
#    first. Same length, different values, and therefore a different random
#    subspace and a different PCA. The pre-PR test asserted only `len == 21`
#    and could not see this.
mutate_and_expect_red \
    "gpu4_trim_window" \
    "scx-gpu/src/curand.rs" \
    's|let src = buf.try_slice(..total)|let src = buf.try_slice(alloc_count - total..)|' \
    "-p scx-gpu" \
    "test_random_gaussian_gpu_odd_count"

# 2. OPT-GPU-10. Fold the returned host indptr at the wrong offset. If the
#    decoder's vector were not really the shard's indptr, an off-by-one here
#    would be invisible; the framed multi-shard arm is what makes it visible.
mutate_and_expect_red \
    "gpu10_indptr_fold" \
    "scx-gpu/src/gpu_csr_assemble.rs" \
    's|for &v in &shard_indptr\[1..=rows\] {|for \&v in \&shard_indptr[0..rows] {|' \
    "-p scx-gpu" \
    "test_decode_csr_shards_to_device_matches_host_concat_framed"

# 3. OPT-GPU-1. Invert the guard: skip the pool tie in 1-vs-rest too, where it
#    IS read. This is the direction the deletion could have been wrong in, and
#    the one-vs-rest parity test is what stands between them.
mutate_and_expect_red \
    "gpu1_tie_guard" \
    "scx-accel/src/diffexp/gpu.rs" \
    's|^    if !is_ref_mode {$|    if false \&\& !is_ref_mode {|' \
    "-p scx-accel --features gpu" \
    "test_wilcoxon_gpu_dense_matches_cpu_one_vs_rest"

cd "${SCX_DIR}" || true
git -C "${SCX_DIR}" worktree remove --force "${WT}" 2>/dev/null; rm -rf "${WT}"

echo ""
echo "=== watched red: status ${STATUS} (logs under ${WORK}/red_*.log) ==="
exit ${STATUS}
