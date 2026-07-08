#!/bin/bash
# Focused GPU correctness run for the ShufDeltaZstd decode path
# (GPU-SHUFDELTA-DECODE Phases 1/1.5/2). Runs the shufdelta + framed decode
# tests, isolated from the flaky rapids UMAP test that can abort the full
# harness. Invoked under sbatch.
#
# Runs under conda `scx-bench-gpu` so `$CONDA_PREFIX/lib/libnvcomp.so.5` is
# dlopen-able. A single pass covers all three decode paths: the default tests
# use the Phase-1.5 pipeline; `test_shard_decode_gpu_shufdelta_nvcomp` self-sets
# `SCX_SHUFDELTA_NVCOMP=1` for the opt-in Phase-2 nvcomp path; and the
# pipeline-vs-sequential equality test forces the Phase-1 sequential path.
set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${SCX_BENCH_GPU_ENV:-scx-bench-gpu}"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo

# `cargo test` takes a single TESTNAME substring filter, so run one invocation
# per filter (each --test-threads=1 to keep CUDA serial). Aggregate exit codes
# without `set -e` so one failure does not skip the others.
overall=0
run_filter() {
    echo "=== cargo test -p scx-gpu --release '$1' ==="
    cargo test -p scx-gpu --release "$1" -- --nocapture --test-threads=1
    local rc=$?
    echo "=== '$1' exit: ${rc} ==="
    [ "${rc}" -ne 0 ] && overall=1
}

# "shufdelta" matches the shufdelta decode tests (default pipeline path + the
# opt-in nvcomp test that self-enables SCX_SHUFDELTA_NVCOMP + the
# pipeline-vs-sequential equality). "test_nvcomp_zstd_batch" is the spike. The
# framed test covers the Scx1+ShufDeltaZstd parity loop; merge is CPU-only.
run_filter shufdelta
run_filter test_nvcomp_zstd_batch
run_filter test_shard_decode_gpu_framed
run_filter test_device_decode_stats_merge_counters

echo "=== overall exit: ${overall} ==="
exit ${overall}
