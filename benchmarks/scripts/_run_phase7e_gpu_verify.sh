#!/bin/bash
# Phase 7e GPU verification — the Harmony M-step on the device arm.
#
#   1. scx-gpu `gpu_harmony` kernel smoke.
#   2. scx-accel harmonypy reference arms, including `gpu_m_step_matches_harmonypy`.
#   3. The three graph-capture contracts, which are what a badly-placed M-step
#      would break: `test_gpu_harmony_captures_once_across_outer_iters`,
#      `test_gpu_harmony_graph_vs_direct_parity`, `test_gpu_harmony_reports_graph_replay`.
#   4. `test_gpu_vs_cpu_per_pc_correlation` — the arm that reds if the M-step
#      lands on one arm only.
#
# NOT `set -e`: one failing step must not abort the rest, mirroring the GPU
# test harness's loose-failure stance.

set -uo pipefail

# GPU tests are `#[ignore]`d (docs/testing.md § GPU Test Skip Behavior), so
# `--include-ignored` is required or this script exits 0 having run nothing.
# `--lib --tests` keeps the flag away from rustdoc, where ```ignore fences mean
# the same thing. SCX_REQUIRE_GPU=1 makes a missing device a failure, not a skip.
export SCX_REQUIRE_GPU=1

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}" || exit 1
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Phase 7e addition. The Phase 7 gate (job 2839940) died at
# CUDA_ERROR_INVALID_IMAGE because cargo replayed a cached *stub* PTX from a
# target directory that had once seen a CPU-only build — nvcc's presence is not
# part of the build script's fingerprint. This makes that a build failure in
# 20 seconds instead of a runtime failure four steps away.
export SCX_GPU_REQUIRE_NVCC=1

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
command -v nvcc && nvcc --version | tail -1
echo

rc=0

# The guard above only fires when the build script actually RERUNS. Touch it so
# a cached stub cannot survive into this job.
touch scx-gpu/build.rs

echo "=== [0] PTX must not be a stub ==="
cargo build -p scx-gpu --release 2>&1 | tail -3
stub=0
while IFS= read -r f; do
    sz=$(wc -c < "$f")
    if [ "$sz" -lt 1000 ]; then
        echo "STUB PTX: $f ($sz bytes)"
        stub=1
    fi
done < <(find target/release/build -name 'harmony.ptx' -newer scx-gpu/build.rs 2>/dev/null)
if [ "$stub" -ne 0 ]; then
    echo "FAILED: empty PTX stubs present; every GPU arm below would be meaningless."
    exit 1
fi
echo "ok"
echo

echo "=== [1] scx-gpu gpu_harmony kernels (release) ==="
cargo test -p scx-gpu --release --lib --tests gpu_harmony \
    -- --include-ignored --nocapture --test-threads=1 || rc=1
echo

echo "=== [2] scx-accel harmonypy reference arms, CPU + GPU (release) ==="
cargo test -p scx-accel --features gpu --release --lib --tests harmony_reference \
    -- --include-ignored --nocapture --test-threads=1 || rc=1
echo

echo "=== [3] graph-capture contracts + CPU/GPU parity (release) ==="
cargo test -p scx-accel --features gpu --release --lib --tests harmony::cpu::tests \
    -- --include-ignored --nocapture --test-threads=1 || rc=1
echo

echo "=== [4] LISI reference arms (CPU only, but they ride this build) ==="
cargo test -p scx-accel --features gpu --release --lib --tests lisi_reference \
    -- --include-ignored --nocapture --test-threads=1 || rc=1
echo

echo "=== rc=${rc} ==="
exit "${rc}"
