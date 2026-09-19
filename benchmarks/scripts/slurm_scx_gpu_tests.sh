#!/bin/bash
# Run scx-gpu and scx-accel(gpu) test suites on a Chimera GPU node.
# Sanity check after touching scx-gpu/kernels/harmony.cu and the
# scx-accel GPU dispatcher.
#
# This is the only place the GPU suites actually execute: they are
# `#[ignore]`d, and the driver re-selects them with --include-ignored under
# SCX_REQUIRE_GPU=1. A green `cargo test --workspace` elsewhere says nothing
# about them by design — it now says so out loud, as `170 ignored`.
#
# Optional first argument: a SLURM dependency spec, e.g.
#   bash slurm_scx_gpu_tests.sh afterany:123456:123457
# Use it rather than submitting alongside a running scx job — a co-scheduled
# build both slows this run and biases the other job's timings.

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

mkdir -p benchmarks/logs

DEP_ARGS=()
if [[ $# -ge 1 && -n "$1" ]]; then
    DEP_ARGS=(--dependency="$1")
    echo "Queuing behind dependency: $1"
fi

# Not `preemptible`: this is a short job and the preemptible queue can starve
# for a day. And not the bare `gpu` partition either — it is the congested one
# (measured: ~11 h queued against ~1 min on the other two). Slurm takes the
# first partition in the list that can run the job.
JOB_ID=$(sbatch --parsable \
    --job-name=scx_gpu_tests \
    --partition=gpu_high_mem,ctc_gpu_priority,gpu \
    --qos=normal \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=01:30:00 \
    "${DEP_ARGS[@]}" \
    --output="benchmarks/logs/scx_gpu_tests_%j.log" \
    --error="benchmarks/logs/scx_gpu_tests_%j.err" \
    --wrap="bash ${SCX_DIR}/benchmarks/scripts/_run_scx_gpu_tests.sh")

echo "Submitted job ${JOB_ID}"
echo "  log:  ${SCX_DIR}/benchmarks/logs/scx_gpu_tests_${JOB_ID}.log"
echo "  err:  ${SCX_DIR}/benchmarks/logs/scx_gpu_tests_${JOB_ID}.err"
