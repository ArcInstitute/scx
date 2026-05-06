#!/bin/bash
# Run scx-gpu and scx-accel(gpu) test suites on a Chimera GPU node.
# Sanity check after touching scx-gpu/kernels/harmony.cu and the
# scx-accel GPU dispatcher.

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

mkdir -p benchmarks/logs

JOB_ID=$(sbatch --parsable \
    --job-name=scx_gpu_tests \
    --partition=preemptible \
    --qos=normal \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=00:45:00 \
    --output="benchmarks/logs/scx_gpu_tests_%j.log" \
    --error="benchmarks/logs/scx_gpu_tests_%j.err" \
    --wrap="bash ${SCX_DIR}/benchmarks/scripts/_run_scx_gpu_tests.sh")

echo "Submitted job ${JOB_ID}"
echo "  log:  ${SCX_DIR}/benchmarks/logs/scx_gpu_tests_${JOB_ID}.log"
echo "  err:  ${SCX_DIR}/benchmarks/logs/scx_gpu_tests_${JOB_ID}.err"
