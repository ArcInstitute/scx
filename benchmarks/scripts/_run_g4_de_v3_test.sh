#!/bin/bash
# Driver for the G4.3 (GPU DE v3 — CSC-first + CSR fallback) parity tests.
# Runs the scx-gpu suite + the new test_pdex_ref_gpu_v3_* tests in
# scx-accel under release.

set -euo pipefail

# GPU tests are `#[ignore]`d (see docs/testing.md § GPU Test Skip Behavior), so
# `--include-ignored` is required or this script exits 0 having run nothing.
# `--lib --tests` keeps the flag away from rustdoc, where ```ignore fences mean
# the same thing. SCX_REQUIRE_GPU=1 makes a missing device a failure, not a skip.
export SCX_REQUIRE_GPU=1

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench-gpu

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo

echo "=== cargo test -p scx-gpu --release ==="
cargo test -p scx-gpu --release --lib --tests -- --include-ignored --nocapture --test-threads=1
echo

echo "=== cargo test -p scx-accel --features gpu --release test_pdex_ref_gpu_v3 ==="
cargo test -p scx-accel --features gpu --release --lib --tests test_pdex_ref_gpu_v3 -- --include-ignored --nocapture --test-threads=1
echo

echo "=== cargo test -p scx-accel --features gpu --release pdex_ref_csc_matches_csr ==="
cargo test -p scx-accel --features gpu --release --lib --tests pdex_ref_csc_matches_csr -- --include-ignored --nocapture --test-threads=1
echo

echo "=== cargo test -p scx-accel --features gpu --release test_pdex_ref_gpu_v2 ==="
cargo test -p scx-accel --features gpu --release --lib --tests test_pdex_ref_gpu_v2 -- --include-ignored --nocapture --test-threads=1
echo

echo "=== done ==="
