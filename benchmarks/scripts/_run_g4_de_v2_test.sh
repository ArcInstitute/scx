#!/bin/bash
# Driver for the G4.1 (GPU DE v2) parity test. Runs the scx-gpu test
# suite + the targeted pdex_ref_gpu tests in scx-accel under release.

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

echo "=== cargo test -p scx-accel --features gpu --release test_pdex_ref_gpu ==="
cargo test -p scx-accel --features gpu --release --lib --tests test_pdex_ref_gpu -- --include-ignored --nocapture --test-threads=1
echo

echo "=== done ==="
