#!/bin/bash
# Driver invoked by slurm_scx_gpu_tests.sh under sbatch --wrap.
# Kept separate so the wrap payload can be `bash <driver>` and we
# can use bash-isms (pipefail, arrays).

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
echo

echo "=== cargo test -p scx-gpu --release ==="
cargo test -p scx-gpu --release -- --nocapture --test-threads=1
echo

echo "=== cargo test -p scx-accel --features gpu --release ==="
cargo test -p scx-accel --features gpu --release -- --nocapture --test-threads=1
echo

echo "=== done ==="
