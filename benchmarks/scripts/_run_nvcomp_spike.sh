#!/bin/bash
# GPU ShufDeltaZstd decode — Phase 2 nvcomp spike (GATE): validate nvcomp 5.1 batched GPU
# zstd decode is byte-exact vs CPU zstd + print GPU-vs-CPU throughput on
# representative per-group frames. Invoked under sbatch on a GPU node.
#
# Runs under conda `scx-bench-gpu` so `$CONDA_PREFIX/lib/libnvcomp.so.5` is
# dlopen-able; LD_LIBRARY_PATH is exported BEFORE the test process starts so
# libnvcomp's transitive deps resolve (setting it from within the process is
# too late — ld.so caches it at startup).
set -uo pipefail

# GPU tests are `#[ignore]`d (see docs/testing.md § GPU Test Skip Behavior), so
# `--include-ignored` is required or this script exits 0 having run nothing.
# `--lib --tests` keeps the flag away from rustdoc, where ```ignore fences mean
# the same thing. SCX_REQUIRE_GPU=1 makes a missing device a failure, not a skip.
export SCX_REQUIRE_GPU=1

SCX_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${SCX_DIR}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv || true

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${SCX_BENCH_GPU_ENV:-scx-bench-gpu}"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== libnvcomp present? ==="
ls -l "${CONDA_PREFIX}/lib/"libnvcomp.so* 2>/dev/null || echo "WARNING: libnvcomp not in CONDA_PREFIX/lib"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo

# This job exists to measure nvcomp. Without this, a node missing
# libnvcomp.so.5 makes every nvcomp test a listed skip and the spike
# reports success having measured nothing.
export SCX_REQUIRE_NVCOMP=1
echo "=== cargo test -p scx-gpu --release nvcomp (spike) ==="
cargo test -p scx-gpu --release --lib --tests nvcomp -- --include-ignored --nocapture --test-threads=1
rc=$?
echo "=== spike exit: ${rc} ==="
exit ${rc}
