#!/bin/bash
# GPU-SHUFDELTA-DECODE Phase 2 spike (GATE): validate nvcomp 5.1 batched GPU
# zstd decode is byte-exact vs CPU zstd + print GPU-vs-CPU throughput on
# representative per-group frames. Invoked under sbatch on a GPU node.
#
# Runs under conda `scx-bench-gpu` so `$CONDA_PREFIX/lib/libnvcomp.so.5` is
# dlopen-able; LD_LIBRARY_PATH is exported BEFORE the test process starts so
# libnvcomp's transitive deps resolve (setting it from within the process is
# too late — ld.so caches it at startup).
set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
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

echo "=== cargo test -p scx-gpu --release nvcomp (spike) ==="
cargo test -p scx-gpu --release nvcomp -- --nocapture --test-threads=1
rc=$?
echo "=== spike exit: ${rc} ==="
exit ${rc}
