#!/bin/bash
# Driver invoked under sbatch --wrap. Validates the pdex_ref GPU cpm_filter +
# numeric (NaN->0) changes on an H100: builds pyscx with the gpu feature under
# conda scx-bench-gpu (nvrtc 12.6 — .venv has nvrtc 13.0 which fails on H100),
# runs the GPU parity test files in isolated processes (CUDA graph-capture
# cross-test cascade), and runs the scx-accel GPU pdex cargo tests to exercise
# the refactored GPU dispatch at runtime.

set -uo pipefail

# Resolve the repo root from this script's location (benchmarks/scripts/<this>)
# so the driver is portable rather than tied to one checkout path. Override with
# SCX_DIR / SCX_GPU_CONDA_ENV in the environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCX_DIR="${SCX_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${SCX_GPU_CONDA_ENV:-scx-bench-gpu}"
cd "${SCX_DIR}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Locate conda from CONDA_EXE / common prefixes rather than a hardcoded path.
CONDA_BASE="$(conda info --base 2>/dev/null || dirname "$(dirname "${CONDA_EXE:-}")")"
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"
echo "python: $(which python)"
python -c "import sys; print('py', sys.version.split()[0])"

rc=0

echo "=== maturin develop pyscx --features hdf5,gpu (scx-bench-gpu) ==="
( cd pyscx && maturin develop --release --features hdf5,gpu ) || { echo "BUILD FAILED"; exit 1; }

python -c "import pdex, inspect; print('pdex sig', inspect.signature(pdex.pdex))" || true

export SCX_DISABLE_CUDA_GRAPHS=1

for f in tests/test_pdex_ref_gpu_parity.py tests/test_pdex_ref_gpu_csc_parity.py; do
    echo "=== pytest (isolated): ${f} ==="
    ( cd pyscx && python -m pytest "${f}" -v 2>&1 ) || rc=1
done

echo "=== cargo test -p scx-accel --features gpu --release (pdex + csc) ==="
cargo test -p scx-accel --features gpu --release pdex -- --nocapture --test-threads=1 || rc=1

echo "=== done (rc=${rc}) ==="
exit ${rc}
