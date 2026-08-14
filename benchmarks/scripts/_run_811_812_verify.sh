#!/bin/bash
# §8.11 + §8.12 GPU verification driver (sbatch --wrap payload).
#
# Chained AFTER the scx-gpu/scx-accel cargo suite job, never concurrent with it:
# this one runs `maturin develop`, which rewrites the in-tree .so that every
# venv and conda env resolves to, and would invalidate any concurrent job that
# imports pyscx (CLAUDE.local.md).
#
# Covers the two pytest surfaces the cargo suite cannot reach:
#   - test_gpu_pca_resident.py         §8.11 resident vs streaming + spmm_policy
#   - test_accel_route_metadata.py     the @gpu_only PCA tuning-metadata arms
#   - test_harmony.py                  route stamp incl. graph_replay

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCX_DIR="${SCX_DIR:-/home/nickyoungblut/dev/rust/scx}"
CONDA_ENV="${SCX_GPU_CONDA_ENV:-scx-bench-gpu}"
cd "${SCX_DIR}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || dirname "$(dirname "${CONDA_EXE:-}")")"
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"
echo "python: $(which python)"

echo "=== maturin develop --release --features hdf5,gpu ==="
( cd pyscx && maturin develop --release --features hdf5,gpu ) || { echo "BUILD FAILED"; exit 1; }

# Preflight the feature being measured: if the GPU PCA route cannot run at all,
# fail here rather than after an hour of confusing test output.
echo "=== preflight: one real GPU PCA ==="
python - <<'PY' || { echo "PREFLIGHT FAILED: GPU PCA did not run"; exit 1; }
import numpy as np, scipy.sparse as sp, anndata, pyscx, sys
rng = np.random.default_rng(0)
x = rng.poisson(2.0, size=(300, 60)).astype(np.float32)
a = anndata.AnnData(X=sp.csr_matrix(x))
pyscx.accel.pca(a, n_comps=4, device="gpu", method="randomized")
info = a.uns["scx_accel"]["pca"]
print("preflight route:", info["route"], "resident_csr:", info.get("resident_csr"))
sys.exit(0 if info["route"].startswith("gpu") else 1)
PY

rc=0
# Isolated processes: CUDA graph-capture state cascades across tests in one
# process (see reference notes on GPU pytest isolation).
for f in tests/test_gpu_pca_resident.py \
         tests/test_accel_route_metadata.py \
         tests/test_harmony.py; do
    echo "=== pytest (isolated): ${f} ==="
    ( cd pyscx && python -m pytest "${f}" -v 2>&1 ) || rc=1
done

echo "=== EXIT ${rc} ==="
exit ${rc}
