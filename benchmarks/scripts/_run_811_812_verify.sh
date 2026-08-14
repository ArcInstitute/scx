#!/bin/bash
# §8.11 + §8.12 GPU verification driver (sbatch --wrap payload).
#
# Chained AFTER the scx-gpu/scx-accel cargo suite job, never concurrent with it:
# this one runs `maturin develop`, which rewrites the in-tree
# `pyscx/python/pyscx/*.so` that every venv and conda env resolves to through
# its editable install — so it would invalidate any concurrent job that merely
# *imports* pyscx, whether or not that job builds anything.
#
# Covers the two pytest surfaces the cargo suite cannot reach:
#   - test_gpu_pca_resident.py         §8.11 resident vs streaming + spmm_policy
#   - test_accel_route_metadata.py     the @gpu_only PCA tuning-metadata arms
#   - test_harmony.py                  route stamp incl. graph_replay

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Derive the repo root from this script's own location (benchmarks/scripts/<this>),
# as `_run_pdex_gpu_parity.sh` does. An earlier version computed SCRIPT_DIR and then
# ignored it in favour of a hardcoded home path, so the committed harness pointed at
# one developer's checkout from any other clone. That matters more than it looks:
# with `set -uo pipefail` and no `-e`, a failed `cd` does not stop the script — it
# carries on in whatever cwd sbatch started in, which is exactly how this driver
# once reported "feature broken" for an environment that had never activated.
SCX_DIR="${SCX_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${SCX_GPU_CONDA_ENV:-scx-bench-gpu}"
cd "${SCX_DIR}" || { echo "PREFLIGHT FAILED: cannot cd to SCX_DIR=${SCX_DIR}" >&2; exit 1; }

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2
    exit 1
fi

# Bootstrap conda BEFORE trying to run it. Under `sbatch --wrap` the shell is
# non-interactive and does not read the profile, so `conda` is not on PATH and
# `conda info --base` — the usual discovery step — is itself a
# `command not found`. Source the hook from a known prefix first, and only fall
# back to querying an already-present conda.
#
# This is not hypothetical: the first run of this script did exactly that, and
# the failure surfaced two steps later as `python: command not found` reported
# by the preflight, which reads as "the feature is broken" rather than "the env
# never activated". Worse, `maturin` (resolved from ~/.local/bin) had already
# built against the uv `.venv` instead — whose nvrtc is 13.0 and fails on H100.
for prefix in "${CONDA_PREFIX_ROOT:-}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -n "${prefix}" ] && [ -f "${prefix}/etc/profile.d/conda.sh" ]; then
        source "${prefix}/etc/profile.d/conda.sh"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "PREFLIGHT FAILED: conda not found; cannot activate ${CONDA_ENV}." >&2
    exit 1
fi
conda activate "${CONDA_ENV}" || { echo "PREFLIGHT FAILED: conda activate ${CONDA_ENV}" >&2; exit 1; }
echo "python: $(command -v python)"
# Fail loudly if we are about to build against the uv venv rather than the
# conda env — the whole reason this driver exists (nvrtc 12.6 vs 13.0).
python -c "import sys; print('py', sys.version.split()[0], sys.prefix)"
case "$(python -c 'import sys; print(sys.prefix)')" in
    *"${CONDA_ENV}"*) ;;
    *) echo "PREFLIGHT FAILED: python is not from ${CONDA_ENV}" >&2; exit 1 ;;
esac

# Isolated target dir, wiped WHOLESALE. Sharing the workspace `target/` across
# runs of this driver produces:
#
#   maturin failed: Failed to parse ELF file at 'target/maturin/libpyscx.so':
#   Malformed entity: Object is too small
#
# — because cargo finds everything fresh, relinks nothing ("Finished in 1.08s"),
# and maturin then reads whatever truncated artifact a previous run left behind.
# Clearing only `$TARGET/maturin` does not help: cargo re-links a 0-byte
# artifact from `release/` and maturin dies the same way. Wipe the whole tree.
# Costs a full rebuild per run, which is the price of a build you can trust.
export CARGO_TARGET_DIR="${SCX_DIR}/target-gpu-verify"
# Guarded, for the same reason the `cd` above is: `set -uo pipefail` has no
# `-e`, so a failed wipe (permissions, a busy NFS artifact) would otherwise fall
# straight through into `maturin develop` against the stale tree this line
# exists to remove — reintroducing the failure silently.
rm -rf "${CARGO_TARGET_DIR}" || {
    echo "PREFLIGHT FAILED: could not wipe ${CARGO_TARGET_DIR}" >&2; exit 1; }
if [ -e "${CARGO_TARGET_DIR}" ]; then
    echo "PREFLIGHT FAILED: ${CARGO_TARGET_DIR} still exists after rm -rf" >&2; exit 1
fi

echo "=== maturin develop --release --features hdf5,gpu (target: ${CARGO_TARGET_DIR}) ==="
( cd pyscx && maturin develop --release --features hdf5,gpu ) || { echo "BUILD FAILED"; exit 1; }

# Preflight the feature being measured: if the native GPU PCA route cannot run
# at all, fail here rather than after an hour of confusing test output.
#
# SCX_FORCE_NATIVE_GPU=1 is load-bearing, not decoration, and the first version
# of this script got it wrong: with an in-memory `X` and rapids-singlecell
# installed, PCA routes to `rapids_singlecell_gpu`, so a `route.startswith("gpu")`
# check fails on a perfectly healthy GPU. The subject here is the native
# resident/streaming pair, which is what the tests below pin too — so pin it,
# and accept any route naming a GPU rather than one spelling of it.
echo "=== preflight: one real native GPU PCA ==="
SCX_FORCE_NATIVE_GPU=1 python - <<'PY' || { echo "PREFLIGHT FAILED: native GPU PCA did not run"; exit 1; }
import numpy as np, scipy.sparse as sp, anndata, pyscx, sys
rng = np.random.default_rng(0)
x = rng.poisson(2.0, size=(300, 60)).astype(np.float32)
a = anndata.AnnData(X=sp.csr_matrix(x))
pyscx.accel.pca(a, n_comps=4, device="gpu", method="randomized")
info = a.uns["scx_accel"]["pca"]
print("preflight route:", info["route"],
      "resident_csr:", info.get("resident_csr"),
      "spmm_policy:", info.get("spmm_policy"))
sys.exit(0 if "gpu" in info["route"] else 1)
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
