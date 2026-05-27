#!/bin/bash
# Smoke test that the bench fixture actually exercises the v3-CSC
# dispatch path on a single small dataset (pbmc3k). Runs the
# accel-only gate restricted to pbmc3k with:
#   - SCX_GPU_DE_V3=1        — turn on v3 dispatch
#   - SCX_GPU_DE_V3_TRACE=1  — emit one stderr line per call indicating
#                              csc-direct vs csr-direct
#   - SCX_BENCH_REBUILD_CSC=1 — force a fresh build of the temp CSC
#                              SCX fixture (cached otherwise)
#
# Success: SLURM stderr contains
#   `[scx-accel/pdex_ref] v3 dispatch route: csc-direct (CSC sidecar present)`
# Failure: contains `csr-direct (no CSC sidecar)` instead.

set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh

# Rebuild pyscx so the worker sees the freshly-compiled scx-accel with
# the v3 dispatch traces + the code-review fixups. Editable .so is
# shared across envs; build under scx-bench-gpu for clarity.
conda activate scx-bench-gpu
echo "=== rebuild pyscx --features gpu in scx-bench-gpu ==="
cd "${SCX_DIR}/pyscx"
"${SCX_DIR}/.venv/bin/maturin" develop --release --features gpu --quiet
cd "${SCX_DIR}"
python -c "
import pyscx
info = pyscx.accel.gpu_info()
assert info, 'pyscx missing GPU features'
print('pyscx gpu_info OK:', info['device'])
"

# Orchestrator runs from scx-bench per workspace convention.
conda deactivate
conda activate scx-bench
set -a
source .env
set +a

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo
echo "=== run accel-only gate on pbmc3k with SCX_GPU_DE_V3=1 + trace ==="
SCX_GPU_DE_V3=1 \
SCX_GPU_DE_V3_TRACE=1 \
SCX_BENCH_REBUILD_CSC=1 \
python "${SCX_DIR}/benchmarks/comprehensive/scripts/gate_candidate.py" \
    --name "g4_de_v3_csc_smoke_pbmc3k" \
    --tier small \
    --accel-only \
    --skip-preflight \
    --datasets pbmc3k \
    --benchmarks accel_de \
    2>&1 | tee "${SCX_DIR}/benchmarks/scripts/g4_de_v3_csc_smoke_pbmc3k.log"

echo
echo "=== grepping SLURM stderr for dispatch route ==="
# accel_de GPU jobs route through scx-bench-gpu env. Look in their
# submitit stderr for the trace lines.
SUBMITIT_DIR="${SCX_DIR}/benchmarks/comprehensive/logs/submitit"
echo "--- recent submitit logs in ${SUBMITIT_DIR}/bench ---"
ls -t "${SUBMITIT_DIR}/bench"/*_log.err 2>/dev/null | head -8

echo
echo "--- dispatch-route lines across recent stderr (last 1h) ---"
find "${SUBMITIT_DIR}/bench" -name "*_log.err" -mmin -60 -print0 2>/dev/null \
    | xargs -0 grep -h "v3 dispatch route" 2>/dev/null \
    | sort | uniq -c

echo
echo "=== done ==="
