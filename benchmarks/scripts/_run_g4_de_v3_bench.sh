#!/bin/bash
# G4.3 bench: GPU DE v3 (CSC-first / CSR-fallback) — now the unconditional
# default route (the former v1/v2/v3 gate sweep collapsed to one run when the
# SCX_GPU_DE_V2/SCX_GPU_DE_V3 gates were removed). Submitted as a single
# orchestrator SLURM job that runs sequentially — the orchestrator itself
# submits per-bench-row child SLURM jobs.
#
# Per-fixture CSC status (built by `_add_csc_to_fixtures.sh`):
#   pbmc3k_auto.scx       → has CSC (v3 exercises CSC-direct)
#   pbmc10k_auto.scx      → has CSC
#   smartseq2_auto.scx    → has CSC
#   tabula_sapiens_100k   → SKIPPED (pre-existing codec bug)
#   census_500k_auto.scx  → NO CSC (build-csc u16 limit; v3 falls back to CSR-direct)
#   census_1m_auto.scx    → NO CSC (same)
#
# Small tier covers the 3 CSC-equipped fixtures; we restrict to those to
# isolate the v3-CSC perf signal.

set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench
set -a
source .env
set +a

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

GATE="benchmarks/comprehensive/scripts/gate_candidate.py"
COMMON=(--tier small --accel-only --skip-preflight
        --datasets pbmc3k pbmc10k smartseq2)

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
echo

run_one () {
    local label="$1"
    shift
    echo
    echo "================================================================="
    echo "=== G4.3 bench: ${label}"
    echo "================================================================="
    env "$@" python "${GATE}" --name "g4_de_v3_${label}_small" "${COMMON[@]}" \
        2>&1 | tee "/home/nickyoungblut/dev/rust/scx/benchmarks/scripts/g4_de_v3_${label}_small.log"
    echo "=== ${label} exit=$? ==="
}

run_one "v3_default"

echo
echo "=== done ==="
