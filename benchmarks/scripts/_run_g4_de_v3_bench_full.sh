#!/bin/bash
# G4.3 bench (FULL tier): GPU DE v3 (CSC-first) — now the unconditional default
# route (the former v1/v2/v3 gate sweep collapsed to one run when the
# SCX_GPU_DE_V2/SCX_GPU_DE_V3 gates were removed). All 6 accel-gate fixtures
# now have CSC sidecars (small + full tier; census_500k / census_1m unlocked by
# the u16 fix in scx-format/writer.rs).

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
COMMON=(--tier full --accel-only --skip-preflight)

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
echo

run_one () {
    local label="$1"
    shift
    echo
    echo "================================================================="
    echo "=== G4.3 bench (FULL): ${label}"
    echo "================================================================="
    env "$@" python "${GATE}" --name "g4_de_v3_${label}_full" "${COMMON[@]}" \
        2>&1 | tee "/home/nickyoungblut/dev/rust/scx/benchmarks/scripts/g4_de_v3_${label}_full.log"
    echo "=== ${label} exit=$? ==="
}

run_one "v3_default"

echo
echo "=== done ==="
