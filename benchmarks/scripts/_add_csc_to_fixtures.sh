#!/bin/bash
# Add CSC sidecars to existing SCX bench fixtures via `scx build-csc`.
# Used to prepare fixtures for the G4.3 bench: GPU DE exercises the v3
# CSC-direct path when a CSC sidecar is present; without CSC it falls back
# to CSR-direct atomicAdd.
#
# Atomic rename pattern: build into <fixture>.csc.tmp, then `mv` over the
# original. Other concurrent bench runs are unaffected because the SCX
# format is append-only and the rename is single-file atomic.

set -uo pipefail  # NOT -e: continue past per-fixture failures so one bad
                  # fixture (e.g., pre-existing codec bug in tabula_sapiens)
                  # doesn't block CSC builds for the rest.

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench

# .env populates SCX_WORK_DIR / SCX_DATA_DIR.
set -a
source .env
set +a

# Build scx-cli with HDF5 disabled (h5 not needed for build-csc).
echo "=== building scx-cli ==="
cargo build --release -p scx-cli
SCX_BIN="${SCX_DIR}/target/release/scx"
[[ -x "${SCX_BIN}" ]] || { echo "scx binary missing at ${SCX_BIN}"; exit 1; }
echo "scx binary: ${SCX_BIN}"
echo

# Small + full tier accel gate fixtures (auto-codec only — that's what
# `gate_candidate.py --accel-only` exercises).
DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"
echo "=== data dir: ${DATA_DIR} ==="
FIXTURES=(
    "pbmc3k_auto.scx"
    "pbmc10k_auto.scx"
    "smartseq2_auto.scx"
    "tabula_sapiens_100k_auto.scx"
    "census_500k_auto.scx"
    "census_1m_auto.scx"
)

for fname in "${FIXTURES[@]}"; do
    src="${DATA_DIR}/${fname}"
    tmp="${DATA_DIR}/${fname}.csc.tmp"
    if [[ ! -f "${src}" ]]; then
        echo "skip: ${src} (file missing)"
        continue
    fi
    # Check via 'scx info' for existing CSC sidecar. The output has a
    # 'has_csc' field; if it's true, skip rebuild.
    if "${SCX_BIN}" info "${src}" 2>/dev/null | grep -qE '(has[_ ]csc|csc_shards)[: ]+(true|[1-9])'; then
        echo "skip: ${fname} (already has CSC sidecar)"
        continue
    fi
    echo
    echo "=== building CSC sidecar for ${fname} ==="
    rm -f "${tmp}"
    t0=$(date +%s)
    if "${SCX_BIN}" build-csc "${src}" "${tmp}" --memory-limit 32G --force; then
        t1=$(date +%s)
        mv "${tmp}" "${src}"
        echo "wrote CSC for ${fname} in $((t1 - t0))s"
    else
        echo "FAILED: build-csc on ${fname} — continuing"
        rm -f "${tmp}"
    fi
done

echo
echo "=== done ==="
