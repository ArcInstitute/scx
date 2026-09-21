#!/bin/bash
# Stage one atlas-scale multimodal fixture for `multimodal_atlas_streaming`.
#
#   sbatch benchmarks/scripts/slurm_build_multimodal_atlas.sh multiome_atlas_500k
#   sbatch --dependency=afterany:$PREV \
#          benchmarks/scripts/slurm_build_multimodal_atlas.sh citeseq_atlas_1m
#
# ONE FIXTURE PER JOB, and chained — never two at once. CLAUDE.local.md's rule
# is about the shared `pyscx/python/pyscx/*.so` that every env resolves through
# its editable install: any job that rebuilds it invalidates every other job
# that will *import* pyscx, and this one imports it for the h5mu -> scx
# conversion. Slurm will also co-schedule two of your jobs on one node, which
# turns the second fixture's write bandwidth into the first's noise.
#
# `cpu_high_mem` rather than `cpu_preemptible`: this is a multi-hour
# single-shot build whose partial output is worthless, and preemption would
# discard it. 8 of its 9 nodes carry ~2 TB and the partition allows 14 days.
# The generator's own peak is one row block (~2.5 GB at the ATAC geometry), so
# 256 GB is headroom for the streaming conversion, not for the generator.
#
#SBATCH --job-name=build_mm_atlas
#SBATCH --partition=cpu_high_mem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/logs/build_multimodal_atlas_%j.log
#SBATCH --error=benchmarks/logs/build_multimodal_atlas_%j.err

set -euo pipefail

DATASET="${1:?usage: sbatch slurm_build_multimodal_atlas.sh <dataset> [extra args]}"
shift || true

# Under sbatch, $0 points at SLURM's spool copy, so dirname($0) is not the
# repo. SLURM_SUBMIT_DIR is.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/nickyoungblut/dev/rust/scx}"
cd "$REPO_ROOT"

if [[ -z "${SCX_WORK_DIR:-}" && -f "${REPO_ROOT}/.env" ]]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi
export SCX_WORK_DIR="${SCX_WORK_DIR:?set SCX_WORK_DIR or provide a repo-root .env}"
export SCX_DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"

# Multiple readers of one HDF5 file otherwise hit `BlockingIOError: errno 11`.
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p benchmarks/logs

echo "=== build_multimodal_atlas: ${DATASET} ==="
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "RAM:   $(free -g | awk '/Mem/{print $2}') GB"
echo "Disk:  $(df -h "${SCX_DATA_DIR}" | tail -1)"
echo ""

PY="${REPO_ROOT}/.venv/bin/python"

# Preflight. This job runs for hours and then hands its output to a benchmark
# that imports pyscx; finding out at the end that from_h5mu is missing from
# this build wastes the whole allocation. Exercise the exact call chain on a
# fixture small enough to be free.
echo "--- preflight: h5mu -> scx -> narrowed to_mudata round trip"
"$PY" - <<'PREFLIGHT'
import sys, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0, "benchmarks/scripts")
import pyscx
for name in ("from_h5mu", "open"):
    if not hasattr(pyscx, name):
        sys.exit(f"PREFLIGHT FAIL: pyscx has no {name}")
import build_multimodal_atlas as B

spec = B.AtlasSpec(n_obs=256, modalities=(
    B.ModalitySpec("rna", 200, 0.05, "rna", "GENE"),
    B.ModalitySpec("adt", 8, 1.0, "protein", "ADT"),
))
with tempfile.TemporaryDirectory(prefix="mm_preflight_") as d:
    h5mu = Path(d) / "pf.h5mu"
    scx = Path(d) / "pf_multimodal.scx"
    rng = np.random.default_rng(0)
    B.write_skeleton(h5mu, spec, rng)
    realised = B.fill_matrices(h5mu, spec, rng, 64)
    B.verify_h5mu(h5mu, spec, realised)
    B.convert_to_scx(h5mu, scx, spec)
    exp = pyscx.open(str(scx))
    narrow = exp.to_mudata(data_dtype="uint16")
    wide = exp.to_mudata()
    for m in ("rna", "adt"):
        assert str(narrow.mod[m].X.dtype) == "uint16", narrow.mod[m].X.dtype
        assert float(narrow.mod[m].X.sum()) == float(wide.mod[m].X.sum()), m
print("preflight ok", flush=True)
PREFLIGHT
echo ""

echo "--- build"
"$PY" benchmarks/scripts/build_multimodal_atlas.py \
    --datasets "${DATASET}" -v "$@"

echo ""
echo "=== Done: $(date) ==="
ls -la "${SCX_DATA_DIR}/${DATASET}".h5mu "${SCX_DATA_DIR}/${DATASET}"_multimodal.scx 2>/dev/null || true
