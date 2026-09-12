#!/bin/bash
# Capture the `accel_pca__pyscx_gpu_backed` arm so its floors come from a
# measurement rather than an estimate.
#
# This arm exists because PR-12's 1.36-1.40x on out-of-core GPU PCA had nothing
# gating it: every other `accel_pca` variant runs on the runner's in-memory
# adata, which reaches GPU PCA through pyscx's single-shard `BorrowedCsrSource`,
# so none of them can observe the multi-shard column-means pass at all. Measured
# on pbmc3k an in-memory `device="gpu"` PCA reports `route:
# rapids_singlecell_gpu` — it never enters the native path.
#
# Scoped to the two multi-shard datasets (tabula_sapiens_100k = 7 shards,
# census_500k = 31). pbmc3k is one shard at the writer's default
# `shard_target_rows`, where the decode-prefetch takes its sequential fallback.
#
# Uses `scx-bench-gpu`'s existing pyscx — it does NOT run `maturin develop`, so
# it repoints nothing and can run beside other jobs. Verify the install points
# at the repo before trusting the numbers; the preflight below does.
#
#SBATCH --job-name=scx-pca-backed-cap
#SBATCH --partition=gpu_high_mem,ctc_gpu_priority,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr12/pca_backed_cap_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr12/pca_backed_cap_%j.out

set -uo pipefail

# Exit status tracks the MEASUREMENT, not the last echo. Every one of these
# scripts used to end on a successful `echo`, so a crashed profiler or a
# half-finished capture produced a SLURM job reporting COMPLETED 0:0 — job
# 2938439 "COMPLETED" in 15 s having measured nothing. GitHub CI runs none of
# the GPU tests, so a green SLURM job is the only signal here, and a green one
# that measured nothing is worse than a red one.
STATUS=0
fail() { echo "!! $*" >&2; STATUS=1; }
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
OUT="/home/nickyoungblut/scx-bench-pr12/pca_backed_cap_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2; exit 1
fi
echo "branch: $(git -C "${SCX_DIR}" rev-parse HEAD)"

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# The install must point at the repo, not at a deleted A/B worktree. A `.pth`
# naming a missing directory imports as an EMPTY namespace package — `__file__
# is None` — which fails at first use rather than at import, so check it.
echo "  pth: $(cat "${ENV}"/lib/python*/site-packages/pyscx.pth 2>/dev/null)"
python - <<'PY' || { echo "FATAL: pyscx unusable or not a gpu build"; exit 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
assert pyscx.__file__, "pyscx imported as an empty namespace package"
print("  pyscx:", pyscx.__file__)
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
    raise
print("  preflight ok")
PY

cd "${SCX_DIR}" || exit 1
set -a; . ./.env; set +a

# The summary checks that THIS job produced each scoped dataset, not that some
# file with the right name exists in the shared raw directory.
# Nanoseconds: `date +%s` is integral, and `st_mtime < started` then accepts a
# pre-existing file written in the same second (codex, round 3). The expected
# paths are also removed up front, so "exists and is newer" cannot be satisfied
# by anything this job did not write.
JOB_START_EPOCH=$(date +%s.%N)
EXPECTED_DATASETS=$(python -c "
import sys; sys.path.insert(0, '${SCX_DIR}')
from benchmarks.comprehensive.benchmarks import accel_pca
print(','.join(sorted(accel_pca.FORMAT_DATASET_SCOPE['accel_pca__pyscx_gpu_backed'])))
") || { echo "FATAL: could not resolve the arm's dataset scope"; exit 1; }
export JOB_START_EPOCH EXPECTED_DATASETS
echo "expecting results for: ${EXPECTED_DATASETS}"
RAW_DIR="${SCX_DIR}/benchmarks/comprehensive/results/raw"
for d in ${EXPECTED_DATASETS//,/ }; do
    rm -f "${RAW_DIR}/accel_pca__accel_pca__pyscx_gpu_backed__${d}.json"
done

# Drive the benchmark module directly rather than through `run_all.py`: its
# `ALL_FORMATS` does not contain accel variants at all (they come from
# `config.get_formats(include_accel=True)`, which only `run_parallel.py` calls),
# so `--formats accel_pca__*` resolves to nothing and the run prints
# "SKIP: no single-modality-compatible formats in scope" having measured
# nothing -- with the real cause one `WARNING: Unknown format key` line further
# up. This is the call a run_parallel worker makes, minus submitit.
python - <<'PYEOF' 2>&1 | tail -60
import sys

sys.path.insert(0, "/home/nickyoungblut/dev/rust/scx")
from benchmarks.comprehensive.benchmarks import accel_pca
from benchmarks.comprehensive.config import DATASETS
from benchmarks.comprehensive.results import write_result

KEY = "accel_pca__pyscx_gpu_backed"
variant = next(v for v in accel_pca.accel_pca_variants() if v.key == KEY)
scope = accel_pca.FORMAT_DATASET_SCOPE[KEY]
# EVERY scoped dataset must produce a result. The first version started `rc = 1`
# and set it to 0 on the first success, so one dataset succeeding made the job
# green while the other silently produced nothing — under a `fail` message that
# claimed to reject exactly that (all three reviewers, round 2).
missing = []
for name in sorted(scope):
    print(f"\n=== {name} ===", flush=True)
    res = accel_pca.run(DATASETS[name], variant, n_runs=3)
    if res is None:
        print(f"  {name}: run() returned None (variant unavailable here)")
        missing.append(name)
        continue
    print(f"  wrote {write_result(res)}", flush=True)
if missing:
    print(f"!! no result for: {', '.join(missing)}", file=sys.stderr)
sys.exit(1 if (missing or not scope) else 0)
PYEOF
[ ${PIPESTATUS[0]} -eq 0 ] || fail "the capture produced no result for at least one dataset"

echo ""
echo "########## FLOOR INPUTS ##########"
env -u LD_LIBRARY_PATH python - "${OUT}" <<'PY'
import json, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
# `write_result` writes to the fixed RAW_RESULTS_DIR, not to --output-dir. That
# directory is SHARED across runs, so a plain glob lets a stale file from an
# earlier invocation stand in for a dataset this one failed to produce. Require
# the exact set this run was scoped to, and require each file to be newer than
# the job's start (all three reviewers, round 2).
import os
import time

raw_dir = Path("/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/results/raw")
started = float(os.environ.get("JOB_START_EPOCH", "0"))
expected = sorted(
    (raw_dir / f"accel_pca__accel_pca__pyscx_gpu_backed__{d}.json")
    for d in os.environ["EXPECTED_DATASETS"].split(",")
)
stale = [p.name for p in expected if p.exists() and p.stat().st_mtime < started]
absent = [p.name for p in expected if not p.exists()]
if absent or stale:
    print(f"MISSING: {absent}   STALE (predate this job): {stale}")
    raise SystemExit(1)
raws = expected
if not raws:
    print("NO RAW RESULTS — the arm produced nothing"); raise SystemExit(1)
for p in raws:
    d = json.loads(p.read_text())
    runs = d.get("runs") or []
    walls = [r["extra"]["pca_backed_wall_s"] for r in runs if "pca_backed_wall_s" in r.get("extra", {})]
    shards = [r["extra"].get("pca_backed_n_shards") for r in runs if "pca_backed_n_shards" in r.get("extra", {})]
    routes = {r.get("extra", {}).get("gpu_dispatch_route") for r in runs}
    if not walls:
        print(f"{d.get('dataset')}: NO pca_backed_wall_s in extras — the floor would read 'missing'")
        continue
    med = statistics.median(walls)
    print(f"{d.get('dataset')}: n_runs={len(walls)} shards={shards[0] if shards else '?'} "
          f"routes={routes}")
    print(f"  median pca_backed_wall_s = {med:.3f}s   -> floor at 1.25x = {med*1.25:.2f}")
PY
[ $? -eq 0 ] || fail "the floor-input summary could not read the results"

echo ""
echo "=== done; raw under ${OUT} ==="

if [ "${STATUS}" -ne 0 ]; then
    echo "=== FAILED: at least one step above did not complete; see !! lines ==="
fi
exit "${STATUS}"
