#!/bin/bash
# Phase-4 task 4.5 — is the parallel shard validation what moved the hvg control?
#
# The main-vs-branch capture (2709095) gave GPU DE 27–36x and, on the *control*,
# GPU HVG **0.93–0.99x** with its summed host-decode up ~10 %. Consistently
# below 1.0 across three datasets is not obviously noise, and a measured
# regression on an op the change was not aiming at is worth resolving rather
# than defending.
#
# The hypothesis: commit C parallelised `validate_shard_for_gpu_de`, which runs
# on the **consuming** thread. On GPU DE that is a win — residency put
# validation on the critical path by removing the 122 redundant decodes that
# used to dwarf it. On GPU HVG it can only lose: HVG validates every shard,
# gains nothing from residency, and is decode-bound, so a parallel scan on the
# consumer competes with the very prefetch workers feeding it.
#
# One build, two arms, via `SCX_GPU_VALIDATE_PAR_MIN_NNZ`. Pinning it above any
# real shard's nnz takes the `else` branch, which is the pre-4.5 serial scan
# **unchanged** — a genuine baseline by construction, not an "off" arm that
# means something new. (#373's lesson was that a knob's "off" state is not
# automatically the old behaviour; here it is, and the reason is checkable in
# the diff.)
#
# Both ops run so the trade is quantified in both directions: whatever the
# parallel scan costs HVG, it should be visibly buying DE something.
#
# **Run alone** — see `_run_4_5_gpu_verify.sh`'s header on the shared `.so`.
#SBATCH --job-name=scx-4.5-validate-ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.5/validate_ab_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.5/validate_ab_%j.out

set -uo pipefail
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
OUT="/home/nickyoungblut/scx-bench-4.5/validate_ab_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8

TARGET=/home/nickyoungblut/.cargo-target-45-valab
rm -rf "${TARGET}"
( cd "${SCX_DIR}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${TARGET}" \
    "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -2
cd "${SCX_DIR}" || exit 1
python - <<'PY' || { echo "FATAL: gpu feature missing"; exit 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
PY
set -a; . ./.env; set +a

# 5 runs, not 3: the effect under test is ~6 %, which a median-of-3 cannot
# separate from cluster noise. That is the whole reason this job exists.
run_arm() {
    local name="$1" thresh="$2"
    echo ""
    echo "########## ARM ${name} (SCX_GPU_VALIDATE_PAR_MIN_NNZ=${thresh}) ##########"
    SCX_GPU_PROFILE=1 GPU_DE_RUNS=5 \
        SCX_GPU_VALIDATE_PAR_MIN_NNZ="${thresh}" \
        GPU_DE_OPS=hvg \
        GPU_DE_DATASETS=tabula_sapiens_100k,census_500k,census_1m \
        GPU_DE_OUT="${OUT}/hvg_${name}.json" \
        python benchmarks/scripts/profile_gpu_de_resident.py \
        2>&1 | tee "${OUT}/hvg_${name}.txt" | grep -E "^  (census|tabula)"
    SCX_GPU_PROFILE=1 GPU_DE_RUNS=5 \
        SCX_GPU_VALIDATE_PAR_MIN_NNZ="${thresh}" \
        GPU_DE_OPS=pdex_ref,wilcoxon \
        GPU_DE_DATASETS=tabula_sapiens_100k,census_500k \
        GPU_DE_OUT="${OUT}/de_${name}.json" \
        python benchmarks/scripts/profile_gpu_de_resident.py \
        2>&1 | tee "${OUT}/de_${name}.txt" | grep -E "^  (census|tabula)"
}

# 4611686018427387904 = 2^62: above any conceivable shard nnz, so every shard
# takes the serial `else` branch — i.e. the pre-4.5 scan verbatim.
run_arm serial   4611686018427387904
run_arm parallel 65536

echo ""
echo "########## SUMMARY ##########"
python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])


def load(name):
    rows = []
    for kind in ("hvg", "de"):
        p = out / f"{kind}_{name}.json"
        if p.exists():
            rows.extend(json.loads(p.read_text()))
    return {(r["dataset"], r["op"]): r for r in rows}


ser, par = load("serial"), load("parallel")
keys = sorted(set(ser) & set(par))
hdr = (f"{'dataset':<22} {'op':<9} {'serial ms':>11} {'parallel ms':>12} "
       f"{'par/ser':>8}   verdict")
print(hdr)
print("-" * len(hdr))
for k in keys:
    s, p = ser[k], par[k]
    ratio = p["wall_ms"] / s["wall_ms"] if s["wall_ms"] else float("nan")
    if ratio < 0.97:
        verdict = "parallel helps"
    elif ratio > 1.03:
        verdict = "parallel HURTS"
    else:
        verdict = "indistinguishable"
    print(f"{k[0]:<22} {k[1]:<9} {s['wall_ms']:11.1f} {p['wall_ms']:12.1f} "
          f"{ratio:8.3f}   {verdict}")
print()
print("Reading it: hvg should show the cost of the parallel scan (it validates "
      "but gains nothing from residency); DE should show the benefit. If hvg is "
      "'indistinguishable' the 0.93-0.99 in job 2709095 was noise and commit C "
      "stays as-is. If hvg is 'parallel HURTS' and DE is not 'parallel helps', "
      "commit C is not paying for itself and should go.")
PY

echo ""
echo "=== done; artifacts under ${OUT} ==="
