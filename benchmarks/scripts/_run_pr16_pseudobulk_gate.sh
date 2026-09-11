#!/bin/bash
# OPT-ACCEL-4 (PR-16) — no-regression capture + gate for the parallel pseudobulk scatter.
#
# The pseudobulk CSR scatter (streaming, in-memory, CSC) now partitions its
# output across the rayon pool instead of running one serial `+=` per nonzero.
# The kernel-level speedup is measured by `cargo bench -p scx-accel --bench
# pseudobulk` (criterion, before/after in one build); this job is the
# real-dataset check that the change is a no-regression on the comprehensive
# suite's pseudobulk-bearing cells and that the route floors still fire:
#
#   bench_csc_dispatch × {bench_csc__pseudobulk_csr, bench_csc__pseudobulk_csc}
#                     × tabula_sapiens_100k   (pseudobulk_dex; `csc_dispatch_correct` ≥ 1.0)
#
# `pseudobulk_dex`'s wall is mostly the pydeseq2 fit, so no speedup claim is
# made from this capture; the aggregation share is what can move.
#
#   sbatch benchmarks/scripts/_run_pr16_pseudobulk_gate.sh
#   PR16_ARM=base sbatch --export=ALL benchmarks/scripts/_run_pr16_pseudobulk_gate.sh   # on main, pre-change .so
#
# ONE job (the orchestrator; `capture_baseline.py` submits the per-cell SLURM
# jobs itself and waits). It must NOT rebuild the `.so` — `pyscx/python/pyscx/*.so`
# is cluster-global; build `maturin develop --release` before submitting.
# `--skip-smoke` is load-bearing: the pre-submit runner smoke matches `--formats`
# against the converter runners, and `bench_csc__*` keys have none.
#SBATCH --job-name=scx-pr16-pb-gate
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr16/capture_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr16/capture_%j.out

set -euo pipefail

export REPO=/home/nickyoungblut/dev/rust/scx
WORK=/home/nickyoungblut/scx-bench-pr16
OUT="$WORK/run_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

cd "$REPO"
SHA=$(git rev-parse --short=7 HEAD)
# `PR16_ARM=base` names the same capture taken on `main` with the pre-change
# `.so`, for the same-build A/B that attributes a delta against LATEST (which
# may predate unrelated merges) to this change or not.
NAME="candidate_pr16_${PR16_ARM:-pseudobulk}_${SHA}"

# The orchestrator MUST run from `scx-bench`: `run_parallel` only activates a
# conda env on each worker when its own CONDA_PREFIX contains "scx-bench".
CONDA_BASE="$HOME/miniforge3"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate scx-bench
# Import pyscx from THIS checkout's editable build, not the wheel `scx-bench`
# carries from an earlier branch A/B; forwarded to every worker by
# `run_parallel._slurm_setup_cmds`.
export PYTHONPATH="$REPO/pyscx/python:$REPO${PYTHONPATH:+:$PYTHONPATH}"

DATASETS="tabula_sapiens_100k"
FORMATS="bench_csc__pseudobulk_csr bench_csc__pseudobulk_csc"
CAPTURE=benchmarks/comprehensive/scripts/capture_baseline.py
COMMON=(--benchmarks bench_csc_dispatch --formats $FORMATS --datasets $DATASETS
        --skip-convert --skip-fingerprints --skip-smoke --include-accel --no-gpu)

echo "=== PR-16 parallel pseudobulk scatter: no-regression capture ==="
echo "host      : $(hostname)"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
DIRTY=$(git status --porcelain | grep -v '^??' || true)
echo "dirty     : $(printf '%s' "$DIRTY" | grep -c . || true) tracked file(s) modified"
if [ -n "$DIRTY" ]; then
    # A capture attributed to a SHA must be of that SHA's tree; name what is not.
    echo "$DIRTY" | sed 's/^/             /'
fi
echo "so sha256 : $(sha256sum "$REPO"/pyscx/python/pyscx/pyscx.cpython-*.so | cut -c1-16)…"
echo "python    : $(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "pyscx     : $(python -c 'import pyscx, os; print(pyscx.__file__)')"
echo "name      : $NAME"
echo "out       : $OUT"
ls -la "$REPO"/pyscx/python/pyscx/pyscx.cpython-*.so

# ---------------------------------------------------------------------------
# Preflight: the `.so` on disk is this checkout's, and the parallel scatter is
# what runs — an unsorted scipy CSR must agree bit for bit with the sorted one
# (the group partition vs the column blocks), and a backed read must agree with
# the in-memory one. A capture is also a test of the build it measures.
# ---------------------------------------------------------------------------
PREFLIGHT="$OUT/preflight.py"
cat > "$PREFLIGHT" <<'PY'
import os, sys, tempfile
import numpy as np, scipy.sparse as sp, anndata as ad, pandas as pd
import pyscx

expected_prefix = os.path.join(os.environ["REPO"], "pyscx", "python") + os.sep
if not pyscx.__file__.startswith(expected_prefix):
    sys.exit(f"preflight: pyscx resolved to {pyscx.__file__}, not this checkout's build")

rng = np.random.default_rng(3)
n_obs, n_vars = 5000, 800
X = sp.random(n_obs, n_vars, density=0.08, format="csr", random_state=7, dtype=np.float32)
X.sort_indices()
X.data = np.exp(rng.uniform(np.log(1e-5), np.log(1e5), size=X.nnz)).astype(np.float32)
obs = pd.DataFrame({"g": [f"g{i % 37}" for i in range(n_obs)]})
adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"v{j}" for j in range(n_vars)]))
means, groups = pyscx.accel.pseudobulk_means(adata, "g")

indptr, indices, data = X.indptr.astype(np.int64).copy(), X.indices.astype(np.int32).copy(), X.data.copy()
for r in range(n_obs):
    lo, hi = int(indptr[r]), int(indptr[r + 1])
    indices[lo:hi] = indices[lo:hi][::-1]; data[lo:hi] = data[lo:hi][::-1]
un = adata.copy(); un.X = sp.csr_matrix((data, indices, indptr), shape=X.shape); un.X.has_sorted_indices = False
means_un, groups_un = pyscx.accel.pseudobulk_means(un, "g")
if groups != groups_un or not np.array_equal(means.view(np.uint64), means_un.view(np.uint64)):
    sys.exit("preflight: unsorted CSR did not reproduce the sorted result bitwise")

with tempfile.TemporaryDirectory(dir=os.environ["REPO"] + "/target") as d:
    p = os.path.join(d, "pf.scx"); pyscx.from_anndata(adata, p)
    backed = pyscx.open(p).to_anndata(backed=True)
    means_b, groups_b = pyscx.accel.pseudobulk_means(backed, "g")
if groups != groups_b or not np.array_equal(means.view(np.uint64), means_b.view(np.uint64)):
    sys.exit("preflight: backed streaming did not reproduce the in-memory result bitwise")
print("preflight ok: sorted == unsorted == backed, bitwise")
PY
echo ""
echo "=== preflight ==="
python "$PREFLIGHT" | tee "$OUT/preflight.log"

# ---------------------------------------------------------------------------
# Dry run: the narrowed invocation must resolve to 1 dataset × 2 formats.
# ---------------------------------------------------------------------------
echo ""
echo "=== dry run ==="
python "$CAPTURE" --mode dry-run --name "${NAME}_dry" "${COMMON[@]}" 2>&1 | tee "$OUT/dry_run.log"
if grep -q "SKIP" "$OUT/dry_run.log"; then
    echo "dry run reported a SKIP — the narrowing resolved away a cell" >&2
    exit 2
fi

echo ""
echo "=== capture: $NAME ==="
python "$CAPTURE" --name "$NAME" "${COMMON[@]}" 2>&1 | tee "$OUT/capture.log"

echo ""
echo "=== gate: $NAME vs LATEST ==="
set +e
python benchmarks/comprehensive/scripts/gate_candidate.py --skip-capture --name "$NAME" \
    --benchmarks bench_csc_dispatch --formats $FORMATS --datasets $DATASETS --no-gpu \
    -- --report-json "$OUT/gate_report.json" 2>&1 | tee "$OUT/gate.log"
GATE_RC=${PIPESTATUS[0]}
set -e
echo "gate exit: $GATE_RC"

echo ""
echo "=== cells ==="
python - "$NAME" <<'PY'
import json, statistics, sys, pathlib
name = sys.argv[1]
base = json.load(open("benchmarks/comprehensive/results/baselines/LATEST/summary.json"))["rows"]
for f in sorted(pathlib.Path(f"benchmarks/comprehensive/results/{name}/raw").glob("bench_csc_dispatch__*.json")):
    d = json.load(open(f)); k = f"{d['benchmark']}__{d['format']}__{d['dataset']}"
    b = base.get(k, {})
    print(f"{k}: wall {d.get('median_wall_s'):.3f}s (LATEST {b.get('median_wall_s')}), "
          f"rss {statistics.median([r['peak_rss_mb'] for r in d['runs']]):.0f} MB (LATEST {b.get('peak_rss_mb_median')}), "
          f"csc_dispatch_correct {[r['extra'].get('csc_dispatch_correct') for r in d['runs']]}")
PY

echo ""
echo "=== done (gate exit $GATE_RC) ==="
exit "$GATE_RC"
