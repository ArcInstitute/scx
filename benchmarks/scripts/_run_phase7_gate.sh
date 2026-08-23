#!/bin/bash
# The gate owed by Organization Phase 7, covering 7a-7d in one run.
#
# ONE run for four sub-phases, deliberately. 7a and 7b (PR #452) never produced
# a measurement -- no gate script existed -- and 7c's tie-run primitive touches
# the dense DE inner loop, so the debt compounds rather than resolving. 7d adds
# Harmony's missing k-means M-step, which is the largest perf change of the four.
#
# NOT YET SUBMITTED. It is written now so the numbers it will report are
# reviewable before the hours are spent, and because a gate script that does not
# exist is how Phase 6 arrived at a gate that had never run. Submit it once 7d
# has landed; run `--benchmarks accel_de accel_de_nb_glm accel_hvg accel_pca
# accel_eval_metrics bench_csc_dispatch` alone if you want a 7a-7c answer sooner.
#
# ONE job, and nothing else submitted alongside: `maturin develop` rewrites the
# in-tree `.so` that every pyscx-importing job resolves through, so a second scx
# job overlapping this one would measure a binary neither of us intended.
#
#SBATCH --job-name=scx7-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/phase7-gate-%j.out

set -euo pipefail
REPO=/home/nickyoungblut/dev/rust/scx
cd "$REPO"

# sbatch exports the submitting shell's environment, and this repo's `.venv`
# sets VIRTUAL_ENV. maturin refuses outright when both VIRTUAL_ENV and
# CONDA_PREFIX are set ("Please unset one of them") and the job dies in seconds,
# before the build. Unset here rather than trusting the submitting shell.
unset VIRTUAL_ENV
unset PYTHONHOME PYTHONPATH

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench-gpu

echo "=== HEAD: $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- build, release: a debug .so runs 4-10x slower and poisons every timing ---
echo "=== maturin develop --release --features hdf5,gpu"
( cd pyscx && maturin develop --release --features hdf5,gpu )

# --- preflight the features about to be measured for hours -------------------
# Not a smoke test. Each arm asserts that THIS branch's code is what is loaded,
# using a signal that separates it from main, and exits non-zero otherwise.
echo "=== preflight"
python - <<'PY'
import sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata, pyscx

fail = []

# 1. GPU HVG runs at all. 7a gated four GPU column-moment finalizes on a
#    finiteness check; if that gate is wrong, this raises rather than degrading
#    quietly, and every accel_hvg number below would describe a CPU fallback.
rng = np.random.default_rng(0)
X = rng.poisson(1.0, size=(4000, 800)).astype(np.float32)
ad = anndata.AnnData(X=sp.csr_matrix(X))
try:
    pyscx.accel.highly_variable_genes(ad, n_top_genes=200, flavor="seurat_v3", device="gpu")
    route = ad.uns.get("scx_accel", {}).get("highly_variable_genes", {}).get("route")
    print(f"preflight: GPU HVG route={route}")
    if route and "gpu" not in str(route) and "rapids" not in str(route):
        fail.append(f"GPU HVG silently fell back to {route}")
except Exception as e:                                    # noqa: BLE001
    fail.append(f"GPU HVG raised: {e!r}")

# 2. The tie-run primitive is the one in this build. `tie_correct=True` must
#    match scipy's mannwhitneyu EXACTLY -- that equality is what 7c pinned, and
#    it is a sharper staleness probe than any timing.
from scipy.stats import mannwhitneyu
n = 240
d = rng.poisson(2.0, size=(n, 12)).astype(np.float32)
d[:, 3] = 3.0                                             # a total tie
g = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
a2 = anndata.AnnData(
    X=sp.csr_matrix(d),
    obs=pd.DataFrame({"g": pd.Categorical(g)}, index=[f"c{i}" for i in range(n)]),
    var=pd.DataFrame(index=[f"g{j}" for j in range(12)]),
)
pyscx.accel.rank_genes_groups(a2, "g", tie_correct=True, device="cpu")
rgg = a2.uns["rank_genes_groups"]
D = np.asarray(a2.X.todense(), dtype=np.float64)
worst = 0.0
for grp in rgg["names"].dtype.names:
    m = dict(zip(rgg["names"][grp], np.asarray(rgg["scores"][grp], np.float64)))
    mask = g == grp
    for j in range(12):
        x, y = D[mask, j], D[~mask, j]
        if x.min() == x.max() == y.min() == y.max():
            continue
        u1, _ = mannwhitneyu(x, y, use_continuity=False, alternative="two-sided",
                             method="asymptotic")
        n1, n2 = len(x), len(y)
        _, cnt = np.unique(np.concatenate([x, y]), return_counts=True)
        tc = float(np.sum(cnt**3 - cnt))
        s2 = (n1 * n2 / 12.0) * ((n + 1) - tc / (n * (n - 1)))
        if s2 <= 0:
            continue
        worst = max(worst, abs((u1 - n1 * n2 / 2.0) / np.sqrt(s2) - m[f"g{j}"]))
print(f"preflight: max |z - scipy| with tie_correct=True = {worst:.3e}")
if worst > 1e-9:
    fail.append(f"tie-corrected z diverges from scipy by {worst:.3e}; the .so is "
                f"stale or the tie-run walk changed")

if fail:
    for f in fail:
        print(f"PREFLIGHT FAILED: {f}", file=sys.stderr)
    sys.exit(1)
print("preflight OK")
PY

# --- the gate -----------------------------------------------------------------
# On a GPU node, not --no-gpu: `accel_hvg` carries an `hvg_overlap_vs_scanpy`
# floor, 7a gated two scx-gpu finalize sites plus two more in scx-accel/hvg/gpu.rs,
# and 7c added a GPU arm to the Wilcoxon reference test. --no-gpu would leave all
# of that unmeasured while still printing a green gate.
#
# accel_harmony is included ONLY if 7d registered it in ALL_BENCHMARKS.
# `capture_baseline.py` rejects an off-list name outright -- that exact failure
# killed Phase 6's first gate attempt in 2.2s -- so check before adding it:
#   python -c "from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS as A; print('accel_harmony' in A)"
#
# READ THE RESULT CAREFULLY, on two counts.
#
#   1. `LATEST` predates a lot of accel work. Where a benchmark x dataset triple
#      has zero baseline rows the REGRESSION arm cannot fire at all --
#      `diff_summaries` classifies a baseline-absent row as appearing, not
#      regressing -- so that triple tests its absolute floors and nothing else.
#      Reporting the run as "no regressions against LATEST" would be false.
#   2. `accel_eval_metrics`' two GPU route floors have never been evaluated. If
#      they come back unevaluated again, say so; do not report them as passed.
#
# --skip-smoke: the pre-submit contract check sweeps all 14 runners on pbmc3k and
# blocks submission if any fails. Right for a full capture, wrong here -- every
# benchmark named below is SCX-only and touches no competitor format.
echo "=== gate_candidate.py"
python benchmarks/comprehensive/scripts/gate_candidate.py \
    --benchmarks accel_de accel_de_nb_glm accel_hvg accel_pca \
                 accel_eval_metrics bench_csc_dispatch \
    --datasets pbmc3k census_1m \
    --name phase7-accel-gate \
    --skip-smoke \
    -v
