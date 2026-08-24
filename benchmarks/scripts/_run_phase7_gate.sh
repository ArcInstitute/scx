#!/bin/bash
# The gate owed by Organization Phase 7, covering 7a-7e in one run.
#
# ONE run for four sub-phases, deliberately. 7a and 7b (PR #452) never produced
# a measurement -- no gate script existed -- and 7c's tie-run primitive touches
# the dense DE inner loop, so the debt compounds rather than resolving.
#
# What 7d actually changed, which is not what an earlier version of this comment
# predicted:
#
#   * The PFlog alpha estimator no longer filters candidates by dispersion
#     (SS7.14). Every `pflog(alpha=None)` caller gets a different alpha on the
#     same counts, so the transform itself moved. No pflog benchmark exists, so
#     the preflight below is the only measurement of it in this job.
#   * The gemm distance expansion recomputes exactly below its error floor
#     (SS7.12), in the exact-kNN path and in the blocked pairwise kernel. That is
#     why `accel_knn` joins the list.
#
# What 7e changed:
#
#   * Harmony's soft k-means sub-loop gained the M-step it never had (SS7.4), on
#     BOTH arms: `Y = normalize(Z_cos R')` plus the distances it invalidates, per
#     sub-iteration. That is `max_iter x max_iter_kmeans` = 60 extra passes over
#     a d x N embedding per run, so it is the largest perf change in this job --
#     and `accel_harmony` now exists to measure it.
#   * On the GPU arm those are two cuBLAS gemms plus a normalize per sub-iter,
#     dispatched OUTSIDE the captured graph, so the capture still happens once.
#   * LISI's default neighbourhood dropped from 90 to 89 (SS7.19). No LISI
#     benchmark exists; it is gated Rust-side by `lisi_reference_tests.rs`.
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
# Derived, with the submitting path as the fallback: sbatch runs this from an
# arbitrary cwd, and `--output` above cannot take a variable. Keep the two in
# sync if the checkout ever moves.
# `sbatch` copies the batch script into slurmd's spool and runs it from there, so
# `${BASH_SOURCE[0]}` is something like /var/spool/slurmd/job123/slurm_script.
# `dirname/../..` is then /var/spool -- which EXISTS, so `cd` succeeds and the
# `||` fallback never fires. Validate the candidate instead of trusting the exit
# status, and fail loudly if neither candidate is a checkout.
REPO_FALLBACK=/home/nickyoungblut/dev/rust/scx
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd || true)
if [ ! -f "${REPO:-/nonexistent}/pyscx/Cargo.toml" ]; then
  REPO=$REPO_FALLBACK
fi
if [ ! -f "$REPO/pyscx/Cargo.toml" ]; then
  echo "cannot locate the scx checkout (tried \"$REPO\"); set REPO_FALLBACK" >&2
  exit 1
fi
cd "$REPO"
echo "=== repo: $REPO"

# sbatch exports the submitting shell's environment, and this repo's `.venv`
# sets VIRTUAL_ENV. maturin refuses outright when both VIRTUAL_ENV and
# CONDA_PREFIX are set ("Please unset one of them") and the job dies in seconds,
# before the build. Unset here rather than trusting the submitting shell.
unset VIRTUAL_ENV
unset PYTHONHOME PYTHONPATH

# Derived from whichever conda is on PATH, with the submitting host's install as
# the fallback. `#SBATCH --output` above stays absolute of necessity: sbatch
# parses those directives before any shell runs, so it cannot take a variable.
CONDA_SH=$( { conda info --base 2>/dev/null || echo /home/nickyoungblut/miniforge3; } )/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate scx-bench-gpu

echo "=== HEAD: $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- build, release: a debug .so runs 4-10x slower and poisons every timing ---
#
# Whether nvcc exists is NOT part of scx-gpu's build-script fingerprint, so a
# target directory that once saw a CPU-only build replays the stub branch -- and
# its warning -- on a machine that has nvcc. That is exactly how the first
# attempt at this gate (job 2839940) died: it built a wheel from 41-byte PTX
# files and failed at CUDA_ERROR_INVALID_IMAGE 82 seconds in, having measured
# nothing. Two defences, because either alone is insufficient:
#   * `touch` makes the build script RERUN, so the probe happens at all;
#   * `SCX_GPU_REQUIRE_NVCC=1` makes a missing nvcc a 20-second build failure
#     instead of a runtime symptom four steps from its cause.
export SCX_GPU_REQUIRE_NVCC=1
export PATH=/usr/local/cuda/bin:${PATH}
touch scx-gpu/build.rs
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
    print(f"preflight: GPU HVG route={route!r}")
    # A MISSING route is a failure, not a pass. `if route and ...` treated None
    # and "" as "nothing to complain about", which is the one outcome this check
    # exists to catch: a stale build or a broken route stamp would have sailed
    # through the probe whose whole purpose is to stop hours of CPU-fallback
    # numbers being reported as GPU results.
    if not route:
        fail.append("GPU HVG stamped no route at all (stale build or broken "
                    "route metadata) -- cannot confirm the op ran on the device")
    elif "gpu" not in str(route) and "rapids" not in str(route):
        fail.append(f"GPU HVG silently fell back to {route!r}")
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
# Compare the P-VALUE against scipy's own `.pvalue`, not a z this script
# reconstructs. The first version of this probe took `U` from scipy and rebuilt z
# with `s2 = (n1*n2/12)*((n+1) - tc/(n(n-1)))` -- SCX's own variance formula --
# so a bug in the tie term would have matched and the probe would have passed.
# The same defect was found and fixed in the reference GENERATOR one round
# earlier; it survived here because the fix went file by file instead of
# following the formula. scipy's `.pvalue` applies scipy's own tie correction,
# which is the whole reason it can falsify SCX's.
worst = 0.0
for grp in rgg["names"].dtype.names:
    m = dict(zip(rgg["names"][grp], np.asarray(rgg["pvals"][grp], np.float64)))
    mask = g == grp
    for j in range(12):
        x, y = D[mask, j], D[~mask, j]
        if x.min() == x.max() == y.min() == y.max():
            continue
        res = mannwhitneyu(x, y, use_continuity=False, alternative="two-sided",
                           method="asymptotic")
        if not np.isfinite(res.pvalue):
            continue
        worst = max(worst, abs(res.pvalue - m[f"g{j}"]))
print(f"preflight: max |p - scipy.pvalue| with tie_correct=True = {worst:.3e}")
if worst > 1e-12:
    fail.append(f"tie-corrected p diverges from scipy by {worst:.3e}; the .so is "
                f"stale or the tie-run walk changed")

# --- 3. the PFlog alpha estimator is the post-SS7.14 one ----------------------
#
# There is no pflog benchmark, so without this arm the job would spend hours
# without touching the one numeric change 7d made to a user-facing transform. The
# probe is the review's own regime: counts simulated from a KNOWN alpha, where the
# pre-fix estimator biases high by discarding the under-dispersed genes.
#
# The assertion is on `n_genes_used`, not on alpha. At 50 cells the estimator's
# sampling spread is wider than the bias it is being distinguished from, so an
# alpha tolerance cannot separate the two estimators -- measured, and the reason
# `pflog_reference_tests.rs` rests on exact arms instead. The pool size is exact:
# every gene with a mean above mu_min, where the old filter kept only the
# over-dispersed ones.
import anndata, scipy.sparse as sp

rng = np.random.default_rng(70301)
n_cells, n_genes, alpha_true = 60, 80, 0.02
mus = np.exp(rng.uniform(np.log(0.5), np.log(200.0), n_genes))
cols = [
    rng.negative_binomial(1.0 / alpha_true, 1.0 / (1.0 + alpha_true * mu), size=n_cells)
    for mu in mus
]
counts = np.stack(cols, axis=1).astype(np.float32)
ad = anndata.AnnData(sp.csr_matrix(counts))
ad.var_names = [f"g{i}" for i in range(n_genes)]
ad.obs_names = [f"c{i}" for i in range(n_cells)]
pyscx.accel.pflog(ad, store="baseline")
meta = ad.uns.get("pflog", {})
n_used = int(meta.get("n_genes_used", -1))
n_expected = int((counts.mean(axis=0) > 1e-3).sum())
print(
    f"preflight: pflog alpha={float(meta.get('alpha', float('nan'))):.6f} "
    f"(alpha_true={alpha_true}), n_genes_used={n_used}/{n_expected}"
)
if meta.get("alpha_source") != "estimated":
    fail.append(f"pflog did not estimate alpha (alpha_source={meta.get('alpha_source')!r})")
elif n_used != n_expected:
    fail.append(
        f"pflog pooled {n_used} of {n_expected} genes with a mean above mu_min; the "
        f"dispersion pre-filter is back, or the .so predates SS7.14"
    )

# --- 3b. the gate env's harmonypy is the one the pins were made against ------
#
# `accel_harmony` scores every arm against harmonypy, and `scx-bench-gpu` did
# not carry it at all -- so without this the benchmark would emit
# `missing_reason=no_harmonypy` for all three variants and its floors would come
# back as missing-metric violations hours in, rather than as a 20-second failure
# here. And `pip install harmonypy` resolves to **2.0.0**, while the Rust pins
# in `harmony_reference_values.rs` were produced by **0.2.0** -- which is what
# `.venv`, `scx-bench` and `scx-gpu` all carry. A silent major-version swap
# would make the benchmark floor and the cargo-test pins disagree about what
# "harmonypy" means.
import importlib.metadata as _md
try:
    _hv = _md.version("harmonypy")
    print(f"preflight: harmonypy {_hv}")
    if _hv != "0.2.0":
        fail.append(
            f"harmonypy {_hv} in this env, but the Rust pins were produced by "
            f"0.2.0 -- `pip install --no-deps 'harmonypy==0.2.0'`"
        )
except _md.PackageNotFoundError:
    fail.append("harmonypy is not installed; accel_harmony scores every arm "
                "against it -- `pip install --no-deps 'harmonypy==0.2.0'`")

# --- 4. Harmony's k-means sub-loop has the SS7.4 M-step, in THIS build --------
#
# The three arms above touch HVG, Wilcoxon and pflog; none of them sees Harmony,
# so without this the job could spend hours reporting `accel_harmony` numbers
# from a binary whose centroids never move.
#
# The signal is the OBJECTIVE CURVE, and it was chosen by measuring candidates
# rather than by reasoning about them. Two that look obvious do not work:
# between-batch variance after correction separates the two builds by 0.5152 vs
# 0.5136 (nothing), and `n_iterations` is 7 either way. The relative drop across
# the run does separate them, because the M-step is what actually minimises
# `sum R.dist` at fixed R:
#
#     rel_drop = (obj[0] - obj[-1]) / |obj[0]|
#
#     with the M-step:     obj 185.60 -> -53.18   rel_drop 1.286
#     without (reverted):  obj 251.57 -> 241.66   rel_drop 0.039
#
# Both measured through pyscx on THIS fixture, with the M-step call site removed
# and the wheel rebuilt for the second. A ratio rather than an absolute so the
# bar does not need recalibrating whenever the fixture is touched; 0.5 sits an
# order of magnitude above 0.039 and well below 1.286.
rng = np.random.default_rng(90210)
n_per, d_pcs, n_batch = 300, 12, 3
base = rng.normal(0.0, 1.0, size=(n_per * n_batch, d_pcs))
shift = rng.normal(0.0, 3.0, size=(n_batch, d_pcs))
lab = np.repeat(np.arange(n_batch), n_per)
pcs = (base + shift[lab]).astype(np.float32)
ah = anndata.AnnData(
    X=sp.csr_matrix(np.zeros((len(lab), 1), dtype=np.float32)),
    obs=pd.DataFrame({"b": pd.Categorical([f"b{v}" for v in lab])},
                     index=[f"c{i}" for i in range(len(lab))]),
)
ah.obsm["X_pca"] = pcs
try:
    pyscx.accel.harmony_integrate(ah, "b", device="cpu", random_state=0)
    obj = list(ah.uns["harmony"]["objective_harmony"])
    if len(obj) < 2 or not np.isfinite(obj[0]) or obj[0] == 0.0:
        fail.append(f"harmony objective curve is unusable: {obj[:3]}")
    else:
        rel_drop = (obj[0] - obj[-1]) / abs(obj[0])
        print(f"preflight: harmony objective {obj[0]:.4f} -> {obj[-1]:.4f} "
              f"(rel_drop {rel_drop:.4f}, n_iter={ah.uns['harmony']['n_iterations']})")
        if rel_drop < 0.5:
            fail.append(
                f"harmony objective fell by only {rel_drop:.3f} of its initial "
                f"value; the k-means sub-loop is not moving centroids, so the .so "
                f"predates SS7.4's M-step (reverted build measures 0.039)"
            )
except Exception as e:                                    # noqa: BLE001
    fail.append(f"harmony_integrate raised: {e!r}")

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
# accel_harmony IS here as of 7e, which wrote the module and its five
# registration sites. `capture_baseline.py` rejects an off-list name outright --
# that exact failure killed Phase 6's first gate attempt in 2.2s -- so the
# registration was falsified before this line was added: dropping the name from
# ALL_BENCHMARKS reds `test_every_floored_benchmark_is_in_all_benchmarks`.
# Confirm before any future edit here:
#   python -c "from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS as A; print('accel_harmony' in A)"
#
# Its floors are BASELINE-ABSENT by construction -- the benchmark did not exist
# when `LATEST` was captured -- so its regression arm cannot fire and this run
# tests its absolute floors only. Its `mean_per_pc_r_vs_harmonypy` floor is
# provisional at 0.90 pending this first measurement; report what it measured so
# it can be set from data rather than from the withdrawn R-harmony figure.
#
# accel_knn is here because SS7.12 changed the precision of the distance the
# exact-kNN path returns for near-duplicate rows. `thresholds.yaml` already
# carries `recall_vs_scanpy >= 0.90` floors for it, which is the signal.
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
#   3. Accelerator FINGERPRINTS will move, and the fingerprint check is not
#      filtered by --only-benchmarks. Expect drift in the kNN fingerprint
#      (SS7.12), any that touches pflog (SS7.14), and any that touches Harmony or
#      LISI (SS7.4, SS7.19 -- both change output for every caller). Do NOT pass
#      `-- --allow-fingerprint-drift` pre-emptively: run it, read which
#      fingerprints moved, confirm each one is explained by those two changes, and
#      only then re-run with the flag -- naming the drifted fingerprints in the
#      report. Phase 6 waved seven through and that is how a real change hides.
#
# --skip-smoke: the pre-submit contract check sweeps all 14 runners on pbmc3k and
# blocks submission if any fails. Right for a full capture, wrong here -- every
# benchmark named below is SCX-only and touches no competitor format.
echo "=== gate_candidate.py"
python benchmarks/comprehensive/scripts/gate_candidate.py \
    --benchmarks accel_de accel_de_nb_glm accel_hvg accel_pca accel_knn \
                 accel_harmony accel_eval_metrics bench_csc_dispatch \
    --datasets pbmc3k census_1m \
    --name phase7-accel-gate \
    --skip-smoke \
    -v
