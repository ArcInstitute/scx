#!/bin/bash
# PR-12 (OPT-GPU-1/-4/-10/-13) — main vs branch, two builds, one host.
#
# Deliberately narrow. The filed item asks for "one build, one capture" over
# `accel_de` + `accel_pca` + `accel_to_gpu_anndata`, but two of its four items
# cannot move any of those:
#
#   * OPT-GPU-4's host round-trip is unreachable at every pyscx default
#     (`k = n_comps + n_oversamples = 60`, so `n_vars x k` is always even and
#     the odd branch never runs). It ships as correctness, not speed.
#   * OPT-GPU-13 is invisible to `accel_pca`: that runner builds an in-memory
#     adata, so the source is pyscx's single-shard `BorrowedCsrSource` and the
#     prefetch pipeline takes its sequential fallback. A backed X is the only
#     multi-shard GPU-PCA path, and no harness arm drives one --
#     `profile_gpu_freebies.py` does it directly instead.
#
# So this measures the two that can move (OPT-GPU-1 via `accel_de`'s GPU arms,
# OPT-GPU-10 via `to_gpu_anndata`), plus the falsification and the GPU-13
# micro-measurement.
#
# **Run alone.** Both arms `maturin develop` against the shared in-tree
# editable `.so`, so a co-scheduled job that merely *imports* pyscx is
# invalidated. Chain with `--dependency=afterany:<prev>`.
#
#SBATCH --job-name=scx-pr12-gpu-ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr12/gpu_ab_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr12/gpu_ab_%j.out

set -uo pipefail

# Exit status tracks the MEASUREMENT, not the last echo.
#
# These scripts run `set -uo pipefail` (not `-e`: an arm that fails should still
# let the other arm and the summary run), and every one of them used to end on a
# successful `echo`, so a failed build, a crashed profiler or a half-finished
# capture produced a SLURM job that reported COMPLETED 0:0. That is not
# hypothetical here: job 2938439 "COMPLETED" in 15 s having measured nothing,
# and 2938468 "COMPLETED" with one of its two arms dead. Since GitHub CI runs
# none of the GPU tests, a green SLURM job is the only signal these produce, and
# a green one that measured nothing is worse than a red one.
#
# `fail <msg>` records a failure and keeps going; the script exits non-zero at
# the end if anything called it.
STATUS=0
fail() { echo "!! $*" >&2; STATUS=1; }
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-pr12
OUT="${WORK}/gpu_ab_${SLURM_JOB_ID:-manual}"
WT_MAIN="${WORK}/wt-pr12-main"
WT_BRANCH="${WORK}/wt-pr12-branch"
BASE=$(git -C "${SCX_DIR}" rev-parse main)
# Pinned at submit time via `sbatch --export=ALL,PR12_BRANCH_SHA=$(git rev-parse HEAD)`.
# Without the pin this resolves `HEAD` when the job *starts*, which on a
# queue that spans a review round is a different tree from the one the
# numbers get attributed to.
HEAD_SHA=${PR12_BRANCH_SHA:-$(git -C "${SCX_DIR}" rev-parse HEAD)}
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo "main   : ${BASE}"
echo "branch : ${HEAD_SHA}"
echo "out    : ${OUT}"

# Fail before spending the allocation, not after.
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2
    exit 1
fi

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# `maturin develop` REPOINTS the env's editable install at whatever directory it
# was run from — here, a worktree this script then deletes. Without a restore
# the env is left with a `pyscx.pth` naming a path that no longer exists, and
# `import pyscx` then *succeeds* as an empty namespace package (`__file__ is
# None`) and fails at first use. That is worse than an ImportError, because
# nothing announces it. Observed on job 2938363, which left `scx-bench-gpu`
# broken exactly this way.
#
# Runs on every exit path, including the `FATAL:` returns and a SIGTERM from
# the scheduler.
restore_shared_env() {
    echo ""
    echo "=== restoring ${ENV}'s pyscx from ${SCX_DIR} ==="
    # Wholesale, exactly as the arm builds do. A reused target dir lets cargo
    # report "Finished in 0.59s" and re-link a 0-byte artifact from release/,
    # and maturin then dies on `Object is too small` — which is how job 2938504
    # left the env pointing at a worktree it had just deleted, the very thing
    # this function exists to prevent.
    rm -rf /home/nickyoungblut/.cargo-target-pr12-restore
    ( cd "${SCX_DIR}/pyscx" \
      && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR=/home/nickyoungblut/.cargo-target-pr12-restore \
         "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) \
        >"${OUT}/restore.log" 2>&1 \
      && env -u LD_LIBRARY_PATH "${ENV}/bin/python" -c "import pyscx, sys; sys.exit(0 if pyscx.__file__ else 1)" \
      && echo "  ok  ${ENV} restored" \
      || {
          echo "  !! RESTORE FAILED — ${ENV} may still point at a deleted worktree."
          echo "     See ${OUT}/restore.log. Manual fix:"
          echo "       cd ${SCX_DIR}/pyscx && VIRTUAL_ENV=${ENV} ${ENV}/bin/maturin develop --release --features hdf5,gpu"
      }
}
trap restore_shared_env EXIT

run_arm() {
    local name="$1" dir="$2"
    local target="/home/nickyoungblut/.cargo-target-pr12-${name}"
    echo ""
    echo "########## ARM ${name} (${dir}) ##########"
    # Wholesale, not just $TARGET/maturin: clearing only the maturin subdir
    # lets cargo re-link a 0-byte artifact from release/ and maturin dies on
    # "Object is too small".
    rm -rf "${target}"
    ( cd "${dir}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${target}" \
        "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -3
    local build_rc=${PIPESTATUS[0]}
    [ "${build_rc}" -eq 0 ] || {
        # A failed rebuild leaves the shared editable install pointing at the
        # PREVIOUS arm, so the import check and the GPU preflight below both
        # pass — on the wrong binary — and the job reports an A/B it never ran
        # (all three reviewers, round 2).
        echo "  !! arm ${name}: maturin build exited ${build_rc}"
        return "${build_rc}"
    }
    cd "${dir}" || return 1
    python -c "import pyscx; print('so:', pyscx.__file__)"

    # Preflight: one real GPU op, non-zero on failure. A build without the gpu
    # feature runs every measurement below on the CPU and reports plausible
    # numbers for the wrong thing.
    python - <<'PY' || { echo "FATAL: gpu feature missing on arm"; return 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
    raise
print("preflight ok")
PY

    set -a; . ./.env; set +a

    mkdir -p "${OUT}/emb-${name}"

    # OPT-GPU-10 + OPT-GPU-13 + the PCA falsification.
    GPU_FB_RUNS=3 \
        GPU_FB_DATASETS=tabula_sapiens_100k,census_500k \
        GPU_FB_SMALL=pbmc3k \
        GPU_FB_OUT="${OUT}/freebies_${name}.json" \
        GPU_FB_EMB_DIR="${OUT}/emb-${name}" \
        python benchmarks/scripts/profile_gpu_freebies.py \
        2>&1 | tee "${OUT}/freebies_${name}.txt" | grep -vE "warn|^\s*$"
    local fb_rc=${PIPESTATUS[0]}

    # OPT-GPU-1: the GPU DE routes. `hvg` is the control -- this PR touches
    # neither the HVG kernels nor the staging driver they share, so a moved
    # hvg means something other than the tie-term deletions is being measured.
    SCX_GPU_PROFILE=1 GPU_DE_RUNS=3 \
        GPU_DE_OPS=pdex_ref,wilcoxon,hvg \
        GPU_DE_DATASETS=pbmc3k,tabula_sapiens_100k \
        GPU_DE_OUT="${OUT}/de_${name}.json" \
        python benchmarks/scripts/profile_gpu_de_resident.py \
        2>&1 | tee "${OUT}/de_${name}.txt" | grep -vE "warn|^\s*$"
    # PIPESTATUS, not $?: both pipelines end in `grep`, whose status says
    # nothing about whether the profiler ran. Checked per measurement block.
    local de_rc=${PIPESTATUS[0]}
    [ "${fb_rc:-0}" -eq 0 ] || { echo "  !! arm ${name}: freebies profile exited ${fb_rc}"; return "${fb_rc}"; }
    [ "${de_rc}" -eq 0 ] || { echo "  !! arm ${name}: DE profile exited ${de_rc}"; return "${de_rc}"; }
    return 0
}

# BOTH arms run from a detached worktree at a pinned SHA, including the branch
# one. The precedent scripts run the branch arm out of the live tree, which
# makes the measurement hostage to any edit landing while the job sits in the
# queue -- and on this PR the queue spans review rounds.
rm -rf "${WT_MAIN}" "${WT_BRANCH}"; git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT_MAIN}" "${BASE}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add (main) failed"; exit 1; }
git -C "${SCX_DIR}" worktree add --detach "${WT_BRANCH}" "${HEAD_SHA}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add (branch) failed"; exit 1; }
# `profile_gpu_freebies.py` does not exist at main; pin it into that arm so the
# only difference between the two is the library. Benchmark-only, and the
# manifest rules permit it -- the main arm will report `git_dirty`.
cp "${SCX_DIR}/benchmarks/scripts/profile_gpu_freebies.py" "${WT_MAIN}/benchmarks/scripts/"
cp "${SCX_DIR}/.env" "${WT_MAIN}/.env"
cp "${SCX_DIR}/.env" "${WT_BRANCH}/.env"

run_arm main   "${WT_MAIN}" || fail "run_arm main failed"
run_arm branch "${WT_BRANCH}" || fail "run_arm branch failed"

for w in "${WT_MAIN}" "${WT_BRANCH}"; do
    git -C "${SCX_DIR}" worktree remove --force "$w" 2>/dev/null; rm -rf "$w"
done

echo ""
echo "########## SUMMARY ##########"
# `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:...` above makes numpy's MKL
# pick up the wrong libcblas and abort with `Intel oneMKL FATAL ERROR: Cannot
# load .../libcblas.so.3` — which killed this summary on job 2938363 *after*
# every measurement had been written. The raw JSON and the .npy embeddings are
# all on disk, so the summary is a convenience; drop the CUDA prefix for it
# rather than risk losing the run's readable output again.
env -u LD_LIBRARY_PATH python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:  # noqa: BLE001
        print(f"  (could not read {p}: {e})")
        return None


fb_main, fb_branch = load(out / "freebies_main.json"), load(out / "freebies_branch.json")
de_pre_m, de_pre_b = load(out / "de_main.json"), load(out / "de_branch.json")
# `load` returning None used to skip the table silently and still exit 0 — an
# A/B that compared nothing, reported as success (all three reviewers, round 2).
_absent = [n for n, v in (("freebies_main", fb_main), ("freebies_branch", fb_branch),
                          ("de_main", de_pre_m), ("de_branch", de_pre_b)) if not v]
if fb_main and fb_branch:
    print("\n-- wall (min of 3), main -> branch --")
    print(f"{'measurement':<46} {'main':>10} {'branch':>10} {'speedup':>9}")
    by_main = {(r.get('op'), r.get('variant')): r for r in fb_main}
    for r in fb_branch:
        k = (r.get('op'), r.get('variant'))
        m = by_main.get(k)
        if not m or 'error' in r or 'error' in m:
            print(f"{str(k):<46} ERROR: {r.get('error') or m.get('error')}")
            continue
        a, b = m['wall_s'], r['wall_s']
        label = k[0] + (f" [{k[1].split('__')[-1]}]" if k[1] else "")
        print(f"{label:<46} {a:>10.3f} {b:>10.3f} {a/b if b else float('nan'):>8.2f}x")
        if k[0] == 'to_gpu_anndata':
            print(f"{'':<46} shards={m.get('n_shards')} mode={r.get('transfer_mode')} "
                  f"uploaded main={m.get('bytes_uploaded')} branch={r.get('bytes_uploaded')}")
        if k[0] == 'pca_backed':
            print(f"{'':<46} shards={m.get('n_shards')} route={r.get('route')} "
                  f"rss main={m.get('peak_rss_mb'):.0f}MB branch={r.get('peak_rss_mb'):.0f}MB")
        if k[0] == 'pca_inmem':
            print(f"{'':<46} self-cosine main={m.get('self_cosine_min')} "
                  f"branch={r.get('self_cosine_min')}")

print("\n-- PCA falsification: cross-arm |cosine| per component, vs the same-arm floor --")
for name in ("pca_inmem__pbmc3k", "pca_backed__tabula_sapiens_100k",
             "pca_backed__census_500k"):
    pa, pb = out / "emb-main" / f"{name}.npy", out / "emb-branch" / f"{name}.npy"
    if not (pa.exists() and pb.exists()):
        print(f"  {name}: missing ({pa.exists()}/{pb.exists()})")
        continue
    a, b = np.load(pa), np.load(pb)
    if a.shape != b.shape:
        print(f"  {name}: shape mismatch {a.shape} vs {b.shape}")
        continue
    cos = []
    for k in range(a.shape[1]):
        u, v = a[:, k], b[:, k]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cos.append(1.0 if nu == 0 and nv == 0 else abs(float(u @ v) / (nu * nv + 1e-300)))
    ident = np.array_equal(a, b)
    print(f"  {name}: min|cos|={min(cos):.12f}  mean={np.mean(cos):.12f}  "
          f"bit-identical={ident}")

print("\n-- GPU DE (OPT-GPU-1); hvg is the control and should not move --")
de_main, de_branch = load(out / "de_main.json"), load(out / "de_branch.json")
if de_main and de_branch:
    # `profile_gpu_de_resident.py` writes a flat list of rows keyed by
    # (dataset, op), with a **median** `wall_ms` of 3 runs.
    def key(r):
        return (r.get('dataset'), r.get('op'))
    bm = {key(r): r for r in de_main}
    print(f"{'dataset/op':<34} {'main ms':>10} {'branch ms':>10} {'speedup':>9}  route")
    for r in de_branch:
        m = bm.get(key(r))
        if not m:
            continue
        a, b = m.get('wall_ms'), r.get('wall_ms')
        if not (a and b):
            continue
        label = f"{key(r)[0]}/{key(r)[1]}"
        print(f"{label:<34} {a:>10.1f} {b:>10.1f} {a/b:>8.2f}x  "
              f"{m.get('route')} -> {r.get('route')}")

if _absent:
    print(f"\n!! MISSING RESULT FILES: {', '.join(_absent)} — nothing was compared")
    raise SystemExit(1)
PY
[ $? -eq 0 ] || fail "the A/B summary could not compare both arms"

echo ""
echo "=== done; raw under ${OUT} ==="

if [ "${STATUS}" -ne 0 ]; then
    echo "=== FAILED: at least one step above did not complete; see !! lines ==="
fi
exit "${STATUS}"
