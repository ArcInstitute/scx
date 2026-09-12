#!/bin/bash
# Can `accel_pca__pyscx_gpu_backed` actually SEE the change it exists to gate?
#
# The arm runs on the runner's preprocessed fixture, which is HVG-subset to
# 2,000 genes. The 1.36-1.40x PR-12 measured was on the full gene axis
# (tabula 6.16 -> 4.54 s, census_500k 24.05 -> 17.14 s). On the subset the same
# datasets take 1.10 s and 3.68 s, so most of that wall is fixed GPU/setup cost
# and the host-side column-means pass — the only thing the change touches — is
# a much smaller share.
#
# A floor authored without checking this is the classic shape: a tolerance
# structurally unable to carry the claim it is written for. So measure the arm
# on main vs branch and only author the ceiling if the delta clears it.
#
# Two worktrees at pinned SHAs, one host, same-node, cargo+maturin per arm.
#
#SBATCH --job-name=scx-pca-backed-sens
#SBATCH --partition=gpu_high_mem,ctc_gpu_priority,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr12/pca_backed_sens_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr12/pca_backed_sens_%j.out

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
OUT="${WORK}/pca_backed_sens_${SLURM_JOB_ID:-manual}"
WT_MAIN="${WORK}/wt-sens-main"
WT_BRANCH="${WORK}/wt-sens-branch"
BASE=$(git -C "${SCX_DIR}" rev-parse main)
HEAD_SHA=${PR12_BRANCH_SHA:-$(git -C "${SCX_DIR}" rev-parse HEAD)}
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible." >&2; exit 1
fi
echo "main   : ${BASE}"
echo "branch : ${HEAD_SHA}"

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# `maturin develop` repoints the env's editable install at the worktree it runs
# from, and this script deletes those. Restore on every exit path or the env is
# left importing an empty namespace package (see job 2938363).
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
      || echo "  !! RESTORE FAILED — see ${OUT}/restore.log"
}
trap restore_shared_env EXIT

run_arm() {
    local name="$1" dir="$2"
    local target="/home/nickyoungblut/.cargo-target-sens-${name}"
    echo ""
    echo "########## ARM ${name} (${dir}) ##########"
    rm -rf "${target}"
    ( cd "${dir}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${target}" \
        "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -2
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
    set -a; . ./.env; set +a
    # Force a rebuild of the backed fixture per arm: it is written by the arm's
    # own pyscx, and a cached one from the other arm would silently make both
    # arms read a file only one of them produced.
    SCX_BENCH_REBUILD_BACKED=1 ARM="${name}" OUTDIR="${OUT}" python - <<'PYEOF' 2>&1 | tail -30
import json, os, statistics, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from benchmarks.comprehensive.benchmarks import accel_pca
from benchmarks.comprehensive.config import DATASETS

KEY = "accel_pca__pyscx_gpu_backed"
variant = next(v for v in accel_pca.accel_pca_variants() if v.key == KEY)
rows = {}
for name in ("tabula_sapiens_100k", "census_500k"):
    res = accel_pca.run(DATASETS[name], variant, n_runs=3)
    if res is None:
        print(f"  {name}: run() returned None"); continue
    walls = [r.extra["pca_backed_wall_s"] for r in res.runs if "pca_backed_wall_s" in r.extra]
    shards = [r.extra.get("pca_backed_n_shards") for r in res.runs]
    rows[name] = {"median_wall_s": statistics.median(walls), "walls": walls,
                  "n_shards": shards[0] if shards else None}
    print(f"  {name}: median={rows[name]['median_wall_s']:.3f}s shards={rows[name]['n_shards']}",
          flush=True)
Path(os.environ["OUTDIR"], f"sens_{os.environ['ARM']}.json").write_text(json.dumps(rows, indent=2))
PYEOF
    # PIPESTATUS, not $?: the pipeline above ends in `tail`, whose status is 0
    # even when the python inside died. Job 2938468's branch arm exited on a
    # StopIteration and the job still reported COMPLETED 0:0.
    local rc=${PIPESTATUS[0]}
    [ "${rc}" -eq 0 ] || { echo "  !! arm ${name}: measurement exited ${rc}"; return "${rc}"; }
    return 0
}

rm -rf "${WT_MAIN}" "${WT_BRANCH}"; git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT_MAIN}" "${BASE}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add (main) failed"; exit 1; }
git -C "${SCX_DIR}" worktree add --detach "${WT_BRANCH}" "${HEAD_SHA}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add (branch) failed"; exit 1; }
# The arm itself does not exist at main; pin the whole benchmark module and its
# test-visible siblings in, so the ONLY difference between the arms is the
# compiled library.
cp "${SCX_DIR}/benchmarks/comprehensive/benchmarks/accel_pca.py" \
   "${WT_MAIN}/benchmarks/comprehensive/benchmarks/accel_pca.py"
cp "${SCX_DIR}/.env" "${WT_MAIN}/.env"
cp "${SCX_DIR}/.env" "${WT_BRANCH}/.env"

run_arm main   "${WT_MAIN}" || fail "run_arm main failed"
run_arm branch "${WT_BRANCH}" || fail "run_arm branch failed"

for w in "${WT_MAIN}" "${WT_BRANCH}"; do
    git -C "${SCX_DIR}" worktree remove --force "$w" 2>/dev/null; rm -rf "$w"
done

echo ""
echo "########## SENSITIVITY ##########"
env -u LD_LIBRARY_PATH python - "${OUT}" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
try:
    m = json.loads((out / "sens_main.json").read_text())
    b = json.loads((out / "sens_branch.json").read_text())
except Exception as e:
    print("could not read both arms:", e); raise SystemExit(1)

print(f"{'dataset':<24} {'main s':>9} {'branch s':>9} {'speedup':>8} {'1.25x ceiling catches?':>24}")
for k in sorted(set(m) & set(b)):
    a, c = m[k]["median_wall_s"], b[k]["median_wall_s"]
    speed = a / c if c else float("nan")
    # A ceiling at 1.25x the branch median fails only if a regression pushes the
    # wall past it. It can catch a full revert only if main is already past it.
    catches = "YES" if a > c * 1.25 else "NO — floor cannot see a revert"
    print(f"{k:<24} {a:>9.3f} {c:>9.3f} {speed:>7.2f}x {catches:>24}")
    print(f"{'':<24} shards main={m[k]['n_shards']} branch={b[k]['n_shards']}")
PYEOF
# Capture got this check in round 1 and sensitivity did not — an incomplete copy
# of the same patch. Without `set -e` the summary's `SystemExit(1)` (missing or
# unreadable arm JSON) leaves the shell running straight on to `exit "${STATUS}"`.
[ $? -eq 0 ] || fail "the sensitivity summary could not compare both arms"

echo ""
echo "=== done; raw under ${OUT} ==="

if [ "${STATUS}" -ne 0 ]; then
    echo "=== FAILED: at least one step above did not complete; see !! lines ==="
fi
exit "${STATUS}"
