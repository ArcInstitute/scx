#!/bin/bash
# Phase 0 — same-build A/B of the `ml_loader` harness edit, plus the R1
# re-capture that carries the wait-tail metrics.
#
# WHY: the phase-0 `p` capture's gate reported two absolute-floor violations on
# `ml_loader / scx_fast / census_1m` — `batches_per_sec__raw` 36.4 against a
# floor of 48, and `batches_per_sec__hvg_norm` 44.2 against 54. Neither
# scenario is touched by this branch (the diff is confined to
# `_run_gpu_train_epoch`, `_GpuEpochResult`, the `gpu_train` emitter and a
# module import), and the scenario that IS touched came out at 67.2 against a
# floor of 56 — faster than `LATEST`'s 64.7. But "my diff cannot have caused
# it" is an argument, not a measurement, and a floor violation in a PR is the
# PR's to explain.
#
# So: both arms run the SAME `.so` and the SAME fixtures, differing only in
# which checkout supplies `benchmarks/`. `base` is a `git worktree` at `main`;
# `head` is this branch. PYTHONPATH puts the main repo's `pyscx/python` first
# in both arms, so the measured extension module is byte-identical and the
# only variable is the Python harness.
#
#   sbatch benchmarks/scripts/_run_phase0_ml_loader_ab.sh
#
# The `head` arm doubles as the R1 re-capture: the first `p` capture predates
# `batch_wait_ms_p99` / `batch_wait_ms_max`, which is where census_1m's wait
# actually lives (p50 13 microseconds, p95 799 microseconds, yet a 0.76
# steady-state fraction — the mass is past p95).
#SBATCH --job-name=scx-phase0-mlab
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase0/mlab_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase0/mlab_%j.out

set -euo pipefail
export REPO=/home/nickyoungblut/dev/rust/scx
BASE_TREE=/large_storage/arcinfra/projects/scx/scratch/phase0/main-tree
WORK=/large_storage/arcinfra/projects/scx/scratch/phase0
OUT="$WORK/mlab_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

cd "$REPO"
SHA=$(git rev-parse --short=7 HEAD)
NAME_HEAD="candidate_phase0_ml_head_${SHA}${ORDER:+_${ORDER}}"
NAME_BASE="candidate_phase0_ml_base_main${ORDER:+_${ORDER}}"

CONDA_BASE="$HOME/miniforge3"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate scx-bench

export SCX_BENCH_GPU_PARTITION="${SCX_BENCH_GPU_PARTITION:-gpu_high_mem,ctc_gpu_priority,gpu}"
DATASETS="census_1m"
FORMATS="scx_fast"
CAPTURE=benchmarks/comprehensive/scripts/capture_baseline.py
COMMON=(--benchmarks ml_loader --formats $FORMATS --datasets $DATASETS
        --skip-convert --skip-fingerprints --skip-smoke --no-accel)

echo "=== phase-0 ml_loader same-build A/B ==="
echo "host   : $(hostname)"
echo "head   : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "base   : $(git -C "$BASE_TREE" rev-parse HEAD) (worktree at main)"
echo "gpu    : $SCX_BENCH_GPU_PARTITION"
echo "out    : $OUT"

# The two arms differ ONLY in which tree supplies `benchmarks/`. `pyscx` comes
# from the main repo's editable build in both, so the .so is byte-identical —
# this is a harness A/B, not a build A/B.
diff <(git show main:benchmarks/comprehensive/benchmarks/ml_loader.py) \
     "$BASE_TREE/benchmarks/comprehensive/benchmarks/ml_loader.py" \
  || { echo "base worktree is not at main's ml_loader" >&2; exit 2; }

run_base() {
  echo ""
  echo "=== arm BASE (main's harness): $NAME_BASE ==="
  ( cd "$BASE_TREE" \
    && PYTHONPATH="$REPO/pyscx/python:$BASE_TREE" \
       python "$CAPTURE" --name "$NAME_BASE" "${COMMON[@]}" ) 2>&1 \
    | tee "$OUT/capture_base.log"
}
run_head() {
  echo ""
  echo "=== arm HEAD (this branch's harness): $NAME_HEAD ==="
  PYTHONPATH="$REPO/pyscx/python:$REPO" \
    python "$CAPTURE" --name "$NAME_HEAD" "${COMMON[@]}" 2>&1 \
    | tee "$OUT/capture_head.log"
}

# ORDER decides which arm runs FIRST, and it exists because the first run of
# this A/B put base first and measured a uniform ~5% head deficit across all
# five scenarios — including `hvg` and `norm`, which this branch does not
# touch, while `gpu_train`, the only scenario it does touch, moved least
# (0.981). A deficit that tracks arm POSITION rather than arm CONTENT is
# drift, not code. Swapping the order is the falsification: if `head` first
# now measures the *higher* numbers, the ordering explains it.
echo "arm order: ${ORDER:-base-first}"
if [ "${ORDER:-base-first}" = "head-first" ]; then
  run_head; run_base
else
  run_base; run_head
fi

echo ""
echo "=== A/B ==="
PYTHONPATH="$REPO/pyscx/python:$REPO" \
  AB_BASE_ROOT="$BASE_TREE" AB_BASE_NAME="$NAME_BASE" \
  AB_HEAD_ROOT="$REPO" AB_HEAD_NAME="$NAME_HEAD" \
  python - <<'PY' | tee "$OUT/ab.txt"
# NOTE the QUOTED heredoc delimiter. With an unquoted <<PY the shell performs
# command substitution INSIDE the script body, so a backtick in a Python
# comment is executed: an earlier version whose comment named
# capture_baseline.py in backticks produced three "command not found" lines
# in the job log (job 2945208). Harmless there because the backticks wrapped
# prose, but it is arbitrary command execution out of a comment. Values now
# arrive through the environment instead of being interpolated by the shell.
import json, os, statistics

def med(root, name, key):
    # Each arm writes under ITS OWN checkout: capture_baseline.py resolves
    # results/ relative to the tree it runs from, and the base arm runs from
    # the worktree. Reading both from the main repo would silently find one
    # and report the other as None.
    path = os.path.join(root, "benchmarks", "comprehensive", "results", name,
                        "raw", "ml_loader__scx_fast__census_1m.json")
    try:
        d = json.load(open(path))
    except Exception:
        return None
    v = [r["extra"][key] for r in d["runs"]
         if r.get("extra", {}).get(key) is not None]
    return statistics.median(v) if v else None

B_ROOT, B_NAME = os.environ["AB_BASE_ROOT"], os.environ["AB_BASE_NAME"]
H_ROOT, H_NAME = os.environ["AB_HEAD_ROOT"], os.environ["AB_HEAD_NAME"]
FLOORS = {"batches_per_sec__raw": 48.0, "batches_per_sec__hvg_norm": 54.0,
          "batches_per_sec__gpu_train": 56.0}
hdr = "{:34s} {:>12s} {:>13s} {:>8s} {:>7s}".format(
    "metric", "base(main)", "head(branch)", "ratio", "floor")
print(hdr)
for k in ("batches_per_sec__raw", "batches_per_sec__hvg_norm",
          "batches_per_sec__gpu_train", "batches_per_sec__norm",
          "batches_per_sec__hvg"):
    b, h = med(B_ROOT, B_NAME, k), med(H_ROOT, H_NAME, k)
    ratio = "{:.3f}".format(h / b) if (b and h) else "-"
    f = FLOORS.get(k)
    flag = ""
    if f is not None:
        if b is not None and b < f: flag += " base<floor"
        if h is not None and h < f: flag += " head<floor"
    print("{:34s} {:>12s} {:>13s} {:>8s} {:>7s}{}".format(
        k,
        "-" if b is None else "{:.1f}".format(b),
        "-" if h is None else "{:.1f}".format(h),
        ratio,
        "-" if f is None else "{:.1f}".format(f),
        flag))
PY
echo ""
echo "=== tracked manifest rows in the main repo (must be unmodified) ==="
git -C "$REPO" status --porcelain benchmarks/comprehensive/results/raw/ | head
echo ""
echo "=== done ==="
