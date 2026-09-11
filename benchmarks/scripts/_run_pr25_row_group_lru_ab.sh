#!/bin/bash
# OPT-FORMATIO-1 (PR-25) — same-build A/B of the row-group LRU on `read_scattered`.
#
# The scattered gather over a row-group-framed file now retains the row groups
# it decodes in the shard LRU (`CacheKey::Group`), where before it decoded the
# touched groups every batch and dropped them. Both arms run the SAME `.so`;
# the arm is selected at runtime by the reader-layer kill-switch
# `SCX_ROW_GROUP_CACHE=0` (read once per process, forwarded to every worker by
# `run_parallel._slurm_setup_cmds`). Same host pool, same fixtures, same plans
# (`read_scattered` seeds its random plans), so the only difference is whether
# a group decoded once is decoded again.
#
#   sbatch benchmarks/scripts/_run_pr25_row_group_lru_ab.sh
#
# ONE job: this is the orchestrator (`capture_baseline.py` submits the per-cell
# SLURM jobs itself and waits), and the two arms run back to back inside it, so
# no two captures of ours are ever in flight together. It must NOT rebuild the
# `.so` — `pyscx/python/pyscx/*.so` is cluster-global, and the release build is
# done before submitting (the harness refuses a debug `.so` on its own).
#
# `--formats` is load-bearing, not tuning: `read_scattered` is scoped to the
# four `scx_compact_trial_g*` keys, all of which are "additional"-tier, so a
# capture without it produces NOTHING and the benchmark's 14 route floors are
# skipped silently by `check_absolute_floors`. `--skip-smoke` is load-bearing
# too: the pre-submit runner smoke matches `--formats` against the converter
# runners, and these keys have none, so it refuses to submit ("matched none of
# the 13 available runners") — a refusal `--mode dry-run` cannot show, because
# the dry run never runs the smoke (job 2930269 died on exactly this after a
# clean dry run).
#
# Preflight before either arm: one real gather through the feature under test,
# in both arms, exiting non-zero if the counters do not move the way the arm
# says — a capture is also a test of the diagnostics, and a silent no-op arm
# would read as "no change".
#
# The orchestrator itself is tiny (it waits on submitit); the cells are sized by
# `run_parallel.py`. `cpu_batch`, not preemptible: killing the orchestrator
# strands the arm with results on disk and no snapshot.
#SBATCH --job-name=scx-pr25-rg-ab
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pr25/capture_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pr25/capture_%j.out

set -euo pipefail

export REPO=/home/nickyoungblut/dev/rust/scx
# /tmp is node-local on Chimera; everything the job needs must live under /home.
WORK=/home/nickyoungblut/scx-bench-pr25
OUT="$WORK/run_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

cd "$REPO"
SHA=$(git rev-parse --short=7 HEAD)
NAME_OFF="candidate_pr25_rg_off_${SHA}"
NAME_ON="candidate_pr25_rg_on_${SHA}"

# The orchestrator MUST run from `scx-bench`: `run_parallel` only activates a
# conda env on each worker when its own CONDA_PREFIX contains "scx-bench".
CONDA_BASE="$HOME/miniforge3"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate scx-bench
# Import pyscx from THIS checkout's editable build, not whatever the env holds:
# `scx-bench` has carried a wheel installed from an earlier branch A/B
# (`pip show pyscx` → `direct_url.json` names `/tmp/pr08-wheel-branch-*`), and
# `run_parallel._slurm_setup_cmds` forwards PYTHONPATH to every worker, so this
# one line selects the build for the whole capture without touching the env.
# `$REPO` itself is on it too: `python /abs/path/preflight.py` puts the script's
# directory on sys.path, not the cwd, and the preflight imports the harness.
export PYTHONPATH="$REPO/pyscx/python:$REPO${PYTHONPATH:+:$PYTHONPATH}"

DATASETS="pbmc3k smartseq2 tabula_sapiens_100k"
FORMATS="scx_compact_trial_g256 scx_compact_trial_g512"
CAPTURE=benchmarks/comprehensive/scripts/capture_baseline.py
COMMON=(--benchmarks read_scattered --formats $FORMATS --datasets $DATASETS
        --skip-convert --skip-fingerprints --skip-smoke --include-additional --no-accel)

echo "=== PR-25 row-group LRU same-build A/B ==="
echo "host      : $(hostname)"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "dirty     : $(git status --porcelain | grep -vc '^??' || true) tracked file(s) modified"
echo "python    : $(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "pyscx     : $(python -c 'import pyscx, os; print(pyscx.__file__)')"
echo "arms      : $NAME_OFF / $NAME_ON"
echo "out       : $OUT"
ls -la "$REPO"/pyscx/python/pyscx/pyscx.cpython-*.so

# ---------------------------------------------------------------------------
# Preflight: the `.so` on disk carries the feature and both arms select.
# ---------------------------------------------------------------------------
PREFLIGHT="$OUT/preflight.py"
cat > "$PREFLIGHT" <<'PY'
import json, os, subprocess, sys
import numpy as np
import pyscx
from benchmarks.comprehensive.config import DATASETS

path = str(DATASETS["tabula_sapiens_100k"].path_for_format("scx_compact_trial_g256"))
rng = np.random.default_rng(42)
n = pyscx.open(path).n_obs
plan = [(int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(64)]

def drive():
    ds = pyscx.IndexPlanDataset(path, normalize=False, cache_shards=128,
                                sort_by_shard=True, lookahead=0,
                                max_memory_mb=8192, scatter_block_index=True)
    list(ds.iter_with_plans(iter([list(plan), list(plan)]), lookahead=0))
    return ds.cache_metrics()

expected_prefix = os.path.join(os.environ["REPO"], "pyscx", "python") + os.sep
if not pyscx.__file__.startswith(expected_prefix):
    sys.exit(f"preflight: pyscx resolved to {pyscx.__file__}, not this checkout's build")
arm = os.environ.get("SCX_ROW_GROUP_CACHE", "unset")
m = drive()
for k in ("row_group_hits", "row_group_misses", "block_index_groups"):
    if k not in m:
        sys.exit(f"preflight: cache_metrics() lacks {k!r} — the .so on disk predates PR-25")
if m["block_index_groups"] == 0 or m["full_shard_groups"] != 0:
    sys.exit(f"preflight: framed fixture did not take the block-index route: {m}")
if arm == "0":
    ok = m["row_group_hits"] == 0 and m["row_group_misses"] == 0
else:
    ok = m["row_group_misses"] > 0 and m["row_group_hits"] == m["row_group_misses"]
print(json.dumps({"arm": arm, "cache_metrics": m}))
if not ok:
    sys.exit(f"preflight: arm SCX_ROW_GROUP_CACHE={arm} did not select: {m}")
PY
echo ""
echo "=== preflight: arm ON (default) ==="
( unset SCX_ROW_GROUP_CACHE; python "$PREFLIGHT" ) | tee "$OUT/preflight_on.json"
echo "=== preflight: arm OFF (SCX_ROW_GROUP_CACHE=0) ==="
SCX_ROW_GROUP_CACHE=0 python "$PREFLIGHT" | tee "$OUT/preflight_off.json"

# ---------------------------------------------------------------------------
# Dry run: the narrowed invocation must resolve to 3 datasets × 2 formats.
# ---------------------------------------------------------------------------
echo ""
echo "=== dry run ==="
python "$CAPTURE" --mode dry-run --name "${NAME_ON}_dry" "${COMMON[@]}" 2>&1 | tee "$OUT/dry_run.log"
if grep -q "SKIP" "$OUT/dry_run.log"; then
    echo "dry run reported a SKIP — the narrowing resolved away a cell" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Arm OFF — the pre-change regime, same build.
# ---------------------------------------------------------------------------
echo ""
echo "=== arm OFF: $NAME_OFF ==="
SCX_ROW_GROUP_CACHE=0 python "$CAPTURE" --name "$NAME_OFF" "${COMMON[@]}" 2>&1 \
    | tee "$OUT/capture_off.log"

# ---------------------------------------------------------------------------
# Arm ON — the shipped default. Also carries the `scx_auto` loader benchmarks:
# those fixtures are framed too (format v4), so `index_plan`'s scattered
# scenarios exercise the row-group path under the default `IndexPlanDataset`
# budget (the 26–57x in docs/performance.md), while `cellset_gather` takes the
# whole-shard route by default and must come out flat against LATEST.
# ---------------------------------------------------------------------------
echo ""
echo "=== arm ON: $NAME_ON ==="
unset SCX_ROW_GROUP_CACHE
python "$CAPTURE" --name "$NAME_ON" \
    --benchmarks read_scattered index_plan cellset_gather \
    --formats $FORMATS scx_auto --datasets $DATASETS \
    --skip-convert --skip-fingerprints --skip-smoke --include-additional --no-accel 2>&1 \
    | tee "$OUT/capture_on.log"

# ---------------------------------------------------------------------------
# Gate the ON arm against LATEST, and summarise the A/B.
# ---------------------------------------------------------------------------
echo ""
echo "=== gate: $NAME_ON vs LATEST ==="
set +e
# `--no-gpu` as well as `--no-accel`: without it the gate's pre-flight submits a
# GPU probe to the starved `preemptible` partition and waits 600 s for it —
# there is nothing GPU-shaped in this capture (job 2930275 sat in that wait).
python benchmarks/comprehensive/scripts/gate_candidate.py --skip-capture --name "$NAME_ON" \
    --benchmarks read_scattered index_plan cellset_gather \
    --formats $FORMATS scx_auto --datasets $DATASETS --no-accel --no-gpu \
    -- --report-json "$OUT/gate_report.json" 2>&1 | tee "$OUT/gate.log"
GATE_RC=${PIPESTATUS[0]}
set -e
echo "gate exit: $GATE_RC"

echo ""
echo "=== A/B summary ==="
python benchmarks/scripts/_pr25_row_group_ab_summary.py \
    "benchmarks/comprehensive/results/$NAME_OFF" \
    "benchmarks/comprehensive/results/$NAME_ON" \
    --out "$OUT/ab_summary.json"

echo ""
echo "=== done (gate exit $GATE_RC) ==="
exit "$GATE_RC"
