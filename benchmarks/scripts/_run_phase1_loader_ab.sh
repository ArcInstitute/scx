#!/bin/bash
# Phase 1 of the ML-loader plan — two-build A/B of the tier-1 loader fixes.
#
# W1 pre-sizes the cell-set gather's `indices`/`data` from a plan prescan, W2
# replaces `collate_cell`'s per-row withheld-gene `HashSet` with a per-SET
# sorted query panel + binary search, W4 adds an opt-in batch charge (off by
# default, so it must move nothing here). The headline is `us_per_cell__collate`,
# whose masking branch was measured at ~34% of collate wall (78.7 us/cell with
# an empty mask against 120.1 with 25% withheld, pbmc3k).
#
#   sbatch benchmarks/scripts/_run_phase1_loader_ab.sh
#
# TWO BUILDS, NOT ONE. Every other A/B in this directory selects its arm at
# runtime through a kill switch (`SCX_ROW_GROUP_CACHE=0`, `SCX_SCATTER_BLOCK_INDEX=0`)
# and so runs both arms off one `.so`. W1/W2/W4 have no such switch — they are
# unconditional code — so the arms are two worktrees, two builds, two
# `CARGO_TARGET_DIR`s. The alternative considered and rejected was shipping a
# `SCX_COLLATE_LEGACY_MASK` switch purely to make this job cheaper: it would
# leave a second crop-mask path in the tree for phase 3 to delete, and it would
# only cover W2.
#
# ONE HOST, all four arms. `capture_baseline.py` submits a SLURM job per
# (benchmark, dataset, format) cell, which would scatter the arms across nodes
# — and phase 0 measured a 5-7% position/host effect on this cluster. So this
# job calls `run_parallel._run_benchmark` in process instead: same node, same
# page cache, same everything but the `.so`. It still writes real
# `BenchmarkResult` JSONs through `write_result`, which is what the manifest
# rule requires.
#
# INTERLEAVED ROUNDS, not four blocks. The first version of this job ran
# before-block then after-block then after-block then before-block, and that
# design failed exactly as a shared node makes it fail: on a quiet node
# (GPUCACE) the within-arm spread was ~1% and the signal was clean, while on a
# contended one (GPU389E, co-tenant array jobs arriving and leaving) the spread
# reached 7-34% and every verdict but two came back "within noise" -- with the
# two exceptions disagreeing with the quiet run. Neighbour load drifts on the
# timescale of a block, so a block design aliases it straight onto the arm.
#
# So the arms alternate per REPETITION, and the statistic is paired: each round
# runs before and after back to back on the same dataset, the ratio is taken
# WITHIN the round, and the reported figure is the median of those ratios plus a
# sign test over them. Drift that is slow compared to one round cancels, because
# it moves both members of a pair together. The within-pair order also flips on
# alternate rounds, so a systematic first-slot advantage cancels too.
#
# This cannot rescue a node so noisy that the effect is below round-to-round
# variation -- nothing can -- but it reports that honestly as a sign test near
# 50% rather than as a confident ratio.
#
# `PYTHONPATH` selects the arm. Each worktree gets its own
# `pyscx/python/pyscx/*.so` from its own `maturin develop`, and PYTHONPATH
# precedes the venv's editable install in `sys.path`, so pointing it at a
# worktree selects that build without touching the venv between arms. The
# `.venv`'s own editable install is restored at exit regardless of how the job
# ends.
#
# The harness is PINNED from the after worktree into the before worktree. W2
# edited `cellset_gather.py`'s prose (the arm's premise now describes a binary
# search rather than a HashSet build) and nothing else, so the two are
# functionally identical — but "functionally identical" is exactly the kind of
# claim that should not be load-bearing in a timing A/B.
#
# Datasets are pbmc3k and tabula_sapiens_100k, and census is NOT here: a
# `cellset_gather` census cell does not finish. census_500k was killed at 205
# minutes still on run 1 of 3 of `cache_undersized`, and census_1m/scx_fast at
# 355. That is recorded in thresholds.yaml's deferred block and is why the
# collate floors were prescribed on tabula alone.
#SBATCH --job-name=scx-phase1-loader-ab
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase1/ab_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase1/ab_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
# /tmp is node-local on Chimera and /home is near full; scratch lives on
# /large_storage, as every other capture in this directory does.
WORK=/large_storage/arcinfra/projects/scx/scratch/phase1
OUT="$WORK/ab_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

BEFORE_SHA=$(git -C "$REPO" rev-parse main)
AFTER_SHA=$(git -C "$REPO" rev-parse HEAD)

DATASETS="pbmc3k tabula_sapiens_100k"
FORMAT_KEY="scx_auto"
FORMAT_RUNNER="scx_runner"
# One timed run per invocation; the repetition that matters is ROUNDS, because
# only a round boundary alternates the arm. n_runs>1 would just make each
# block longer and each pair further apart in time.
N_RUNS=1
ROUNDS=${ROUNDS:-12}

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
# The gather arms hold ten batches resident; the cap the harness honours.
export SCX_BENCH_OOC_MEM_CAP_GB=${SCX_BENCH_OOC_MEM_CAP_GB:-48}

echo "=== phase 1: two-build loader A/B ==="
echo "host    : $(hostname)"
echo "before  : $BEFORE_SHA (main)"
echo "after   : $AFTER_SHA ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
echo "dirty   : $(git -C "$REPO" status --porcelain | grep -vc '^??' || true) tracked file(s) modified"
echo "datasets: $DATASETS"
echo "out     : $OUT"
nproc; free -g | head -2

# The venv's editable install points at whichever worktree built last. Put it
# back whatever happens, or the next interactive `import pyscx` in this repo
# silently resolves to a scratch worktree that this job is about to delete.
restore_dev_venv() {
    echo ""
    echo "=== restoring the dev venv from the primary checkout ==="
    ( cd "$REPO/pyscx" && VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release ) \
        >"$OUT/restore.log" 2>&1 || echo "WARNING: restore build failed, see restore.log"
    git -C "$REPO" worktree remove --force "$WORK/wt-before" 2>/dev/null || true
    git -C "$REPO" worktree remove --force "$WORK/wt-after" 2>/dev/null || true
}
trap restore_dev_venv EXIT

# ---------------------------------------------------------------------------
# Build both arms.
# ---------------------------------------------------------------------------
build_arm() {
    local name="$1" sha="$2"
    local wt="$WORK/wt-$name"
    local target="$WORK/target-$name"

    echo ""
    echo "=== building $name ($sha) ==="
    git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || true
    rm -rf "$wt"
    git -C "$REPO" worktree add --detach "$wt" "$sha"

    # Wholesale, never just `$target/maturin`: a reused target dir leaves a
    # 0-byte `libpyscx.so` that cargo re-links in 0.9s and maturin then dies on
    # with `Object is too small`.
    rm -rf "$target"

    # Pin the harness and the env into both worktrees, so the only difference
    # between the arms is the extension module.
    #
    # Code only. `benchmarks/comprehensive/{results,logs}` is 9.5 GB of prior
    # snapshots and would be copied twice for nothing; the worktree already has
    # the git-tracked manifest rows from its own checkout, and each arm writes
    # its results inside its own worktree, so the primary checkout's rows are
    # never touched by this job.
    rsync -a --delete \
        --exclude 'comprehensive/results/' \
        --exclude 'comprehensive/logs/' \
        --exclude 'logs/' \
        --exclude '__pycache__/' \
        "$REPO/benchmarks/" "$wt/benchmarks/"
    mkdir -p "$wt/benchmarks/comprehensive/results/raw"
    cp "$REPO/.env" "$wt/.env"

    ( cd "$wt/pyscx" && VIRTUAL_ENV="$VENV" CARGO_TARGET_DIR="$target" \
        "$VENV/bin/maturin" develop --release ) >"$OUT/build-$name.log" 2>&1
    ls -la "$wt"/pyscx/python/pyscx/pyscx.cpython-*.so
}

build_arm before "$BEFORE_SHA"
build_arm after  "$AFTER_SHA"

# ---------------------------------------------------------------------------
# Preflight: each arm's build is the one it claims to be.
#
# A capture is also a test of its own provenance. The failure this guards is
# not hypothetical: the phase-0 captures ran for hours against a wheel the
# conda env carried from an earlier branch A/B, same `__version__`, missing the
# counters under measurement.
# ---------------------------------------------------------------------------
cat > "$OUT/preflight.py" <<'PY'
import os, sys
import pyscx

wt = os.environ["ARM_WT"]
arm = os.environ["ARM_NAME"]
want = os.path.join(wt, "pyscx", "python") + os.sep
if not pyscx.__file__.startswith(want):
    sys.exit(f"preflight[{arm}]: pyscx resolved to {pyscx.__file__}, not {want}")

# W3's `gather` is the arm marker: it exists only on the after build, so it
# distinguishes the two `.so`s by capability rather than by path alone. A
# PYTHONPATH that silently failed to take would show up here.
has_gather = hasattr(pyscx.SparseCellSetDataset, "gather")
if arm == "after" and not has_gather:
    sys.exit("preflight[after]: this build has no SparseCellSetDataset.gather")
if arm == "before" and has_gather:
    sys.exit("preflight[before]: this build HAS gather — it is not main")

# The collate kernel is the subject; refuse to measure a build that cannot run
# it, rather than reporting an absent arm as a flat one.
if not hasattr(pyscx, "collate_cellset_gathered"):
    sys.exit(f"preflight[{arm}]: build predates the collate kernel")
v = getattr(pyscx, "COLLATE_CELLSET_CONTRACT_VERSION", None)
if v is None or int(v) < 2:
    sys.exit(f"preflight[{arm}]: collate contract {v}, need >= 2")

print(f"preflight[{arm}]: ok  {pyscx.__file__}  gather={has_gather}  contract=v{v}")
PY

# ---------------------------------------------------------------------------
# One arm = one in-process pass over the cells.
# ---------------------------------------------------------------------------
cat > "$OUT/run_one.py" <<'PY'
"""Run cellset_gather once, for one arm and one dataset, in process.

Emits one JSON line of the metrics the A/B compares, so the shell can
accumulate rounds without the two arms ever sharing a process — each has its
own pyscx extension module, so they cannot coexist in one interpreter.
"""
import json, os, pathlib, statistics, sys

from benchmarks.comprehensive.scripts.run_parallel import _run_benchmark

METRICS = (
    "us_per_cell__collate",
    "cellsets_per_sec__collate_rust",
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
)

ds = os.environ["ARM_DATASET"]
res = _run_benchmark(
    bench_name="cellset_gather",
    dataset_name=ds,
    format_key=os.environ["ARM_FORMAT_KEY"],
    format_runner=os.environ["ARM_FORMAT_RUNNER"],
    format_params={"codec": "auto"},
    n_runs=int(os.environ["ARM_N_RUNS"]),
    cold_cache=True,
    converted_path_str=None,
)
if res.get("skipped"):
    sys.exit(f"{ds} was skipped: {res}")

out = {"arm": os.environ["ARM_NAME"], "round": int(os.environ["ARM_ROUND"]),
       "dataset": ds}
for m in METRICS:
    vals = []
    for r in res.get("runs", []) or []:
        v = (r.get("extra") or {}).get(m)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    out[m] = statistics.median(vals) if vals else None
out["peak_rss_mb_median"] = res.get("peak_rss_mb_median")

pathlib.Path(os.environ["ARM_OUT"]).write_text(json.dumps(out))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in out.items()}), flush=True)
PY

run_one() {
    local name="$1" round="$2" ds="$3"
    local wt="$WORK/wt-$name"

    export ARM_NAME="$name" ARM_ROUND="$round" ARM_WT="$wt" ARM_DATASET="$ds"
    export ARM_OUT="$OUT/rounds/${ds}__r$(printf '%02d' "$round")__${name}.json"
    export ARM_FORMAT_KEY="$FORMAT_KEY" ARM_FORMAT_RUNNER="$FORMAT_RUNNER"
    export ARM_N_RUNS="$N_RUNS"
    # This worktree's pyscx AND this worktree's harness, ahead of everything.
    export PYTHONPATH="$wt/pyscx/python:$wt"

    ( cd "$wt" && set -a && . ./.env && set +a \
      && "$VENV/bin/python" "$OUT/preflight.py" >/dev/null \
      && "$VENV/bin/python" "$OUT/run_one.py" )
}

mkdir -p "$OUT/rounds"

# Preflight both builds ONCE and loudly before the timed rounds start. It also
# runs inside every invocation, silently, as a cheap guard against a PYTHONPATH
# that stops taking partway through a long job.
for arm in before after; do
    ARM_NAME="$arm" ARM_WT="$WORK/wt-$arm" \
        PYTHONPATH="$WORK/wt-$arm/pyscx/python:$WORK/wt-$arm" \
        "$VENV/bin/python" "$OUT/preflight.py"
done

for ds in $DATASETS; do
    echo ""
    echo "=== $ds: $ROUNDS interleaved rounds ==="
    for round in $(seq 1 "$ROUNDS"); do
        # Flip the within-pair order on alternate rounds, so a first-slot
        # advantage — page cache left warm by the sibling process, say —
        # cancels across the pair instead of accruing to one arm.
        if [ $((round % 2)) -eq 1 ]; then
            run_one before "$round" "$ds"
            run_one after  "$round" "$ds"
        else
            run_one after  "$round" "$ds"
            run_one before "$round" "$ds"
        fi
    done
done

echo ""
echo "=== done ==="
echo "rounds   : $OUT/rounds/"
echo "summarise: .venv/bin/python benchmarks/scripts/_phase1_loader_ab_summary.py $OUT"
