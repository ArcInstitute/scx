#!/bin/bash
# Phase 3 of the ML-loader plan — tokenisation kernel set (W6).
#
#   sbatch benchmarks/scripts/_run_phase3_tokenize_ab.sh
#
# TWO THINGS AT ONCE, and they are different kinds of measurement.
#
# 1. `us_per_cell__collate` FLAT. The crop moved out of `collate_cell` into
#    `tokenize::crop::top_k` and the kernel now takes reusable scratch instead
#    of allocating three `Vec`s per cell. The crop extraction is meant to be
#    byte-identical — the cross-repo golden says it is — so the only thing that
#    can have moved is the wall, in either direction: the scratch removes
#    allocations, the call boundary adds one. **Flat is the expected result and
#    a move in either direction is a finding**, not a budget. Paired,
#    interleaved, sign-tested, exactly as phases 1 and 2.
#
# 2. THE FOUR NEW ARMS, captured for the first time. `us_per_cell__{crop,rank,
#    bin,sample}` exist only on the `after` build, so they have no ratio and are
#    NOT part of the A/B table. They are reported separately by
#    `_phase3_tokenize_summary.py`, twice per dataset per the phase's gate, and
#    **no floor is proposed from this capture** — thresholds.yaml item 22 says
#    why, and says what would have to be true to author one.
#
# Two of the four arms are fixture-gated and report themselves not applicable
# rather than producing a number over the wrong branch: `bin` needs at least two
# distinct non-zero values per row, `sample` needs median nnz/cell above its
# `n`. Whether pbmc3k clears the second is a measurement this job makes, not one
# it assumes.
#
# TWO BUILDS, NOT ONE: the change is unconditional code with no runtime kill
# switch, so one `.so` cannot produce both arms.
#
# INTERLEAVED ROUNDS AND A SIGN TEST, inherited from phase 1, where a block
# design on a contended node reported a 1.85x speedup as a 0.81x regression.
# Each round runs before and after back to back on the same dataset, the ratio
# is taken WITHIN the round, and the statistic is the median of those ratios
# plus an exact sign test. The within-pair order flips on alternate rounds.
#
# The harness is pinned from the after worktree into the before worktree, and
# the `.venv`'s editable install is restored on exit however the job ends.
#
# Datasets are pbmc3k and tabula_sapiens_100k. Census is absent for the reason
# recorded in thresholds.yaml's deferred block: a `cellset_gather` census cell
# does not finish (census_500k killed at 205 minutes on run 1 of 3).
#SBATCH --job-name=scx-phase3-tokenize-ab
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase3/ab_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase3/ab_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
# /tmp is node-local on Chimera and /home is near full; scratch lives on
# /large_storage, as every other capture in this directory does.
WORK=/large_storage/arcinfra/projects/scx/scratch/phase3
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

echo "=== phase 3: two-build tokenisation-kernel A/B ==="
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

# `pyscx.tokenize` is the arm marker: the namespace exists only on the after
# build, so the two `.so`s are distinguished by capability rather than by path
# alone. A PYTHONPATH that silently failed to take would show up here. Probed by
# CALLING a kernel, not by `hasattr`: a namespace that imports but whose
# functions are absent would pass the latter.
import numpy as np
has_tokenize = False
try:
    import pyscx.tokenize as _tok
    _out = _tok.top_k(
        np.array([0, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int32),
        np.array([5.0, 1.0], dtype=np.float32),
        2, 100,
    )
    has_tokenize = _out["ids"].tolist() == [1, 3]
except Exception:
    has_tokenize = False
if arm == "after" and not has_tokenize:
    sys.exit("preflight[after]: this build has no working pyscx.tokenize")
if arm == "before" and has_tokenize:
    sys.exit("preflight[before]: this build HAS pyscx.tokenize — it is not main")

# The collate kernel is the other subject; refuse to measure a build that cannot
# run it, rather than reporting an absent arm as a flat one. The contract
# version is also the second arm marker, and the two must agree: an `after`
# build at v2 would mean the PYTHONPATH took for the namespace but not the
# module, which is not a state this job should average over.
if not hasattr(pyscx, "collate_cellset_gathered"):
    sys.exit(f"preflight[{arm}]: build predates the collate kernel")
v = getattr(pyscx, "COLLATE_CELLSET_CONTRACT_VERSION", None)
if v is None or int(v) < 2:
    sys.exit(f"preflight[{arm}]: collate contract {v}, need >= 2")
want_v = 3 if arm == "after" else 2
if int(v) != want_v:
    sys.exit(f"preflight[{arm}]: collate contract v{v}, expected v{want_v} for this arm")

print(f"preflight[{arm}]: ok  {pyscx.__file__}  tokenize={has_tokenize}  contract=v{v}")
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

# The first six are paired A/B metrics: both builds emit them, so a ratio is
# defined. The last eight exist ONLY on the after build and therefore carry no
# ratio — `_phase1_loader_ab_summary.py` skips a metric with no complete pairs,
# which is the right behaviour for a ratio table, and
# `_phase3_tokenize_summary.py` reports them one-armed.
METRICS = (
    "us_per_cell__collate",
    "cellsets_per_sec__collate_rust",
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
    "us_per_cell__crop",
    "us_per_cell__rank",
    "us_per_cell__bin",
    "us_per_cell__sample",
    "cellsets_per_sec__tokenize_crop",
    "cellsets_per_sec__tokenize_rank",
    "cellsets_per_sec__tokenize_bin",
    "cellsets_per_sec__tokenize_sample",
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

# Which of the four kernel arms this fixture could actually exercise, and why
# not where it could not. An absent metric alone reads as "not captured"; this
# is what says "captured, and deliberately skipped, for this reason".
out["tokenize"] = (res.get("metadata") or {}).get("tokenize")

# Keep the FULL BenchmarkResult beside the reduced record. Each arm writes it
# inside its own scratch worktree, which the job removes on exit, so without
# this the committed evidence is metric scalars only and does not carry the
# schema_version / system / runs envelope docs/benchmark_manifest.md describes.
out["result"] = res

pathlib.Path(os.environ["ARM_OUT"]).write_text(json.dumps(out))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in out.items() if k != "result"}), flush=True)
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

# Provenance, written by the JOB rather than by hand afterwards.
#
# Phase 1's equivalent file was hand-written after the fact, which works exactly
# once: the facts live in the job's stdout, and a capture whose provenance is
# assembled by whoever remembers to is a capture that eventually ships without
# one. Everything here is read from the running job.
cat > "$OUT/provenance.json" <<JSON
{
 "job_id": "${SLURM_JOB_ID:-manual}",
 "host": "$(hostname)",
 "before_sha": "$BEFORE_SHA",
 "after_sha": "$AFTER_SHA",
 "before_ref": "main",
 "after_ref": "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)",
 "datasets": "$DATASETS",
 "format_key": "$FORMAT_KEY",
 "rounds": $ROUNDS,
 "n_runs_per_invocation": $N_RUNS,
 "rayon_num_threads": "${RAYON_NUM_THREADS}",
 "design": "Two builds, one host. Arms interleaved per round with the within-pair order flipped on alternate rounds; the statistic is the median of WITHIN-round ratios plus an exact two-sided sign test. A block design was tried in phase 1 and failed on a contended node, reporting a 1.85x speedup as a 0.81x regression.",
 "what_is_under_test": "Two things. (1) us_per_cell__collate FLAT: the top-K crop moved out of collate_cell into tokenize::crop::top_k and the kernel now takes reusable scratch instead of allocating three Vecs per cell. The crop is byte-identical (the cross-repo encoder_crop_golden.json is unchanged and green), so only the wall can have moved, in either direction — the scratch removes allocations, the call boundary adds one. Flat is the expected result and a move either way is a finding, not a budget. (2) The four per-kernel tokenisation arms, captured for the first time.",
 "what_this_is_not": "The four us_per_cell__{crop,rank,bin,sample} metrics exist ONLY on the after build and carry no ratio; they are absent from the A/B table by design and are reported one-armed by _phase3_tokenize_summary.py. NO floor is proposed from this capture — thresholds.yaml item 22 records what would have to be true to author one. Two of the four arms are fixture-gated (bin needs >= 2 distinct non-zero values per row, sample needs median nnz/cell above n=1024) and report themselves not applicable rather than producing a number over the wrong branch; the per-arm verdict is in each round record's tokenize block.",
 "records_are_reduced": "Each round is the median over the benchmark's own runs of the named metrics, not a full BenchmarkResult.to_dict() — but the full result is carried alongside under result, so the schema_version / system / runs envelope docs/benchmark_manifest.md describes is present.",
 "crop_premise_not_gated": "The crop arm does NOT assert that it truncates. Its cost is the sort over the row's positive entries, which runs whatever k is; k only decides how much of the output is PAD-init. At k=2048 against ~1950 nnz/cell on tabula it does not truncate at all, which Phase 1 measured. crop_fill_ratio is recorded instead so a reader can see how much of the wall is PAD-init."
}
JSON

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
echo "summarise (paired A/B):  .venv/bin/python benchmarks/scripts/_phase1_loader_ab_summary.py $OUT"
echo "summarise (new arms):    .venv/bin/python benchmarks/scripts/_phase3_tokenize_summary.py $OUT"
