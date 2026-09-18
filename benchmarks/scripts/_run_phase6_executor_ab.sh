#!/usr/bin/env bash
# Phase 6 (W11) — TWO-arm A/B for the multi-set batch executor, on one build.
#
# Arms, both one build, selected by `SCX_CELLSET_EXECUTOR`:
#   plan — shipped: one executor over the whole batch. A raw-local single-file
#          plan is read verbatim in plan order and moved out AS the batch; a
#          plan spanning files or carrying a remap/downsample is assembled from
#          a deduplicated read, each unique `(file, row)` transformed once.
#   set  — the pre-W11 walk: one read per set, a per-set
#          `Vec<Option<(Vec<i32>, Vec<f32>)>>`, an `extend_from_slice` copy of
#          every row, and no dedup at all
#
# ⚠️ Named `_executor_`, not `_gate_`: `_run_phase6_gate.sh` already exists and
# belongs to the scx-convert phase series, not to this one.
#
# ⚠️ What is already measured and is NOT what this job is for: the allocation
# count, by `scx-loader/tests/gather_allocation.rs` — 2,698 -> 81 for a
# 1,024-row plan, with peak live bytes DOWN as well. This job answers the question that test
# cannot: whether that shows up as throughput on a real fixture, and at what
# cost in peak RSS, on the scenarios that carry floors.
#
#SBATCH --job-name=scx-phase6-executor-ab
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase6/executor_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase6/executor_%j.out

set -euo pipefail

# ⚠️ Not derived from `${BASH_SOURCE[0]}`: sbatch copies the batch script into
# its own spool before running it, so `dirname $0/../..` resolves somewhere
# under the spool and would silently name the wrong tree rather than failing.
# Two scripts in this directory do derive it that way and are wrong for the
# same reason. An env override is the portable half without the hazard.
# (Review on #542, Antigravity - Gemini 3.8 Flash.)
REPO="${SCX_REPO:-/home/nickyoungblut/dev/rust/scx}"
VENV="$REPO/.venv"
WORK=/large_storage/arcinfra/projects/scx/scratch/phase6
OUT="$WORK/executor_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT/rounds"

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
ROUNDS=${ROUNDS:-12}
N_RUNS=1
ARMS="plan set"

# `benchmark|dataset|format_key|format_runner`.
#
# `cellset_gather` is the only benchmark this phase touches: `read_scattered`
# and `index_plan` drive `IndexPlanDataset`, whose gather W11 does not change.
# tabula first, deliberately — it carries four of the twelve live floors, both
# set sizes, and `gather_grouped_s512`, the one floored scenario that
# duplicates rows heavily (its `rng.choice(..., replace=g.size < S)` pads an
# under-full covariate group). census_500k is the second fixture and the
# schedule risk: phase 1's driver excluded census from `cellset_gather` after a
# census_500k cell was killed at 205 min on run 1 of 3, so it is chained as its
# own job rather than sharing this one's wall clock.
CELLS_DEFAULT="cellset_gather|tabula_sapiens_100k|scx_auto|scx_runner"
IFS=';' read -r -a CELLS <<< "${CELLS:-$CELLS_DEFAULT}"

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export SCX_BENCH_OOC_MEM_CAP_GB=${SCX_BENCH_OOC_MEM_CAP_GB:-64}
export PYTHONPATH="$REPO/pyscx/python:$REPO"

echo "=== phase 6: two-arm A/B (multi-set batch executor) ==="
echo "host   : $(hostname)"
echo "head   : $HEAD_SHA ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
echo "dirty  : $(git -C "$REPO" status --porcelain | grep -vc '^??' || true) tracked file(s) modified"
echo "arms   : $ARMS"
echo "rounds : $ROUNDS"
echo "cells  : ${CELLS[*]}"
echo "out    : $OUT"
nproc; free -g | head -2

cat > "$OUT/preflight.py" <<'PY'
"""Assert each arm BY BEHAVIOUR. One build, so a version string cannot tell the
arms apart even in principle — and unlike the W10 arms, the two executors are
byte-identical in their OUTPUT by construction, so the output cannot either.

What does differ is how many shard request groups a plan produces.
`block_index_groups` counts one per `(set, shard)` under the per-set walk and
one per `(plan, shard)` under the batch executor, so four single-shard sets read
as 4 against 1. A build that predates W11 answers 4 on both arms, which is what
makes this a build check as well as an arm check.
"""
import json, os, sys, tempfile

import numpy as np
import pyscx

arm = os.environ["ARM_NAME"]
print(f"pyscx: {pyscx.__file__}", file=sys.stderr)

import anndata as ad
import scipy.sparse as sp

N_OBS, N_VARS = 2_000, 400
N_SETS, SET_SIZE = 4, 2
X = sp.random(N_OBS, N_VARS, density=0.05, format="csr", random_state=0)
X.data = np.round(X.data * 10 + 1).astype(np.float32)
adata = ad.AnnData(X=X)
adata.obs["cell_id"] = [f"c{i}" for i in range(N_OBS)]
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "pf.scx")
    # One shard, framed, so every set names the same shard and the two arms
    # differ only in how many request groups they make of it.
    pyscx.from_anndata(adata, p, codec="shufdelta", row_group_rows=16,
                       shard_size=N_OBS)
    ds = pyscx.SparseCellSetDataset([p], cache_shards=4, scatter_block_index=True)
    rng = np.random.default_rng(11)
    rows = [int(rng.integers(0, N_OBS)) for _ in range(N_SETS * SET_SIZE)]
    # A row repeated across two sets, so the dedup has something to do.
    rows[SET_SIZE] = rows[0]
    plan = (
        [0] * len(rows),
        rows,
        [0] * len(rows),
        [s * SET_SIZE for s in range(N_SETS + 1)],
    )
    batch = ds.gather(*plan)
    cm = ds.cache_metrics()

if batch["shape"][0] != N_SETS * SET_SIZE:
    sys.exit(f"preflight gathered {batch['shape'][0]} rows, expected {N_SETS * SET_SIZE}")
if cm["block_index_groups"] <= 0:
    sys.exit(f"preflight never took the block-index route: {cm}")

want = 1 if arm == "plan" else N_SETS
if cm["block_index_groups"] != want:
    sys.exit(
        f"arm {arm!r}: block_index_groups={cm['block_index_groups']}, expected {want}. "
        "Either the arm did not reach this process (SCX_CELLSET_EXECUTOR is "
        "OnceLock-cached and must be exported before pyscx is imported), or the "
        f"build predates W11 and has only the per-set walk. cache_metrics={cm}"
    )

print(json.dumps({"arm": arm, "pyscx": pyscx.__file__, **cm}))
PY

cat > "$OUT/run_one.py" <<'PY'
"""One benchmark run, one arm, one cell, in process."""
import json, os, pathlib, statistics, sys

from benchmarks.comprehensive.scripts.run_parallel import _run_benchmark
from benchmarks.comprehensive.config import DATASETS

bench = os.environ["ARM_BENCH"]
ds = os.environ["ARM_DATASET"]
fk = os.environ["ARM_FORMAT_KEY"]
converted = str(DATASETS[ds].path_for_format(fk))

res = _run_benchmark(
    bench_name=bench,
    dataset_name=ds,
    format_key=fk,
    format_runner=os.environ["ARM_FORMAT_RUNNER"],
    format_params={"codec": "auto"},
    n_runs=int(os.environ["ARM_N_RUNS"]),
    cold_cache=True,
    converted_path_str=converted,
)
if res.get("skipped"):
    sys.exit(f"{bench}/{ds}/{fk} was skipped: {res}")

out = {"arm": os.environ["ARM_NAME"], "round": int(os.environ["ARM_ROUND"]),
       "bench": bench, "dataset": ds, "format": fk,
       "cell": f"{bench}|{ds}|{fk}"}

# Every numeric key any run emitted, reduced to its median -- per SCENARIO as
# well as overall, because one `cellset_gather` result carries four scenarios
# plus every arm block, and a metric named without its scenario is a median
# over things that are not comparable.
keys = set()
for r in res.get("runs", []) or []:
    for k, v in (r.get("extra") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keys.add(k)
for k in sorted(keys):
    vals = [float((r.get("extra") or {})[k]) for r in res["runs"]
            if isinstance((r.get("extra") or {}).get(k), (int, float))
            and not isinstance((r.get("extra") or {}).get(k), bool)]
    out[k] = statistics.median(vals) if vals else None
out["peak_rss_mb_median"] = res.get("peak_rss_mb_median")
out["result"] = res
pathlib.Path(os.environ["ARM_OUT"]).write_text(json.dumps(out))
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in out.items() if k != "result"})[:900], flush=True)
PY

# Plain `export` + `unset`, never a conditional command prefix — the phase-5
# driver died on `ARM_NAME=plan: command not found` writing it the other way.
# `unset` and not "set it to empty": the reader treats anything other than
# `set` as the default, so a stale value from a previous arm is how an arm
# comes to measure its neighbour's executor.
arm_env() {
    local name="$1"
    case "$name" in
        plan) unset SCX_CELLSET_EXECUTOR ;;
        set)  export SCX_CELLSET_EXECUTOR=set ;;
        *) echo "unknown arm $name" >&2; return 1 ;;
    esac
    export ARM_NAME="$name"
}

# `IFS` is set and RESTORED around the split, not left as an assignment prefix
# on `read`: a leaked `IFS='|'` stops `for arm in $ARMS` splitting on spaces.
split_cell() {
    local cell="$1" old_ifs="$IFS"
    IFS='|'
    # shellcheck disable=SC2086
    set -- $cell
    IFS="$old_ifs"
    CELL_BENCH="$1"; CELL_DS="$2"; CELL_FK="$3"; CELL_RUNNER="$4"
}

run_one() {
    local name="$1" round="$2" cell="$3"
    split_cell "$cell"
    arm_env "$name"
    export ARM_ROUND="$round"
    export ARM_BENCH="$CELL_BENCH" ARM_DATASET="$CELL_DS"
    export ARM_FORMAT_KEY="$CELL_FK" ARM_FORMAT_RUNNER="$CELL_RUNNER"
    export ARM_OUT="$OUT/rounds/${CELL_BENCH}__${CELL_DS}__${CELL_FK}__r$(printf '%02d' "$round")__${name}.json"
    export ARM_N_RUNS="$N_RUNS"
    ( cd "$REPO" && set -a && . ./.env && set +a \
      && "$VENV/bin/python" "$OUT/preflight.py" >/dev/null \
      && "$VENV/bin/python" "$OUT/run_one.py" )
}

# Preflight every arm once, loudly, before any timed round.
for arm in $ARMS; do
    arm_env "$arm"
    ( cd "$REPO" && set -a && . ./.env && set +a \
      && "$VENV/bin/python" "$OUT/preflight.py" )
done

for cell in "${CELLS[@]}"; do
    echo ""
    echo "=== $cell: $ROUNDS rounds x 2 arms ==="
    for round in $(seq 1 "$ROUNDS"); do
        # Alternate the arm order by round so neither arm takes every
        # cold-cache slot. Over 12 rounds each arm goes first six times.
        if [ $((round % 2)) -eq 1 ]; then order="plan set"; else order="set plan"; fi
        for arm in $order; do run_one "$arm" "$round" "$cell"; done
    done
done

"$VENV/bin/python" - <<PY
import json, os, pathlib
pathlib.Path("$OUT/provenance.json").write_text(json.dumps({
    "phase": 6, "what": "W11 two-arm A/B: multi-set batch executor vs the per-set walk",
    "head_sha": "$HEAD_SHA",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    "rounds": $ROUNDS, "n_runs": $N_RUNS,
    "arms": {
      "plan": ("shipped: whole-batch executor. Raw-local single-file plans are read "
               "verbatim and moved out as the batch; plans spanning files or carrying a "
               "remap/downsample are assembled from a deduplicated read."),
      "set":  "SCX_CELLSET_EXECUTOR=set (the pre-W11 per-set walk)",
    },
    "contrasts": {"executor": "plan vs set"},
    "cells": """${CELLS[*]}""".split(),
    "design": ("one build, two arms selected by environment; arm order alternated "
               "by round so neither takes every cold-cache slot; exact two-sided "
               "sign test over per-round ratios within a cell"),
    "arms_verified_by": ("behaviour, per arm, before every timed invocation: four "
                         "single-shard sets produce 4 block-index request groups "
                         "under the per-set walk and 1 under the batch executor"),
    "already_measured_elsewhere": ("the allocation count, by "
                                   "scx-loader/tests/gather_allocation.rs: 2,698 -> 81 "
                                   "for a 1,024-row plan, 2.63 -> 0.08 per row, peak "
                                   "live 421,192 -> 371,304 B"),
    "what_this_is_not": ("not a consumer-side measurement: R2's data-wait fraction "
                         "has never been measured, so what this buys a STATE3-class "
                         "trainer is not answered here"),
}, indent=1))
PY

echo ""
echo "=== done ==="
echo "rounds   : $OUT/rounds/"
echo "summarise: .venv/bin/python benchmarks/scripts/_phase6_executor_summary.py $OUT"
