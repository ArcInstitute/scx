#!/usr/bin/env bash
# Phase 5 (W10) — THREE-arm A/B isolating the phase's two independent factors,
# on one build.
#
# ⚠️ Why this exists. The first phase-5 capture (job 2964125) compared
# `SCX_ROW_GROUP_ADMIT=plan` against `reuse` on `cellset_gather` and reported
# "flat". Two things were wrong with that as a merge decision:
#
#   1. The chunked parallel group decode is NOT gated by
#      `SCX_ROW_GROUP_ADMIT`, so it was identical in both arms and cancelled
#      out of every ratio. The phase's biggest change had no number at all.
#   2. `cellset_gather`'s hot/cold arm caps its hit rate at 8/64 = 0.125 by
#      construction, and `index_plan` / `read_scattered` — the benchmarks whose
#      regime (R3) funded the phase — were never run. Measured since:
#      `read_scattered` on tabula sits at a 0.000 row-group hit rate under its
#      auto-tuned two-shard budget, which `thresholds.yaml` documents as the
#      pre-change design, and that is exactly the regime the admission policy
#      targets.
#
# Arms, all one build:
#   plan   — pre-W10 admission (all-or-nothing per plan), parallel decode ON
#   reuse  — shipped: reuse-signal admission, parallel decode ON
#   serial — shipped admission, `SCX_ROW_GROUP_DECODE_CHUNK=1` (one group per
#            chunk, i.e. the pre-change serial decode)
#
# `reuse` vs `plan` isolates the admission policy; `reuse` vs `serial` isolates
# the decode. Both from the same interleaved rounds, so neither is compared
# across jobs or nodes.
#
#SBATCH --job-name=scx-phase5-factors-ab
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase5/factors_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase5/factors_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
WORK=/large_storage/arcinfra/projects/scx/scratch/phase5
OUT="$WORK/factors_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT/rounds"

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
ROUNDS=${ROUNDS:-12}
N_RUNS=1
ARMS="plan reuse serial"

# `benchmark|dataset|format_key|format_runner` — the cells where the two
# factors are actually observable.
#   read_scattered × tabula g256   : the 0.000 admission regime (the decisive cell)
#   read_scattered × smartseq2 g256: a second witness of the same regime
#   index_plan     × tabula auto   : ~19k parallel group fetches per run, and
#                                    three floored scenarios, so it is where
#                                    the decode factor and the floors both live
CELLS=(
  "read_scattered|tabula_sapiens_100k|scx_compact_trial_g256|scx_runner"
  "read_scattered|smartseq2|scx_compact_trial_g256|scx_runner"
  "index_plan|tabula_sapiens_100k|scx_auto|scx_runner"
)

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export SCX_BENCH_OOC_MEM_CAP_GB=${SCX_BENCH_OOC_MEM_CAP_GB:-64}
export PYTHONPATH="$REPO/pyscx/python:$REPO"

echo "=== phase 5: three-arm factor A/B (admission x decode) ==="
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
arms apart even in principle."""
import json, os, sys, tempfile

import numpy as np
import pyscx

arm = os.environ["ARM_NAME"]
print(f"pyscx: {pyscx.__file__}", file=sys.stderr)

import anndata as ad
import scipy.sparse as sp

n_obs, n_vars = 20_000, 400
X = sp.random(n_obs, n_vars, density=0.05, format="csr", random_state=0)
X.data = np.round(X.data * 10 + 1).astype(np.float32)
adata = ad.AnnData(X=X)
adata.obs["cell_id"] = [f"c{i}" for i in range(n_obs)]
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "pf.scx")
    pyscx.from_anndata(adata, p, codec="shufdelta", row_group_rows=16, shard_size=2000)
    ds = pyscx.IndexPlanDataset(p, normalize=False, cache_shards=4, sort_by_shard=True,
                                scatter_block_index=True, lookahead=2,
                                max_memory_mb=60, max_plan_size=512)
    rng = np.random.default_rng(7)
    plans = []
    for _ in range(6):
        pl = [(c, int(rng.integers(0, n_obs))) for c in (5, 1007, 2009)]
        pl += [(int(rng.integers(0, n_obs)), int(rng.integers(0, n_obs))) for _ in range(200)]
        plans.append(pl)
    list(ds.iter_with_plans(iter([list(x) for x in plans]), lookahead=2))
    cm = ds.cache_metrics()

missing = [k for k in ("reuse_admissions", "parallel_group_decodes") if k not in cm]
if missing:
    sys.exit(f"build predates W10: cache_metrics() lacks {missing}")
if cm["block_index_groups"] <= 0:
    sys.exit(f"preflight never took the block-index route: {cm}")

if arm == "plan":
    if cm["reuse_admissions"] or cm["row_group_hits"]:
        sys.exit(f"arm 'plan' still retained: {cm}")
    if cm["parallel_group_decodes"] <= 0:
        sys.exit(f"arm 'plan' must keep the parallel decode: {cm}")
elif arm == "reuse":
    if not cm["reuse_admissions"] or not cm["row_group_hits"]:
        sys.exit(f"arm 'reuse' retained nothing: {cm}")
    if cm["parallel_group_decodes"] <= 0:
        sys.exit(f"arm 'reuse' must have the parallel decode: {cm}")
elif arm == "serial":
    if not cm["reuse_admissions"]:
        sys.exit(f"arm 'serial' must keep the reuse admission: {cm}")
    if cm["parallel_group_decodes"] != 0:
        sys.exit(f"arm 'serial' still decoded in parallel: {cm}")
else:
    sys.exit(f"unknown arm {arm!r}")

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

# `read_scattered` runs on the `additional`-tier trial fixtures, which the
# orchestrator does not convert — pass the path explicitly, as its own driver
# does.
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

# Every numeric key any run emitted, reduced to its median. The two factors
# move different metrics in different benchmarks, and hard-coding a list is how
# the first capture came to have no number for the decode change.
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

# Export the arm's environment into the CURRENT shell, for a subshell to
# inherit.
#
# ⚠️ Written as plain `export` + `unset`, never as a conditional command prefix
# (`${chunk:+VAR=$chunk} cmd`). The first version of this driver did that and
# died on `ARM_NAME=plan: command not found` — an assignment where a command
# name was expected. `unset` and not "set it to empty": the reader treats any
# value that is not a positive integer as absent, but leaving a stale value from
# a previous arm in the environment is how an arm comes to measure its
# neighbour's policy.
arm_env() {
    local name="$1"
    case "$name" in
        plan)   export SCX_ROW_GROUP_ADMIT=plan;  unset SCX_ROW_GROUP_DECODE_CHUNK ;;
        reuse)  export SCX_ROW_GROUP_ADMIT=reuse; unset SCX_ROW_GROUP_DECODE_CHUNK ;;
        serial) export SCX_ROW_GROUP_ADMIT=reuse; export SCX_ROW_GROUP_DECODE_CHUNK=1 ;;
        *) echo "unknown arm $name" >&2; return 1 ;;
    esac
    export ARM_NAME="$name"
}

# `IFS` is set and RESTORED around the split, not left as an assignment prefix
# on `read`: a leaked `IFS='|'` stops `for arm in $ARMS` splitting on spaces,
# which is the other half of what broke the first version.
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
    echo "=== $cell: $ROUNDS rounds x 3 arms ==="
    for round in $(seq 1 "$ROUNDS"); do
        # Rotate the arm order by round so each arm occupies each slot an equal
        # number of times over 12 rounds. A fixed order would give the first arm
        # every cold-cache advantage and the last every warm one.
        case $((round % 3)) in
            1) order="plan reuse serial" ;;
            2) order="reuse serial plan" ;;
            0) order="serial plan reuse" ;;
        esac
        for arm in $order; do run_one "$arm" "$round" "$cell"; done
    done
done

"$VENV/bin/python" - <<PY
import json, os, pathlib
pathlib.Path("$OUT/provenance.json").write_text(json.dumps({
    "phase": 5, "what": "W10 three-arm factor A/B: admission x parallel decode",
    "head_sha": "$HEAD_SHA",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    "rounds": $ROUNDS, "n_runs": $N_RUNS,
    "arms": {
      "plan":   "SCX_ROW_GROUP_ADMIT=plan (pre-W10 admission), parallel decode on",
      "reuse":  "shipped: reuse-signal admission, parallel decode on",
      "serial": "shipped admission, SCX_ROW_GROUP_DECODE_CHUNK=1 (pre-change serial decode)",
    },
    "contrasts": {
      "admission": "reuse vs plan",
      "parallel_decode": "reuse vs serial",
    },
    "cells": """${CELLS[*]}""".split(),
    "design": ("one build, three arms selected by environment; arm order rotated "
               "by round so each arm occupies each slot equally; exact two-sided "
               "sign test over per-round ratios within a cell"),
    "why_three_arms": ("the parallel decode is not gated by SCX_ROW_GROUP_ADMIT, "
                       "so a two-arm admission A/B holds it constant and cannot "
                       "measure it at all — which is what job 2964125 did"),
    "what_this_is_not": ("not a census measurement, and not a claim about "
                         "cellset_gather, whose own arm caps its hit rate at "
                         "8/64 by construction"),
    "arms_verified_by": "behaviour, per arm, before every timed invocation",
}, indent=1))
PY

echo ""
echo "=== done ==="
echo "rounds   : $OUT/rounds/"
echo "summarise: .venv/bin/python benchmarks/scripts/_phase5_factors_summary.py $OUT"
