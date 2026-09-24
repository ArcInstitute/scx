#!/usr/bin/env bash
# Phase 5 (W10) — reuse-signal admission A/B, ONE build, two arms.
#
# ⚠️ **SUPERSEDED by `_run_phase5_factors_ab.sh`. Do not start a new capture
# here.** This driver holds the chunked parallel decode constant across both its
# arms — that change is not gated by `SCX_ROW_GROUP_ADMIT` — so it structurally
# cannot measure the larger of the phase's two factors, and reporting its result
# as the phase's verdict is the mistake job 2964125 produced. The factors driver
# runs three arms and isolates both.
#
# Kept, rather than deleted as review suggested, for one reason:
# `results/raw/phase5_admission/` is cited by name in `docs/performance/loader-index-plan.md`, and
# `docs/benchmark_manifest.md` requires a committed performance claim to have a
# reproducible producer. Deleting this script would leave that row unreproducible.
#
# `SCX_ROW_GROUP_ADMIT=plan` restores the pre-W10 all-or-nothing verdict and the
# unset default is `reuse`, so unlike phases 1-3 this needs no second worktree
# and no per-arm `maturin develop`: the arm is selected by an environment
# variable the reader caches in a `OnceLock` on first read. That is PR-25's
# mechanism.
#
# The DESIGN is phase 1's, not PR-25's. PR-25 ran one pass per arm back to back;
# phase 1 measured that a block design on a contended node can turn a 1.85x
# speedup into a 0.81x regression, so rounds are interleaved with the within-pair
# order flipped on alternate rounds and the summary is an exact two-sided sign
# test over per-round ratios.
#
# ⚠️ The arms MUST NOT share a process. Both knobs are `OnceLock`-cached, so a
# second arm in the same interpreter would silently measure the first one's
# policy — which is why `run_one` is a fresh `python` per invocation and why the
# preflight below distinguishes the arms BY BEHAVIOUR (does a hot-control plan
# sequence retain anything?) rather than by a version string.
#
# Datasets: pbmc3k and tabula_sapiens_100k. Census is NOT here for the reason
# phase 1 recorded — a `cellset_gather` census cell does not finish (census_500k
# killed at 205 minutes still on run 1 of 3). The census headroom figures in
# docs/performance/loader-index-plan.md came from `bench_cellset_scatter_routes.py`'s
# `cache_metrics()` probes, and that is the vehicle that can reproduce them.
#
#SBATCH --job-name=scx-phase5-admission-ab
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase5/ab_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase5/ab_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
WORK=/large_storage/arcinfra/projects/scx/scratch/phase5
OUT="$WORK/ab_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT/rounds"

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
DATASETS="pbmc3k tabula_sapiens_100k"
FORMAT_KEY="scx_auto"
FORMAT_RUNNER="scx_runner"
N_RUNS=1
ROUNDS=${ROUNDS:-12}

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export SCX_BENCH_OOC_MEM_CAP_GB=${SCX_BENCH_OOC_MEM_CAP_GB:-48}
export PYTHONPATH="$REPO/pyscx/python:$REPO"

echo "=== phase 5: same-build reuse-signal admission A/B ==="
echo "host    : $(hostname)"
echo "head    : $HEAD_SHA ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
echo "dirty   : $(git -C "$REPO" status --porcelain | grep -vc '^??' || true) tracked file(s) modified"
echo "datasets: $DATASETS"
echo "rounds  : $ROUNDS"
echo "out     : $OUT"
nproc; free -g | head -2

cat > "$OUT/preflight.py" <<'PY'
"""Assert the arm by BEHAVIOUR, then that the build can measure it at all.

A version string has been misleading twice in this repo (a stale editable `.so`,
a wheel from another branch), and here it cannot distinguish the arms even in
principle — they are one build. So the preflight drives a hot-control plan
sequence and checks what the policy did with it.
"""
import json, os, sys, tempfile

import numpy as np
import pyscx

arm = os.environ["ARM_ADMIT"]
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

# The four W10 keys must exist. The `scx-bench` conda env has carried a pyscx
# wheel reporting the right `__version__` while lacking `row_group_*` entirely,
# and a capture that silently measures a build without the counters reports the
# arms as identical.
missing = [k for k in ("reuse_admissions", "admitted_group_bytes",
                       "rejected_group_bytes", "parallel_group_decodes")
           if k not in cm]
if missing:
    sys.exit(f"build predates W10: cache_metrics() lacks {missing}")
if cm["block_index_groups"] <= 0:
    sys.exit(f"preflight never took the block-index route: {cm}")

if arm == "plan":
    if cm["reuse_admissions"] or cm["row_group_hits"] or cm["row_group_bytes_inserted"]:
        sys.exit(f"SCX_ROW_GROUP_ADMIT=plan still retained: {cm}")
else:
    if not cm["reuse_admissions"] or not cm["row_group_hits"]:
        sys.exit(f"SCX_ROW_GROUP_ADMIT=reuse retained nothing: {cm}")

print(json.dumps({"arm": arm, "pyscx": pyscx.__file__, **cm}))
PY

cat > "$OUT/run_one.py" <<'PY'
"""One `cellset_gather` run, one arm, one dataset, in process."""
import json, os, pathlib, statistics, sys

from benchmarks.comprehensive.scripts.run_parallel import _run_benchmark

# The new arm's metrics first, then every pre-existing `cellset_gather` metric:
# "nothing else moved" has to be measured on the same runs, not inferred.
METRICS = (
    "cellsets_per_sec__gather_hot_control_cold_tail",
    "us_per_cell__gather_hot_control_cold_tail",
    "row_group_hit_rate__gather_hot_control_cold_tail",
    "peak_rss_mb__gather_hot_control_cold_tail",
    "reuse_admissions__gather_hot_control_cold_tail",
    "admitted_group_bytes__gather_hot_control_cold_tail",
    "rejected_group_bytes__gather_hot_control_cold_tail",
    "cellsets_per_sec__gather_random",
    "cellsets_per_sec__gather_grouped",
    "cellsets_per_sec__gather_random_s512",
    "cellsets_per_sec__gather_grouped_s512",
    "us_per_cell__collate",
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
       "dataset": ds, "admit": os.environ["ARM_ADMIT"]}
for m in METRICS:
    vals = []
    for r in res.get("runs", []) or []:
        v = (r.get("extra") or {}).get(m)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    out[m] = statistics.median(vals) if vals else None
out["peak_rss_mb_median"] = res.get("peak_rss_mb_median")
# Why the arm skipped where it skipped, carried per round rather than inferred
# from a missing metric.
out["hot_meta"] = (res.get("metadata") or {}).get("hot_control_cold_tail")
out["result"] = res
pathlib.Path(os.environ["ARM_OUT"]).write_text(json.dumps(out))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in out.items() if k != "result"}), flush=True)
PY

run_one() {
    local name="$1" round="$2" ds="$3"
    export ARM_NAME="$name" ARM_ROUND="$round" ARM_DATASET="$ds"
    export ARM_ADMIT="$name"
    export ARM_OUT="$OUT/rounds/${ds}__r$(printf '%02d' "$round")__${name}.json"
    export ARM_FORMAT_KEY="$FORMAT_KEY" ARM_FORMAT_RUNNER="$FORMAT_RUNNER"
    export ARM_N_RUNS="$N_RUNS"
    ( cd "$REPO" && set -a && . ./.env && set +a \
      && SCX_ROW_GROUP_ADMIT="$name" "$VENV/bin/python" "$OUT/preflight.py" >/dev/null \
      && SCX_ROW_GROUP_ADMIT="$name" "$VENV/bin/python" "$OUT/run_one.py" )
}

for arm in plan reuse; do
    ( cd "$REPO" && set -a && . ./.env && set +a \
      && ARM_ADMIT="$arm" SCX_ROW_GROUP_ADMIT="$arm" "$VENV/bin/python" "$OUT/preflight.py" )
done

for ds in $DATASETS; do
    echo ""
    echo "=== $ds: $ROUNDS interleaved rounds ==="
    for round in $(seq 1 "$ROUNDS"); do
        if [ $((round % 2)) -eq 1 ]; then
            run_one plan  "$round" "$ds"
            run_one reuse "$round" "$ds"
        else
            run_one reuse "$round" "$ds"
            run_one plan  "$round" "$ds"
        fi
    done
done

# Provenance written from the RUNNING job, not from the #SBATCH directives —
# `sbatch --partition=` silently overrides them, which phase 4 was caught by.
"$VENV/bin/python" - <<PY
import json, os, pathlib, subprocess
pathlib.Path("$OUT/provenance.json").write_text(json.dumps({
    "phase": 5, "what": "W10 reuse-signal admission, same-build A/B",
    "head_sha": "$HEAD_SHA",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    "rounds": $ROUNDS, "n_runs": $N_RUNS,
    "datasets": "$DATASETS".split(),
    "design": ("one build, two arms selected by SCX_ROW_GROUP_ADMIT; rounds "
               "interleaved with the within-pair order flipped on alternate "
               "rounds; exact two-sided sign test over per-round ratios"),
    "what_this_is_not": ("not a census measurement (a cellset_gather census "
                         "cell does not finish), and not a claim about any "
                         "workload whose hot set is a different fraction of "
                         "its lookups than this arm's 8-of-64"),
    "arms_verified_by": "behaviour (a hot-control plan sequence must retain on reuse and not on plan)",
}, indent=1))
PY

echo ""
echo "=== done ==="
echo "rounds   : $OUT/rounds/"
echo "summarise: .venv/bin/python benchmarks/scripts/_phase5_admission_summary.py $OUT"
