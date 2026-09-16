#!/usr/bin/env bash
# Phase 4 capture: the two neighbourhood-plan arms on the spatial fixture.
#
# ⚠️ ONE-ARMED, not the paired A/B phases 1-3 ran, and that is not a shortcut.
# `gather_neighborhood_graph` / `_coords` do not exist on `main` — there is no
# "before" build that emits them, so there is no ratio to take and no sign test
# to run. A two-build job here would spend half its wall producing a column of
# `None`. What the two builds WOULD have told us — that nothing else regressed —
# is covered instead by the four pre-existing `cellsets_per_sec__gather_*`
# metrics, which this job records on the same runs: they come from the same
# module and the same fixture, so a change in the shared gather path shows up in
# them.
#
# The arms are captured TWICE (ROUNDS=2 by default) before any floor is
# proposed, per the phase gate. `thresholds.yaml` item 23 records what would
# have to be true to author one, and why the answer is not simply "read the
# median".
#
# ⚠️ ONE DATASET, and no other can substitute. `visium_lymph_node` is the only
# fixture in the registry with an `obsm` or an `obsp` at all — every other
# dataset's arms self-skip with a named reason rather than producing a number.
# It is also deliberately outside every `capture_baseline.TIERS` list, which is
# why this driver exists rather than the arms riding a default gate run.
#
# ⚠️ Read the rates together with `metadata.neighborhood.arms.<arm>.locality`.
# A Visium file's obs order is barcode order, NOT spatial: a 7-cell
# neighbourhood spans a median of ~3,000 row indices out of 4,035 and every
# batch touches every shard. These numbers are the pessimal scattered-read case;
# a spatially sorted file is the other extreme and is not measured here.
#
#SBATCH --job-name=scx-phase4-neighborhood
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase4/nb_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase4/nb_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
# /tmp is node-local on Chimera and /home is near full; scratch lives on
# /large_storage, as every other capture in this directory does.
WORK=/large_storage/arcinfra/projects/scx/scratch/phase4
OUT="$WORK/nb_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT/rounds"

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)

DATASETS="visium_lymph_node"
FORMAT_KEY="scx_auto"
FORMAT_RUNNER="scx_runner"
N_RUNS=${N_RUNS:-3}
ROUNDS=${ROUNDS:-8}

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export SCX_BENCH_OOC_MEM_CAP_GB=${SCX_BENCH_OOC_MEM_CAP_GB:-48}

echo "=== phase 4: neighbourhood plan arms (one-armed) ==="
echo "host    : $(hostname)"
echo "part    : ${SLURM_JOB_PARTITION:-unknown}"
echo "head    : $HEAD_SHA ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
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
    git -C "$REPO" worktree remove --force "$WORK/wt-head" 2>/dev/null || true
}
trap restore_dev_venv EXIT

# ---------------------------------------------------------------------------
# Build the one arm.
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

    # Pin the harness and the env into the worktree. Code only —
    # `benchmarks/comprehensive/{results,logs}` is ~9.5 GB of prior snapshots.
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

build_arm head "$HEAD_SHA"

# ---------------------------------------------------------------------------
# Preflight: assert the CAPABILITY, not the path.
# ---------------------------------------------------------------------------
cat > "$OUT/preflight.py" <<'PY'
"""Refuse to time anything until the build under test can do the thing.

Checks the capability by CALLING it, not by importing a name: a build whose
plan builder is present but broken would pass an `hasattr` probe and then
produce a number for a gather over an empty plan list.
"""
import os
import sys

import numpy as np
import pyscx

wt = os.environ["ARM_WT"]
if not pyscx.__file__.startswith(wt):
    sys.exit(f"pyscx resolved to {pyscx.__file__}, not inside {wt}")

for name in ("neighborhood_plans_from_graph", "neighborhood_plans_from_coords",
             "batch_plans"):
    if not hasattr(pyscx, name):
        sys.exit(f"this build has no pyscx.{name}")

path = os.environ["ARM_FIXTURE"]
exp = pyscx.open(path)
if "connectivities" not in exp.obsp_keys():
    sys.exit(f"{path} has no obsp['connectivities'] (has {exp.obsp_keys()})")
if "spatial" not in exp.obsm_keys():
    sys.exit(f"{path} has no obsm['spatial'] (has {exp.obsm_keys()})")

# Call both builders and check the shape of what comes back, so a builder that
# returns an empty list cannot pass.
g_plans, g_centers = pyscx.neighborhood_plans_from_graph(path, k=6, file_id=0)
c_plans, c_centers = pyscx.neighborhood_plans_from_coords(path, k=6, file_id=0)
for label, plans, centers in (("graph", g_plans, g_centers),
                              ("coords", c_plans, c_centers)):
    if len(plans) != exp.n_obs or len(centers) != exp.n_obs:
        sys.exit(f"{label}: got {len(plans)} plans for {exp.n_obs} cells")
    rows = np.asarray(plans[0][1])
    tags = np.asarray(plans[0][2])
    if rows.size < 2 or int(tags[0]) != 0 or not (tags[1:] == 1).all():
        sys.exit(f"{label}: first plan is not a role-tagged neighbourhood: "
                 f"rows={rows} tags={tags}")

# And the bounded obsp read, which the arms' premise probe depends on.
block = exp.read_obsp_rows("connectivities", 0, min(exp.n_obs, 512))
if block.nnz == 0:
    sys.exit("read_obsp_rows returned an empty block")

print(f"preflight OK: {pyscx.__file__}")
print(f"  {exp.n_obs} cells, graph nnz/row in first block = "
      f"{block.nnz / block.shape[0]:.2f}")
PY

# ---------------------------------------------------------------------------
# One timed invocation.
# ---------------------------------------------------------------------------
cat > "$OUT/run_one.py" <<'PY'
"""Run cellset_gather once and reduce it to the metrics this capture is about.

Emits one JSON line plus a file, so the shell can accumulate rounds without the
benchmark ever sharing a process between rounds.
"""
import json, os, pathlib, statistics, sys

from benchmarks.comprehensive.scripts.run_parallel import _run_benchmark

# The first six are the NEW arms; the last four are the pre-existing gather
# scenarios, recorded on the same runs so that "nothing else moved" is a
# measurement on this fixture rather than an inference from another one.
METRICS = (
    "cellsets_per_sec__gather_neighborhood_graph",
    "cellsets_per_sec__gather_neighborhood_coords",
    "us_per_cell__neighborhood_graph",
    "us_per_cell__neighborhood_coords",
    "plan_build_s__neighborhood_graph",
    "plan_build_s__neighborhood_coords",
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
# Per ARM, from the runs. `res["peak_rss_mb_median"]` is None on this
# benchmark — it pools every scenario, which for a module with a dozen arms is
# not a number about any of them. The neighbourhood arms' own RSS turned out to
# be the most informative field in the first capture: it is what distinguished a
# fast round from a slow one.
for arm in ("graph", "coords"):
    scenario = f"gather_neighborhood_{arm}"
    vals = [
        float(r["peak_rss_mb"])
        for r in (res.get("runs") or [])
        if (r.get("extra") or {}).get("scenario") == scenario
        and isinstance(r.get("peak_rss_mb"), (int, float))
    ]
    out[f"peak_rss_mb__neighborhood_{arm}"] = statistics.median(vals) if vals else None
out["peak_rss_mb_median"] = res.get("peak_rss_mb_median")

# Which arms the fixture could exercise, why not where it could not, and the
# measured locality of the plans. An absent metric alone reads as "not
# captured"; this is what says "captured, over sets of THIS shape".
out["neighborhood"] = (res.get("metadata") or {}).get("neighborhood")

# Keep the FULL BenchmarkResult beside the reduced record. The arm writes it
# inside a scratch worktree the job removes on exit, so without this the
# committed evidence is metric scalars only and does not carry the
# schema_version / system / runs envelope docs/benchmark_manifest.md describes.
out["result"] = res

pathlib.Path(os.environ["ARM_OUT"]).write_text(json.dumps(out))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in out.items() if k not in ("result", "neighborhood")}),
      flush=True)
PY

run_one() {
    local name="$1" round="$2" ds="$3"
    local wt="$WORK/wt-$name"

    export ARM_NAME="$name" ARM_ROUND="$round" ARM_WT="$wt" ARM_DATASET="$ds"
    export ARM_OUT="$OUT/rounds/${ds}__r$(printf '%02d' "$round")__${name}.json"
    export ARM_FORMAT_KEY="$FORMAT_KEY" ARM_FORMAT_RUNNER="$FORMAT_RUNNER"
    export ARM_N_RUNS="$N_RUNS"
    export PYTHONPATH="$wt/pyscx/python:$wt"

    ( cd "$wt" && set -a && . ./.env && set +a \
      && ARM_FIXTURE="$FIXTURE" "$VENV/bin/python" "$OUT/preflight.py" >/dev/null \
      && "$VENV/bin/python" "$OUT/run_one.py" )
}

# Resolve the fixture path from the registry rather than hard-coding it, so a
# moved SCX_DATA_DIR is a failure here and not a mystery later.
FIXTURE=$(cd "$WORK/wt-head" && set -a && . ./.env && set +a && \
    PYTHONPATH="$WORK/wt-head" "$VENV/bin/python" -c \
    "from benchmarks.comprehensive.config import DATASETS; print(DATASETS['visium_lymph_node'].scx_auto_path)")
echo "fixture : $FIXTURE"

# Provenance, written by the JOB rather than by hand afterwards. Phase 1's
# equivalent file was hand-written after the fact, which works exactly once.
cat > "$OUT/provenance.json" <<JSON
{
 "job_id": "${SLURM_JOB_ID:-manual}",
 "host": "$(hostname)",
 "partition": "${SLURM_JOB_PARTITION:-unknown}",
 "partition_note": "Read from the running job, not from the #SBATCH directive, because sbatch --partition= overrides it. The directive asks for cpu_batch (what phases 1-3 used); a capture that landed elsewhere says so here rather than being compared across hardware classes by someone who assumed.",
 "head_sha": "$HEAD_SHA",
 "head_ref": "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)",
 "datasets": "$DATASETS",
 "fixture": "$FIXTURE",
 "format_key": "$FORMAT_KEY",
 "rounds": $ROUNDS,
 "rounds_note": "Eight, not the two the phase gate asks for, because two could not resolve these arms: the first capture (job 2959318) measured 9,597 vs 13,387 sets/s on the graph arm between two rounds while the three runs WITHIN each round agreed to ~3%, and peak RSS tracked the difference (1,580 MB in the slow round, 2,045 MB in the fast one). The variation is between processes, not within one.",
 "n_runs_per_invocation": $N_RUNS,
 "rayon_num_threads": "${RAYON_NUM_THREADS}",
 "design": "ONE-ARMED. The two neighbourhood arms do not exist on main, so there is no before build to interleave against and no ratio to take. Two rounds on one host, three timed runs per round, cold page cache per run. The four pre-existing gather_* metrics are recorded on the same runs so that 'nothing else moved on this fixture' is measured rather than inferred.",
 "what_is_under_test": "gather_neighborhood_graph and gather_neighborhood_coords: the cost of building neighbourhood plans (plan_build_s, timed apart from the gather) and of gathering what they produce, at k=6 and 146 sets per batch on the only fixture in the registry with an obsm or an obsp.",
 "what_this_is_not": "NOT a floor. thresholds.yaml item 23 records the three separate reasons one cannot be authored from this capture, one of which is that this dataset is outside every capture_baseline tier. NOT a locality result either: a Visium file's obs order is barcode order, so a neighbourhood spans ~3000 of 4035 row indices and every batch touches every shard. These are the pessimal scattered-read numbers; a spatially sorted file is the other extreme and is not measured. NOT a scale result: 4,035 spots, not Xenium's 10^5, and plan_build_s is the term that would move most at that scale.",
 "records_are_reduced": "Each round is the median over the benchmark's own runs of the named metrics, not a full BenchmarkResult.to_dict() — but the full result is carried alongside under result, so the schema_version / system / runs envelope docs/benchmark_manifest.md describes is present.",
 "premises_gated_in_the_arm": "The arm raises rather than reporting a number when the sets are not neighbourhoods (median set radius above 10% of the fixture's coordinate extent) or when the coordinate extent is degenerate; it SKIPS with a named reason when the fixture has no graph or no coordinates, or when mean degree is below 2. Set overlap is recorded and NOT gated: measured 1.1023 in centre order, 1.1084 shuffled, random-plan control of the same set size and batch width 1.1286 — random sets duplicate rows more than neighbourhoods do, so a bar there would have passed on random plans."
}
JSON

echo ""
echo "=== preflight ==="
ARM_NAME=head ARM_WT="$WORK/wt-head" ARM_FIXTURE="$FIXTURE" \
    PYTHONPATH="$WORK/wt-head/pyscx/python:$WORK/wt-head" \
    "$VENV/bin/python" "$OUT/preflight.py"

for ds in $DATASETS; do
    echo ""
    echo "=== $ds: $ROUNDS rounds x $N_RUNS runs ==="
    for round in $(seq 1 "$ROUNDS"); do
        run_one head "$round" "$ds"
    done
done

echo ""
echo "=== done ==="
echo "rounds   : $OUT/rounds/"
echo "summarise: .venv/bin/python benchmarks/scripts/_phase4_neighborhood_summary.py $OUT"
