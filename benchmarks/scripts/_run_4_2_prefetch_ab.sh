#!/bin/bash
# Phase-4 task 4.2 — before/after capture for the relocated decode-prefetch.
#
# 4.2 routed four sets of shard loops that previously could not reach the
# Phase-2.1 pipeline through it: `scx-format-io`'s backed aggregations, pyscx's
# column-projected twins, pyscx's lazy/transformed streaming kernels, and
# `scx-gpu`'s staging path (captured separately, on a GPU node).
#
# Unlike the 4.0b capture this needs **one build, two environments**, not two
# worktrees: the pipeline is disabled at runtime by `SCX_ACCEL_PREFETCH_DEPTH=1`,
# which is the same A/B protocol 2.1 used. Same host, same `.so`, so the only
# difference is whether decode overlaps.
#
#   sbatch benchmarks/scripts/_run_4_2_prefetch_ab.sh
#
# Peak RSS is a first-class number here, not a footnote: decoded-but-unconsumed
# shards go from one to `depth` (default 4), and that is the one way this change
# can regress. `profile_cpu_stages_backed.py` records `peak_rss_mb` per run.
#
# Sizing follows the 4.0b harness: the `cpu` partition runs QOS `cpu_interact`,
# capped at 64 CPU-equivalents per user, with memory billing at 4 GB = 1 CPU.
# 96G/24c bills 24. The extra in-flight shards raise the ceiling over 4.0b's
# ~3.5 GB, but not near 96G.
#SBATCH --job-name=scx-4.2-prefetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.2/capture_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.2/capture_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
# /tmp is node-local on Chimera; everything the job needs must live under /home.
WORK=/home/nickyoungblut/scx-bench-4.2
OUT="$WORK/run_${SLURM_JOB_ID:-manual}"

SHA=$(cd "$REPO" && git rev-parse HEAD)

DATASETS="pbmc10k smartseq2 tabula_sapiens_100k census_500k census_1m"
# `qc` / `filter_genes*` / `filter_cells` are the ops whose kernels 4.2 rewired.
# `normalize` exercises the lazy/transformed streaming twins. `hvg` was already
# prefetched by 2.1 and `pca` is still a deferred path — both are controls and
# should come out flat; if either moves, something unintended changed.
OPS="qc filter_genes filter_genes_real filter_cells normalize hvg pca"
N_RUNS=3

# Pin the rayon pool so both arms are identical even if the cgroup's reported
# parallelism moves; recorded in each result's provenance. Note the prefetch
# depth is additionally capped by this value.
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

mkdir -p "$OUT"
echo "=== 4.2 decode-prefetch A/B capture ==="
echo "host       : $(hostname)"
echo "commit     : $SHA"
echo "datasets   : $DATASETS"
echo "ops        : $OPS"
echo "n_runs     : $N_RUNS"
echo "out        : $OUT"
nproc; free -g | head -2

echo ""
echo "=== building once (both arms share this .so) ==="
( cd "$REPO/pyscx" && VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release --features hdf5 ) \
    >"$OUT/build.log" 2>&1
ls -la "$REPO"/pyscx/python/pyscx/pyscx.cpython-*.so

RAW_DIR="$REPO/benchmarks/comprehensive/results/raw"

# `results/raw/` accumulates across captures — the 4.0b run left 41 files there.
# The arms must not inherit each other's output, but blowing the directory away
# would destroy earlier captures, so quarantine what is already there and put it
# back on the way out.
STASH="$WORK/preexisting_raw_${SLURM_JOB_ID:-manual}"
mkdir -p "$STASH"
mv "$RAW_DIR"/accel_cpu_profile_backed__*.json "$STASH/" 2>/dev/null || true
echo "quarantined $(ls "$STASH" | wc -l) pre-existing result files -> $STASH"

restore_preexisting() {
    echo "=== restoring pre-existing result files ==="
    cp -n "$STASH"/*.json "$RAW_DIR/" 2>/dev/null || true
}
trap restore_preexisting EXIT

run_arm() {
    local name="$1" depth="$2"

    echo ""
    echo "=== arm $name (SCX_ACCEL_PREFETCH_DEPTH=$depth) ==="
    ( cd "$REPO" \
      && set -a && . ./.env && set +a \
      && SCX_CPU_PROFILE=1 SCX_ACCEL_PREFETCH_DEPTH="$depth" \
         "$VENV/bin/python" benchmarks/scripts/profile_cpu_stages_backed.py \
            --datasets $DATASETS --ops $OPS --n-runs "$N_RUNS" ) \
        2>&1 | tee "$OUT/capture-$name.md"

    mkdir -p "$OUT/raw-$name"
    cp "$RAW_DIR"/accel_cpu_profile_backed__*.json "$OUT/raw-$name/" 2>/dev/null \
        || echo "WARNING: no raw JSON for $name"
    # This arm's files only — the quarantine above means nothing older is here.
    rm -f "$RAW_DIR"/accel_cpu_profile_backed__*.json
    echo "--- $name: $(ls "$OUT/raw-$name" | wc -l) result files ---"
}

# `off` first so the `on` arm cannot benefit from a warmer page cache.
run_arm off 1
run_arm on  4

echo ""
echo "=== done; raw JSON under $OUT/raw-{off,on} ==="
