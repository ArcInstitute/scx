#!/bin/bash
# Phase-4 task 4.0b — before/after capture for the axis-subset path.
#
# 4.0b hands the axis subset to anndata: `adata[idx]` -> `_mutated_copy` ->
# `_init_as_actual`. That builds a whole replacement AnnData — `uns` deep-copied,
# `obs` / `var` and every non-handle aligned member copied — and swaps it in, so
# for a moment both the old and the new frames are live. The acceptance question
# is whether that transient shows up at census scale, and whether the matrix is
# still never materialized.
#
# Both arms are built from isolated git worktrees on one host, so only the
# library differs. The harness is pinned too: the *current* profile script is
# copied into both worktrees, because the `filter_cells` / `filter_genes_real` /
# `subset_obs` ops it needs did not exist at the "before" commit.
#
# `subset_obs` is the sharp probe — a fixed 50 % mask, no threshold scan, so
# wall and peak RSS are open + rebuild and nothing else.
#
#   sbatch benchmarks/scripts/_run_4_0b_axis_subset_rss.sh
#
# Sizing: the `cpu` partition runs QOS `cpu_interact`, which caps a user at
# **64 CPU-equivalents**, and memory bills against it at `MaxMemPerCPU=4096`
# (4 GB = 1 CPU). So `--mem` is the binding constraint, not `--cpus-per-task`:
# 200G alone would bill 50 and queue behind any other job of mine. Census_1m
# peaks near 3.5 GB, so 96G is already ~25x headroom and bills the same 24 as
# the cores.
#SBATCH --job-name=scx-4.0b-rss
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.0b/capture_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.0b/capture_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
# /tmp is node-local on Chimera; everything the job needs must live under /home.
WORK=/home/nickyoungblut/scx-bench-4.0b
OUT="$WORK/run_${SLURM_JOB_ID:-manual}"

BEFORE_SHA=59783883
AFTER_SHA=$(cd "$REPO" && git rev-parse HEAD)

DATASETS="pbmc10k smartseq2 tabula_sapiens_100k census_500k census_1m"
OPS="filter_cells filter_genes_real subset_obs filter_genes"
N_RUNS=3

# Pin the rayon pool so the two arms are identical even if the cgroup's
# reported parallelism moves; recorded in each result's provenance.
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

mkdir -p "$OUT"
echo "=== 4.0b axis-subset capture ==="
echo "host       : $(hostname)"
echo "before     : $BEFORE_SHA"
echo "after      : $AFTER_SHA"
echo "datasets   : $DATASETS"
echo "ops        : $OPS"
echo "n_runs     : $N_RUNS"
echo "out        : $OUT"
nproc; free -g | head -2

# `maturin develop` repoints the *shared* editable install, so the two arms
# cannot run concurrently and the dev venv must be put back at the end however
# this exits.
restore_dev_venv() {
    echo "=== restoring the dev venv from the main worktree ==="
    cd "$REPO/pyscx" && VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release --features hdf5 \
        >"$OUT/restore.log" 2>&1 || echo "WARNING: restore build failed, see restore.log"
}
trap restore_dev_venv EXIT

run_arm() {
    local name="$1" sha="$2"
    local wt="$WORK/wt-$name"
    local target="$WORK/target-$name"

    echo ""
    echo "=== arm $name ($sha) ==="
    rm -rf "$wt"
    git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || true
    git -C "$REPO" worktree add --detach "$wt" "$sha"

    # Pin the harness: the "before" commit predates the ops being captured.
    cp "$REPO/benchmarks/scripts/profile_cpu_stages_backed.py" \
       "$wt/benchmarks/scripts/profile_cpu_stages_backed.py"
    cp "$REPO/.env" "$wt/.env"

    echo "--- building $name ---"
    ( cd "$wt/pyscx" && VIRTUAL_ENV="$VENV" CARGO_TARGET_DIR="$target" \
        "$VENV/bin/maturin" develop --release --features hdf5 ) >"$OUT/build-$name.log" 2>&1
    ls -la "$target/release/libpyscx.so"

    echo "--- capturing $name ---"
    ( cd "$wt" \
      && set -a && . ./.env && set +a \
      && SCX_CPU_PROFILE=1 "$VENV/bin/python" benchmarks/scripts/profile_cpu_stages_backed.py \
            --datasets $DATASETS --ops $OPS --n-runs "$N_RUNS" ) \
        2>&1 | tee "$OUT/capture-$name.md"

    mkdir -p "$OUT/raw-$name"
    cp "$wt"/benchmarks/comprehensive/results/raw/accel_cpu_profile_backed__*.json \
       "$OUT/raw-$name/" 2>/dev/null || echo "WARNING: no raw JSON for $name"
    echo "--- $name: $(ls "$OUT/raw-$name" | wc -l) result files ---"

    git -C "$REPO" worktree remove --force "$wt"
}

run_arm before "$BEFORE_SHA"
run_arm after  "$AFTER_SHA"

echo ""
echo "=== done; raw JSON under $OUT/raw-{before,after} ==="
