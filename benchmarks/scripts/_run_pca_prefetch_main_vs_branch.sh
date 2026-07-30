#!/bin/bash
# PCA decode-prefetch — two-build confirmation of the one-build A/B's premise.
#
# `_run_pca_prefetch_ab.sh` is the headline capture: one build, two environments,
# `SCX_ACCEL_PREFETCH_DEPTH=1` vs the default. Its whole validity rests on depth 1
# being what `main` actually did, and #373 is the reason that claim gets tested
# rather than asserted — there, the "off" arm had zero decode threads where
# `main` had one, and the published numbers were inflated until the review caught
# it.
#
# So: two worktrees, two builds, one host, `main` at its own defaults against the
# branch at its own defaults. If `main` and the depth-1 arm agree, the cheap
# capture is sound and is the one to quote. If they do not, the cheap capture is
# void and this job's numbers are the headline instead.
#
# Deliberately narrow — census_1m and `pca` / `pca_hvg` only. This exists to
# settle one premise, not to re-measure the tier.
#
#   sbatch --dependency=afterany:<ab-job-id> \
#          benchmarks/scripts/_run_pca_prefetch_main_vs_branch.sh
#
# The dependency is not optional: both jobs run `maturin develop` against the
# shared in-tree editable install, so a second job starting while the first is
# still importing pyscx swaps the library underneath it.
#
# Sizing as the A/B job — `--mem` binds at 4 GB = 1 CPU, so 200G bills 50 of the
# 64 CPU-equivalents QOS `cpu_interact` allows.
#SBATCH --job-name=scx-pca-2build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pca/twobuild_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pca/twobuild_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
WORK=/home/nickyoungblut/scx-bench-pca
OUT="$WORK/twobuild_${SLURM_JOB_ID:-manual}"

BEFORE_SHA=$(cd "$REPO" && git rev-parse main)
AFTER_SHA=$(cd "$REPO" && git rev-parse HEAD)

DATASETS="census_1m"
OPS="pca pca_hvg"
N_RUNS=3

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

mkdir -p "$OUT"
echo "=== PCA prefetch two-build confirmation ==="
echo "host       : $(hostname)"
echo "before     : $BEFORE_SHA (main)"
echo "after      : $AFTER_SHA"
echo "datasets   : $DATASETS"
echo "ops        : $OPS"
echo "out        : $OUT"
nproc; free -g | head -2

# `maturin develop` repoints the *shared* editable install, so the arms cannot
# run concurrently and the dev venv must be put back however this exits.
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

    # Remove the target dir **wholesale**. A reused one leaves a 0-byte
    # `libpyscx.so` hardlink; cargo reports "Finished in 0.9s", re-links it, and
    # maturin dies on `Object is too small`. Clearing only `$target/maturin` is
    # not enough — cargo re-links from `release/`.
    rm -rf "$target"

    # Pin the harness into both worktrees: `pca_hvg` does not exist at the
    # "before" commit. Benchmark-only, and permitted by docs/benchmark_manifest.md
    # — both arms will report `git_dirty`.
    cp "$REPO/benchmarks/scripts/profile_cpu_stages_backed.py" \
       "$wt/benchmarks/scripts/profile_cpu_stages_backed.py"
    cp "$REPO/.env" "$wt/.env"

    echo "--- building $name ---"
    ( cd "$wt/pyscx" && VIRTUAL_ENV="$VENV" CARGO_TARGET_DIR="$target" \
        "$VENV/bin/maturin" develop --release --features hdf5 ) >"$OUT/build-$name.log" 2>&1
    ls -la "$target/release/libpyscx.so"

    # Preflight this arm's build before spending its share of the allocation.
    "$VENV/bin/python" -c "import pyscx; print('preflight', '$name', pyscx.__file__)" \
        || { echo "PREFLIGHT FAILED for $name"; exit 1; }

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
echo "Compare the 'before' wall_s against the one-build capture's depth-1 arm."
echo "They should agree within host noise; if they do not, the one-build"
echo "headline is void and these numbers replace it."
