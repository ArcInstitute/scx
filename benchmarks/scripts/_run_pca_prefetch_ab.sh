#!/bin/bash
# PCA decode-prefetch — before/after capture.
#
# Streaming CPU PCA's six shard loops now go through the bounded ordered
# decode-prefetch pipeline. At census_1m 4.2 measured `pca` at 179.9 s wall with
# 101.4 s (56 %) of it spent decoding one shard at a time on the calling thread
# — it was that A/B's *control*, precisely because its prefetch was deferred.
#
# One build, two environments — not two worktrees. The pipeline is disabled at
# runtime by `SCX_ACCEL_PREFETCH_DEPTH=1`, so the same `.so` serves both arms and
# the only difference is whether decode overlaps.
#
# **Why depth 1 is a genuine baseline here, and was not in #373.** With depth <= 1
# `for_each_ordered` takes a literal `for idx { read(idx); consume(idx, shard) }`
# — byte-for-byte the loop PCA ran before, with zero decode threads. #373's GPU
# staging A/B called depth 1 `main` when `main` had unconditionally spawned a
# one-ahead thread, which inflated its numbers. The distinction is checkable in
# the diff, and `pca::cpu::tests::depth_one_never_overlaps` pins it. The
# `memory_budget` reserve is also written as `depth - 1` shards precisely so this
# arm reserves nothing and stays byte-for-byte the old behaviour.
# `_run_pca_prefetch_main_vs_branch.sh` closes the premise empirically anyway.
#
# **Two controls, in opposite directions.** `qc` was wired for prefetch in 4.2,
# so the knob *must* move it (~2x at census scale) — if it does not, the knob is
# not binding and every other number here is void. `subset_obs` decodes no shards
# at all (fixed 50 % mask, open + rebuild only), so the knob must *not* move it —
# if it does, the host is too noisy to interpret.
#
#   sbatch benchmarks/scripts/_run_pca_prefetch_ab.sh
#
# Peak RSS is an acceptance number, not a footnote: the reserve predicts
# memory-neutrality against the sequential loop, and this is what confirms or
# refutes it. `profile_cpu_stages_backed.py` records `peak_rss_mb` per run, plus
# `n_shards` so the decode-bucket count reads as passes over the matrix.
#
# Sizing: the `cpu` partition runs QOS `cpu_interact`, capped at 64
# CPU-equivalents per user with memory billed at 4 GB = 1 CPU, so `--mem` binds
# before `--cpus-per-task`. 200G bills 50. PCA holds an 8 GiB decoded-shard LRU
# on top of the working matrices, which is why this asks for more than 4.2's 96G.
#SBATCH --job-name=scx-pca-prefetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-pca/capture_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-pca/capture_%j.out

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
VENV="$REPO/.venv"
# /tmp is node-local on Chimera; everything the job needs must live under /home.
WORK=/home/nickyoungblut/scx-bench-pca
OUT="$WORK/run_${SLURM_JOB_ID:-manual}"

SHA=$(cd "$REPO" && git rev-parse HEAD)

DATASETS="pbmc10k smartseq2 tabula_sapiens_100k census_500k census_1m"
# `pca` = randomized route (every fixture carries a full gene set).
# `pca_hvg` = covariance route, via a 2000-gene mask — the only op that reaches
# it, and the one that measures the `spmm_forward_into` swap.
# `qc` / `subset_obs` are the two controls described above.
OPS="pca pca_hvg qc subset_obs"
N_RUNS=3

# Pin the rayon pool so both arms are identical even if the cgroup's reported
# parallelism moves; recorded in each result's provenance. The prefetch depth is
# additionally capped by this value.
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

mkdir -p "$OUT"
echo "=== PCA decode-prefetch A/B capture ==="
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

# Preflight, before spending the allocation. Two checks, because they fail
# differently: the first catches a stale or broken `.so` in seconds, the second
# catches the failure mode this whole task is designed against — the pipeline
# silently declining to engage, which produces no speedup and no error.
echo ""
echo "=== preflight ==="
"$VENV/bin/python" - <<'PY' || { echo "PREFLIGHT FAILED: pyscx PCA is not runnable"; exit 1; }
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx, tempfile, pathlib
x = sp.random(200, 24, density=0.4, format="csr", dtype=np.float32, random_state=0)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(200)]
a.var_names = [f"g{i}" for i in range(24)]
p = pathlib.Path(tempfile.mkdtemp()) / "pf.scx"
pyscx.from_anndata(a, str(p), shard_size=25)
b = pyscx.open(str(p)).to_anndata(backed=True)
pyscx.accel.pca(b, n_comps=3, device="cpu")
assert np.isfinite(np.asarray(b.obsm["X_pca"])).all()
print("preflight: pca ok")
PY
( cd "$REPO" && CARGO_TARGET_DIR="$WORK/target-preflight" cargo test --release -p scx-accel --lib \
    pca::cpu::tests::whole_pca_ops_decode_shards_concurrently -- --nocapture ) \
    >"$OUT/preflight-gauge.log" 2>&1 \
    || { echo "PREFLIGHT FAILED: prefetch does not engage — see preflight-gauge.log"; exit 1; }
echo "preflight: decode-prefetch engages"

RAW_DIR="$REPO/benchmarks/comprehensive/results/raw"

# `results/raw/` accumulates across captures. The arms must not inherit each
# other's output, but blowing the directory away would destroy earlier captures,
# so quarantine what is already there and put it back on the way out.
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
echo "=== summary (median wall_s, off -> on) ==="
"$VENV/bin/python" - "$OUT" <<'PY'
import json, pathlib, statistics, sys
out = pathlib.Path(sys.argv[1])


def med(d, key):
    vals = [r.get(key) for r in d["runs"] if r.get(key) is not None]
    return statistics.median(vals) if vals else float("nan")


def load(arm):
    got = {}
    for f in sorted((out / f"raw-{arm}").glob("accel_cpu_profile_backed__*.json")):
        d = json.loads(f.read_text())
        got[(d["dataset"], d["format"])] = d
    return got


off, on = load("off"), load("on")
print(f"| dataset | op | off wall_s | on wall_s | speedup | off RSS MB | on RSS MB | RSS delta |")
print(f"|---|---|--:|--:|--:|--:|--:|--:|")
for key in sorted(off.keys() & on.keys()):
    ds, op = key
    a, b = med(off[key], "wall_s"), med(on[key], "wall_s")
    ra, rb = med(off[key], "peak_rss_mb"), med(on[key], "peak_rss_mb")
    print(
        f"| {ds} | {op} | {a:.2f} | {b:.2f} | {a / b if b else float('nan'):.2f}x "
        f"| {ra:.0f} | {rb:.0f} | {rb - ra:+.0f} |"
    )
PY

echo ""
echo "=== done; raw JSON under $OUT/raw-{off,on} ==="
