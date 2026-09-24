#!/bin/bash
# Phase 0 of the ML-loader plan — the per-regime data-wait fraction `p`.
#
# Tier-3 loader work (reuse-signal admission, the multi-set executor, derived
# token artifacts) is gated on `p`, the fraction of a training step spent
# waiting on data, because a decoder speedup `s` over an exposed fraction `p`
# is worth `1 / [(1 - p) + p/s]` and no more — 1.006x at the one `p` on record
# (STATE3, 0.006). This job produces `p` for two regimes:
#
#   R1  i.i.d. minibatches   `ml_loader`'s `gpu_train` scenario: a real scVI-
#                            equivalent VAE step against `TrainingDataset`.
#   R3  paired batches       `index_plan`'s `pyscx_index_plan_dataset_workers2`
#                            scenario, which has NO model step of its own, so
#                            `SCX_BENCH_R3_NULL_MODEL_MS` buys it a fixed one.
#                            Without that the fraction is ~1.0 by construction.
#
# R2 is consumer-side (STATE3) and is not measured here.
#
#   sbatch benchmarks/scripts/_run_phase0_data_wait_gate.sh
#
# ONE job: this is the orchestrator (`capture_baseline.py` submits the per-cell
# SLURM jobs itself and waits), and the two captures run back to back inside
# it. It must NOT rebuild the `.so` — `pyscx/python/pyscx/*.so` is
# cluster-global. Phase 0 changes no Rust, so there is nothing to rebuild.
#
# Two `--` details are load-bearing rather than tuning:
#   * `--skip-smoke` — the pre-submit runner smoke matches `--formats` against
#     the converter runners and refuses to submit when a narrowed run has none.
#   * `SCX_BENCH_GPU_PARTITION` — `ml_loader x scx_*` routes itself to the GPU
#     partition (`run_parallel._resolve_partition`), ignoring `--partition`,
#     and its default (`preemptible`) is the starved queue.
#
# The R3 capture is named separately and is deliberately NOT promoted: it runs
# with a null model the registered rows do not have, so its walls are not
# comparable to `LATEST`'s and must never be diffed against them.
#SBATCH --job-name=scx-phase0-datawait
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/large_storage/arcinfra/projects/scx/scratch/phase0/capture_%j.out
#SBATCH --error=/large_storage/arcinfra/projects/scx/scratch/phase0/capture_%j.out

set -euo pipefail

export REPO=/home/nickyoungblut/dev/rust/scx
# /tmp is node-local on Chimera, so the job's outputs must live on a shared
# filesystem. /large_storage rather than /home: /home is at 98% and a capture
# that fills it takes the login node down with it.
WORK=/large_storage/arcinfra/projects/scx/scratch/phase0
OUT="$WORK/run_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

cd "$REPO"
SHA=$(git rev-parse --short=7 HEAD)
NAME_R1="candidate_phase0_p_r1_${SHA}"
NAME_R3="candidate_phase0_p_r3_nullmodel_${SHA}"

# The orchestrator MUST run from `scx-bench`: `run_parallel` only activates a
# conda env on each worker when its own CONDA_PREFIX contains "scx-bench".
CONDA_BASE="$HOME/miniforge3"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate scx-bench
# Import pyscx from THIS checkout's editable build, not whatever the env holds
# (`scx-bench` carries a wheel from an earlier branch A/B).
# `run_parallel._slurm_setup_cmds` forwards PYTHONPATH to every worker, so this
# one line selects the build for the whole capture without touching the env.
export PYTHONPATH="$REPO/pyscx/python:$REPO${PYTHONPATH:+:$PYTHONPATH}"

# The bare `gpu` partition is the congested one; ask for the priority queues
# first. GPU cells ignore `--partition` / `SCX_BENCH_PARTITION` by design.
export SCX_BENCH_GPU_PARTITION="${SCX_BENCH_GPU_PARTITION:-gpu_high_mem,ctc_gpu_priority,gpu}"

DATASETS_R1="tabula_sapiens_100k census_1m"
DATASETS_R3="tabula_sapiens_100k"
CAPTURE=benchmarks/comprehensive/scripts/capture_baseline.py

echo "=== phase 0: per-regime data-wait fraction ==="
echo "host      : $(hostname)"
echo "commit    : $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "dirty     : $(git status --porcelain | grep -vc '^??' || true) tracked file(s) modified"
echo "python    : $(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "pyscx     : $(python -c 'import pyscx; print(pyscx.__file__)')"
echo "gpu part  : $SCX_BENCH_GPU_PARTITION"
echo "arms      : $NAME_R1 / $NAME_R3"
echo "out       : $OUT"
ls -la "$REPO"/pyscx/python/pyscx/pyscx.cpython-*.so

# ---------------------------------------------------------------------------
# Preflight: the instrumentation under measurement actually produces a number.
#
# A capture is also a test of its own diagnostics. The `gpu_train` emitter sits
# inside a broad `except Exception` that degrades a raise to zero runs, so a
# broken metric would come back as "GPU scenario unavailable" after hours on a
# GPU node. Exercise the same code path on CPU first, cheaply.
# ---------------------------------------------------------------------------
PREFLIGHT="$OUT/preflight.py"
cat > "$PREFLIGHT" <<'PY'
import json, os, sys
import numpy as np
import pyscx
from benchmarks.comprehensive.config import DATASETS
from benchmarks.comprehensive.data_wait import (
    data_wait_fraction, steady_state_wait, wait_percentiles,
)

expected_prefix = os.path.join(os.environ["REPO"], "pyscx", "python") + os.sep
if not pyscx.__file__.startswith(expected_prefix):
    sys.exit(f"preflight: pyscx resolved to {pyscx.__file__}, not this checkout's build")

# 1. The helper's contract: unmeasured is None, never 0.0 — the gate skips a
#    null metric but would median a zero as a real observation.
assert data_wait_fraction([], 1.0) is None, "unmeasured must be None"
assert data_wait_fraction([0.1], 0.0) is None, "no wall must be None"
assert abs(data_wait_fraction([0.25, 0.25], 1.0) - 0.5) < 1e-9
assert data_wait_fraction([2.0], 1.0) == 1.0, "fraction must clamp to 1"
assert all(v is None for v in wait_percentiles([]).values())

# 2. The real R1 loop shape, on CPU: drive `TrainingDataset` through an
#    explicit iterator and confirm a wait is measurable at all. If `next()`
#    always returns instantly on a warm tiny fixture the fraction is small but
#    it must still be a float, not None.
import time
path = str(DATASETS["pbmc3k"].path_for_format("scx_auto"))
ds = pyscx.TrainingDataset(path, batch_size=512, normalize=True, log1p=True, seed=0)
it = iter(ds)
waits, n = [], 0
t0 = time.perf_counter()
while True:
    w0 = time.perf_counter()
    try:
        b = next(it)
    except StopIteration:
        break
    waits.append(time.perf_counter() - w0)
    n += 1
wall = time.perf_counter() - t0
p = data_wait_fraction(waits, wall)
pct = wait_percentiles(waits)
st = steady_state_wait(waits, wall)
if n == 0:
    sys.exit("preflight: TrainingDataset yielded no batches on pbmc3k")
if p is None:
    sys.exit(f"preflight: {n} steps ran but the fraction is None")
if not (0.0 <= p <= 1.0):
    sys.exit(f"preflight: fraction {p} out of range")
# The headline `p` must exist and must be the startup-excluded one. A run that
# produced only the all-steps figure would publish a time-to-first-batch
# measurement under a heading that says "data-wait fraction".
if n > 1 and st["data_wait_fraction_steady"] is None:
    sys.exit(f"preflight: {n} steps ran but the steady-state fraction is None")
if st["ttfb_s"] is None:
    sys.exit("preflight: no time-to-first-batch recorded")
print(json.dumps({"n_steps": n, "wall_s": round(wall, 4),
                  "data_wait_fraction": round(p, 5),
                  "data_wait_fraction_steady": st["data_wait_fraction_steady"],
                  "ttfb_s": round(st["ttfb_s"], 4),
                  "n_steady_steps": st["n_steady_steps"],
                  **{f"batch_wait_{k}": v for k, v in pct.items()}}))

# 3. The R3 knob reads the env, and defaults off.
from benchmarks.comprehensive.benchmarks import index_plan as ip
os.environ.pop("SCX_BENCH_R3_NULL_MODEL_MS", None)
if ip._null_model_ms() != 0.0:
    sys.exit("preflight: R3 null model is not off by default")
os.environ["SCX_BENCH_R3_NULL_MODEL_MS"] = "5"
if ip._null_model_ms() != 5.0:
    sys.exit("preflight: R3 null model does not read its env var")
os.environ.pop("SCX_BENCH_R3_NULL_MODEL_MS", None)
print("preflight: R3 null-model knob ok")
PY
echo ""
echo "=== preflight ==="
python "$PREFLIGHT" | tee "$OUT/preflight.json"

# ---------------------------------------------------------------------------
# Dry run: the narrowed invocation resolves to the cells we expect.
# ---------------------------------------------------------------------------
echo ""
echo "=== dry run (R1) ==="
python "$CAPTURE" --mode dry-run --name "${NAME_R1}_dry" \
    --benchmarks ml_loader --formats scx_auto scx_fast --datasets $DATASETS_R1 \
    --skip-convert --skip-fingerprints --skip-smoke --no-accel 2>&1 \
    | tee "$OUT/dry_run_r1.log"
if grep -q "SKIP" "$OUT/dry_run_r1.log"; then
    echo "dry run reported a SKIP — the narrowing resolved away a cell" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# R1: `p` on the real training path. `ml_loader x scx_*` routes itself to a GPU
# node so the `gpu_train` scenario can fire.
# ---------------------------------------------------------------------------
echo ""
echo "=== R1: $NAME_R1 ==="
python "$CAPTURE" --name "$NAME_R1" \
    --benchmarks ml_loader --formats scx_auto scx_fast --datasets $DATASETS_R1 \
    --skip-convert --skip-fingerprints --skip-smoke --no-accel 2>&1 \
    | tee "$OUT/capture_r1.log"

# ---------------------------------------------------------------------------
# R3: `p` against a fixed-cost step. NOT promoted, and never diffed against
# LATEST: the null model is wall-clock this benchmark's registered rows do not
# contain, so every timing row here is incomparable by construction.
# ---------------------------------------------------------------------------
echo ""
echo "=== R3: $NAME_R3 (null model ${SCX_BENCH_R3_NULL_MODEL_MS:-25} ms/batch) ==="
SCX_BENCH_R3_NULL_MODEL_MS="${SCX_BENCH_R3_NULL_MODEL_MS:-25}" \
python "$CAPTURE" --name "$NAME_R3" \
    --benchmarks index_plan --formats scx_auto --datasets $DATASETS_R3 \
    --skip-convert --skip-fingerprints --skip-smoke --no-accel 2>&1 \
    | tee "$OUT/capture_r3.log"

# ⚠️ Restore the tracked `index_plan` manifest rows this arm just overwrote.
#
# Every benchmark writes to `results/raw/<triple>.json` before the snapshot is
# copied, and three `index_plan` rows there are **git-tracked** (force-added
# past .gitignore) because `docs/performance/loader-index-plan.md`'s OPT-FORMATIO-1 claims cite
# them. This arm runs with a 25 ms/batch null model, so its numbers are
# incomparable by construction — leaving them in place silently replaces a
# published 25.14 batches/s row with a 16.89 one whose slowdown is a `sleep`
# this job inserted. Observed: the first run of this job did exactly that.
#
# Narrow on purpose: only the tracked files, only under `results/raw/`, and
# only after the arm that contaminates them. The snapshot under
# `results/$NAME_R3/raw/` keeps the null-model numbers, which is where they
# belong.
echo "restoring tracked index_plan manifest rows clobbered by the null-model arm"
# NOT `2>/dev/null || true`: a restore that fails (wrong cwd, missing file,
# permissions) would leave the null-model numbers standing in a tracked,
# published row and let the job continue into the gate — the exact
# contamination this step exists to undo. `set -e` is in force, so a failure
# here stops the job.
git -C "$REPO" checkout -- \
    benchmarks/comprehensive/results/raw/index_plan__scx_auto__pbmc3k.json \
    benchmarks/comprehensive/results/raw/index_plan__scx_auto__smartseq2.json \
    benchmarks/comprehensive/results/raw/index_plan__scx_auto__tabula_sapiens_100k.json
if ! git -C "$REPO" diff --quiet -- \
        benchmarks/comprehensive/results/raw/index_plan__scx_auto__pbmc3k.json \
        benchmarks/comprehensive/results/raw/index_plan__scx_auto__smartseq2.json \
        benchmarks/comprehensive/results/raw/index_plan__scx_auto__tabula_sapiens_100k.json; then
    echo "tracked index_plan rows still differ from HEAD after restore" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Gate the R1 capture against LATEST. The instrumentation added two
# `perf_counter()` calls per batch to a timed epoch that carries
# `batches_per_sec__gpu_train` floors; this is the check that it cost nothing.
# ---------------------------------------------------------------------------
echo ""
echo "=== gate: $NAME_R1 vs LATEST ==="
set +e
python benchmarks/comprehensive/scripts/gate_candidate.py --skip-capture --name "$NAME_R1" \
    --benchmarks ml_loader --formats scx_auto scx_fast --datasets $DATASETS_R1 \
    --no-accel --no-gpu \
    -- --report-json "$OUT/gate_report.json" 2>&1 | tee "$OUT/gate.log"
GATE_RC=${PIPESTATUS[0]}
set -e
echo "gate exit: $GATE_RC"

# ---------------------------------------------------------------------------
# Summarise the `p` values out of the raw JSONs, so the docs table is
# transcribed from one artifact rather than from a log.
# ---------------------------------------------------------------------------
echo ""
echo "=== p summary ==="
python benchmarks/scripts/_phase0_data_wait_summary.py \
    "benchmarks/comprehensive/results/$NAME_R1" \
    "benchmarks/comprehensive/results/$NAME_R3" \
    --out "$OUT/p_summary.json" | tee "$OUT/p_summary.txt"

echo ""
echo "=== done (gate exit $GATE_RC) ==="
exit "$GATE_RC"
