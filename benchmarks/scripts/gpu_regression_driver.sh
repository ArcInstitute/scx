#!/bin/bash
# ============================================================================
# DEPRECATED as of Phase 9 (2026-04-23). Use the comprehensive framework:
#
#     bash benchmarks/comprehensive/scripts/gate_candidate.sh
#
# That gates the current head against the canonical baseline at
# benchmarks/comprehensive/results/baselines/LATEST → v0.6.0-gpu-phase1-7,
# which now covers both format-level AND accelerator benchmarks. This
# script remains in-tree for one release for rollback convenience; future
# regression runs should use gate_candidate.sh instead.
# ============================================================================
#
# gpu_regression_driver.sh — orchestrate GPU accelerator regression bench +
# diff against a pre-change baseline.  Implements Phase 8 tasks 8.2–8.6 of
# GPU-ACC-SPEED-UP.md in a single idempotent invocation.
#
# Must run on a node with SLURM submit access — not the login node.  On this
# cluster, interactive dev work already runs inside an srun-allocated worker
# node, so just invoke this script directly.  For long runs that need to
# survive disconnects, wrap this driver in its own sbatch submission instead.
# Can run the benches inline (if already on a GPU node) or submit each GPU
# benchmark as an individual SLURM job.
#
# Usage:
#
#   # Full run: preflight + tests + SLURM submit + wait + diff.
#   bash benchmarks/scripts/gpu_regression_driver.sh
#
#   # Skip the cargo/pytest preflight (already ran).
#   bash benchmarks/scripts/gpu_regression_driver.sh --skip-tests
#
#   # Diff only: existing post/ already populated, just run the diff.
#   bash benchmarks/scripts/gpu_regression_driver.sh --diff-only
#
#   # Different pre baseline.
#   bash benchmarks/scripts/gpu_regression_driver.sh \
#       --pre benchmarks/results/pre_v0.6_2026_06
#
#   # Inline (no sbatch) on a node that already has a GPU — useful for
#   # quick iteration on small datasets.  Datasets hard to size this way;
#   # prefer --submit for anything ≥ 100K cells.
#   bash benchmarks/scripts/gpu_regression_driver.sh --inline
#
# Exit codes:
#   0   pass  (all benchmarks within tolerance vs baseline)
#   1   fail  (regression or hard-floor flag — see report)
#   2   infra (missing inputs, SLURM errors, build failures)

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

PRE_DEFAULT="benchmarks/results/pre_phases_1_7_baseline_2026_03"
POST="benchmarks/results"
PRE="$PRE_DEFAULT"
SKIP_TESTS=0
DIFF_ONLY=0
INLINE=0
DATASET_FOR_PIPELINE="${DATASET_FOR_PIPELINE:-census_1m}"
CONDA_PREFIX_DEFAULT="/home/nickyoungblut/miniforge3/envs/scx-gpu"
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-$CONDA_PREFIX_DEFAULT}"

# ── Argument parsing ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pre)          PRE="$2"; shift 2;;
        --post)         POST="$2"; shift 2;;
        --skip-tests)   SKIP_TESTS=1; shift;;
        --diff-only)    DIFF_ONLY=1; shift;;
        --inline)       INLINE=1; shift;;
        --dataset)      DATASET_FOR_PIPELINE="$2"; shift 2;;
        --conda-prefix) CONDA_PREFIX="$2"; shift 2;;
        -h|--help)
            sed -n '2,33p' "$0"; exit 0;;
        *)
            echo "unknown flag: $1" >&2; exit 2;;
    esac
done

[[ -d "$PRE" ]]  || { echo "error: --pre '$PRE' is not a directory" >&2; exit 2; }
[[ -d "$POST" ]] || { echo "error: --post '$POST' is not a directory" >&2; exit 2; }

mkdir -p benchmarks/logs "$POST"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="benchmarks/logs/gpu_regression_driver_${STAMP}.log"
echo "gpu_regression_driver.sh — log: $LOG"

# Shell `tee` to capture everything below to the log.
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "GPU accelerator regression driver"
echo "============================================================"
echo "repo:    $REPO"
echo "pre:     $PRE"
echo "post:    $POST"
echo "dataset: $DATASET_FOR_PIPELINE"
echo "mode:    $([[ $INLINE -eq 1 ]] && echo inline || echo sbatch)"
echo "started: $(date -Iseconds)"
echo ""

# ── Detect GPU locally (informational; the sbatch workers bring their own) ──
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    echo "(no nvidia-smi on this host — using sbatch for all GPU work)"
fi

if [[ $DIFF_ONLY -eq 1 ]]; then
    echo "--diff-only: skipping build + tests + sbatch, running diff only."
else
    # ── Bind the conda env for every downstream subprocess ─────────────
    if [[ -d "$CONDA_PREFIX" ]]; then
        # Export CONDA_PREFIX itself — cupy's CUDA-path detection and some
        # RAPIDS packages read it directly to find `targets/<arch>/include/`.
        # Without the export cupy raises `TypeError: expected str, bytes or
        # os.PathLike object, not NoneType` in `_get_conda_cuda_path`.
        export CONDA_PREFIX
        export CUDA_HOME="${CONDA_PREFIX}"
        export CUDA_PATH="${CONDA_PREFIX}"
        export PATH="${CONDA_PREFIX}/bin:$PATH"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        PYTHON="${CONDA_PREFIX}/bin/python"
        PYTEST="${CONDA_PREFIX}/bin/pytest"
        MATURIN="${CONDA_PREFIX}/bin/maturin"
    else
        echo "warn: conda prefix '$CONDA_PREFIX' not found — falling back to .venv/" >&2
        PYTHON="${REPO}/.venv/bin/python"
        PYTEST="${REPO}/.venv/bin/pytest"
        MATURIN="${REPO}/.venv/bin/maturin"
    fi
    [[ -x "$PYTHON" ]] || { echo "error: no python at $PYTHON" >&2; exit 2; }

    # ── 8.2 Rebuild pyscx with --release --features gpu ────────────────
    echo "--- 8.2: rebuild pyscx ---"
    "$MATURIN" develop --manifest-path pyscx/Cargo.toml --release --features gpu
    echo ""

    # ── 8.3 GPU correctness tests (cargo + pytest) ─────────────────────
    if [[ $SKIP_TESTS -eq 0 ]]; then
        echo "--- 8.3a: cargo test --workspace --features gpu ---"
        cargo test --workspace --features gpu -- --nocapture 2>&1 | \
            tee "benchmarks/logs/gpu_cargo_tests_${STAMP}.log"
        echo ""

        echo "--- 8.3b: pytest pyscx GPU tests ---"
        "$PYTEST" -v \
            pyscx/tests/test_accel_pca_gpu.py \
            pyscx/tests/test_accel_pipeline_gpu.py \
            2>&1 | tee "benchmarks/logs/gpu_pytest_${STAMP}.log"
        echo ""
    else
        echo "--- 8.3: skipped (--skip-tests) ---"
    fi

    # ── 8.4 + 8.5 Submit GPU benchmarks in parallel ───────────────────
    #
    # Default model: N parallel sbatches via slurm_gpu_regression_cell.sh,
    # one per entry in CELLS below. Each cell gets its own GPU allocation;
    # the driver waits on all of them. --inline forces sequential in-
    # allocation execution (debug / interactive-node use only).
    if [[ $INLINE -eq 1 ]]; then
        echo "--- 8.4 (inline): running GPU benchmarks directly ---"
        "$PYTHON" benchmarks/scripts/benchmark_gpu_pca.py        --mode all || true
        "$PYTHON" benchmarks/scripts/benchmark_gpu_knn.py        --mode all || true
        "$PYTHON" benchmarks/scripts/benchmark_gpu_umap.py       --mode all || true
        "$PYTHON" benchmarks/scripts/benchmark_gpu_preprocess.py --mode all || true
    else
        echo "--- 8.4: submitting parallel SLURM cell grid ---"

        # Each entry is "BENCHMARK[:DATASET]" — DATASET is only used by
        # the `pipeline` cell because that's the only script that writes
        # a single JSON per run regardless of dataset (forcing a 1-cell
        # per dataset split would clobber the shared output). Other
        # cells iterate datasets internally.
        CELLS=(
            "pca"
            "knn"
            "umap"
            "preprocess"
            "pipeline:${DATASET_FOR_PIPELINE}"
        )

        CELL_WRAPPER="benchmarks/scripts/slurm_gpu_regression_cell.sh"
        [[ -f "$CELL_WRAPPER" ]] || {
            echo "error: $CELL_WRAPPER missing" >&2; exit 2; }

        JOB_IDS=()
        GRID_MANIFEST="benchmarks/logs/gpu_regression_grid_${STAMP}.txt"
        : > "$GRID_MANIFEST"
        for cell in "${CELLS[@]}"; do
            bench="${cell%%:*}"
            dataset="${cell##*:}"
            [[ "$bench" == "$dataset" ]] && dataset=""  # no colon → unused
            export_vars="ALL,BENCHMARK=${bench}"
            [[ -n "$dataset" ]] && export_vars="${export_vars},DATASET=${dataset}"

            id="$(sbatch --parsable \
                --job-name="scx_gpu_cell_${bench}" \
                --export="${export_vars}" \
                "$CELL_WRAPPER")"
            printf '%s\t%s\t%s\n' "$id" "$bench" "${dataset:-n/a}" \
                | tee -a "$GRID_MANIFEST"
            JOB_IDS+=("$id")
        done

        if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
            echo "error: no cells submitted" >&2; exit 2
        fi
        echo ""
        echo "Grid manifest: $GRID_MANIFEST"
        echo "Waiting for ${#JOB_IDS[@]} cell(s) to finish: ${JOB_IDS[*]}"

        # Poll squeue. Each pass waits 30 s; bail out once no submitted jobs
        # remain in the queue (done or failed).
        while :; do
            remaining=0
            for id in "${JOB_IDS[@]}"; do
                if squeue -h -j "$id" 2>/dev/null | grep -q .; then
                    remaining=$((remaining + 1))
                fi
            done
            if [[ $remaining -eq 0 ]]; then
                break
            fi
            echo "  $(date +%H:%M:%S)  $remaining/${#JOB_IDS[@]} still running…"
            sleep 30
        done

        # Capture final per-job states for the log.
        echo ""
        echo "=== Grid cell exit states ==="
        FAILED_CELLS=0
        for id in "${JOB_IDS[@]}"; do
            state=$(sacct -j "$id" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
            echo "  cell $id: $state"
            [[ "$state" == "COMPLETED" ]] || FAILED_CELLS=$((FAILED_CELLS + 1))
        done
        if [[ $FAILED_CELLS -gt 0 ]]; then
            echo "warn: $FAILED_CELLS of ${#JOB_IDS[@]} cells did not complete cleanly"
            echo "      — resubmit the failing ones via:"
            echo "      sbatch --export=ALL,BENCHMARK=<bench> $CELL_WRAPPER"
            # Don't abort — let the diff run with whatever data was produced
            # so partial progress is still visible.
        fi
    fi

    # In inline mode, the Phase 7.2/7.3 captures run after the main grid
    # (in grid mode, the `preprocess` and `pipeline` cells above already
    # include them via the cell wrapper's dispatch).
    #
    # Note: `benchmark_gpu_preprocess.py --mode all` in step 8.4 already
    # runs the `device` sub-mode (see its argparse), so we do NOT re-run
    # `--mode device` here. Only the pipeline benchmark is Phase-7-
    # specific and not covered by `--mode all`.
    if [[ $INLINE -eq 1 ]]; then
        echo ""
        echo "--- 8.5 (inline): Phase 7 pipeline capture ---"
        "$PYTHON" benchmarks/scripts/benchmark_gpu_pipeline.py \
            --dataset "$DATASET_FOR_PIPELINE" \
            --attribute-preprocessing \
            --pca-variants \
            --n-runs 3
        echo ""
    fi
fi

# ── 8.6 Diff post vs pre ──────────────────────────────────────────────
echo "--- 8.6: diff post vs pre ---"
REPORT_MD="$POST/phases_1_7_gpu_regression_report_${STAMP}.md"
REPORT_JSON="$POST/phases_1_7_gpu_regression_report_${STAMP}.json"
# Also write a stable-named "latest" copy for operators that grep for it.
REPORT_MD_LATEST="$POST/phases_1_7_gpu_regression_report.md"
REPORT_JSON_LATEST="$POST/phases_1_7_gpu_regression_report.json"

DIFF_PY="${REPO}/benchmarks/scripts/gpu_regression_diff.py"
[[ -x "$DIFF_PY" || -f "$DIFF_PY" ]] || {
    echo "error: diff tool missing at $DIFF_PY" >&2; exit 2; }

# Pick whichever python is available (conda env when doing a full run,
# plain python3 when --diff-only on a CPU machine).
PY_FOR_DIFF="${PYTHON:-python3}"

set +e
"$PY_FOR_DIFF" "$DIFF_PY" \
    --pre "$PRE" --post "$POST" \
    --report "$REPORT_MD" --json-out "$REPORT_JSON"
diff_rc=$?
set -e

cp "$REPORT_MD"   "$REPORT_MD_LATEST"
cp "$REPORT_JSON" "$REPORT_JSON_LATEST"

echo ""
echo "============================================================"
echo "Done: $(date -Iseconds)"
echo "  report:       $REPORT_MD_LATEST"
echo "  verdict JSON: $REPORT_JSON_LATEST"
echo "  driver log:   $LOG"
echo "  diff exit:    $diff_rc  (0=pass, 1=regression/flag, 2=infra)"
echo "============================================================"

exit $diff_rc
