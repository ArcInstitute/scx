#!/bin/bash
# DEPRECATED (Phase 9 2026-04-23). Comprehensive framework replacement:
#     python benchmarks/comprehensive/scripts/run_parallel.py \
#         --benchmarks accel_pca accel_knn accel_umap accel_leiden \
#                      accel_preprocess accel_hvg \
#         --datasets pbmc3k tabula_sapiens_100k census_1m \
#         --skip-convert
# Kept in-tree one release for rollback.
#
# Per-cell SLURM wrapper for Phase-8 parallel-grid submission.
#
# Runs ONE GPU accelerator benchmark (all datasets that benchmark
# supports internally) under a dedicated GPU allocation.  The calling
# script (`gpu_regression_driver.sh`) submits N of these in parallel —
# one per entry in the benchmark list below — matching the
# "parallel SLURM job submission" pattern in AGENTS.md line 21.
#
# Why one cell per benchmark (not per benchmark × dataset)?  Each
# `benchmark_gpu_*.py` script writes to a fixed JSON file
# (`gpu_<name>_timing.json`, etc.), and multiple cells sharing that
# file would clobber each other.  Each cell's script iterates its
# datasets internally — that's still one parallel SLURM job per
# benchmark type, which is the parallelism that matters on a 5-cell
# grid.  Per-benchmark-per-dataset granularity requires patching each
# script to write per-dataset files; tracked as Phase-9 follow-up.
#
# Usage (from driver):
#   sbatch --parsable --export=ALL,BENCHMARK=pca benchmarks/scripts/slurm_gpu_regression_cell.sh
#
# Supported BENCHMARK values: pca | knn | umap | preprocess | pipeline
#
# For `pipeline`, an additional DATASET env var selects the dataset
# (default: census_1m) — this is the only cell that is dataset-
# parameterised because the script writes a single fixed JSON per run.

#SBATCH --job-name=scx_gpu_cell
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/logs/gpu_cell_%x_%j.log

set -euo pipefail
REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"

BENCHMARK="${BENCHMARK:?BENCHMARK env var required (pca|knn|umap|preprocess|pipeline)}"
DATASET="${DATASET:-census_1m}"   # only used by `pipeline`

CONDA_PREFIX_DEFAULT="/home/nickyoungblut/miniforge3/envs/scx-gpu"
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-$CONDA_PREFIX_DEFAULT}"

echo "=== SLURM cell $SLURM_JOB_ID on $(hostname) ==="
echo "BENCHMARK=$BENCHMARK  DATASET=$DATASET  started=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo ""

# Source .env (SCX_WORK_DIR) so bench_env.py works without python-dotenv.
if [[ -f "$REPO/.env" ]]; then
    set -a; . "$REPO/.env"; set +a
fi

# Bind the scx-gpu conda env. Export CONDA_PREFIX / CUDA_HOME / CUDA_PATH —
# cupy's CUDA-path detection reads these and crashes with a TypeError when
# CONDA_PREFIX is just a local shell var.
export CONDA_PREFIX
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_PREFIX}/bin/python"

mkdir -p benchmarks/results benchmarks/logs

# Dispatch.
case "$BENCHMARK" in
    pca)
        "$PYTHON" benchmarks/scripts/benchmark_gpu_pca.py --mode all
        ;;
    knn)
        "$PYTHON" benchmarks/scripts/benchmark_gpu_knn.py --mode all
        ;;
    umap)
        "$PYTHON" benchmarks/scripts/benchmark_gpu_umap.py --mode all
        ;;
    preprocess)
        # `--mode all` runs validate + bench + device (the last populates
        # gpu_preprocess_device.json for Phase 7.3). No second run needed.
        "$PYTHON" benchmarks/scripts/benchmark_gpu_preprocess.py --mode all
        ;;
    pipeline)
        # Phase 7.2 flag capture. `--attribute-preprocessing` and
        # `--pca-variants` populate the Phase-7.2 keys in
        # gpu_pipeline_timing.json that gpu_regression_diff.py reads.
        "$PYTHON" benchmarks/scripts/benchmark_gpu_pipeline.py \
            --dataset "$DATASET" \
            --attribute-preprocessing \
            --pca-variants
        ;;
    *)
        echo "error: unknown BENCHMARK=$BENCHMARK" >&2
        exit 2
        ;;
esac

echo ""
echo "=== cell $SLURM_JOB_ID done: $BENCHMARK @ $(date -Iseconds) ==="
