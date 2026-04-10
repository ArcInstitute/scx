#!/bin/bash
# Phase 4: ML Data Loader Throughput benchmark (§3.6)
#
# Submits parallel SLURM jobs: one CPU job + one GPU job per dataset.
# CPU jobs benchmark all 4 loaders (SCX, AnnData, TileDB-SOMA-ML, scDataLoader)
# across 4 data scenarios (raw, hvg, norm, hvg_norm).
# GPU jobs benchmark SCX only with a scVI-equivalent training loop (gpu_train).
#
# Datasets: pbmc3k, tabula_sapiens_100k, census_1m
#
# Usage:
#   cd /home/nickyoungblut/dev/rust/scx
#   bash benchmarks/scripts/slurm_phase4_ml_loader.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p benchmarks/logs

DATASETS=("pbmc3k" "tabula_sapiens_100k" "census_1m")
FORMATS="scx_auto h5ad_none h5ad_gzip tiledb_soma"
SCX_FORMATS="scx_auto"

# Resource allocation per dataset (CPU jobs)
declare -A CPU_MEM=( [pbmc3k]=32G [tabula_sapiens_100k]=80G [census_1m]=200G )
declare -A CPU_TIME=( [pbmc3k]=01:00:00 [tabula_sapiens_100k]=02:00:00 [census_1m]=06:00:00 )

# Resource allocation per dataset (GPU jobs)
declare -A GPU_MEM=( [pbmc3k]=32G [tabula_sapiens_100k]=64G [census_1m]=128G )
declare -A GPU_TIME=( [pbmc3k]=00:30:00 [tabula_sapiens_100k]=01:00:00 [census_1m]=02:00:00 )

SCX_DIR="$(pwd)"

echo "=== Phase 4: ML Data Loader Throughput Benchmark ==="
echo "Date: $(date)"
echo "Repo: ${SCX_DIR}"
echo ""

# -------------------------------------------------------------------------
# Phase 1: CPU throughput jobs (all loaders, data scenarios)
# -------------------------------------------------------------------------
echo "--- Submitting CPU jobs ---"
CPU_JOBS=()
for ds in "${DATASETS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="p4_ml_cpu_${ds}" \
        --partition=cpu \
        --cpus-per-task=32 \
        --mem="${CPU_MEM[$ds]}" \
        --time="${CPU_TIME[$ds]}" \
        --output="benchmarks/logs/phase4_ml_cpu_${ds}_%j.log" \
        --error="benchmarks/logs/phase4_ml_cpu_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && \
            .venv/bin/maturin develop --release --manifest-path pyscx/Cargo.toml && \
            .venv/bin/python benchmarks/comprehensive/scripts/run_all.py \
                --benchmarks ml_loader \
                --datasets ${ds} \
                --formats ${FORMATS}")
    echo "  CPU ${ds}: job ${JOB_ID} (mem=${CPU_MEM[$ds]}, time=${CPU_TIME[$ds]})"
    CPU_JOBS+=("${JOB_ID}")
done

# -------------------------------------------------------------------------
# Phase 2: GPU training jobs (SCX only, gpu_train scenario)
# Depends on CPU jobs so SCX files are pre-converted.
# -------------------------------------------------------------------------
echo ""
echo "--- Submitting GPU jobs ---"
DEP_STR=$(IFS=:; echo "${CPU_JOBS[*]}")
for ds in "${DATASETS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="p4_ml_gpu_${ds}" \
        --partition=gpu \
        --gpus-per-node=1 \
        --cpus-per-task=16 \
        --mem="${GPU_MEM[$ds]}" \
        --time="${GPU_TIME[$ds]}" \
        --dependency="afterok:${DEP_STR}" \
        --output="benchmarks/logs/phase4_ml_gpu_${ds}_%j.log" \
        --error="benchmarks/logs/phase4_ml_gpu_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && \
            source activate scx-gpu && \
            maturin develop --features gpu --release --manifest-path pyscx/Cargo.toml && \
            python benchmarks/comprehensive/scripts/run_all.py \
                --benchmarks ml_loader \
                --datasets ${ds} \
                --formats ${SCX_FORMATS}")
    echo "  GPU ${ds}: job ${JOB_ID} (depends on CPU jobs, mem=${GPU_MEM[$ds]})"
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor:  squeue -u \$USER"
echo "CPU logs: benchmarks/logs/phase4_ml_cpu_*"
echo "GPU logs: benchmarks/logs/phase4_ml_gpu_*"
echo "Results:  benchmarks/comprehensive/results/raw/ml_loader__*"
