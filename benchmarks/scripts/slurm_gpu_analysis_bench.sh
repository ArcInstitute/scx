#!/bin/bash
#SBATCH --job-name=scx_gpu_analysis
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/logs/gpu_analysis_bench_%j.log

set -euo pipefail

REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"

# Use the scx-gpu conda environment for RAPIDS (cuVS, cuGraph) compatibility.
# The conda env resolves the full CUDA + RAPIDS dependency tree, avoiding
# CUDA version mismatches (e.g., pip cuVS needing CUDA 12.9 vs driver 535/12.2).
export CONDA_PREFIX="/home/nickyoungblut/miniforge3/envs/scx-gpu"
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p benchmarks/logs benchmarks/results

echo "============================================================"
echo "SCX GPU Analysis Benchmark Suite"
echo "============================================================"
echo ""

echo "=== System info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Building pyscx in release mode (with GPU features) ==="
cargo clean -p scx-gpu -p pyscx --release 2>/dev/null || true
cd pyscx && maturin develop --release --features gpu && cd ..
echo ""

# ── 1. GPU PCA Validation & Benchmark ──
echo "============================================================"
echo "1/5  GPU PCA Validation & Benchmark"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_pca.py --mode all 2>&1
echo ""

# ── 2. GPU kNN Validation & Benchmark (existing script) ──
echo "============================================================"
echo "2/5  GPU kNN Validation & Benchmark"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_knn.py --mode all 2>&1
echo ""

# ── 3. GPU UMAP Validation & Benchmark ──
echo "============================================================"
echo "3/5  GPU UMAP Validation & Benchmark"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_umap.py --mode all 2>&1
echo ""

# ── 4. GPU Preprocessing Benchmark ──
echo "============================================================"
echo "4/5  GPU Preprocessing Benchmark"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_preprocess.py --mode all 2>&1
echo ""

# ── 5. End-to-End Pipeline & Go/No-Go Gate ──
echo "============================================================"
echo "5/5  End-to-End Pipeline & Go/No-Go Gate"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_pipeline.py --mode all 2>&1
echo ""

# ── Summary ──
echo "============================================================"
echo "All Benchmark Results"
echo "============================================================"
echo ""
echo "--- JSON outputs ---"
ls -la benchmarks/results/gpu_*.json 2>/dev/null || echo "No JSON files found"
echo ""
echo "--- Markdown reports ---"
ls -la benchmarks/results/gpu_*_benchmark.md 2>/dev/null || echo "No reports found"
echo ""

# Print Go/No-Go gate result
echo "--- Go/No-Go Gate ---"
if [ -f benchmarks/results/gpu_gonogo.json ]; then
    cat benchmarks/results/gpu_gonogo.json
else
    echo "gpu_gonogo.json not generated"
fi
echo ""

echo "=== DONE ==="
