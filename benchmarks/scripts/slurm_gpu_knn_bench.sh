#!/bin/bash
#SBATCH --job-name=scx_gpu_knn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/logs/gpu_knn_bench_%j.log

set -euo pipefail

REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"

# Export VIRTUAL_ENV so Rust cuVS loader can find libcuvs_c.so in site-packages
export VIRTUAL_ENV="${REPO}/.venv"

# Add cuVS and RAPIDS library paths to LD_LIBRARY_PATH for dlopen resolution
CUVS_LIB_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)/libcuvs/lib64
RAPIDS_LIB_DIRS="${VIRTUAL_ENV}/lib/python3.13/site-packages/libcuvs/lib64"
RAPIDS_LIB_DIRS="${RAPIDS_LIB_DIRS}:${VIRTUAL_ENV}/lib/python3.13/site-packages/libraft/lib64"
RAPIDS_LIB_DIRS="${RAPIDS_LIB_DIRS}:${VIRTUAL_ENV}/lib/python3.13/site-packages/librmm/lib64"
RAPIDS_LIB_DIRS="${RAPIDS_LIB_DIRS}:${VIRTUAL_ENV}/lib/python3.13/site-packages/rapids_logger/lib64"
RAPIDS_LIB_DIRS="${RAPIDS_LIB_DIRS}:${VIRTUAL_ENV}/lib/python3.13/site-packages/nvidia/cublas/lib"
RAPIDS_LIB_DIRS="${RAPIDS_LIB_DIRS}:${VIRTUAL_ENV}/lib/python3.13/site-packages/nvidia/cusparse/lib"
RAPIDS_LIB_DIRS="${RAPIDS_LIB_DIRS}:${VIRTUAL_ENV}/lib/python3.13/site-packages/nvidia/cusolver/lib"
export LD_LIBRARY_PATH="${RAPIDS_LIB_DIRS}:${LD_LIBRARY_PATH:-}"

mkdir -p benchmarks/logs benchmarks/results

echo "=== System info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Building pyscx in release mode (with GPU features) ==="
cargo clean -p scx-gpu -p pyscx --release 2>/dev/null || true
cd pyscx && ../.venv/bin/maturin develop --release --features gpu && cd ..
echo ""

echo "=== Running GPU kNN validation & benchmark ==="
.venv/bin/python benchmarks/scripts/benchmark_gpu_knn.py --mode all 2>&1
echo ""

echo "=== Results ==="
cat benchmarks/results/gpu_knn_benchmark.md
echo ""

echo "=== JSON outputs ==="
ls -la benchmarks/results/gpu_knn_*.json 2>/dev/null || echo "No JSON files found"
echo ""

echo "=== DONE ==="
