#!/bin/bash
#SBATCH --job-name=bench_gpu_codec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=8
# Relative log paths resolve against the submission dir — submit from repo root
# (`sbatch benchmarks/scripts/slurm_bench_gpu_codec.sh`).
#SBATCH --output=benchmarks/comprehensive/logs/bench_gpu_codec.%j.out
#SBATCH --error=benchmarks/comprehensive/logs/bench_gpu_codec.%j.err

# GPU ShufDeltaZstd decode — Phase 0 (tasks 0a/0c): profile to_gpu_anndata across
# scx1 / compact_trial / shufdelta encodings of census_1m + census_500k and
# print the host-bounce-vs-Scx1 throughput ratio + go/no-go for Phase 1.
#
# Rebuilds pyscx (release, gpu) in the bench-gpu conda env, then runs the
# standalone driver. Paths are derived dynamically so any maintainer can run
# this unedited.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p benchmarks/comprehensive/logs
source .env
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${SCX_BENCH_GPU_ENV:-scx-bench-gpu}"
# .env activates the dev .venv (exports VIRTUAL_ENV); maturin refuses when both
# VIRTUAL_ENV and CONDA_PREFIX are set. Drop VIRTUAL_ENV so maturin targets the
# active conda env (scx-bench-gpu).
unset VIRTUAL_ENV

echo "=== GPU visible ==="
nvidia-smi -L || true

# Rebuild pyscx so the bench-gpu env carries current HEAD (device-decode +
# uns stamping). --features gpu matches the route-gate template; this bench
# needs no hdf5 (fixtures are pre-built .scx).
echo "=== rebuilding pyscx (release, gpu) ==="
( cd pyscx && maturin develop --release --features gpu )

# Per-stage GPU/host profiler is OnceLock-cached in scx-gpu, so it must be set
# in the environment BEFORE the process starts (an in-process os.environ set
# after import is too late → all-zero buckets).
export SCX_GPU_PROFILE=1
# So the Phase-2 nvcomp path can dlopen libnvcomp.so.5 (+ its transitive deps)
# from the conda env; set before the process starts (ld.so caches it).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

OUT_DIR="${REPO_ROOT}/benchmarks/results/gpu_codec"
echo "=== running bench_gpu_codec.py → ${OUT_DIR} ==="
# --compare-sequential runs the Phase-1.5 pipeline vs the Phase-1 sequential
# path (SCX_SHUFDELTA_GPU_SEQUENTIAL=1) for shufdelta / compact_trial.
python benchmarks/scripts/bench_gpu_codec.py \
    --datasets census_1m census_500k \
    --n-runs 3 \
    --compare-sequential \
    --out-dir "${OUT_DIR}"
echo "=== bench_gpu_codec exit: $? ==="

# T4 macro A/B: how much the parallel obs/var metadata decode moves the
# metadata-bound to_gpu_anndata wall (serial vs parallel, in-process toggle).
echo "=== running bench_metadata_ab.py (T4 macro A/B) → ${OUT_DIR} ==="
python benchmarks/scripts/bench_metadata_ab.py \
    --fixtures census_1m_scx1 census_1m_compact_trial census_500k_scx1 \
    --n-runs 3 \
    --out-dir "${OUT_DIR}"
echo "=== bench_metadata_ab exit: $? ==="
