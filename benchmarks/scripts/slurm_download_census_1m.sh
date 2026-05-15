#!/bin/bash
#SBATCH --job-name=download_census_1m
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=benchmarks/logs/download_census_1m_%j.log
#SBATCH --error=benchmarks/logs/download_census_1m_%j.err

# Download census_1m.h5ad (1M human blood cells from CELLxGENE Census).
# The Census API materialises every matching cell into RAM before
# subsampling — the full blood-cell pull dwarfs the 1 M target, hence
# the conservative `--mem=512G`. Lambda HPC `standard` nodes have
# ~1.8 TB RAM; this leaves headroom for the AnnData copy + h5ad
# write buffers.
#
# Output: ${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}/census_1m.h5ad
# Approx output size: 11 GB (per benchmarks/comprehensive/config.py D6).
#
# Usage: sbatch benchmarks/scripts/slurm_download_census_1m.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"

if [ -f "${SCX_DIR}/.env" ]; then
    set -a; source "${SCX_DIR}/.env"; set +a
fi
if [ -z "${SCX_WORK_DIR:-}" ]; then
    echo "ERROR: SCX_WORK_DIR is not set. Define it in ${SCX_DIR}/.env or export it." >&2
    exit 1
fi

PYTHON="${SCX_DIR}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: ${PYTHON} not executable" >&2
    exit 1
fi

# `bench_env.py` ships under benchmarks/comprehensive/ but the download
# scripts import it as a top-level module (`from bench_env import DATA_DIR`).
# Surface it on PYTHONPATH so the import resolves when sbatch runs the
# script from the repo root.
export PYTHONPATH="${SCX_DIR}/benchmarks/comprehensive:${PYTHONPATH:-}"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Download census_1m ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "RAM: $(free -g | awk '/Mem/{print $2}') GB"
echo "SCX_WORK_DIR: ${SCX_WORK_DIR}"
echo "SCX_DATA_DIR: ${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"
echo ""

${PYTHON} benchmarks/scripts/download_census_1m.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
