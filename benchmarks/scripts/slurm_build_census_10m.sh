#!/bin/bash
#SBATCH --job-name=build_10m
#SBATCH --partition=cpu_high_mem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/logs/build_census_10m_%j.log
#SBATCH --error=benchmarks/logs/build_census_10m_%j.err

# Build census_10m.h5ad from census chunk files (99 × 100K ≈ 10M cells)
# Requires ~500-1000 GB RAM
# Usage: sbatch benchmarks/scripts/slurm_build_census_10m.sh

set -euo pipefail

cd /home/nickyoungblut/dev/rust/scx

echo "=== Build census_10m ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "RAM: $(free -g | awk '/Mem/{print $2}') GB"
echo ""

.venv/bin/python benchmarks/scripts/build_census_10m.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
