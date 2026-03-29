#!/bin/bash
#SBATCH --job-name=build_5m
#SBATCH --partition=cpu_high_mem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=8:00:00
#SBATCH --output=benchmarks/logs/build_census_5m_%j.log
#SBATCH --error=benchmarks/logs/build_census_5m_%j.err

# Build census_5m.h5ad from census chunk files (50 × 100K = 5M cells)
# Requires ~200-300 GB RAM
# Usage: sbatch benchmarks/scripts/slurm_build_census_5m.sh

set -euo pipefail

cd /home/nickyoungblut/dev/rust/scx

echo "=== Build census_5m ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "RAM: $(free -g | awk '/Mem/{print $2}') GB"
echo ""

.venv/bin/python benchmarks/scripts/build_census_5m.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
