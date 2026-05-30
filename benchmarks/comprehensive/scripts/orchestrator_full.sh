#!/bin/bash
# Orchestrator sentinel: runs run_parallel.py for the full-tier dataset list
# (small tier + census_500k + census_1m + multimodal pair). Wraps the
# orchestrator in a long-walltime sbatch so it survives ssh drops and can be
# monitored with `squeue` + watch.py.
#SBATCH --job-name=scx-bench-full
#SBATCH --partition=cpu_preemptible
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/nickyoungblut/dev/rust/scx

# Load conda (init lines are commented out in ~/.bashrc on this account)
source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench

# Load SCX_WORK_DIR and friends
set -a
source .env
set +a

echo "=== full orchestrator boot ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} time=$(date -Iseconds)"
echo "python=$(which python)"
python --version
echo "=============================="

# full tier datasets — TIERS["full"] from capture_baseline.py plus the
# multimodal pair so multimodal_* benchmarks pick up coverage too.
python benchmarks/comprehensive/scripts/run_parallel.py \
  --datasets \
    pbmc3k pbmc10k smartseq2 tabula_sapiens_100k \
    census_500k census_1m \
    cite_seq_pbmc multiome_pbmc \
  --skip-smoke
