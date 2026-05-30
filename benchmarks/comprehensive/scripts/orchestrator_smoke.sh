#!/bin/bash
# Orchestrator sentinel: runs run_parallel.py for the small-tier dataset list.
# Submitted via sbatch so the orchestrator survives ssh drops and can be
# monitored with `squeue` + watch.py.
#SBATCH --job-name=scx-bench-small
#SBATCH --partition=cpu_preemptible
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /home/nickyoungblut/dev/rust/scx

# Source the conda init (commented out in ~/.bashrc on this account; load directly)
source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench

# Load SCX_WORK_DIR and friends
set -a
source .env
set +a

echo "=== orchestrator boot ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} time=$(date -Iseconds)"
echo "python=$(which python)"
python --version
echo "=========================="

python benchmarks/comprehensive/scripts/run_parallel.py \
  --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k cite_seq_pbmc multiome_pbmc \
  --skip-smoke
