#!/bin/bash
#SBATCH --job-name=de_csr_census1m
#SBATCH --partition=cpu_preemptible
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/de_csr_census1m.%j.out
#SBATCH --error=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/de_csr_census1m.%j.err

cd /home/nickyoungblut/dev/rust/scx
source .env
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true
.venv/bin/python /home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/scripts/_oneshot_de_csr_census1m.py
