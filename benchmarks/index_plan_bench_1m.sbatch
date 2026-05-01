#!/usr/bin/env bash
#SBATCH --job-name=scx_index_plan_1m
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/scx_index_plan_1m_%j.out
#SBATCH --error=/home/%u/scx_index_plan_1m_%j.err

# Phase 7.2 — 1M-cell IndexPlanDataset throughput benchmark.
# Spec: PER-CELL-CONTROL-PAIRING.md §Phase 7.2 (1024 pairs/batch, 1000 batches).
# Two passes: the IndexPlanDataset configs at full 1000 batches, then the slow
# backed_python_baseline at 50 batches (the 200K bench showed it's ~60× slower
# than the indexed path; full 1000-batch run would take ~5 hours of bench wall
# in addition to the indexed configs).

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
SCRATCH=/scratch/$(id -u)
FIXTURE_DIR=${SCRATCH}/scx_index_plan_1m
RESULTS_DIR=${REPO}/benchmarks/comprehensive/results/index_plan_phase7
mkdir -p "${FIXTURE_DIR}" "${RESULTS_DIR}"

cd "${REPO}"

echo "=== Lambda HPC node: $(hostname) ==="
echo "    cpus available  : $(nproc)"
echo "    mem available   : $(free -h | awk '/^Mem:/ {print $7}')"
echo "    SLURM_CPUS_ON_NODE: ${SLURM_CPUS_ON_NODE:-?}"
echo "    SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-?}"
echo "    fixture dir     : ${FIXTURE_DIR}"
echo "    results dir     : ${RESULTS_DIR}"
echo

# --- Pass 1: IndexPlanDataset configs + sequential ceiling (1000 batches) ---
echo "=== Pass 1/2: IndexPlanDataset + TrainingDataset, 1000 batches ==="
./.venv/bin/python benchmarks/index_plan_bench.py \
  --n-cells 1000000 \
  --n-vars 2000 \
  --n-pairs-per-batch 1024 \
  --n-batches 1000 \
  --hvg-size 2000 \
  --fixture-dir "${FIXTURE_DIR}" \
  --config index_plan_random,index_plan_locality,sort_off,sort_on,lookahead0,lookahead4,lookahead8,training_seq \
  --output "${RESULTS_DIR}/index_plan_1m_indexed.json"

echo
echo "=== Pass 2/2: backed_python baseline, 50 batches ==="
./.venv/bin/python benchmarks/index_plan_bench.py \
  --n-cells 1000000 \
  --n-vars 2000 \
  --n-pairs-per-batch 1024 \
  --n-batches 50 \
  --hvg-size 2000 \
  --fixture-dir "${FIXTURE_DIR}" \
  --config backed_python \
  --output "${RESULTS_DIR}/index_plan_1m_backed_python.json"

echo
echo "=== Done. Results in ${RESULTS_DIR} ==="
ls -la "${RESULTS_DIR}"
