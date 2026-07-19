#!/usr/bin/env bash
#SBATCH --job-name=scx_mm_query_bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/scx_mm_query_bench_%j.out
#SBATCH --error=/home/%u/scx_mm_query_bench_%j.err

# Phase 7 — multimodal query pushdown vs `scx subset` micro-benchmark.
# Both paths are the SAME release `scx` CLI binary (apples-to-apples: pushdown
# vs full-materialization, no Python-interpreter RSS baseline). CPU-only.
#
# Prereq (run on the submit/worker node before sbatch):
#   cargo build --release -p scx-cli

set -euo pipefail

REPO=/home/nickyoungblut/dev/rust/scx
SCRATCH=/scratch/ctc/nickyoungblut
FIXTURE_DIR=${SCRATCH}/scx_mm_query
RESULTS_DIR=${REPO}/benchmarks/results/multimodal_query
mkdir -p "${FIXTURE_DIR}" "${RESULTS_DIR}"

cd "${REPO}"

echo "=== node: $(hostname) ==="
echo "    cpus: $(nproc)   mem-avail: $(free -h | awk '/^Mem:/ {print $7}')"
echo "    scx : $(./target/release/scx --version 2>/dev/null || echo MISSING)"
echo

# Two workload sizes so the RSS win is visible at scale and the trend is clear.
for NC in 100000 300000; do
  echo "=== workload: ${NC} cells ==="
  ./.venv/bin/python benchmarks/multimodal_query_bench.py \
    --n-cells "${NC}" \
    --rna-vars 3000 \
    --adt-vars 30 \
    --density 0.05 \
    --rare-frac 0.05 \
    --filter "cell_type == 'rare'" \
    --repeats 3 \
    --scx-bin target/release/scx \
    --fixture-dir "${FIXTURE_DIR}" \
    --output "${RESULTS_DIR}/mm_query_${NC}.json"
  echo
done

echo "=== Done. Results in ${RESULTS_DIR} ==="
ls -la "${RESULTS_DIR}"
