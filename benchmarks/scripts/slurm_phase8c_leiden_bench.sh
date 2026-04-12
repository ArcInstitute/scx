#!/bin/bash
# Phase 8c: Leiden Benchmark — Sequential + Parallel Rust vs Python leidenalg
#
# Submits parallel SLURM jobs: one per dataset, with env vars for memory control.
#
# Usage: bash benchmarks/scripts/slurm_phase8c_leiden_bench.sh

set -euo pipefail

SCX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SCX_DIR}/.venv"
PYTHON="${VENV}/bin/python"
MATURIN="${VENV}/bin/maturin"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs benchmarks/results

echo "=== Phase 8c: Leiden Benchmark — Seq/Par Rust vs Python leidenalg ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# ── Step 0: Build pyscx in release mode ──────────────────────────────────────

BUILD_JOB=$(sbatch --parsable \
    --job-name="p8c_build" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --output="benchmarks/logs/phase8c_build_%j.log" \
    --error="benchmarks/logs/phase8c_build_%j.err" \
    --wrap="cd ${SCX_DIR}/pyscx && ${MATURIN} develop --release && echo 'Build complete'")
echo "Build job: ${BUILD_JOB}"

# ── Helper: create a wrapper script that sets env vars before Python ─────────
# This ensures RAYON_NUM_THREADS and MALLOC_ARENA_MAX are set BEFORE
# the Python process (and thus rayon's global thread pool) initializes.

cat > /tmp/p8c_run_bench.sh << 'WRAPPER_EOF'
#!/bin/bash
set -euo pipefail
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MALLOC_ARENA_MAX=2
echo "ENV: RAYON_NUM_THREADS=${RAYON_NUM_THREADS} MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX} CPUS=${SLURM_CPUS_PER_TASK:-?}"
cd "$1"
shift
exec "$@"
WRAPPER_EOF
chmod +x /tmp/p8c_run_bench.sh

# ── Step 1: Leiden benchmark per dataset ─────────────────────────────────────

JOB_IDS=()

# tabula_sapiens_100k (100K cells)
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p8c_leiden_tabula" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=64G \
    --time=04:00:00 \
    --output="benchmarks/logs/phase8c_leiden_tabula_%j.log" \
    --error="benchmarks/logs/phase8c_leiden_tabula_%j.err" \
    --wrap="/tmp/p8c_run_bench.sh ${SCX_DIR} ${PYTHON} benchmarks/scripts/benchmark_leiden.py --mode all --datasets tabula_sapiens_100k --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Leiden x tabula_sapiens_100k -> job ${JOB_ID}"

# census_1m (1M cells)
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p8c_leiden_census" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=128G \
    --time=16:00:00 \
    --output="benchmarks/logs/phase8c_leiden_census_%j.log" \
    --error="benchmarks/logs/phase8c_leiden_census_%j.err" \
    --wrap="/tmp/p8c_run_bench.sh ${SCX_DIR} ${PYTHON} benchmarks/scripts/benchmark_leiden.py --mode all --datasets census_1m --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Leiden x census_1m -> job ${JOB_ID}"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs + 1 build job."
echo "Build job:   ${BUILD_JOB}"
echo ""
echo "Monitor with:  squeue -u \$USER --name=p8c_"
echo "Results in:    benchmarks/results/leiden_benchmark.json"
echo "Logs in:       benchmarks/logs/phase8c_*.log"
