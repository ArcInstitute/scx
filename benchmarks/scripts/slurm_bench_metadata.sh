#!/bin/bash
#SBATCH --job-name=bench_metadata
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
# Relative log paths resolve against the submission dir — submit from repo root
# (`sbatch benchmarks/scripts/slurm_bench_metadata.sh`).
#SBATCH --output=benchmarks/comprehensive/logs/bench_metadata.%j.out
#SBATCH --error=benchmarks/comprehensive/logs/bench_metadata.%j.err

# T4 micro-benchmark: obs/var sharded-metadata decode, serial vs parallel +
# rayon thread scaling. Drives the `read_obs_timing` diagnostic
# (scx-format-io/tests/read_obs_timing.rs), which times bare
# `ScxReader::read_obs()` / `read_var()` — the path parallelized by
# `read_sharded_layout_by_prefix`. Pure CPU; no GPU needed.
#
#   serial baseline : SCX_METADATA_DECODE_SERIAL=1  (forces single-thread decode)
#   parallel scaling: RAYON_NUM_THREADS in {1,2,4,8,16,32}  (rayon reads it at
#                     pool init → one process per thread count)
set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p benchmarks/comprehensive/logs
source .env
# .env may export VIRTUAL_ENV / activate a venv; irrelevant to cargo, but unset
# SLURM cpu vars that can throw off rayon's default parallelism detection.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true

DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"
echo "=== node: $(hostname) | cores: $(nproc) | DATA_DIR=${DATA_DIR} ==="

# Build the test binary once (release); later invocations reuse the cache.
echo "=== building read_obs_timing (release) ==="
cargo test --release -p scx-format-io --test read_obs_timing --no-run 2>&1 | tail -3

run_cfg() {
    # $1 = fixture basename, $2 = label, then remaining = extra env assignments
    local fixture="$1"; shift
    local label="$1"; shift
    local path="${DATA_DIR}/${fixture}.scx"
    if [ ! -f "${path}" ]; then
        echo "  [skip] ${fixture}: missing at ${path}"
        return
    fi
    echo "=== ${fixture} [${label}] ==="
    env "$@" REPRO_SHARD="${path}" \
        cargo test --release -p scx-format-io --test read_obs_timing \
        -- --ignored --nocapture 2>&1 | grep -E "read_obs_timing:|mode=|open=|read_obs:|read_var:"
}

for fixture in census_1m_scx1 census_500k_scx1; do
    # Serial baseline (single-threaded decode).
    run_cfg "${fixture}" "serial" SCX_METADATA_DECODE_SERIAL=1
    # Parallel scaling curve.
    for T in 1 2 4 8 16 32; do
        run_cfg "${fixture}" "parallel_t${T}" RAYON_NUM_THREADS="${T}"
    done
done

echo "=== done ==="
