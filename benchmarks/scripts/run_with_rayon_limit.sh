#!/bin/bash
# Wrapper script to limit thread/arena counts before running a command.
# On SLURM nodes, os.cpu_count() reports ALL hardware CPUs (e.g., 192),
# and SLURM may allocate more CPUs than requested to satisfy memory.
# This causes rayon, OpenBLAS, and GLIBC to create too many threads,
# leading to OOM from per-thread stacks and arenas.
#
# We cap at 16 threads regardless of SLURM_CPUS_PER_TASK to keep
# memory overhead predictable (~2-4 GB for thread stacks + arenas).
set -euo pipefail
NCPUS=16
export RAYON_NUM_THREADS=${NCPUS}
export OMP_NUM_THREADS=${NCPUS}
export OPENBLAS_NUM_THREADS=${NCPUS}
export MKL_NUM_THREADS=${NCPUS}
export NUMEXPR_MAX_THREADS=${NCPUS}
export MALLOC_ARENA_MAX=2
echo "ENV: THREADS=${NCPUS} SLURM_CPUS=${SLURM_CPUS_PER_TASK:-?} ARENA=${MALLOC_ARENA_MAX}"
exec "$@"
