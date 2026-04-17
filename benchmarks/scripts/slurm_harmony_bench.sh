#!/bin/bash
# Phase 6 — Harmony + LISI scaling sweep. Parallel SLURM submission.
#
# One job per (impl, dataset, device) triple + PC/K secondary sweeps on D4
# + a final report-generation job depending afterok on all benchmark jobs.
#
# Usage:  cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_harmony_bench.sh
#
# Environment:
#   SCX_WORK_DIR from .env — used by bench_env.py
#   Four conda envs required (see plan):
#     rscx (R harmony + lisi), scx-bench (harmonypy), scx-gpu (GPU),
#     project .venv (scx-accel CPU).

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON_VENV="${SCX_DIR}/.venv/bin/python"
PYTHON_SCX_BENCH="/home/nickyoungblut/miniforge3/envs/scx-bench/bin/python"
PYTHON_SCX_GPU="/home/nickyoungblut/miniforge3/envs/scx-gpu/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs benchmarks/results/harmony/runs
RUNS_DIR="${SCX_DIR}/benchmarks/results/harmony/runs"

echo "=== Phase 6: Harmony + LISI scaling sweep ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Runs will land in: ${RUNS_DIR}"
echo ""

JOB_IDS=()

# Core invoker: dispatches one benchmark_harmony.py run under sbatch.
# Args: impl dataset device partition mem_gb time_h cpus
submit_harmony () {
    local impl="$1" ds="$2" dev="$3" part="$4" mem="$5" hrs="$6" cpus="$7"
    local extra_flags="${8:-}"

    # The harmony driver internally picks the right interpreter per impl/device.
    local cmd="cd ${SCX_DIR} && ${PYTHON_VENV} benchmarks/scripts/benchmark_harmony.py \
        --dataset ${ds} --impl ${impl} --device ${dev} ${extra_flags}"

    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="hy_${impl:0:3}_${ds}_${dev}" \
        --partition="${part}" \
        --qos=normal \
        --cpus-per-task="${cpus}" \
        --mem="${mem}G" \
        --time="${hrs}:00:00" \
        --output="benchmarks/logs/harmony_${impl}_${ds}_${dev}_%j.log" \
        --error="benchmarks/logs/harmony_${impl}_${ds}_${dev}_%j.err" \
        $( [[ "$dev" == "gpu" ]] && echo "--gres=gpu:1" ) \
        --wrap="${cmd}")
    JOB_IDS+=("${JOB_ID}")
    echo "  Submitted: ${impl}/${ds}/${dev} → job ${JOB_ID}"
}

submit_lisi () {
    local impl="$1" ds="$2" part="$3" mem="$4" hrs="$5"
    local cmd="cd ${SCX_DIR} && ${PYTHON_VENV} benchmarks/scripts/benchmark_lisi.py \
        --dataset ${ds} --impl ${impl}"
    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="lisi_${impl}_${ds}" \
        --partition="${part}" \
        --qos=normal \
        --cpus-per-task=8 \
        --mem="${mem}G" \
        --time="${hrs}:00:00" \
        --output="benchmarks/logs/lisi_${impl}_${ds}_%j.log" \
        --error="benchmarks/logs/lisi_${impl}_${ds}_%j.err" \
        --wrap="${cmd}")
    JOB_IDS+=("${JOB_ID}")
    echo "  Submitted: lisi/${impl}/${ds} → job ${JOB_ID}"
}

# ── Cell-count scaling sweep (CPU) ───────────────────────────────
# D1–D4 — small/medium; cpu_preemptible 80 GB, 2 h
for impl in scx_accel_cpu harmonypy r_harmony; do
    for ds in pbmc3k pbmc10k smartseq2 tabula_sapiens_100k; do
        submit_harmony "${impl}" "${ds}" cpu cpu_preemptible 80 2 16
    done
done

# D5–D6 — 500k / 1M; cpu_preemptible 256 GB, 4 h
for impl in scx_accel_cpu harmonypy r_harmony; do
    for ds in census_500k census_1m; do
        submit_harmony "${impl}" "${ds}" cpu cpu_preemptible 256 4 32
    done
done

# D7 — 5M; cpu_high_mem 500 GB, 8 h
for impl in scx_accel_cpu harmonypy r_harmony; do
    submit_harmony "${impl}" census_5m cpu cpu_high_mem 500 8 32
done

# ── GPU sweep (scx-accel GPU only) ───────────────────────────────
for ds in pbmc10k smartseq2 tabula_sapiens_100k census_500k census_1m census_5m; do
    submit_harmony scx_accel_gpu "${ds}" gpu preemptible 128 2 16
done

# ── Secondary axes on D4 (tabula_sapiens_100k) ───────────────────
for d in 10 20 30 50 100; do
    submit_harmony scx_accel_cpu tabula_sapiens_100k cpu cpu_preemptible 80 2 16 "--n-pcs ${d} --tag pc${d}"
done
for K in 50 100 200 400; do
    submit_harmony scx_accel_cpu tabula_sapiens_100k cpu cpu_preemptible 80 2 16 "--n-clusters ${K} --tag K${K}"
done

# ── LISI (scx-accel vs R on D1–D4) ────────────────────────────────
# LISI is O(N^2 d) in brute-force mode, so we cap at D4 to stay tractable.
for ds in pbmc3k pbmc10k smartseq2 tabula_sapiens_100k; do
    submit_lisi scx_accel "${ds}" cpu_preemptible 128 2
    submit_lisi r_lisi   "${ds}" cpu_preemptible 128 2
done

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs."

# ── Final report job — fires after all benchmarks finish (afterok OR afterany) ──
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
REPORT_ID=$(sbatch --parsable \
    --dependency=afterany:"${DEPS}" \
    --job-name=hy_report \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --output="benchmarks/logs/harmony_report_%j.log" \
    --error="benchmarks/logs/harmony_report_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON_VENV} benchmarks/scripts/report_harmony.py")

echo "Submitted archival job: ${REPORT_ID} (depends on all ${#JOB_IDS[@]} benchmark jobs)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Report: ${SCX_DIR}/benchmarks/results/harmony/REPORT.md"

# Emit a JSON manifest so the caller can programmatically poll squeue.
printf '{"benchmark_jobs": [%s], "report_job": "%s"}\n' \
    "$(IFS=,; echo "\"${JOB_IDS[*]}\"" | sed 's/,/","/g')" "${REPORT_ID}" \
    > "${SCX_DIR}/benchmarks/logs/harmony_sweep_manifest.json"
