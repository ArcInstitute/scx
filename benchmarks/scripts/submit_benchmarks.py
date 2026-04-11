#!/usr/bin/env python3
"""
Submit SCX training loader benchmarks to SLURM via submitit.

Usage:
    # Smoke test (PBMC 3K, CPU-only, quick)
    python benchmarks/scripts/submit_benchmarks.py --smoke

    # Full benchmark suite
    python benchmarks/scripts/submit_benchmarks.py

    # Single dataset, GPU
    python benchmarks/scripts/submit_benchmarks.py --dataset tabula_sapiens_100k --n-gpus 1

    # All datasets with GPU sweep
    python benchmarks/scripts/submit_benchmarks.py --all-datasets --gpu-sweep
"""

import argparse
import json
import sys
from pathlib import Path

import submitit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
SLURM_LOG_DIR = REPO_ROOT / "benchmarks" / "slurm_logs"

sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build
import benchmark_loader  # noqa: E402 — must be after sys.path insert


def run_benchmark_job(dataset_name: str, n_gpus: int = 0,
                      smoke: bool = False) -> dict:
    """Entry point for a single benchmark job (runs inside SLURM allocation)."""
    return benchmark_loader.run_all_benchmarks(
        dataset_name, n_gpus=n_gpus, smoke=smoke,
    )


def main():
    parser = argparse.ArgumentParser(description="Submit SCX benchmarks to SLURM")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (PBMC 3K, CPU-only)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to benchmark")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Benchmark all available datasets")
    parser.add_argument("--n-gpus", type=int, default=0,
                        help="GPUs per job (0=CPU-only)")
    parser.add_argument("--gpu-sweep", action="store_true",
                        help="Run GPU scaling sweep (1, 2, 4 GPUs)")
    parser.add_argument("--local", action="store_true",
                        help="Run locally (no SLURM)")
    parser.add_argument("--partition", type=str, default="preemptible",
                        help="SLURM partition (default: preemptible for GPU jobs)")
    parser.add_argument("--time", type=str, default="4:00:00",
                        help="SLURM time limit")
    args = parser.parse_args()

    SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    if args.smoke:
        datasets = ["pbmc3k"]
    elif args.all_datasets:
        datasets = [name for name, ds in benchmark_loader.DATASETS.items()
                     if ds["h5ad"].exists()]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        # Default: run on all available datasets
        datasets = [name for name, ds in benchmark_loader.DATASETS.items()
                     if ds["h5ad"].exists()]

    # Determine GPU configurations
    if args.gpu_sweep:
        gpu_configs = [0, 1, 2, 4]
    elif args.n_gpus > 0:
        gpu_configs = [args.n_gpus]
    else:
        gpu_configs = [0]

    print("=" * 60)
    print("SCX Benchmark Submission")
    print("=" * 60)
    print(f"Datasets: {datasets}")
    print(f"GPU configs: {gpu_configs}")
    print(f"Partition: {args.partition}")
    print(f"Smoke test: {args.smoke}")
    print(f"Local: {args.local}")
    print()

    ensure_release_build()

    if args.local:
        # Run locally without SLURM
        all_results = []
        for ds in datasets:
            for n_gpus in gpu_configs:
                print(f"\nRunning: {ds} (GPUs={n_gpus})...")
                result = run_benchmark_job(ds, n_gpus=n_gpus, smoke=args.smoke)
                all_results.append(result)

        # Generate combined report
        report = benchmark_loader.generate_report(all_results)
        report_path = RESULTS_DIR / "training_loader_benchmark.md"
        report_path.write_text(report)
        print(f"\nReport: {report_path}")

        json_path = RESULTS_DIR / "training_loader_benchmark.json"
        json_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"JSON: {json_path}")
        return

    # Submit to SLURM via submitit
    jobs = []
    job_configs = []

    for ds in datasets:
        for n_gpus in gpu_configs:
            partition = args.partition if n_gpus > 0 else "cpu_preemptible"
            time_limit = args.time

            executor = submitit.AutoExecutor(folder=str(SLURM_LOG_DIR))
            executor.update_parameters(
                slurm_partition=partition,
                slurm_gpus_per_node=n_gpus if n_gpus > 0 else 0,
                slurm_cpus_per_task=min(8 * max(n_gpus, 1), 32),
                slurm_mem_per_cpu="10G",
                slurm_time=time_limit,
                slurm_job_name=f"scx-bench-{ds}-g{n_gpus}",
                slurm_setup=[
                    # Activate the project venv
                    f"source {REPO_ROOT / '.venv' / 'bin' / 'activate'}",
                ],
            )

            job = executor.submit(run_benchmark_job, ds, n_gpus, args.smoke)
            jobs.append(job)
            job_configs.append({"dataset": ds, "n_gpus": n_gpus})
            print(f"Submitted: {ds} (GPUs={n_gpus}) → Job {job.job_id}")

    # Wait for all jobs
    print(f"\nWaiting for {len(jobs)} job(s) to complete...")
    all_results = []

    for job, config in zip(jobs, job_configs):
        try:
            result = job.result()  # Blocks until job completes
            all_results.append(result)
            print(f"  Completed: {config['dataset']} (GPUs={config['n_gpus']})")
        except Exception as e:
            print(f"  FAILED: {config['dataset']} (GPUs={config['n_gpus']}): {e}")
            all_results.append({
                "dataset": config["dataset"],
                "error": str(e),
                "benchmarks": {},
            })

    # Generate combined report
    report = benchmark_loader.generate_report(all_results)
    report_path = RESULTS_DIR / "training_loader_benchmark.md"
    report_path.write_text(report)
    print(f"\nReport: {report_path}")

    json_path = RESULTS_DIR / "training_loader_benchmark.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
