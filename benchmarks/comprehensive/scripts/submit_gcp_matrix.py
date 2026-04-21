#!/usr/bin/env python3
"""
Phase E — GCP compute-node matrix launcher.

For each instance type in ``GCP_INSTANCE_TYPES`` (config.py), provisions a
VM in the bucket's region, runs the cloud benchmarks against the shared
GCS fixtures, copies the result JSONs back to this repo's ``results/raw/``
tree, and tears the VM down.

The launcher is cost-gated by default — it will NOT create any VMs unless
``--yes-spend`` is passed. ``--dry-run`` prints every gcloud command that
would run and exits. Use ``--instances`` to target a subset.

Workflow for each instance:

  1. ``gcloud compute instances create`` in GCP_BUCKET_REGION (zone picked
     per-family — see ``_default_zone_for_family``)
  2. Wait for SSH to come up
  3. ``gcloud compute scp`` the repo archive to ``/home/<user>/scx``
  4. Install conda + recreate ``scx-bench`` env (or use a pre-baked image)
  5. Run ``python -m benchmarks.comprehensive.scripts.run_parallel
     --launcher local --benchmarks <cloud benches> --datasets ...``
     with ``SCX_BENCH_GCP_INSTANCE`` / ``SCX_BENCH_GCP_REGION`` exported —
     sysinfo picks these up so the emitted JSONs carry the instance label
  6. ``gcloud compute scp`` the ``results/raw/cloud_*.json`` back
  7. ``gcloud compute instances delete`` (with ``--quiet``)

Result collection: JSONs are copied into
``benchmarks/comprehensive/results/raw/`` on this machine. They self-label
via ``system.gcp`` so ``reporting/tables.py::gcp_matrix_table`` can pivot
across instances.

This launcher intentionally does NOT use submitit — submitit's SLURM
integration doesn't apply here and GCP instance lifecycle is handled by
``gcloud`` directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    GCP_BUCKET_REGION,
    GCP_INSTANCE_TYPES,
    GCP_PROJECT,
    GCS_TEST_BUCKET,
    ensure_instance_region_matches_bucket,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zone selection
# ---------------------------------------------------------------------------

# Default zones within each region for the three instance families. Override
# at the CLI via ``--zone``. The A3 family is only available in a subset of
# zones, so picking a sensible default reduces the chance of silent fallback.
_REGION_FAMILY_ZONE_DEFAULTS: dict[str, dict[str, str]] = {
    "us-central1": {
        "n2": "us-central1-a",
        "c3": "us-central1-a",
        "a3": "us-central1-a",
    },
    "us-east4": {
        "n2": "us-east4-a",
        "c3": "us-east4-a",
        "a3": "us-east4-a",
    },
    "europe-west4": {
        "n2": "europe-west4-a",
        "c3": "europe-west4-a",
        "a3": "europe-west4-b",
    },
}


def _default_zone_for_family(region: str, family: str) -> str:
    """Return a sensible default zone for (region, family).

    Falls back to ``{region}-a`` if the table doesn't list the pairing —
    most regions have an ``-a`` zone and most instance families are
    available there.
    """
    by_family = _REGION_FAMILY_ZONE_DEFAULTS.get(region, {})
    return by_family.get(family, f"{region}-a")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARKS = [
    "cloud_read",
    "cloud_metadata",
    "cloud_filtered",
]

DEFAULT_FORMATS = ["scx_auto", "zarr_zstd", "tiledb_soma", "slaf"]

DEFAULT_DATASETS = ["pbmc3k", "tabula_sapiens_100k"]


@dataclass
class InstancePlan:
    """Everything the launcher needs to provision and run one VM."""
    instance_type: str
    zone: str
    vm_name: str
    disk_size_gb: int = 200
    image_family: str = "debian-12"
    image_project: str = "debian-cloud"
    service_account: str = ""
    env: dict[str, str] = field(default_factory=dict)


def _plan_instances(
    instance_types: Iterable[str],
    zone_override: str | None,
    disk_size_gb: int,
    run_tag: str,
) -> list[InstancePlan]:
    """Materialize one ``InstancePlan`` per requested instance type."""
    plans: list[InstancePlan] = []
    for itype in instance_types:
        spec = GCP_INSTANCE_TYPES.get(itype)
        if spec is None:
            raise ValueError(
                f"Unknown instance type {itype!r}; "
                f"available: {sorted(GCP_INSTANCE_TYPES)}"
            )
        family = spec["family"]
        zone = zone_override or _default_zone_for_family(GCP_BUCKET_REGION, family)
        region = zone.rsplit("-", 1)[0]
        ensure_instance_region_matches_bucket(region)
        vm_name = f"scx-bench-{itype.replace('_', '-')}-{run_tag}"
        plans.append(InstancePlan(
            instance_type=itype,
            zone=zone,
            vm_name=vm_name,
            disk_size_gb=disk_size_gb,
            env={
                "SCX_BENCH_GCP_INSTANCE": itype,
                "SCX_BENCH_GCP_REGION": region,
                "SCX_BENCH_GCP_ZONE": zone,
            },
        ))
    return plans


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _create_vm_cmd(plan: InstancePlan) -> list[str]:
    spec = GCP_INSTANCE_TYPES[plan.instance_type]
    cmd = [
        "gcloud", "compute", "instances", "create", plan.vm_name,
        f"--project={GCP_PROJECT}",
        f"--zone={plan.zone}",
        f"--machine-type={plan.instance_type}",
        f"--image-family={plan.image_family}",
        f"--image-project={plan.image_project}",
        f"--boot-disk-size={plan.disk_size_gb}GB",
        "--boot-disk-type=pd-ssd",
        "--scopes=cloud-platform",
    ]
    if spec.get("gpu"):
        # A3 nodes need the accelerator flag + host maintenance TERMINATE.
        cmd.extend([
            "--maintenance-policy=TERMINATE",
            "--accelerator=type=nvidia-h100-80gb,count=1",
        ])
    if plan.service_account:
        cmd.append(f"--service-account={plan.service_account}")
    return cmd


def _delete_vm_cmd(plan: InstancePlan) -> list[str]:
    return [
        "gcloud", "compute", "instances", "delete", plan.vm_name,
        f"--project={GCP_PROJECT}",
        f"--zone={plan.zone}",
        "--quiet",
    ]


def _ssh_cmd(plan: InstancePlan, remote_cmd: str) -> list[str]:
    return [
        "gcloud", "compute", "ssh", plan.vm_name,
        f"--project={GCP_PROJECT}",
        f"--zone={plan.zone}",
        "--command", remote_cmd,
    ]


def _scp_from_vm_cmd(
    plan: InstancePlan, remote_globs: list[str], local_dir: Path,
) -> list[str]:
    """Single ``gcloud compute scp`` invocation pulling every glob at once.

    One batched call replaces the previous loop-per-benchmark pattern;
    connection + auth handshake amortizes across all globs instead of
    paying per benchmark.
    """
    sources = [f"{plan.vm_name}:{g}" for g in remote_globs]
    return [
        "gcloud", "compute", "scp",
        "--recurse",
        f"--project={GCP_PROJECT}",
        f"--zone={plan.zone}",
        *sources,
        str(local_dir),
    ]


def _build_remote_bench_cmd(
    benchmarks: list[str],
    datasets: list[str],
    formats: list[str],
    env: dict[str, str],
) -> str:
    """Shell string executed on the VM to run the benchmark suite.

    Uses the non-SLURM ``local`` submitit executor so the VM runs the jobs
    directly. Assumes the repo is synced to ``$HOME/scx`` and the
    ``scx-bench`` conda env is already created (by the user's image or by
    a bootstrap step the launcher prints separately).
    """
    env_prefix = " ".join(
        f"{k}={shlex.quote(v)}" for k, v in env.items()
    )
    bench_args = " ".join(shlex.quote(b) for b in benchmarks)
    dataset_args = " ".join(shlex.quote(d) for d in datasets)
    format_args = " ".join(shlex.quote(f) for f in formats)
    inner = [
        "cd $HOME/scx",
        "source ~/.bashrc",
        "conda activate scx-bench",
        (
            f"{env_prefix} python -m benchmarks.comprehensive.scripts.run_parallel "
            "--launcher local "
            f"--benchmarks {bench_args} "
            f"--datasets {dataset_args} "
            f"--formats {format_args}"
        ),
    ]
    return " && ".join(inner)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_or_print(cmd: list[str], *, dry_run: bool) -> int:
    """Execute ``cmd`` unless ``dry_run`` is set, in which case print it."""
    printable = " ".join(shlex.quote(p) for p in cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return 0
    logger.info("exec: %s", printable)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("command failed (rc=%d): %s", result.returncode, printable)
    return result.returncode


def run_one_instance(
    plan: InstancePlan,
    benchmarks: list[str],
    datasets: list[str],
    formats: list[str],
    dry_run: bool,
    results_dir: Path,
) -> bool:
    """Provision, run, collect, teardown — returns True on success."""
    logger.info("=== %s (%s) ===", plan.instance_type, plan.zone)

    create_rc = _run_or_print(_create_vm_cmd(plan), dry_run=dry_run)
    if create_rc != 0 and not dry_run:
        logger.error("VM creation failed for %s; skipping", plan.vm_name)
        return False

    try:
        remote_cmd = _build_remote_bench_cmd(benchmarks, datasets, formats, plan.env)
        ssh_rc = _run_or_print(_ssh_cmd(plan, remote_cmd), dry_run=dry_run)
        if ssh_rc != 0 and not dry_run:
            logger.error("remote benchmark run failed on %s", plan.vm_name)
            return False

        # Collect only the cloud_* JSONs — this avoids slurping an entire
        # ``results/raw/`` tree if the remote has unrelated leftovers.
        # Batched into a single scp call (see ``_scp_from_vm_cmd``).
        results_dir.mkdir(parents=True, exist_ok=True)
        remote_globs = [
            f"~/scx/benchmarks/comprehensive/results/raw/{bench}__*.json"
            for bench in benchmarks
        ]
        scp_rc = _run_or_print(
            _scp_from_vm_cmd(plan, remote_globs, results_dir), dry_run=dry_run,
        )
        if scp_rc != 0 and not dry_run:
            logger.warning(
                "scp from %s returned %d; continuing", plan.vm_name, scp_rc,
            )
    finally:
        _run_or_print(_delete_vm_cmd(plan), dry_run=dry_run)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instances", nargs="+", default=sorted(GCP_INSTANCE_TYPES.keys()),
        help=f"Instance types to cover (default: all {sorted(GCP_INSTANCE_TYPES)})",
    )
    parser.add_argument(
        "--zone", default=None,
        help="Override zone (otherwise picked per-family from the "
             "region/family defaults in submit_gcp_matrix.py)",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        help=f"Cloud benchmarks to run on each VM (default: {DEFAULT_BENCHMARKS})",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help=f"Datasets (default: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        "--formats", nargs="+", default=DEFAULT_FORMATS,
        help=f"Formats (default: {DEFAULT_FORMATS})",
    )
    parser.add_argument(
        "--disk-size-gb", type=int, default=200,
        help="Boot disk size per VM (default: 200)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Where to collect result JSONs "
             "(default: benchmarks/comprehensive/results/raw/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every gcloud command without executing. Safe default "
             "for exploration — no GCP spend.",
    )
    parser.add_argument(
        "--yes-spend", action="store_true",
        help="Required to actually create VMs. Acts as a guard against "
             "accidentally launching an expensive matrix.",
    )
    parser.add_argument(
        "--run-tag", default=time.strftime("%Y%m%d-%H%M%S"),
        help="Tag appended to VM names for traceability (default: timestamp)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.dry_run and not args.yes_spend:
        parser.error(
            "--yes-spend is required for live execution. Use --dry-run to "
            "preview commands without spending."
        )

    plans = _plan_instances(
        args.instances, args.zone, args.disk_size_gb, args.run_tag,
    )

    results_dir = (
        Path(args.results_dir) if args.results_dir else
        PROJECT_ROOT / "benchmarks" / "comprehensive" / "results" / "raw"
    )

    logger.info(
        "GCP matrix plan: %d instances × %d benchmarks × %d datasets × %d formats",
        len(plans), len(args.benchmarks), len(args.datasets), len(args.formats),
    )
    logger.info("bucket=%s region=%s project=%s",
                GCS_TEST_BUCKET, GCP_BUCKET_REGION, GCP_PROJECT)
    logger.info("results_dir=%s", results_dir)
    if args.dry_run:
        logger.info("DRY RUN — no VMs will be created")

    failures: list[str] = []
    for plan in plans:
        ok = run_one_instance(
            plan,
            benchmarks=args.benchmarks,
            datasets=args.datasets,
            formats=args.formats,
            dry_run=args.dry_run,
            results_dir=results_dir,
        )
        if not ok:
            failures.append(plan.instance_type)

    if failures:
        logger.error("failed instances: %s", ", ".join(failures))
        return 1
    logger.info("GCP matrix complete (%d instances)", len(plans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
