#!/usr/bin/env python3
"""
DEPRECATED — Phase C moved cloud benchmarking into the comprehensive harness.

The original 7-benchmark script has been replaced by four modules under
``benchmarks/comprehensive/benchmarks/``:

  * ``cloud_push``     — SCX push throughput
  * ``cloud_pull``     — SCX pull throughput
  * ``cloud_read``     — cross-format cloud full-read
  * ``cloud_metadata`` — cross-format metadata-open latency

Run them through the standard SLURM launcher — submitit handles per-triple
SLURM jobs, result JSON emission, and reporting just like the local
benchmarks:

    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --benchmarks cloud_push cloud_pull cloud_read cloud_metadata \\
        --datasets tabula_sapiens_100k \\
        --formats scx_auto zarr_zstd tiledb_soma slaf

This shim forwards to that command (defaults: ``tabula_sapiens_100k`` with
SCX auto, matching the legacy script's default dataset/format). Any extra
CLI arguments are passed through to ``run_parallel.py``.

Will be removed one release cycle after the comprehensive cloud benchmarks
land. See the ``Cloud Benchmarks (GCP)`` section of ``benchmarks/README.md``
for the current workflow.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    print(
        "=" * 70,
        "DEPRECATED: benchmark_cloud.py has been replaced by the comprehensive",
        "benchmark harness. Forwarding to:",
        "",
        "  benchmarks/comprehensive/scripts/run_parallel.py \\",
        "    --benchmarks cloud_push cloud_pull cloud_read cloud_metadata \\",
        "    --datasets tabula_sapiens_100k --formats scx_auto",
        "",
        "See benchmarks/README.md 'Cloud Benchmarks (GCP)' for details.",
        "=" * 70,
        sep="\n",
        file=sys.stderr,
    )

    launcher = REPO_ROOT / "benchmarks" / "comprehensive" / "scripts" / "run_parallel.py"
    extra = sys.argv[1:]
    defaults = [
        "--benchmarks", "cloud_push", "cloud_pull", "cloud_read", "cloud_metadata",
        "--datasets", "tabula_sapiens_100k",
        "--formats", "scx_auto",
    ]
    cmd = [sys.executable, str(launcher), *(extra if extra else defaults)]

    env = os.environ.copy()
    default_key = Path.home() / ".gcp" / "scx-bench.json"
    if "GOOGLE_APPLICATION_CREDENTIALS" not in env and default_key.exists():
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_key)

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
