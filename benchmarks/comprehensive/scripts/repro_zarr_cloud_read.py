#!/usr/bin/env python3
"""
Minimal reproducer for the zarr cloud-read zstd-decompression failure.

Opens a cloud zarr store via ``zarr.open(gs://…)`` and fully reads the
``indices`` and ``data`` arrays — the two that failed in Phase 5
cloud benchmarking. Sweepable via ``--concurrency N`` to isolate the
zarr ``async.concurrency`` knob as the trigger.

Intended uses:

  * Regression probe — re-run after any zarr/gcsfs/aiohttp bump. If it
    fails at concurrency=10 but passes at concurrency=1, KI.1 is still
    present and the harness mitigation (async.concurrency=1 in
    ZarrRunner.read_cloud) must stay in place.
  * Upstream ticket attachment — minimal, credential-aware, no
    submitit. Reproduces the decompression error deterministically.

Expected behavior on the affected stack (zarr 3.1.5 + gcsfs 2025.12 +
aiohttp 3.13.3):

    repro_zarr_cloud_read.py --concurrency 1  --runs 5  → 5/5 pass
    repro_zarr_cloud_read.py --concurrency 10 --runs 5  → partial fails

Exit 0 on all-runs-pass, 1 on any failure (usable as a CI regression
check once the harness mitigation lands).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# config import triggers .env load + tilde-expands GOOGLE_APPLICATION_CREDENTIALS
from benchmarks.comprehensive.config import (  # noqa: E402
    GCS_TEST_BUCKET,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = f"{GCS_TEST_BUCKET}/pbmc3k.zarr"


def _attempt_read(url: str, concurrency: int) -> tuple[bool, str, float]:
    """Return (success, message, wall_s) for one cloud-read attempt.

    Opens the store, reads ``indptr``, ``indices``, and ``data`` fully.
    Mirrors what ``ZarrRunner.read_cloud`` does for the 3-array CSR layout.
    """
    import zarr

    t0 = time.perf_counter()
    try:
        with zarr.config.set({"async.concurrency": concurrency}):
            store = zarr.open(url, mode="r")
            # Touch attrs first (the cloud_metadata path — always works)
            shape = tuple(store.attrs.get("shape", ()))
            # Full reads of all three arrays — the failing path
            indptr = store["indptr"][:]
            indices = store["indices"][:]
            data = store["data"][:]
        wall = time.perf_counter() - t0
        return (
            True,
            f"shape={shape} nnz={len(data)} indptr[0]={int(indptr[0])}",
            wall,
        )
    except Exception as exc:  # noqa: BLE001 — repro captures every failure mode
        wall = time.perf_counter() - t0
        return False, f"{type(exc).__name__}: {exc}", wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Cloud zarr URL to probe (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="zarr async.concurrency value (default 10 — reproduces KI.1; "
             "set 1 to confirm the mitigation path succeeds)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of attempts (default 5 — KI.1 is flaky, not fully "
             "deterministic, so multiple runs increase signal)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("url: %s", args.url)
    logger.info("concurrency: %d  runs: %d", args.concurrency, args.runs)
    logger.info("")

    passes = 0
    fails = 0
    for i in range(args.runs):
        ok, msg, wall = _attempt_read(args.url, args.concurrency)
        status = "PASS" if ok else "FAIL"
        logger.info("run %d/%d  [%s] %.2fs  %s", i + 1, args.runs, status, wall, msg)
        if ok:
            passes += 1
        else:
            fails += 1

    logger.info("")
    logger.info("summary: %d pass / %d fail", passes, fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
