#!/usr/bin/env python3
"""
Minimal reproducer for the SLAF cloud-path resolution failure (KI.2).

Opens ``gs://arc-ctc-nextflow/scx-test/pbmc3k.slaf`` via ``SLAFArray`` and
touches ``.shape``. Passes when the required cloud deps
(``smart_open[gcs]`` → ``google-cloud-storage``) are installed; fails
with a misleading ``FileNotFoundError`` when they aren't.

Intended uses:

  * Regression probe — re-run after any `scx-bench-slaf.yml` rebuild. A
    failure means `google-cloud-storage` (or equivalent) was dropped.
  * Upstream-issue attachment — minimal, credential-aware, no submitit.
    Demonstrates that SLAF's `_path_exists` catches an ImportError and
    silently returns False, surfacing as "SLAF dataset not found".

Exit 0 on all-runs-pass, 1 on any failure.
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

DEFAULT_URL = f"{GCS_TEST_BUCKET}/pbmc3k.slaf"


def _attempt_open(url: str) -> tuple[bool, str, float]:
    """Return (success, message, wall_s) for one SLAFArray cloud open."""
    t0 = time.perf_counter()
    try:
        from slaf import SLAFArray

        arr = SLAFArray(url)
        wall = time.perf_counter() - t0
        return True, f"shape={arr.shape}", wall
    except Exception as exc:  # noqa: BLE001 — repro captures every failure mode
        wall = time.perf_counter() - t0
        return False, f"{type(exc).__name__}: {exc}", wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Cloud SLAF URL to probe (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of attempts (default 3). Misleading errors in this "
             "class tend to be deterministic, so multiple runs mostly "
             "confirm consistency.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("url: %s", args.url)
    logger.info("runs: %d", args.runs)
    logger.info("")

    passes = 0
    fails = 0
    for i in range(args.runs):
        ok, msg, wall = _attempt_open(args.url)
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
