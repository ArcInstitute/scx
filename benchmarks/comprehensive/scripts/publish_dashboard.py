#!/usr/bin/env python3
"""
Rsync the HTML snapshot tree to a static-hosting target (Phase G.4).

Reads ``DASHBOARD_PUBLISH_TARGET`` from ``config.py`` (env-backed). When
unset, prints a skip message and exits 0 — CI can unconditionally invoke
this script from a post-report step without hardcoding credentials.

Protocols supported, auto-selected by URL prefix:
  * ``gs://…``            → ``gsutil -m rsync -r``
  * ``s3://…``            → ``aws s3 sync``
  * ``user@host:…`` / ``rsync://`` → ``rsync -avz``
  * local path            → ``rsync -a`` (useful for a mounted NFS target)

Only the current ``BENCHMARK_REPORT.{html,md,pdf}`` + ``dashboard_history.json``
+ ``figures/`` tree are published — raw JSONs stay off the public target.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    DASHBOARD_PUBLISH_TARGET,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)

# Files/dirs copied to the publish target — deliberately allowlisted so
# the publish step cannot leak raw result JSONs or internal debug output.
_PUBLISH_ITEMS = (
    "BENCHMARK_REPORT.html",
    "BENCHMARK_REPORT.md",
    "BENCHMARK_REPORT.pdf",
    "BENCHMARK_REPORT.json",
    "LINT_WARNINGS.json",
    "dashboard_history.json",
    "figures",
)


def _command_for(target: str, items: list[Path]) -> list[str]:
    item_args = [str(p) for p in items if p.exists()]
    if target.startswith("gs://"):
        # gsutil rsync is directory-to-directory; stage items by copying to
        # a temp list. Simpler path: call `gsutil cp -r` on each item.
        return ["gsutil", "-m", "cp", "-r", *item_args, target]
    if target.startswith("s3://"):
        # aws s3 cp takes one source at a time when not using sync — use cp
        # per item to mirror the gsutil shape.
        raise NotImplementedError(
            "s3:// publish targets are not yet wired — use an rsync-over-SSH "
            "target or request S3 support via a roadmap ticket."
        )
    if target.startswith("rsync://") or "@" in target.split("/", 1)[0]:
        return ["rsync", "-avz", *item_args, target]
    # Local / mounted NFS
    return ["rsync", "-a", *item_args, target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target", default=DASHBOARD_PUBLISH_TARGET,
        help="Override DASHBOARD_PUBLISH_TARGET for this invocation. Empty = skip.",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help=f"Source directory containing the HTML snapshot (default: {REPORTS_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the transfer command without executing it.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    target = (args.target or "").strip()
    if not target:
        logger.info(
            "DASHBOARD_PUBLISH_TARGET is not configured — skipping publish. "
            "Set DASHBOARD_PUBLISH_TARGET=<rsync-or-gs-target> to enable.",
        )
        return 0

    items = [args.reports_dir / name for name in _PUBLISH_ITEMS]
    missing = [str(p) for p in items if not p.exists()]
    if all(p in missing for p in (str(i) for i in items)):
        logger.error(
            "No publishable artifacts found under %s. Run write_report() first.",
            args.reports_dir,
        )
        return 1
    if missing:
        logger.info(
            "Some expected items are missing (will be skipped): %s", missing,
        )

    cmd = _command_for(target, items)
    printable = " ".join(shlex.quote(p) for p in cmd)
    if args.dry_run:
        print(f"[dry-run] {printable}")
        return 0

    logger.info("publishing → %s", target)
    logger.info("exec: %s", printable)
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
