#!/usr/bin/env python3
"""
Stamp ``schema_version`` on pre-versioned raw result JSONs (Phase I.2).

Walks ``benchmarks/comprehensive/results/raw/`` (or a custom directory
passed via ``--dir``), reads each ``*.json``, adds the current
``SCHEMA_VERSION`` if absent, and rewrites the file in place. Safe to
re-run — a result that already carries a version at or above the
current one is left untouched.

Usage:

    python benchmarks/comprehensive/scripts/migrate_results.py
    python benchmarks/comprehensive/scripts/migrate_results.py \\
        --dir benchmarks/comprehensive/results/candidate_abc1234 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import RAW_RESULTS_DIR  # noqa: E402
from benchmarks.comprehensive.results import SCHEMA_VERSION  # noqa: E402

logger = logging.getLogger(__name__)


def migrate_file(path: Path, *, dry_run: bool) -> str:
    """Return one of: 'stamped', 'already_current', 'too_new', 'invalid'."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("skipping %s: invalid JSON (%s)", path, exc)
        return "invalid"
    current = data.get("schema_version")
    if current is None:
        data["schema_version"] = SCHEMA_VERSION
        if not dry_run:
            path.write_text(json.dumps(data, indent=2, default=str))
        return "stamped"
    if current > SCHEMA_VERSION:
        logger.warning(
            "%s already at schema_version %d > current %d — leaving alone",
            path, current, SCHEMA_VERSION,
        )
        return "too_new"
    return "already_current"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=RAW_RESULTS_DIR,
        help=f"Directory to scan (default: {RAW_RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without modifying files.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.dir.is_dir():
        print(f"ERROR: {args.dir} is not a directory", file=sys.stderr)
        return 1

    counts = {"stamped": 0, "already_current": 0, "too_new": 0, "invalid": 0}
    for f in sorted(args.dir.glob("*.json")):
        status = migrate_file(f, dry_run=args.dry_run)
        counts[status] += 1
    verb = "would stamp" if args.dry_run else "stamped"
    print(
        f"{verb} {counts['stamped']}, already current {counts['already_current']}, "
        f"too_new {counts['too_new']}, invalid {counts['invalid']} "
        f"(target schema_version={SCHEMA_VERSION})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
