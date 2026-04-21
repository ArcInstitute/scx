#!/usr/bin/env python3
"""
Promote a candidate snapshot to a canonical baseline (Phase G.2).

Copies ``summary.json`` + ``environment.json`` + ``MANIFEST.sha256`` from a
``capture_baseline.py`` snapshot tree into
``results/baselines/<version>/``. The raw `raw/*.json` files are NOT
copied — they stay in the original snapshot (or re-generated from the git
SHA recorded in environment.json), keeping the committed baseline tree
small enough to live in git while still being tamper-evident via the
manifest.

Usage:

    python benchmarks/comprehensive/scripts/promote_baseline.py \\
        --snapshot benchmarks/comprehensive/results/candidate_2026_04_18_batch_d_t3 \\
        --version v0.5.0-phase5

    # Overwrite an existing <version>/ (rare — typically you'd bump the version):
    python ... --version v0.5.0-phase5 --force

Refuses to overwrite an existing version directory unless ``--force`` —
canonical baselines should only change with a deliberate promotion.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINES_DIR = (
    PROJECT_ROOT / "benchmarks" / "comprehensive" / "results" / "baselines"
)

_REQUIRED_FILES = ("summary.json", "environment.json", "MANIFEST.sha256")
LATEST_LINK = "LATEST"

logger = logging.getLogger(__name__)


def _update_latest_symlink(baselines_dir: Path, version: str) -> None:
    """Point ``baselines/LATEST`` → ``baselines/<version>`` atomically.

    The on-demand gate workflow reads ``baselines/LATEST`` when
    ``--baseline`` is omitted. Using a symlink keeps the default dereference
    O(1) and avoids hardcoding version labels in every `compare` invocation.
    Falls back to a plain text pointer file on platforms / filesystems that
    reject symlinks (rare on Linux; belt-and-suspenders).
    """
    link = baselines_dir / LATEST_LINK
    target_relative = Path(version)  # relative so moves/clones stay valid
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target_relative, target_is_directory=True)
        logger.info("Updated %s → %s", link, target_relative)
    except OSError as exc:
        # Pointer file fallback: "LATEST" holds the version string.
        logger.warning("symlink failed (%s); writing pointer file instead", exc)
        link.write_text(f"{version}\n")


def promote(snapshot: Path, version: str, *, force: bool = False) -> Path:
    if not snapshot.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {snapshot}")

    missing = [f for f in _REQUIRED_FILES if not (snapshot / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"snapshot {snapshot} is missing required files: {missing}. "
            f"Is this a capture_baseline.py output?"
        )

    target = BASELINES_DIR / version
    if target.exists():
        if not force:
            raise FileExistsError(
                f"baseline {target} already exists. Pass --force to overwrite "
                f"(or pick a new --version label)."
            )
        logger.warning("Overwriting existing baseline at %s", target)
        shutil.rmtree(target)

    target.mkdir(parents=True)
    for name in _REQUIRED_FILES:
        shutil.copy2(snapshot / name, target / name)

    _update_latest_symlink(BASELINES_DIR, version)

    logger.info("Promoted %s → %s (%d files)", snapshot, target, len(_REQUIRED_FILES))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot", type=Path, required=True,
        help="Path to a capture_baseline.py snapshot directory",
    )
    parser.add_argument(
        "--version", required=True,
        help="Canonical label for this baseline (e.g. 'v0.5.0-phase5'). "
             "Typically a git tag.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing baselines/<version>/ (use sparingly).",
    )
    parser.add_argument(
        "--no-latest", action="store_true",
        help="Don't update baselines/LATEST after promoting. Useful when "
             "backfilling historical baselines out-of-order.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG logging",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        target = promote(
            args.snapshot, args.version,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.no_latest:
        # promote() always updates LATEST; if the operator asked to opt out,
        # revert to whatever was there before (best-effort).
        latest = BASELINES_DIR / "LATEST"
        if latest.is_symlink() and latest.resolve().name == args.version:
            latest.unlink()
            logger.info("--no-latest: removed %s", latest)

    print(f"Promoted snapshot to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
