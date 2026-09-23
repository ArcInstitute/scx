#!/usr/bin/env python3
"""Re-convert benchmark fixtures from h5ad source, optionally re-push to cloud.

When the source h5ad fixtures change shape (e.g. the 2026-05-09
``obs.n_counts`` augmentation), the derived SCX / Zarr / TileDB local
files and their cloud-pushed copies go stale. This script reruns the
full conversion + push pipeline for selected (dataset, format) pairs so
operators don't have to drive each runner by hand.

The 2026-05-09 → 2026-05-11 firefight ran this campaign ad-hoc out of
``/tmp/reconvert_stale.py``; promoting the workflow here makes future
fixture refreshes a single command.

Usage::

    # Re-convert default fixture matrix locally (all datasets x
    # {scx_auto, tiledb_soma, zarr_zstd}); skip cloud push.
    python benchmarks/scripts/reconvert_fixtures.py

    # Cherry-pick datasets + formats, force overwrite, and re-push.
    python benchmarks/scripts/reconvert_fixtures.py \
        --datasets pbmc3k pbmc10k tabula_sapiens_100k \
        --formats scx_auto tiledb_soma \
        --cloud-push

    # See what would happen without touching anything.
    python benchmarks/scripts/reconvert_fixtures.py --dry-run --cloud-push

CSC sidecars: ``scx_auto`` fixtures are converted with ``csc="auto"``
(``benchmarks.comprehensive.convert.FIXTURE_CSC_POLICY``), so every fixture
that clears the ingest rule (n_obs >= 50,000 and n_vars >= 5,000) comes back
with a sidecar, and a reconvert cannot drop the one the route floors in
``thresholds.yaml`` depend on. The other SCX codec variants stay CSR-only.

Pre-reqs:
  - ``SCX_DATA_DIR`` resolves to the staged h5ad fixtures (or
    ``SCX_WORK_DIR/benchmarks/datasets`` if not set explicitly).
  - For ``--cloud-push``: ``GOOGLE_APPLICATION_CREDENTIALS`` (or
    Application Default Credentials) configured for the
    ``GCS_TEST_BUCKET`` project. ``gsutil`` on PATH.
  - Active conda env carries the format-runner dependencies for the
    formats requested. ``scx-bench`` covers the default trio.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Repo root so the benchmarks package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.comprehensive.config import (  # noqa: E402
    DATASETS,
    ALL_FORMATS,
    DatasetConfig,
    FormatVariant,
)
from benchmarks.comprehensive.convert import convert_dataset_format  # noqa: E402

logger = logging.getLogger("reconvert_fixtures")


# Default re-conversion matrix — the formats whose local files got
# stale in the 2026-05-09 fixture campaign. h5ad rows are intentionally
# excluded (the augmented h5ad IS the source of truth); h5mu / parquet
# / bpcells / slaf rows are excluded because they're either downstream
# of separate pipelines or carry their own runner-specific quirks.
_DEFAULT_FORMATS = ("scx_auto", "tiledb_soma", "zarr_zstd")


def _resolve_formats(format_keys: list[str] | None) -> list[FormatVariant]:
    by_key = {fv.key: fv for fv in ALL_FORMATS}
    keys = list(format_keys) if format_keys else list(_DEFAULT_FORMATS)
    resolved: list[FormatVariant] = []
    unknown: list[str] = []
    for k in keys:
        if k in by_key:
            resolved.append(by_key[k])
        else:
            unknown.append(k)
    if unknown:
        raise SystemExit(
            f"Unknown format key(s): {unknown}. Known: {sorted(by_key)}"
        )
    return resolved


def _resolve_datasets(dataset_names: list[str] | None) -> list[DatasetConfig]:
    names = list(dataset_names) if dataset_names else list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(
            f"Unknown dataset name(s): {unknown}. Known: {sorted(DATASETS)}"
        )
    return [DATASETS[n] for n in names]


def _gsutil_rm_recursive(url: str) -> None:
    """Best-effort ``gsutil rm -r`` to clear a stale cloud fixture.

    ``ensure_cloud_fixture`` short-circuits when ``_is_fixture_complete``
    returns True (URL + ``.blake3`` sidecar both present on GCS). For a
    forced re-push after local re-conversion we want to invalidate that
    short-circuit by nuking the cloud copy outright. Failures are
    logged but not fatal — if the URL doesn't exist yet, the upload
    that follows is the same operation we'd want anyway.
    """
    import subprocess
    try:
        subprocess.run(
            ["gsutil", "-m", "rm", "-rf", url],
            timeout=120, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.debug("  gsutil rm skipped for %s (%s)", url, exc)


def _push_to_cloud(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    local_path: Path,
) -> str:
    """Upload a freshly-converted fixture to its canonical cloud URL.

    Clears the stale cloud copy first (both the main directory and the
    BLAKE3 sidecar) so ``ensure_cloud_fixture`` does not short-circuit
    on the prior completion marker. Then delegates to
    ``ensure_cloud_fixture`` so the upload path, lock file, sidecar
    write, and pyscx.push behaviour all match what the live benchmarks
    see at read-time.
    """
    from benchmarks.comprehensive.cloud_fixtures import (
        cloud_url_for,
        ensure_cloud_fixture,
        require_gcp_credentials,
    )

    require_gcp_credentials()
    url = cloud_url_for(dataset, format_variant)
    # Invalidate the completion short-circuit. Two paths to clear:
    # (1) the main directory / file (covers exploded .scxd and zarr
    #     trees), (2) the .blake3 sidecar in case the dir clear leaves
    #     it behind on some buckets (a `gsutil rm -rf` on a prefix
    #     normally takes everything; belt-and-suspenders).
    _gsutil_rm_recursive(url.rstrip("/"))
    _gsutil_rm_recursive(url.rstrip("/") + ".blake3")
    return ensure_cloud_fixture(dataset, format_variant, local_path)


def _convert_one(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    *,
    cloud_push: bool,
    dry_run: bool,
) -> dict:
    """Re-convert a single (dataset, format) pair. Returns a status dict."""
    label = f"{dataset.name}/{format_variant.key}"
    out_path = dataset.path_for_format(format_variant.key)
    record = {
        "label": label,
        "local_path": str(out_path),
        "converted": False,
        "cloud_pushed": False,
        "cloud_url": None,
        "wall_s": 0.0,
        "error": None,
    }
    if dry_run:
        logger.info("[dry-run] would re-convert %s → %s", label, out_path)
        if cloud_push:
            logger.info("[dry-run] would push %s to GCS", label)
        return record

    t0 = time.perf_counter()
    try:
        convert_dataset_format(dataset, format_variant, overwrite=True)
        record["converted"] = True
    except Exception as exc:
        record["error"] = f"convert: {type(exc).__name__}: {exc}"
        logger.error("  FAILED convert %s: %s", label, exc)
        return record

    if cloud_push:
        try:
            url = _push_to_cloud(dataset, format_variant, out_path)
            record["cloud_pushed"] = True
            record["cloud_url"] = url
        except Exception as exc:
            record["error"] = f"push: {type(exc).__name__}: {exc}"
            logger.error("  FAILED push %s: %s", label, exc)

    record["wall_s"] = time.perf_counter() - t0
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-convert benchmark fixtures from h5ad source, optionally "
            "re-push to cloud. Promoted from the 2026-05-09 ad-hoc campaign."
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset names to reconvert (default: all). "
             "Run with `--list` to see known names.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=None,
        help=f"Format keys to emit (default: {' '.join(_DEFAULT_FORMATS)}).",
    )
    parser.add_argument(
        "--cloud-push", action="store_true",
        help="After local conversion, upload to the canonical GCS path "
             "via ensure_cloud_fixture. Requires GCP credentials and "
             "gsutil on PATH.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without converting or uploading.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print known datasets and format keys, then exit.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (`-v` INFO, `-vv` DEBUG).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    if args.list:
        print("Datasets:")
        for name in sorted(DATASETS):
            print(f"  {name}")
        print("Formats:")
        for fv in ALL_FORMATS:
            print(f"  {fv.key}")
        return 0

    datasets = _resolve_datasets(args.datasets)
    formats = _resolve_formats(args.formats)

    logger.info(
        "Plan: %d datasets × %d formats = %d conversions (cloud_push=%s, dry_run=%s)",
        len(datasets), len(formats), len(datasets) * len(formats),
        args.cloud_push, args.dry_run,
    )

    records: list[dict] = []
    for ds in datasets:
        for fv in formats:
            rec = _convert_one(
                ds, fv,
                cloud_push=args.cloud_push,
                dry_run=args.dry_run,
            )
            records.append(rec)

    if args.dry_run:
        return 0

    # Summary
    converted = sum(1 for r in records if r["converted"])
    pushed = sum(1 for r in records if r["cloud_pushed"])
    failed = sum(1 for r in records if r["error"])
    total_wall = sum(r["wall_s"] for r in records)
    logger.warning(
        "Done: %d converted, %d pushed, %d failed (total wall: %.1f s)",
        converted, pushed, failed, total_wall,
    )
    if failed:
        for r in records:
            if r["error"]:
                logger.warning("  FAIL %s: %s", r["label"], r["error"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
