#!/usr/bin/env python3
"""
Inspect cloud fixtures vs local files and report mismatches (Phase I.4).

Dry-run safe — never uploads or deletes. For each format × dataset in the
comprehensive registry, compares:

  * Whether the local `.scx` / `.zarr` / `.soma` / `.slaf` exists
  * Whether the cloud URL exists (`gsutil ls` probe)
  * The BLAKE3 digest recorded by ``ensure_cloud_fixture`` in
    ``{cloud_url}.blake3`` vs the local file's current BLAKE3

Output is a markdown table + exit code summary (0 = all match, 1 = at
least one mismatch). Use it before a benchmark run to catch stale cloud
copies that would otherwise silently produce stale numbers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.cloud_fixtures import (  # noqa: E402
    cloud_path_exists,
    cloud_url_for,
    compute_blake3,
    read_blake3_sidecar,
)
from benchmarks.comprehensive.config import (  # noqa: E402
    ALL_FORMATS,
    DATASETS,
    DatasetConfig,
    FormatVariant,
)

logger = logging.getLogger(__name__)


def _status(local: Path, url: str, compute: bool) -> tuple[str, str]:
    """Return ``(status, note)`` for one (dataset, format) pair."""
    if not local.exists():
        return "NO_LOCAL", "local fixture absent (convert first)"
    try:
        exists = cloud_path_exists(url)
    except Exception as exc:  # noqa: BLE001
        return "PROBE_FAIL", f"gsutil ls failed: {exc}"
    if not exists:
        return "NOT_UPLOADED", "cloud path missing"

    sidecar = read_blake3_sidecar(url)
    if sidecar is None:
        return "NO_SIDECAR", "cloud fixture present but no .blake3 sidecar"
    if not compute:
        return "OK_NO_VERIFY", f"cloud sidecar={sidecar[:16]}… (local not hashed)"
    local_digest = compute_blake3(local)
    if local_digest != sidecar:
        return "DRIFT", (
            f"local={local_digest[:16]}… cloud={sidecar[:16]}… (re-upload needed)"
        )
    return "OK", f"digest match ({local_digest[:16]}…)"


def _iter_targets(
    datasets: list[str] | None,
    formats: list[str] | None,
):
    ds_names = datasets or list(DATASETS.keys())
    fmt_keys = formats or [f.key for f in ALL_FORMATS]
    fmt_map = {f.key: f for f in ALL_FORMATS}
    for ds_name in ds_names:
        ds: DatasetConfig | None = DATASETS.get(ds_name)
        if ds is None:
            logger.warning("unknown dataset %r; skipping", ds_name)
            continue
        for fmt_key in fmt_keys:
            fmt: FormatVariant | None = fmt_map.get(fmt_key)
            if fmt is None:
                continue
            try:
                url = cloud_url_for(ds, fmt)
            except ValueError:
                # Formats without a cloud layout (e.g. h5ad) — skip silently.
                continue
            try:
                local = ds.path_for_format(fmt_key)
            except ValueError:
                continue
            yield ds_name, fmt_key, local, url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--formats", nargs="*", default=None)
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the local BLAKE3 recompute (faster on large datasets; "
             "reports NO_SIDECAR / OK_NO_VERIFY instead of OK / DRIFT).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = []
    mismatches = 0
    for ds_name, fmt_key, local, url in _iter_targets(args.datasets, args.formats):
        status, note = _status(local, url, compute=not args.no_verify)
        rows.append((ds_name, fmt_key, status, note, url))
        if status in ("NOT_UPLOADED", "NO_SIDECAR", "DRIFT", "PROBE_FAIL"):
            mismatches += 1

    # Markdown table on stdout so it pastes cleanly into PR comments.
    print("| Dataset | Format | Status | Note |")
    print("|---|---|---|---|")
    for ds, fmt, status, note, _url in rows:
        print(f"| {ds} | `{fmt}` | {status} | {note} |")

    print()
    print(
        f"{len(rows)} pairs checked — {mismatches} mismatch(es). "
        "Re-upload with `setup_cloud_test_data.sh`."
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
