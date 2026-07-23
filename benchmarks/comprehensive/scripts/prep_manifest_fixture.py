#!/usr/bin/env python
"""Build a multi-file manifest fixture for the ``obs_open`` benchmark's
manifest-scale scenario (data-load Phase 0).

The real STATE3 obs-open cost is dominated by *per-file* open across manifests
of ~26k files (``basecount_homo_sapiens_train_int.csv``), which can't be
committed. This script shards one benchmark dataset into ``--n-files`` small
single-modality files (``.scx`` and ``.h5ad``) under
``DATA_DIR/manifest_<dataset>/`` plus a ``manifest.csv``, so ``obs_open`` can
measure per-file open cost at manifest scale and extrapolate.

Usage::

    conda activate scx-bench
    python benchmarks/comprehensive/scripts/prep_manifest_fixture.py \
        --dataset tabula_sapiens_100k --n-files 256

    # SCX only (skip the h5ad arm):
    python .../prep_manifest_fixture.py --dataset pbmc10k --n-files 64 --formats scx

The manifest CSV has one row per shard with ``scx_auto_path`` / ``h5ad_none_path``
columns; ``obs_open._manifest_paths`` filters by format + extension.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep_manifest_fixture")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    sys.path.insert(0, str(_repo_root()))
    from benchmarks.comprehensive.config import DATA_DIR, DATASETS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tabula_sapiens_100k", help="Source dataset name (config.DATASETS key)")
    ap.add_argument("--n-files", type=int, default=256, help="Number of shard files to emit")
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["scx", "h5ad"],
        choices=["scx", "h5ad"],
        help="Which per-shard formats to write",
    )
    ap.add_argument("--overwrite", action="store_true", help="Rewrite existing shards")
    args = ap.parse_args()

    if args.dataset not in DATASETS:
        logger.error("Unknown dataset %r; known: %s", args.dataset, ", ".join(sorted(DATASETS)))
        return 2
    ds = DATASETS[args.dataset]
    src = ds.h5ad_path
    if not src.exists():
        logger.error("Source h5ad missing: %s — run dataset prep first.", src)
        return 2

    out_dir = DATA_DIR / f"manifest_{ds.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "manifest.csv"

    import anndata
    import numpy as np

    want_scx = "scx" in args.formats
    if want_scx:
        import pyscx  # noqa: F401

    logger.info("Reading %s (backed) …", src)
    adata = anndata.read_h5ad(src, backed="r")
    n_obs = adata.n_obs
    n_files = max(1, min(args.n_files, n_obs))
    bounds = np.linspace(0, n_obs, n_files + 1, dtype=np.int64)
    logger.info("Sharding %d cells into %d files under %s", n_obs, n_files, out_dir)

    rows: list[dict[str, str]] = []
    for i in range(n_files):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue
        # Materialize just this row range (bounded memory).
        chunk = adata[lo:hi].to_memory()
        row: dict[str, str] = {"shard": str(i), "lo": str(lo), "hi": str(hi)}
        if "h5ad" in args.formats:
            h5p = out_dir / f"shard_{i:05d}.h5ad"
            if args.overwrite or not h5p.exists():
                chunk.write_h5ad(h5p)
            row["h5ad_none_path"] = str(h5p)
        if want_scx:
            import pyscx

            scxp = out_dir / f"shard_{i:05d}.scx"
            if args.overwrite or not scxp.exists():
                pyscx.from_anndata(chunk, str(scxp))
            row["scx_auto_path"] = str(scxp)
        rows.append(row)
        if (i + 1) % 25 == 0:
            logger.info("  … %d/%d shards", i + 1, n_files)

    if getattr(adata, "isbacked", False) and adata.file is not None:
        adata.file.close()

    fieldnames = ["shard", "lo", "hi"]
    if "h5ad" in args.formats:
        fieldnames.append("h5ad_none_path")
    if want_scx:
        fieldnames.append("scx_auto_path")
    with open(manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    logger.info("Wrote manifest %s (%d shards, formats=%s)", manifest_csv, len(rows), args.formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
