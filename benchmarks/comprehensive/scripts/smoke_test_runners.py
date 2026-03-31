#!/usr/bin/env python3
"""Smoke test for all format runners on pbmc3k.

Usage:
    python -m benchmarks.comprehensive.scripts.smoke_test_runners [--data-dir /path/to/datasets]

Runs convert → file_size → read_full → read_subset for each runner that has
its dependencies installed. Skips runners whose libraries are missing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.config import DATA_DIR


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def smoke_test_runner(runner, h5ad_path: Path, tmp_dir: Path) -> bool:
    """Run the four core operations and return True if all pass."""
    tag = runner.key
    ok = True

    # convert
    out_path = tmp_dir / f"{tag}_output"
    try:
        cr = runner.convert_from_h5ad(h5ad_path, out_path)
        _log(f"convert: {cr.wall_s:.2f}s, {cr.output_size_bytes} bytes")
    except Exception as exc:
        _log(f"convert FAILED: {exc}")
        return False

    # file_size
    try:
        sz = runner.file_size(out_path)
        _log(f"file_size: {sz} bytes")
    except Exception as exc:
        _log(f"file_size FAILED: {exc}")
        ok = False

    # read_full
    try:
        tr = runner.read_full(out_path)
        _log(f"read_full: {tr.wall_s:.2f}s, RSS {tr.peak_rss_mb:.1f} MB")
    except Exception as exc:
        _log(f"read_full FAILED: {exc}")
        ok = False

    # read_subset (10 random cells, 50 random genes)
    rng = np.random.default_rng(42)
    cell_idx = rng.choice(2700, size=10, replace=False).tolist()
    gene_idx = rng.choice(32738, size=50, replace=False).tolist()
    try:
        tr = runner.read_subset(out_path, cell_indices=cell_idx, gene_indices=gene_idx)
        _log(f"read_subset: {tr.wall_s:.2f}s, RSS {tr.peak_rss_mb:.1f} MB")
    except Exception as exc:
        _log(f"read_subset FAILED: {exc}")
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test benchmark runners")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    h5ad_path = args.data_dir / "pbmc3k.h5ad"
    if not h5ad_path.exists():
        print(f"ERROR: pbmc3k.h5ad not found at {h5ad_path}")
        sys.exit(1)

    runners = []

    # h5ad variants
    from benchmarks.comprehensive.runners.h5ad_runner import H5adRunner
    for comp in [None, "gzip", "lzf"]:
        runners.append(H5adRunner(compression=comp))

    # Zarr variants
    try:
        from benchmarks.comprehensive.runners.zarr_runner import ZarrRunner
        runners.append(ZarrRunner(compressor="zstd", level=3))
        runners.append(ZarrRunner(compressor="lz4", level=5))
    except ImportError as exc:
        print(f"SKIP Zarr runners: {exc}")

    # TileDB-SOMA
    try:
        from benchmarks.comprehensive.runners.tiledb_runner import TileDBRunner
        runners.append(TileDBRunner())
    except (ImportError, RuntimeError) as exc:
        print(f"SKIP TileDB runner: {exc}")

    # SCX variants
    try:
        from benchmarks.comprehensive.runners.scx_runner import ScxRunner
        for codec in ["auto", "none", "scx1", "zstd"]:
            runners.append(ScxRunner(codec=codec))
    except (ImportError, RuntimeError) as exc:
        print(f"SKIP SCX runners: {exc}")

    # BPCells
    try:
        from benchmarks.comprehensive.runners.bpcells_runner import BPCellsRunner
        runners.append(BPCellsRunner())
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"SKIP BPCells runner: {exc}")

    # Parquet
    try:
        from benchmarks.comprehensive.runners.parquet_runner import ParquetRunner
        runners.append(ParquetRunner(compression="zstd"))
    except ImportError as exc:
        print(f"SKIP Parquet runner: {exc}")

    print(f"\nRunning smoke tests on {h5ad_path} ({len(runners)} runners)\n")

    passed = 0
    failed = 0
    with tempfile.TemporaryDirectory(prefix="scx_smoke_") as tmp:
        tmp_dir = Path(tmp)
        for runner in runners:
            print(f"--- {runner.name} ({runner.key}) ---")
            try:
                if smoke_test_runner(runner, h5ad_path, tmp_dir):
                    print(f"  PASS\n")
                    passed += 1
                else:
                    print(f"  FAIL (partial)\n")
                    failed += 1
            except Exception as exc:
                print(f"  FAIL: {exc}\n")
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
