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
    """Run the core + declared-optional operations; return True if all pass.

    Phase I.8: probe every capability the runner's ``capabilities`` set
    advertises. A declared capability that raises ``NotImplementedError``
    is a contract violation and fails the smoke test loudly.
    """
    tag = runner.key
    ok = True
    caps = getattr(runner, "capabilities", frozenset())

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

    # read_full + shape/nnz capture as the baseline for parity checks below.
    baseline_shape = None
    baseline_nnz = None
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

    # Optional capability probes — each declared capability must have a
    # working method, else the runner's manifest is lying. Contract
    # violations fail the smoke loudly (vs the old silent-skip on NotImpl).
    if "backed_mode" in caps:
        try:
            tr = runner.read_backed(out_path)
            _log(f"read_backed (declared): {tr.wall_s:.3f}s")
        except NotImplementedError as exc:
            _log(f"read_backed CONTRACT VIOLATION (capability declared): {exc}")
            ok = False
        except Exception as exc:
            _log(f"read_backed raised: {exc}")
            ok = False
        try:
            tr = runner.read_backed_slice(out_path, start=0, count=10)
            _log(f"read_backed_slice (declared): {tr.wall_s:.3f}s")
        except NotImplementedError as exc:
            _log(f"read_backed_slice CONTRACT VIOLATION: {exc}")
            ok = False
        except Exception as exc:
            _log(f"read_backed_slice raised: {exc}")
            ok = False

    if "filtered_query" in caps:
        from benchmarks.comprehensive.queries import RandomSamplePredicate
        try:
            tr = runner.read_filtered_query(
                out_path, RandomSamplePredicate(fraction=0.01, seed=42, name="smoke_rand"),
            )
            _log(f"read_filtered_query (declared): {tr.wall_s:.3f}s mech="
                 f"{(tr.extra or {}).get('native_mechanism', '?')}")
        except NotImplementedError as exc:
            _log(f"read_filtered_query CONTRACT VIOLATION: {exc}")
            ok = False
        except Exception as exc:
            _log(f"read_filtered_query raised: {exc}")
            ok = False

    # Cloud capabilities — smoke against a local .scxd-style directory is
    # impractical without real fixtures; declared capabilities are probed
    # by checking the method exists and raises the right default. True
    # cloud coverage lives in the cloud_* benchmarks' own CI runs.
    for cap, method_name in (
        ("cloud_read", "read_cloud"),
        ("cloud_subset", "read_cloud_subset"),
        ("cloud_push", "push"),
        ("cloud_pull", "pull"),
        ("cloud_filtered", "read_cloud_filtered_query"),
    ):
        if cap in caps:
            fn = getattr(runner, method_name, None)
            if fn is None or fn.__qualname__.startswith("FormatRunner."):
                _log(f"{method_name} CONTRACT VIOLATION: capability {cap!r} "
                     f"declared but method not overridden")
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
