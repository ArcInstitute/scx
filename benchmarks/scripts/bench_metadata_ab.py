#!/usr/bin/env python3
"""
T4 macro A/B — obs/var parallel metadata decode, end-to-end `to_gpu_anndata`.

The `to_gpu_anndata` wall is metadata-assembly-bound (Phase-2.x profiling: the
codec-agnostic obs/var assembly + cupy handoff dominate, not X decode). T4
parallelized the per-shard obs/var decode (`reader::read_sharded_layout_by_prefix`).
This script isolates T4's contribution to that wall: it times
`pyscx.open(p).to_gpu_anndata(device="gpu")` on sharded-obs fixtures with the
parallel decode ON (default) vs forced serial via `SCX_METADATA_DECODE_SERIAL=1`
(read per-call by the Rust reader, so both run in one process).

Companion to `bench_gpu_codec.py` (reuses its DATA_DIR + profile helpers); run on
a GPU node under conda `scx-bench-gpu` (see `slurm_bench_gpu_codec.sh`, which
invokes this after the decode matrix).

Usage:
    python benchmarks/scripts/bench_metadata_ab.py \
        --fixtures census_1m_scx1 census_1m_compact_trial --n-runs 3 \
        --out-dir benchmarks/results/gpu_codec
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.comprehensive.bench_env import DATA_DIR  # noqa: E402


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _profile_reset(pyscx) -> None:
    try:
        pyscx.accel.gpu_profile_reset()
    except Exception:
        pass


def _time_to_gpu(pyscx, path: Path, serial: bool, n_runs: int, n_warmup: int) -> dict | None:
    """Median `to_gpu_anndata` wall with metadata decode serial vs parallel.

    `SCX_METADATA_DECODE_SERIAL` is read per-call by the reader, so toggling
    os.environ here flips the path in-process (no rebuild / restart)."""
    if serial:
        os.environ["SCX_METADATA_DECODE_SERIAL"] = "1"
    else:
        os.environ.pop("SCX_METADATA_DECODE_SERIAL", None)

    if not path.exists():
        print(f"  [skip] {path.name}: fixture missing", file=sys.stderr)
        return None

    try:
        for _ in range(n_warmup):
            warm = pyscx.open(str(path)).to_gpu_anndata(device="gpu")
            del warm
            gc.collect()
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {path.name} (serial={serial}): warm-up failed ({exc})", file=sys.stderr)
        return {"fixture": path.name, "serial": serial, "error": f"warmup: {exc}"}

    walls: list[float] = []
    transfer_mode = None
    n_obs = n_vars = nnz = 0
    for i in range(n_runs):
        gc.collect()
        _profile_reset(pyscx)
        t0 = time.perf_counter()
        adata = pyscx.open(str(path)).to_gpu_anndata(device="gpu")
        wall = time.perf_counter() - t0
        walls.append(wall)
        info = adata.uns["scx_accel"]["to_gpu_anndata"]
        transfer_mode = info.get("transfer_mode")
        n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])
        nnz = int(adata.X.nnz)
        print(
            f"  {path.name}[{'serial' if serial else 'parallel'}] run {i + 1}/{n_runs}: "
            f"wall={wall:.3f}s transfer_mode={transfer_mode}"
        )
        del adata
        gc.collect()

    os.environ.pop("SCX_METADATA_DECODE_SERIAL", None)
    return {
        "fixture": path.name,
        "serial": serial,
        "median_wall_s": _median(walls),
        "walls_s": walls,
        "transfer_mode": transfer_mode,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "nnz": nnz,
        "n_runs": n_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        nargs="+",
        default=["census_1m_scx1", "census_1m_compact_trial"],
        help="fixture basenames under $SCX_DATA_DIR (without .scx)",
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "gpu_codec",
    )
    args = parser.parse_args()

    try:
        import pyscx
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: pyscx import failed: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        import cupy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cupy import failed (need a GPU node): {exc}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    for fixture in args.fixtures:
        path = DATA_DIR / f"{fixture}.scx"
        print(f"\n=== {fixture} — metadata decode serial vs parallel ===")
        # Parallel first (warm the page cache), then serial on the same warm cache
        # so the A/B isolates decode parallelism, not disk.
        par = _time_to_gpu(pyscx, path, serial=False, n_runs=args.n_runs, n_warmup=args.n_warmup)
        ser = _time_to_gpu(pyscx, path, serial=True, n_runs=args.n_runs, n_warmup=args.n_warmup)
        if par:
            records.append(par)
        if ser:
            records.append(ser)
        if par and ser and "median_wall_s" in par and "median_wall_s" in ser:
            p, s = par["median_wall_s"], ser["median_wall_s"]
            speedup = s / p if p > 0 else float("nan")
            print(
                f"  → {fixture}: parallel {p:.3f}s vs serial {s:.3f}s "
                f"→ {speedup:.2f}x faster (Δ {s - p:.3f}s off the to_gpu_anndata wall)"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "metadata_ab.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
