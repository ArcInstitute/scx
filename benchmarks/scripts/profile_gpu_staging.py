#!/usr/bin/env python
"""Phase-4 task 4.2 — GPU staging host-decode profile.

Until 4.2, `scx-gpu`'s `GpuShardSource` decoded on **one** scoped worker thread
one shard ahead of the stager (finding §9.12): a single CPU decode thread
feeding an H100, which makes every GPU streaming op host-decode-bound. It now
uses the shared bounded decode-prefetch pipeline, so `depth` shards decode
concurrently while consumption stays on the calling thread in shard order.

This captures the `gpu_profile` buckets that show whether that moved the
needle. The A/B knob is the same one the CPU capture uses --
`SCX_ACCEL_PREFETCH_DEPTH=1` disables the pipeline and restores sequential
decode on the staging thread.

Two things about the host-decode number, both of which make the raw
milliseconds misleading on their own:

* It is a **sum over concurrent workers**, so with prefetch on it can exceed
  wall-clock. Compare the ratio against wall, not the absolute.
* The GPU-side buckets (HTOD, compute) are what should *not* move. If they do,
  something other than host decode changed.

Run under `sbatch` on a GPU node, in the `scx-bench-gpu` conda env::

    SCX_ACCEL_PREFETCH_DEPTH=1 python benchmarks/scripts/profile_gpu_staging.py
    SCX_ACCEL_PREFETCH_DEPTH=4 python benchmarks/scripts/profile_gpu_staging.py

Env overrides: GPU_STAGE_DATASETS (comma-separated), GPU_STAGE_OPS
(comma-separated subset of hvg,de), GPU_STAGE_RUNS.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.bench_env import DATA_DIR  # noqa: E402

# Multi-shard only: the pipeline no-ops on a single-shard file, exactly as the
# 2.1 capture found, so a small fixture would report a flat "no change" that
# says nothing about the change.
DATASETS = tuple(
    d.strip()
    for d in os.environ.get(
        "GPU_STAGE_DATASETS", "tabula_sapiens_100k,census_500k,census_1m"
    ).split(",")
    if d.strip()
)
OPS = tuple(
    o.strip() for o in os.environ.get("GPU_STAGE_OPS", "hvg,de").split(",") if o.strip()
)
N_RUNS = int(os.environ.get("GPU_STAGE_RUNS", 3))


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _flat(snap: dict) -> dict[str, float]:
    out: dict[str, float] = {"gpu_profile_enabled": 1.0 if snap.get("enabled") else 0.0}
    for bucket in (
        "host_decode_scx1",
        "host_decode_generic",
        "htod_scx1",
        "htod_generic",
        "gpu_decode",
        "compute",
    ):
        st = snap.get(bucket) or {}
        out[f"{bucket}_ms"] = float(st.get("ms", 0.0))
        out[f"{bucket}_count"] = float(st.get("count", 0) or st.get("shards", 0))
        out[f"{bucket}_bytes"] = float(st.get("bytes", 0))
    return out


def _run_op(op: str, scx_path: Path) -> None:
    import pyscx

    adata = pyscx.open(str(scx_path)).to_anndata(backed=True)
    if op == "hvg":
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=2000, flavor="seurat_v3", device="gpu"
        )
    elif op == "de":
        # A two-level grouping over the obs axis; DE streams every shard per
        # gene chunk, which is the path §9.11 and §9.12 both touch.
        import numpy as np

        adata.obs["grp"] = np.where(np.arange(adata.n_obs) % 2 == 0, "a", "b")
        pyscx.accel.rank_genes_groups(adata, "grp", device="gpu")
    else:
        raise ValueError(op)


def main() -> int:
    import pyscx

    depth = os.environ.get("SCX_ACCEL_PREFETCH_DEPTH", "<unset>")
    print(f"=== GPU staging profile (SCX_ACCEL_PREFETCH_DEPTH={depth}) ===")
    print(f"datasets: {', '.join(DATASETS)}")
    print(f"ops     : {', '.join(OPS)}   runs: {N_RUNS}")
    print()

    rows = []
    for dataset in DATASETS:
        scx_path = DATA_DIR / f"{dataset}_auto.scx"
        if not scx_path.exists():
            print(f"  SKIP {dataset}: {scx_path} not found")
            continue
        for op in OPS:
            walls = []
            snaps = []
            for _ in range(N_RUNS):
                pyscx.accel.gpu_profile_reset()
                t0 = time.perf_counter()
                try:
                    _run_op(op, scx_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {dataset}/{op}: {type(exc).__name__}: {exc}")
                    walls = []
                    break
                walls.append(time.perf_counter() - t0)
                snaps.append(_flat(pyscx.accel.gpu_profile_snapshot()))
            if not walls:
                continue
            walls.sort()
            median = walls[len(walls) // 2]
            snap = snaps[len(walls) // 2]
            hd = snap["host_decode_scx1_ms"] + snap["host_decode_generic_ms"]
            htod = snap["htod_scx1_ms"] + snap["htod_generic_ms"]
            row = {
                "dataset": dataset,
                "op": op,
                "depth": depth,
                "wall_ms": median * 1000.0,
                "host_decode_ms": hd,
                "host_decode_over_wall": hd / (median * 1000.0) if median else 0.0,
                "htod_ms": htod,
                "compute_ms": snap["compute_ms"],
                "peak_rss_mb": _peak_rss_mb(),
            }
            rows.append(row)
            print(
                f"  {dataset:<22} {op:<4} wall {row['wall_ms']:9.1f} ms | "
                f"host-decode {hd:9.1f} ms ({row['host_decode_over_wall'] * 100:5.1f}% of wall) | "
                f"HTOD {htod:8.1f} ms | compute {snap['compute_ms']:8.1f} ms | "
                f"peak {row['peak_rss_mb']:.0f} MB"
            )

    print()
    print(json.dumps(rows, indent=2))
    if not rows:
        print("NOTE: nothing captured — no dataset produced a usable run")
        return 1
    if not rows[0] and not any(r["host_decode_ms"] for r in rows):
        print("NOTE: set SCX_GPU_PROFILE=1 to populate the gpu_profile buckets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
