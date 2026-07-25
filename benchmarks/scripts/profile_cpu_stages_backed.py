#!/usr/bin/env python
"""Phase-2 task 2.0 — CPU stage-profile capture on the *backed streaming* path.

The comprehensive `accel_*` benchmarks load fixtures with `anndata.read_h5ad`
(in-memory), so their timed op never decodes SCX shards — the `decode`/`io`
buckets are structurally ~0 and they cannot rank the §5.1 decode-prefetch work.
This capture instead opens each dataset **backed** (`pyscx.open(scx).to_anndata(
backed=True)`) and runs the accelerator ops over the streaming shard source, so
the profiler records the real decode / I-O / reduction / marshalling split of the
out-of-core path — the ranking oracle the Phase-2 tasks are gated on.

Requires `SCX_CPU_PROFILE=1` in the environment (read once at profiler init).
Writes one manifest-shaped result JSON per (op, dataset) into `results/raw/`
(benchmark = `accel_cpu_profile_backed`, format = `<op>_backed`) with the
`cpu_profile_*` breakdown in `runs[].extra`, and prints a summary table.

Usage:
    SCX_CPU_PROFILE=1 python benchmarks/scripts/profile_cpu_stages_backed.py \
        --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.bench_env import DATA_DIR  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult, write_result  # noqa: E402

OPS = ("pca", "hvg", "normalize", "qc", "filter_genes")
BENCHMARK = "accel_cpu_profile_backed"

# Gene subsets for the `qc` op. Three qc_vars is the common analyst call
# (`["mt", "ribo", "hb"]`); before the Phase-4.1 fusion each one cost its own
# full shard decode on top of the four axis statistics, so the decode-bucket
# count is the headline signal for this op.
QC_VARS = ("mt", "ribo", "hb")


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _scx_path(dataset: str) -> Path:
    return DATA_DIR / f"{dataset}_auto.scx"


def _run_op(op: str, scx_path: Path, n_comps: int) -> None:
    """Open backed and run one accelerator op over the streaming shard source."""
    import pyscx

    adata = pyscx.open(str(scx_path)).to_anndata(backed=True)
    if op == "pca":
        pyscx.accel.pca(adata, n_comps=n_comps, device="cpu")
    elif op == "hvg":
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=2000, flavor="seurat_v3", device="cpu",
        )
    elif op == "normalize":
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        _ = adata.X[:, :]  # force materialization of the lazy transform
    elif op == "qc":
        _assign_qc_masks(adata)
        pyscx.accel.calculate_qc_metrics(adata, qc_vars=list(QC_VARS))
    elif op == "filter_genes":
        # Both thresholds → exercises the fused column pass. Values are
        # deliberately permissive: the cost is the full scan, not the cut.
        pyscx.accel.filter_genes(adata, min_cells=1, min_counts=1.0)
    else:
        raise ValueError(op)


def _assign_qc_masks(adata) -> None:
    """Attach three deterministic gene subsets as boolean `var` columns.

    Real MT/ribo/hb prefixes are absent from several fixtures (Census var_names
    are integer strings), so select by position instead — the accumulator cost
    depends on subset *size*, not on which genes are in it.
    """
    import numpy as np

    n_vars = adata.n_vars
    idx = np.arange(n_vars)
    # ~1% / ~5% / ~0.5% of genes, echoing typical MT / ribo / hb fractions.
    for name, stride in zip(QC_VARS, (100, 20, 200)):
        adata.var[name] = (idx % stride) == 0


def _snapshot_flat() -> dict[str, float]:
    import pyscx

    snap = pyscx.accel.cpu_profile_snapshot()
    out: dict[str, float] = {"cpu_profile_enabled": 1.0 if snap.get("enabled") else 0.0}
    for b in ("io", "decode_scx1", "decode_generic", "reduction", "marshalling"):
        st = snap.get(b) or {}
        out[f"cpu_profile_{b}_ms"] = float(st.get("ms", 0.0))
        out[f"cpu_profile_{b}_count"] = float(st.get("count", 0))
        out[f"cpu_profile_{b}_bytes"] = float(st.get("bytes", 0))
    return out


def capture(dataset: str, op: str, n_runs: int, n_comps: int) -> BenchmarkResult | None:
    import pyscx

    scx_path = _scx_path(dataset)
    if not scx_path.exists():
        print(f"  SKIP {op}/{dataset}: {scx_path} missing", file=sys.stderr)
        return None

    result = BenchmarkResult(benchmark=BENCHMARK, format=f"{op}_backed", dataset=dataset)
    result.metadata["source"] = "backed_streaming"
    result.metadata["n_comps"] = n_comps

    # Warmup (page cache + one-time init), discarded.
    try:
        _run_op(op, scx_path, n_comps)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {op}/{dataset} warmup: {e}", file=sys.stderr)
        return None

    for _ in range(n_runs):
        gc.collect()
        pyscx.accel.cpu_profile_reset()
        rss0 = _peak_rss_mb()
        t0 = time.perf_counter()
        _run_op(op, scx_path, n_comps)
        wall = time.perf_counter() - t0
        extra = _snapshot_flat()
        result.add_run(wall_s=wall, peak_rss_mb=max(rss0, _peak_rss_mb()), **extra)

    write_result(result)
    return result


def _fmt_row(dataset: str, op: str, r: BenchmarkResult) -> str:
    import statistics

    def med(key: str) -> float:
        vals = [run.extra.get(key, 0.0) for run in r.runs]
        return statistics.median(vals) if vals else 0.0

    wall_ms = (r.median_wall_s or 0.0) * 1e3
    dec = med("cpu_profile_decode_scx1_ms") + med("cpu_profile_decode_generic_ms")
    io = med("cpu_profile_io_ms")
    red = med("cpu_profile_reduction_ms")
    mar = med("cpu_profile_marshalling_ms")
    # Shard-decode COUNT, not just time: for the multi-pass ops (qc,
    # filter_genes) the number of passes over the matrix is what a pass-fusion
    # change moves. Note this is `passes x n_shards`, so it is NOT comparable
    # across datasets — pbmc10k has 1 shard (qc -> 2) while census_1m has 62
    # (qc -> 124). Divide by the dataset's shard count to recover the passes.
    n_dec = med("cpu_profile_decode_scx1_count") + med("cpu_profile_decode_generic_count")
    tot = io + dec + red + mar
    frac = f"{tot / wall_ms:.0%}" if wall_ms else "n/a"
    return (f"| {dataset} | {op} | {wall_ms:.1f} | {io:.1f} | {dec:.1f} | "
            f"{red:.1f} | {mar:.1f} | {n_dec:.0f} | {frac} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--n-comps", type=int, default=50)
    args = ap.parse_args()

    if os.environ.get("SCX_CPU_PROFILE", "") in ("", "0"):
        print("ERROR: set SCX_CPU_PROFILE=1 before running (read at profiler init).",
              file=sys.stderr)
        return 2

    rows = []
    for dataset in args.datasets:
        for op in args.ops:
            print(f"capturing {op}/{dataset} ...", file=sys.stderr)
            r = capture(dataset, op, args.n_runs, args.n_comps)
            if r is not None:
                rows.append(_fmt_row(dataset, op, r))

    print("\n| dataset | op | wall_ms | io_ms | decode_ms | reduction_ms | "
          "marshalling_ms | n_decodes | Σ/wall |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
