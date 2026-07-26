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

OPS = (
    "pca",
    "pca_hvg",
    "hvg",
    "normalize",
    "qc",
    "filter_genes",
    "filter_cells",
    "filter_genes_real",
    "subset_obs",
)
BENCHMARK = "accel_cpu_profile_backed"

# `filter_cells` / `filter_genes_real` / `subset_obs` were added for the Phase-4
# task 4.0b capture, where the question is not "how fast is the scan" but "what
# does the axis subset itself cost". 4.0b hands the subset to anndata, which
# builds a whole replacement AnnData (`_mutated_copy` deep-copies `uns` and
# copies `obs` / `var` / every non-handle aligned member) and swaps it in — so
# both the old and the new frames are live at once. At census scale that
# transient is the thing worth measuring, and `subset_obs` isolates it: a fixed
# mask, no threshold scan, so wall and peak RSS are open + rebuild and nothing
# else.
#
# The pre-existing `filter_genes` thresholds (`min_cells=1, min_counts=1.0`) are
# deliberately permissive — 4.1 wanted the *scan*, not the cut — which means
# they may keep every gene. That is fine for 4.1's pass-count question but makes
# them useless for 4.0b's, because a subset that keeps everything is now
# correctly skipped. `filter_genes_real` uses scanpy's canonical `min_cells=3`
# so the cut actually happens. Every op records `kept_frac_*` in `extra`, so a
# no-op cut is visible in the capture rather than being read as a speedup.
SUBSET_OBS_STRIDE = 2  # keep every other cell

# Gene subsets for the `qc` op. Three qc_vars is the common analyst call
# (`["mt", "ribo", "hb"]`); before the Phase-4.1 fusion each one cost its own
# full shard decode on top of the four axis statistics, so the decode-bucket
# count is the headline signal for this op.
QC_VARS = ("mt", "ribo", "hb")

# `pca_hvg` exists because `pca` on these fixtures never reaches the covariance
# route. `pyscx.accel.pca` picks covariance when the effective var count is
# <= COVARIANCE_PCA_THRESHOLD (5000) and randomized otherwise, and every dataset
# in the tier carries a full gene set — census_1m has 61,497 — so the whole
# capture measures one of the two routes. Masking to a fixed gene count puts the
# other one under measurement, and routes through `ProjectedShardSource` while
# it is there.
#
# The mask is a positional stride, not a real HVG selection: the question is
# which code path runs and how much it decodes, and a biologically meaningful
# choice would only make the two arms harder to compare. `mask_n_vars` is
# recorded per run so a fixture change that silently flips the route back is
# visible in the capture instead of being read as a speedup.
PCA_HVG_N_VARS = 2000
# `scx_accel::pca::COVARIANCE_PCA_THRESHOLD`, pinned by the doc-drift guard in
# tests/scx-integration-tests. Asserted rather than commented because exceeding
# it is silent: the op still runs, still reports a number, and measures the
# randomized route a second time.
COVARIANCE_PCA_THRESHOLD = 5000
assert PCA_HVG_N_VARS <= COVARIANCE_PCA_THRESHOLD, (
    f"pca_hvg masks to {PCA_HVG_N_VARS} genes, which routes to randomized PCA, "
    f"not the covariance route this op exists to cover"
)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _scx_path(dataset: str) -> Path:
    return DATA_DIR / f"{dataset}_auto.scx"


def _run_op(op: str, scx_path: Path, n_comps: int) -> dict[str, float]:
    """Open backed and run one accelerator op over the streaming shard source.

    Returns the shape before / after the op, so a capture can prove the op did
    the work it claims — an axis subset that kept every element is a no-op, and
    would otherwise read as a speedup.
    """
    import numpy as np
    import pyscx

    exp = pyscx.open(str(scx_path))
    # Grab the shard count while the Experiment is still in hand: the decode
    # bucket counts *shards decoded*, and only `count / n_shards` turns that
    # into the number that matters — how many times the op walked the matrix.
    n_shards = float(exp.shard_count)
    adata = exp.to_anndata(backed=True)
    n_obs0, n_vars0 = adata.n_obs, adata.n_vars
    if op == "filter_cells":
        # scanpy's canonical default; drops a real fraction of real data.
        pyscx.accel.filter_cells(adata, min_genes=200)
    elif op == "filter_genes_real":
        pyscx.accel.filter_genes(adata, min_cells=3)
    elif op == "subset_obs":
        keep = (np.arange(n_obs0) % SUBSET_OBS_STRIDE) == 0
        pyscx.accel.subset_obs(adata, keep)
    elif op == "pca":
        pyscx.accel.pca(adata, n_comps=n_comps, device="cpu")
    elif op == "pca_hvg":
        adata.var["highly_variable"] = _stride_mask(n_vars0, PCA_HVG_N_VARS)
        pyscx.accel.pca(
            adata, n_comps=n_comps, device="cpu", mask_var="highly_variable"
        )
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
    out = {
        "n_obs_before": float(n_obs0),
        "n_vars_before": float(n_vars0),
        "n_obs_after": float(adata.n_obs),
        "n_vars_after": float(adata.n_vars),
        "kept_frac_obs": adata.n_obs / n_obs0 if n_obs0 else 1.0,
        "kept_frac_vars": adata.n_vars / n_vars0 if n_vars0 else 1.0,
        "n_shards": n_shards,
    }
    if op == "pca_hvg":
        # Premise, not decoration: if this ever exceeded the routing ceiling the
        # op would silently measure the randomized route a second time and the
        # covariance route would go uncovered with nothing to show for it.
        out["mask_n_vars"] = float(int(adata.var["highly_variable"].sum()))
    return out


def _stride_mask(n_vars: int, keep: int):
    """A deterministic boolean mask over `keep` evenly-spaced genes."""
    import numpy as np

    idx = np.arange(n_vars)
    stride = max(1, n_vars // max(1, keep))
    mask = (idx % stride) == 0
    # Trim the tail so the count is exactly `keep` when the stride overshoots.
    surplus = int(mask.sum()) - keep
    if surplus > 0:
        mask[np.flatnonzero(mask)[-surplus:]] = False
    return mask


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
        shape = _run_op(op, scx_path, n_comps)
        wall = time.perf_counter() - t0
        extra = _snapshot_flat()
        extra.update(shape)
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
    # `ru_maxrss` is a process high-water mark, so take the max across runs —
    # the same convention the 4.1 capture used.
    peak = max((run.peak_rss_mb or 0.0) for run in r.runs) if r.runs else 0.0
    # Kept fraction makes a no-op cut visible: a subset that keeps everything is
    # correctly skipped, and would otherwise read as a speedup.
    kept = f"{med('kept_frac_obs'):.2f}/{med('kept_frac_vars'):.2f}"
    return (f"| {dataset} | {op} | {wall_ms:.1f} | {peak:.0f} | {kept} | {io:.1f} | {dec:.1f} | "
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

    print("\n| dataset | op | wall_ms | peak_rss_mb | kept obs/vars | io_ms | decode_ms | "
          "reduction_ms | marshalling_ms | n_decodes | Σ/wall |")
    print("|---|---|--:|--:|:-:|--:|--:|--:|--:|--:|--:|")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
