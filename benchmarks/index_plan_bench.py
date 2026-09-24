#!/usr/bin/env python3
"""Phase 7 — `IndexPlanDataset` throughput benchmark.

Self-contained driver covering:

  - **Mode A** (random pairing): IndexPlanDataset with uniformly-random
    `(pert, ctrl)` pairs. Pessimistic locality. Stresses shard cache.
  - **Mode B** (controlled-locality): IndexPlanDataset with `(cell_type, batch)`-
    keyed pairs (sample both pert and ctrl from the same group). Realistic
    locality, the canonical ST training workload.
  - **Backed Python baseline**: `pyscx.ScxBackedSparseDataset` with a Python
    loop that gathers rows + applies projection/normalize manually. The
    current cell-load-scx path that this dataset is meant to replace.
  - **TrainingDataset ceiling**: `pyscx.TrainingDataset` sequential pipeline.
    Different access pattern but the throughput ceiling for SCX reads.

Default workload is sized for laptop-class quick iteration. The 1M-cell
workload is enabled with `--n-cells 1_000_000` (single run takes
several minutes; allow ~10 GB disk for the fixture).

Switches surfacing the locality optimisation deltas:
  - shard sort: `--config sort_off` vs `--config sort_on`.
  - Phase 4 (lookahead): `--config lookahead0` vs `--config lookahead4`.
  - Phase 5 (zero-allocation dense gather) is exercised by every
    HVG-projected configuration; see `docs/performance/loader-index-plan.md`
    § "Zero-allocation dense gather" for A/B numbers.

Output goes to stdout (and optional `--output FILE.json`).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def build_fixture(
    path: Path,
    n_cells: int,
    n_vars: int,
    density: float = 0.05,
    n_cell_types: int = 5,
    n_batches: int = 8,
    seed: int = 42,
) -> dict:
    """Write a synthetic .scx fixture with `cell_type` and `batch` categorical
    obs columns. Returns the obs dataframe so plan generators can sample from
    `(cell_type, batch)` groups without re-reading the file."""
    import anndata
    import pyscx

    if path.exists():
        # Already built — read obs back. AnnData needs the file but for our
        # purposes we just need the obs metadata.
        from pyscx import ScxBackedSparseDataset  # noqa: F401

        # Cheapest: pickle alongside the .scx.
        meta_path = path.with_suffix(".scx.meta.parquet")
        if meta_path.exists():
            obs = pd.read_parquet(meta_path)
            return {"obs": obs}

    rng = np.random.default_rng(seed)
    print(f"Building fixture {n_cells} × {n_vars} (density={density}) → {path}", flush=True)
    t0 = time.perf_counter()

    # Sparse integer counts (scipy CSR) — generated row-by-row so we don't
    # materialize a dense (n_cells, n_vars) array.
    nnz_per_row = max(1, int(n_vars * density))
    indptr = np.arange(0, (n_cells + 1) * nnz_per_row, nnz_per_row, dtype=np.int64)
    cols = np.empty(n_cells * nnz_per_row, dtype=np.int32)
    vals = np.empty(n_cells * nnz_per_row, dtype=np.float32)
    for r in range(n_cells):
        c = rng.choice(n_vars, size=nnz_per_row, replace=False)
        c.sort()
        cols[r * nnz_per_row : (r + 1) * nnz_per_row] = c
        vals[r * nnz_per_row : (r + 1) * nnz_per_row] = rng.integers(
            1, 200, size=nnz_per_row
        ).astype(np.float32)
    x = sp.csr_matrix((vals, cols, indptr), shape=(n_cells, n_vars))

    cell_types = rng.choice(
        [f"type_{i}" for i in range(n_cell_types)], size=n_cells
    )
    batches = rng.choice([f"batch_{i}" for i in range(n_batches)], size=n_cells)
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "cell_type": pd.Categorical(cell_types),
            "batch": pd.Categorical(batches),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    ad = anndata.AnnData(X=x, obs=obs, var=var)

    pyscx.from_anndata(ad, str(path))
    obs.to_parquet(path.with_suffix(".scx.meta.parquet"))

    print(f"  fixture built in {time.perf_counter() - t0:.1f}s", flush=True)
    return {"obs": obs}


# ---------------------------------------------------------------------------
# Plan generators
# ---------------------------------------------------------------------------


def gen_random_plans(
    n_cells: int, n_pairs_per_batch: int, n_batches: int, seed: int = 0
) -> Iterator[list[tuple[int, int]]]:
    """Mode A — uniformly random pert/ctrl indices. Pessimistic locality."""
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        pert = rng.integers(0, n_cells, size=n_pairs_per_batch)
        ctrl = rng.integers(0, n_cells, size=n_pairs_per_batch)
        yield list(zip(map(int, pert), map(int, ctrl)))


def gen_locality_plans(
    obs: pd.DataFrame,
    n_pairs_per_batch: int,
    n_batches: int,
    seed: int = 0,
) -> Iterator[list[tuple[int, int]]]:
    """Mode B — `(cell_type, batch)`-keyed pairs. Both pert and ctrl come from
    the same `(cell_type, batch)` group; this is the realistic ST training
    workload (the `BatchMappingStrategy` pattern in cell-load-scx)."""
    rng = np.random.default_rng(seed)
    obs_int = obs.reset_index(drop=True)
    groups = obs_int.groupby(["cell_type", "batch"], observed=True).indices
    group_keys = list(groups.keys())
    if not group_keys:
        raise ValueError("no (cell_type, batch) groups found in obs")
    for _ in range(n_batches):
        plan = []
        for _ in range(n_pairs_per_batch):
            key = group_keys[rng.integers(0, len(group_keys))]
            members = groups[key]
            pert = int(members[rng.integers(0, len(members))])
            ctrl = int(members[rng.integers(0, len(members))])
            plan.append((pert, ctrl))
        yield plan


# ---------------------------------------------------------------------------
# RSS helper
# ---------------------------------------------------------------------------


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    n_batches: int
    n_pairs_per_batch: int
    n_output_cols: int
    elapsed_s: float
    batches_per_sec: float
    cells_per_sec: float  # pairs × 2 (pert+ctrl) × batches / elapsed
    peak_rss_mb: float
    notes: str = ""

    def to_row(self) -> str:
        return (
            f"{self.name:<32}"
            f"{self.batches_per_sec:>10.1f} bps  "
            f"{self.cells_per_sec:>12,.0f} cps  "
            f"{self.peak_rss_mb:>8.0f} MB  "
            f"{self.elapsed_s:>7.2f}s  "
            f"{self.notes}"
        )


# ---------------------------------------------------------------------------
# Bench: IndexPlanDataset
# ---------------------------------------------------------------------------


def bench_index_plan(
    name: str,
    path: str,
    plans_factory: Callable[[], Iterator[list[tuple[int, int]]]],
    n_pairs_per_batch: int,
    n_batches: int,
    *,
    sort_by_shard: bool = True,
    lookahead: int = 4,
    cache_shards: int = 128,
    hvg_indices: np.ndarray | None = None,
    normalize: bool = True,
    notes: str = "",
) -> BenchResult:
    import pyscx

    gc.collect()
    rss0 = peak_rss_mb()

    ds = pyscx.IndexPlanDataset(
        path,
        hvg_indices=hvg_indices,
        normalize=normalize,
        cache_shards=cache_shards,
        sort_by_shard=sort_by_shard,
        lookahead=lookahead,
        max_plan_size=max(n_pairs_per_batch, 16384),
        max_memory_mb=4096,
    )

    plans = plans_factory()
    t0 = time.perf_counter()
    seen = 0
    for batch in ds.iter_with_plans(plans, lookahead=lookahead):
        seen += 1
        # Touch the data so the optimizer doesn't elide the read.
        _ = batch["X"].shape[0]
    elapsed = time.perf_counter() - t0
    rss = peak_rss_mb() - rss0

    return BenchResult(
        name=name,
        n_batches=seen,
        n_pairs_per_batch=n_pairs_per_batch,
        n_output_cols=ds.n_output_genes,
        elapsed_s=elapsed,
        batches_per_sec=seen / elapsed if elapsed > 0 else float("inf"),
        cells_per_sec=(seen * n_pairs_per_batch * 2) / elapsed
        if elapsed > 0
        else float("inf"),
        peak_rss_mb=rss,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Bench: backed Python baseline (the ScxBackedSparseDataset path)
# ---------------------------------------------------------------------------


def bench_backed_python(
    name: str,
    path: str,
    plans_factory: Callable[[], Iterator[list[tuple[int, int]]]],
    n_pairs_per_batch: int,
    *,
    hvg_indices: np.ndarray | None,
    normalize: bool,
    target_sum: float = 1e4,
    notes: str = "",
) -> BenchResult:
    import pyscx

    gc.collect()
    rss0 = peak_rss_mb()

    # Open via the public API: pyscx.open + to_anndata(backed=True). adata.X
    # is the ScxBackedSparseDataset that cell-load-scx currently consumes.
    adata = pyscx.open(path).to_anndata(backed=True)
    ds = adata.X
    n_vars = ds.shape[1]
    n_out = len(hvg_indices) if hvg_indices is not None else n_vars

    plans = plans_factory()
    t0 = time.perf_counter()
    seen = 0
    for plan in plans:
        n = len(plan)
        all_idx = np.empty(2 * n, dtype=np.int64)
        for i, (p, c) in enumerate(plan):
            all_idx[i] = p
            all_idx[n + i] = c
        sub = ds[all_idx]
        dense = sub.toarray().astype(np.float32)
        if hvg_indices is not None:
            dense = dense[:, hvg_indices]
        if normalize:
            sums = dense.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            dense = np.log1p(dense * (target_sum / sums))
        pert = dense[:n]
        ctrl = dense[n:]
        _ = pert.shape, ctrl.shape
        seen += 1
    elapsed = time.perf_counter() - t0
    rss = peak_rss_mb() - rss0

    return BenchResult(
        name=name,
        n_batches=seen,
        n_pairs_per_batch=n_pairs_per_batch,
        n_output_cols=n_out,
        elapsed_s=elapsed,
        batches_per_sec=seen / elapsed if elapsed > 0 else float("inf"),
        cells_per_sec=(seen * n_pairs_per_batch * 2) / elapsed
        if elapsed > 0
        else float("inf"),
        peak_rss_mb=rss,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Bench: TrainingDataset (sequential ceiling)
# ---------------------------------------------------------------------------


def bench_training_seq(
    name: str,
    path: str,
    n_pairs_per_batch: int,
    n_batches_target: int,
    *,
    hvg_indices: np.ndarray | None,
    normalize: bool,
    notes: str = "",
) -> BenchResult:
    """Cap to the same total cells as the IndexPlan run for fairness."""
    import pyscx

    gc.collect()
    rss0 = peak_rss_mb()

    ds = pyscx.TrainingDataset(
        path,
        batch_size=2 * n_pairs_per_batch,  # pert + ctrl side combined per pair
        hvg_indices=list(hvg_indices) if hvg_indices is not None else None,
        normalize=normalize,
        log1p=normalize,
    )
    target = n_batches_target

    t0 = time.perf_counter()
    seen = 0
    for batch in ds:
        seen += 1
        _ = batch["X"].shape[0]
        if seen >= target:
            break
    elapsed = time.perf_counter() - t0
    rss = peak_rss_mb() - rss0

    n_out = ds.n_output_genes
    return BenchResult(
        name=name,
        n_batches=seen,
        n_pairs_per_batch=n_pairs_per_batch,
        n_output_cols=n_out,
        elapsed_s=elapsed,
        batches_per_sec=seen / elapsed if elapsed > 0 else float("inf"),
        cells_per_sec=(seen * 2 * n_pairs_per_batch) / elapsed
        if elapsed > 0
        else float("inf"),
        peak_rss_mb=rss,
        notes=notes + " (sequential, different access pattern)",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-cells", type=int, default=50_000)
    p.add_argument("--n-vars", type=int, default=2_000)
    p.add_argument("--n-pairs-per-batch", type=int, default=512)
    p.add_argument("--n-batches", type=int, default=200)
    p.add_argument("--hvg-size", type=int, default=2_000,
                   help="HVG projection size; set <= 0 to disable HVG (use full gene matrix).")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--fixture-dir", type=str, default=None,
                   help="Directory for the .scx fixture (default: a fresh tempdir).")
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON output path for results.")
    p.add_argument(
        "--config",
        type=str,
        default="all",
        help="Comma-separated subset of: all, index_plan_random, index_plan_locality, "
        "sort_off, sort_on, lookahead0, lookahead4, lookahead8, "
        "backed_python, training_seq",
    )
    args = p.parse_args()

    # --- fixture ---
    if args.fixture_dir is None:
        td = tempfile.mkdtemp(prefix="scx_index_plan_bench_")
    else:
        os.makedirs(args.fixture_dir, exist_ok=True)
        td = args.fixture_dir
    fixture_path = Path(td) / f"bench_{args.n_cells}x{args.n_vars}.scx"
    fixture_meta = build_fixture(fixture_path, args.n_cells, args.n_vars)
    obs = fixture_meta["obs"]

    hvg = (
        np.random.default_rng(0).choice(args.n_vars, size=args.hvg_size, replace=False).astype(np.uint32)
        if args.hvg_size and args.hvg_size > 0
        else None
    )
    if hvg is not None:
        hvg.sort()

    selected = (
        {"index_plan_random", "index_plan_locality", "backed_python", "training_seq",
         "sort_off", "sort_on", "lookahead0", "lookahead4", "lookahead8"}
        if args.config == "all"
        else set(s.strip() for s in args.config.split(","))
    )

    common = dict(
        n_pairs_per_batch=args.n_pairs_per_batch,
        n_batches=args.n_batches,
        hvg_indices=hvg,
        normalize=args.normalize,
    )
    path_str = str(fixture_path)
    results: list[BenchResult] = []

    def random_plans():
        return gen_random_plans(args.n_cells, args.n_pairs_per_batch, args.n_batches)

    def locality_plans():
        return gen_locality_plans(obs, args.n_pairs_per_batch, args.n_batches)

    def emit(r: BenchResult) -> None:
        results.append(r)
        print(r.to_row(), flush=True)

    print()
    print(
        f"=== IndexPlanDataset bench === "
        f"n_cells={args.n_cells} n_vars={args.n_vars} "
        f"hvg={args.hvg_size if hvg is not None else 'off'} "
        f"normalize={args.normalize} "
        f"pairs/batch={args.n_pairs_per_batch} batches={args.n_batches}"
    )
    print(f"{'config':<32}{'bps':>14} {'cps':>16} {'rss':>11} {'time':>8}  notes")
    print("-" * 100)

    # --- IndexPlanDataset Mode A (random) ---
    if "index_plan_random" in selected:
        emit(bench_index_plan(
            "index_plan_random", path_str, random_plans, **common,
            sort_by_shard=True, lookahead=4, notes="Mode A (random pairing)"
        ))

    # --- IndexPlanDataset Mode B (locality) ---
    if "index_plan_locality" in selected:
        emit(bench_index_plan(
            "index_plan_locality", path_str, locality_plans, **common,
            sort_by_shard=True, lookahead=4, notes="Mode B (cell_type+batch keyed)"
        ))

    # --- Phase 2 (sort_by_shard) delta ---
    if "sort_off" in selected:
        emit(bench_index_plan(
            "phase2_sort_off", path_str, locality_plans, **common,
            sort_by_shard=False, lookahead=4, notes="Phase 2 baseline"
        ))
    if "sort_on" in selected:
        emit(bench_index_plan(
            "phase2_sort_on", path_str, locality_plans, **common,
            sort_by_shard=True, lookahead=4, notes="Phase 2 +shard sort"
        ))

    # --- Phase 4 (lookahead) deltas ---
    if "lookahead0" in selected:
        emit(bench_index_plan(
            "phase4_lookahead0", path_str, locality_plans, **common,
            sort_by_shard=True, lookahead=0, notes="Phase 4 baseline (sync decode)"
        ))
    if "lookahead4" in selected:
        emit(bench_index_plan(
            "phase4_lookahead4", path_str, locality_plans, **common,
            sort_by_shard=True, lookahead=4, notes="Phase 4 lookahead=4"
        ))
    if "lookahead8" in selected:
        emit(bench_index_plan(
            "phase4_lookahead8", path_str, locality_plans, **common,
            sort_by_shard=True, lookahead=8, notes="Phase 4 lookahead=8"
        ))

    # --- Backed Python baseline ---
    if "backed_python" in selected:
        emit(bench_backed_python(
            "backed_python_baseline", path_str, locality_plans,
            n_pairs_per_batch=args.n_pairs_per_batch,
            hvg_indices=hvg, normalize=args.normalize,
            notes="ScxBackedSparseDataset + Python loop (current cell-load-scx path)",
        ))

    # --- TrainingDataset ceiling ---
    if "training_seq" in selected:
        emit(bench_training_seq(
            "training_seq_ceiling", path_str,
            n_pairs_per_batch=args.n_pairs_per_batch,
            n_batches_target=args.n_batches,
            hvg_indices=hvg, normalize=args.normalize,
            notes="TrainingDataset ceiling",
        ))

    # --- Phase deltas summary ---
    by_name = {r.name: r for r in results}
    print()
    print("=== Phase 7.3 — locality optimisation deltas ===")
    if "phase2_sort_off" in by_name and "phase2_sort_on" in by_name:
        a = by_name["phase2_sort_off"].batches_per_sec
        b = by_name["phase2_sort_on"].batches_per_sec
        if a > 0:
            print(f"  Phase 2 (shard sort):       {b / a:.2f}× ({a:.1f} → {b:.1f} bps)")
    if "phase4_lookahead0" in by_name and "phase4_lookahead4" in by_name:
        a = by_name["phase4_lookahead0"].batches_per_sec
        b = by_name["phase4_lookahead4"].batches_per_sec
        if a > 0:
            print(f"  Phase 4 (lookahead 0→4):    {b / a:.2f}× ({a:.1f} → {b:.1f} bps)")
    if "phase4_lookahead4" in by_name and "phase4_lookahead8" in by_name:
        a = by_name["phase4_lookahead4"].batches_per_sec
        b = by_name["phase4_lookahead8"].batches_per_sec
        if a > 0:
            print(f"  Phase 4 (lookahead 4→8):    {b / a:.2f}× ({a:.1f} → {b:.1f} bps)")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
