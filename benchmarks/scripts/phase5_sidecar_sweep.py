#!/usr/bin/env python3
"""Phase 5 / T5.1+T5.2 — IndexPlanDataset sidecar on/off × batch-size sweep.

Measures the loader-level win of the scx1 decode sidecar (L1+L2) vs. the legacy
full-shard-decode path, across realistic STATE_TX batch sizes, for the random and
locality plan generators. Reuses the canonical measurement path
(`index_plan._run_index_plan`) so numbers are comparable to the Phase-0 baselines.

Per (scenario × batch_size × scatter_sidecar) combo it records throughput
(batches/s, cells/s), consumer-observed per-batch gather latency (mean/p50/p99 ms),
peak RSS (T5.2 — must not regress with the sidecar), and the sidecar adoption rate.

Usage::

    python benchmarks/scripts/phase5_sidecar_sweep.py --dataset census_1m --n-batches 60
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.comprehensive.benchmarks import index_plan
from benchmarks.comprehensive.config import DATASETS, RAW_RESULTS_DIR

_BATCH_SIZES = (2, 32, 128)
_SCENARIOS = ("random", "locality")


def _adoption(sidecar: int | None, full: int | None) -> float | None:
    if sidecar is None or full is None or (sidecar + full) == 0:
        return None
    return round(sidecar / (sidecar + full), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="dataset name (config.DATASETS)")
    ap.add_argument("--n-batches", type=int, default=60, help="batches per combo")
    ap.add_argument("--cache-shards", type=int, default=128)
    ap.add_argument("--lookahead", type=int, default=4)
    args = ap.parse_args()

    if args.dataset not in DATASETS:
        print(f"unknown dataset {args.dataset!r}", file=sys.stderr)
        return 2
    dataset = DATASETS[args.dataset]
    scx_path = str(dataset.path_for_format("scx_auto"))
    n_obs = dataset.n_obs
    n_vars = dataset.n_vars
    print(f"=== phase5 sidecar sweep: {args.dataset} (n_obs={n_obs}, n_vars={n_vars}) ===")
    print(f"    fixture: {scx_path}")

    results: list[dict] = []
    for scenario in _SCENARIOS:
        for batch_size in _BATCH_SIZES:
            # pairs_per_batch must not exceed n_obs (random) / group (locality);
            # tiny fixtures are clamped so the generators stay valid.
            ppb = min(batch_size, max(1, n_obs // 4))
            for scatter_sidecar in (True, False):
                if scenario == "random":
                    def factory(ppb=ppb):
                        return index_plan._random_plans(n_obs, ppb, args.n_batches, seed=0)
                else:
                    def factory(ppb=ppb):
                        return index_plan._locality_plans(
                            n_obs, index_plan._LOCALITY_GROUP_SIZE, ppb, args.n_batches, seed=0
                        )

                outcome = index_plan._run_index_plan(
                    scx_path,
                    factory,
                    ppb,
                    hvg_indices=None,
                    normalize=True,
                    sort_by_shard=True,
                    lookahead=args.lookahead,
                    cache_shards=args.cache_shards,
                    max_plan_size=max(ppb, 256),
                    scatter_sidecar=scatter_sidecar,
                )
                wall = outcome.wall_s
                row = {
                    "dataset": args.dataset,
                    "scenario": scenario,
                    "batch_size": batch_size,
                    "pairs_per_batch": ppb,
                    "scatter_sidecar": scatter_sidecar,
                    "n_batches": outcome.n_batches,
                    "batches_per_sec": round(outcome.n_batches / wall, 3) if wall > 0 else 0.0,
                    "cells_per_sec": round(outcome.n_cells / wall, 1) if wall > 0 else 0.0,
                    "gather_latency_ms_mean": outcome.gather_latency_ms_mean,
                    "gather_latency_ms_p50": outcome.gather_latency_ms_p50,
                    "gather_latency_ms_p99": outcome.gather_latency_ms_p99,
                    "peak_rss_mb": round(outcome.peak_rss_mb, 1),
                    "sidecar_groups": outcome.sidecar_groups,
                    "full_shard_groups": outcome.full_shard_groups,
                    "sidecar_adoption_rate": _adoption(
                        outcome.sidecar_groups, outcome.full_shard_groups
                    ),
                }
                results.append(row)
                print(
                    f"    {scenario:9s} bs={batch_size:<4d} sidecar={str(scatter_sidecar):5s} "
                    f"-> {row['batches_per_sec']:>8.2f} b/s  "
                    f"p50={row['gather_latency_ms_p50']} ms  "
                    f"rss={row['peak_rss_mb']} MB  adopt={row['sidecar_adoption_rate']}"
                )

    out_dir = RAW_RESULTS_DIR.parent / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sidecar_sweep__{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"dataset": args.dataset, "n_obs": n_obs, "n_vars": n_vars,
             "n_batches": args.n_batches, "rows": results},
            f, indent=2, default=str,
        )
    print(f"=== wrote {out_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
