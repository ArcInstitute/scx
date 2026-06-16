#!/usr/bin/env python3
"""Microbenchmark: Experiment.gather_rows_sparse vs backed adata.X[rows].

SCX-DATA-LOADER Phase 0.3. Measures the per-gather cost (cold cache per call —
the map-style ``__getitem__`` seam being replaced) of the synchronous sparse
gather against today's backed-AnnData random-access path, for a realistic
cell-set draw (consecutive S-block) and a scattered worst case, sweeping
``cache_shards``. Informational only (§0): does not gate the native-loader build.

Usage:
    cd pyscx && ../.venv/bin/maturin develop --release
    ../.venv/bin/python ../benchmarks/scripts/benchmark_gather_rows_sparse.py \
        [--scx PATH] [--s 64] [--runs 20]
"""

import argparse
import time

import numpy as np
import pyscx

DEFAULT_SCX = "/data/transcriptomics/perturbation/replogle/HepG2.scx"


def timer(fn, warmup=2, runs=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6  # µs/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scx", default=DEFAULT_SCX)
    ap.add_argument("--s", type=int, default=64, help="cells per set draw")
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    exp = pyscx.open(args.scx)
    n_obs, n_vars = exp.shape
    n_shards = exp.shard_count
    rows_per_shard = n_obs / n_shards
    print(f"# {args.scx}")
    print(f"# shape=({n_obs},{n_vars}) shards={n_shards} codec_id={exp.codec_id} S={args.s}")

    rng = np.random.default_rng(0)
    # Realistic draw: a consecutive block (same_file_sets=True, consecutive
    # row sampling) — touches ~1 shard.
    start = int(rng.integers(0, n_obs - args.s))
    consec = np.arange(start, start + args.s, dtype=np.uint64)
    # Worst case: scattered across the whole file — touches every shard.
    scattered = rng.choice(n_obs, size=args.s, replace=False).astype(np.uint64)

    def touched(rows):
        return len({int(r // rows_per_shard) for r in rows})

    patterns = {
        f"consecutive (~{touched(consec)} shard)": consec,
        f"scattered (~{touched(scattered)} shards)": scattered,
    }

    # Parity check (cache_shards irrelevant to correctness).
    for name, rows in patterns.items():
        got = exp.gather_rows_sparse(rows)
        ref = pyscx.open(args.scx).to_anndata(backed=True).X[rows.tolist()]
        np.testing.assert_array_equal(got.toarray(), ref.toarray())
    print("# parity OK (gather == backed X[rows]) for all patterns\n")

    header = f"{'pattern':<28}{'cache':>6}{'gather µs':>14}{'X[rows] µs':>14}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for name, rows in patterns.items():
        rows_list = rows.tolist()
        for c in (4, 16, n_shards):
            # Cold per call: fresh reader each iteration (the __getitem__ cost).
            g = timer(
                lambda: pyscx.open(args.scx).gather_rows_sparse(rows, cache_shards=c),
                runs=args.runs,
            )
            b = timer(
                lambda: pyscx.open(args.scx)
                .to_anndata(backed=True, cache_shards=c)
                .X[rows_list],
                runs=args.runs,
            )
            print(f"{name:<28}{c:>6}{g:>14.1f}{b:>14.1f}{b / g:>9.2f}x")


if __name__ == "__main__":
    main()
