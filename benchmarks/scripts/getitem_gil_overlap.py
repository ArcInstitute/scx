#!/usr/bin/env python
"""2-thread GIL-overlap micro-bench for backed ``__getitem__`` (P1 / OPT-1.1).

The backed row-gather path (``X[rows]``) decodes shards (LZ4/Zstd/Pcodec/Scx1)
in Rust. Before OPT-1.1 that decode ran *while holding the GIL*, so two Python
threads each gathering disjoint rows were serialized. After the fix the decode
runs inside ``detached(py, || ...)`` (GIL released), so two threads overlap.

This bench measures:

  * ``t_serial``   — two gather workloads run back-to-back on one thread
  * ``t_parallel`` — the same two workloads run on two threads

Pass criterion (the review's AC): ``t_parallel < t_serial`` — wall-clock is
less than the sum of parts, i.e. the decodes overlapped. Before the fix the
ratio is ~1.0 (GIL-serialized); after, it should drop toward ~0.5-0.7 on a
2-core decode-bound gather.

Usage:
    python getitem_gil_overlap.py --path /path/to/file.scx \
        [--rows 50000] [--iters 4] [--repeats 3]

Caching is disabled (``cache_shards=0``) so every gather decodes — otherwise a
warm shard cache would serve repeats without decode and hide the overlap.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

import pyscx


def _make_workloads(n_obs: int, n_rows: int, seed: int):
    """Two disjoint random row-index arrays (lower / upper half of the axis)."""
    half = n_obs // 2
    rng = np.random.default_rng(seed)
    n = min(n_rows, half)
    lower = rng.choice(half, size=n, replace=False).astype(np.int64)
    upper = (half + rng.choice(n_obs - half, size=n, replace=False)).astype(np.int64)
    return lower, upper


def _gather(X, idx: np.ndarray, iters: int) -> int:
    """Repeatedly gather ``idx`` from backed ``X``; return total nnz touched."""
    total = 0
    for _ in range(iters):
        sub = X[idx]
        total += int(sub.nnz)
    return total


def _run_serial(X, wl_a, wl_b, iters: int) -> tuple[float, tuple[int, int]]:
    t0 = time.perf_counter()
    nnz_a = _gather(X, wl_a, iters)
    nnz_b = _gather(X, wl_b, iters)
    return time.perf_counter() - t0, (nnz_a, nnz_b)


def _run_parallel(X, wl_a, wl_b, iters: int) -> tuple[float, tuple[int, int]]:
    results: list[int | None] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def worker(slot: int, idx: np.ndarray) -> None:
        # Capture any failure so it can be re-raised on the main thread —
        # Thread.join() does not propagate worker exceptions, and a silent
        # failure in the parallel path is exactly what this bench must catch.
        try:
            results[slot] = _gather(X, idx, iters)
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            errors[slot] = exc

    t_a = threading.Thread(target=worker, args=(0, wl_a))
    t_b = threading.Thread(target=worker, args=(1, wl_b))
    t0 = time.perf_counter()
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    elapsed = time.perf_counter() - t0

    for exc in errors:
        if exc is not None:
            raise RuntimeError("parallel gather worker failed") from exc
    if any(r is None for r in results):
        raise RuntimeError("parallel gather worker did not produce a result")
    return elapsed, (results[0], results[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="Path to a .scx file")
    ap.add_argument("--rows", type=int, default=50_000,
                    help="Rows gathered per workload (per half-axis)")
    ap.add_argument("--iters", type=int, default=4,
                    help="Gather repeats per workload")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Measurement repeats; best (min) wall time is reported")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # cache_shards=0 → every gather decodes (no warm-cache shortcut).
    adata = pyscx.open(args.path).to_anndata(backed=True, cache_shards=0)
    X = adata.X
    n_obs = X.shape[0]
    print(f"opened {args.path}: shape={X.shape}, cache_shards=0")

    wl_a, wl_b = _make_workloads(n_obs, args.rows, args.seed)
    print(f"workloads: {len(wl_a)} + {len(wl_b)} disjoint rows, "
          f"iters={args.iters}, repeats={args.repeats}")

    # Warm up (touch the file once; cache_shards=0 means no decode is retained).
    _gather(X, wl_a[:1024], 1)

    serials, parallels = [], []
    for r in range(args.repeats):
        ts, nnz_serial = _run_serial(X, wl_a, wl_b, args.iters)
        tp, nnz_parallel = _run_parallel(X, wl_a, wl_b, args.iters)
        # Correctness guard: the parallel path must return the same data as the
        # serial path — catches a decode/projection drift, not just a timing win.
        if nnz_serial != nnz_parallel:
            raise RuntimeError(
                f"serial/parallel nnz mismatch: {nnz_serial} != {nnz_parallel} "
                "— parallel gather returned different data"
            )
        serials.append(ts)
        parallels.append(tp)
        print(f"  rep {r}: serial={ts:.3f}s  parallel={tp:.3f}s  "
              f"ratio={tp / ts:.3f}")

    t_serial = min(serials)
    t_parallel = min(parallels)
    ratio = t_parallel / t_serial
    speedup = t_serial / t_parallel

    print("\n=== RESULT ===")
    print(f"t_serial   (best): {t_serial:.3f}s")
    print(f"t_parallel (best): {t_parallel:.3f}s")
    print(f"parallel/serial ratio: {ratio:.3f}  (lower is better; <1 = overlap)")
    print(f"speedup: {speedup:.2f}x")

    if t_parallel < t_serial:
        print("PASS: parallel gather overlapped (wall-clock < sum-of-parts).")
        return 0
    print("FAIL: no overlap — decode appears GIL-serialized.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
