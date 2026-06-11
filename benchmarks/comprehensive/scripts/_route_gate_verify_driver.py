#!/usr/bin/env python
"""Self-contained route-gate verification driver (ACC-RUST-OPT-V2 § 4).

Runs the touched accelerator benchmark triples in-process on a GPU node,
writes their result JSON into ``<out>/raw/``, then evaluates the absolute
floors from ``thresholds.yaml`` and reports the route-gate signals + any
violations of the new route metrics. Exits non-zero iff a new route floor
is violated.

This is a focused contract check (does the GPU route actually dispatch and
does the gate signal fire), independent of the full baseline-timing gate.

Usage:
    python _route_gate_verify_driver.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# GPU DE v3 is the unconditional default, so the pdex_ref GPU triple takes the
# gpu_csc_v3 route whenever a CSC fixture is present (exercises
# de_route_csc_direct) — no env gate needed.

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from benchmarks.comprehensive.scripts.run_parallel import _run_benchmark  # noqa: E402
from benchmarks.comprehensive.scripts.compare_against_baseline import (  # noqa: E402
    _load_thresholds_yaml,
    check_absolute_floors,
)

# Metrics introduced by section 4. Only violations of these fail this driver.
_NEW_ROUTE_METRICS = {
    "de_route_csc_direct",
    "wilcoxon_route_gpu_correct",
    "wilcoxon_route_csc_direct",
    "csc_dispatch_correct",
    "pca_route_gpu_correct",
    # knn/umap native in-VRAM routes were removed in ACC-RUST-OPT-V4 Phase 3
    # (in-VRAM kNN/UMAP route to rapids-singlecell); their correctness is now
    # gated as `knn_route_rapids_correct` / `umap_route_rapids_correct` on the
    # rapids variants, not here.
    "leiden_route_gpu_correct",
    "hvg_route_gpu_correct",
    "preprocess_route_gpu_correct",
}

# (benchmark, dataset, format_key). pbmc3k for the single-shard route checks;
# tabula_sapiens_100k for the multi-shard CSC-direct + csc_dispatch path.
_TRIPLES = [
    ("accel_de", "pbmc3k", "accel_de__pyscx_pdex_ref_gpu"),
    ("accel_de", "pbmc3k", "accel_de__pyscx_wilcoxon_gpu"),
    ("accel_de", "tabula_sapiens_100k", "accel_de__pyscx_pdex_ref_gpu"),
    ("accel_de", "tabula_sapiens_100k", "accel_de__pyscx_wilcoxon_gpu"),
    # Native in-VRAM covariance PCA / CAGRA kNN / UMAP variants were removed in
    # ACC-RUST-OPT-V4 Phase 3. PCA's native route gate (`pca_route_gpu_correct`)
    # now rides the surviving randomized variant; in-VRAM kNN/UMAP route to
    # rapids and are gated as `*_route_rapids_correct` elsewhere.
    ("accel_pca", "pbmc3k", "accel_pca__pyscx_gpu_rand_hh"),
    ("accel_pca", "tabula_sapiens_100k", "accel_pca__pyscx_gpu_rand_hh"),
    ("accel_leiden", "pbmc3k", "accel_leiden__pyscx_gpu"),
    ("accel_hvg", "pbmc3k", "accel_hvg__pyscx_gpu"),
    ("accel_preprocess", "pbmc3k", "accel_preprocess__pyscx_gpu"),
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__qc_metrics_csc"),
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__hvg_csc"),
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__de_csc"),
    # pseudobulk needs pydeseq2, which lives in the `scx-bench` env (these
    # keys have no `_gpu`, so the real gate routes them there). This driver
    # runs under `scx-bench-gpu`, which lacks pydeseq2, so these two SKIP here
    # by design — verify them on a `scx-bench` (or .venv) CPU run instead.
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__pseudobulk_csc"),
    # CSR counterpart: confirm the csr variant does NOT take a csc route.
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__pseudobulk_csr"),
    # pdex_ref on tabula — covers the live thresholds.yaml csc_dispatch floor.
    # `_open_backed` now sizes the LRU cache to the shard count, so the CSR
    # variant no longer re-decodes every shard per gene chunk (was hours-long
    # at cache_shards=4 < n_shards=7); both complete in minutes.
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__pdex_ref_csc"),
    ("bench_csc_dispatch", "tabula_sapiens_100k", "bench_csc__pdex_ref_csr"),
]


def _routes_and_signals(d: dict) -> str:
    runs = d.get("runs", []) or []
    keys = sorted(
        {k for r in runs for k in (r.get("extra", {}) or {})
         if "route" in k or "dispatch" in k}
    )
    parts = []
    for k in keys:
        vals = [
            (r.get("extra", {}) or {}).get(k)
            for r in runs
            if k in (r.get("extra", {}) or {})
        ]
        parts.append(f"{k}={vals[0]!r}" if vals else f"{k}=?")
    return ", ".join(parts) if parts else "(no route extras)"


def main() -> int:
    out = Path(sys.argv[1])
    (out / "raw").mkdir(parents=True, exist_ok=True)

    print("[route-gate] GPU DE v3 is the unconditional default route")
    for bench, ds, fmt in _TRIPLES:
        try:
            d = _run_benchmark(bench, ds, fmt, "accel_runner", {}, 3, False, None)
        except Exception as e:  # noqa: BLE001
            print(f"[route-gate] ERROR {bench} {fmt} {ds}: {e}")
            continue
        if d.get("skipped"):
            print(f"[route-gate] SKIP  {bench} {fmt} {ds}")
            continue
        (out / "raw" / f"{bench}__{fmt}__{ds}.json").write_text(json.dumps(d))
        print(f"[route-gate] OK    {bench} {fmt} {ds}: {_routes_and_signals(d)}")

    floors = _load_thresholds_yaml(_REPO / "benchmarks/comprehensive/thresholds.yaml")
    violations = check_absolute_floors(out, floors)
    new_v = [v for v in violations if v.metric in _NEW_ROUTE_METRICS]

    print("\n[route-gate] === new route-floor evaluation ===")
    if not new_v:
        print("[route-gate] PASS — all new route floors satisfied (or skipped).")
    else:
        for v in new_v:
            obs = "missing" if v.observed is None else f"{v.observed:.3f}"
            print(
                f"[route-gate] FAIL  {v.benchmark} {v.fmt} {v.dataset} "
                f"{v.metric}: observed={obs} {v.direction} {v.threshold}"
            )
    return 1 if new_v else 0


if __name__ == "__main__":
    raise SystemExit(main())
