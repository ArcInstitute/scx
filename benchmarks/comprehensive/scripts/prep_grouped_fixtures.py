#!/usr/bin/env python
"""Stage the real Perturb-seq fixtures the grouped-sharding benchmark needs.

Copies read-only loader h5ad files into the benchmark data dir under the names
the ``grouped_sort`` benchmark expects (``DATA_DIR/{dataset}.h5ad``). Idempotent:
skips a destination that already exists with the same size.

The loader sources under ``/large_storage/arcinfra/projects/state-designer/loader/``
are **read-only** — this script only ever reads from them.

Synthetic grouping datasets (``pert_synth_10k`` / ``nb_glm_synth``) are NOT
staged here — the benchmark materializes them in-process via ``_pert_synth``.

Usage:
    python benchmarks/comprehensive/scripts/prep_grouped_fixtures.py
    python benchmarks/comprehensive/scripts/prep_grouped_fixtures.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from benchmarks.comprehensive.bench_env import DATA_DIR

# dataset name (== grouped_sort GROUP_SPEC key) -> read-only loader source.
_LOADER = Path("/large_storage/arcinfra/projects/state-designer/loader")
FIXTURES: dict[str, Path] = {
    "replogle_k562": _LOADER / "Replogle2022" / "k562_n600.h5ad",
    "tahoe_c38": _LOADER / "tahoe-100m" / "c38-n10.h5ad",
    # Real RAW-COUNT Perturb-seq fixture (integer UMIs, CSR) — shardad's home
    # turf for the integer-count compression + grouped head-to-head. The other
    # two real fixtures are float/normalized. 136,051 cells x 18,151 genes,
    # ~909M nnz; obs `target_gene` (2354 KOs) + `non-targeting` control.
    "chemogenetic_rgfp": _LOADER / "chemogenetic_h1" / "run1" / "RGFP-n5.h5ad",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true", help="re-copy even if the dest exists"
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, src in FIXTURES.items():
        dst = DATA_DIR / f"{name}.h5ad"
        if not src.exists():
            print(f"[MISS] {name}: source not found: {src}", file=sys.stderr)
            rc = 1
            continue
        if dst.exists() and not args.force and dst.stat().st_size == src.stat().st_size:
            print(f"[skip] {name}: {dst} already present ({dst.stat().st_size} bytes)")
            continue
        print(f"[copy] {src} -> {dst} ({src.stat().st_size / 1e9:.2f} GB)")
        shutil.copy2(src, dst)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
