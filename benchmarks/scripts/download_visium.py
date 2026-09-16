#!/usr/bin/env python3
"""Download a 10x Visium spatial sample and build the spatial fixture.

Source: 10x Genomics public Visium datasets, via `scanpy.datasets.visium_sge`.
This is the benchmark suite's **only** dataset with spatial coordinates or a
pairwise graph — `benchmarks/comprehensive/config.py` records that no source
h5ad in the suite carries an `obsm` key, and nothing carries an `obsp` at all,
so the `cellset_gather` neighbourhood arms have nothing to run on without it.

## The graph is built here, not downloaded

A Visium h5ad ships `obsm["spatial"]` (integer pixel coordinates) and **no**
`obsp`. The neighbourhood arms need a stored graph, so this script computes one
from the coordinates with `sc.pp.neighbors(..., use_rep="spatial")`, which
writes `obsp["connectivities"]` and `obsp["distances"]`. At this sample's size
scanpy takes its exact kNN path, so the graph is reproducible from the recorded
parameters rather than being an artefact of an approximate index.

The sample id, `n_neighbors` and the scanpy version are stamped into
`uns["scx_spatial_fixture"]` so the fixture's identity is on disk and a figure
captured against it can be traced back to how it was built.

## After this

    python benchmarks/scripts/download_visium.py
    python benchmarks/scripts/reconvert_fixtures.py \
        --datasets visium_lymph_node --formats scx_auto

The conversion's `--shard-size` matters: the neighbourhood arms exercise the
**bounded** obsp read, and a shard target above `n_obs` emits one obsp shard,
which turns that read into a whole-matrix decode without saying so. This script
prints the shard size to use.
"""

import os
import sys
from pathlib import Path

import scanpy as sc

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "comprehensive"))
from bench_env import DATA_DIR  # noqa: E402

# scanpy's own default Visium sample. Chosen over a bespoke Xenium download
# because it is reachable through a pinned library call rather than a URL that
# can move, and because it needs no accessory parsing.
SAMPLE_ID = "V1_Human_Lymph_Node"
N_NEIGHBORS = 6  # Visium's hexagonal lattice has six immediate neighbours.
OUTPUT_PATH = os.path.join(str(DATA_DIR), "visium_lymph_node.h5ad")
# Enough shards that a row-range obsp read touches a subset, not the whole file.
SUGGESTED_SHARD_SIZE = 512


def main():
    if os.path.exists(OUTPUT_PATH):
        import anndata as ad

        adata = ad.read_h5ad(OUTPUT_PATH, backed="r")
        print(f"Output already exists: {OUTPUT_PATH}")
        print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
        print(f"  obsm: {list(adata.obsm.keys())}  obsp: {list(adata.obsp.keys())}")
        return

    os.makedirs(str(DATA_DIR), exist_ok=True)
    # `visium_sge` caches into `sc.settings.datasetdir`, which defaults to
    # `./data` — i.e. it litters whatever directory the script was run from,
    # which for this repo is the repo root. Point it at the benchmark data dir
    # instead, beside the fixture it produces.
    sc.settings.datasetdir = Path(str(DATA_DIR)) / "_visium_cache"
    print(f"Downloading Visium sample {SAMPLE_ID} via scanpy.datasets.visium_sge...")
    adata = sc.datasets.visium_sge(sample_id=SAMPLE_ID)
    adata.var_names_make_unique()
    print(f"  Shape: {adata.n_obs} x {adata.n_vars}")

    if "spatial" not in adata.obsm:
        raise RuntimeError(
            f"{SAMPLE_ID} has no obsm['spatial'] — the sample layout changed and "
            "this script's premise is no longer true"
        )
    print(f"  obsm['spatial'] dtype={adata.obsm['spatial'].dtype} "
          f"shape={adata.obsm['spatial'].shape}")

    # A Visium h5ad carries no obsp; build one from the coordinates.
    print(f"Building a spatial kNN graph (k={N_NEIGHBORS}) from obsm['spatial']...")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="spatial")
    for key in ("connectivities", "distances"):
        if key not in adata.obsp:
            raise RuntimeError(f"sc.pp.neighbors did not write obsp['{key}']")
        print(f"  obsp['{key}']: nnz={adata.obsp[key].nnz:,}")

    # X arrives as raw counts; keep it that way — the gather arms time a
    # decode, and a normalised float matrix would measure a different codec.
    import scipy.sparse as sp

    if not sp.issparse(adata.X) or adata.X.format != "csr":
        adata.X = sp.csr_matrix(adata.X)
        print("  Converted X to CSR")

    adata.uns["scx_spatial_fixture"] = {
        "sample_id": SAMPLE_ID,
        "n_neighbors": N_NEIGHBORS,
        "graph_source": "sc.pp.neighbors(use_rep='spatial')",
        "scanpy_version": sc.__version__,
    }

    # `visium_sge` leaves the image payload in uns; it is large, unused by every
    # benchmark here, and would dominate the h5ad. Dropped explicitly rather
    # than carried silently.
    if "spatial" in adata.uns:
        del adata.uns["spatial"]
        print("  Dropped uns['spatial'] (tissue images — unused by the benchmarks)")

    print(f"Writing h5ad to: {OUTPUT_PATH}")
    adata.write_h5ad(OUTPUT_PATH)
    print(f"  h5ad size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")
    print()
    print("Next:")
    print("  python benchmarks/scripts/reconvert_fixtures.py \\")
    print("      --datasets visium_lymph_node --formats scx_auto")
    print(f"  (convert with --shard-size {SUGGESTED_SHARD_SIZE} so the obsp is "
          "sharded; see this script's docstring)")


if __name__ == "__main__":
    main()
