"""Scanpy pipeline test (Task 15.12): SCX → to_anndata → full scanpy workflow."""

import numpy as np
import pytest


def test_scanpy_pipeline(synthetic_adata, tmp_dir):
    """15.12: SCX → to_anndata → sc.pp.filter_cells → normalize → log1p → PCA → neighbors → leiden."""
    import pyscx
    import scanpy as sc

    adata = synthetic_adata
    path = str(tmp_dir / "pipeline.scx")
    pyscx.from_anndata(adata, path)

    # Load from SCX
    adata2 = pyscx.open(path).to_anndata()

    # Standard scanpy preprocessing pipeline
    sc.pp.filter_cells(adata2, min_genes=1)
    sc.pp.filter_genes(adata2, min_cells=1)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)
    sc.pp.pca(adata2, n_comps=min(10, adata2.n_vars - 1))
    sc.pp.neighbors(adata2, n_neighbors=10)
    sc.tl.leiden(adata2, flavor="igraph", n_iterations=2, directed=False)

    # Verify the pipeline completed without errors
    assert "leiden" in adata2.obs.columns
    assert adata2.obs["leiden"].nunique() >= 1
    assert "X_pca" in adata2.obsm
    assert adata2.obsm["X_pca"].shape[0] == adata2.n_obs
