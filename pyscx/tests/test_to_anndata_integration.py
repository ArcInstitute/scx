"""Tests for improved to_anndata() integration (Phase4 Step 5).

Validates:
- var_names: gene projection at load time
- obs_filter: predicate-based cell filtering via query engine
- layers: selective layer loading
- Combined parameters
- Error cases
"""

import numpy as np
import pytest


def test_var_names_projects_genes(synthetic_adata, scx_from_adata):
    """to_anndata(var_names=[...]) returns only the requested genes."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames.scx")
    target_genes = ["gene_0", "gene_5", "gene_10"]
    adata = pyscx.open(path).to_anndata(var_names=target_genes)

    assert adata.n_vars == len(target_genes)
    assert adata.n_obs == synthetic_adata.n_obs
    # var index should contain exactly the requested genes
    for gene in target_genes:
        assert gene in adata.var.index.tolist()


def test_var_names_data_correct(synthetic_adata, scx_from_adata):
    """var_names projection produces correct X data."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_data.scx")
    target_genes = ["gene_0", "gene_1"]

    # Load with var_names
    filtered = pyscx.open(path).to_anndata(var_names=target_genes)

    # Load full and manually slice
    full = pyscx.open(path).to_anndata()
    gene_mask = full.var.index.isin(target_genes)
    expected = full[:, gene_mask].copy()

    np.testing.assert_array_equal(
        filtered.X.toarray(), expected.X.toarray()
    )


def test_obs_filter_filters_cells(query_adata, scx_from_adata):
    """to_anndata(obs_filter=...) returns only matching cells."""
    import pyscx

    path = scx_from_adata(query_adata, "obsfilter.scx")
    adata = pyscx.open(path).to_anndata(obs_filter="cell_type == 'T cell'")

    assert adata.n_obs == 40  # 40 T cells in query_adata fixture
    assert all(adata.obs["cell_type"] == "T cell")


def test_obs_filter_matches_query_pipeline(query_adata, scx_from_adata):
    """obs_filter produces same result as the query pipeline."""
    import pyscx

    path = scx_from_adata(query_adata, "obsfilter_qp.scx")

    # to_anndata(obs_filter=...)
    adata_filter = pyscx.open(path).to_anndata(obs_filter="cell_type == 'B cell'")

    # query pipeline
    adata_qp = (
        pyscx.open(path)
        .query()
        .filter_obs("cell_type == 'B cell'")
        .collect()
        .to_anndata()
    )

    assert adata_filter.n_obs == adata_qp.n_obs
    np.testing.assert_array_equal(
        adata_filter.X.toarray(), adata_qp.X.toarray()
    )


def test_obs_filter_with_var_names(query_adata, scx_from_adata):
    """Combined obs_filter + var_names correctly filters both dimensions."""
    import pyscx

    path = scx_from_adata(query_adata, "combined.scx")
    adata = pyscx.open(path).to_anndata(
        obs_filter="cell_type == 'NK cell'",
        var_names=["gene_0", "gene_1", "gene_2"],
    )

    assert adata.n_obs == 40  # 40 NK cells
    assert adata.n_vars == 3
    assert all(adata.obs["cell_type"] == "NK cell")


def test_layers_selective(synthetic_adata, scx_from_adata):
    """to_anndata(layers=[...]) loads only requested layers."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "layers.scx")

    # Load only "raw" layer
    adata = pyscx.open(path).to_anndata(layers=["raw"])
    assert "raw" in adata.layers
    assert adata.n_obs == synthetic_adata.n_obs

    # Load with empty list — no layers
    adata_no_layers = pyscx.open(path).to_anndata(layers=[])
    assert len(adata_no_layers.layers) == 0


def test_layers_backed(synthetic_adata, scx_from_adata):
    """to_anndata(backed=True, layers=[...]) loads only requested layers."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "layers_backed.scx")

    # Load only "raw" layer in backed mode
    adata = pyscx.open(path).to_anndata(backed=True, layers=["raw"])
    assert "raw" in adata.layers

    # Load with empty list — no layers
    adata_no = pyscx.open(path).to_anndata(backed=True, layers=[])
    assert len(adata_no.layers) == 0


def test_obs_filter_backed(query_adata, scx_from_adata):
    """to_anndata(backed=True, obs_filter=...) filters cells in backed mode."""
    import pyscx

    path = scx_from_adata(query_adata, "obsfilter_backed.scx")
    adata = pyscx.open(path).to_anndata(backed=True, obs_filter="cell_type == 'T cell'")

    assert adata.n_obs == 40
    assert all(adata.obs["cell_type"] == "T cell")

    # X should be a backed dataset
    assert isinstance(adata.X, pyscx.ScxBackedSparseDataset)

    # Materialize and verify shape
    x_dense = adata.X.to_memory()
    assert x_dense.shape == (40, 40)


def test_var_names_backed(synthetic_adata, scx_from_adata):
    """backed=True + var_names produces correct shape and var."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_backed.scx")
    target_genes = ["gene_0", "gene_5", "gene_10"]
    adata = pyscx.open(path).to_anndata(backed=True, var_names=target_genes)

    assert adata.n_vars == len(target_genes)
    assert adata.n_obs == synthetic_adata.n_obs
    for gene in target_genes:
        assert gene in adata.var.index.tolist()

    # X should be a ScxBackedSparseDataset
    assert isinstance(adata.X, pyscx.ScxBackedSparseDataset)
    assert adata.X.shape == (synthetic_adata.n_obs, len(target_genes))

    # Materialize and check shape
    x_mat = adata.X.to_memory()
    assert x_mat.shape == (synthetic_adata.n_obs, len(target_genes))


def test_var_names_backed_data_matches_nonbacked(synthetic_adata, scx_from_adata):
    """backed + var_names produces same data as non-backed + var_names."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_backed_match.scx")
    target_genes = ["gene_0", "gene_1"]

    # Non-backed
    eager = pyscx.open(path).to_anndata(var_names=target_genes)
    # Backed
    backed = pyscx.open(path).to_anndata(backed=True, var_names=target_genes)
    backed_x = backed.X.to_memory()

    np.testing.assert_array_equal(backed_x.toarray(), eager.X.toarray())


def test_var_names_backed_combined_obs_filter(query_adata, scx_from_adata):
    """backed + var_names + obs_filter work together."""
    import pyscx

    path = scx_from_adata(query_adata, "backed_combined.scx")
    adata = pyscx.open(path).to_anndata(
        backed=True,
        var_names=["gene_0", "gene_1", "gene_2"],
        obs_filter="cell_type == 'T cell'",
    )

    assert adata.n_obs == 40
    assert adata.n_vars == 3
    assert all(adata.obs["cell_type"] == "T cell")
    assert isinstance(adata.X, pyscx.ScxBackedSparseDataset)
    assert adata.X.shape == (40, 3)


def test_var_names_none_found_raises(synthetic_adata, scx_from_adata):
    """var_names with no matching genes raises error."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_bad.scx")

    with pytest.raises((RuntimeError, ValueError)):
        pyscx.open(path).to_anndata(var_names=["nonexistent_gene"])


def test_no_params_unchanged(synthetic_adata, scx_from_adata):
    """to_anndata() with no new params produces same result as before."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "no_params.scx")

    adata = pyscx.open(path).to_anndata()
    assert adata.n_obs == synthetic_adata.n_obs
    assert adata.n_vars == synthetic_adata.n_vars
    assert "raw" in adata.layers
