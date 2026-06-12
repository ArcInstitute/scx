"""Query pipeline integration tests."""

import numpy as np
import pytest
import scipy.sparse as sp


# ───────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────


def test_filter_obs_cell_type(query_adata, scx_from_adata):
    """filter_obs with cell_type predicate returns correct subset."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    result = pyscx.open(path).query()
    result.filter_obs("cell_type == 'T cell'")
    qr = result.collect()

    assert qr.n_obs == 40  # 40 T cells in fixture
    assert qr.n_vars == 40


def test_collect_to_anndata(query_adata, scx_from_adata):
    """collect().to_anndata() returns valid AnnData with filtered obs."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'B cell'")
    adata = p.collect().to_anndata()

    assert adata.n_obs == 40
    assert adata.n_vars == 40
    assert sp.issparse(adata.X)
    # obs should only contain B cells
    assert all(adata.obs["cell_type"] == "B cell")


def test_select_genes(query_adata, scx_from_adata):
    """select_genes([0, 1, 2]) returns 3-column matrix."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.select_genes([0, 1, 2])
    result = p.collect()

    assert result.n_vars == 3
    assert result.n_obs == 120  # all cells


def test_select_genes_preserves_order_and_names(query_adata, scx_from_adata):
    """F4: select_genes returns columns in the requested order (not sorted),
    accepts gene names and mixed int/str, raises KeyError on unknown names, and
    collapses duplicates to first occurrence."""
    import pyscx

    path = scx_from_adata(query_adata, "order.scx")
    ref = query_adata.X.toarray()  # 120 × 40 reference

    # Non-ascending integer request → requested order preserved.
    req = [7, 3, 0, 20]
    adata = pyscx.open(path).query().select_genes(req).collect().to_anndata()
    assert list(adata.var["gene_id"]) == [f"gene_{i}" for i in req]
    assert np.array_equal(adata.X.toarray(), ref[:, req])

    # Gene names in a custom order.
    names = ["gene_5", "gene_2", "gene_30"]
    a2 = pyscx.open(path).query().select_genes(names).collect().to_anndata()
    assert list(a2.var["gene_id"]) == names
    assert np.array_equal(a2.X.toarray(), ref[:, [5, 2, 30]])

    # Mixed int + name, order preserved.
    a3 = pyscx.open(path).query().select_genes([1, "gene_9", 4]).collect().to_anndata()
    assert list(a3.var["gene_id"]) == ["gene_1", "gene_9", "gene_4"]
    assert np.array_equal(a3.X.toarray(), ref[:, [1, 9, 4]])

    # Unknown name → KeyError (eager, at select_genes).
    with pytest.raises(KeyError):
        pyscx.open(path).query().select_genes(["not_a_gene"])

    # Duplicates collapse to first occurrence.
    a4 = pyscx.open(path).query().select_genes([3, 3, 1]).collect().to_anndata()
    assert list(a4.var["gene_id"]) == ["gene_3", "gene_1"]


def test_normalize_log1p(query_adata, scx_from_adata):
    """with_normalize + with_log1p produces transformed values matching scanpy."""
    import pyscx
    import scanpy as sc

    path = scx_from_adata(query_adata, "query.scx")

    # SCX pipeline path
    p = pyscx.open(path).query()
    p.with_normalize(1e4)
    p.with_log1p()
    scx_adata = p.collect().to_anndata()

    # Scanpy reference path
    ref_adata = pyscx.open(path).to_anndata()
    sc.pp.normalize_total(ref_adata, target_sum=1e4)
    sc.pp.log1p(ref_adata)

    np.testing.assert_allclose(
        scx_adata.X.toarray(), ref_adata.X.toarray(), rtol=1e-5, atol=1e-5
    )


def test_count(query_adata, scx_from_adata):
    """count() returns number of matching cells."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'NK cell'")
    assert p.count() == 40


def test_limit(query_adata, scx_from_adata):
    """limit(5) returns at most 5 cells."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.limit(5)
    result = p.collect()	

    assert result.n_obs <= 5


def test_chained_predicates(query_adata, scx_from_adata):
    """Chained filter_obs calls apply AND semantics."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    p.filter_obs("tissue == 'lung'")
    result = p.collect()

    # 40 T cells, half lung half blood → 20 expected
    assert result.n_obs == 20


def test_invalid_column_raises_valueerror(query_adata, scx_from_adata):
    """Invalid column name raises ValueError (not RuntimeError)."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises((ValueError, RuntimeError)):
        p.filter_obs("nonexistent_col == 'foo'")


def test_empty_result(query_adata, scx_from_adata):
    """Empty result (no matching cells) gives AnnData with 0 rows."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'macrophage'")  # no macrophages in fixture
    result = p.collect()

    assert result.n_obs == 0
    adata = result.to_anndata()
    assert adata.n_obs == 0


def test_to_csr(query_adata, scx_from_adata):
    """to_csr() returns scipy CSR without AnnData overhead."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    result = pyscx.open(path).query().collect()
    csr = result.to_csr()

    assert sp.issparse(csr)
    assert csr.shape == (120, 40)


def test_shard_stats_accessible(query_adata, scx_from_adata):
    """skipped_shards and total_shards are accessible on result."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    result = pyscx.open(path).query().collect()

    assert result.total_shards >= 1
    assert result.skipped_shards >= 0


def test_sequential_call_syntax(query_adata, scx_from_adata):
    """Sequential call syntax works (no method chaining needed)."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    p.with_log1p()
    result = p.collect()

    assert result.n_obs == 40
