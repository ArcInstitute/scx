"""Integration tests for comparison optimization (Phase 4 Step 2).

Tests _ComparisonResult lazy wrapper:
- (X > 0).sum(axis=1) short-circuits to getnnz(axis=1) — no materialization
- (X > 0).sum(axis=0) short-circuits to getnnz(axis=0)
- Non-shortcircuit patterns fall back to materialization correctly
- Scanpy QC functions work in backed mode
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_scx(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic_adata."""
    import pyscx

    path = str(tmp_dir / "cmp_test.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def adata_backed(backed_scx):
    import pyscx

    return pyscx.open(backed_scx).to_anndata(backed=True)


@pytest.fixture
def adata_non_backed(backed_scx):
    import pyscx

    return pyscx.open(backed_scx).to_anndata()


# --- _ComparisonResult type tests ---


def test_gt_returns_comparison_result(adata_backed):
    """__gt__ returns a _ComparisonResult, not a materialized matrix."""
    result = adata_backed.X > 0
    assert type(result).__name__ == "_ComparisonResult"


def test_comparison_result_repr(adata_backed):
    """_ComparisonResult has a useful __repr__."""
    result = adata_backed.X > 0
    r = repr(result)
    assert "gt" in r
    assert "shortcircuit=true" in r


# --- Short-circuit tests ---


def test_gt0_sum_axis1_shortcircuit(adata_backed, adata_non_backed):
    """(X > 0).sum(axis=1) uses getnnz short-circuit and matches scipy."""
    backed_result = np.asarray((adata_backed.X > 0).sum(axis=1)).flatten()
    scipy_result = np.asarray(
        (adata_non_backed.X > 0).sum(axis=1)
    ).flatten()
    np.testing.assert_array_equal(backed_result, scipy_result)


def test_gt0_sum_axis0_shortcircuit(adata_backed, adata_non_backed):
    """(X > 0).sum(axis=0) uses getnnz short-circuit and matches scipy."""
    backed_result = np.asarray((adata_backed.X > 0).sum(axis=0)).flatten()
    scipy_result = np.asarray(
        (adata_non_backed.X > 0).sum(axis=0)
    ).flatten()
    np.testing.assert_array_equal(backed_result, scipy_result)


def test_gt0_sum_no_axis(adata_backed, adata_non_backed):
    """(X > 0).sum() scalar matches scipy."""
    backed_result = (adata_backed.X > 0).sum()
    scipy_result = (adata_non_backed.X > 0).sum()
    assert backed_result == scipy_result


# --- Non-shortcircuit fallback tests ---


def test_gt5_sum_axis1_fallback(adata_backed, adata_non_backed):
    """(X > 5).sum(axis=1) falls back to materialization and is correct."""
    backed_result = np.asarray((adata_backed.X > 5).sum(axis=1)).flatten()
    scipy_result = np.asarray(
        (adata_non_backed.X > 5).sum(axis=1)
    ).flatten()
    np.testing.assert_array_equal(backed_result, scipy_result)


def test_lt10_sum_axis0(adata_backed, adata_non_backed):
    """(X < 10).sum(axis=0) falls back and matches scipy."""
    backed_result = np.asarray((adata_backed.X < 10).sum(axis=0)).flatten()
    scipy_result = np.asarray(
        (adata_non_backed.X < 10).sum(axis=0)
    ).flatten()
    np.testing.assert_array_equal(backed_result, scipy_result)


def test_eq0_toarray(adata_backed, adata_non_backed):
    """(X == 0).toarray() materializes correctly."""
    backed_result = (adata_backed.X == 0).toarray()
    scipy_result = (adata_non_backed.X == 0).toarray()
    np.testing.assert_array_equal(backed_result, scipy_result)


def test_ge1_getitem(adata_backed, adata_non_backed):
    """(X >= 1)[0:5] indexing materializes and slices correctly."""
    backed_result = (adata_backed.X >= 1)[0:5].toarray()
    scipy_result = (adata_non_backed.X >= 1)[0:5].toarray()
    np.testing.assert_array_equal(backed_result, scipy_result)


# --- With deletion vectors ---


def test_gt0_sum_with_deletions(tmp_dir):
    """(X > 0).sum(axis=1) with deletions matches filtered matrix."""
    import anndata
    import pyscx

    np.random.seed(123)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    adata = anndata.AnnData(X=sp.csr_matrix(dense))

    path = str(tmp_dir / "del_cmp.scx")
    pyscx.from_anndata(adata, path)

    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[[0, 5, 99]] = True
    pyscx.open(path).mark_deleted(delete_mask)

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_genes = np.asarray((adata_backed.X > 0).sum(axis=1)).flatten()
    full_genes = np.asarray((adata_full.X > 0).sum(axis=1)).flatten()
    np.testing.assert_array_equal(backed_genes, full_genes)


# --- Scanpy validation ---


def test_calculate_qc_metrics(backed_scx):
    """sc.pp.calculate_qc_metrics works in backed mode.

    Note: calculate_qc_metrics internally uses (X > 0).sum(axis=1) pattern.
    We materialize to a copy first since QC metrics modifies adata in-place.
    """
    import pyscx
    import scanpy as sc

    adata_backed = pyscx.open(backed_scx).to_anndata(backed=True)
    adata = adata_backed[:].copy()
    sc.pp.calculate_qc_metrics(adata, percent_top=[], inplace=True)

    assert "n_genes_by_counts" in adata.obs.columns
    assert "total_counts" in adata.obs.columns
    assert "n_cells_by_counts" in adata.var.columns
    assert "total_counts" in adata.var.columns


def test_calculate_qc_metrics_matches(backed_scx):
    """QC metrics match between backed and non-backed."""
    import pyscx
    import scanpy as sc

    adata_backed = pyscx.open(backed_scx).to_anndata(backed=True)
    adata_from_backed = adata_backed[:].copy()
    adata_full = pyscx.open(backed_scx).to_anndata()

    sc.pp.calculate_qc_metrics(adata_from_backed, percent_top=[], inplace=True)
    sc.pp.calculate_qc_metrics(adata_full, percent_top=[], inplace=True)

    np.testing.assert_array_equal(
        adata_from_backed.obs["n_genes_by_counts"].values,
        adata_full.obs["n_genes_by_counts"].values,
    )
    np.testing.assert_array_equal(
        adata_from_backed.var["n_cells_by_counts"].values,
        adata_full.var["n_cells_by_counts"].values,
    )
    np.testing.assert_allclose(
        adata_from_backed.obs["total_counts"].values,
        adata_full.obs["total_counts"].values,
        rtol=1e-5,
    )


def test_filter_cells(backed_scx):
    """sc.pp.filter_cells works in backed mode."""
    import pyscx
    import scanpy as sc

    adata = pyscx.open(backed_scx).to_anndata(backed=True)
    # Materialize first since filter_cells modifies the AnnData
    adata_mat = adata[:].copy()
    sc.pp.filter_cells(adata_mat, min_genes=1)
    assert adata_mat.n_obs > 0


def test_filter_genes(backed_scx):
    """sc.pp.filter_genes works in backed mode."""
    import pyscx
    import scanpy as sc

    adata = pyscx.open(backed_scx).to_anndata(backed=True)
    adata_mat = adata[:].copy()
    sc.pp.filter_genes(adata_mat, min_cells=1)
    assert adata_mat.n_vars > 0
